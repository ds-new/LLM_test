"""token 频次与欠训练 token 探测 + GSM8K 评测污染检查（训练结束后、pass@k 之后运行）。

三件事，都用这次训练自己的数据和权重，因此每个结论都有"真实频次"作对照：
  1. 频次统计：在全部训练 shard 上数每个 token 出现的次数（Zipf 曲线、覆盖率、从未出现 / 罕见 token 数）；
  2. 欠训练 token：
     * 嵌入范数（初始化时每行约 0.02·√d；从不拿到梯度的行只受 AdamW 权重衰减，会按 ∏(1 − lr·wd) 缩小）；
     * 去掉所有 token 共有的主方向（第一主成分）后，与"未使用 token"平均向量的余弦相似度
       （思路来自 Land & Bartolo 2024《Fishing for Magikarp》）；
     * 用真实频次检验这两个指标：罕见（< 100 次）vs 常见（≥ 1000 次）的 AUC、Spearman 相关；
     * 行为验证：少样本"复述"提示，看模型能否原样复述该 token；
  3. 评测污染：GSM8K test（题目、原始答案、去掉 <<…>> 的答案）的 13-gram 是否在训练数据里出现
     （GPT-3 论文的做法，在 token 序列上做，滚动哈希 + 逐个核对，排除哈希碰撞）。

频次统计与污染检查只用 CPU，可在训练进行时提前算好并缓存（--precompute，低优先级运行）；
训练结束后的阶段直接复用缓存，只做需要 GPU 的嵌入分析和复述测试。结果写入 RUN_DIR/token_probe/。
冒烟测试（CPU 小模型）：python -m decoder_from_scratch.a800.token_freq_probe --smoke
"""

import argparse
import hashlib
import math
import re
import time

import numpy as np
import torch

from .cleaning import normalize_zh_punct
from .common import RUN_DIR, now, read_json, stamp, write_json, write_text

NAME = 'token_probe'
DEFAULTS = {'enabled': True, 'ngram': 13, 'chunk_tokens': 64 * 1024 * 1024, 'repeat_samples': 100,
            'suspicious_top': 50, 'batch': 256}
SPECIAL = 4          # id 0–3 是特殊 token（pad/unk/bos/eos），不参与欠训练分析
P = np.uint64(0x9E3779B97F4A7C15)
TABLE_BITS = 26


# ---------------------------------------------------------------- n-gram 哈希
def ngram_hashes(ids, n):
    """ids: 一维整数数组 → 每个起点的 n-gram 多项式哈希（uint64，溢出即取模 2^64）。"""
    x = np.asarray(ids).astype(np.uint64)
    m = len(x) - n + 1
    if m <= 0:
        return np.zeros(0, np.uint64)
    h = np.zeros(m, np.uint64)
    with np.errstate(over='ignore'):
        for i in range(n):
            h *= P
            h += x[i:i + m]
    return h


def gsm8k_refs(tok, cache_dir):
    """GSM8K test 每题的三段文本（题目、原始答案、去掉计算标注的答案）→ token 序列。"""
    from .passk_probe import load_gsm8k
    rows = load_gsm8k(cache_dir)['test']
    refs = []
    for i, row in enumerate(rows):
        ans = row['answer']
        clean = re.sub(r'<<[^>]*>>', '', ans)
        for field, text in (('question', row['question']), ('answer', ans), ('answer_clean', clean)):
            refs.append((i, field, tok.encode(normalize_zh_punct(text), add_bos=False)))
    return refs, len(rows)


def scan(manifest, refs, vocab, n=13, chunk=DEFAULTS['chunk_tokens'], log=print):
    """一次遍历全部训练 shard：token 频次 + 参考 n-gram 命中。返回 (counts, per_source_tokens, hits)。
    hits: {哈希: {'ids': n-gram, 'positions': 命中次数, 'sources': {来源: 次数}}}（已逐个核对，排除碰撞）。"""
    ref_grams = {}
    for _, _, ids in refs:
        ids = np.asarray(ids, np.int64)
        for j, h in enumerate(ngram_hashes(ids, n).tolist()):
            ref_grams.setdefault(h, tuple(ids[j:j + n].tolist()))
    keys = np.array(sorted(ref_grams), dtype=np.uint64)
    table = np.zeros(1 << TABLE_BITS, bool)
    mask = np.uint64((1 << TABLE_BITS) - 1)
    if len(keys):
        table[(keys & mask).astype(np.int64)] = True
    counts = np.zeros(vocab, np.int64)
    per_source, hits = {}, {}
    total = sum(s['tokens'] for e in manifest['sources'].values() for s in e['train'])
    done, started = 0, time.time()
    for name, entry in manifest['sources'].items():
        per_source[name] = 0
        for shard in entry['train']:
            arr = np.memmap(RUN_DIR / shard['path'], dtype='<u2', mode='r')
            for start in range(0, len(arr), chunk):
                body = np.asarray(arr[start:start + chunk])
                counts += np.bincount(body, minlength=vocab)[:vocab]
                per_source[name] += len(body)
                if len(keys):
                    x = np.asarray(arr[start:start + chunk + n - 1])
                    h = ngram_hashes(x, n)
                    cand = np.nonzero(table[(h & mask).astype(np.int64)])[0]
                    if len(cand):
                        hc = h[cand]
                        pos = np.searchsorted(keys, hc)
                        pos[pos >= len(keys)] = 0
                        ok = keys[pos] == hc
                        for hv, p in zip(hc[ok].tolist(), cand[ok].tolist()):
                            gram = ref_grams[hv]
                            if tuple(x[p:p + n].tolist()) != gram:
                                continue                      # 哈希碰撞
                            rec = hits.setdefault(hv, {'ids': list(gram), 'positions': 0, 'sources': {}})
                            rec['positions'] += 1
                            rec['sources'][name] = rec['sources'].get(name, 0) + 1
                done += len(body)
            del arr
        log(f'[{stamp()}] 扫描 {name} 完成：累计 {done / 1e9:.2f}B / {total / 1e9:.2f}B token，'
            f'{(time.time() - started) / 60:.1f} 分钟，命中 n-gram {len(hits)}')
    return counts, per_source, hits


def contamination_summary(refs, n_items, hits, n, probed=200):
    """按题汇总：任一 n-gram 命中即"有重叠"；≥ 50% 的 n-gram 命中视为"强重叠"（基本整段出现）。"""
    hit_keys = set(hits)
    per_item = {}
    for i, field, ids in refs:
        hs = ngram_hashes(np.asarray(ids, np.int64), n).tolist()
        if not hs:
            continue
        found = sum(h in hit_keys for h in hs)
        d = per_item.setdefault(i, {})
        d[field] = {'ngrams': len(hs), 'found': found, 'frac': round(found / len(hs), 3)}

    def rates(items):
        items = list(items)
        if not items:
            return {}
        out = {'items': len(items)}
        for field in ('question', 'answer', 'answer_clean'):
            fr = [per_item[i][field]['frac'] for i in items if field in per_item[i]]
            fd = [per_item[i][field]['found'] for i in items if field in per_item[i]]
            if not fd:
                continue
            out[field] = {'any_overlap': round(float(np.mean([f > 0 for f in fd])), 4),
                          'strong_overlap': round(float(np.mean([f >= 0.5 for f in fr])), 4)}
        out['any_field_any_overlap'] = round(float(np.mean(
            [any(v['found'] > 0 for v in per_item[i].values()) for i in items])), 4)
        return out

    by_source = {}
    for rec in hits.values():
        for s, c in rec['sources'].items():
            by_source[s] = by_source.get(s, 0) + c
    worst = sorted(per_item.items(), key=lambda kv: -max(v['frac'] for v in kv[1].values()))[:10]
    return {'ngram': n, 'all_test': rates(per_item.keys()),
            f'probed_first_{probed}': rates(i for i in per_item if i < probed),
            'hit_positions_by_source': by_source, 'distinct_hit_ngrams': len(hits),
            'worst_items': [{'index': i, **v} for i, v in worst]}


# ---------------------------------------------------------------- 统计工具
def ranks(x):
    r = np.empty(len(x))
    r[np.argsort(x, kind='mergesort')] = np.arange(len(x))
    return r


def spearman(a, b):
    ra, rb = ranks(np.asarray(a, float)), ranks(np.asarray(b, float))
    return float(np.corrcoef(ra, rb)[0, 1])


def auc(scores, labels):
    """labels 为 True 的样本得分高于 False 的概率（Mann-Whitney）。0.5 = 无区分力。"""
    scores, labels = np.asarray(scores, float), np.asarray(labels, bool)
    pos, neg = labels.sum(), (~labels).sum()
    if pos == 0 or neg == 0:
        return None
    r = ranks(scores) + 1
    return round(float((r[labels].sum() - pos * (pos + 1) / 2) / (pos * neg)), 4)


BUCKETS = [('0 次', 0, 1), ('1–99', 1, 100), ('100–999', 100, 1000), ('1千–1万', 1000, 10**4),
           ('1万–10万', 10**4, 10**5), ('10万–100万', 10**5, 10**6), ('≥100万', 10**6, 1 << 62)]


def freq_stats(counts):
    c = counts[SPECIAL:]
    total = int(c.sum())
    order = np.sort(c)[::-1]
    cum = np.cumsum(order) / max(total, 1)
    cover = {f'{int(q * 100)}%': int(np.searchsorted(cum, q) + 1) for q in (0.5, 0.9, 0.99)}
    nz = order[order > 0]
    lo, hi = 10, min(10000, len(nz))
    slope = float(np.polyfit(np.log(np.arange(lo, hi) + 1), np.log(nz[lo:hi]), 1)[0]) if hi - lo > 10 else None
    return {'total_tokens': total, 'vocab_non_special': len(c), 'zero': int((c == 0).sum()),
            'lt100': int((c < 100).sum()), 'lt1000': int((c < 1000).sum()),
            'tokens_to_cover': cover, 'zipf_slope_rank10_10000': round(slope, 3) if slope is not None else None,
            'buckets': {name: int(((c >= a) & (c < b)).sum()) for name, a, b in BUCKETS}}


# ---------------------------------------------------------------- 嵌入分析
def embedding_analysis(emb, counts, init_std, decay_factor):
    E = emb.float().cpu()
    V, d = E.shape
    norms = E.norm(dim=1).numpy()
    # 第一主成分（含均值方向，所有 token 共享）→ 投影掉
    top = torch.linalg.eigh(E.T @ E).eigenvectors[:, -1]
    Er = E - (E @ top)[:, None] * top[None, :]
    ids = np.arange(V)
    usable = ids >= SPECIAL
    zero = usable & (counts == 0)
    if zero.sum() >= 5:
        ref_ids, ref_desc = ids[zero], f'训练数据中出现 0 次的 {int(zero.sum())} 个 token'
    else:
        cand = ids[usable][np.argsort(counts[usable], kind='mergesort')[:20]]
        ref_ids, ref_desc = cand, f'出现次数最少的 20 个 token（出现 0 次的只有 {int(zero.sum())} 个）'
    centroid = Er[ref_ids].mean(0)
    indicator = torch.nn.functional.cosine_similarity(Er, centroid[None, :], dim=1).numpy()
    lc = np.log10(counts + 1)
    rare, common = usable & (counts < 100), usable & (counts >= 1000)
    sel = rare | common
    # 参照组本身会让指标"自证"，计算 AUC 时排除参照组
    sel_ex = sel & ~np.isin(ids, ref_ids)
    buckets = []
    for name, a, b in BUCKETS:
        m = usable & (counts >= a) & (counts < b)
        if m.sum():
            buckets.append({'bucket': name, 'tokens': int(m.sum()), 'norm_median': round(float(np.median(norms[m])), 4),
                            'norm_p10': round(float(np.percentile(norms[m], 10)), 4),
                            'norm_p90': round(float(np.percentile(norms[m], 90)), 4),
                            'indicator_median': round(float(np.median(indicator[m])), 4)})
    init_norm = init_std * math.sqrt(d)
    return {
        'norms': norms, 'indicator': indicator, 'ref_ids': ref_ids,
        'summary': {'d': d, 'init_norm_expected': round(init_norm, 4), 'decay_factor': round(decay_factor, 4),
                    'never_updated_norm_expected': round(init_norm * decay_factor, 4), 'reference': ref_desc,
                    'spearman_logcount_norm': round(spearman(lc[usable], norms[usable]), 4),
                    'spearman_logcount_indicator': round(spearman(lc[usable], indicator[usable]), 4),
                    'auc_rare_low_norm': auc(-norms[sel_ex], rare[sel_ex]),
                    'auc_rare_high_indicator': auc(indicator[sel_ex], rare[sel_ex]),
                    'rare_tokens(<100)': int(rare.sum()), 'common_tokens(>=1000)': int(common.sum()),
                    'buckets': buckets}}


def repeat_test(model, tok, token_ids, device, precision, batch):
    """少样本复述：模型能否原样输出给定 token。返回逐个结果。"""
    from .weight_edit import batch_generate
    head = tok.encode('原文：苹果\n复述：苹果\n原文：world\n复述：world\n原文：数据\n复述：数据\n原文：', add_bos=True)
    tail = tok.encode('\n复述：', add_bos=False)
    if not token_ids:
        return []
    prompts = torch.tensor([head + [int(t)] + tail for t in token_ids], dtype=torch.long)
    out = batch_generate(model, prompts, 3, device, precision, greedy=True, batch=batch).cpu().tolist()
    res = []
    for t, g in zip(token_ids, out):
        s = tok.decode([int(t)])
        ok = g[0] == int(t) or (bool(s.strip()) and '�' not in s and tok.decode(g).startswith(s))
        res.append({'id': int(t), 'text': s, 'output': tok.decode(g), 'ok': bool(ok)})
    return res


# ---------------------------------------------------------------- 主流程
def manifest_key(manifest):
    return hashlib.sha256(repr(sorted((k, [s['path'] for s in v['train']]) for k, v in
                                      manifest['sources'].items())).encode()).hexdigest()[:16]


def precompute(config, manifest, tok, vocab, log=print):
    """CPU 部分：频次 + 污染。已有同一份数据的缓存则直接读取。"""
    cfg = dict(DEFAULTS, **config.get('token_probe', {}))
    out = RUN_DIR / 'token_probe'
    out.mkdir(parents=True, exist_ok=True)
    key = manifest_key(manifest)
    meta = read_json(out / 'scan_meta.json', {})
    if meta.get('key') == key and meta.get('ngram') == cfg['ngram'] and (out / 'counts.npy').exists():
        log('复用已缓存的频次与污染扫描结果')
        return np.load(out / 'counts.npy'), meta
    started = now()
    errors = []
    try:
        refs, n_items = gsm8k_refs(tok, RUN_DIR / 'passk')
    except Exception as exc:  # noqa: BLE001
        refs, n_items = [], 0
        errors.append(f'GSM8K 读取失败：{type(exc).__name__}: {exc}')
    counts, per_source, hits = scan(manifest, refs, vocab, cfg['ngram'], cfg['chunk_tokens'], log)
    np.save(out / 'counts.npy', counts)
    meta = {'key': key, 'ngram': cfg['ngram'], 'time': stamp(), 'minutes': round((now() - started) / 60, 1),
            'per_source_tokens': per_source, 'errors': errors,
            'contamination': contamination_summary(refs, n_items, hits, cfg['ngram']) if refs else None,
            'hit_examples': [{'text': tok.decode(r['ids']), 'positions': r['positions'], 'sources': r['sources']}
                             for r in sorted(hits.values(), key=lambda r: -r['positions'])[:15]]}
    write_json(out / 'scan_meta.json', meta)
    return counts, meta


def plots(out, counts, norms, indicator):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except Exception:  # noqa: BLE001
        return []
    lc = np.log10(counts[SPECIAL:] + 1)
    files = []
    fig, ax = plt.subplots(figsize=(6, 4.5))
    nz = np.sort(counts[SPECIAL:])[::-1]
    nz = nz[nz > 0]
    ax.loglog(np.arange(1, len(nz) + 1), nz)
    ax.set_xlabel('rank'), ax.set_ylabel('count in training data'), ax.set_title('Token rank-frequency (Zipf)')
    fig.tight_layout(), fig.savefig(out / 'zipf.png', dpi=130), plt.close(fig)
    files.append('zipf.png')
    for name, y, label in (('norm_vs_freq.png', norms[SPECIAL:], 'embedding L2 norm'),
                           ('indicator_vs_freq.png', indicator[SPECIAL:], 'cos to unused-token centroid (PC1 removed)')):
        fig, ax = plt.subplots(figsize=(6, 4.5))
        ax.scatter(lc, y, s=2, alpha=0.3)
        ax.set_xlabel('log10(count + 1)'), ax.set_ylabel(label)
        fig.tight_layout(), fig.savefig(out / name, dpi=130), plt.close(fig)
        files.append(name)
    return files


def run_token_probe(config, bus, trainer, tok, device, manifest=None, log=print):
    cfg = dict(DEFAULTS, **config.get('token_probe', {}))
    out = RUN_DIR / 'token_probe'
    out.mkdir(parents=True, exist_ok=True)
    started = now()
    manifest = manifest or read_json(RUN_DIR / 'data' / 'tokens' / 'manifest.json')
    model = trainer.model
    vocab = model.config.vocab_size
    if bus is not None:
        bus.post(NAME, 'decision', 'progress', {'action': 'start'})
    counts, meta = precompute(config, manifest, tok, vocab, log)
    final = RUN_DIR / 'model' / 'final.pt'
    tokens_seen = None
    if final.exists():
        ck = torch.load(final, map_location='cpu', weights_only=True)
        model.load_state_dict(ck['model'])
        tokens_seen = ck.get('tokens_seen')
    emb = model.embedding.weight.detach()
    wd = config.get('train', {}).get('weight_decay', 0.1)
    try:
        steps = trainer.plan['total_steps']
        log_decay = sum(math.log1p(-trainer.lr_at(s) * wd) for s in range(steps))
        decay = math.exp(log_decay)
    except Exception:  # noqa: BLE001
        decay = float('nan')
    ana = embedding_analysis(emb, counts, 0.02, decay)
    norms, indicator = ana['norms'], ana['indicator']
    ids = np.arange(vocab)
    usable = ids >= SPECIAL
    rng = np.random.default_rng(20261105)

    def sample(mask, k):
        pool = ids[mask & usable]
        return sorted(rng.choice(pool, size=min(k, len(pool)), replace=False).tolist()) if len(pool) else []
    k = cfg['repeat_samples']
    sus_order = ids[usable][np.argsort(-indicator[usable])][:cfg['suspicious_top']].tolist()
    groups = {f'指标最可疑的 {len(sus_order)} 个': sus_order, '出现 0 次': sample(counts == 0, k),
              '1–99 次': sample((counts >= 1) & (counts < 100), k), '1千–1万次': sample((counts >= 1000) & (counts < 10**4), k),
              '≥10万次': sample(counts >= 10**5, k)}
    precision = trainer.plan.get('precision', 'bf16')
    repeat = {}
    for g, tids in groups.items():
        if tids:
            r = repeat_test(model, tok, tids, device, precision, cfg['batch'])
            repeat[g] = {'n': len(r), 'ok_rate': round(sum(x['ok'] for x in r) / len(r), 4),
                         'examples': [x for x in r if not x['ok']][:8] + [x for x in r if x['ok']][:3]}
    suspicious = [{'id': int(t), 'text': tok.decode([int(t)]), 'count': int(counts[t]),
                   'norm': round(float(norms[t]), 4), 'indicator': round(float(indicator[t]), 4)} for t in sus_order[:40]]
    train_total = int(counts.sum())
    result = {'time': stamp(), 'minutes': round((now() - started) / 60, 1), 'config': cfg,
              'freq': freq_stats(counts), 'tokens_seen': tokens_seen, 'train_unique_tokens': train_total,
              'epochs_equiv': round(tokens_seen / train_total, 3) if tokens_seen and train_total else None,
              'embedding': ana['summary'], 'suspicious': suspicious, 'repeat': repeat,
              'contamination': meta.get('contamination'), 'hit_examples': meta.get('hit_examples'),
              'scan_minutes': meta.get('minutes'), 'errors': meta.get('errors', [])}
    result['plots'] = plots(out, counts, norms, indicator)
    np.save(out / 'embedding_norms.npy', norms)
    np.save(out / 'indicator.npy', indicator)
    write_json(out / 'result.json', result)
    write_text(out / 'REPORT.md', render_report(result))
    return result


def verdict(result):
    e, rep = result['embedding'], result['repeat']
    a1, a2 = e.get('auc_rare_low_norm'), e.get('auc_rare_high_indicator')
    best = max([x for x in (a1, a2) if x is not None] or [0.5])
    sus = next((v for g, v in rep.items() if g.startswith('指标')), None)
    freq = rep.get('≥10万次')
    parts = [f'用真实频次检验：范数 AUC {a1}，方向指标 AUC {a2}（≥0.8 说明权重本身就能认出罕见 token）']
    if sus and freq:
        parts.append(f'复述成功率：最可疑组 {sus["ok_rate"]:.0%}，高频组 {freq["ok_rate"]:.0%}')
    parts.append('欠训练 token 明显' if best >= 0.8 and sus and freq and sus['ok_rate'] + 0.3 < freq['ok_rate']
                 else '欠训练迹象有限或指标区分力不足')
    return '；'.join(parts)


def render_report(r):
    f, e, c = r['freq'], r['embedding'], r.get('contamination')
    L = ['# token 频次 / 欠训练 token / 评测污染', '',
         f'- 时间：{r["time"]}，用时 {r["minutes"]} 分钟（其中数据扫描 {r.get("scan_minutes")} 分钟，可能来自缓存）',
         f'- 训练数据唯一 token {r["train_unique_tokens"]:,}，训练共看过 {r["tokens_seen"] or 0:,} token，'
         f'约 {r["epochs_equiv"]} 遍（各来源采样权重不同，单个 token 的实际次数 ≈ 频次 × 该来源遍数）', '',
         '## 1. 频次分布', '',
         f'- 非特殊 token {f["vocab_non_special"]:,} 个；出现 0 次 {f["zero"]}，<100 次 {f["lt100"]}，<1000 次 {f["lt1000"]}',
         f'- 覆盖语料 50% / 90% / 99% 所需 token 数：{f["tokens_to_cover"]["50%"]:,} / {f["tokens_to_cover"]["90%"]:,} / '
         f'{f["tokens_to_cover"]["99%"]:,}',
         f'- Zipf 斜率（秩 10–10000，log-log）：{f["zipf_slope_rank10_10000"]}（自然语言的词约 −1；BPE 合并会让它变平）', '',
         '| 频次区间 | token 数 |', '|---|---|'] + [f'| {k} | {v:,} |' for k, v in f['buckets'].items()]
    L += ['', '## 2. 欠训练 token', '',
          f'- 预期：初始化范数 ≈ 0.02×√{e["d"]} = {e["init_norm_expected"]}；从未拿到梯度的行只受权重衰减，'
          f'∏(1−lr·wd) = {e["decay_factor"]} → 约 {e["never_updated_norm_expected"]}。'
          '绑定嵌入下罕见 token 仍会被 softmax 统一下压（Adam 会把这个微小但方向一致的梯度放大），所以范数可能偏小也可能偏大，'
          '更稳定的特征是"彼此很像"。',
          f'- 方向指标参照组：{e["reference"]}',
          f'- Spearman(log 频次, 范数) = {e["spearman_logcount_norm"]}；Spearman(log 频次, 方向指标) = {e["spearman_logcount_indicator"]}',
          f'- 罕见（<100 次，{e["rare_tokens(<100)"]} 个）vs 常见（≥1000 次，{e["common_tokens(>=1000)"]} 个）的区分力：'
          f'范数越小越罕见 AUC = {e["auc_rare_low_norm"]}；方向指标越高越罕见 AUC = {e["auc_rare_high_indicator"]}'
          '（0.5 = 无区分力；<0.5 表示方向相反）', '',
          '| 频次区间 | token 数 | 范数中位数 | 范数 P10–P90 | 方向指标中位数 |', '|---|---|---|---|---|']
    L += [f'| {b["bucket"]} | {b["tokens"]:,} | {b["norm_median"]} | {b["norm_p10"]}–{b["norm_p90"]} | {b["indicator_median"]} |'
          for b in e['buckets']]
    L += ['', '### 复述测试（少样本"原文：X\\n复述："，贪心）', '', '| 组 | 数量 | 复述成功率 |', '|---|---|---|']
    L += [f'| {g} | {v["n"]} | {v["ok_rate"]:.0%} |' for g, v in r['repeat'].items()]
    L += ['', '失败样例：', '']
    for g, v in r['repeat'].items():
        bad = [x for x in v['examples'] if not x['ok']][:4]
        if bad:
            L.append(f'- {g}：' + '；'.join(f'`{x["text"]!r}` → `{x["output"]!r}`' for x in bad))
    L += ['', '### 方向指标最可疑的 token（前 20）', '', '| id | 文本 | 频次 | 范数 | 指标 |', '|---|---|---|---|---|']
    L += [f'| {s["id"]} | `{s["text"]!r}` | {s["count"]:,} | {s["norm"]} | {s["indicator"]} |' for s in r['suspicious'][:20]]
    L += ['', f'**结论**：{verdict(r)}', '', '## 3. GSM8K 评测污染（训练数据中的 13-gram 重叠）', '']
    if not c:
        L.append('- 未完成（' + '；'.join(r.get('errors') or ['无 GSM8K 数据']) + '）')
    else:
        for title, key in (('全部 test 题', 'all_test'), ('pass@k 实际用到的前 200 题', 'probed_first_200')):
            s = c.get(key) or {}
            if not s:
                continue
            L += [f'**{title}**（{s["items"]} 题）：任一字段有重叠 {s["any_field_any_overlap"]:.1%}', '',
                  '| 字段 | 有重叠（≥1 个 13-gram） | 强重叠（≥50% 的 13-gram） |', '|---|---|---|']
            L += [f'| {fld} | {s[fld]["any_overlap"]:.1%} | {s[fld]["strong_overlap"]:.1%} |'
                  for fld in ('question', 'answer', 'answer_clean') if fld in s]
            L.append('')
        L += [f'- 命中位置按来源：{c["hit_positions_by_source"] or "无"}；不同的命中 13-gram {c["distinct_hit_ngrams"]} 个']
        if r.get('hit_examples'):
            L += ['- 出现最多的命中片段：'] + [f'  - `{h["text"]!r}` ×{h["positions"]}（{h["sources"]}）'
                                           for h in r['hit_examples'][:8]]
        L += ['', '判读：零星的"有重叠"多为通用短语（如 "How much money does she have left"），'
              '强重叠（题目或答案过半出现）才说明题目被收录；若强重叠比例不低，pass@k 中 GSM8K 的分数要打折看。']
    if r.get('plots'):
        L += ['', '## 图', ''] + [f'![{p}]({p})' for p in r['plots']]
    return '\n'.join(L) + '\n'


# ---------------------------------------------------------------- 入口
def _load_tok_and_manifest():
    from .data_agent import load_bbpe
    manifest = read_json(RUN_DIR / 'data' / 'tokens' / 'manifest.json')
    return load_bbpe(RUN_DIR / manifest['tokenizer']['path']), manifest


def smoke():
    import tempfile
    from pathlib import Path
    from ..config import ModelConfig
    from ..model import DecoderLM

    class Tok:
        def encode(self, text, add_bos=False):
            return ([2] if add_bos else []) + [5 + (ord(ch) % 250) for ch in text]

        def decode(self, ids):
            return ''.join(chr(97 + (int(i) % 26)) for i in ids)

    rng = np.random.default_rng(0)
    vocab = 256
    tmp = Path(tempfile.mkdtemp(dir=RUN_DIR))
    planted = list(range(10, 40))                       # 30 个 token，含 18 个 13-gram
    data = rng.integers(4, 200, 5000).astype('<u2')     # id 200–255 从不出现
    data[1234:1264] = planted
    data[3000:3030] = planted
    (tmp / 'a.bin').write_bytes(data.tobytes())
    (tmp / 'b.bin').write_bytes(rng.integers(4, 200, 3000).astype('<u2').tobytes())
    manifest = {'sources': {'s1': {'train': [{'path': str((tmp / 'a.bin').relative_to(RUN_DIR)), 'tokens': 5000}]},
                            's2': {'train': [{'path': str((tmp / 'b.bin').relative_to(RUN_DIR)), 'tokens': 3000}]}}}
    refs = [(0, 'question', planted[:20]), (0, 'answer', planted[5:30]), (1, 'question', list(range(210, 240)))]
    counts, per_source, hits = scan(manifest, refs, vocab, 13, chunk=1000)
    assert counts.sum() == 8000 and counts[200:].sum() == 0 and per_source == {'s1': 5000, 's2': 3000}
    c = contamination_summary(refs, 2, hits, 13, probed=1)
    assert c['all_test']['question']['any_overlap'] == 0.5 and c['probed_first_1']['question']['strong_overlap'] == 1.0, c
    assert sum(r['positions'] for r in hits.values()) >= 2 * 8 and set(c['hit_positions_by_source']) == {'s1'}
    assert auc([3, 2, 1], [True, False, False]) == 1.0 and abs(spearman([1, 2, 3], [2, 4, 9]) - 1) < 1e-9
    model = DecoderLM(ModelConfig(vocab_size=vocab, hidden_size=64, num_layers=2, num_heads=4, num_kv_heads=2,
                                  intermediate_size=128, max_seq_len=256))
    ana = embedding_analysis(model.embedding.weight.detach(), counts, 0.02, 0.5)
    assert ana['summary']['init_norm_expected'] == round(0.02 * 8, 4)
    rep = repeat_test(model, Tok(), [50, 60, 220], torch.device('cpu'), 'fp32', 8)
    assert len(rep) == 3 and all('ok' in x for x in rep)
    print('smoke ok', ana['summary']['buckets'][:2], c['all_test'])

    class T:
        pass
    t = T()
    t.model, t.plan, t.lr_at = model, {'precision': 'fp32', 'total_steps': 100}, (lambda s: 4e-4)
    cfg = {'train': {'weight_decay': 0.1},
           'token_probe': {'chunk_tokens': 1000, 'repeat_samples': 5, 'suspicious_top': 5, 'batch': 8}}
    res = run_token_probe(cfg, None, t, Tok(), torch.device('cpu'), manifest=manifest)
    assert res['freq']['zero'] == 56 and abs(res['embedding']['decay_factor'] - (1 - 4e-5) ** 100) < 1e-4, res['freq']
    print((RUN_DIR / 'token_probe' / 'REPORT.md').read_text()[:3000])


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--precompute', action='store_true', help='只做 CPU 部分（频次 + 污染），结果缓存供训练后阶段复用')
    a = p.parse_args()
    if a.smoke:
        smoke()
    elif a.precompute:
        from .common import load_config
        tok, manifest = _load_tok_and_manifest()
        cfg = load_config()
        precompute(cfg, manifest, tok, cfg.get('token_probe', {}).get('vocab', manifest['tokenizer']['vocab_size']))
