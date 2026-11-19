"""数据工程（CPU 阶段，与路线 A 并行运行，不占 GPU）。

对路线 A 实际会用到的数据做三项统计，并给第 2 段（退火）生成剔除清单：
  1. 文档长度分布（v1 tokenizer）：有多少 token 位于文档第 1024 / 4096 个 token 之后，即扩上下文时真正有长程信号的比例；
  2. 跨文件近似重复：按 cpt.build_stage 的选文件规则，推算第 1 段、第 2 段会用哪些文件（第 2 段多算 25% 余量）；
     数学和行业语料另外加上 v1 用过的文件（上游没有确认做过全局去重，也最可能和 v1 重复）；
  3. 评测污染：v1 的验证 / 测试文档（路线 A 的评估就用它们）在路线 A 训练数据里有没有近似重复。
方法：与正式构建完全相同的清洗和文件内去重；MinHash 112 个哈希，5-gram（中文按字，其余按词）；按来源做 LSH（14 段 × 8 行），
候选对用签名一致率 ≥ 0.8 验证（≈ Jaccard 相似度 0.8），并查集聚类，每簇保留最早出现的一篇（评测文档 < v1 < 第 1 段 < 第 2 段）。
输出（/root/autodl-tmp/data_eng）：REPORT.md、stats.json、doc_lengths.json；
  stage2_dedup.json：每个来源在第 2 段是否启用去重（规则：第 2 段可剔除比例按字符计 ≥ 3%，用户 2026-11-17 同意）；
  stage2_exclude.npz：要剔除的文档（来源、文档哈希、文件编号、原因）。与 v1 验证/测试文档近似重复的一律剔除。
每个文件的签名缓存在 /dev/shm/data_eng（内存盘，不占磁盘），可断点续跑；全部完成后清掉缓存。
"""

from collections import Counter, defaultdict
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
import json
import multiprocessing
import os
from pathlib import Path
import random
import re
import shutil
import sys
import time
import traceback
import zlib

import numpy as np

OUT = Path(os.environ.get('DE_OUT', '/root/autodl-tmp/data_eng'))
SHM = Path(os.environ.get('DE_SHM', '/dev/shm/data_eng'))
CACHE, DL = SHM / 'cache', SHM / 'downloads'
V1 = Path('/root/autodl-tmp/a800_pretrain_v1')
AB = Path('/root/autodl-tmp/a800_tok_ab')
V1_TOK_HF = V1 / 'tokenizer' / 'tokenizer_hf.json'
K, BANDS, ROWS, NGRAM = 112, 14, 8, 5
THRESH = 0.8
APPLY_RATE = 0.03
MAX_UNITS = 20000
CFG = {'workers': 32, 'parquet_gb': 24.0, 'stage2_margin': 1.25, 'v1_sources': ['math', 'zh_industry'], 'sources': None}
CFG.update(json.loads(os.environ.get('DE_CFG', '{}')))
REASONS = {1: '评测污染', 2: '与 v1 重复', 3: '与第 1 段重复', 4: '同段其他文件', 5: '同一文件内'}

_rng = np.random.default_rng(20261117)
HA = (_rng.integers(0, 2 ** 63, size=K, dtype=np.uint64) << np.uint64(1)) | np.uint64(1)
HB = _rng.integers(0, 2 ** 63, size=K, dtype=np.uint64)
MIX = (_rng.integers(0, 2 ** 63, size=NGRAM, dtype=np.uint64) << np.uint64(1)) | np.uint64(1)
BAND_MIX = (_rng.integers(0, 2 ** 63, size=ROWS, dtype=np.uint64) << np.uint64(1)) | np.uint64(1)
UNIT_RE = re.compile(r'[㐀-鿿豈-﫿]|[0-9a-z_]+')


def stamp():
    return time.strftime('%Y-%m-%d %H:%M:%S')


def log(msg):
    print(f'[{stamp()}] {msg}', flush=True)


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=1), encoding='utf-8')
    os.replace(tmp, path)


def read_json(path, default=None):
    path = Path(path)
    return json.loads(path.read_text(encoding='utf-8')) if path.exists() else default


# ------------------------------------------------------------------ MinHash ----------
def signature(text):
    us = UNIT_RE.findall(text.lower())[:MAX_UNITS]
    if not us:
        return None
    h = np.fromiter((zlib.crc32(u.encode('utf-8')) for u in us), dtype=np.uint64, count=len(us))
    n = len(h)
    if n >= NGRAM:
        g = h[:n - NGRAM + 1] * MIX[0]
        for j in range(1, NGRAM):
            g = g + h[j:n - NGRAM + 1 + j] * MIX[j]
    else:
        g = np.array([(h * MIX[:n]).sum()], dtype=np.uint64)
    g = np.unique(g >> np.uint64(32))
    v = (g[:, None] * HA[None, :] + HB[None, :]) >> np.uint64(32)
    return v.min(axis=0).astype(np.uint32)


def candidate_pairs(sig):
    """LSH 找候选对（每个桶：星形 + 链式连边），再用签名一致率验证。返回验证通过的 (lo, hi)。"""
    N = len(sig)
    empty = (np.zeros(0, np.int64), np.zeros(0, np.int64))
    if N < 2:
        return empty
    ar = np.arange(N)
    us, vs = [], []
    for b in range(BANDS):
        key = (sig[:, b * ROWS:(b + 1) * ROWS].astype(np.uint64) * BAND_MIX).sum(axis=1)
        order = np.argsort(key, kind='stable')
        ks = key[order]
        same = np.zeros(N, bool)
        same[1:] = ks[1:] == ks[:-1]
        m = np.flatnonzero(same)
        if len(m) == 0:
            continue
        start = np.maximum.accumulate(np.where(same, 0, ar))
        us += [order[start[m]], order[m - 1]]
        vs += [order[m], order[m]]
    if not us:
        return empty
    u, v = np.concatenate(us), np.concatenate(vs)
    lo, hi = np.minimum(u, v).astype(np.int64), np.maximum(u, v).astype(np.int64)
    pair = np.unique(lo * N + hi)
    lo, hi = pair // N, pair % N
    ok = np.zeros(len(lo), bool)
    for i in range(0, len(lo), 200_000):
        ok[i:i + 200_000] = (sig[lo[i:i + 200_000]] == sig[hi[i:i + 200_000]]).mean(axis=1) >= THRESH
    return lo[ok], hi[ok]


def components(a, b):
    """并查集；总把较大的根挂到较小的根下，所以每个簇的根就是簇里编号最小（最早出现）的文档。"""
    parent = {}

    def find(x):
        r = parent.setdefault(x, x)
        if r == x:
            return x
        root = r
        while parent[root] != root:
            root = parent[root]
        while parent[x] != root:
            parent[x], x = root, parent[x]
        return root

    for x, y in zip(a.tolist(), b.tolist()):
        rx, ry = find(x), find(y)
        if rx != ry:
            if rx < ry:
                parent[ry] = rx
            else:
                parent[rx] = ry
    return {x: find(x) for x in list(parent)}


# ------------------------------------------------------------------ 单文件任务（子进程） ----------
def sig_worker(job):
    from . import cleaning
    from .data_agent import _iter_texts
    out = Path(job['out'])
    if out.exists():
        return {'source': job['source'], 'index': job['index'], 'stage': job['stage'], 'cached': True}
    t0 = time.time()
    seen_docs, seen_paras = set(), set()
    hashes, lens, sigs, rows, nosig = [], [], [], 0, 0
    try:
        for raw in _iter_texts(job):
            rows += 1
            text, reason = cleaning.clean_text(raw, job['kind'])
            if reason:
                continue
            h = cleaning.doc_hash64(text)
            if h in seen_docs:
                continue
            paras = cleaning.paragraph_hashes(text)
            if len(paras) >= 3 and sum(p in seen_paras for p in paras) * 2 >= len(paras):
                continue
            seen_docs.add(h)
            seen_paras.update(paras)
            s = signature(text)
            if s is None:
                nosig += 1
                continue
            hashes.append(h)
            lens.append(len(text))
            sigs.append(s)
    finally:
        if job['format'] == 'parquet':
            for p in (Path(job['download_path']), Path(job['download_path'] + '.partial')):
                p.unlink(missing_ok=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name[:-4] + '.tmp.npz')
    np.savez(tmp, hashes=np.array(hashes, dtype=np.uint64), lens=np.array(lens, dtype=np.int64),
             sigs=np.stack(sigs) if sigs else np.zeros((0, K), np.uint32))
    os.replace(tmp, out)
    return {'source': job['source'], 'index': job['index'], 'stage': job['stage'], 'docs': len(hashes), 'rows': rows,
            'nosig': nosig, 'seconds': round(time.time() - t0, 1)}


# ------------------------------------------------------------------ 计划：路线 A 会用哪些文件 ----------
def plan():
    from .cpt import CPT_DEFAULT
    from .sources import SOURCES
    fl = read_json(V1 / 'data' / 'file_lists.json')
    v1_done = {}
    for p in (V1 / 'data' / 'receipts').glob('*/*.json'):
        r = read_json(p)
        v1_done.setdefault(r['source'], {})[r['index']] = r['stats']
    tpb = {}
    for name, files in fl.items():
        size = {f['index']: f['size'] for f in files}
        d = v1_done.get(name, {})
        tpb[name] = sum(x.get('tokens_train', 0) for x in d.values()) / max(1, sum(size[i] for i in d if i in size))
    used = set()
    for d in (V1 / 'data' / 'receipts', AB / 'data' / 'receipts'):
        for p in d.glob('*/*.json'):
            r = read_json(p)
            used.add((r['source'], r['index']))

    def select(st, used_, factor=1.0):
        picks = {}
        for name, w in st['weights'].items():
            target = int(st['tokens'] * w) * factor
            est, chosen = 0.0, []
            for f in fl[name]:
                if (name, f['index']) in used_:
                    continue
                if est >= target:
                    break
                est += f['size'] * tpb[name]
                chosen.append(f)
            picks[name] = chosen
        return picks

    s1 = select(CPT_DEFAULT['stage1'], used)
    used1 = used | {(n, f['index']) for n, fs in s1.items() for f in fs}
    s2 = select(CPT_DEFAULT['stage2'], used1)
    s2m = select(CPT_DEFAULT['stage2'], used1, CFG['stage2_margin'])
    specs = {s['name']: s for s in SOURCES}
    jobs, summary = [], {}
    for name in fl:
        if CFG['sources'] and name not in CFG['sources']:
            continue
        entries = []
        if name in CFG['v1_sources']:
            byidx = {f['index']: f for f in fl[name]}
            entries += [(0, byidx[i]) for i in sorted(v1_done.get(name, {})) if i in byidx]
        entries += [(1, f) for f in s1.get(name, [])]
        idx2 = {f['index'] for f in s2.get(name, [])}
        entries += [(2 if f['index'] in idx2 else 3, f) for f in s2m.get(name, [])]
        sp = specs[name]
        for stage, f in entries:
            jobs.append({'source': name, 'kind': sp['kind'], 'format': sp['format'], 'fields': sp['fields'], 'url': f['url'],
                         'path': f['path'], 'size': f['size'], 'index': f['index'], 'stage': stage,
                         'download_path': str(DL / name / f'{f["index"]:05d}.parquet'),
                         'out': str(CACHE / name / f'{f["index"]:05d}.npz')})
        summary[name] = {str(st): {'files': sum(1 for s_, _ in entries if s_ == st),
                                   'gb': round(sum(f['size'] for s_, f in entries if s_ == st) / 1e9, 2),
                                   'est_tokens_B': round(sum(f['size'] for s_, f in entries if s_ == st) * tpb[name] / 1e9, 2)}
                         for st in (0, 1, 2, 3)}
    return jobs, summary


# ------------------------------------------------------------------ 统计 ----------
def doc_lengths():
    """用对照实验已经按 v1 tokenizer 编码好的数据（和路线 A 同一批来源、相邻的文件）统计文档长度；它不在了就用 v1 验证集。"""
    path = OUT / 'doc_lengths.json'
    if path.exists():
        return read_json(path)
    shards, origin = {}, ''
    man = read_json(AB / 'data' / 'manifest.json')
    if man and all(Path(s['path']).exists() for e in man['sources'].values() for s in e.get('v1', [])):
        shards = {n: [s['path'] for s in e['v1']] for n, e in man['sources'].items() if n != 'arith'}
        origin = 'tokenizer 对照实验的训练数据（v1 编码）'
    else:
        vm = read_json(V1 / 'data' / 'tokens' / 'manifest.json')
        shards = {n: [str(V1 / s['path']) for s in e['val']] for n, e in vm['sources'].items()}
        origin = 'v1 验证集'
    res = {'_origin': origin}
    for name, paths in shards.items():
        Ls = []
        for p in paths:
            arr = np.memmap(p, dtype='<u2', mode='r')
            b = np.flatnonzero(arr == 2)
            if len(b) == 0:
                continue
            ends = np.append(b[1:], len(arr))
            L = ends - b - 2
            Ls.append(L[L > 0])
        if not Ls:
            continue
        L = np.concatenate(Ls).astype(np.int64)
        tot = float(L.sum())
        res[name] = {'docs': int(len(L)), 'mean': round(float(L.mean()), 1), 'median': int(np.median(L)),
                     'p90': int(np.percentile(L, 90)), 'p99': int(np.percentile(L, 99)),
                     'docs_gt_1024': round(float((L > 1024).mean()), 4), 'docs_gt_4096': round(float((L > 4096).mean()), 4),
                     'tokens_beyond_1024': round(float(np.maximum(L - 1024, 0).sum()) / tot, 4),
                     'tokens_beyond_4096': round(float(np.maximum(L - 4096, 0).sum()) / tot, 4)}
        log(f'文档长度 {name}: {res[name]}')
    write_json(path, res)
    return res


def eval_docs(name):
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(str(V1_TOK_HF))
    man = read_json(V1 / 'data' / 'tokens' / 'manifest.json')
    docs = []
    for split, code in (('val', -1), ('test', -2)):
        for s in man['sources'].get(name, {}).get(split, []):
            arr = np.fromfile(V1 / s['path'], dtype='<u2')
            b = np.flatnonzero(arr == 2)
            ends = np.append(b[1:], len(arr))
            for st, en in zip(b.tolist(), ends.tolist()):
                ids = arr[st + 1:en].tolist()
                if ids and ids[-1] == 3:
                    ids = ids[:-1]
                if ids:
                    docs.append((code, tok.decode(ids, skip_special_tokens=True)))
    return docs


def analyze_source(name, jobs_src):
    from . import cleaning
    stage_l, file_l, hash_l, len_l, sig_l = [], [], [], [], []
    ev = [(c, t) for c, t in eval_docs(name)]
    ev_sigs = [(c, t, signature(t)) for c, t in ev]
    ev_sigs = [x for x in ev_sigs if x[2] is not None]
    if ev_sigs:
        stage_l.append(np.full(len(ev_sigs), -1, np.int8))
        file_l.append(np.array([c for c, _, _ in ev_sigs], np.int64))
        hash_l.append(np.array([cleaning.doc_hash64(t) for _, t, _ in ev_sigs], np.uint64))
        len_l.append(np.array([len(t) for _, t, _ in ev_sigs], np.int64))
        sig_l.append(np.stack([s for _, _, s in ev_sigs]))
    missing = []
    for job in sorted(jobs_src, key=lambda j: (j['stage'], j['index'])):
        p = Path(job['out'])
        if not p.exists():
            missing.append(job['index'])
            continue
        z = np.load(p)
        n = len(z['hashes'])
        if n == 0:
            continue
        stage_l.append(np.full(n, job['stage'], np.int8))
        file_l.append(np.full(n, job['index'], np.int64))
        hash_l.append(z['hashes'])
        len_l.append(z['lens'].astype(np.int64))
        sig_l.append(z['sigs'])
    stage, fileidx = np.concatenate(stage_l), np.concatenate(file_l)
    hashes, lens, sig = np.concatenate(hash_l), np.concatenate(len_l), np.concatenate(sig_l)
    N = len(sig)
    a, b = candidate_pairs(sig)
    root = components(a, b)
    st = {'docs': {}, 'chars': {}, 'removable': {}, 'missing_files': missing, 'pairs': int(len(a)),
          'clusters': len(set(root.values()))}
    for s in (-1, 0, 1, 2, 3):
        m = stage == s
        st['docs'][str(s)] = int(m.sum())
        st['chars'][str(s)] = int(lens[m].sum())
    rem = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    excl_h, excl_f, excl_r = [], [], []
    members = defaultdict(list)
    for x, r in root.items():
        members[r].append(x)
        if x == r:
            continue
        sx, sr = int(stage[x]), int(stage[r])
        if sx == -1:
            continue
        if sr == -1:
            reason = 1
        elif sr == 0 and sx >= 1:
            reason = 2
        elif sr == 1 and sx >= 2:
            reason = 3
        else:
            reason = 5 if fileidx[x] == fileidx[r] else 4
        key = '2' if sx >= 2 else str(sx)
        if sx != 3:
            rem[key][reason][0] += 1
            rem[key][reason][1] += int(lens[x])
        if sx >= 2:
            excl_h.append(int(hashes[x]))
            excl_f.append(int(fileidx[x]))
            excl_r.append(reason)
    st['removable'] = {k: {str(rr): v for rr, v in d.items()} for k, d in rem.items()}
    hit = {'0': 0, '1': 0, '2': 0}
    for r, mem in members.items():
        stages = {int(stage[x]) for x in mem}
        if -1 not in stages:
            continue
        n_eval = sum(1 for x in mem if stage[x] == -1)
        for s in (0, 1, 2):
            if s in stages:
                hit[str(s)] += n_eval
    st['eval'] = {'docs': st['docs']['-1'], 'hit': hit}
    files = sorted({int(f) for f, s in zip(fileidx.tolist(), stage.tolist()) if s in (1, 2)})
    rng = random.Random(7)
    growth = {}
    for frac in (0.25, 0.5, 1.0):
        sub = set(files) if frac == 1.0 else set(rng.sample(files, max(1, int(len(files) * frac))))
        mask = np.isin(fileidx, list(sub)) & ((stage == 1) | (stage == 2))
        keep = mask[a] & mask[b]
        rr = components(a[keep], b[keep])
        growth[str(frac)] = {'docs': int(mask.sum()), 'removable': round(sum(1 for x, y in rr.items() if x != y) / max(1, int(mask.sum())), 4)}
    st['growth'] = growth
    excl = (np.array(excl_h, np.uint64), np.array(excl_f, np.int64), np.array(excl_r, np.int8))
    return st, excl


def stage2_decision(stats):
    out = {}
    for name, st in stats.items():
        chars = max(1, st['chars'].get('2', 0))
        rem = st['removable'].get('2', {})
        dup = sum(v[1] for r, v in rem.items() if r != '1')
        dup_docs = sum(v[0] for r, v in rem.items() if r != '1')
        ev_chars, ev_docs = rem.get('1', [0, 0])[1], rem.get('1', [0, 0])[0]
        apply_dup = dup / chars >= APPLY_RATE
        out[name] = {'stage2_docs': st['docs'].get('2', 0), 'dup_rate_chars': round(dup / chars, 4),
                     'dup_rate_docs': round(dup_docs / max(1, st['docs'].get('2', 0)), 4), 'eval_docs': ev_docs,
                     'eval_rate_chars': round(ev_chars / chars, 5), 'apply_dup': bool(apply_dup),
                     'drop_rate': round((ev_chars + (dup if apply_dup else 0)) / chars, 4)}
    return out


def pct(x):
    return f'{x:.2%}' if x is not None else '-'


def write_report(lens, summary, stats, decision, failed, hours):
    from .cpt import CPT_DEFAULT
    w2 = CPT_DEFAULT['stage2']['weights']
    L = ['# 数据工程（CPU 阶段）报告', '', f'- 生成时间：{stamp()}；耗时 {hours:.1f} 小时',
         '- 范围：路线 A 第 1 段、第 2 段（另加 25% 余量）会用到的文件；数学和行业语料另加 v1 用过的文件；v1 验证 / 测试文档用于检查评测污染',
         f'- 方法：与正式构建相同的清洗和文件内去重；MinHash {K} 个哈希、{NGRAM}-gram（中文按字，其余按词）；LSH {BANDS} 段 × {ROWS} 行；'
         f'签名一致率 ≥ {THRESH} 判为近似重复；并查集聚类，每簇保留最早出现的一篇', '']
    L += ['## 1. 文档长度（v1 tokenizer，来源：' + lens.get('_origin', '') + '）', '',
          '| 来源 | 文档数 | 平均 token | 中位数 | P90 | 超过 1024 的文档 | 超过 4096 的文档 | 位于 1024 之后的 token | 位于 4096 之后的 token |',
          '|---|---|---|---|---|---|---|---|---|']
    for n, d in lens.items():
        if n.startswith('_'):
            continue
        L.append(f'| {n} | {d["docs"]:,} | {d["mean"]} | {d["median"]} | {d["p90"]} | {pct(d["docs_gt_1024"])} | {pct(d["docs_gt_4096"])} | '
                 f'{pct(d["tokens_beyond_1024"])} | {pct(d["tokens_beyond_4096"])} |')
    ws = sum(w for n, w in w2.items() if n in lens)
    if ws:
        b1 = sum(lens[n]['tokens_beyond_1024'] * w for n, w in w2.items() if n in lens) / ws
        b4 = sum(lens[n]['tokens_beyond_4096'] * w for n, w in w2.items() if n in lens) / ws
        L += ['', f'- 按第 2 段配比加权：约 {b1:.1%} 的 token 位于所在文档第 1024 个 token 之后，{b4:.1%} 位于 4096 之后。'
              '扩到 4096 时，只有前一部分 token 能用上 1024 以外的上下文；其余窗口里是多篇文档拼在一起。']
    L += ['', '## 2. 跨文件近似重复（可剔除 = 簇里不是最早出现的那篇）', '',
          '| 来源 | 阶段 | 文档数 | 可剔除（文档） | 可剔除（字符） | 评测污染 | 与 v1 重复 | 与第 1 段重复 | 同段其他文件 | 同一文件内 |',
          '|---|---|---|---|---|---|---|---|---|---|']
    for n, st in stats.items():
        for s, label in (('0', 'v1（仅数学、行业）'), ('1', '第 1 段'), ('2', '第 2 段')):
            docs = st['docs'].get(s, 0)
            if not docs:
                continue
            rem = st['removable'].get(s, {})
            rd = sum(v[0] for v in rem.values())
            rc = sum(v[1] for v in rem.values())
            cols = [f'{rem.get(str(r), [0, 0])[0]:,}' for r in (1, 2, 3, 4, 5)]
            L.append(f'| {n} | {label} | {docs:,} | {rd / docs:.2%} | {rc / max(1, st["chars"].get(s, 1)):.2%} | ' + ' | '.join(cols) + ' |')
    L += ['', '## 3. 评测污染：v1 验证 / 测试文档在训练数据里有近似重复', '',
          '| 来源 | v1 验证+测试文档 | v1 训练（数学、行业） | 第 1 段 | 第 2 段 |', '|---|---|---|---|---|']
    for n, st in stats.items():
        e = st['eval']
        dn = max(1, e['docs'])
        L.append(f'| {n} | {e["docs"]:,} | {e["hit"]["0"]}（{e["hit"]["0"] / dn:.2%}） | {e["hit"]["1"]}（{e["hit"]["1"] / dn:.2%}） | '
                 f'{e["hit"]["2"]}（{e["hit"]["2"] / dn:.2%}） |')
    L += ['', '## 4. 重复率随数据规模增长（第 1+2 段的文件随机抽 25% / 50% / 全部）', '',
          '| 来源 | 25% | 50% | 全部 |', '|---|---|---|---|']
    for n, st in stats.items():
        g = st['growth']
        L.append(f'| {n} | {g["0.25"]["removable"]:.2%} | {g["0.5"]["removable"]:.2%} | {g["1.0"]["removable"]:.2%} |')
    L += ['', '- 近似重复是"成对"出现的，数据越多，找到配对的机会越多，所以小样本测出的重复率会低估全量的重复率。', '',
          '## 5. 第 2 段（退火）的处理决定', '',
          f'规则：某来源第 2 段的可剔除比例（按字符）≥ {APPLY_RATE:.0%} 时启用全局去重；与 v1 验证/测试文档近似重复的一律剔除（保持评估干净）。'
          '剔除后每个文件产出的 token 变少，第 2 段会按预计剔除比例多选文件，token 总量不变。', '',
          '| 来源 | 第 2 段文档 | 重复（按字符） | 评测污染文档 | 启用去重 | 预计剔除（按字符） |', '|---|---|---|---|---|---|']
    for n, d in decision.items():
        L.append(f'| {n} | {d["stage2_docs"]:,} | {d["dup_rate_chars"]:.2%} | {d["eval_docs"]} | {"是" if d["apply_dup"] else "否"} | {d["drop_rate"]:.2%} |')
    if failed:
        L += ['', '## 未处理的文件', ''] + [f'- {f["source"]} #{f["index"]}：{f["error"]}' for f in failed]
    L += ['', '## 文件范围', '', '| 来源 | v1 文件 | 第 1 段 | 第 2 段 | 第 2 段余量 |', '|---|---|---|---|---|']
    for n, s in summary.items():
        L.append(f'| {n} | ' + ' | '.join(f'{s[k]["files"]} 个 / {s[k]["est_tokens_B"]}B' for k in ('0', '1', '2', '3')) + ' |')
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'REPORT.md').write_text('\n'.join(L) + '\n', encoding='utf-8')


# ------------------------------------------------------------------ 主流程 ----------
def run_jobs(jobs):
    # 大文件先开工（单个文件只能由一个进程处理，最大的文件决定总耗时），小文件填空隙
    pending = sorted((j for j in jobs if not Path(j['out']).exists()), key=lambda j: -j['size'])
    total, done, failed, attempts = len(pending), 0, [], Counter()
    budget = CFG['parquet_gb'] * 1e9
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=CFG['workers'], mp_context=multiprocessing.get_context('spawn')) as ex:
        futs = {}
        while pending or futs:
            inflight = sum(j['size'] for j in futs.values() if j['format'] == 'parquet')
            i = 0
            while i < len(pending) and len(futs) < CFG['workers']:
                j = pending[i]
                if j['format'] == 'parquet' and inflight > 0 and inflight + j['size'] > budget:
                    i += 1
                    continue
                pending.pop(i)
                futs[ex.submit(sig_worker, j)] = j
                if j['format'] == 'parquet':
                    inflight += j['size']
            if not futs:
                break
            fin, _ = wait(list(futs), timeout=60, return_when=FIRST_COMPLETED)
            for f in fin:
                j = futs.pop(f)
                try:
                    f.result()
                    done += 1
                except Exception as exc:  # noqa: BLE001
                    attempts[(j['source'], j['index'])] += 1
                    if attempts[(j['source'], j['index'])] < 3:
                        pending.append(j)
                    else:
                        failed.append({'source': j['source'], 'index': j['index'], 'error': f'{type(exc).__name__}: {str(exc)[:200]}'})
                        log(f'放弃 {j["source"]} #{j["index"]}：{exc}')
            write_json(OUT / 'progress.json', {'time': stamp(), 'done': done, 'total': total, 'failed': len(failed),
                                               'running': len(futs), 'minutes': round((time.time() - t0) / 60, 1)})
    return failed


def run():
    OUT.mkdir(parents=True, exist_ok=True)
    for d in (CACHE, DL):
        d.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    try:
        log('开始：文档长度统计')
        lens = doc_lengths()
        jobs, summary = plan()
        write_json(OUT / 'plan.json', summary)
        log(f'计划：{len(jobs)} 个文件；' + json.dumps(summary, ensure_ascii=False))
        failed = run_jobs(jobs)
        log(f'签名完成，失败 {len(failed)} 个文件；开始分析')
        stats, excl = {}, {}
        by_src = defaultdict(list)
        for j in jobs:
            by_src[j['source']].append(j)
        for name, js in by_src.items():
            stats[name], excl[name] = analyze_source(name, js)
            log(f'{name}: ' + json.dumps({k: stats[name][k] for k in ('docs', 'removable', 'eval', 'growth')}, ensure_ascii=False))
        decision = stage2_decision(stats)
        arrays = {}
        for name, (h, f, r) in excl.items():
            if len(h):
                arrays[f'{name}__hash'], arrays[f'{name}__file'], arrays[f'{name}__reason'] = h, f, r
        np.savez(OUT / 'stage2_exclude.npz', **arrays)
        write_json(OUT / 'stats.json', stats)
        write_json(OUT / 'stage2_dedup.json', {'status': 'done', 'generated': stamp(), 'threshold': APPLY_RATE,
                                               'exclude_path': str(OUT / 'stage2_exclude.npz'), 'sources': decision,
                                               'failed_files': failed})
        write_report(lens, summary, stats, decision, failed, (time.time() - t0) / 3600)
        shutil.rmtree(SHM, ignore_errors=True)
        log('完成，报告见 ' + str(OUT / 'REPORT.md'))
        return 0
    except Exception as exc:  # noqa: BLE001
        write_json(OUT / 'FAILED.json', {'time': stamp(), 'error': f'{type(exc).__name__}: {exc}', 'traceback': traceback.format_exc()[-4000:]})
        log(traceback.format_exc())
        return 1


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'run':
        sys.exit(run())
    print(json.dumps(read_json(OUT / 'progress.json', {}), ensure_ascii=False))
