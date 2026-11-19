"""E 阶段评测：SFT 后的模型（HF 目录）在 MATH-500 / AMC23 / AIME24 / AIME25 / AIME 2026 上生成、打分，并和 R1-Zero 配对比较。
  gen  ：python sft/eval_e.py gen --model DIR --out DIR --shard i --nshards 4 [--limit N] [--max-new 16384]
  score：python sft/eval_e.py score --out DIR [--compare FILE ...]
口径（2026-11-19 定，和 R1-Zero 的完整评测对齐）：
  * 提示 = 题目 + "Please reason step by step, and put your final answer within \boxed{}."，套 Qwen3 chat 模板（让模型自己写 <think>）；
  * temperature 0.6、top_p 0.95，最长生成 16384 token；AMC/AIME 每题 16 次（mean@16），MATH-500 每题 1 次；
  * 判分用 ma/reward_v2（只看 </think> 之后的部分）；R1-Zero 的生成用同一个判分函数重新判一遍再比较；
  * 主指标：四个原评测集等权平均，按题目配对 bootstrap 10000 次的 95% 区间（下限 > 0 显著提升，上限 < 0 显著下降，否则无显著差异）；
    AIME 2026（留出集）单独报告。
  * 另外统计可用性：正常结束（写完 </think> 且自己停下）的比例、含中文字符的比例、同一行重复 ≥5 次的比例、平均长度。
"""
import argparse
import collections
import glob
import json
import os
import re
import sys

import numpy as np

R = '/root/autodl-tmp/r1zero'
INSTR = 'Please reason step by step, and put your final answer within \\boxed{}.'
SETS4 = ['math500', 'amc23', 'aime24', 'aime25']
ALL5 = SETS4 + ['aime26']
NAMES = {'math500': 'MATH-500', 'amc23': 'AMC23', 'aime24': 'AIME24', 'aime25': 'AIME25', 'aime26': 'AIME 2026（留出）'}
CJK = re.compile(r'[㐀-鿿]')


def eval_rows():
    import pyarrow.parquet as pq
    out = []
    for f in ['val.parquet', 'test_aime26.parquet']:
        for r in pq.read_table(os.path.join(R, 'run/data', f), columns=['data_source', 'reward_model', 'extra_info']).to_pylist():
            out.append(dict(ds=r['data_source'], q=r['extra_info']['question'], gt=str(r['reward_model']['ground_truth'])))
    return out


def gen(a):
    from transformers import AutoTokenizer
    import sglang as sgl
    items = eval_rows()
    if a.limit:
        cnt, keep = collections.Counter(), []
        for it in items:
            if cnt[it['ds']] < a.limit:
                cnt[it['ds']] += 1
                keep.append(it)
        items = keep
    items = items[a.shard::a.nshards]
    tok = AutoTokenizer.from_pretrained(a.model)
    prompts = [tok.apply_chat_template([{'role': 'user', 'content': it['q'] + '\n\n' + INSTR}], tokenize=False, add_generation_prompt=True)
               for it in items]
    eng = sgl.Engine(model_path=a.model, tp_size=1, mem_fraction_static=0.85, context_length=a.max_new + 2048,
                     random_seed=1234 + a.shard, log_level='warning')
    sp = {'temperature': 0.6, 'top_p': 0.95, 'max_new_tokens': a.max_new, 'stop_token_ids': [151645, 151643], 'skip_special_tokens': False}
    outs = eng.generate(prompt=prompts, sampling_params=sp)
    tmp = os.path.join(a.out, 'gen_%d.jsonl.tmp' % a.shard)
    with open(tmp, 'w') as fo:
        for it, o in zip(items, outs):
            mi = o.get('meta_info', {}) or {}
            fr = mi.get('finish_reason')
            fr = fr.get('type') if isinstance(fr, dict) else fr
            fo.write(json.dumps(dict(it, output=o['text'], finish=fr, n_tok=mi.get('completion_tokens')), ensure_ascii=False) + '\n')
    os.replace(tmp, os.path.join(a.out, 'gen_%d.jsonl' % a.shard))
    print('GEN_DONE', len(items), flush=True)
    eng.shutdown()


def rep_line(x):
    lines = [l.strip() for l in x.split('\n') if len(l.strip()) > 20]
    c = collections.Counter(lines)
    return bool(c) and c.most_common(1)[0][1] >= 5


def judge(ds, text, gt):
    from reward_v2 import compute_score
    try:
        return float(compute_score(ds, text, gt).get('acc', 0.0))
    except Exception:  # noqa: BLE001
        return 0.0


def load_e(out):
    rows = []
    for f in sorted(glob.glob(os.path.join(out, 'gen_*.jsonl'))):
        rows += [json.loads(l) for l in open(f)]
    for r in rows:
        o = r['output'].replace('<|im_end|>', '').replace('<|endoftext|>', '')
        tail = o.split('</think>')[-1] if '</think>' in o else o[-3000:]
        r['acc'] = judge(r['ds'], tail, r['gt'])
        r['complete'] = r.get('finish') == 'stop' and '</think>' in o
        r['cjk'] = bool(CJK.search(o))
        r['rep'] = rep_line(o)
    return rows


def load_r1zero(path):
    qmap = {(x['q']): (x['ds'], x['gt']) for x in eval_rows()}
    per = collections.defaultdict(list)
    for l in open(path):
        r = json.loads(l)
        try:
            q = r['input'].split('the answer to the problem.\n\n', 1)[1].rsplit('\n\nRemember to put', 1)[0]
        except IndexError:
            continue
        if q in qmap:
            ds, gt = qmap[q]
            per[(ds, q)].append(judge(ds, r['output'], gt))
    return per


def paired(E, Z, n=10000, seed=0):
    rng = np.random.default_rng(seed)
    res, diffs = {}, {}
    for ds in ALL5:
        ks = sorted(k for k in E if k[0] == ds and k in Z)
        if ks:
            diffs[ds] = np.array([np.mean(E[k]) - np.mean(Z[k]) for k in ks])
    boots = {}
    for ds, d in diffs.items():
        boots[ds] = d[rng.integers(0, len(d), (n, len(d)))].mean(1)
        res[ds] = dict(n=len(d), diff=float(d.mean()), lo=float(np.percentile(boots[ds], 2.5)), hi=float(np.percentile(boots[ds], 97.5)))
    if all(ds in diffs for ds in SETS4):
        bm = np.mean([boots[ds] for ds in SETS4], axis=0)
        res['macro4'] = dict(diff=float(np.mean([diffs[ds].mean() for ds in SETS4])), lo=float(np.percentile(bm, 2.5)), hi=float(np.percentile(bm, 97.5)))
    return res


def verdict(x):
    return '显著提升' if x['lo'] > 0 else ('显著下降' if x['hi'] < 0 else '无显著差异')


def score(a):
    sys.path.insert(0, os.path.join(R, 'ma'))
    rows = load_e(a.out)
    by = collections.defaultdict(list)
    for r in rows:
        by[r['ds']].append(r)
    summ = {}
    for ds in ALL5:
        rs = by.get(ds, [])
        if not rs:
            continue
        summ[ds] = dict(rows=len(rs), problems=len({r['q'] for r in rs}), acc=float(np.mean([r['acc'] for r in rs])),
                        complete=float(np.mean([r['complete'] for r in rs])), cjk=float(np.mean([r['cjk'] for r in rs])),
                        rep=float(np.mean([r['rep'] for r in rs])), mean_tokens=float(np.mean([r.get('n_tok') or 0 for r in rs])))
    macro4 = float(np.mean([summ[s]['acc'] for s in SETS4])) if all(s in summ for s in SETS4) else None
    E = collections.defaultdict(list)
    for r in rows:
        E[(r['ds'], r['q'])].append(r['acc'])
    cmps = {}
    files = a.compare or sorted(glob.glob('/root/r1zero_valgen/final_final_eval_r3_frontier_*/*.jsonl')) or sorted(glob.glob('/root/r1zero_valgen/final_final_eval_r3_lr2dyn_*/*.jsonl'))
    for f in files:
        Z = load_r1zero(f)
        zs = {ds: float(np.mean([np.mean(v) for k, v in Z.items() if k[0] == ds])) for ds in ALL5 if any(k[0] == ds for k in Z)}
        cmps[f] = dict(r1zero=zs, paired=paired(E, Z))
    res = dict(summary=summ, macro4=macro4, compare=cmps)
    json.dump(res, open(os.path.join(a.out, 'summary.json'), 'w'), indent=1, ensure_ascii=False)
    L = ['# E 阶段（蒸馏 SFT）评测', '', '| 评测集 | 准确率 | 题数 × 次数 | 正常结束 | 含中文 | 行重复 | 平均长度 |', '|---|---|---|---|---|---|---|']
    for ds in ALL5:
        if ds in summ:
            s = summ[ds]
            L.append('| %s | %.1f%% | %d × %d | %.1f%% | %.1f%% | %.1f%% | %d |' % (NAMES[ds], 100 * s['acc'], s['problems'], round(s['rows'] / max(1, s['problems'])),
                                                                         100 * s['complete'], 100 * s['cjk'], 100 * s['rep'], s['mean_tokens']))
    if macro4 is not None:
        L += ['', '四个原评测集等权平均：%.2f%%' % (100 * macro4)]
    for f, c in cmps.items():
        L += ['', '## 和 R1-Zero 比较：%s' % f, '', '| 评测集 | R1-Zero | 差值（E − R1-Zero） | 95% 区间 | 题数 |', '|---|---|---|---|---|']
        for ds in ALL5 + ['macro4']:
            p = c['paired'].get(ds)
            if not p:
                continue
            z = c['r1zero'].get(ds)
            L.append('| %s | %s | %+.1f | [%+.1f, %+.1f] | %s |' % (NAMES.get(ds, '四集平均'), '%.1f%%' % (100 * z) if z is not None else '-',
                                                                 100 * p['diff'], 100 * p['lo'], 100 * p['hi'], p.get('n', '')))
        if 'macro4' in c['paired']:
            L += ['', '结论（四集平均，事先定的标准）：' + verdict(c['paired']['macro4'])]
    open(os.path.join(a.out, 'summary.md'), 'w').write('\n'.join(L) + '\n')
    print('\n'.join(L))
    print('SCORE_DONE')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('cmd')
    p.add_argument('--model')
    p.add_argument('--out', required=True)
    p.add_argument('--shard', type=int, default=0)
    p.add_argument('--nshards', type=int, default=1)
    p.add_argument('--limit', type=int, default=0)
    p.add_argument('--max-new', type=int, default=16384)
    p.add_argument('--compare', nargs='*')
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    gen(a) if a.cmd == 'gen' else score(a)
