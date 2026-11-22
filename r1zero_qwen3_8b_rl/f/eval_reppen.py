"""E 加重复惩罚的推理试验（2026-11-21 用户同意，约 30 元）：E 的模型在 AIME24 / AIME25 / AIME 2026 上，除了多一个 presence_penalty
以外，和 E 的完整评测（sft/eval_e.py：temperature 0.6、top_p 0.95、最长 16384 token、同样的提示）完全一样；每题 16 次里前 8 次用
presence_penalty 0.5、后 8 次用 1.5（Qwen3 官方建议的范围是 0-2）。目的：看 AIME 上约一半写不完的回答（其中 21-26% 是重复循环）
能不能在推理时直接减少，以及正确率会不会受影响。
说明：本机的 SGLang 0.5.5 接受 repetition_penalty 参数但采样器里没有实现（penaltylib 只有 frequency / presence / min_new_tokens），
frequency_penalty 按出现次数累加，长数学推导里数字和符号出现几百次，会被罚得没法用；所以用 presence_penalty（出现过的 token 统一减一个常数）。
  gen  ：python f/eval_reppen.py gen --model DIR --out DIR --shard i --nshards 4 [--dry]
  score：python f/eval_reppen.py score --out DIR [--selftest]   -> run/audit/E_PRESENCE_PENALTY_RESULT.md
基线：sft/eval_full/gen_*.jsonl（E，同题 16 次，无惩罚）。比较按题目配对（每题：试验组均值 - 基线均值），bootstrap 10000 次。"""
import argparse
import collections
import glob
import json
import os
import sys
import zlib

import numpy as np

R = '/root/autodl-tmp/r1zero'
sys.path.insert(0, R + '/sft')
sys.path.insert(0, R + '/ma')
import eval_e  # noqa: E402

SETS = ['aime24', 'aime25', 'aime26']
ARMS = [('pp0.5', 0.5), ('pp1.5', 1.5)]


def items_with_arms():
    cnt, out = collections.Counter(), []
    for it in eval_e.eval_rows():
        if it['ds'] not in SETS:
            continue
        k = (it['ds'], it['q'])
        i = cnt[k]
        cnt[k] += 1
        arm = ARMS[0] if i < 8 else ARMS[1]
        out.append(dict(it, arm=arm[0], pp=arm[1], rep_idx=i))
    return out


def gen(a):
    items = items_with_arms()[a.shard::a.nshards]
    if a.dry:
        print('shard', a.shard, 'items', len(items), collections.Counter((it['ds'], it['arm']) for it in items))
        return
    from transformers import AutoTokenizer
    import sglang as sgl
    tok = AutoTokenizer.from_pretrained(a.model)
    prompts = [tok.apply_chat_template([{'role': 'user', 'content': it['q'] + '\n\n' + eval_e.INSTR}], tokenize=False, add_generation_prompt=True)
               for it in items]
    eng = sgl.Engine(model_path=a.model, tp_size=1, mem_fraction_static=0.85, context_length=a.max_new + 2048,
                     random_seed=4321 + a.shard, log_level='warning')
    sps = [{'temperature': 0.6, 'top_p': 0.95, 'max_new_tokens': a.max_new, 'stop_token_ids': [151645, 151643],
            'skip_special_tokens': False, 'presence_penalty': it['pp']} for it in items]
    outs = eng.generate(prompt=prompts, sampling_params=sps)
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


def annotate(rows):
    from reward_f import compute_score
    for r in rows:
        o = r['output']
        try:
            r['acc'] = float(compute_score(r['ds'], o, r['gt'])['acc'])
        except Exception:  # noqa: BLE001
            r['acc'] = 0.0
        r['complete'] = float(r.get('finish') == 'stop' and '</think>' in o)
        lines = [l.strip() for l in o.split('\n') if len(l.strip()) > 20]
        c = collections.Counter(lines)
        top = c.most_common(1)[0][1] if c else 0
        tail = o[-8000:].encode('utf-8', 'ignore')
        r['loop'] = float(top >= 20 or len(zlib.compress(tail, 6)) / max(1, len(tail)) < 0.12)
        r['cjk'] = float(bool(eval_e.CJK.search(o)))
        r['tok'] = float(r.get('n_tok') or 0)
    return rows


def load(d):
    rows = []
    for f in sorted(glob.glob(os.path.join(d, 'gen_*.jsonl'))):
        rows += [json.loads(l) for l in open(f)]
    return [r for r in rows if r['ds'] in SETS]


def score(a):
    base = annotate(load(R + '/sft/eval_full'))
    if a.selftest:
        cnt, new = collections.Counter(), []
        for r in base:
            k = (r['ds'], r['q'])
            new.append(dict(r, arm=ARMS[0][0] if cnt[k] < 8 else ARMS[1][0]))
            cnt[k] += 1
    else:
        new = annotate(load(a.out))
    B = collections.defaultdict(list)
    for r in base:
        B[(r['ds'], r['q'])].append(r)
    rng = np.random.default_rng(0)
    L = ['# E 加 presence_penalty 的推理试验（AIME24 / AIME25 / AIME 2026，每题每组 8 次；基线 = E 完整评测的同题 16 次）', '',
         '除 presence_penalty 以外的设置和 E 的完整评测相同（temperature 0.6、top_p 0.95、最长 16384 token）。差值按题目配对，方括号是 bootstrap 95% 区间。', '']
    keys_all = sorted(B)
    for arm, pp in [(None, 0.0)] + ARMS:
        if arm is None:
            A = B
            L += ['## 基线（presence_penalty 0）', '']
        else:
            A = collections.defaultdict(list)
            for r in new:
                if r['arm'] == arm:
                    A[(r['ds'], r['q'])].append(r)
            L += ['', '## presence_penalty %s（%d 条）' % (pp, sum(len(v) for v in A.values())), '']
        L += ['| 评测集 | 正确率 | 写完的比例 | 重复循环 | 含中文 | 平均 token |' + (' 正确率差值 | 写完比例差值 |' if arm else ''),
              '|---|---|---|---|---|---|' + ('---|---|' if arm else '')]
        for ds in SETS + ['all']:
            ks = [k for k in keys_all if (ds == 'all' or k[0] == ds) and k in A]
            if not ks:
                continue
            m = lambda f, X=A, ks=ks: float(np.mean([np.mean([r[f] for r in X[k]]) for k in ks]))  # noqa: E731
            row = '| %s | %.1f%% | %.1f%% | %.1f%% | %.1f%% | %.0f |' % ('三套合计' if ds == 'all' else eval_e.NAMES[ds], 100 * m('acc'), 100 * m('complete'),
                                                                  100 * m('loop'), 100 * m('cjk'), m('tok'))
            if arm:
                for f in ['acc', 'complete']:
                    d = np.array([np.mean([r[f] for r in A[k]]) - np.mean([r[f] for r in B[k]]) for k in ks])
                    bt = d[rng.integers(0, len(d), (10000, len(d)))].mean(1)
                    row += ' %+.1f [%+.1f, %+.1f] |' % (100 * d.mean(), 100 * np.percentile(bt, 2.5), 100 * np.percentile(bt, 97.5))
            L.append(row)
    out = '/tmp/E_PRESENCE_PENALTY_selftest.md' if a.selftest else R + '/run/audit/E_PRESENCE_PENALTY_RESULT.md'
    open(out, 'w').write('\n'.join(L) + '\n')
    print('\n'.join(L))
    print('SCORE_DONE', out)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('cmd')
    p.add_argument('--model')
    p.add_argument('--out', required=True)
    p.add_argument('--shard', type=int, default=0)
    p.add_argument('--nshards', type=int, default=1)
    p.add_argument('--max-new', type=int, default=16384)
    p.add_argument('--dry', action='store_true')
    p.add_argument('--selftest', action='store_true')
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    gen(a) if a.cmd == 'gen' else score(a)
