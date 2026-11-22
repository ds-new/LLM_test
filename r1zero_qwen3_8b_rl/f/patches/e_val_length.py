"""Mean overlong penalty (reward - score) of E on the F validation set, per data source, from E's eval token counts
(sft/eval_full/gen_*.jsonl n_tok; penalty = -clip((n_tok - 12288)/4096, 0, 1), same as reward_model.overlong_buffer len 4096 / max 16384).
Also the share of samples that hit 16384 tokens (no </think> or truncated). Compare with F's val-aux reward/mean@1 - score/mean@1."""
import json, math, re, sys
from collections import defaultdict
import pyarrow.parquet as pq
R = '/root/autodl-tmp/r1zero'
def norm(s):
    return re.sub(r'\s+', ' ', (s or '')).strip()
pen_q = defaultdict(list); trunc_q = defaultdict(list)
for k in range(4):
    for l in open(R + '/sft/eval_full/gen_%d.jsonl' % k):
        r = json.loads(l)
        n = int(r['n_tok'])
        pen_q[(r['ds'], norm(r['q']))].append(-min(max((n - 12288) / 4096.0, 0.0), 1.0))
        trunc_q[(r['ds'], norm(r['q']))].append(1.0 if (n >= 16384 or r.get('finish') == 'length') else 0.0)
va = pq.read_table(R + '/run/data/val_select_f.parquet').to_pylist()
by = defaultdict(list)
for r in va:
    key = (r['data_source'], norm((r.get('extra_info') or {}).get('question')))
    if key in pen_q:
        by[r['data_source']].append(key)
F = {'math500': (0.671414794921875, 0.74), 'amc23': (0.261712646484375, 0.45), 'aime24': (-1.0333089192708333, -0.43333333333333335), 'aime25': (-0.9460734049479167, -0.4)}
for ds in ['math500', 'amc23', 'aime24', 'aime25']:
    keys = by[ds]; N = len(keys)
    pe = sum(sum(pen_q[k]) / len(pen_q[k]) for k in keys) / N
    tr = sum(sum(trunc_q[k]) / len(trunc_q[k]) for k in keys) / N
    allp = [p for k in keys for p in pen_q[k]]
    m = sum(allp) / len(allp); sd = math.sqrt(sum((p - m) ** 2 for p in allp) / len(allp))
    fpen = F[ds][0] - F[ds][1]
    print('%-8s N=%d  E penalty %.3f (E truncated %.1f%%)  F penalty %.3f  diff %+.3f  ~SD of F single-sample mean %.3f' % (ds, N, pe, 100 * tr, fpen, fpen - pe, sd / math.sqrt(N)))
