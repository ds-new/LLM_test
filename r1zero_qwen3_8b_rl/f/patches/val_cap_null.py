"""Null check: if F step 20 had E's per-question truncation probability, how unusual is F's AIME cap rate (1 sample per row)?"""
import json, re, math, random
from collections import defaultdict
import pyarrow.parquet as pq
R = '/root/autodl-tmp/r1zero'
def norm(s):
    return re.sub(r'\s+', ' ', (s or '')).strip()
cap = defaultdict(list)
for k in range(4):
    for l in open(R + '/sft/eval_full/gen_%d.jsonl' % k):
        r = json.loads(l)
        n = int(r['n_tok'] or 0)
        cap[(r['ds'], norm(r['q']))].append(1.0 if (n >= 16384 or r.get('finish') == 'length') else 0.0)
ps = []
for r in pq.read_table(R + '/run/data/val_select_f.parquet').to_pylist():
    if r['data_source'] in ('aime24', 'aime25'):
        key = (r['data_source'], norm((r.get('extra_info') or {}).get('question')))
        xs = cap[key]
        ps.append(sum(xs) / len(xs))
n = len(ps)
mu = sum(ps) / n
sd = math.sqrt(sum(p * (1 - p) for p in ps)) / n
rng = random.Random(0)
sims = sorted(sum(1 for p in ps if rng.random() < p) / n for _ in range(20000))
obs = 0.375
print('n', n, 'E mean cap %.3f, null SD %.3f, z for F20 %.2f, P(sim <= %.3f) = %.4f' % (mu, sd, (obs - mu) / sd, obs, sum(1 for s in sims if s <= obs) / len(sims)))
print('null 2.5%%/97.5%%: %.3f / %.3f' % (sims[int(0.025 * len(sims))], sims[int(0.975 * len(sims))]))
