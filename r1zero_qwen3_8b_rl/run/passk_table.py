import sys, glob, os
sys.path.insert(0, '/root/autodl-tmp/r1zero/sft'); sys.path.insert(0, '/root/autodl-tmp/r1zero/ma')
import numpy as np
from math import comb
import eval_e as E
def passk(v, k):
    n, c = len(v), int(sum(v))
    return 1.0 if n - c < k else 1 - comb(n - c, k) / comb(n, k)
runs = [('46', 'r2_best_v2'), ('76', 'r2_last_v2'), ('116', 'r3_upd8dyn_last'), ('136', 'r3_lr2dyn_last'), ('148', 'r3_frontier_best'), ('156', 'r3_frontier_last')]
print('step | set: pass@1 / pass@4 / pass@16 / (题数 0<c<16 的比例, 全错比例)')
for step, d in runs:
    f = sorted(glob.glob('/root/r1zero_valgen/final_final_eval_%s/*.jsonl' % d))[0]
    Z = E.load_r1zero(f)
    row = []
    for ds in ['amc23', 'aime24', 'aime25', 'aime26']:
        vs = [v for k, v in Z.items() if k[0] == ds]
        p = [np.mean([passk(v, k) for v in vs]) for k in (1, 4, 16)]
        mixed = np.mean([0 < sum(v) < len(v) for v in vs]); zero = np.mean([sum(v) == 0 for v in vs])
        row.append('%s %.1f/%.1f/%.1f (%.0f%%,%.0f%%)' % (ds, 100*p[0], 100*p[1], 100*p[2], 100*mixed, 100*zero))
    print(step, '|', ' ; '.join(row), flush=True)
