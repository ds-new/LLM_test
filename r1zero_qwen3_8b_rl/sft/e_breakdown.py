import sys, os, collections
from math import comb
import numpy as np
R = '/root/autodl-tmp/r1zero'
for p in (R, R + '/sft', R + '/ma'):
    sys.path.insert(0, p)
from eval_e import load_e
rows = load_e(R + '/sft/eval_full')
by = collections.defaultdict(list)
for r in rows:
    by[r['ds']].append(r)
for ds, rs in by.items():
    comp = [r for r in rs if r['complete']]
    inc = [r for r in rs if not r['complete']]
    acc_c = np.mean([r['acc'] for r in comp]) if comp else float('nan')
    acc_i = np.mean([r['acc'] for r in inc]) if inc else float('nan')
    rep_i = np.mean([r['rep'] for r in inc]) if inc else float('nan')
    per = collections.defaultdict(list)
    for r in rs:
        per[r['q']].append(r['acc'])
    def pk(k):
        v = []
        for a in per.values():
            n = len(a); c = int(round(sum(a)))
            if n < k:
                continue
            v.append(1 - comb(n - c, k) / comb(n, k))
        return 100 * np.mean(v) if v else float('nan')
    allw = sum(1 for a in per.values() if sum(a) == 0)
    lens_ok = [r['n_tok'] for r in comp if r['acc'] > 0]
    print(ds, 'n', len(rs), 'complete', len(comp), 'acc_all %.1f' % (100 * np.mean([r['acc'] for r in rs])),
          'acc|complete %.1f' % (100 * acc_c), 'acc|trunc %.1f' % (100 * acc_i), 'rep|trunc %.2f' % rep_i,
          'pass@1/4/16 %.1f/%.1f/%.1f' % (pk(1), pk(4), pk(16)), 'allwrong %d/%d' % (allw, len(per)),
          'len_correct_mean %d' % (np.mean(lens_ok) if lens_ok else 0), flush=True)
print('DONE')
