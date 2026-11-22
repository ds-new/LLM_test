import json
import sys
import time

sys.path.insert(0, '/root/autodl-tmp/r1zero/f')
import eval_pp_F as P  # noqa: E402
import eval_e  # noqa: E402

s = json.load(open(P.BASE + '/summary.json'))['summary']
print('summary.json acc:', {k: round(100 * v['acc'], 1) for k, v in s.items()})
t0 = time.time()
par = P.load(P.BASE, 24)
t1 = time.time()
seq = eval_e.load_e(P.BASE)
t2 = time.time()
print('par %.0fs seq %.0fs' % (t1 - t0, t2 - t1))
from reward_v2 import compute_score  # noqa: E402
mm = 0
for a, b in zip(par, seq):
    if a['acc'] != b['acc']:
        mm += 1
        if mm <= 6:
            o = a['output'].replace('<|im_end|>', '').replace('<|endoftext|>', '')
            tail = o.split('</think>')[-1] if '</think>' in o else o[-3000:]
            print('MISMATCH', a['ds'], 'gt=', a['gt'], 'par', a['acc'], 'seq', b['acc'], 'now', compute_score(a['ds'], tail, a['gt']))
print('mismatches', mm, 'of', len(par))
for ds in P.SETS:
    print(ds, 'par %.1f' % (100 * sum(r['acc'] for r in par if r['ds'] == ds) / max(1, sum(1 for r in par if r['ds'] == ds))),
          'seq %.1f' % (100 * sum(r['acc'] for r in seq if r['ds'] == ds) / max(1, sum(1 for r in seq if r['ds'] == ds))))
