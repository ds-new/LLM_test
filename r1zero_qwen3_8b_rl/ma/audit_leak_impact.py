import json, sys, collections
sys.path.insert(0, 'ma')
import reward_v2 as R2, reward as R1
import pandas as pd
LEAK = ['8^5 \\cdot 5^{10} \\cdot 15^5', '\\log_{7x}2023', 'u \\otimes v = ac + bdi', 'y=ax+c$, $y=ax+d$', '\\text{mod}_5 (n)', 'F(z)=\\frac{z+i}{z-i}']
v = pd.read_parquet('run/data/val_select.parquet')
k2s = {r['prompt'][0]['content'][-200:]: r['data_source'] for _, r in v.iterrows()}
W = ['math500', 'amc23', 'aime24', 'aime25']
for st in [46, 48]:
    c = collections.Counter()
    for l in open('/root/r1zero_valgen/%d.jsonl' % st):
        d = json.loads(l)
        src = next((s for k, s in k2s.items() if k in d['input']), '?')
        a = R2.compute_score(src, d['output'], d['gts'])['acc']
        lk = any(x in d['input'] for x in LEAK)
        c[(src, lk, 'n')] += 1; c[(src, lk, 'a')] += a
    tot_n = sum(c[(s, l, 'n')] for s in W for l in (0, 1)); tot_a = sum(c[(s, l, 'a')] for s in W for l in (0, 1))
    cl_n = sum(c[(s, False, 'n')] for s in W); cl_a = sum(c[(s, False, 'a')] for s in W)
    print(st, 'amc23 leaked acc %.3f (n=%d)  clean acc %.3f (n=%d)' % (c[('amc23', True, 'a')] / max(1, c[('amc23', True, 'n')]), c[('amc23', True, 'n')], c[('amc23', False, 'a')] / c[('amc23', False, 'n')], c[('amc23', False, 'n')]),
          '| select v2 all %.4f  clean %.4f' % (tot_a / tot_n, cl_a / cl_n))
R1._kill_pool()
