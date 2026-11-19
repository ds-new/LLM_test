import json, glob, sys, collections
sys.path.insert(0, 'ma')
import reward_v2 as R2, reward as R1
import pandas as pd
v = pd.read_parquet('run/data/val_select.parquet')
key2src = {r['prompt'][0]['content'][-200:]: r['data_source'] for _, r in v.iterrows()}
W = {'math500': 200, 'amc23': 80, 'aime24': 60, 'aime25': 60}
flips = []; per = collections.defaultdict(lambda: collections.Counter())
for f in sorted(glob.glob('/root/r1zero_valgen/*.jsonl'), key=lambda p: int(p.split('/')[-1][:-6])):
    st = int(f.split('/')[-1][:-6])
    for l in open(f):
        d = json.loads(l)
        src = next((s for k, s in key2src.items() if k in d['input']), '?')
        a1 = float(d['acc']) > .5
        r2 = R2.compute_score(src, d['output'], d['gts'])
        a2 = r2['acc'] > .5
        per[st][src + '_n'] += 1; per[st][src + '_v1'] += a1; per[st][src + '_v2'] += a2
        if a1 != a2:
            flips.append(dict(step=st, src=src, dir='wrong->right' if a2 else 'right->wrong', gt=d['gts'], v1=d['pred'], v2=r2['pred'], how=r2['src'], tail=d['output'][-160:]))
rows = []
for st, c in sorted(per.items()):
    row = {'step': st}
    for k in ('v1', 'v2'):
        row['score_' + k] = round(sum(c[s + '_' + k] for s in W) / sum(c[s + '_n'] for s in W), 4)
        for s in W: row[s + '_' + k] = round(c[s + '_' + k] / max(1, c[s + '_n']), 3)
    rows.append(row)
json.dump(dict(rows=rows, flips=flips), open('run/audit/rescore_v2.json', 'w'), ensure_ascii=False, indent=1)
for r in rows: print(r['step'], 'v1', r['score_v1'], 'v2', r['score_v2'], {s: (r[s + '_v1'], r[s + '_v2']) for s in W})
print('flips', collections.Counter((x['dir'], x['src']) for x in flips))
R1._kill_pool()
