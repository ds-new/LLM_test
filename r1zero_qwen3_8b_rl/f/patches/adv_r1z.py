import json, collections
R = '/root/autodl-tmp/r1zero/run/'
rows = {}
for l in open(R + 'metrics_r1zero_qwen3_8b_dapo.jsonl'):
    o = json.loads(l); d = o.get('data', o); s = o.get('step', d.get('step'))
    if s is not None and 'critic/advantages/max' in d:
        rows[int(s)] = d
ks = sorted(rows)
print('R1-Zero steps with adv metrics:', len(ks))
c = collections.Counter((round(rows[k]['critic/advantages/max'], 3), round(rows[k]['critic/advantages/min'], 3)) for k in ks)
print('most common (max, min):', c.most_common(8))
for k in ks[-6:]:
    d = rows[k]
    print(k, round(d['critic/advantages/max'], 3), round(d['critic/advantages/min'], 3), round(d['critic/score/max'], 3), round(d['critic/score/min'], 3))
