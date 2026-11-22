"""F 每步的奖励、熵、梯度相关指标（训练批，动态采样后）。"""
import json
R = '/root/autodl-tmp/r1zero/run/'
rows = [json.loads(x) for x in open(R + 'metrics_f_stage1_1120_1606.jsonl')] + [json.loads(x) for x in open(R + 'metrics_f_stage1_1120_2113.jsonl')]
ks = sorted(k for k in rows[-1]['data'] if any(s in k for s in ('acc', 'score', 'reward', 'entropy', 'pg_clip', 'grad_norm', 'advantages')) and not k.startswith('val'))
print(ks)
for o in rows:
    d = o['data']
    print(o['step'], ' '.join('%.3f' % d[k] if isinstance(d.get(k), (int, float)) else str(d.get(k)) for k in ks))
