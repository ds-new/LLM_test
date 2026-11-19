"""定时检查用：打印最近几步的耗时拆分和关键指标。"""
import json, sys
n = int(sys.argv[1]) if len(sys.argv) > 1 else 6
rows = []
for l in open('run/metrics_r1zero_qwen3_8b_dapo.jsonl'):
    d = json.loads(l)
    s = d.get('step') or d.get('training/global_step')
    if s and 'timing_s/step' in d and d.get('t', '') >= '2026-11-16 21:42':
        rows.append((s, d))
for s, d in rows[-n:]:
    t = {k.split('/')[1]: round(v) for k, v in d.items() if k.startswith('timing_s/') and k.split('/')[1] in ('step', 'gen', 'update_actor', 'old_log_prob', 'testing')}
    print(s, d.get('t', '')[5:16], t, 'len', round(d.get('response_length/mean', 0)), 'clip', round(d.get('response_length/clip_ratio', 0), 3),
          'score', round(d.get('critic/score/mean', 0), 3), 'ent', round(d.get('actor/entropy', 0), 3),
          'gn', round(d.get('actor/grad_norm', 0), 3))
print('steps since resume:', len(rows), 'avg step s:', round(sum(d['timing_s/step'] for _, d in rows) / max(1, len(rows))))
