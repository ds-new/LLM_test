import json
R = '/root/autodl-tmp/r1zero'
rows = {}
for l in open(R + '/run/metrics_r1zero_qwen3_8b_dapo.jsonl'):
    try:
        d = json.loads(l)
    except Exception:
        continue
    s = d.get('step')
    dd = d.get('data', d)
    if s is None:
        s = dd.get('training/global_step')
    rows[s] = dd
ks = ['actor/lr', 'train/num_gen_batches', 'response_length/clip_ratio', 'response_length/mean', 'actor/entropy', 'actor/pg_clipfrac', 'actor/ppo_kl', 'timing_s/step']
print(len(rows), list(rows)[:3], list(rows)[-3:])
for s in sorted(k for k in rows if isinstance(k, (int, float))):
    if s >= 135:
        d = rows[s]
        print(s, ' '.join(('%.3g' % d[k]) if k in d else 'NA' for k in ks))
