import json, glob, os, time
R = '/root/autodl-tmp/r1zero'
rows = {}
for f in sorted(glob.glob(R + '/run/metrics_f_stage1_*.jsonl')):
    for l in open(f):
        d = json.loads(l)
        rows[d['step']] = (d['data'], os.path.basename(f))
keys = ['actor/lr', 'train/num_gen_batches', 'response_length/clip_ratio', 'response_length/mean', 'actor/entropy', 'critic/score/mean', 'actor/pg_clipfrac', 'actor/ppo_kl', 'actor/grad_norm', 'timing_s/gen', 'timing_s/step']
print('step ' + ' '.join(k.split('/')[-1] for k in keys))
for s in sorted(rows):
    d = rows[s][0]
    print(s, ' '.join(('%.4g' % d[k]) if k in d else 'NA' for k in keys))
d = rows[max(rows)][0]
print([k for k in d if 'acc' in k or 'reward' in k or 'score' in k or 'val' in k][:40])
