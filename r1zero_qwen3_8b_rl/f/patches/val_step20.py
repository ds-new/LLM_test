import json
R = '/root/autodl-tmp/r1zero/run/'
rows = [json.loads(x) for x in open(R + 'metrics_f_stage1_1120_2113.jsonl')]
for o in rows:
    d = o['data']
    v = {k: d[k] for k in d if k.startswith('val-core') or k.startswith('val-aux')}
    if v:
        print('step', o['step'])
        for k in sorted(v):
            if 'acc' in k or 'score' in k or 'reward' in k:
                print('  %-45s %.4f' % (k, v[k]))
o = rows[-1]; d = o['data']
print('last step', o['step'], 'step_min %.1f gen_min %.1f nb %s clip %.3f len %.0f ent %.3f score %.3f lr %s' % (d['timing_s/step'] / 60, d['timing_s/gen'] / 60, d.get('train/num_gen_batches'), d['response_length/clip_ratio'], d['response_length/mean'], d['actor/entropy'], d['critic/score/mean'], d.get('actor/lr')))
print('testing time min', round(d.get('timing_s/testing', 0) / 60, 1), 'save', round(d.get('timing_s/save_checkpoint', 0) / 60, 1))
print('adv', round(d['critic/advantages/max'], 3), round(d['critic/advantages/min'], 3))
