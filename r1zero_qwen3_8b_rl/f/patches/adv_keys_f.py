import json
R = '/root/autodl-tmp/r1zero/run/'
rows = [json.loads(x) for x in open(R + 'metrics_f_stage1_1120_2113.jsonl')]
ks = sorted(k for k in rows[-1]['data'] if k.startswith('rollout_corr') or k.startswith('critic/') or k.startswith('response_length') or 'mask' in k or 'valid' in k)
for k in ks:
    print('%-55s' % k, ' '.join('%.4g' % o['data'][k] if isinstance(o['data'].get(k), (int, float)) else '-' for o in rows))
