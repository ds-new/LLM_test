R = '/root/autodl-tmp/r1zero'
s = open(R + '/f/resume_f6.sh').read()
lines = s.split('\n')
assert lines[0].startswith('#!')
hdr = ['#!/usr/bin/env bash',
 '# 2026-11-21 Claude: started only by f/restart_if_frag.sh, when the training process (WorkerDict) keeps so much fragmented GPU memory after a step',
 '# that the next generation would likely OOM at SGLang 0.55. Same as f/resume_f6.sh, but F_MIN_TIER=2 (SGLang gpu_memory_utilization 0.50,',
 '# ppo/log_prob max token len 17408); run_f.sh resumes from the latest checkpoint. Note: the LR warmup restarts on resume (first step lr 0).']
body = [l for l in lines[1:] if not (l.startswith('#') and lines.index(l) < 4)]
t = '\n'.join(hdr + body)
assert t.count('export F_MIN_TIER=1') == 1
t = t.replace('export F_MIN_TIER=1', 'export F_MIN_TIER=2')
assert t.count('F resume 6 (f/resume_f6.sh pid') == 1
t = t.replace('F resume 6 (f/resume_f6.sh pid', 'F resume 7 (memory tier 2; f/resume_f7.sh pid')
open(R + '/f/resume_f7.sh', 'w').write(t)
print('ok')
