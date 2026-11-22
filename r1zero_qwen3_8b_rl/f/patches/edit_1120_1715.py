import shutil
R = '/root/autodl-tmp/r1zero'
P = R + '/f/make_command_f.py'
shutil.copy(P, '/tmp/make_command_f.py.bak1120d')
s = open(P).read()
assert 'F_GMU' not in s
anchor = "\nfor k, v in OV.items():\n"
assert s.count(anchor) == 1
add = ("\nraw_tier = int(sys.argv[3]) if len(sys.argv) > 3 else 0\n"
       "if raw_tier == 0 and os.environ.get('F_GMU'):  # 11/20 17:14 用户同意：第 2 步实测余量够就把 SGLang 显存比例提到 0.60；OOM 升档（raw tier >= 1）后不再用，回到档位默认值\n"
       "    OV['actor_rollout_ref.rollout.gpu_memory_utilization'] = os.environ['F_GMU']\n")
s = s.replace(anchor, add + anchor)
open(P, 'w').write(s)
print('make_command_f.py: F_GMU added')
src = open(R + '/f/resume_f5.sh').read().split('\n')
out = []
for l in src:
    if l.startswith('# 2026-11-20 16:0x') or l.startswith('# 均衡补丁从第 1 步起生效'):
        continue
    if l.startswith('flog "F resume 5'):
        l = 'flog "F resume 6 (f/resume_f6.sh pid $$): after the step-2 checkpoint, SGLang gpu_memory_utilization 0.55 -> 0.60 (F_GMU, back to 0.55 on OOM); balanced rollout, min memory tier 1, save every 2 steps, chunked_prefill_size 2048; stage1 resumes from the latest checkpoint"'
    out.append(l)
    if l.startswith('#!/usr/bin/env bash'):
        out.append('# 2026-11-20 17:14 用户同意：第 2 步生成时实测显存余量够，就在第 2 步存盘后重启，把 SGLang 显存比例从 0.55 提到 0.60（F_GMU）。')
        out.append('# 由 f/restart_after_step.sh 启动；run_f.sh stage1 按 resume_mode auto 从最新存盘续训（重试次数重新算），再 OOM 就回到 0.55；之后评测、和 E 比较，同 resume_f5.sh。')
    if l.startswith('export F_MIN_TIER=1'):
        out.append('export F_GMU=0.60')
s6 = '\n'.join(out)
assert 'F resume 6' in s6 and 'export F_GMU=0.60' in s6 and 'export F_MIN_TIER=1' in s6 and 'run_f.sh stage1' in s6 and '16:0x' not in s6
open(R + '/f/resume_f6.sh', 'w').write(s6)
print('resume_f6.sh written')
