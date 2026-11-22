R = '/root/autodl-tmp/r1zero'
src = open(R + '/f/resume_f4.sh').read().split('\n')
out = []
for l in src:
    if l.startswith('# 2026-11-20 13:1x') or l.startswith('# 只在最后写指标文件时崩') or l.startswith('# 冒烟的目的'):
        continue
    if l.startswith('rm -rf $R/run/ckpt/f_smoke'):
        continue
    if l.startswith('flog "F resume 4'):
        l = 'flog "F resume 5 (f/resume_f5.sh pid $$): restart after step 5 with balanced rollout (the 16 samples of each prompt spread over the 4 SGLang engines, f/patches/dapo_ray_trainer.interleave.py); stage1 resumes from the latest checkpoint (resume_mode auto)"'
    out.append(l)
    if l.startswith('#!/usr/bin/env bash'):
        out.append('# 2026-11-20 15:25 用户同意：第一阶段第 5 步存盘后重启一次，换上"每道题的 16 条回答拆成每卡 4 条"的生成补丁。')
        out.append('# 由 f/restart_step5.sh 在第 5 步存盘后启动；run_f.sh stage1 按 resume_mode auto 从最新存盘续训（重试次数重新算 3 次），之后评测、和 E 比较，同 resume_f4.sh。')
s = '\n'.join(out)
assert 'F resume 5' in s and 'f_smoke' not in s and 'run_f.sh stage1' in s and 'NCCL_P2P_DISABLE=1' in s
open(R + '/f/resume_f5.sh', 'w').write(s)
print('wrote f/resume_f5.sh,', len(out), 'lines')
