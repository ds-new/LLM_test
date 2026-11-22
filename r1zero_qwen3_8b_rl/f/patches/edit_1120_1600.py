import shutil
R = '/root/autodl-tmp/r1zero'
P = R + '/f/make_command_f.py'
shutil.copy(P, '/tmp/make_command_f.py.bak1120c')
s = open(P).read()
reps = [
    ('import re\n', 'import os\nimport re\n'),
    ("tier = int(sys.argv[3]) if len(sys.argv) > 3 else 0\n",
     "tier = int(sys.argv[3]) if len(sys.argv) > 3 else 0\n"
     "tier = max(tier, int(os.environ.get('F_MIN_TIER', '0')))  # 11/20 16:0x：0 档第 2 步会 OOM（训练进程生成时还占 16.5 GiB），resume_f5 设 F_MIN_TIER=1\n"),
    ("    'trainer.save_freq': '5',\n",
     "    'trainer.save_freq': '2',  # 11/20 16:0x：第 2 步 OOM 时还没有存盘，第 1 步整步作废；改成每 2 步存一次\n"),
    ("    'custom_reward_function.path'",
     "    '+actor_rollout_ref.rollout.engine_kwargs.sglang.chunked_prefill_size': '2048',  # 11/20 16:0x：15:39 的 OOM 是一次 4.64 GiB 分配（8192 token × 词表 × fp32 logits），分块预填改小\n"
     "    'custom_reward_function.path'"),
]
for a, b in reps:
    assert s.count(a) == 1, a
    s = s.replace(a, b)
open(P, 'w').write(s)
print('make_command_f.py updated')
src = open(R + '/f/resume_f4.sh').read().split('\n')
out = []
for l in src:
    if l.startswith('# 2026-11-20 13:1x') or l.startswith('# 只在最后写指标文件时崩') or l.startswith('# 冒烟的目的'):
        continue
    if l.startswith('rm -rf $R/run/ckpt/f_smoke'):
        continue
    if l.startswith('flog "F resume 4'):
        l = 'flog "F resume 5 (f/resume_f5.sh pid $$): stage1 from step 1 again with balanced rollout (interleave trainer), min memory tier 1, save every 2 steps, SGLang chunked_prefill_size 2048"'
    out.append(l)
    if l.startswith('#!/usr/bin/env bash'):
        out.append('# 2026-11-20 16:0x：15:39 第 2 步生成时显存不足，run_f.sh 降档从头重跑；Claude 决定现在就重启（用户 15:25 同意换均衡补丁，原定第 5 步后）：')
        out.append('# 均衡补丁从第 1 步起生效，最低显存档 1（F_MIN_TIER），每 2 步存盘，SGLang 分块预填 2048。之后评测、和 E 比较，同 resume_f4.sh。')
    if l.startswith('export PYTHONUNBUFFERED'):
        out.append('export F_MIN_TIER=1')
s5 = '\n'.join(out)
assert 'F resume 5' in s5 and 'f_smoke' not in s5 and 'export F_MIN_TIER=1' in s5 and 'run_f.sh stage1' in s5
open(R + '/f/resume_f5.sh', 'w').write(s5)
print('resume_f5.sh written')
