import shutil
R = '/root/autodl-tmp/r1zero'
P = R + '/f/make_command_f.py'
shutil.copy(P, '/tmp/make_command_f.py.bak1120e')
s = open(P).read()
assert 'F_GEN_BATCH' not in s
anchor = "\nfor k, v in OV.items():\n"
assert s.count(anchor) == 1
add = ("if os.environ.get('F_GEN_BATCH'):  # 11/20 17:2x：第 1 步 48 题只有 29 组可用、要再生成一整批；从第 3 步起生成批 64（训练批仍 32 题 x 16 条）\n"
       "    OV['data.gen_batch_size'] = os.environ['F_GEN_BATCH']\n")
s = s.replace(anchor, "\n" + add + anchor.lstrip("\n") if False else anchor.replace("\nfor", "\n" + add + "for"))
open(P, 'w').write(s)
print('make_command_f.py: F_GEN_BATCH added')
src = open(R + '/f/resume_f6.sh').read().split('\n')
out = []
for l in src:
    if l.startswith('export F_GMU=0.60'):
        continue
    if l.startswith('# 2026-11-20 17:14') or l.startswith('# 由 f/restart_after_step.sh 启动'):
        continue
    if l.startswith('flog "F resume 6'):
        l = 'flog "F resume 6 (f/resume_f6.sh pid $$): after the step-2 checkpoint; gen_batch_size $F_GEN_BATCH, SGLang gpu_memory_utilization override [$F_GMU] (empty = tier default 0.55; back to 0.55 on OOM); balanced rollout, min memory tier 1, save every 2 steps, chunked_prefill_size 2048; stage1 resumes from the latest checkpoint"'
    out.append(l)
    if l.startswith('#!/usr/bin/env bash'):
        out.append('# 2026-11-20 17:14 用户同意第 2 步实测显存后把 SGLang 显存比例提到 0.60；17:2x 第 1 步 48 题只有 29 组可用、要再生成一整批，生成批一并改成 64。')
        out.append('# 由 f/restart_after_step.sh 在第 2 步存盘后启动：显存检查通过时它在环境里设 F_GMU=0.60，否则不设（0.55）；run_f.sh 从最新存盘续训，再 OOM 回到 0.55。')
    if l.startswith('export F_MIN_TIER=1'):
        out.append('export F_GEN_BATCH=64')
s6 = '\n'.join(out)
assert 'export F_GEN_BATCH=64' in s6 and 'export F_GMU' not in s6 and 'F resume 6' in s6 and 'run_f.sh stage1' in s6
open(R + '/f/resume_f6.sh', 'w').write(s6)
W = R + '/f/restart_after_step.sh'
w = open(W).read()
reps = [
    ('  flog "memory check passed: restart with SGLang gpu_memory_utilization 0.60"\nelse\n  flog "memory check failed: keep 0.55, no restart"\n  [ "$DRY" = 1 ] || exit 2\nfi\n',
     '  GMU_ENV="F_GMU=0.60"; flog "memory check passed: restart with gen_batch_size 64 and SGLang gpu_memory_utilization 0.60"\nelse\n  GMU_ENV=""; flog "memory check failed: restart with gen_batch_size 64 only, SGLang stays at 0.55"\nfi\n'),
    ('setsid nohup bash $R/f/resume_f6.sh', 'env $GMU_ENV setsid nohup bash $R/f/resume_f6.sh'),
    ('stage1 resumes from step $STEP with gpu_memory_utilization 0.60"', 'stage1 resumes from step $STEP with gen_batch_size 64, [$GMU_ENV]"'),
    ('# 2026-11-20 17:14 用户同意（"按照建议进行"）：第 2 步生成时实测显存，余量够就在第 2 步存盘后重启，把 SGLang 显存比例从 0.55 提到 0.60。',
     '# 2026-11-20 17:14 用户同意（"按照建议进行"）：第 2 步生成时实测显存，余量够就在第 2 步存盘后重启，把 SGLang 显存比例从 0.55 提到 0.60。\n# 17:2x 加：第 1 步 48 题只有 29 组可用要再生成一整批，重启时生成批一并改成 64；显存检查不通过也照样重启（只改生成批，保持 0.55）。'),
]
for a, b in reps:
    assert w.count(a) == 1, a[:60]
    w = w.replace(a, b)
open(W, 'w').write(w)
print('restart_after_step.sh updated')
