import shutil, re, os
R = '/root/autodl-tmp/r1zero'
p = R + '/f/make_command_f.py'
src = open(p).read()
if 'F_RESUME_LR' in src:
    print('already patched')
else:
    shutil.copy(p, R + '/f/patches/make_command_f.py.bak1121b')
    anchor = "for k, v in OV.items():\n    pat = re.compile("
    assert src.count(anchor) == 1, 'anchor'
    add = '''# 11/21 16:4x（用户同意）：存盘里没有学习率调度器状态，续训时 Megatron 调度器从 num_steps=0 重新预热，lr_warmup_init=0 时
# 续训后的第一步学习率是 0（只更新 Adam 动量、不改权重，白跑一步）。已经有存盘（resume_mode=auto 会从它续训）时改成：
# 第一步 lr/3、第二步起满学习率（lr_warmup_steps=1、lr_warmup_init=lr/3）。F_RESUME_LR=0 可以关掉。只影响以后新生成的命令。
if mode == 'stage1' and os.environ.get('F_RESUME_LR', '1') != '0':
    try:
        _it = int(open(OV['trainer.default_local_dir'] + '/latest_checkpointed_iteration.txt').read().strip())
    except Exception:  # noqa: BLE001
        _it = 0
    if _it >= 1:
        _m = re.search(r'actor_rollout_ref\\.actor\\.optim\\.lr=(\\S+)', s)
        _lr = float(_m.group(1)) if _m else 1e-6
        OV['actor_rollout_ref.actor.optim.lr_warmup_steps'] = '1'
        OV['actor_rollout_ref.actor.optim.lr_warmup_init'] = '%.6g' % (_lr / 3)
        print('resume from checkpoint %d: lr warmup 1 step from %.6g to %.6g' % (_it, _lr / 3, _lr))
'''
    src = src.replace(anchor, add + anchor)
    open(p, 'w').write(src)
    print('patched')
t = src.replace("open(R + '/run/command_f_%s.sh' % mode, 'w')", "open('/tmp/command_f_test_%s_' % mode + sys.argv[3] + '.sh', 'w')")
assert t != src
open('/tmp/mcf_test.py', 'w').write(t)
