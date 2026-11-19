import sys
root = sys.argv[1]
t = open(root + "/ma/trainer_agent.py").read()
a = 'steps=steps or 150, val_before=True, test_freq=25, save_freq=25, exp="r1zero_qwen3_8b_dapo")'
b = 'steps=steps or 150, val_before=True, test_freq=25, save_freq=10, exp="r1zero_qwen3_8b_dapo")'
if a in t: t = t.replace(a, b)
a2 = '        f"+{R}.engine_kwargs.sglang.disable_custom_all_reduce=True",\n'
b2 = a2 + '        # 两个推理引擎生成长度不均时，先完成的一组会在集合通信上等另一组；默认 10 分钟超时会把训练杀掉（第 20 步实测）\n        "actor_rollout_ref.nccl_timeout=3600",\n'
if "nccl_timeout" not in t:
    assert a2 in t; t = t.replace(a2, b2)
open(root + "/ma/trainer_agent.py", "w").write(t)
l = open(root + "/ma/loop.py").read()
a3 = '''                p = plan("formal", level, FORMAL_STEPS)
                ok, err = trainer.validate(p)'''
b3 = '''                p = plan("formal", level, FORMAL_STEPS)
                if board.get("baseline_done"):  # 基线验证已有结果：续训时不再重复训练前验证（省约 25 分钟）
                    p["val_before"] = False
                ok, err = trainer.validate(p)'''
if "baseline_done" not in l:
    assert a3 in l; l = l.replace(a3, b3)
a4 = '''                r = trainer.train(p, hang_min=45, wall_hours=left)
'''
b4 = '''                r = trainer.train(p, hang_min=45, wall_hours=left)
                if r.get("last_step", 0) >= 1:
                    board.set(baseline_done=True)
'''
if "board.set(baseline_done=True)" not in l:
    assert a4 in l; l = l.replace(a4, b4)
l = l.replace('if resumed < 1 and r["status"]', 'if resumed < 3 and r["status"]')
l = l.replace('从最近检查点续训一次"})', '从最近检查点续训（第 {resumed}/3 次）"})')
open(root + "/ma/loop.py", "w").write(l)
print("patched")
