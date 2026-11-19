
R = "/root/autodl-tmp/r1zero/ma/trainer_agent.py"
s = open(R).read()
anchor = '''        f"{R}.name=sglang", f"{R}.tensor_model_parallel_size=2", f"{R}.gpu_memory_utilization={p['gmu']}",\n'''
add = '''        # 本机 GPU 直连(P2P)不可用：SGLang 自带的 custom all-reduce 走 P2P 会在解码时卡死，改用 NCCL\n        f"+{R}.engine_kwargs.sglang.disable_custom_all_reduce=True",\n'''
if "disable_custom_all_reduce" not in s:
    assert anchor in s
    s = s.replace(anchor, anchor + add)
    open(R, "w").write(s)
print("ok")
