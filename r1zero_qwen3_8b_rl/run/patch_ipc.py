
import re, shutil, os
V = "/root/miniconda3/lib/python3.12/site-packages/verl/utils/device.py"
if not os.path.exists(V + ".orig"):
    shutil.copy2(V, V + ".orig")
s = open(V).read()
old = '''    if is_cuda_available:
        torch.cuda.memory._set_allocator_settings(f"expandable_segments:{enable}")'''
new = '''    # r1zero 补丁：容器内核不支持 pidfd_open，expandable_segments 下分配的张量无法经 CUDA IPC 同步给 SGLang
    import os as _os
    if _os.environ.get("VERL_DISABLE_EXPANDABLE_SEGMENTS") == "1":
        enable = False
    if is_cuda_available:
        torch.cuda.memory._set_allocator_settings(f"expandable_segments:{enable}")'''
if "VERL_DISABLE_EXPANDABLE_SEGMENTS" not in s:
    assert old in s, "pattern not found"
    s = s.replace(old, new); open(V, "w").write(s)
print("verl patched")
R = "/root/autodl-tmp/r1zero/ma/"
c = open(R + "common.py").read()
if "VERL_DISABLE_EXPANDABLE_SEGMENTS" not in c:
    c = c.replace('''            "OMP_NUM_THREADS": "8",''', '''            # 容器内核不支持 pidfd_open：关掉 expandable_segments，否则 Megatron→SGLang 的 CUDA IPC 权重同步失败
            "VERL_DISABLE_EXPANDABLE_SEGMENTS": "1", "MEGATRON_CI_DISABLE_EXPANDABLE_SEGMENTS": "1",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:False",
            "OMP_NUM_THREADS": "8",''', 1)
    open(R + "common.py", "w").write(c)
t = open(R + "trainer_agent.py").read()
t = t.replace('''    t = text
    if re.search(r"out of memory|OutOfMemoryError|CUDA error: out of memory|\bOOM\b", t):
        return "oom"''', '''    t = text
    # 容器内核不支持 pidfd_open：CUDA IPC 权重同步失败，与显存无关（不能降档解决）
    if "pidfd_open" in t:
        return "ipc"
    # 只认 CUDA 的显存不足；Ray 的通用提示里有 "OOM killer" 字样，不能据此判成显存不足
    if re.search(r"CUDA out of memory|OutOfMemoryError|CUDA error: out of memory|torch\.OutOfMemoryError", t):
        return "oom"''')
t = t.replace('"cls": classify(t) if status', '"cls": classify(tail(log, 400000)) if status')
open(R + "trainer_agent.py", "w").write(t)
print("ma patched")
