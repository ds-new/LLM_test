
R = "/root/autodl-tmp/r1zero/ma/trainer_agent.py"
lines = open(R).read().split("\n")
out = []
i = 0
while i < len(lines):
    l = lines[i]
    if "OutOfMemoryError" in l and "OOM" in l and l.strip().startswith("if re.search"):
        out += ['    # 容器内核不支持 pidfd_open：CUDA IPC 权重同步失败，与显存无关（不能降档解决）',
                '    if "pidfd_open" in t:',
                '        return "ipc"',
                '    # 只认 CUDA 的显存不足；Ray 的通用提示里有 "OOM killer" 字样，不能据此判成显存不足',
                '    if re.search(r"CUDA out of memory|OutOfMemoryError|CUDA error: out of memory|torch\\.OutOfMemoryError", t):']
        i += 1
        continue
    out.append(l); i += 1
open(R, "w").write("\n".join(out))
print("done")
