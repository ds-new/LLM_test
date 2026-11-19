
import shutil, os
V = "/root/miniconda3/lib/python3.12/site-packages/verl/workers/rollout/sglang_rollout/sglang_rollout.py"
if not os.path.exists(V + ".orig"):
    shutil.copy2(V, V + ".orig")
s = open(V).read()
old = '''                "dist_timeout": 1800,
            }
'''
new = '''                "dist_timeout": 1800,
                # r1zero 补丁：同步 rollout 路径原先不传 engine_kwargs（异步服务路径会传），补上以便关闭 custom all-reduce 等
                **engine_kwargs,
            }
'''
if "r1zero 补丁" not in s:
    assert s.count(old) == 1, s.count(old)
    s = s.replace(old, new); open(V, "w").write(s)
print("patched")
