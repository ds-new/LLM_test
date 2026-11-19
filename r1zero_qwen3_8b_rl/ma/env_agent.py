"""环境智能体：检查并补齐 verl 0.6.1 + SGLang 0.5.5 + Megatron-core 0.13.1 + TransformerEngine + mbridge 环境。"""
import os
import json
import shutil
import subprocess
import time

from .common import ENV, LOGS, cleanup_gpu, run

NAME = "env"
CHECK = r'''
import json, importlib
out = {}
def ver(m):
    try:
        mod = importlib.import_module(m); return getattr(mod, "__version__", "ok")
    except Exception as e:
        return "ERR: " + repr(e)[:300]
for m in ["torch", "numpy", "pyarrow", "pandas", "datasets", "transformers", "ray", "verl", "sglang",
          "megatron.core", "transformer_engine", "transformer_engine.pytorch", "mbridge", "math_verify", "flash_attn"]:
    out[m] = ver(m)
try:
    import torch; out["cuda"] = torch.cuda.is_available(); out["gpus"] = torch.cuda.device_count()
except Exception as e:
    out["cuda"] = repr(e)
try:
    from mbridge import AutoBridge; out["mbridge_autobridge"] = "ok"
except Exception as e:
    out["mbridge_autobridge"] = "ERR: " + repr(e)[:300]
try:
    import sglang.srt.entrypoints.engine; out["sglang_engine"] = "ok"
except Exception as e:
    out["sglang_engine"] = "ERR: " + repr(e)[:300]
try:  # 只 import 包名不够：site-packages 里可能只有空的 flash_attn 目录（命名空间包），要能拿到真正的函数
    from flash_attn import flash_attn_varlen_func  # noqa: F401
    import importlib.metadata as _md; out["flash_attn_real"] = _md.version("flash-attn")
except Exception as e:
    out["flash_attn_real"] = "ERR: " + repr(e)[:200]
try:
    from verl.workers.megatron_workers import ActorRolloutRefWorker; out["verl_megatron_worker"] = "ok"
except Exception as e:
    out["verl_megatron_worker"] = "ERR: " + repr(e)[:300]
print("ENVCHECK " + json.dumps(out))
'''


class EnvAgent:
    def __init__(self, board):
        self.board = board

    def check(self):
        p = subprocess.run(["python", "-c", CHECK], capture_output=True, text=True, env=ENV, timeout=900)
        line = [l for l in p.stdout.splitlines() if l.startswith("ENVCHECK ")]
        return json.loads(line[-1][9:]) if line else {"error": (p.stdout + p.stderr)[-2000:]}

    def pip(self, spec, tag, timeout=3600, extra_env=None):
        env = dict(ENV)
        env.update(extra_env or {})
        code, t = run(f"pip install {spec}", LOGS / f"pip_{tag}.log", timeout=timeout, env=env)
        self.board.post(NAME, "pip", {"spec": spec, "code": code, "tail": t[-400:] if code else ""})
        return code == 0

    def ensure(self):
        c = self.check()
        self.board.post(NAME, "check", c)
        bad = lambda k: str(c.get(k, "ERR")).startswith("ERR")  # noqa: E731
        # numpy 被 verl/sglang 降到 1.x 后，新版 pyarrow/pandas 无法导入：换成兼容 numpy 1.x 的版本
        if bad("pyarrow") or bad("pandas") or bad("datasets"):
            self.pip("'pyarrow>=15,<20' 'pandas>=2.1,<2.3' 'datasets>=3,<4' -c /root/autodl-tmp/r1zero/constraints.txt", "pyarrow")
        if bad("math_verify"):
            # 必须用 antlr4 4.9.3：verl 依赖的 omegaconf/hydra 用 4.9.3 生成的解析器，4.13 会让 import verl 报 ATN 错误
            self.pip("'math-verify[antlr4_9_3]' -c /root/autodl-tmp/r1zero/constraints.txt", "mathverify")
        if bad("transformer_engine") or bad("transformer_engine.pytorch"):
            cudnn = subprocess.run(["python", "-c", "import nvidia.cudnn,os;print(os.path.dirname(nvidia.cudnn.__file__))"],
                                   capture_output=True, text=True, env=ENV).stdout.strip()
            te_env = {"NVTE_FRAMEWORK": "pytorch", "MAX_JOBS": "64", "NVTE_BUILD_THREADS_PER_JOB": "2",
                      "CUDA_HOME": "/usr/local/cuda", "CUDNN_PATH": cudnn, "CUDNN_HOME": cudnn,
                      "CPATH": f"{cudnn}/include:" + ENV.get("CPATH", "")}
            self.pip("pybind11 ninja -c /root/autodl-tmp/r1zero/constraints.txt", "te_deps")
            ok = False
            for v in ("2.6.0.post1", "2.5.0", "2.8.0"):
                if self.pip(f"-v --no-build-isolation 'transformer_engine[pytorch]=={v}' -c /root/autodl-tmp/r1zero/constraints.txt",
                            f"te_{v}", timeout=7200, extra_env=te_env):
                    ok = True
                    break
            if not ok:
                return False, "TransformerEngine 安装失败（见 logs/pip_te_*.log）"
        c = self.check()
        # 兜底：任何一次安装把 numpy 升到 2.x 或把 antlr4 换成 4.13，都会让 verl 无法导入（verl 要求 numpy<2）
        if (str(c.get("numpy", "")).startswith("2") or bad("verl") or bad("verl_megatron_worker")
                or bad("math_verify")):
            self.pip("'numpy==1.26.4' 'antlr4-python3-runtime==4.9.3' 'math-verify[antlr4_9_3]' 'pyarrow>=15,<20' "
                     "'pandas>=2.1,<2.3' 'datasets>=3,<4' -c /root/autodl-tmp/r1zero/constraints.txt", "pin_fix")
            c = self.check()
        # Megatron 的打包序列（THD）注意力在 sm80 上要靠 flash-attn（TE 2.6 支持 2.1.1~2.8.1）
        if bad("flash_attn_real"):
            self.try_flash_attn()
            c = self.check()
        if str(c.get("flash_attn_real", "")) > "2.8.1":  # 装了更新的版本时才放宽 TE 的上限
            self.patch_te_flash_max()
        if bad("mbridge_autobridge") or str(c.get("mbridge_autobridge", "")).startswith("ERR"):
            self.pip("'mbridge==0.13.1' -c /root/autodl-tmp/r1zero/constraints.txt", "mbridge")
            c = self.check()
        self.board.post(NAME, "check_after", c)
        must = ["torch", "numpy", "pyarrow", "pandas", "verl", "sglang", "megatron.core", "transformer_engine.pytorch",
                "mbridge_autobridge", "sglang_engine", "verl_megatron_worker", "flash_attn_real"]
        missing = [k for k in must if str(c.get(k, "ERR")).startswith("ERR")]
        if c.get("cuda") is not True or c.get("gpus", 0) < 4:
            missing.append(f"cuda={c.get('cuda')} gpus={c.get('gpus')}")
        self.board.set(env=c)
        return (not missing), ("缺少/无法导入：" + ", ".join(missing)) if missing else "ok"

    def patch_te_flash_max(self):
        """TE 2.6 的 flash-attn 版本上限 2.8.1 → 2.8.3（补丁版本，接口相同）；原文件备份为 .orig。"""
        code = r'''
import os, shutil, transformer_engine.pytorch.attention.dot_product_attention.utils as u
p = u.__file__
s = open(p).read()
if 'max_version = PkgVersion("2.8.1")' in s:
    if not os.path.exists(p + ".orig"):
        shutil.copy2(p, p + ".orig")
    open(p, "w").write(s.replace('max_version = PkgVersion("2.8.1")', 'max_version = PkgVersion("2.8.3")  # r1zero 补丁'))
    print("patched")
else:
    print("skip")
'''
        r = subprocess.run(["python", "-c", code], capture_output=True, text=True, env=ENV, timeout=300)
        self.board.post(NAME, "patch_te_flash_max", {"out": (r.stdout + r.stderr)[-300:]})

    def try_flash_attn(self):
        """从 PyPI 镜像源码编译 flash-attn 2.8.1（只编 sm80，112 核约 15~25 分钟）。
        GitHub 预编译轮子在本机下载极慢；2.8.1 也正好是 TE 2.6 支持的上限。"""
        env = {"FLASH_ATTENTION_FORCE_BUILD": "TRUE", "FLASH_ATTN_CUDA_ARCHS": "80", "MAX_JOBS": "64",
               "NVCC_THREADS": "2", "CUDA_HOME": "/usr/local/cuda"}
        return self.pip("-v --no-build-isolation --no-deps 'flash-attn==2.8.1'", "flash_attn_build", timeout=7200,
                        extra_env=env)


    # ---------------- 第二轮：续训前预检 ----------------
    DATA_DISK, SYS_DISK = "/root/autodl-tmp", "/root"
    CKPT_GB = 16.5  # Qwen3-8B 的 HF 权重（bf16）约 16 GB

    def disk(self):
        return {"data_free_gb": round(shutil.disk_usage(self.DATA_DISK).free / 2**30, 1),
                "sys_free_gb": round(shutil.disk_usage(self.SYS_DISK).free / 2**30, 1)}

    def gpus_busy(self):
        r = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                           capture_output=True, text=True)
        return [int(x) for x in r.stdout.split()] if r.returncode == 0 else None

    def preflight(self, sys_reclaim_gb=0.0):
        """sys_reclaim_gb：系统盘上旧的最佳 checkpoint 会在复制新最佳前删掉，这部分空间可算作可用。
        续训前检查：依赖可导入、4 卡空闲、磁盘够放「最新 + 正在写的新 checkpoint」（数据盘）和「最佳」（系统盘）。"""
        c = self.check()
        must = ["torch", "verl", "sglang", "megatron.core", "transformer_engine.pytorch", "mbridge_autobridge",
                "sglang_engine", "verl_megatron_worker", "flash_attn_real", "math_verify"]
        missing = [k for k in must if str(c.get(k, "ERR")).startswith("ERR")]
        if c.get("cuda") is not True or c.get("gpus", 0) < 4:
            missing.append(f"cuda={c.get('cuda')} gpus={c.get('gpus')}")
        if missing:
            ok, _ = self.ensure()  # 第一轮的自动修复逻辑
            if not ok:
                return False, "环境缺少/无法导入：" + ", ".join(missing)
        used = self.gpus_busy()
        if used is None or max(used) > 2000:
            cleanup_gpu()
            time.sleep(10)
            used = self.gpus_busy()
            if used is None or max(used) > 2000:
                return False, f"GPU 被占用：{used} MiB"
        # Ray 没在跑时清掉旧的会话日志（在系统盘上，长时间训练会越积越多）
        if os.environ.get("R1Z_CLEAN_RAY") == "1":  # 2026-11-17：默认不再自动删（用户没选清理系统盘临时文件）
            subprocess.run("pgrep -f 'ray::' >/dev/null || rm -rf /tmp/ray/session_*", shell=True)
        d = self.disk()
        self.board.post(NAME, "preflight", {**d, "gpu_mem_mib": used})
        if d["data_free_gb"] < self.CKPT_GB + 1:
            return False, f"数据盘剩余 {d['data_free_gb']} GB，不够写下一个 checkpoint（需要约 {self.CKPT_GB + 1} GB）"
        if d["sys_free_gb"] + sys_reclaim_gb < self.CKPT_GB + 3:
            return False, f"系统盘剩余 {d['sys_free_gb']} GB，放不下最佳 checkpoint"
        return True, "ok"
