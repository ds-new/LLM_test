"""R1-Zero 多智能体 loop 的公共部分：路径、状态、事件总线、子进程执行、关机。"""
import json
import os
import re
import subprocess
import time
from pathlib import Path

ROOT = Path(os.environ.get("R1Z_ROOT", "/root/autodl-tmp/r1zero"))
RUN = ROOT / "run"
CODE = ROOT / "ma"
MODEL = Path(os.environ.get("R1Z_MODEL", "/root/autodl-tmp/models/Qwen3-8B-Base"))
RAW = ROOT / "raw"
DATA = RUN / "data"
LOGS = RUN / "logs"
for d in (RUN, DATA, LOGS):
    d.mkdir(parents=True, exist_ok=True)

ENV = dict(os.environ)
ENV.update({"PYTHONUNBUFFERED": "1",  # Ray 里训练进程的 stdout 是块缓冲，指标行会延迟出现
            "NCCL_P2P_DISABLE": "1",  # 本机 PCIe A800 容器内 GPU 直连(P2P)会让 NCCL 集合通信卡死，实测关闭后正常
            # 容器内核不支持 pidfd_open：关掉 expandable_segments，否则 Megatron→SGLang 的 CUDA IPC 权重同步失败
            "VERL_DISABLE_EXPANDABLE_SEGMENTS": "1", "MEGATRON_CI_DISABLE_EXPANDABLE_SEGMENTS": "1",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:False",
            "OMP_NUM_THREADS": "8", "CUDA_DEVICE_MAX_CONNECTIONS": "1", "HYDRA_FULL_ERROR": "1",
            "RAY_DEDUP_LOGS": "0", "HF_ENDPOINT": "https://hf-mirror.com", "TOKENIZERS_PARALLELISM": "false",
            "PYTHONPATH": str(ROOT) + os.pathsep + os.environ.get("PYTHONPATH", "")})


def now():
    return time.strftime("%Y-%m-%d %H:%M:%S")


def read_json(p, default=None):
    try:
        return json.loads(Path(p).read_text(encoding="utf-8"))
    except Exception:
        return default


def write_json(p, obj):
    p = Path(p)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(p)


class Board:
    """共享状态（state.json）+ 事件日志（events.jsonl）。各智能体通过它交流。"""
    def __init__(self):
        self.path = RUN / "state.json"
        self.state = read_json(self.path, {}) or {}

    def get(self, k, d=None):
        return self.state.get(k, d)

    def set(self, **kw):
        self.state.update(kw)
        write_json(self.path, self.state)

    def post(self, sender, kind, content):
        rec = {"t": now(), "from": sender, "kind": kind, **content}
        with (RUN / "events.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        with (LOGS / "loop.log").open("a", encoding="utf-8") as f:
            f.write(f"[{rec['t']}] {sender} {kind}: {json.dumps(content, ensure_ascii=False)[:1500]}\n")


def run(cmd, log, timeout=None, env=None, cwd=None):
    """执行命令（shell 字符串或列表），输出写到 log；返回 (exit_code, 末尾 4000 字符)。超时返回 124。"""
    log = Path(log)
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a", encoding="utf-8") as f:
        f.write(f"\n===== {now()} $ {cmd if isinstance(cmd, str) else ' '.join(map(str, cmd))}\n")
        f.flush()
        argv = ["bash", "-c", cmd] if isinstance(cmd, str) else cmd
        p = subprocess.Popen(argv, stdout=f, stderr=subprocess.STDOUT,
                             env=env or ENV, cwd=cwd, start_new_session=True)
        try:
            code = p.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            kill_group(p)
            code = 124
    return code, tail(log)


def kill_group(p):
    import signal
    try:
        os.killpg(os.getpgid(p.pid), signal.SIGTERM)
        time.sleep(10)
        os.killpg(os.getpgid(p.pid), signal.SIGKILL)
    except Exception:
        pass


def tail(p, n=4000):
    try:
        b = Path(p).read_bytes()
        return b[-n:].decode("utf-8", errors="replace")
    except Exception:
        return ""


def cleanup_gpu():
    """结束残留的推理/训练进程（ray、sglang 调度器），避免下一次启动时显存被占。"""
    subprocess.run("ray stop --force >/dev/null 2>&1; pkill -9 -f 'sglang::' ; pkill -9 -f 'verl.trainer.main_ppo'; "
                   "sleep 5", shell=True, env=ENV)


def shutdown(board, reason, delay=60):
    board.post("decision", "shutdown", {"reason": reason, "in_seconds": delay})
    if (RUN / "KEEP_RUNNING").exists():
        board.post("decision", "shutdown_cancelled", {"why": "KEEP_RUNNING 文件存在"})
        return
    subprocess.Popen(["bash", "-c", f"sleep {delay}; /usr/bin/shutdown"], start_new_session=True)


ANSI = re.compile(r"\x1b\[[0-9;]*m")
