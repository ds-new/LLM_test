"""模型训练智能体：启动 verl 训练/评估进程、解析日志里的每步指标、判断卡死/崩溃并给出错误类别。

配置由模型框架智能体（framework_agent）负责；这里保留 LEVELS / plan / overrides 的导出，兼容第一轮的 loop.py。
"""
import json
import re
import subprocess
import time

from .common import ANSI, ENV, LOGS, RUN, cleanup_gpu, kill_group, now, tail
from .framework_agent import LEVELS, FrameworkAgent, command, overrides, plan  # noqa: F401

NAME = "trainer"
STEP_RE = re.compile(r"step:(\d+) - (.*)")
KV_RE = re.compile(r"([\w@/\-.]+):([-+0-9.eE]+|nan|inf)")
VAL_RE = re.compile(r"['\"]?(val-core/[^'\":]+/acc/mean@\d+)['\"]?\s*[:=]\s*([0-9.]+)")


def classify(text):
    t = text
    # 容器内核不支持 pidfd_open：CUDA IPC 权重同步失败，与显存无关（不能降档解决）
    if "pidfd_open" in t:
        return "ipc"
    if re.search(r"is not a distributed checkpoint|CheckpointingException", t):
        return "ckpt"
    if re.search(r"No space left on device|Disk quota exceeded", t):
        return "disk"
    # 只认 CUDA 的显存不足；Ray 的通用提示里有 "OOM killer" 字样，不能据此判成显存不足
    if re.search(r"CUDA out of memory|OutOfMemoryError|CUDA error: out of memory|torch\.OutOfMemoryError", t):
        return "oom"
    # SGLang 显存比例设得太高时的报错，按显存不足处理。只认报错原文：参数名 mem_fraction_static 每次启动都会打印，不能用来判断
    if re.search(r"Not enough memory\. Please try to increase --mem-fraction-static|Not enough memory to allocate", t):
        return "oom"
    # checkpoint 损坏或不完整（缺元数据、缺权重分片）
    if re.search(r"is not a distributed checkpoint|CheckpointingException|safetensors_rust\.SafetensorError|No such file or directory: .*global_step_\d+", t):
        return "ckpt"
    if "No module named 'flash_attn'" in t or "flash_attn" in t and "ModuleNotFoundError" in t:
        return "flash_attn"
    if re.search(r"Could not override|Key '.*' is not in struct|ConfigAttributeError|ConfigKeyError", t):
        return "config"
    if re.search(r"NCCL|Watchdog caught collective operation timeout", t):
        return "nccl"
    return "other"


class TrainerAgent:
    def __init__(self, board):
        self.board = board
        self.fw = FrameworkAgent(board)

    def validate(self, p):
        return self.fw.validate(p)

    def train(self, p, hang_min=45, wall_hours=None, on_tick=None):
        """启动并监控一次训练/评估。on_tick() 每 30 秒调用一次，返回非空字符串表示致命异常、立即停止。
        返回 dict(status=done|crash|hang|deadline|fatal, cls, last_step, metrics, log, hours, tail)。"""
        cleanup_gpu()
        log = LOGS / f"train_{p['exp']}_L{p['level']}_{time.strftime('%m%d_%H%M')}.log"
        mfile = RUN / f"metrics_{p['exp']}.jsonl"
        cmd = command(p)
        (RUN / f"command_{p['exp']}.sh").write_text("cd " + str(RUN) + "\n" + " \\\n  ".join(cmd) + "\n", encoding="utf-8")
        self.board.post(NAME, "launch", {"exp": p["exp"], "level": p["level"], "log": str(log),
                                         **{k: p[k] for k in ("tp", "recompute", "max_tok", "gmu", "bsz", "n", "resp", "steps")}})
        f = log.open("w", encoding="utf-8")
        proc = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, env=ENV, cwd=str(RUN), start_new_session=True)
        t0, last_size, last_change, seen_steps, val_seen = time.time(), 0, time.time(), set(), set()
        status, fatal, pos, carry = None, None, 0, ""
        while True:
            time.sleep(30)
            # 只读新增部分（日志会到几百 MB，每 30 秒全量读太慢）
            with log.open("r", encoding="utf-8", errors="replace") as lf:
                lf.seek(pos)
                chunk = lf.read()
                pos = lf.tell()
            if chunk:
                last_change = time.time()
            text = carry + ANSI.sub("", chunk)
            cut = text.rfind("\n")
            text, carry = (text[:cut + 1], text[cut + 1:]) if cut >= 0 else ("", text)
            for m in STEP_RE.finditer(text):
                step = int(m.group(1))
                kv = {k: float(v) for k, v in KV_RE.findall(m.group(2))}
                key = (step, any(k.startswith("val-core/") for k in kv), "critic/score/mean" in kv)
                if key in seen_steps:
                    continue
                seen_steps.add(key)
                kv.update(step=step, t=now(), level=p["level"])
                with mfile.open("a", encoding="utf-8") as mf:
                    mf.write(json.dumps(kv) + "\n")
                brief = {k: round(v, 4) for k, v in kv.items() if k in (
                    "critic/score/mean", "response_length/mean", "response_length/clip_ratio", "actor/entropy",
                    "timing_s/step", "actor/pg_clipfrac", "actor/grad_norm") or k.startswith("val-core/")}
                self.board.post(NAME, "step", {"exp": p["exp"], "step": step, **brief})
            for k, v in VAL_RE.findall(text):
                if (k, v) not in val_seen:
                    val_seen.add((k, v))
                    self.board.post(NAME, "val", {"exp": p["exp"], "metric": k, "value": float(v)})
            if on_tick:
                try:
                    fatal = on_tick()
                except Exception as e:  # noqa: BLE001  回调出错不能拖垮训练监控
                    self.board.post(NAME, "tick_error", {"error": repr(e)[:500]})
                    fatal = None
                if fatal:
                    status = "fatal"
                    kill_group(proc)
                    break
            code = proc.poll()
            if code is not None:
                status = "done" if code == 0 else "crash"
                break
            if time.time() - last_change > hang_min * 60:
                status = "hang"
                kill_group(proc)
                break
            if wall_hours and time.time() - t0 > wall_hours * 3600:
                status = "deadline"
                kill_group(proc)
                break
        f.close()
        cleanup_gpu()
        steps = [s for s, _v, train in seen_steps if train]
        res = {"status": status, "cls": (classify(tail(log, 400000)) if status not in ("done", "fatal") else None),
               "fatal": fatal, "last_step": max(steps) if steps else 0, "log": str(log), "metrics": str(mfile),
               "hours": round((time.time() - t0) / 3600, 2), "tail": tail(log, 20000)[-1500:] if status != "done" else ""}
        self.board.post(NAME, "result", {k: v for k, v in res.items() if k != "tail"} | {"tail": res["tail"][-600:]})
        return res
