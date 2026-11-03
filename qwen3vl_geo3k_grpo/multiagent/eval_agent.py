"""模型测试评估智能体。

  * 基线：原始模型在 val / test 上的贪心准确率；在训练集上每题采样 k 次得到通过率（交给数据智能体做难度过滤）；
  * 导出：把 FSDP 检查点合并为 HuggingFace 格式（verl.model_merger，旧版回退 scripts/model_merger.py），
    已含 safetensors 的 actor/huggingface 目录直接使用；
  * 评估：调用 evaluate_geo3k.py 子进程（sglang → vllm → hf；独占 GPU，结束即释放），读取 JSON 结果并做失败类型分析；
  * 纪律：test 只评两次——基线模型和最终选出的最佳模型；所有选择都只看 val。
"""

import os
import shutil
import signal
import subprocess
import time
from pathlib import Path

from .common import (DRY_RUN, PROJECT_DIR, RUN_DIR, RestartRequested, StopRequested, control_signal, project_env,
                     python_executable, read_json, stamp, write_json)

NAME = "evaluator"


class EvalAgent:
    def __init__(self, config, bus):
        self.config = config
        self.ecfg = config["eval"]
        self.bus = bus
        self.dir = RUN_DIR / "eval"

    def report(self, action, content):
        self.bus.post(NAME, "decision", "report", {"action": action, **content})

    # ------------------------------------------------------------------
    def evaluate(self, model_path, data_file, tag, n_samples=1, temperature=0.0, tp=1, timeout_h=4.0,
                 max_new_tokens=None):
        max_new_tokens = int(max_new_tokens or self.ecfg["max_new_tokens"])
        out = self.dir / f"{tag}.json"
        cached = read_json(out)
        if cached and cached["summary"].get("model_path") == str(model_path) and \
                cached["summary"].get("n_samples") == n_samples and \
                cached["summary"].get("max_new_tokens") == max_new_tokens:
            self.report("evaluate", {"tag": tag, "cached": True, **self._brief(cached["summary"])})
            return cached
        backend = "fake" if DRY_RUN else self.ecfg["backend"]
        cmd = [python_executable(), str(PROJECT_DIR / "evaluate_geo3k.py"), "--model_path", str(model_path),
               "--input_file", str(data_file), "--backend", backend, "--n_samples", str(n_samples),
               "--temperature", str(temperature), "--max_new_tokens", str(max_new_tokens),
               "--tp", str(tp), "--output_json", str(out)]
        if temperature > 0:
            cmd += ["--top_p", "1.0"]
        log = self.dir / "logs" / f"{tag}.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        self.bus.post(NAME, "decision", "status", {"action": "evaluate_start", "tag": tag, "model": str(model_path),
                                                   "data": str(data_file), "n_samples": n_samples})
        code = self._run(cmd, log, timeout_h, "w")
        if code != 0 or not out.exists():
            tail = log.read_text(encoding="utf-8", errors="replace")[-2000:]
            # 推理引擎出错时，只有单次采样（val/test 贪心评估）才退回 HF；多次采样（难度过滤，上万次生成）
            # 用 HF 要跑数小时，直接报错让人修复引擎，而不是悄悄烧掉计费时间
            if self.ecfg["backend"] == "auto" and not DRY_RUN and n_samples == 1:
                self.bus.post(NAME, "decision", "status", {"action": "fallback_hf", "tag": tag, "engine_error": tail[-600:]})
                cmd[cmd.index("--backend") + 1] = "hf"
                code = self._run(cmd, log, timeout_h * 3, "a")
            if code != 0 or not out.exists():
                raise RuntimeError(f"评估失败 {tag}: {tail}")
        res = read_json(out)
        self.report("evaluate", {"tag": tag, **self._brief(res["summary"])})
        return res

    def _run(self, cmd, log, timeout_h, mode):
        """子进程运行评估；每 10 秒检查一次 STOP/RESTART，收到即结束整个进程组（含 vLLM 引擎子进程）。"""
        with log.open(mode, encoding="utf-8") as f:
            proc = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=str(PROJECT_DIR), env=project_env(),
                                    start_new_session=True)
            deadline = time.time() + timeout_h * 3600
            while True:
                try:
                    return proc.wait(timeout=10 if not DRY_RUN else 0.05)
                except subprocess.TimeoutExpired:
                    pass
                sig = control_signal()
                if sig or time.time() > deadline:
                    for s_, w in ((signal.SIGTERM, 20), (signal.SIGKILL, 5)):
                        try:
                            os.killpg(proc.pid, s_)
                            proc.wait(timeout=w)
                            break
                        except (ProcessLookupError, subprocess.TimeoutExpired):
                            continue
                    if sig == "stop":
                        raise StopRequested("STOP file present (during evaluation)")
                    if sig == "restart":
                        raise RestartRequested("RESTART file present (during evaluation)")
                    return -9

    @staticmethod
    def _brief(s):
        keys = ["accuracy", "n_items", "n_samples", "format_rate", "boxed_rate", "truncated_rate", "failure_types",
                "pass_rate_hist", "backend", "seconds"]
        out = {k: s.get(k) for k in keys if k in s}
        for k, v in s.items():
            if k.startswith(("mean@", "pass@")):
                out[k] = v
        return out

    # ------------------------------------------------------------------
    def export_hf(self, actor_dir, target):
        """FSDP 分片 → HF 目录。返回可直接加载的模型路径。"""
        actor_dir, target = Path(actor_dir), Path(target)
        hf = actor_dir / "huggingface"
        if any(hf.glob("*.safetensors")):
            return str(hf)
        if (target / "config.json").exists() and (any(target.glob("*.safetensors")) or DRY_RUN):
            return str(target)
        target.mkdir(parents=True, exist_ok=True)
        if DRY_RUN:
            q = read_json(hf / "fake_quality.json", {"acc": 0.35})
            write_json(target / "fake_quality.json", q)
            write_json(target / "config.json", {"fake": True})
            return str(target)
        tries = [[python_executable(), "-m", "verl.model_merger", "merge", "--backend", "fsdp",
                  "--local_dir", str(actor_dir), "--target_dir", str(target)]]
        try:
            import importlib.util
            spec = importlib.util.find_spec("verl")
            if spec and spec.origin:
                legacy = Path(spec.origin).parent.parent / "scripts" / "model_merger.py"
                if legacy.exists():
                    tries.append([python_executable(), str(legacy), "merge", "--backend", "fsdp",
                                  "--local_dir", str(actor_dir), "--target_dir", str(target)])
        except Exception:
            pass
        errors = []
        for cmd in tries:
            r = subprocess.run(cmd, capture_output=True, text=True, env=project_env(), timeout=3 * 3600)
            if r.returncode == 0 and any(target.glob("*.safetensors")):
                # 处理器 / 分词器文件从原模型补齐（合并器有时只写权重和 config）
                src = Path(self.config["model"]["path"])
                for name in ("preprocessor_config.json", "video_preprocessor_config.json", "chat_template.json",
                             "chat_template.jinja", "tokenizer.json", "tokenizer_config.json", "vocab.json",
                             "merges.txt", "generation_config.json", "special_tokens_map.json"):
                    if (src / name).exists() and not (target / name).exists():
                        shutil.copy2(src / name, target / name)
                self.report("export_hf", {"actor_dir": str(actor_dir), "target": str(target)})
                return str(target)
            errors.append((r.stderr or r.stdout)[-800:])
        raise RuntimeError(f"检查点合并失败 {actor_dir}: {errors}")

    # ------------------------------------------------------------------
    def compare(self, base, new):
        """逐题对比：新模型修复了多少、弄坏了多少（判断提升是否真实，而不只是总分波动）。"""
        b = {it["index"]: it["correct"] for it in base["items"]}
        fixed = sum(1 for it in new["items"] if it["correct"] and b.get(it["index"]) is False)
        broken = sum(1 for it in new["items"] if not it["correct"] and b.get(it["index"]) is True)
        n = len(new["items"])
        # 配对样本符号检验的正态近似
        z = (fixed - broken) / max(1.0, (fixed + broken) ** 0.5)
        out = {"n": n, "base_acc": base["summary"]["accuracy"], "new_acc": new["summary"]["accuracy"],
               "fixed": fixed, "broken": broken, "sign_test_z": round(z, 2), "significant_95": abs(z) >= 1.96,
               "time": stamp()}
        return out

