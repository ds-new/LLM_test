"""模型结构与算法实现智能体。

职责：
  * 模型：确认 Qwen3-VL-4B 权重就绪（缺失时从 ModelScope / HF 镜像下载）、读取结构与参数量；
  * 框架能力探测：用 `python -m verl.trainer.main_ppo --cfg job` 取得已安装 verl 的完整配置树，
    之后所有 override 都按"键存在才使用"生成，不再用 `++` 静默添加不存在的键（旧脚本里拼错的
    键不会报错，也不会生效）；
  * 奖励函数实现自检（典型几何答案写法的判分用例）；
  * 算法预设与选择：grpo / dapo_lite（clip-higher、去 KL）/ drgrpo（去长度与方差归一偏差），
    依据上一轮的诊断（熵塌缩、截断率、提升幅度）提出下一轮方案，交决策智能体批准。
"""

import re
import subprocess
from pathlib import Path

from .common import DRY_RUN, PROJECT_DIR, RUN_DIR, project_env, python_executable, read_json, write_json

NAME = "architect"

PRESETS = {
    "grpo": {
        "desc": "标准 GRPO：组内标准化优势 + low_var_kl 正则（verl geo3k 官方配方）",
        "use_kl_loss": True, "kl_loss_coef": 0.01, "kl_loss_type": "low_var_kl",
        "clip_ratio_low": 0.2, "clip_ratio_high": 0.2, "loss_agg_mode": "token-mean",
        "norm_adv_by_std_in_grpo": True, "entropy_coeff": 0.0, "format_bonus": 0.1, "lr_scale": 1.0,
    },
    "dapo_lite": {
        "desc": "DAPO 要点：clip-higher（上界 0.28）防熵塌缩、去掉 KL、纯准确率奖励、token 级损失",
        "use_kl_loss": False, "kl_loss_coef": 0.0, "kl_loss_type": "low_var_kl",
        "clip_ratio_low": 0.2, "clip_ratio_high": 0.28, "loss_agg_mode": "token-mean",
        "norm_adv_by_std_in_grpo": True, "entropy_coeff": 0.0, "format_bonus": 0.0, "lr_scale": 1.0,
    },
    "drgrpo": {
        "desc": "Dr.GRPO：不按组内标准差归一、序列求和/常数归一，消除长度与难度偏置",
        "use_kl_loss": False, "kl_loss_coef": 0.0, "kl_loss_type": "low_var_kl",
        "clip_ratio_low": 0.2, "clip_ratio_high": 0.2, "loss_agg_mode": "seq-mean-token-sum-norm",
        "norm_adv_by_std_in_grpo": False, "entropy_coeff": 0.0, "format_bonus": 0.0, "lr_scale": 1.0,
    },
}

# 每个预设依赖的 verl 配置键；缺失则该预设不可用（旧版 verl）
PRESET_KEYS = {
    "dapo_lite": ["actor_rollout_ref.actor.clip_ratio_high"],
    "drgrpo": ["algorithm.norm_adv_by_std_in_grpo"],
}


def flatten(d, prefix=""):
    out = set()
    if isinstance(d, dict):
        for k, v in d.items():
            key = f"{prefix}.{k}" if prefix else str(k)
            out.add(key)
            out |= flatten(v, key)
    return out


class ArchitectAgent:
    def __init__(self, config, bus):
        self.config = config
        self.bus = bus

    def report(self, action, content):
        self.bus.post(NAME, "decision", "report", {"action": action, **content})

    # ------------------------------------------------------------------ 模型
    def ensure_model(self):
        path = Path(self.config["model"]["path"])
        if DRY_RUN:
            path.mkdir(parents=True, exist_ok=True)
            write_json(path / "config.json", {"architectures": ["FakeQwen3VL"], "model_type": "qwen3_vl",
                                             "text_config": {"num_hidden_layers": 36, "hidden_size": 2560},
                                             "vision_config": {"depth": 24}})
        if DRY_RUN or ((path / "config.json").exists() and any(path.glob("*.safetensors"))):
            return self.inspect_model()
        attempts = []
        py = python_executable()
        ms = (f"from modelscope import snapshot_download; "
              f"snapshot_download({self.config['model']['modelscope_id']!r}, local_dir={str(path)!r})")
        hf = (f"from huggingface_hub import snapshot_download; "
              f"snapshot_download({self.config['model']['hf_id']!r}, local_dir={str(path)!r})")
        # 用 python -c 调库函数，不依赖 modelscope / huggingface-cli 命令是否在 PATH 中
        for name, cmd in (("modelscope", [py, "-c", ms]), ("hf-mirror", [py, "-c", hf])):
            self.bus.post(NAME, "decision", "status", {"action": "download_model", "via": name, "target": str(path)})
            try:
                r = subprocess.run(cmd, env=project_env(), capture_output=True, text=True, timeout=3 * 3600)
                attempts.append({"via": name, "code": r.returncode, "tail": (r.stderr or r.stdout)[-500:]})
                if r.returncode == 0 and (path / "config.json").exists():
                    return self.inspect_model()
            except Exception as e:  # noqa
                attempts.append({"via": name, "error": f"{type(e).__name__}: {e}"})
        raise RuntimeError(f"模型不存在且下载失败: {path}; {attempts}")

    def inspect_model(self):
        path = Path(self.config["model"]["path"])
        cfg = read_json(path / "config.json", {})
        text = cfg.get("text_config", cfg)
        vis = cfg.get("vision_config", {})
        index = read_json(path / "model.safetensors.index.json", {})
        total = (index.get("metadata") or {}).get("total_size")
        params_b = round(total / 2 / 1e9, 2) if total else self.config["train"]["params_b"]  # bf16 = 2 字节/参数
        info = {"path": str(path), "architectures": cfg.get("architectures"), "model_type": cfg.get("model_type"),
                "text_layers": text.get("num_hidden_layers"), "hidden_size": text.get("hidden_size"),
                "vision_depth": vis.get("depth"), "params_b": params_b,
                "tie_word_embeddings": text.get("tie_word_embeddings", cfg.get("tie_word_embeddings"))}
        self.report("inspect_model", info)
        return info

    # ------------------------------------------------------------------ verl 能力
    def probe_verl(self):
        from .trainer_agent import verl_base_cmd
        cmd = verl_base_cmd() + ["--cfg", "job"]
        try:
            r = subprocess.run(cmd, env=project_env(), capture_output=True, text=True, timeout=600, cwd=str(PROJECT_DIR))
            if r.returncode != 0:
                raise RuntimeError(r.stderr[-1500:])
            keys = sorted(flatten(extract_yaml(r.stdout)))
            if len(keys) < 50:
                raise RuntimeError(f"配置树过小（{len(keys)} 个键），输出可能不是 verl 配置")
        except Exception as e:  # noqa
            out = {"ok": False, "error": f"{type(e).__name__}: {str(e)[-800:]}", "mode": "legacy_plus_plus"}
            write_json(RUN_DIR / "verl_capabilities.json", out)
            self.report("probe_verl", out)
            return out
        version = None
        try:
            v = subprocess.run([python_executable(), "-c", "import verl;print(getattr(verl,'__version__','?'))"],
                               env=project_env(), capture_output=True, text=True, timeout=120)
            version = v.stdout.strip() or None
        except Exception:
            pass
        presets = [p for p in PRESETS if all(k in keys for k in PRESET_KEYS.get(p, []))]
        out = {"ok": True, "mode": "strict", "verl_version": version, "n_keys": len(keys), "keys": keys,
               "available_presets": presets}
        write_json(RUN_DIR / "verl_capabilities.json", out)
        self.report("probe_verl", {k: v for k, v in out.items() if k != "keys"})
        return out

    # ------------------------------------------------------------------ 奖励自检
    def reward_selftest(self):
        import os
        from reward_fn_geo3k import compute_score, grade
        os.environ.setdefault("GEO3K_FORMAT_BONUS", "0.1")
        T = "<think>r</think> "
        cases = [
            (T + "\\boxed{2\\sqrt{221}}", "2 \\sqrt { 221 }", 1), (T + "\\boxed{\\sqrt{884}}", "2 \\sqrt { 221 }", 1),
            (T + "\\boxed{29.73}", "2 \\sqrt { 221 }", 1), (T + "\\boxed{30}", "2 \\sqrt { 221 }", 0),
            (T + "\\boxed{\\dfrac{26}{3}}", "\\frac { 26 } { 3 }", 1), (T + "\\boxed{120^{\\circ}}", "120", 1),
            (T + "\\boxed{x=3}", "3", 1), (T + "\\boxed{12\\pi}", "12 \\pi", 1),
            ("<think> so x = 3 </think> answer 3", "3", 0), (T + "\\boxed{3.01}", "3", 0),
            (T + "\\boxed{4} wait \\boxed{3}", "3", 1),
        ]
        failures = []
        for s, gt, exp in cases:
            got = grade(s, gt)["acc"]
            if got != exp:
                failures.append({"output": s, "gt": gt, "expected": exp, "got": got})
        sample = compute_score("hiyouga/geometry3k", T + "\\boxed{3}", "3")
        ok = not failures and isinstance(sample, dict) and abs(sample["score"] - 1.1) < 1e-9
        out = {"passed": ok, "cases": len(cases), "failures": failures, "sample_return": sample}
        self.report("reward_selftest", out)
        return out

    # ------------------------------------------------------------------ 算法选择
    def available(self):
        caps = read_json(RUN_DIR / "verl_capabilities.json", {}) or {}
        return caps.get("available_presets") or ["grpo"]

    def propose(self, rounds):
        """rounds: 已完成各轮的结果（含 diagnostics）。返回下一轮的算法方案和理由。"""
        order = [p for p in self.config["algorithm"]["presets_order"] if p in self.available()] or ["grpo"]
        tried = [r["preset"] for r in rounds]
        if not rounds:
            name = order[0]
            return {"preset": name, **PRESETS[name], "max_response_length": self.config["train"]["max_response_length"],
                    "rationale": f"首轮用 {name}（{PRESETS[name]['desc']}）建立可复现基线"}
        last = rounds[-1]
        diag = last.get("diagnostics", {})
        resp_len = last["plan"]["max_response_length"]
        reasons, name = [], None
        if diag.get("entropy_collapse") and "dapo_lite" in order and "dapo_lite" not in tried:
            name = "dapo_lite"
            reasons.append(f"上一轮熵从 {diag.get('entropy_first')} 降到 {diag.get('entropy_last')}（塌缩），改用 clip-higher")
        elif (diag.get("response_clip_ratio") or 0) > 0.08 and "drgrpo" in order and "drgrpo" not in tried:
            name = "drgrpo"
            resp_len = min(4096, int(resp_len * 1.5))
            reasons.append(f"截断比例 {diag['response_clip_ratio']:.1%} 偏高：回复长度上限提到 {resp_len}，并用 Dr.GRPO 去除长度偏置")
        if name is None:
            rest = [p for p in order if p not in tried]
            if not rest:
                return None
            name = rest[0]
            gain = last.get("val_gain")
            reasons.append(f"上一轮验证提升 {gain} 个百分点；按顺序尝试下一个预设 {name}（{PRESETS[name]['desc']}）做对照")
        if (diag.get("response_clip_ratio") or 0) > 0.08 and resp_len == last["plan"]["max_response_length"]:
            resp_len = min(4096, int(resp_len * 1.5))
            reasons.append(f"截断比例偏高，回复长度上限提到 {resp_len}")
        return {"preset": name, **PRESETS[name], "max_response_length": resp_len, "rationale": "；".join(reasons)}


def describe_presets():
    return {k: v["desc"] for k, v in PRESETS.items()}


def extract_yaml(stdout):
    """`--cfg job` 的输出前面可能混有 import 时打印的日志，从第一个顶层键开始解析。"""
    import yaml
    lines = stdout.splitlines()
    start = next((i for i, l in enumerate(lines) if re.match(r"^[A-Za-z_][\w]*:(\s|$)", l)), 0)
    return yaml.safe_load("\n".join(lines[start:])) or {}
