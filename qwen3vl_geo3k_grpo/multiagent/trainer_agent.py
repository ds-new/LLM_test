"""模型训练配置、执行与监控智能体。

配置：按实测 GPU（数量、显存）、模型参数量、数据盘剩余空间和算法预设生成 verl 配置；
      只使用已安装 verl 中存在的配置键（由架构智能体探测），每次启动的完整命令写入 command.sh 便于复现。
执行：verl 作为独立进程组运行（FSDP 训练 + SGLang/vLLM rollout），断点续训（trainer.resume_mode=auto）。
监控：增量解析训练日志里的 "step:N - key:value - …" 行，写入 metrics.jsonl，并把每一步交给决策智能体审核；
      检测进程崩溃（区分 GPU OOM / 内存 OOM / NCCL / 其他）、日志停滞、磁盘不足，按降级阶梯重启：
        L0 默认 → L1 降 SGLang 显存占比、减半每卡 token 上限 → L2 打开优化器 offload
        → L3 再打开参数 offload → L4 每题采样数降到 5。
检查点保护：verl 只保留最近 k 个检查点，最佳验证分数对应的检查点会被轮转删除。每出现新最佳，
      用硬链接把它的模型分片"钉住"（同一文件系统，不额外占空间；原目录被删后仍然保留）。
"""

import json
import os
import re
import shutil
import signal
import subprocess
import time
from pathlib import Path

from .common import (DRY_RUN, PROJECT_DIR, RUN_DIR, control_signal, disk_free_gb, now, project_env, python_executable, read_json,
                     stamp, write_json, write_text)

NAME = "trainer"
MAX_LEVEL = 4


def verl_base_cmd():
    custom = os.environ.get("MA_VERL_CMD")
    if custom:
        return custom.split()
    if DRY_RUN:
        return [python_executable(), "-m", "multiagent.fake_verl"]
    return [python_executable(), "-m", "verl.trainer.main_ppo"]


# ---------------------------------------------------------------------------
# 日志解析
# ---------------------------------------------------------------------------
STEP_RE = re.compile(r"step:(\d+)\s+-\s+(.*)$")
NUM_RE = re.compile(r"^(?:np\.\w+\()?\s*([-+]?(?:\d+\.?\d*(?:[eE][-+]?\d+)?|nan|inf))\s*\)?$")


def parse_step_line(line):
    """解析 verl 控制台日志行；Ray 会加上 "(TaskRunner pid=…)" 前缀，数值可能是 np.float64(…)。"""
    m = STEP_RE.search(line.strip())
    if not m:
        return None
    metrics = {}
    for part in m.group(2).split(" - "):
        key, sep, val = part.partition(":")
        if not sep:
            continue
        nm = NUM_RE.match(val.strip())
        if nm:
            try:
                metrics[key.strip()] = float(nm.group(1))
            except ValueError:
                pass
    return int(m.group(1)), metrics


def val_accuracy(metrics):
    """验证准确率（百分数）。优先奖励函数单独上报的 acc，退回 score（含格式分）时标注来源。"""
    pats = [(r"^val-core/.*/acc/mean@\d+$", "acc"), (r"^val[-/].*acc", "acc"),
            (r"^val[-/].*(test_score|reward|score)", "score")]
    for pat, kind in pats:
        for k, v in metrics.items():
            if re.search(pat, k) and v == v:
                return (v * 100 if abs(v) <= 1.5 else v), kind, k
    return None, None, None


def classify_crash(text):
    t = text[-20000:]
    # 含 vLLM 唤醒时的 "CUDA Error: out of memory"（cumem_allocator）——续训后显存碎片常见，应降级而非原样重试
    if re.search(r"CUDA out of memory|OutOfMemoryError: CUDA|torch\.OutOfMemoryError|CUBLAS_STATUS_ALLOC_FAILED|"
                 r"CUDA Error: out of memory|cudaErrorMemoryAllocation", t):
        return "gpu_oom"
    if re.search(r"node running low on memory|Killed|MemoryError|Cannot allocate memory", t):
        return "host_oom"
    if re.search(r"NCCL (?:error|WARN.*timeout)|ProcessGroupNCCL|Watchdog caught collective", t):
        return "nccl"
    if re.search(r"No space left on device", t):
        return "disk_full"
    # 确定性错误：重试多少次结果都一样，应立即停下等人修
    if re.search(r"ImportError|ModuleNotFoundError|undefined symbol|cannot import name", t):
        return "env_error"
    if re.search(r"ConfigAttributeError|ConfigKeyError|Could not override|is not in struct|"
                 r"omegaconf\.errors|hydra\.errors|Key '.*' not in|KV cache is needed, which is larger than", t):
        return "config_error"
    return "other"


# ---------------------------------------------------------------------------
# 资源探测与规划
# ---------------------------------------------------------------------------
def probe_gpus():
    if DRY_RUN:
        return [{"index": 0, "name": "FakeGPU-80G", "mem_gb": 80.0, "used_gb": 0.5}]
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=index,name,memory.total,memory.used",
                              "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=30).stdout
    except Exception:
        return []
    gpus = []
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    allow = set(visible.split(",")) if visible else None
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 4 and (allow is None or parts[0] in allow):
            gpus.append({"index": int(parts[0]), "name": parts[1], "mem_gb": round(float(parts[2]) / 1024, 1),
                         "used_gb": round(float(parts[3]) / 1024, 1)})
    return gpus


def check_packages():
    if DRY_RUN:
        return {"torch": "dry", "verl": "dry", "sglang": "dry", "vllm": None, "transformers": "dry", "math_verify": "dry"}
    code = ("import importlib,json\nout={}\n"
            "for m in ['torch','verl','sglang','vllm','transformers','math_verify','ray','datasets','huggingface_hub','scipy.special','torch._inductor.select_algorithm']:\n"
            "  try:\n    mod=importlib.import_module(m); out[m]=getattr(mod,'__version__','installed')\n"
            "  except Exception as e:\n    out[m]=None; out.setdefault('import_errors',{})[m]=(type(e).__name__+': '+str(e))[:300]\n"
            "try:\n  import torch; out['cuda']=torch.version.cuda\nexcept Exception: out['cuda']=None\n"
            "print(json.dumps(out))")
    r = subprocess.run([python_executable(), "-c", code], capture_output=True, text=True, timeout=300, env=project_env())
    try:
        return json.loads(r.stdout.strip().splitlines()[-1])
    except Exception:
        return {"error": r.stderr[-800:]}


def choose_attn(want="auto"):
    """flash-attn 的 CUDA 扩展能导入就用 flash_attention_2，否则退回 PyTorch SDPA（较慢但无需编译）。"""
    if want in ("flash_attention_2", "sdpa"):
        return want
    if DRY_RUN:
        return "sdpa" if os.environ.get("FAKE_NO_FLASH_ATTN") else "flash_attention_2"
    r = subprocess.run([python_executable(), "-c", "import flash_attn_2_cuda; from flash_attn import flash_attn_func"],
                       capture_output=True, text=True, timeout=300, env=project_env())
    return "flash_attention_2" if r.returncode == 0 else "sdpa"


def rollout_engine(want, packages):
    if want in ("sglang", "vllm"):
        return want if packages.get(want) else None
    for name in ("sglang", "vllm"):
        if packages.get(name):
            return name
    return None


def memory_plan(gpu_mem_gb, n_gpus, params_b, level, max_len, vocab=151936):
    """FSDP 训练状态约 16 字节/参数（fp32 主权重 4 + 梯度 4 + Adam 8），按卡数分片。"""
    state = 16 * params_b / n_gpus
    budget = 0.55 * gpu_mem_gb
    opt_off = state > budget or level >= 2
    param_off = (opt_off and 8 * params_b / n_gpus > budget) or level >= 3
    if gpu_mem_gb >= 70:
        gmu, k = 0.55, 4
    elif gpu_mem_gb >= 40:
        gmu, k = 0.45, 2
    else:
        gmu, k = 0.35, 1
    if level >= 1:
        gmu, k = max(0.25, gmu - 0.1), max(1, k // 2)
    tp = 2 if (gpu_mem_gb < 30 and n_gpus >= 2) else 1
    # 词表很大（Qwen3-VL 约 15 万）时，一个 micro-batch 的 fp32 logits 及其 softmax/熵中间量约 12 字节 × 词表 / token，
    # 往往比模型本身更占显存：按显存预算限制每卡 token 数（至少放得下一条最长序列），降级时继续收紧
    frac = {0: 0.2, 1: 0.2, 2: 0.15, 3: 0.1}.get(level, 0.08)
    cap = int(frac * gpu_mem_gb * 2 ** 30 / (vocab * 12))
    lp_tokens = max(max_len, min(max_len * k * 2, cap))
    ppo_tokens = max(max_len, min(max_len * k, cap))
    return {"param_offload": param_off, "optimizer_offload": opt_off, "gpu_memory_utilization": round(gmu, 2),
            "max_token_len_per_gpu": ppo_tokens, "log_prob_max_token_len_per_gpu": lp_tokens,
            "rollout_tp": tp, "level": level,
            "note": f"训练状态约 {state:.1f}GB/卡，预算 {budget:.1f}GB/卡"}


def ckpt_size_gb(params_b, with_optimizer=True):
    return params_b * (12 if with_optimizer else 4) * 1.02 + 0.1


# ---------------------------------------------------------------------------
class TrainerAgent:
    def __init__(self, config, bus, board):
        self.config = config
        self.tcfg = config["train"]
        self.bus = bus
        self.board = board

    def report(self, action, content, to="decision"):
        self.bus.post(NAME, to, "report", {"action": action, **content})

    # ------------------------------------------------------------------ 环境
    def env_check(self):
        gpus = probe_gpus()
        pk = check_packages()
        free = disk_free_gb(RUN_DIR)
        out = {"gpus": gpus, "packages": pk, "disk_free_gb": round(free, 1), "run_dir": str(RUN_DIR)}
        problems = []
        if not gpus:
            problems.append("未检测到 NVIDIA GPU（nvidia-smi 不可用）")
        for m in ("torch", "verl", "transformers"):
            if not pk.get(m):
                problems.append(f"缺少 {m}（或导入失败，可能是依赖版本冲突，如 huggingface-hub 与 transformers 不兼容）")
        for m, why in (("scipy.special", "scipy 无法正常导入（常见于 numpy/scipy 有旧版本残留文件），verl 与评估都会失败"),
                       ("torch._inductor.select_algorithm", "torch 编译器模块导入失败（常见于 torch 目录有旧版本残留文件）")):
            if not DRY_RUN and pk.get(m) is None:
                problems.append(f"{why}: {(pk.get('import_errors') or {}).get(m, '')[-200:]}")
        engine = rollout_engine(self.tcfg.get("rollout", "auto"), pk)
        out["rollout_engine"] = engine
        if engine is None:
            problems.append("没有可用的 rollout 引擎：sglang 与 vllm 都未安装（或导入失败）")
        if not pk.get("math_verify"):
            out["warning"] = "math-verify 未安装，判分使用内置规范化 + 数值求值（仍然可用）"
        p_b = self.tcfg["params_b"]
        need = 2 * ckpt_size_gb(p_b, False) + p_b * 2 + 5   # 1 个轮转检查点 + 1 个钉住的最佳 + 导出 + 余量
        good = ckpt_size_gb(p_b, True) + ckpt_size_gb(p_b, False) + p_b * 2 + 5
        if free < need:
            problems.append(f"数据盘剩余 {free:.0f}GB：检查点约 {ckpt_size_gb(p_b, False):.0f}GB（不含优化器）/ "
                            f"{ckpt_size_gb(p_b, True):.0f}GB（含优化器），每轮导出约 {p_b * 2:.0f}GB，至少需要约 {need:.0f}GB")
        elif free < good:
            out["warning_disk"] = (f"数据盘剩余 {free:.0f}GB，只够保存不含优化器状态的检查点（崩溃续训时 Adam 状态重置）；"
                                   f"约 {good:.0f}GB 以上可保存完整检查点")
        out["problems"] = problems
        self.report("env_check", out)
        return out

    # ------------------------------------------------------------------ 规划
    def make_plan(self, round_idx, proposal, data, model_info, level=0, prev_plan=None):
        gpus = probe_gpus()
        want = self.tcfg["gpus"]
        n = len(gpus) if want == "auto" else min(int(want), len(gpus))
        if n == 0:
            raise RuntimeError("没有可用 GPU")
        mem = min(g["mem_gb"] for g in gpus[:n])
        params_b = model_info.get("params_b") or self.tcfg["params_b"]
        max_prompt = data.get("max_prompt_length", self.tcfg["max_prompt_length"])
        max_resp = proposal.get("max_response_length", self.tcfg["max_response_length"])
        mp = memory_plan(mem, n, params_b, level, max_prompt + max_resp)
        nsamp = self.tcfg["n_samples"] if level < 4 else min(5, self.tcfg["n_samples"])
        bsz = self.tcfg["train_batch_size"]
        while (bsz * nsamp) % n:
            bsz -= 1
        mini = min(self.tcfg["ppo_mini_batch_size"], bsz)
        while bsz % mini:
            mini -= 1
        rdir = RUN_DIR / "rounds" / f"round_{round_idx}"
        save_dir = rdir / "ckpt"
        # 磁盘：最佳检查点钉住后只占模型分片（4 字节/参数）；每个轮转检查点含优化器约 12 字节/参数
        # 预留本轮最佳模型的 HF 导出（bf16，2 字节/参数）+ 5GB 余量；之前各轮的导出已经计入已用空间
        # 轮次结束时先删轮转检查点、再从钉住的硬链接副本导出 HF 模型，所以导出不需要额外预留；只留 5GB 余量
        # verl 的轮转：max_ckpt_to_keep=1 时先写新检查点、写完才删旧的（>=2 时先删到 keep-1 个再写），
        # 所以峰值要能同时放下 max(keep, 2) 个轮转检查点，再加上钉住的最佳（fp32 分片，4 字节/参数）
        free = disk_free_gb(RUN_DIR) - 5
        pinned = ckpt_size_gb(params_b, False)
        disk_warning = None
        with_opt = True
        slots = int((free - pinned) // ckpt_size_gb(params_b, True))
        if slots < 2:
            with_opt = False
            slots = int((free - pinned) // ckpt_size_gb(params_b, False))
        if slots < 1:
            raise RuntimeError(f"数据盘剩余空间不足以保存检查点（可用约 {free:.0f}GB）。请在 AutoDL 控制台扩容数据盘。")
        if slots < 2:
            need = pinned + 2 * ckpt_size_gb(params_b, False) + 5
            disk_warning = (f"数据盘只够同时放 1 个轮转检查点（verl 写完新的才删旧的，需要约 {need:.0f}GB，"
                            f"现可用 {free + 5:.0f}GB）：第二次保存时可能写满磁盘，请扩容或清理数据盘")
        keep = min(max(slots, 1), 3)
        p = {
            "round": round_idx, "preset": proposal["preset"], "rationale": proposal.get("rationale"),
            "rollout": (self.board.get("env") or {}).get("rollout_engine") or "sglang",
            "model_path": data.get("init_model") or self.config["model"]["path"],
            "train_file": str(data["train_file"]), "val_file": str(data["val_file"]),
            "n_gpus": n, "gpu_mem_gb": mem, "gpu_name": gpus[0]["name"],
            "train_batch_size": bsz, "ppo_mini_batch_size": mini, "n_samples": nsamp,
            "lr": self.tcfg["lr"] * proposal.get("lr_scale", 1.0),
            "max_prompt_length": max_prompt, "max_response_length": max_resp,
            "total_epochs": self.tcfg["total_epochs"], "save_freq": self.tcfg["save_freq"],
            "keep_ckpts": keep, "save_optimizer": with_opt, "memory": mp, "disk_warning": disk_warning,
            "algo": {k: proposal[k] for k in ("use_kl_loss", "kl_loss_coef", "kl_loss_type", "clip_ratio_low",
                                              "clip_ratio_high", "loss_agg_mode", "norm_adv_by_std_in_grpo",
                                              "entropy_coeff")},
            "reward_env": {"GEO3K_FORMAT_BONUS": proposal.get("format_bonus", 0.1), "GEO3K_STRICT_BOXED": 1},
            "attn_implementation": choose_attn(self.tcfg.get("attn_implementation", "auto")),
            "round_dir": str(rdir), "save_dir": str(save_dir), "created": stamp(),
        }
        warns = []
        if not with_opt:
            warns.append("磁盘不足以保存优化器状态：崩溃续训时 Adam 状态会重置")
        if disk_warning:
            warns.append(disk_warning)
        if warns:
            p["warning"] = "；".join(warns)
        return p

    def overrides(self, plan):
        """返回 (override 列表, 被丢弃的可选键)。strict 模式下键名直接写（拼错会让 Hydra 报错）。"""
        caps = read_json(RUN_DIR / "verl_capabilities.json", {}) or {}
        strict = caps.get("mode") == "strict"
        keys = set(caps.get("keys") or [])
        mp, a = plan["memory"], plan["algo"]
        A, R, F = "actor_rollout_ref.actor", "actor_rollout_ref.rollout", "actor_rollout_ref.ref"
        logger = ["console"] + (["wandb"] if os.environ.get("WANDB_API_KEY") else [])
        items = [  # (键, 值, 是否必需)
            ("algorithm.adv_estimator", "grpo", True),
            ("algorithm.use_kl_in_reward", False, True),
            ("algorithm.norm_adv_by_std_in_grpo", a["norm_adv_by_std_in_grpo"], plan["preset"] == "drgrpo"),
            ("data.train_files", plan["train_file"], True), ("data.val_files", plan["val_file"], True),
            ("data.train_batch_size", plan["train_batch_size"], True),
            ("data.max_prompt_length", plan["max_prompt_length"], True),
            ("data.max_response_length", plan["max_response_length"], True),
            ("data.filter_overlong_prompts", True, False), ("data.truncation", "error", True),
            ("data.image_key", "images", True), ("data.shuffle", True, False),
            ("actor_rollout_ref.model.path", plan["model_path"], True),
            ("actor_rollout_ref.model.trust_remote_code", True, False),
            # 训练进程中显式导入运行时补丁（Qwen3-VL Conv3d → F.linear），与 sitecustomize 互为保险
            ("actor_rollout_ref.model.external_lib", "geo3k_patches", False),
            # remove_padding 把多条序列拼成一条变长序列，只有 flash-attention 能正确处理；SDPA 下必须关闭
            ("actor_rollout_ref.model.use_remove_padding", plan.get("attn_implementation", "flash_attention_2") == "flash_attention_2", True),
            ("actor_rollout_ref.model.enable_gradient_checkpointing", True, True),
            (f"{A}.optim.lr", plan["lr"], True), (f"{A}.ppo_epochs", 1, True),
            (f"{A}.ppo_mini_batch_size", plan["ppo_mini_batch_size"], True),
            (f"{A}.use_dynamic_bsz", True, True),
            (f"{A}.ppo_max_token_len_per_gpu", mp["max_token_len_per_gpu"], True),
            (f"{A}.use_kl_loss", a["use_kl_loss"], True), (f"{A}.kl_loss_coef", a["kl_loss_coef"], True),
            (f"{A}.kl_loss_type", a["kl_loss_type"], True), (f"{A}.entropy_coeff", a["entropy_coeff"], True),
            (f"{A}.clip_ratio", a["clip_ratio_low"], True),
            (f"{A}.clip_ratio_low", a["clip_ratio_low"], False),
            (f"{A}.clip_ratio_high", a["clip_ratio_high"], plan["preset"] == "dapo_lite"),
            (f"{A}.loss_agg_mode", a["loss_agg_mode"], plan["preset"] == "drgrpo"),
            (f"{A}.fsdp_config.param_offload", mp["param_offload"], True),
            (f"{A}.fsdp_config.optimizer_offload", mp["optimizer_offload"], True),
            (f"{R}.name", plan.get("rollout", "sglang"), True), (f"{R}.tensor_model_parallel_size", mp["rollout_tp"], True),
            (f"{R}.gpu_memory_utilization", mp["gpu_memory_utilization"], True),
            (f"{R}.n", plan["n_samples"], True), (f"{R}.free_cache_engine", True, False),
            # 不设时 vLLM 按模型原生上下文（Qwen3-VL 为 262144）预留 KV cache，启动即报显存不足
            (f"{R}.max_model_len", plan["max_prompt_length"] + plan["max_response_length"], False),
            (f"{R}.log_prob_use_dynamic_bsz", True, True),
            (f"{R}.log_prob_max_token_len_per_gpu", mp["log_prob_max_token_len_per_gpu"], True),
            (f"{R}.val_kwargs.do_sample", False, False), (f"{R}.val_kwargs.n", 1, False),
            (f"{F}.log_prob_use_dynamic_bsz", True, True),
            (f"{F}.log_prob_max_token_len_per_gpu", mp["log_prob_max_token_len_per_gpu"], True),
            # ref 只做前向、bf16 存储（4B 模型分两卡每卡约 4.4GB）；offload 到 CPU 会让每个 micro-batch 都搬一次参数，
            # 实测使 ref log-prob 变成 CPU 瓶颈。只有显存级别 L3（参数 offload）时才一并 offload
            (f"{F}.fsdp_config.param_offload", mp["param_offload"], True),
            ("custom_reward_function.path", str(PROJECT_DIR / "reward_fn_geo3k.py"), True),
            ("custom_reward_function.name", "compute_score", True),
            ("trainer.critic_warmup", 0, True), ("trainer.logger", "[" + ",".join(logger) + "]", True),
            ("trainer.project_name", "qwen3vl_4b_geo3k_multiagent", True),
            ("trainer.experiment_name", f"round{plan['round']}_{plan['preset']}", True),
            ("trainer.n_gpus_per_node", plan["n_gpus"], True), ("trainer.nnodes", 1, True),
            ("trainer.save_freq", plan["save_freq"], True), ("trainer.test_freq", plan["save_freq"], True),
            ("trainer.total_epochs", plan["total_epochs"], True),
            ("trainer.default_local_dir", plan["save_dir"], True),
            ("trainer.resume_mode", "auto", False), ("trainer.val_before_train", True, False),
            ("trainer.max_actor_ckpt_to_keep", plan["keep_ckpts"], False),
        ]
        if not plan["save_optimizer"]:
            if f"{A}.checkpoint.save_contents" in keys:
                items.append((f"{A}.checkpoint.save_contents", "[model,extra]", False))
            elif f"{A}.checkpoint.contents" in keys:
                items.append((f"{A}.checkpoint.contents", "[model,extra]", False))

        def fmt(v):
            if isinstance(v, bool):
                return "True" if v else "False"
            return str(v)

        out, dropped, missing = [], [], []
        for key, val, required in items:
            if strict and key not in keys:
                (missing if required else dropped).append(key)
                continue
            out.append(f"{key}={fmt(val)}" if strict else f"++{key}={fmt(val)}")
        if plan.get("attn_implementation") == "sdpa":
            key = "actor_rollout_ref.model.override_config"
            if strict and key not in keys:
                raise RuntimeError("当前 verl 没有 model.override_config，无法改用 SDPA；请修复 flash-attn")
            out.append(("+" if strict else "++") + key + ".attn_implementation=sdpa")
        if missing:
            raise RuntimeError(f"已安装的 verl 缺少必需配置键（版本过旧？）: {missing}")
        return out, dropped

    # ------------------------------------------------------------------ 执行与监控
    def run_round(self, plan, deadline, review, attempt_state=None):
        """运行一轮训练直到完成 / 决策智能体叫停 / 截止 / 不可恢复的失败。

        review(step, metrics) -> None 或 "stop:<原因>"，每个解析到的步都会调用。
        """
        rdir = Path(plan["round_dir"])
        (rdir / "logs").mkdir(parents=True, exist_ok=True)
        state = attempt_state or {"attempts": 0, "level": plan["memory"]["level"], "history": []}
        while True:
            state["attempts"] += 1
            if state["level"] != plan["memory"]["level"]:
                plan["memory"] = memory_plan(plan["gpu_mem_gb"], plan["n_gpus"],
                                             self.board.get("model_info", {}).get("params_b") or self.tcfg["params_b"],
                                             state["level"], plan["max_prompt_length"] + plan["max_response_length"])
                if state["level"] >= 4:
                    plan["n_samples"] = min(5, plan["n_samples"])
            res = self._launch_and_watch(plan, deadline, review, state["attempts"])
            state["history"].append({k: res[k] for k in ("status", "exit_code", "last_step", "crash_class") if k in res})
            self.board.set("train_attempts", state)
            if res["status"] != "crashed" and res["status"] != "stalled":
                res["attempt_state"] = state
                return res
            if state["attempts"] > self.tcfg["max_crash_retries"]:
                res["status"] = "failed"
                res["attempt_state"] = state
                return res
            cls = res.get("crash_class")
            prev = state["history"][-2]["status"] if len(state["history"]) >= 2 else None
            # GPU OOM 直接降级；卡死先原样重试一次，连续两次卡死再降级（常见原因是显存碎片/交换）
            if cls == "gpu_oom" or (res["status"] == "stalled" and prev == "stalled"):
                if state["level"] >= MAX_LEVEL:
                    res["status"] = "failed"
                    return res
                state["level"] += 1
                self.report("degrade", {"reason": cls or res["status"], "new_level": state["level"],
                                        "step": res.get("last_step")})
            elif cls in ("disk_full", "env_error", "config_error"):
                res["status"] = "failed"
                res["attempt_state"] = state
                return res
            elif cls == "host_oom" and plan["memory"]["param_offload"]:
                self.report("host_oom", {"message": "主机内存不足且已开启参数 offload，无法继续降级"})
                res["status"] = "failed"
                return res
            self.report("restart_training", {"attempt": state["attempts"] + 1, "level": state["level"],
                                             "resume_from_step": res.get("last_saved_step")})
            time.sleep(5 if not DRY_RUN else 0.1)

    def _launch_and_watch(self, plan, deadline, review, attempt):
        rdir = Path(plan["round_dir"])
        args, dropped = self.overrides(plan)
        cmd = verl_base_cmd() + args
        env = project_env({**plan["reward_env"], "HYDRA_FULL_ERROR": "1", "RAY_DEDUP_LOGS": "0"})
        if DRY_RUN:
            env["FAKE_OOM_IF_GMU_ABOVE"] = os.environ.get("FAKE_OOM_IF_GMU_ABOVE", "0.5")
        log = rdir / "logs" / f"train_attempt{attempt}.log"
        write_text(rdir / "command.sh", "#!/bin/bash\n# " + stamp() + f"  attempt {attempt}\ncd " + str(PROJECT_DIR) +
                   "\n" + " ".join(f"{k}={v}" for k, v in plan["reward_env"].items()) + " \\\n" +
                   " \\\n    ".join(cmd) + "\n")
        write_json(rdir / "plan.json", plan)
        self.report("launch", {"round": plan["round"], "attempt": attempt, "preset": plan["preset"],
                               "level": plan["memory"]["level"], "log": str(log), "dropped_optional_keys": dropped,
                               "memory": plan["memory"], "n_gpus": plan["n_gpus"]})
        f = log.open("w", encoding="utf-8")
        proc = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=str(PROJECT_DIR), env=env,
                                start_new_session=True)
        started = now()
        last_progress = started
        last_step, offset, buf = None, 0, ""
        metrics_path = rdir / "metrics.jsonl"
        stop_reason = None
        poll = self.tcfg["poll_seconds"]
        disk_checked = 0
        try:
            while True:
                time.sleep(poll)
                with log.open("r", encoding="utf-8", errors="replace") as lf:
                    lf.seek(offset)
                    chunk = lf.read()
                    offset = lf.tell()
                buf += chunk
                lines = buf.split("\n")
                buf = lines.pop()
                for line in lines:
                    parsed = parse_step_line(line)
                    if not parsed:
                        continue
                    step, metrics = parsed
                    last_step, last_progress = step, now()
                    with metrics_path.open("a", encoding="utf-8") as mf:
                        mf.write(json.dumps({"t": round(now(), 1), "attempt": attempt, "step": step, **metrics}) + "\n")
                    verdict = review(step, metrics)
                    if verdict and verdict.startswith("stop"):
                        stop_reason = verdict
                        break
                if stop_reason:
                    self._terminate(proc)
                    return self._result("stopped", proc, log, last_step, plan, reason=stop_reason)
                code = proc.poll()
                if code is not None:
                    status = "completed" if code == 0 else "crashed"
                    return self._result(status, proc, log, last_step, plan)
                if now() >= deadline:
                    self._terminate(proc)
                    return self._result("deadline", proc, log, last_step, plan)
                limit = (self.tcfg["first_step_minutes"] if last_step is None else self.tcfg["stall_minutes"]) * 60
                if now() - last_progress > limit:
                    self._terminate(proc)
                    return self._result("stalled", proc, log, last_step, plan)
                if now() - disk_checked > 300:
                    disk_checked = now()
                    if disk_free_gb(RUN_DIR) < 3:
                        self._terminate(proc)
                        res = self._result("crashed", proc, log, last_step, plan)
                        res["crash_class"] = "disk_full"
                        return res
                if control_signal():
                    self._terminate(proc)
                    return self._result("interrupted", proc, log, last_step, plan)
        finally:
            f.close()

    def _terminate(self, proc):
        if proc.poll() is not None:
            return
        for sig, wait in ((signal.SIGINT, 60), (signal.SIGTERM, 30), (signal.SIGKILL, 10)):
            try:
                os.killpg(proc.pid, sig)
            except ProcessLookupError:
                break
            try:
                proc.wait(timeout=wait if not DRY_RUN else 2)
                break
            except subprocess.TimeoutExpired:
                continue
        if not DRY_RUN:  # 清理残留的 Ray actor / SGLang 进程，避免下次启动显存被占
            subprocess.run(["ray", "stop", "--force"], capture_output=True, timeout=120)

    def _result(self, status, proc, log, last_step, plan, reason=None):
        text = Path(log).read_text(encoding="utf-8", errors="replace")
        res = {"status": status, "exit_code": proc.poll(), "last_step": last_step, "log": str(log),
               "last_saved_step": self.latest_saved(plan), "time": stamp()}
        if reason:
            res["reason"] = reason
        if status == "crashed":
            res["crash_class"] = classify_crash(text)
            res["log_tail"] = text[-2500:]
            if not DRY_RUN:
                subprocess.run(["ray", "stop", "--force"], capture_output=True, timeout=120)
        self.report("round_status", {k: v for k, v in res.items() if k != "log_tail"})
        return res

    # ------------------------------------------------------------------ 检查点
    @staticmethod
    def latest_saved(plan):
        f = Path(plan["save_dir"]) / "latest_checkpointed_iteration.txt"
        try:
            return int(f.read_text().strip())
        except (OSError, ValueError):
            return None

    def pin_checkpoint(self, plan, step):
        """把 global_step_{step}/actor 的模型分片硬链接到 pinned/（跳过优化器与 extra_state）。"""
        src = Path(plan["save_dir"]) / f"global_step_{step}" / "actor"
        if not src.exists():
            return None
        dst_root = Path(plan["round_dir"]) / "pinned"
        dst = dst_root / f"global_step_{step}" / "actor"
        for root, _, files in os.walk(src):
            rel = Path(root).relative_to(src)
            (dst / rel).mkdir(parents=True, exist_ok=True)
            for name in files:
                if name.startswith(("optim_", "extra_state_")):
                    continue
                target = dst / rel / name
                if not target.exists():
                    try:
                        os.link(Path(root) / name, target)
                    except OSError:
                        shutil.copy2(Path(root) / name, target)
        # 只保留当前最佳的钉住副本
        for old in dst_root.glob("global_step_*"):
            if old.name != f"global_step_{step}":
                shutil.rmtree(old, ignore_errors=True)
        return str(dst)

