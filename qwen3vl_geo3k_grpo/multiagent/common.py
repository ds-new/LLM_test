"""多智能体 loop 的公共设施：路径、配置、原子读写、消息总线（Bus）、共享状态（Blackboard）。

与 decoder_from_scratch/multiagent 保持同一套协作约定：
  * Bus（events.jsonl）：发送方 / 接收方 / 类型 / 内容，记录指令、报告、决策及理由；
  * Blackboard（state.json）：当前阶段、各轮计划与结果、各智能体最近报告；所有写入原子落盘，
    任意时刻被打断（AutoDL 关机、断网）都可以从断点恢复。
决策智能体读取报告并下达指令；其余四个智能体只做职责内的工作并回报结果。

干跑：MA_DRY_RUN=1 时 GPU、verl、SGLang、模型合并全部替换为模拟实现（fake_verl.py、evaluate 的
fake 后端），用合成数据在几分钟内走完整个流程，可在 Mac 上先验证逻辑再租 GPU。
"""

import json
import os
import shutil
import sys
import time
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = PACKAGE_DIR.parent  # qwen3vl_4b_geo3k_grpo/
DRY_RUN = os.environ.get("MA_DRY_RUN", "0") == "1"


def _is_dir(p):
    """Python 3.10 中对无权限路径 is_dir() 会抛 PermissionError（如 Mac 上的 /root），视为不存在。"""
    try:
        return Path(p).is_dir()
    except OSError:
        return False


def _default_root():
    if DRY_RUN:
        return PROJECT_DIR / "runs" / "dryrun"
    # AutoDL：系统盘只有 30G，大文件全部放数据盘 /root/autodl-tmp
    if _is_dir("/root/autodl-tmp"):
        return Path("/root/autodl-tmp/geo3k_runs/multiagent_v1")
    return PROJECT_DIR / "runs" / "multiagent_v1"


RUN_DIR = Path(os.environ.get("MA_RUN_DIR", str(_default_root())))


def _is_qwen3vl(p):
    try:
        cfg = json.loads((Path(p) / "config.json").read_text(encoding="utf-8"))
        return "qwen3_vl" in str(cfg.get("model_type", "")) and any(Path(p).glob("*.safetensors"))
    except (OSError, ValueError):
        return False


def find_local_model():
    """MODEL_PATH → 常见目录名 → /root/autodl-tmp 下一层中任何 Qwen3-VL 权重目录。"""
    cands = [os.environ.get("MODEL_PATH"), "/root/autodl-tmp/qwen3_vl_4b", "/root/autodl-tmp/qwen3_vl_4",
             "/root/autodl-tmp/models/Qwen3-VL-4B-Instruct", "/root/autodl-tmp/Qwen3-VL-4B-Instruct"]
    for p in cands:
        if p and _is_qwen3vl(p):
            return p
    for base in ("/root/autodl-tmp", "/root/autodl-tmp/models"):
        if _is_dir(base):
            try:
                for d in sorted(Path(base).iterdir()):
                    if "4b" in d.name.lower() and _is_qwen3vl(d):
                        return str(d)
            except OSError:
                pass
    return None


def _default_model():
    return find_local_model() or os.environ.get("MODEL_PATH") or "/root/autodl-tmp/models/Qwen3-VL-4B-Instruct"


DEFAULT_CONFIG = {
    "version": 1,
    "budget_hours": 24.0,               # 从首次启动算起的总墙钟预算（AutoDL 按小时计费）
    "final_reserve_hours": 1.0,         # 预留给最终 test 评估与报告
    "shutdown_when_done": False,        # True：完成后执行 `shutdown`（AutoDL 关机停止计费）
    "model": {
        "path": _default_model(),
        "modelscope_id": "Qwen/Qwen3-VL-4B-Instruct",
        "hf_id": "Qwen/Qwen3-VL-4B-Instruct",
    },
    "data": {
        "source": os.environ.get("GEO3K_RAW", "hiyouga/geometry3k"),  # Hub 名称或本地目录
        "max_image_side": 1024,         # 超大图缩放，控制视觉 token 数
        "min_image_side": 28,           # Qwen-VL patch 14 × merge 2
        "drop_opaque_answers": True,    # 无数字且无法求值的答案（如选项字母）无法可靠判分
        "difficulty_filter": True,      # 基线模型每题采样 k 次，去掉全对的题（GRPO 组内优势为 0）
        "difficulty_samples": 8,
        "keep_easy_fraction": 0.1,      # 全对题保留 10% 作为防遗忘锚点
        "min_train_items": 1000,        # 过滤后少于该数则放弃难度过滤
    },
    "train": {
        "gpus": "auto",                 # auto = nvidia-smi 可见的全部 GPU
        "attn_implementation": "auto",  # auto：flash-attn 可用则 flash_attention_2，否则 sdpa
        "rollout": "auto",              # auto：装了 sglang 用 sglang，否则 vllm；也可写死 "sglang" / "vllm"
        "train_batch_size": 64,
        "n_samples": 8,                 # 原脚本为 5；GRPO 组越大优势估计方差越小
        "ppo_mini_batch_size": 32,
        "lr": 1e-6,
        "max_prompt_length": 1024,      # 数据智能体按实测 token 长度更新
        "max_response_length": 2048,    # 起始值；基线截断比例 > 10% 时决策智能体翻倍（上限见下）
        "max_response_length_cap": 4096,
        "total_epochs": 12,
        "save_freq": 8,                 # 同时作为 test_freq：每个保存点都有验证分数
        "patience": 3,                  # 连续 3 次验证无提升（> min_delta）提前结束本轮
        "min_delta": 0.5,               # 百分点
        "min_steps": 48,
        "diverge_drop": 8.0,            # 验证准确率比最佳低 8 个百分点视为退化
        "round_hours_max": 9.0,
        "stall_minutes": 100,           # 训练日志无新步数超过该时长视为卡死（4096 长度下单步可达 30～40 分钟）
        "first_step_minutes": 120,      # 首步包含模型加载、vLLM 启动、训练前验证
        "poll_seconds": 30,
        "max_crash_retries": 5,
        "params_b": 4.4,                # 参数量（十亿），显存规划用；架构智能体按 safetensors 实测更新
    },
    "algorithm": {
        "presets_order": ["grpo", "dapo_lite", "drgrpo"],
        "max_rounds": 2,
    },
    "eval": {
        "backend": "auto",              # auto/sglang/hf
        "max_new_tokens": 2048,
    },
    # 旁路监控（multiagent/wandb_sync.py，由 run_loop.sh 自动拉起，不影响训练进程）
    "monitor": {
        "wandb": "auto",                # auto：已 `wandb login`（或设置 WANDB_API_KEY）就推送；off：不推送
        "project": "qwen3vl_4b_geo3k_multiagent",
        "entity": None,                 # wandb 团队名；None 用账号默认
        "tensorboard": True,            # 同时写 TensorBoard（AutoDL 的 AutoPanel 直接可看）
        "tensorboard_dir": "/root/tf-logs/geo3k_multiagent",
        "poll_seconds": 60,
    },
}

PHASES = ["env_check", "data_acquire", "data_clean", "baseline_eval", "difficulty_filter",
          "plan_round", "train_round", "review_round", "final_eval", "done"]


def now():
    return time.time()


def stamp(ts=None):
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ts or time.time()))


def read_json(path, default=None):
    path = Path(path)
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return default


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def write_text(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def deep_merge(base, override):
    out = dict(base)
    for k, v in (override or {}).items():
        out[k] = deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def load_config():
    """RUN_DIR/config.json 覆盖默认值；首次运行写出完整配置，修改后 restart 生效。"""
    path = RUN_DIR / "config.json"
    cfg = deep_merge(DEFAULT_CONFIG, read_json(path, {}))
    if DRY_RUN:
        cfg = deep_merge(cfg, {"budget_hours": 0.5, "final_reserve_hours": 0.02,
                               "model": {"path": str(RUN_DIR / "fake_model")},
                               "data": {"source": str(RUN_DIR / "raw_synthetic"), "min_train_items": 20},
                               "train": {"train_batch_size": 8, "ppo_mini_batch_size": 4, "total_epochs": 6,
                                         "save_freq": 2, "min_steps": 6, "patience": 2, "poll_seconds": 0.2,
                                         "stall_minutes": 1, "first_step_minutes": 1, "round_hours_max": 0.1},
                               "eval": {"backend": "fake"}})
    if not DRY_RUN:
        # 环境变量 MODEL_PATH 优先；保存的路径不是有效权重目录时自动改用本地找到的权重
        env_model = os.environ.get("MODEL_PATH")
        if env_model and env_model != cfg["model"]["path"]:
            cfg["model"]["path"] = env_model
        elif not _is_qwen3vl(cfg["model"]["path"]):
            found = find_local_model()
            if found:
                cfg["model"]["path"] = found
        if read_json(path, {}).get("model", {}).get("path") != cfg["model"]["path"]:
            saved = read_json(path, None)
            if saved is not None:
                saved.setdefault("model", {})["path"] = cfg["model"]["path"]
                write_json(path, saved)
    if not path.exists():
        write_json(path, cfg)
    return cfg


def disk_free_gb(path):
    if DRY_RUN:  # 干跑不占真实空间，按 AutoDL 扩容后的数据盘模拟
        return 500.0
    p = Path(path)
    while not p.exists():
        p = p.parent
    return shutil.disk_usage(p).free / 1024 ** 3


def dir_size_gb(path):
    total = 0
    for root, _, files in os.walk(path):
        for f in files:
            try:
                total += os.lstat(os.path.join(root, f)).st_size
            except OSError:
                pass
    return total / 1024 ** 3


class Bus:
    """追加式消息总线：每条 = 时间、发送方、接收方、类型、内容；同时打印到日志。"""

    def __init__(self, run_dir=None):
        self.path = Path(run_dir or RUN_DIR) / "events.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def post(self, sender, recipient, kind, content):
        event = {"t": round(now(), 3), "time": stamp(), "from": sender, "to": recipient, "type": kind, "content": content}
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")
        text = content if isinstance(content, str) else json.dumps(content, ensure_ascii=False)
        print(f"[{event['time']}] {sender} -> {recipient} ({kind}): {text[:400]}{'…' if len(text) > 400 else ''}", flush=True)
        return event

    def tail(self, n=30, kinds=None):
        if not self.path.exists():
            return []
        events = [json.loads(l) for l in self.path.read_text(encoding="utf-8").splitlines() if l.strip()]
        if kinds:
            events = [e for e in events if e["type"] in kinds]
        return events[-n:]


class Blackboard:
    def __init__(self, run_dir=None):
        self.path = Path(run_dir or RUN_DIR) / "state.json"
        self.data = read_json(self.path, {}) or {}

    def get(self, key, default=None):
        return self.data.get(key, default)

    def set(self, key, value):
        self.data[key] = value
        self.save()

    def update(self, **items):
        self.data.update(items)
        self.save()

    def save(self):
        self.data["updated"] = stamp()
        write_json(self.path, self.data)


class StopRequested(Exception):
    """STOP 文件或预算耗尽：保存后正常退出。"""


class RestartRequested(Exception):
    """RESTART 文件：退出码 75，由 run_loop.sh 用最新代码重启。"""


def control_signal():
    if (RUN_DIR / "STOP").exists():
        return "stop"
    if (RUN_DIR / "RESTART").exists():
        return "restart"
    return None


def python_executable():
    return os.environ.get("MA_PYTHON", sys.executable)


def project_env(extra=None):
    """子进程环境：项目目录进 PYTHONPATH，HF 走镜像（AutoDL 访问 huggingface.co 不稳定）。"""
    env = dict(os.environ)
    env["PYTHONPATH"] = str(PROJECT_DIR) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    env.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    env.setdefault("PYTHONUNBUFFERED", "1")
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    # vLLM 引擎子进程用 spawn：父进程一旦初始化过 CUDA，fork 出的子进程会报 "CUDA driver initialization failed"
    env.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    env.setdefault("GEO3K_PATCH_CONV3D", "1")  # 见 geo3k_patches.py：绕开 torch 2.9 bf16 Conv3d 性能回退
    env.update({k: str(v) for k, v in (extra or {}).items()})
    return env
