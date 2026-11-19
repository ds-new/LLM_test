"""A800 多智能体预训练的公共设施：路径、配置、原子读写、消息总线、共享黑板、控制信号与心跳。

五个智能体（决策 / 数据清洗 / 结构与算法 / 训练执行与监控 / 测试评估）不直接改彼此的内部状态，
只通过两种方式协作，所有过程都落盘，方便学习和复盘：
  * Bus（events.jsonl）：带 发送方→接收方、类型（directive 指令 / report 报告 / decision 决策 / proposal 提案）的消息；
  * Blackboard（state.json）：当前阶段、训练计划、各智能体最近一次报告等共享状态。
"""

import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time

PACKAGE_DIR = Path(__file__).resolve().parent
DECODER_DIR = PACKAGE_DIR.parent                  # decoder_from_scratch/
PROJECT_DIR = DECODER_DIR.parent                  # 工程根目录（包含 decoder_from_scratch/）

# 数据盘放 token 和最终模型（关机后保留）；系统盘放临时下载和训练检查点（数据盘只有约 50GB）。
RUN_DIR = Path(os.environ.get('A8_RUN_DIR', '/root/autodl-tmp/a800_pretrain_v1'))
SCRATCH_DIR = Path(os.environ.get('A8_SCRATCH_DIR', '/root/a800_scratch'))
TOKENIZER_PROJECT = Path(os.environ.get('A8_TOKENIZER_PROJECT', str(PROJECT_DIR / 'tokenizer_training')))

DEFAULT_CONFIG = {
    'version': 1,
    'device': 'cuda',
    'data': {
        # 唯一 token 目标：受数据盘容量限制（uint16 每个 token 2 字节），实际值由决策智能体按剩余空间确定。
        'unique_tokens_target': 14_000_000_000,
        'disk_margin_gb': 4.0,
        'workers': 0,                      # 0 = CPU 核数 - 2
        'max_parallel_downloads': 4,       # parquet 需要先下载到系统盘再读取，限制并发控制临时空间
        'val_fraction': 0.0015,            # 按文档哈希划分，训练前就固定
        'test_fraction': 0.0015,
        'tokenizer_sample_chars': 24_000_000,
        'tokenizer_vocab': 32000,
        'tokenizer_min_frequency': 2,
        'tokenizer_timeout_hours': 3.0,
        'file_attempts': 3,
    },
    'train': {
        'target_tokens': 25_000_000_000,   # 用户选择：约 7 天、25B token
        'max_train_hours': 192.0,          # 训练阶段硬上限 8 天；吞吐不够时按时间缩减 token 数
        'seq_len': 1024,
        'tokens_per_step': 524_288,        # 全局 batch ≈ 0.5M token
        'peak_lr': 4e-4,
        'min_lr_ratio': 0.1,
        'warmup_steps': 1000,
        'decay_fraction': 0.2,             # WSD：最后 20% 的步数线性衰减到 min_lr
        'weight_decay': 0.1,
        'beta2': 0.95,
        'grad_clip': 1.0,
        'segment_minutes': 60,             # 每段训练后把控制权交回决策智能体（评估 + 决策）
        'checkpoint_minutes': 30,
        'seed': 20261102,
        'compile': True,
    },
    'eval': {
        'val_chunks_per_source': 32,
        'max_new_tokens': 160,
        'top_p': 0.9,
        'temperature': 0.8,
    },
    # 训练结束后的参数空间编辑实验（weight_edit.py，复现 Weight-Space Programming 实验 8），完成后再关机
    'weight_edit': {'enabled': True, 'practice_prompts': 384, 'practice_chunks': 48, 'heldout_prompts': 256,
                    'prompt_len': 48, 'gen_tokens': 128, 'budget': 50, 'rounds': 3, 'max_layers': 3,
                    'columns': [32, 64, 128], 'alphas': [1.0, 0.5], 'loss_tolerance': 0.005, 'min_gain': 0.05},
    # pass@k 能力探测（passk_probe.py），判断哪些可验证任务适合直接做 RLVR
    'passk': {'enabled': True, 'n': 16, 'ks': [1, 4, 8, 16], 'problems': 100, 'gsm8k_problems': 200, 'gsm8k_shots': 3,
              'temperature': 0.8, 'top_p': 0.95, 'batch': 128},
    # token 频次 / 欠训练 token / GSM8K 13-gram 污染（token_freq_probe.py）
    'token_probe': {'enabled': True, 'ngram': 13, 'repeat_samples': 100, 'suspicious_top': 50, 'batch': 256},
    # 最后一步：torch.profiler 抓一个真实训练步 + MFU（profile_probe.py），会做几次优化器更新，所以排在所有评估之后
    'profile': {'enabled': True, 'warmup_steps': 2, 'timed_steps': 5, 'eager_steps': 3, 'peak_tflops': 312.0},
    # 结果分析智能体：趋势窗口（评估次数）、显著性阈值（t 值）、"真的变差"至少要超过几倍单次评估噪声
    'analysis': {'enabled': True, 'window': 8, 't_threshold': 3.0, 'sigma_mult': 3.0},
    'shutdown_on_exit': True,              # 训练完成或异常终止后自动关机（AutoDL 关机即停止计费）
}

PHASES = ['selftest', 'benchmark', 'sample', 'tokenizer', 'fast_tokenizer', 'build_data',
          'plan', 'train', 'final_eval', 'weight_edit', 'passk_probe', 'token_probe', 'profile', 'analysis', 'done']


def now():
    return time.time()


def stamp(ts=None):
    return time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(ts or time.time()))


def read_json(path, default=None):
    path = Path(path)
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding='utf-8'))


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f'.tmp{os.getpid()}')
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')
    os.replace(tmp, path)


def write_text(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f'.tmp{os.getpid()}')
    tmp.write_text(text, encoding='utf-8')
    os.replace(tmp, path)


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def free_gb(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return shutil.disk_usage(path).free / 1e9


def deep_merge(base, override):
    out = dict(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def load_config():
    """运行目录中的 config.json 可覆盖默认值；首次运行时写出完整配置便于查看和修改。"""
    path = RUN_DIR / 'config.json'
    config = deep_merge(DEFAULT_CONFIG, read_json(path, {}))
    if not path.exists():
        write_json(path, config)
    return config


class Bus:
    """追加式消息总线。每条消息 = 时间、发送方、接收方、类型、内容；同时打印到日志。"""

    def __init__(self, run_dir=None):
        self.path = Path(run_dir or RUN_DIR) / 'events.jsonl'
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def post(self, sender, recipient, kind, content):
        event = {'t': round(now(), 3), 'time': stamp(), 'from': sender, 'to': recipient,
                 'type': kind, 'content': content}
        with self.path.open('a', encoding='utf-8') as f:
            f.write(json.dumps(event, ensure_ascii=False) + '\n')
        summary = content if isinstance(content, str) else json.dumps(content, ensure_ascii=False)
        print(f'[{event["time"]}] {sender} -> {recipient} ({kind}): {summary[:500]}', flush=True)
        return event

    def tail(self, n=30, kinds=None):
        if not self.path.exists():
            return []
        events = []
        for line in self.path.read_text(encoding='utf-8').splitlines():
            if line.strip():
                event = json.loads(line)
                if not kinds or event['type'] in kinds:
                    events.append(event)
        return events[-n:]


class Blackboard:
    """共享状态：所有写入都原子落盘，进程随时被杀也能从断点恢复。"""

    def __init__(self, run_dir=None):
        self.path = Path(run_dir or RUN_DIR) / 'state.json'
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
        self.data['updated'] = stamp()
        write_json(self.path, self.data)


class StopRequested(Exception):
    """用户放置 STOP 文件：保存后正常退出（不关机）。"""


class FatalError(Exception):
    """无法自动恢复的错误（检查点丢失、多次回滚仍发散等）：loop 以退出码 2 退出，守护脚本随后关机。"""


class RestartRequested(Exception):
    """放置 RESTART 文件：保存后以退出码 75 退出，由 run_loop.sh 用最新代码重启。"""


def control_signal():
    if (RUN_DIR / 'STOP').exists():
        return 'stop'
    if (RUN_DIR / 'RESTART').exists():
        return 'restart'
    return None


def heartbeat(note=''):
    """守护脚本据此判断 loop 是否卡死（长时间无心跳会被杀掉并重启）。"""
    try:
        (RUN_DIR / 'heartbeat').write_text(f'{now():.0f} {note}\n')
    except OSError:
        pass


def python_executable():
    return sys.executable

def cpu_quota():
    """容器实际可用的 CPU 核数：os.cpu_count() 在 AutoDL 上返回宿主机核数（如 112），要读 cgroup 配额。"""
    for quota_file, period_file in (('/sys/fs/cgroup/cpu.max', None),
                                    ('/sys/fs/cgroup/cpu/cpu.cfs_quota_us', '/sys/fs/cgroup/cpu/cpu.cfs_period_us')):
        try:
            if period_file is None:
                quota, period = open(quota_file).read().split()[:2]
            else:
                quota, period = open(quota_file).read().strip(), open(period_file).read().strip()
            if quota not in ('max', '-1') and int(quota) > 0:
                return max(1, int(int(quota) / int(period)))
        except (OSError, ValueError):
            continue
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return os.cpu_count() or 4
