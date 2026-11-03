"""多智能体 loop 的公共设施：路径、配置、原子读写、消息总线和共享状态（黑板）。

五个智能体之间不直接调用对方内部状态，而是通过两种方式协作：
  * Bus（events.jsonl）：带发送方/接收方/类型的消息，记录指令、报告和决策理由；
  * Blackboard（state.json）：当前阶段、计划、各智能体最近一次报告。
决策智能体读取报告并发出指令；其他智能体只执行职责范围内的工作并回报结果。
"""

import hashlib
import json
import os
from pathlib import Path
import sys
import time

PACKAGE_DIR = Path(__file__).resolve().parent
DECODER_DIR = PACKAGE_DIR.parent                  # decoder_from_scratch/
PROJECT_DIR = DECODER_DIR.parent                  # verl_megatron_sglang/
RUN_DIR = Path(os.environ.get('MA_RUN_DIR', DECODER_DIR / 'runs' / 'multiagent_pretrain_v1'))
TOKENIZER_PROJECT = Path(os.environ.get(
    'MA_TOKENIZER_PROJECT', str(PROJECT_DIR / 'tokenizer_training')))

DEFAULT_CONFIG = {
    'version': 1,
    'budget_hours': 72.0,                 # 从首次启动起算的总墙钟预算（含数据、tokenizer、训练、评估）
    'device': 'mps',
    'data': {
        'source_config': 'configs/scale_sources_v1.json',   # BAAI/IndustryCorpus（ModelScope，固定 revision）
        'total_chars': 1_000_000_000,     # 清洗后目标字符数（约 7 亿 token）
        'weights': {
            'zh_education': 0.22, 'zh_science': 0.13, 'zh_news': 0.10, 'zh_literature': 0.10,
            'zh_math': 0.10, 'programming': 0.10, 'en_education': 0.08, 'zh_travel': 0.05,
            'zh_agriculture': 0.04, 'zh_ai': 0.04, 'zh_sport': 0.02, 'en_programming': 0.02,
        },
        'download_workers': 4,
        'download_attempts': 3,
        'val_docs_per_domain': 300,
        'test_docs_per_domain': 300,
        'tokenizer_chars': 10_000_000,    # tokenizer 只用 train 划分的分层抽样
        'tokenizer_vocab': 32000,
        'tokenizer_min_frequency': 2,
        'tokenizer_timeout_hours': 4.0,
        'encode_workers': 0,              # 0 = cpu_count - 2
    },
    'train': {
        'seq_len': 1024,
        'tokens_per_step': 65536,
        'segment_minutes': 45,
        'checkpoint_minutes': 15,
        'eval_reserve_fraction': 0.06,
        'safety_hours': 1.0,
        'target_tokens_per_param': 20.0,
        'max_epochs_per_domain': 4.0,
        'weight_decay': 0.1,
        'beta2': 0.95,
        'grad_clip': 1.0,
        'min_lr_ratio': 0.1,
        'warmup_fraction': 0.01,
        'seed': 20261031,
    },
    'eval': {
        'val_chunks_per_domain': 16,
        'max_new_tokens': 120,
        'top_p': 0.9,
        'temperature': 0.8,
    },
}

PHASES = ['selftest', 'data_acquire', 'data_merge', 'tokenizer', 'encode',
          'model_select', 'train', 'final_eval', 'done']


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
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')
    os.replace(tmp, path)


def write_text(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(text, encoding='utf-8')
    os.replace(tmp, path)


def file_sha256(path, limit=None):
    digest = hashlib.sha256()
    with open(path, 'rb') as f:
        while True:
            block = f.read(1 << 20)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def deep_merge(base, override):
    out = dict(base)
    for key, value in (override or {}).items():
        # 领域权重整体替换（允许在 config.json 中删除领域），其他字典逐层合并
        if isinstance(value, dict) and isinstance(out.get(key), dict) and key != 'weights':
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def load_config():
    """运行目录中的 config.json 可覆盖默认值；首次运行时写出完整配置便于查看。"""
    path = RUN_DIR / 'config.json'
    saved = read_json(path, {})
    config = deep_merge(DEFAULT_CONFIG, saved)
    if not path.exists():
        write_json(path, config)
    return config


class Bus:
    """追加式消息总线。每条消息 = 发送方、接收方、类型、内容；同时打印到日志。"""

    def __init__(self, run_dir=None):
        self.path = Path(run_dir or RUN_DIR) / 'events.jsonl'
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def post(self, sender, recipient, kind, content):
        event = {'t': round(now(), 3), 'time': stamp(), 'from': sender, 'to': recipient,
                 'type': kind, 'content': content}
        with self.path.open('a', encoding='utf-8') as f:
            f.write(json.dumps(event, ensure_ascii=False) + '\n')
        summary = content if isinstance(content, str) else json.dumps(content, ensure_ascii=False)
        if len(summary) > 400:
            summary = summary[:400] + '…'
        print(f'[{event["time"]}] {sender} -> {recipient} ({kind}): {summary}', flush=True)
        return event

    def tail(self, n=30, kinds=None):
        if not self.path.exists():
            return []
        lines = self.path.read_text(encoding='utf-8').splitlines()
        events = [json.loads(line) for line in lines if line.strip()]
        if kinds:
            events = [e for e in events if e['type'] in kinds]
        return events[-n:]


class Blackboard:
    """共享状态：阶段、计划、各智能体最新报告。所有写入都原子落盘，可随时中断恢复。"""

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
    """用户放置 STOP 文件或到达截止时间：保存后正常退出。"""


class RestartRequested(Exception):
    """放置 RESTART 文件：保存后以退出码 75 退出，由 run_loop.sh 用最新代码重启。"""


def control_signal():
    if (RUN_DIR / 'STOP').exists():
        return 'stop'
    if (RUN_DIR / 'RESTART').exists():
        return 'restart'
    return None


def python_executable():
    return sys.executable
