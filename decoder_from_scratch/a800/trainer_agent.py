"""训练配置、执行与监控智能体（CUDA / A800）。

职责：把决策智能体批准的计划落实为训练——
  * 精度与速度：bf16 autocast、TF32、torch.compile、fused AdamW、FlashAttention（PyTorch SDPA 自动选择）；
  * 优化：AdamW(β2=0.95, wd=0.1, 只对矩阵权重衰减)、梯度裁剪 1.0、梯度累积到约 0.5M token/步；
  * 学习率：WSD（warmup → 恒定 → 最后 20% 线性衰减），改总步数时只移动衰减起点，不会让学习率跳变；
  * 数据：按来源权重抽样，每个来源内部"打乱的块顺序"逐轮遍历，状态可精确恢复；
  * 监控：每 20 步记录 loss / 梯度范数 / 吞吐 / 显存，非有限值跳过并在连续出现时请求回滚；
  * 容错：每 30 分钟和每段结束保存完整检查点（模型 + 优化器 + 采样器 + 随机数状态），保留上一个用于回滚。
"""

import json
import math
import os
from pathlib import Path
import time
import zlib

import numpy as np
import torch

from ..config import ModelConfig
from ..model import DecoderLM
from .common import RUN_DIR, SCRATCH_DIR, control_signal, heartbeat, now, stamp

NAME = 'trainer'


def autocast(device, precision='bf16'):
    if precision == 'bf16' and device.type in ('cuda', 'cpu'):
        return torch.autocast(device_type=device.type, dtype=torch.bfloat16)
    import contextlib
    return contextlib.nullcontext()


def wsd_lr(step, total, peak, warmup, decay_fraction, min_ratio):
    """Warmup-Stable-Decay。decay_start 之前学习率恒定，所以中途调整 total 不会改变当前学习率。"""
    if step < warmup:
        return peak * (step + 1) / warmup
    decay_start = int(total * (1 - decay_fraction))
    if step < decay_start:
        return peak
    progress = min(1.0, (step - decay_start) / max(1, total - decay_start))
    return peak * (1 - (1 - min_ratio) * progress)


class MixtureSampler:
    """多来源、多分片的训练块采样器。

    每个来源把所有分片拼成一条虚拟 token 流，切成长度 seq_len+1 的块；每轮（epoch）用固定种子打乱块顺序，
    逐个取用，取完进入下一轮。来源之间按权重随机选择。状态 = 每个来源的 (epoch, cursor) + 随机数状态。
    """

    def __init__(self, manifest, weights, seq_len, seed):
        self.seq_len, self.seed = seq_len, seed
        self.sources = {}
        for name, entry in manifest['sources'].items():
            arrays, counts = [], []
            for shard in entry['train']:
                arr = np.memmap(RUN_DIR / shard['path'], dtype='<u2', mode='r')
                n = (len(arr) - 1) // seq_len
                if n > 0:
                    arrays.append(arr)
                    counts.append(n)
            if counts:
                self.sources[name] = {'arrays': arrays, 'offsets': np.cumsum([0] + counts), 'n': int(sum(counts))}
        self.state = {name: {'epoch': 0, 'cursor': 0} for name in self.sources}
        self.rng = np.random.default_rng(seed)
        self._perm = {}
        self.set_weights(weights)

    def set_weights(self, weights):
        names = [n for n in self.sources if weights.get(n, 0) > 0]
        w = np.array([weights[n] for n in names], dtype=np.float64)
        self.names, self.probs = names, w / w.sum()
        self.weights = {n: float(p) for n, p in zip(names, self.probs)}

    def _permutation(self, name):
        st = self.state[name]
        key = (name, st['epoch'])
        if key not in self._perm:
            self._perm = {k: v for k, v in self._perm.items() if k[0] != name}
            seed = self.seed + zlib.crc32(name.encode()) * 1009 + st['epoch']
            self._perm[key] = np.random.default_rng(seed).permutation(self.sources[name]['n'])
        return self._perm[key]

    def _chunk(self, name):
        src, st = self.sources[name], self.state[name]
        if st['cursor'] >= src['n']:
            st['epoch'] += 1
            st['cursor'] = 0
        index = int(self._permutation(name)[st['cursor']])
        st['cursor'] += 1
        shard = int(np.searchsorted(src['offsets'], index, side='right') - 1)
        start = (index - int(src['offsets'][shard])) * self.seq_len
        return src['arrays'][shard][start:start + self.seq_len + 1]

    def batch(self, batch_size):
        picks = self.rng.choice(len(self.names), size=batch_size, p=self.probs)
        return torch.from_numpy(np.stack([self._chunk(self.names[i]).astype(np.int64) for i in picks]))

    def epochs(self):
        return {n: round(st['epoch'] + st['cursor'] / self.sources[n]['n'], 4) for n, st in self.state.items()}

    def state_dict(self):
        return {'state': json.loads(json.dumps(self.state)), 'rng': self.rng.bit_generator.state}

    def load_state_dict(self, data):
        for name, st in data['state'].items():
            if name in self.state:
                self.state[name] = dict(st)
        self.rng.bit_generator.state = data['rng']


class TrainerAgent:
    def __init__(self, config, bus, manifest, plan, device, tokenizer_dict, ckpt_dir=None, log_path=None):
        self.config, self.bus, self.manifest, self.plan = config, bus, manifest, plan
        self.tcfg = config['train']
        self.device = torch.device(device)
        self.tokenizer_dict = tokenizer_dict
        self.ckpt_dir = Path(ckpt_dir or SCRATCH_DIR / 'checkpoints')
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = Path(log_path or RUN_DIR / 'train_log.jsonl')
        self.step = 0
        self.tokens_seen = 0
        self.train_seconds = 0.0
        self.lr_scale = 1.0
        self.last_checkpoint = now()
        if self.device.type == 'cuda':
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

    def say(self, recipient, kind, content):
        if self.bus is not None:
            self.bus.post(NAME, recipient, kind, content)

    # ---------------- 构建 / 恢复 ----------------
    def model_config(self):
        return ModelConfig(vocab_size=self.manifest['tokenizer']['vocab_size'], max_seq_len=max(2048, self.plan['seq_len']),
                           dropout=0.0, tie_embeddings=True, attention_backend='sdpa', **self.plan['model'])

    def build(self):
        torch.manual_seed(self.tcfg['seed'])
        self.model = DecoderLM(self.model_config()).to(self.device)
        decay = [p for p in self.model.parameters() if p.dim() >= 2]
        no_decay = [p for p in self.model.parameters() if p.dim() < 2]
        fused = self.device.type == 'cuda'
        self.optimizer = torch.optim.AdamW(
            [{'params': decay, 'weight_decay': self.tcfg['weight_decay']}, {'params': no_decay, 'weight_decay': 0.0}],
            lr=self.plan['peak_lr'], betas=(0.9, self.tcfg['beta2']), eps=1e-8, fused=fused)
        self.train_model = torch.compile(self.model) if self.plan.get('compile') else self.model
        self.sampler = MixtureSampler(self.manifest, self.plan['weights'], self.plan['seq_len'], self.tcfg['seed'])
        latest = self.ckpt_dir / 'latest.pt'
        if latest.exists():
            self.load(latest)
            self.say('decision', 'report', {'action': 'resumed', 'step': self.step, 'tokens_seen': self.tokens_seen})
        else:
            self.say('decision', 'report', {'action': 'initialized', 'params': self.model.count_parameters(),
                                            'compile': bool(self.plan.get('compile'))})

    def load(self, path):
        ckpt = torch.load(path, map_location='cpu', weights_only=False)
        self.model.load_state_dict(ckpt['model'])
        self.optimizer.load_state_dict(ckpt['optimizer'])
        self.step, self.tokens_seen = ckpt['step'], ckpt['tokens_seen']
        self.train_seconds = ckpt.get('train_seconds', 0.0)
        self.lr_scale = ckpt.get('lr_scale', 1.0)
        self.sampler.set_weights(ckpt.get('weights', self.plan['weights']))
        self.sampler.load_state_dict(ckpt['sampler'])
        torch.set_rng_state(ckpt['torch_rng'])

    def save_latest(self):
        payload = {'model': self.model.state_dict(), 'optimizer': self.optimizer.state_dict(), 'step': self.step,
                   'global_step': self.step, 'tokens_seen': self.tokens_seen, 'train_seconds': self.train_seconds,
                   'lr_scale': self.lr_scale, 'sampler': self.sampler.state_dict(), 'weights': dict(self.sampler.weights),
                   'torch_rng': torch.get_rng_state(), 'config': self.model.config.to_dict(),
                   'tokenizer': self.tokenizer_dict, 'plan': self.plan}
        tmp = self.ckpt_dir / 'latest.pt.tmp'
        (self.ckpt_dir / 'prev.pt').unlink(missing_ok=True)  # 先删上上个：写入期间最多占两份（每份约 5.8GB），否则峰值是三份
        torch.save(payload, tmp)
        if (self.ckpt_dir / 'latest.pt').exists():
            os.replace(self.ckpt_dir / 'latest.pt', self.ckpt_dir / 'prev.pt')
        os.replace(tmp, self.ckpt_dir / 'latest.pt')
        self.last_checkpoint = now()

    def save_weights(self, path, **extra):
        """只存权重 + 配置 + 分词器，格式与 decoder_from_scratch/infer.py 兼容。"""
        payload = {'model': self.model.state_dict(), 'config': self.model.config.to_dict(), 'tokenizer': self.tokenizer_dict,
                   'step': self.step, 'global_step': self.step, 'tokens_seen': self.tokens_seen,
                   'normalize_punct': True, **extra}  # 训练数据做过中文标点全角化，推理时提示词也要同样处理
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + '.tmp')
        torch.save(payload, tmp)
        os.replace(tmp, path)

    def rollback(self, lr_factor):
        source = self.ckpt_dir / 'prev.pt'
        if not source.exists():
            source = self.ckpt_dir / 'latest.pt'
        self.load(source)
        self.lr_scale *= lr_factor
        self.say('decision', 'report', {'action': 'rolled_back', 'to_step': self.step, 'lr_scale': self.lr_scale})

    def lr_at(self, step):
        p = self.plan
        return wsd_lr(step, p['total_steps'], p['peak_lr'] * self.lr_scale, p['warmup_steps'],
                      p['decay_fraction'], self.tcfg['min_lr_ratio'])

    # ---------------- 分段训练 ----------------
    def run_segment(self, seconds, hard_deadline):
        plan = self.plan
        micro, accum, T = plan['micro_batch'], plan['accum'], plan['seq_len']
        self.model.train()
        seg_start, step0, tokens0 = now(), self.step, self.tokens_seen
        losses, norms, nonfinite, bad_streak = [], [], 0, 0
        stop_reason = 'segment_time'
        if self.device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats()
        log = self.log_path.open('a', encoding='utf-8')
        prefetch = self.sampler.batch(micro).pin_memory() if self.device.type == 'cuda' else self.sampler.batch(micro)
        try:
            while True:
                if self.step >= plan['total_steps']:
                    stop_reason = 'plan_complete'
                    break
                if now() - seg_start >= seconds:
                    break
                if now() >= hard_deadline:
                    stop_reason = 'deadline'
                    break
                signal = control_signal()
                if signal:
                    stop_reason = signal
                    break
                t0 = time.perf_counter()
                lr = self.lr_at(self.step)
                for group in self.optimizer.param_groups:
                    group['lr'] = lr
                total_loss = torch.zeros((), device=self.device)
                for _ in range(accum):
                    batch = prefetch.to(self.device, non_blocking=True)
                    nxt = self.sampler.batch(micro)  # CPU 准备下一批，与 GPU 计算重叠
                    prefetch = nxt.pin_memory() if self.device.type == 'cuda' else nxt
                    with autocast(self.device, plan['precision']):
                        loss = self.train_model(batch[:, :-1], targets=batch[:, 1:]).loss / accum
                    loss.backward()
                    total_loss += loss.detach()
                grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.tcfg['grad_clip'])
                loss_value, norm_value = float(total_loss), float(grad_norm)
                if not (math.isfinite(loss_value) and math.isfinite(norm_value)):
                    self.optimizer.zero_grad(set_to_none=True)
                    nonfinite += 1
                    bad_streak += 1
                    if bad_streak >= 3:
                        stop_reason = 'nonfinite'
                        break
                    continue
                bad_streak = 0
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)
                self.step += 1
                self.tokens_seen += micro * T * accum
                dt = time.perf_counter() - t0
                self.train_seconds += dt
                losses.append(loss_value)
                norms.append(norm_value)
                if self.step % 20 == 0:
                    rec = {'time': stamp(), 'step': self.step, 'tokens': self.tokens_seen, 'loss': round(loss_value, 4),
                           'lr': lr, 'grad_norm': round(norm_value, 3), 'step_s': round(dt, 3),
                           'tok_s': round(micro * T * accum / dt)}
                    if self.device.type == 'cuda':
                        rec['mem_gb'] = round(torch.cuda.max_memory_allocated() / 1e9, 1)
                    log.write(json.dumps(rec) + '\n')
                    log.flush()
                    heartbeat(f'train step {self.step}')
                if now() - self.last_checkpoint >= self.tcfg['checkpoint_minutes'] * 60:
                    self.save_latest()
        finally:
            log.close()
        if self.step > step0:
            self.save_latest()
        elapsed = max(1e-6, now() - seg_start)
        window = losses[-100:]
        report = {'action': 'segment_done', 'stop_reason': stop_reason, 'step': self.step, 'total_steps': plan['total_steps'],
                  'tokens_seen': self.tokens_seen, 'segment_steps': self.step - step0,
                  'segment_tok_s': round((self.tokens_seen - tokens0) / elapsed),
                  'train_loss': round(sum(window) / len(window), 4) if window else None,
                  'grad_norm_median': round(float(np.median(norms)), 3) if norms else None,
                  'grad_norm_max': round(max(norms), 3) if norms else None, 'nonfinite_steps': nonfinite,
                  'lr': self.lr_at(self.step), 'lr_scale': self.lr_scale, 'epochs': self.sampler.epochs(),
                  'weights': {n: round(w, 4) for n, w in self.sampler.weights.items()}}
        if self.device.type == 'cuda':
            report['peak_mem_gb'] = round(torch.cuda.max_memory_allocated() / 1e9, 1)
        self.say('decision', 'report', report)
        return report

    def apply_weights(self, weights):
        self.sampler.set_weights(weights)
        self.plan['weights'] = dict(weights)
