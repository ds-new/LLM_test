"""模型训练配置、执行与监控智能体。

职责：把决策智能体批准的计划落实为训练（学习率计划、梯度累积、精度、检查点），
分段执行（每段默认 45 分钟后把控制权交回决策智能体），实时监控 loss/梯度/吞吐/非有限值，
并负责断点恢复与回滚。
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
from .architect_agent import autocast_ctx, sync
from .common import RUN_DIR, control_signal, now, read_json, stamp, write_json

NAME = 'trainer'


class MixtureSampler:
    """按领域权重抽样；每个领域内部按"不放回的随机块顺序"遍历（逐 epoch 打乱），可精确恢复。"""

    def __init__(self, manifest, weights, seq_len, seed, state=None):
        self.seq_len, self.seed = seq_len, seed
        self.shards = {}
        for domain, entry in manifest['domains'].items():
            arrays = []
            for shard in entry['train']:
                arr = np.memmap(RUN_DIR / shard['path'], dtype='<u2', mode='r')
                n = max(0, (len(arr) - 1) // seq_len)
                if n:
                    arrays.append((arr, n))
            if arrays:
                self.shards[domain] = arrays
        self.state = {d: {'epoch': 0, 'cursor': 0, 'n': self.total_chunks(d), 'drawn': 0} for d in self.shards}
        self.rng = np.random.default_rng(seed)
        self._perm_cache = {}
        self.set_weights(weights)
        if state:
            self.load_state_dict(state)

    def total_chunks(self, domain):
        return sum(n for _, n in self.shards[domain])

    def set_weights(self, weights):
        domains = [d for d in self.shards if weights.get(d, 0) > 0]
        w = np.array([weights[d] for d in domains], dtype=np.float64)
        self.domains, self.probs = domains, w / w.sum()
        self.weights = {d: float(p) for d, p in zip(domains, self.probs)}

    def add_shard(self, domain, shard):
        arr = np.memmap(RUN_DIR / shard['path'], dtype='<u2', mode='r')
        n = (len(arr) - 1) // self.seq_len
        if n > 0:
            self.shards.setdefault(domain, []).append((arr, n))
            self.state.setdefault(domain, {'epoch': 0, 'cursor': 0, 'n': 0, 'drawn': 0})

    def _perm(self, domain):
        st = self.state[domain]
        key = (domain, st['epoch'], st['n'])
        if key not in self._perm_cache:
            if len(self._perm_cache) > 64:
                self._perm_cache.clear()
            seed = self.seed + zlib.crc32(domain.encode()) * 1000 + st['epoch']
            self._perm_cache[key] = np.random.default_rng(seed).permutation(st['n'])
        return self._perm_cache[key]

    def _chunk(self, domain):
        st = self.state[domain]
        if st['n'] == 0 or st['cursor'] >= st['n']:
            if st['n']:
                st['epoch'] += 1
            st['cursor'], st['n'] = 0, self.total_chunks(domain)  # 新 epoch 才纳入补充分片
        index = int(self._perm(domain)[st['cursor']])
        st['cursor'] += 1
        st['drawn'] += 1
        for arr, n in self.shards[domain]:
            if index < n:
                start = index * self.seq_len
                return arr[start:start + self.seq_len + 1]
            index -= n
        raise IndexError('chunk index out of range')

    def batch(self, batch_size):
        picks = self.rng.choice(len(self.domains), size=batch_size, p=self.probs)
        rows = np.stack([self._chunk(self.domains[i]).astype(np.int64) for i in picks])
        return torch.from_numpy(rows)

    def epochs(self):
        return {d: round(st['epoch'] + st['cursor'] / max(1, st['n']), 3) for d, st in self.state.items()}

    def state_dict(self):
        return {'state': json.loads(json.dumps(self.state)), 'rng': self.rng.bit_generator.state}

    def load_state_dict(self, data):
        for domain, st in data['state'].items():
            if domain in self.state:
                self.state[domain] = dict(st)
        self.rng.bit_generator.state = data['rng']


class TrainerAgent:
    def __init__(self, config, bus, manifest, plan, device, tokenizer_dict):
        self.config, self.bus, self.manifest, self.plan = config, bus, manifest, plan
        self.tcfg = config['train']
        self.device = torch.device(device)
        self.tokenizer_dict = tokenizer_dict
        self.ckpt_dir = RUN_DIR / 'checkpoints'
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = RUN_DIR / 'train_log.jsonl'
        self.model = self.optimizer = self.sampler = None
        self.step = 0
        self.tokens_seen = 0
        self.train_seconds = 0.0
        self.lr_scale = 1.0
        self.nan_events = 0
        self.last_checkpoint_time = now()

    def say(self, recipient, kind, content):
        if self.bus is not None:
            self.bus.post(NAME, recipient, kind, content)

    # ---------------- 构建 / 恢复 ----------------
    def model_config(self):
        return ModelConfig(vocab_size=self.manifest['tokenizer']['vocab_size'],
                           max_seq_len=max(2048, self.plan['seq_len']), dropout=0.0, tie_embeddings=True,
                           attention_backend='sdpa', **self.plan['model'])

    def build(self):
        torch.manual_seed(self.tcfg['seed'])
        self.model = DecoderLM(self.model_config()).to(self.device)
        decay, no_decay = [], []
        for name, p in self.model.named_parameters():
            (decay if p.dim() >= 2 else no_decay).append(p)
        self.optimizer = torch.optim.AdamW(
            [{'params': decay, 'weight_decay': self.tcfg['weight_decay']},
             {'params': no_decay, 'weight_decay': 0.0}],
            lr=self.plan['peak_lr'], betas=(0.9, self.tcfg['beta2']), eps=1e-8)
        self.sampler = MixtureSampler(self.manifest, self.plan['weights'], self.plan['seq_len'], self.tcfg['seed'])
        latest = self.ckpt_dir / 'latest.pt'
        if latest.exists():
            self.load(latest)
            self.say('decision', 'report', {'action': 'resumed', 'step': self.step, 'tokens_seen': self.tokens_seen})
        else:
            self.say('decision', 'report', {'action': 'initialized', 'params': self.model.count_parameters()})

    def load(self, path):
        ckpt = torch.load(path, map_location='cpu', weights_only=False)
        self.model.load_state_dict(ckpt['model'])
        self.optimizer.load_state_dict(ckpt['optimizer'])
        self.step, self.tokens_seen = ckpt['step'], ckpt['tokens_seen']
        self.train_seconds = ckpt.get('train_seconds', 0.0)
        self.lr_scale = ckpt.get('lr_scale', 1.0)
        for shard_domain, shard in ckpt.get('extra_shards', []):
            if not any(str(RUN_DIR / shard['path']) == str(a.filename) for a, _ in self.sampler.shards.get(shard_domain, [])):
                self.sampler.add_shard(shard_domain, shard)
        self.sampler.set_weights(ckpt.get('weights', self.plan['weights']))
        self.sampler.load_state_dict(ckpt['sampler'])
        torch.set_rng_state(ckpt['torch_rng'])

    def checkpoint_payload(self):
        return {'model': self.model.state_dict(), 'optimizer': self.optimizer.state_dict(),
                'step': self.step, 'global_step': self.step, 'tokens_seen': self.tokens_seen,
                'train_seconds': self.train_seconds, 'lr_scale': self.lr_scale,
                'sampler': self.sampler.state_dict(), 'weights': dict(self.sampler.weights),
                'extra_shards': self.plan.get('extra_shards', []), 'torch_rng': torch.get_rng_state(),
                'config': self.model.config.to_dict(), 'tokenizer': self.tokenizer_dict, 'plan': self.plan}

    def save_latest(self):
        latest, prev = self.ckpt_dir / 'latest.pt', self.ckpt_dir / 'prev.pt'
        tmp = self.ckpt_dir / 'latest.pt.tmp'
        torch.save(self.checkpoint_payload(), tmp)
        if latest.exists():
            os.replace(latest, prev)
        os.replace(tmp, latest)
        self.last_checkpoint_time = now()

    def save_best(self, val_loss):
        payload = {'model': self.model.state_dict(), 'config': self.model.config.to_dict(),
                   'tokenizer': self.tokenizer_dict, 'step': self.step, 'global_step': self.step,
                   'tokens_seen': self.tokens_seen, 'val_loss': float(val_loss)}
        tmp = self.ckpt_dir / 'best.pt.tmp'
        torch.save(payload, tmp)
        os.replace(tmp, self.ckpt_dir / 'best.pt')

    def rollback(self, lr_factor):
        prev = self.ckpt_dir / 'prev.pt'
        source = prev if prev.exists() else self.ckpt_dir / 'latest.pt'
        self.load(source)
        self.lr_scale *= lr_factor
        self.say('decision', 'report', {'action': 'rolled_back', 'to_step': self.step, 'lr_scale': self.lr_scale,
                                        'source': source.name})

    # ---------------- 学习率 ----------------
    def lr_at(self, step):
        peak, total = self.plan['peak_lr'] * self.lr_scale, max(1, self.plan['total_steps'])
        warm = self.plan['warmup_steps']
        floor = peak * self.tcfg['min_lr_ratio']
        if step < warm:
            return peak * (step + 1) / warm
        progress = min(1.0, (step - warm) / max(1, total - warm))
        return floor + 0.5 * (peak - floor) * (1 + math.cos(math.pi * progress))

    # ---------------- 分段训练 ----------------
    def run_segment(self, seconds, hard_deadline):
        """训练 seconds 秒（或到计划步数/截止/控制信号）；返回本段监控报告。"""
        plan = self.plan
        micro, accum, T = plan['micro_batch'], plan['accum'], plan['seq_len']
        precision = plan['precision']
        self.model.train()
        seg_start, seg_step0, seg_tokens0 = now(), self.step, self.tokens_seen
        losses, grad_norms, nonfinite, consecutive_bad = [], [], 0, 0
        log = self.log_path.open('a', encoding='utf-8')
        stop_reason = 'segment_time'
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
                total_loss = 0.0
                for _ in range(accum):
                    batch = self.sampler.batch(micro).to(self.device)
                    with autocast_ctx(self.device, precision):
                        loss = self.model(batch[:, :-1], targets=batch[:, 1:]).loss / accum
                    loss.backward()
                    total_loss += float(loss.detach())
                grad_norm = float(torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.tcfg['grad_clip']))
                if not (math.isfinite(total_loss) and math.isfinite(grad_norm)):
                    # 跳过这一步，不更新参数；连续出现则请求回滚
                    self.optimizer.zero_grad(set_to_none=True)
                    nonfinite += 1
                    consecutive_bad += 1
                    self.nan_events += 1
                    if consecutive_bad >= 3:
                        stop_reason = 'nonfinite'
                        break
                    continue
                consecutive_bad = 0
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)
                self.step += 1
                self.tokens_seen += micro * T * accum
                sync(self.device) if self.step % 20 == 0 else None
                dt = time.perf_counter() - t0
                self.train_seconds += dt
                losses.append(total_loss)
                grad_norms.append(grad_norm)
                if self.step % 20 == 0:
                    rec = {'time': stamp(), 'step': self.step, 'tokens': self.tokens_seen, 'loss': round(total_loss, 4),
                           'lr': lr, 'grad_norm': round(grad_norm, 3), 'step_seconds': round(dt, 3),
                           'tok_s': round(micro * T * accum / dt, 1)}
                    log.write(json.dumps(rec) + '\n')
                    log.flush()
                if now() - self.last_checkpoint_time >= self.tcfg['checkpoint_minutes'] * 60:
                    self.save_latest()
        finally:
            log.close()
        sync(self.device)
        if self.step > seg_step0:
            self.save_latest()
        elapsed = max(1e-6, now() - seg_start)
        window = losses[-50:]
        report = {'action': 'segment_done', 'stop_reason': stop_reason, 'step': self.step,
                  'total_steps': plan['total_steps'], 'tokens_seen': self.tokens_seen,
                  'segment_steps': self.step - seg_step0,
                  'segment_tok_s': round((self.tokens_seen - seg_tokens0) / elapsed, 1),
                  'train_loss': round(sum(window) / len(window), 4) if window else None,
                  'grad_norm_median': round(float(np.median(grad_norms)), 3) if grad_norms else None,
                  'grad_norm_max': round(max(grad_norms), 3) if grad_norms else None,
                  'nonfinite_steps': nonfinite, 'lr': self.lr_at(self.step), 'lr_scale': self.lr_scale,
                  'epochs': self.sampler.epochs(), 'weights': {d: round(w, 4) for d, w in self.sampler.weights.items()}}
        self.say('decision', 'report', report)
        return report

    def apply_weights(self, weights):
        self.sampler.set_weights(weights)
        self.plan['weights'] = dict(weights)

    def add_shard(self, domain, shard):
        self.sampler.add_shard(domain, shard)
        self.plan.setdefault('extra_shards', []).append([domain, shard])
