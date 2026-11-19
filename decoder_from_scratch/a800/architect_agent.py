"""模型结构与算法实现智能体：结构正确性自检、0.5B 候选结构吞吐基准、按吞吐与深度选择结构。

模型本身复用 decoder_from_scratch/model.py（Embedding → [Pre-RMSNorm + RoPE/GQA 注意力 + 残差
→ Pre-RMSNorm + SwiGLU FFN + 残差] × L → RMSNorm → 与 Embedding 共享权重的输出层）。

三个候选参数量都约 0.49B，区别是"深而窄"还是"浅而宽"：
  * 宽模型矩阵更大，GPU 利用率更高、跑得更快；
  * 对小模型，同参数量下更深通常效果更好（MobileLLM 等工作的结论）。
选择规则：在吞吐不低于最快候选 85% 的候选里选层数最多的——用不超过 15% 的速度换深度。
"""

import json
import math
import subprocess
import sys
import time

import torch

from ..config import ModelConfig
from ..model import DecoderLM
from .common import PROJECT_DIR, heartbeat

NAME = 'architect'

CANDIDATES = {
    # head_dim 统一 64；GQA 每 4~5 个 Query 头共享一组 KV；SwiGLU 中间层约 2.7 倍 hidden（三个矩阵）
    'd1536_l18': dict(hidden_size=1536, num_layers=18, num_heads=24, num_kv_heads=4, intermediate_size=4096),
    'd1280_l26': dict(hidden_size=1280, num_layers=26, num_heads=20, num_kv_heads=4, intermediate_size=3456),
    'd1024_l40': dict(hidden_size=1024, num_layers=40, num_heads=16, num_kv_heads=4, intermediate_size=2816),
}
A800_BF16_PEAK = 312e12


def flops_per_token(cfg, seq_len):
    """训练 FLOPs/token ≈ 6N（矩阵乘法，前向 2N + 反向 4N）+ 注意力 12·L·d·T（QK^T 与 AV）。"""
    # 共享权重时参数只计一次：嵌入查表不是乘法，但同一矩阵作为 lm_head 做了乘法，所以参与乘法的参数量 = N。
    n = DecoderLM(cfg).count_parameters()
    return 6 * n + 12 * cfg.num_layers * cfg.hidden_size * seq_len


class ArchitectAgent:
    def __init__(self, config, bus):
        self.config, self.bus = config, bus

    def say(self, recipient, kind, content):
        if self.bus is not None:
            self.bus.post(NAME, recipient, kind, content)

    @staticmethod
    def model_config(name, vocab_size=32000, seq_len=1024):
        return ModelConfig(vocab_size=vocab_size, max_seq_len=max(2048, seq_len), dropout=0.0, tie_embeddings=True,
                           attention_backend='sdpa', **CANDIDATES[name])

    # ---------------- 自检 ----------------
    def self_test(self, device):
        """因果性（未来 token 不影响过去）、KV Cache 与完整前向一致、bf16 下能过拟合单个 batch。"""
        torch.manual_seed(0)
        cfg = ModelConfig(vocab_size=512, hidden_size=128, num_layers=2, num_heads=4, num_kv_heads=2,
                          intermediate_size=256, max_seq_len=256)
        model = DecoderLM(cfg).to(device).eval()
        x = torch.randint(5, 512, (2, 64), device=device)
        out = {}
        with torch.no_grad():
            full = model(x).logits.float()
            y = x.clone()
            y[:, 40:] = torch.randint(5, 512, (2, 24), device=device)
            out['causal_max_diff'] = float((full[:, :40] - model(y).logits.float()[:, :40]).abs().max())
            cache, pieces = None, []
            for t in range(x.size(1)):
                o = model(x[:, t:t + 1], past_key_values=cache, use_cache=True)
                cache = o.past_key_values
                pieces.append(o.logits.float())
            out['kv_cache_max_diff'] = float((full - torch.cat(pieces, 1)).abs().max())
        model.train()
        opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
        batch = torch.randint(5, 512, (2, 33), device=device)
        losses = []
        for _ in range(120):
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                loss = model(batch[:, :-1], targets=batch[:, 1:]).loss
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            losses.append(float(loss))
        out.update(overfit_first=round(losses[0], 3), overfit_last=round(losses[-1], 3), device=str(device))
        out['passed'] = bool(out['causal_max_diff'] < 1e-4 and out['kv_cache_max_diff'] < 2e-3
                             and losses[-1] < 0.5 * losses[0] and all(map(math.isfinite, losses)))
        if device.type == 'cuda':
            out['gpu'] = torch.cuda.get_device_name(device)
            out['gpu_mem_gb'] = round(torch.cuda.get_device_properties(device).total_memory / 1e9)
            out['bf16_supported'] = torch.cuda.is_bf16_supported()
        return out

    # ---------------- 吞吐基准（每个组合一个子进程，互不影响显存） ----------------
    def benchmark(self, seq_len, vocab_size=32000, micro_options=(16, 32), compile_model=True):
        rows = []
        for name in CANDIDATES:
            cfg = self.model_config(name, vocab_size, seq_len)
            params = DecoderLM(cfg).count_parameters()
            fpt = flops_per_token(cfg, seq_len)
            best = None
            trials = []
            for micro in micro_options:
                args = json.dumps({'name': name, 'micro': micro, 'seq_len': seq_len, 'vocab': vocab_size,
                                   'compile': compile_model})
                try:
                    proc = subprocess.run([sys.executable, '-m', 'decoder_from_scratch.a800.bench_worker', args],
                                          capture_output=True, text=True, timeout=900, cwd=str(PROJECT_DIR))
                    line = [l for l in proc.stdout.splitlines() if l.startswith('BENCH ')]
                    res = json.loads(line[-1][6:]) if line else {'ok': False, 'error': proc.stderr[-400:]}
                except subprocess.TimeoutExpired:
                    res = {'ok': False, 'error': 'timeout'}
                res['micro'] = micro
                trials.append(res)
                heartbeat(f'benchmark {name} micro={micro}')
                if res.get('ok') and (best is None or res['tok_s'] > best['tok_s']):
                    best = res
            row = {'name': name, 'params': params, 'layers': cfg.num_layers, 'hidden': cfg.hidden_size,
                   'flops_per_token': fpt, 'trials': trials}
            if best:
                row.update(tok_s=best['tok_s'], micro_batch=best['micro'], peak_mem_gb=best['peak_mem_gb'],
                           mfu=round(best['tok_s'] * fpt / A800_BF16_PEAK, 3))
            rows.append(row)
            self.say('decision', 'report', {'action': 'benchmark', **{k: v for k, v in row.items() if k != 'trials'},
                                            'trial_errors': [t.get('error', '')[:120] for t in trials if not t.get('ok')]})
        return rows

    def choose(self, rows):
        ok = [r for r in rows if r.get('tok_s')]
        if not ok:
            raise RuntimeError('no candidate could be benchmarked')
        fastest = max(r['tok_s'] for r in ok)
        eligible = [r for r in ok if r['tok_s'] >= 0.85 * fastest]
        pick = max(eligible, key=lambda r: (r['layers'], r['tok_s']))
        why = (f"最快候选 {fastest:.0f} tok/s；在吞吐 ≥85% 的候选里选层数最多的 {pick['name']}"
               f"（{pick['params'] / 1e6:.0f}M 参数，{pick['layers']} 层，{pick['tok_s']:.0f} tok/s，MFU {pick['mfu']:.0%}）。")
        return dict(pick, model=CANDIDATES[pick['name']], rationale=why)


def bench(name, micro, seq_len, vocab, compile_model, steps=8, warmup=3):
    device = torch.device('cuda')
    torch.backends.cuda.matmul.allow_tf32 = True
    cfg = ArchitectAgent.model_config(name, vocab, seq_len)
    model = DecoderLM(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, fused=True)
    run = torch.compile(model) if compile_model else model
    x = torch.randint(5, vocab, (micro, seq_len + 1), device=device)
    torch.cuda.reset_peak_memory_stats()
    losses = []
    for i in range(warmup + steps):
        if i == warmup:
            torch.cuda.synchronize()
            start = time.perf_counter()
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            loss = run(x[:, :-1], targets=x[:, 1:]).loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        losses.append(float(loss))
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    if not all(map(math.isfinite, losses)):
        raise RuntimeError('non-finite loss')
    return {'ok': True, 'tok_s': round(micro * seq_len * steps / elapsed), 'first_loss': round(losses[0], 3),
            'peak_mem_gb': round(torch.cuda.max_memory_allocated() / 1e9, 1)}
