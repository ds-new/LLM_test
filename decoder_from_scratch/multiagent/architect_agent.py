"""模型结构与算法实现智能体：负责结构正确性自检、候选结构吞吐基准、按预算选择规模和精度。

复用工程 decoder_from_scratch/model.py 的 DecoderLM（Embedding、RoPE、GQA、Pre-RMSNorm、SwiGLU、残差）。
"""

import contextlib
import math
import time

import torch

from ..config import ModelConfig
from ..model import DecoderLM

NAME = 'architect'

# 候选结构：head_dim 统一 64，GQA 组数 4~5，FFN ≈ 2.7×hidden（SwiGLU 三个矩阵）
CANDIDATES = {
    'xs_d384_l8':   dict(hidden_size=384, num_layers=8,  num_heads=6,  num_kv_heads=2, intermediate_size=1024),
    's_d512_l8':    dict(hidden_size=512, num_layers=8,  num_heads=8,  num_kv_heads=2, intermediate_size=1408),
    'm_d640_l10':   dict(hidden_size=640, num_layers=10, num_heads=10, num_kv_heads=2, intermediate_size=1728),
    'l_d768_l12':   dict(hidden_size=768, num_layers=12, num_heads=12, num_kv_heads=4, intermediate_size=2048),
}
PEAK_LR = {384: 1.2e-3, 512: 1.0e-3, 640: 8e-4, 768: 6e-4}


def sync(device):
    if device.type == 'mps':
        torch.mps.synchronize()
    elif device.type == 'cuda':
        torch.cuda.synchronize()


def autocast_ctx(device, precision):
    if precision == 'bf16':
        return torch.autocast(device_type=device.type, dtype=torch.bfloat16)
    return contextlib.nullcontext()


class ArchitectAgent:
    def __init__(self, config, bus):
        self.config, self.bus = config, bus

    def say(self, recipient, kind, content):
        if self.bus is not None:
            self.bus.post(NAME, recipient, kind, content)

    def model_config(self, name, vocab_size, seq_len):
        return ModelConfig(vocab_size=vocab_size, max_seq_len=max(2048, seq_len), dropout=0.0,
                           tie_embeddings=True, attention_backend='sdpa', **CANDIDATES[name])

    # ---------------- 结构自检 ----------------
    def self_test(self, device):
        """在目标设备上验证：因果性（未来 token 不影响过去）、KV Cache 与完整前向一致、能过拟合单个 batch。"""
        results = {}
        torch.manual_seed(0)
        cfg = ModelConfig(vocab_size=512, hidden_size=128, num_layers=2, num_heads=4, num_kv_heads=2,
                          intermediate_size=256, max_seq_len=256)
        model = DecoderLM(cfg).to(device)
        model.eval()
        x = torch.randint(5, 512, (2, 64), device=device)
        with torch.no_grad():
            full = model(x).logits.float()
            y = x.clone()
            y[:, 40:] = torch.randint(5, 512, (2, 24), device=device)
            changed = model(y).logits.float()
            results['causal_max_diff'] = float((full[:, :40] - changed[:, :40]).abs().max())
            cache, pieces = None, []
            for t in range(x.size(1)):
                out = model(x[:, t:t + 1], past_key_values=cache, use_cache=True)
                cache = out.past_key_values
                pieces.append(out.logits.float())
            inc = torch.cat(pieces, dim=1)
            results['kv_cache_max_diff'] = float((full - inc).abs().max())
        model.train()
        opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
        batch = torch.randint(5, 512, (2, 33), device=device)
        losses = []
        for _ in range(120):
            out = model(batch[:, :-1], targets=batch[:, 1:])
            opt.zero_grad(set_to_none=True)
            out.loss.backward()
            opt.step()
            losses.append(float(out.loss))
        results['overfit_first_loss'] = round(losses[0], 4)
        results['overfit_last_loss'] = round(losses[-1], 4)
        ok = (results['causal_max_diff'] < 1e-4 and results['kv_cache_max_diff'] < 1e-3
              and losses[-1] < 0.6 * losses[0] and all(math.isfinite(v) for v in losses))
        results['passed'] = bool(ok)
        results['device'] = str(device)
        del model, opt
        if device.type == 'mps':
            torch.mps.empty_cache()
        return results

    # ---------------- 吞吐基准 ----------------
    def benchmark(self, device, vocab_size, seq_len, names=None, train_seconds=None, target_ratio=20.0):
        """每个(结构, 精度, micro batch)在独立子进程中测速，避免显存残留互相干扰；超时视为过慢。

        v1 在同一进程内用 micro=16 连续测试，32k 词表的 logits 占用过大导致内存抖动、吞吐随候选递减，结果失真。
        从小到大测试；若某结构在剩余时间内连目标 token/参数的 40% 都达不到，更大的结构不再测试。
        """
        import subprocess, sys, json as _json
        rows = []
        for name in (names or CANDIDATES):
            cfg = self.model_config(name, vocab_size, seq_len)
            params = DecoderLM(cfg).count_parameters()
            best_tps = 0.0
            for precision in ('fp32', 'bf16'):
                row = {'name': name, 'precision': precision, 'params': params}
                trials = []
                for micro in (8, 4):
                    args = _json.dumps({'name': name, 'precision': precision, 'micro': micro, 'vocab': vocab_size,
                                        'seq_len': seq_len, 'device': str(device)})
                    try:
                        out = subprocess.run([sys.executable, '-m', 'decoder_from_scratch.multiagent.bench_worker', args],
                                             capture_output=True, text=True, timeout=300)
                        line = [l for l in out.stdout.splitlines() if l.startswith('BENCH_RESULT ')]
                        res = _json.loads(line[-1][len('BENCH_RESULT '):]) if line else {'ok': False, 'error': out.stderr[-300:]}
                    except subprocess.TimeoutExpired:
                        res = {'ok': False, 'error': 'timeout (>300s, too slow)'}
                    res['micro_batch'] = micro
                    trials.append(res)
                ok = [t for t in trials if t.get('ok')]
                if ok:
                    top = max(ok, key=lambda t: t['tokens_per_second'])
                    row.update(tokens_per_second=top['tokens_per_second'], first_loss=top['first_loss'],
                               micro_batch=top['micro_batch'])
                    best_tps = max(best_tps, top['tokens_per_second'])
                else:
                    row['error'] = '; '.join(str(t.get('error'))[:120] for t in trials)
                row['trials'] = [{k: t.get(k) for k in ('micro_batch', 'tokens_per_second', 'ok')} for t in trials]
                rows.append(row)
                self.say('decision', 'report', {'action': 'benchmark', **{k: v for k, v in row.items() if k != 'error'},
                                                **({'error': row['error'][:200]} if 'error' in row else {})})
            if train_seconds and best_tps and best_tps * train_seconds * 0.9 / params < 0.4 * target_ratio:
                self.say('decision', 'report', {'action': 'benchmark_stop', 'reason': f'{name} 已远低于目标 token/参数，不再测试更大结构'})
                break
        return rows

    def _bench_one(self, device, cfg, precision, micro, seq_len, steps=6, warmup=2):
        torch.manual_seed(0)
        model = DecoderLM(cfg).to(device)
        model.train()
        opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
        x = torch.randint(5, cfg.vocab_size, (micro, seq_len + 1), device=device)
        losses = []
        try:
            for i in range(warmup + steps):
                if i == warmup:
                    sync(device)
                    start = time.perf_counter()
                with autocast_ctx(device, precision):
                    loss = model(x[:, :-1], targets=x[:, 1:]).loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                opt.zero_grad(set_to_none=True)
                losses.append(float(loss))
            sync(device)
            elapsed = time.perf_counter() - start
        finally:
            del model, opt, x
            if device.type == 'mps':
                torch.mps.empty_cache()
        if not all(math.isfinite(v) for v in losses):
            raise RuntimeError('non-finite loss in benchmark')
        return {'tokens_per_second': round(micro * seq_len * steps / elapsed, 1),
                'first_loss': round(losses[0], 4)}

    # ---------------- 规模选择 ----------------
    def propose(self, rows, train_seconds, unique_tokens, target_ratio):
        """按 Chinchilla 经验（训练 token ≈ 20×参数）在给定算力下挑选最接近计算最优的结构。

        bf16 只在比 fp32 快 10% 以上、且首步 loss 与 fp32 相差 < 2% 时采用。
        """
        best_by_name = {}
        for name in CANDIDATES:
            fp32 = next((r for r in rows if r['name'] == name and r['precision'] == 'fp32' and 'tokens_per_second' in r), None)
            bf16 = next((r for r in rows if r['name'] == name and r['precision'] == 'bf16' and 'tokens_per_second' in r), None)
            choice = fp32
            if bf16 and fp32 and bf16['tokens_per_second'] > 1.10 * fp32['tokens_per_second'] \
                    and abs(bf16['first_loss'] - fp32['first_loss']) < 0.02 * fp32['first_loss']:
                choice = bf16
            elif bf16 and not fp32:
                choice = bf16
            if choice:
                best_by_name[name] = choice
        if not best_by_name:
            raise RuntimeError('no candidate could be benchmarked')
        scored = []
        for name, row in best_by_name.items():
            # 0.9：数据加载、评估、检查点等实际开销
            tokens = row['tokens_per_second'] * train_seconds * 0.9
            ratio = tokens / row['params']
            scored.append({'name': name, 'precision': row['precision'], 'params': row['params'],
                           'micro_batch': row['micro_batch'], 'tokens_per_second': row['tokens_per_second'],
                           'planned_tokens': int(tokens), 'tokens_per_param': round(ratio, 1),
                           'epochs_over_unique': round(tokens / max(1, unique_tokens), 2),
                           'score': abs(math.log(ratio / target_ratio))})
        scored.sort(key=lambda s: s['score'])
        pick = dict(scored[0])
        hidden = CANDIDATES[pick['name']]['hidden_size']
        pick['peak_lr'] = PEAK_LR[hidden]
        pick['model'] = CANDIDATES[pick['name']]
        pick['alternatives'] = [{k: s[k] for k in ('name', 'precision', 'params', 'tokens_per_second',
                                                    'tokens_per_param')} for s in scored[1:]]
        pick['rationale'] = (f"在剩余约 {train_seconds / 3600:.1f} 小时训练时间下，{pick['name']} "
                             f"({pick['params'] / 1e6:.1f}M 参数, {pick['precision']}, {pick['tokens_per_second']:.0f} tok/s) "
                             f"可训练约 {pick['planned_tokens'] / 1e9:.2f}B token，即 {pick['tokens_per_param']} token/参数，"
                             f"最接近目标 {target_ratio}。")
        return pick
