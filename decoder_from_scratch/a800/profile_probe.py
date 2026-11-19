"""练习 10：训练一步的时间都花在哪 —— torch.profiler + MFU（训练结束后、关机前运行，约 3–5 分钟）。

做法（用真实训练配置：同样的 micro batch / 梯度累积 / 序列长度 / bf16 / torch.compile / fused AdamW）：
  1. 计时：编译版跑 2 步预热 + 5 步计时，得到每步时间、tok/s、峰值显存；再用未编译（eager）模型跑 3 步，看 compile 带来多少加速；
  2. MFU：每 token 计算量 ≈ 6·N_非嵌入 + 12·L·d·T（前向+反向；后一项是注意力 QKᵀ 与 AV），除以 A800 bf16 稠密峰值 312 TFLOPS；
  3. profiler：抓 1 个完整训练步（含全部梯度累积微批、裁剪和优化器），按 GPU 时间把 kernel 归类：
     矩阵乘（GEMM）/ 注意力（FlashAttention）/ Triton 融合 kernel（torch.compile 生成的逐元素、归一化等）/
     优化器 / 拷贝与清零 / 其它；并计算 GPU 忙碌占比（kernel 总时间 ÷ 墙钟时间，差值是 CPU 调度、同步等空隙）；
  4. 导出 Chrome trace（gzip），可在 chrome://tracing 或 https://ui.perfetto.dev 打开看时间线。
注意：会对模型做几次优化器更新，所以放在所有评估之后；模型文件（final.pt / edited.pt）早已保存，不受影响。
结果写入 RUN_DIR/profile/。
"""

import gzip
import json
import math
import os
import re
import shutil
import time

import torch

from .common import RUN_DIR, now, stamp, write_json, write_text
from .trainer_agent import autocast

NAME = 'profile'
DEFAULTS = {'enabled': True, 'warmup_steps': 2, 'timed_steps': 5, 'eager_steps': 3, 'peak_tflops': 312.0}
CATS = [('注意力（FlashAttention 等）', r'flash|fmha|attention|sdpa|_efficient_attention'),
        ('矩阵乘（GEMM）', r'gemm|cutlass|xmma|sm80_|sm90_|ampere_|cublas|matmul|_mm_|splitk'),
        ('优化器', r'adam|multi_tensor|foreach|_fused_adam'),
        ('Triton 融合 kernel', r'triton'),
        ('拷贝 / 清零', r'memcpy|memset|copy_|fill_|zero'),
        ('交叉熵 / softmax', r'softmax|cross_entropy|nll|log_softmax')]


def flops_per_token(cfg, seq_len):
    d, L, V, ffn = cfg.hidden_size, cfg.num_layers, cfg.vocab_size, cfg.intermediate_size
    hd = d // cfg.num_heads
    attn = d * (cfg.num_heads * hd) + 2 * d * (cfg.num_kv_heads * hd) + d * d   # q, k, v, o
    mlp = 3 * d * ffn
    n_layers = L * (attn + mlp)
    n_head = V * d                                                         # 输出投影（与嵌入共享权重，但计算照算）
    return 6 * (n_layers + n_head) + 12 * L * d * seq_len, n_layers, n_head


def _sync():
    torch.cuda.synchronize()


def one_step(trainer, model):
    plan = trainer.plan
    micro, accum = plan['micro_batch'], plan['accum']
    for _ in range(accum):
        batch = trainer.sampler.batch(micro).to(trainer.device, non_blocking=True)
        with autocast(trainer.device, plan['precision']):
            loss = model(batch[:, :-1], targets=batch[:, 1:]).loss / accum
        loss.backward()
    torch.nn.utils.clip_grad_norm_(trainer.model.parameters(), trainer.tcfg['grad_clip'])
    trainer.optimizer.step()
    trainer.optimizer.zero_grad(set_to_none=True)


def timed(trainer, model, warmup, steps):
    for _ in range(warmup):
        one_step(trainer, model)
    _sync()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    for _ in range(steps):
        one_step(trainer, model)
    _sync()
    return (time.perf_counter() - t0) / steps, torch.cuda.max_memory_allocated() / 2 ** 30


def categorize(events):
    total = 0.0
    by_cat, top = {}, []
    # GPU 时间线上的用户注解（"## Call CompiledFxGraph … ##"、"Optimizer.step#AdamW.step"、ProfilerStep）是区间，
    # 不是 kernel，会和其中的 kernel 重复计数，要排除
    kernels = [e for e in events if getattr(e, 'device_type', None) == torch.autograd.DeviceType.CUDA
               and not (e.key.startswith('##') or '#' in e.key or e.key.startswith('ProfilerStep'))]
    for e in kernels or events:   # 只统计 GPU kernel 本身；CPU 侧算子的 "Self CUDA" 与 kernel 重复计数
        t = getattr(e, 'self_device_time_total', None)
        if t is None:
            t = getattr(e, 'self_cuda_time_total', 0.0)
        if not t:
            continue
        name = e.key
        total += t
        cat = next((c for c, pat in CATS if re.search(pat, name, re.I)), '其它')
        by_cat[cat] = by_cat.get(cat, 0.0) + t
        top.append((t, name, e.count))
    top.sort(reverse=True)
    return total, by_cat, top[:25]


def run_profile(config, bus, trainer, device):
    if device.type != 'cuda':
        raise RuntimeError('profiler 阶段只在 CUDA 上运行')
    cfg = dict(DEFAULTS, **config.get('profile', {}))
    out = RUN_DIR / 'profile'
    out.mkdir(parents=True, exist_ok=True)
    started = now()
    plan = trainer.plan
    tokens_per_step = plan['micro_batch'] * plan['accum'] * plan['seq_len']
    trainer.model.train()
    compiled = trainer.train_model
    step_s, mem = timed(trainer, compiled, cfg['warmup_steps'], cfg['timed_steps'])
    eager_s = None
    eager_note = None
    if compiled is not trainer.model and cfg['eager_steps']:
        try:
            eager_s, _ = timed(trainer, trainer.model, 1, cfg['eager_steps'])
        except torch.OutOfMemoryError:
            eager_note = '不编译时显存不足（OOM）：torch.compile 的算子融合也省下了中间激活的显存'
            trainer.optimizer.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
    fpt, n_layers, n_head = flops_per_token(trainer.model.config, plan['seq_len'])
    tok_s = tokens_per_step / step_s
    achieved = fpt * tok_s / 1e12
    from torch.profiler import ProfilerActivity, profile
    _sync()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        t0 = time.perf_counter()
        one_step(trainer, compiled)
        _sync()
        wall = time.perf_counter() - t0
    events = prof.key_averages()
    total_us, by_cat, top = categorize(events)
    trace = out / 'trace.json'
    prof.export_chrome_trace(str(trace))
    with open(trace, 'rb') as src, gzip.open(str(trace) + '.gz', 'wb') as dst:
        shutil.copyfileobj(src, dst)
    os.remove(trace)
    try:
        table = events.table(sort_by='self_device_time_total', row_limit=30)
    except Exception:  # noqa: BLE001
        table = events.table(sort_by='self_cuda_time_total', row_limit=30)
    write_text(out / 'top_kernels.txt', table)
    result = {
        'time': stamp(), 'minutes': round((now() - started) / 60, 1),
        'plan': {k: plan[k] for k in ('micro_batch', 'accum', 'seq_len', 'precision') if k in plan} | {'compile': bool(plan.get('compile'))},
        'tokens_per_step': tokens_per_step, 'step_seconds': round(step_s, 3), 'tok_s': round(tok_s),
        'eager_step_seconds': round(eager_s, 3) if eager_s else None,
        'compile_speedup': round(eager_s / step_s, 3) if eager_s else None, 'eager_note': eager_note,
        'peak_mem_gb': round(mem, 2),
        'flops_per_token': fpt, 'params_layers': n_layers, 'params_head': n_head,
        'attention_flops_share': round(12 * trainer.model.config.num_layers * trainer.model.config.hidden_size * plan['seq_len'] / fpt, 4),
        'achieved_tflops': round(achieved, 1), 'mfu': round(achieved / cfg['peak_tflops'], 4),
        'profiled_wall_s': round(wall, 3), 'gpu_kernel_s': round(total_us / 1e6, 3),
        'gpu_busy': round(total_us / 1e6 / wall, 4) if wall else None,
        'by_category': {k: {'seconds': round(v / 1e6, 3), 'share': round(v / total_us, 4)}
                        for k, v in sorted(by_cat.items(), key=lambda kv: -kv[1])},
        'top_kernels': [{'name': n[:120], 'seconds': round(t / 1e6, 4), 'share': round(t / total_us, 4), 'calls': c}
                        for t, n, c in top],
    }
    write_json(out / 'result.json', result)
    write_text(out / 'REPORT.md', render(result))
    trainer.optimizer.zero_grad(set_to_none=True)
    torch.cuda.empty_cache()
    return result


def render(r):
    L = ['# 训练一步的时间分解（torch.profiler）与 MFU', '',
         f'- 配置：{r["plan"]}，每步 {r["tokens_per_step"]:,} token',
         f'- 编译版：每步 {r["step_seconds"]} s，{r["tok_s"]:,} tok/s，峰值显存 {r["peak_mem_gb"]} GB'
         + (f'；eager（不编译）每步 {r["eager_step_seconds"]} s → torch.compile 加速 ×{r["compile_speedup"]}' if r['eager_step_seconds'] else (f'；{r["eager_note"]}' if r.get('eager_note') else '')),
         f'- 每 token 计算量 ≈ 6·(层内参数 {r["params_layers"] / 1e6:.1f}M + 输出投影 {r["params_head"] / 1e6:.1f}M) + 12·L·d·T '
         f'= {r["flops_per_token"] / 1e9:.2f} GFLOPs（注意力 QKᵀ/AV 占 {r["attention_flops_share"]:.0%}）',
         f'- 实际算力 {r["achieved_tflops"]} TFLOPS，**MFU = {r["mfu"]:.1%}**（A800 bf16 稠密峰值 312 TFLOPS）',
         f'- profiler 抓的那一步：墙钟 {r["profiled_wall_s"]} s，GPU kernel 合计 {r["gpu_kernel_s"]} s，'
         f'**GPU 忙碌 {r["gpu_busy"]:.1%}**（其余是 CPU 调度、数据搬运、同步造成的空隙；profiler 本身也有少量开销）', '',
         '## 按类别', '', '| 类别 | GPU 时间 (s) | 占比 |', '|---|---|---|']
    L += [f'| {k} | {v["seconds"]} | {v["share"]:.1%} |' for k, v in r['by_category'].items()]
    L += ['', '## 耗时最多的 kernel', '', '| kernel | 调用次数 | GPU 时间 (s) | 占比 |', '|---|---|---|---|']
    L += [f'| `{k["name"]}` | {k["calls"]} | {k["seconds"]} | {k["share"]:.1%} |' for k in r['top_kernels'][:15]]
    L += ['', '## 怎么读', '',
          '- 矩阵乘占比越高越好（说明时间花在"有用"的计算上）；Triton 融合 kernel / 拷贝占比高，说明逐元素操作和访存是瓶颈。',
          '- GPU 忙碌明显低于 100%：查数据加载（是否预取）、Python 开销、频繁的 .item()/float() 同步。',
          '- MFU 30–50% 对这种规模的单卡训练是正常区间；继续提升通常靠更大的 micro batch、更长的序列、融合优化器 / 注意力。',
          '- 时间线：把 `trace.json.gz` 解压后在 https://ui.perfetto.dev 打开，能看到每个微批的前向、反向和优化器。']
    return '\n'.join(L) + '\n'
