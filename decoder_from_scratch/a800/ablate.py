"""消融实验队列（第一批）：固定数据、模型和 token 预算，一次只改一个变量。

复用 tokenizer 对照实验（tok_ab）已经编码好的数据、验证文档和运算评测题，所以每组结果可以和对照实验的基准组直接比。
  1. 学习率扫描：峰值学习率 5e-4 / 1e-3（基准，直接取对照实验的结果）/ 2e-3 / 4e-3，其余完全相同；
  2. 多种子方差：基准配置换 2 个随机种子（初始化和数据顺序都随之改变），估计"同一配置重复训练"的噪声 σ；
     有了 σ 才能判断学习率之间、v1/v2 tokenizer 之间的差异是不是真的（两次训练之差的噪声是 √2·σ，超过 3 倍才算显著）。
每组训练额外记录两项诊断：梯度裁剪实际生效的步数比例、每步更新量与权重的范数比（update/weight ratio）。
角色：决策（排队、判定显著性）/ 训练 / 评估 / 结果分析，消息写入 <RUN>/events.jsonl。
阶段：wait（等对照实验结束）→ plan → experiments → report → done。可断点续跑；发散的实验记为 diverged，不中断队列。
"""

import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import threading
import time
import traceback

import numpy as np

from .common import RUN_DIR, Bus, FatalError, StopRequested, control_signal, heartbeat, read_json, stamp, write_json, write_text

ABL = Path(os.environ.get('ABLATE_RUN_DIR', '/root/autodl-tmp/a800_ablate'))
if Path(RUN_DIR) != ABL:
    raise RuntimeError(f'common.RUN_DIR={RUN_DIR} 与 ABLATE_RUN_DIR={ABL} 不一致：请用 run_ablate.sh 启动')
AB = Path(os.environ.get('ABLATE_AB_DIR', '/root/autodl-tmp/a800_tok_ab'))
V1_TOK = Path('/root/autodl-tmp/a800_pretrain_v1/tokenizer/tokenizer_hf.json')
BOS, EOS, VOCAB = 2, 3, 32000
EVAL_WEIGHTS = {'zh_cci3': 0.28, 'zh_fwedu': 0.14, 'zh_industry': 0.13, 'en_fwedu': 0.18, 'en_cosmo': 0.07, 'code': 0.12, 'math': 0.08}
PHASES = ['wait', 'plan', 'experiments', 'report', 'done']

CFG = {
    'tok': None,                 # None = 按对照结论自动选（v2 明显更好 → v2，否则 v1）
    'wait_ab': True,
    'base': {'peak_lr': 1e-3, 'seed': 1234},   # 与 tok_ab 的训练配置一致
    'lrs': [2e-3, 5e-4, 4e-3],
    'seeds': [2, 3],
    'model': dict(hidden_size=768, num_layers=16, num_heads=12, num_kv_heads=4, intermediate_size=2048),
    'seq_len': 1024, 'micro': 64, 'accum': 4, 'warmup': 300, 'decay_fraction': 0.2, 'min_ratio': 0.1,
    'wd': 0.1, 'beta2': 0.95, 'clip': 1.0, 'compile': True,
    'tokens': None,              # None = 把这份数据完整过一遍（与对照实验相同）
    'arith_limit': None,
    'keep_weights': False,       # 评估完删除各组权重，只留日志和结果（省数据盘）
}
CFG.update(json.loads(os.environ.get('ABLATE_CFG', '{}')))


class keepalive:
    def __init__(self, note, every=300):
        self.note, self.every, self.stop = note, every, threading.Event()

    def __enter__(self):
        heartbeat(self.note)

        def loop():
            while not self.stop.wait(self.every):
                heartbeat(self.note)
        threading.Thread(target=loop, daemon=True).start()
        return self

    def __exit__(self, *exc):
        self.stop.set()


def check_control():
    if control_signal() == 'stop':
        raise StopRequested('STOP file present')


def tok_path(tok):
    return V1_TOK if tok == 'v1' else AB / 'tokenizer_v2.json'


def model_config():
    from ..config import ModelConfig
    return ModelConfig(vocab_size=VOCAB, max_seq_len=2048, dropout=0.0, tie_embeddings=True, attention_backend='sdpa', **CFG['model'])


# ------------------------------------------------------------------ 阶段：等待对照实验、排队 ----------
def ab_running():
    return subprocess.run(['pgrep', '-f', 'a800.tok_ab run'], capture_output=True).returncode == 0


def phase_wait(state, bus):
    if not CFG['wait_ab']:
        return
    bus.post('decision', 'all', 'report', {'action': 'waiting_for_tok_ab'})
    idle = 0
    with keepalive('wait tok_ab'):
        while True:
            check_control()
            phase = (read_json(AB / 'state.json', {}) or {}).get('phase')
            running = ab_running()
            if phase == 'done' and not running:
                return
            idle = 0 if running else idle + 1
            if idle >= 40:  # 20 分钟内对照进程一直不在、又没有完成：对照实验失败了
                raise FatalError(f'对照实验没有完成（阶段 {phase}）且进程已不在，无法开始消融')
            time.sleep(30)


def phase_plan(state, bus):
    st = read_json(AB / 'state.json', {}) or {}
    verdict = ((st.get('results') or {}).get('report') or {}).get('verdict', '')
    tok = CFG['tok'] or ('v2' if '明显更好' in verdict else 'v1')
    man = read_json(AB / 'data' / 'manifest.json')
    shards = [s for e in man['sources'].values() for s in e.get(tok, [])]
    missing = [s['path'] for s in shards if not Path(s['path']).exists()]
    if not shards or missing:
        raise FatalError(f'对照实验的 {tok} token 数据不完整（缺 {len(missing)} 个分片），无法复用')
    res = (read_json(AB / 'results.json') or {}).get(tok)
    base = None
    if res:
        base = {'name': 'base', 'group': 'base', 'peak_lr': CFG['base']['peak_lr'], 'seed': CFG['base']['seed'], 'diverged': False,
                'bpb_weighted': res['bpb_weighted'], 'bpb': {s: v['bpb'] for s, v in res['bpb'].items()},
                'arith_mean': res['arith_mean'], 'arith': res['arith'], 'train': (st.get('results') or {}).get('train_' + tok)}
    queue = [{'name': f'lr_{lr:g}', 'group': 'lr', 'peak_lr': lr, 'seed': CFG['base']['seed']} for lr in CFG['lrs']]
    queue += [{'name': f'seed_{s}', 'group': 'seed', 'peak_lr': CFG['base']['peak_lr'], 'seed': s} for s in CFG['seeds']]
    state.update(tok=tok, ab_verdict=verdict, base=base, queue=queue, results=state.get('results') or {})
    bus.post('decision', 'all', 'decision', {'decision': 'plan', 'why': f'对照结论"{verdict}"，消融用 {tok} tokenizer 的数据；'
                                             f'队列：{[e["name"] for e in queue]}；基准组直接取对照实验的结果'})


# ------------------------------------------------------------------ 训练智能体 ----------
def train(exp, tok, bus):
    import torch
    from ..model import DecoderLM
    from .trainer_agent import MixtureSampler, autocast, wsd_lr
    name = exp['name']
    out = ABL / 'runs' / name
    out.mkdir(parents=True, exist_ok=True)
    info_path = out / 'train_info.json'
    if info_path.exists():
        return read_json(info_path)
    man = read_json(AB / 'data' / 'manifest.json')
    manifest = {'sources': {n: {'train': e[tok]} for n, e in man['sources'].items() if e.get(tok)}}
    tokens = {n: sum(s['tokens'] for s in e['train']) for n, e in manifest['sources'].items()}
    total_tokens = sum(tokens.values())
    weights = {n: t / total_tokens for n, t in tokens.items()}
    T, micro, accum = CFG['seq_len'], CFG['micro'], CFG['accum']
    tps = micro * accum * T
    total_steps = max(1, min(total_tokens, CFG['tokens'] or total_tokens) // tps)
    (ABL / 'scratch').mkdir(parents=True, exist_ok=True)
    ckpt = ABL / 'scratch' / f'{name}_latest.pt'
    dev = torch.device('cuda')
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.manual_seed(exp['seed'])
    model = DecoderLM(model_config()).to(dev)
    decay = [p for p in model.parameters() if p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.dim() < 2]
    opt = torch.optim.AdamW([{'params': decay, 'weight_decay': CFG['wd']}, {'params': no_decay, 'weight_decay': 0.0}],
                            lr=exp['peak_lr'], betas=(0.9, CFG['beta2']), eps=1e-8, fused=True)
    sampler = MixtureSampler(manifest, weights, T, exp['seed'])
    step, t_train, clipped, losses, ratios = 0, 0.0, 0, [], []
    if ckpt.exists():
        ck = torch.load(ckpt, map_location='cpu', weights_only=False)
        model.load_state_dict(ck['model'])
        opt.load_state_dict(ck['opt'])
        sampler.load_state_dict(ck['sampler'])
        step, t_train, clipped, ratios = ck['step'], ck['t'], ck.get('clipped', 0), ck.get('ratios', [])
        torch.set_rng_state(ck['rng'])
        del ck
    cmodel = torch.compile(model) if CFG['compile'] else model
    bus.post('trainer', 'decision', 'report', {'action': 'resumed' if step else 'initialized', 'run': name, 'peak_lr': exp['peak_lr'],
                                               'seed': exp['seed'], 'step': step, 'total_steps': total_steps,
                                               'params': model.count_parameters()})

    def save():
        tmp = ckpt.with_name(ckpt.name + '.tmp')
        torch.save({'model': model.state_dict(), 'opt': opt.state_dict(), 'sampler': sampler.state_dict(), 'step': step,
                    't': t_train, 'rng': torch.get_rng_state(), 'clipped': clipped, 'ratios': ratios}, tmp)
        os.replace(tmp, ckpt)

    log = (out / 'train_log.jsonl').open('a', encoding='utf-8')
    last_ck, bad, diverged = time.time(), 0, False
    prefetch = sampler.batch(micro).pin_memory()
    try:
        while step < total_steps:
            check_control()
            t0 = time.perf_counter()
            lr = wsd_lr(step, total_steps, exp['peak_lr'], CFG['warmup'], CFG['decay_fraction'], CFG['min_ratio'])
            for g in opt.param_groups:
                g['lr'] = lr
            tot = torch.zeros((), device=dev)
            for _ in range(accum):
                b = prefetch.to(dev, non_blocking=True)
                prefetch = sampler.batch(micro).pin_memory()
                with autocast(dev):
                    loss = cmodel(b[:, :-1], targets=b[:, 1:]).loss / accum
                loss.backward()
                tot += loss.detach()
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), CFG['clip'])
            lv, gv = float(tot), float(gn)
            if not (math.isfinite(lv) and math.isfinite(gv)):
                opt.zero_grad(set_to_none=True)
                bad += 1
                if bad >= 20:
                    diverged = True
                    break
                continue
            bad = 0
            clipped += int(gv > CFG['clip'])
            measure = step % 200 == 0
            if measure:
                before = [p.detach().clone() for p in decay]
            opt.step()
            opt.zero_grad(set_to_none=True)
            ratio = None
            if measure:
                num = math.sqrt(sum(float(((p.detach() - w) ** 2).sum()) for p, w in zip(decay, before)))
                den = math.sqrt(sum(float((w ** 2).sum()) for w in before))
                ratio = num / max(den, 1e-12)
                ratios.append(ratio)
                del before
            step += 1
            dt = time.perf_counter() - t0
            t_train += dt
            losses.append(lv)
            if step % 50 == 0 or step == total_steps or measure:
                rec = {'time': stamp(), 'step': step, 'total': total_steps, 'loss': round(lv, 4), 'lr': lr,
                       'grad_norm': round(gv, 3), 'tok_s': round(tps / dt)}
                if ratio is not None:
                    rec['update_ratio'] = float(f'{ratio:.3e}')
                log.write(json.dumps(rec) + '\n')
                log.flush()
                heartbeat(f'{name} step {step}/{total_steps}')
            if time.time() - last_ck > 1800:
                save()
                last_ck = time.time()
    finally:
        log.close()
    info = {'steps': step, 'total_steps': total_steps, 'tokens': step * tps, 'train_hours': round(t_train / 3600, 2),
            'tok_s': round(step * tps / max(1e-6, t_train)), 'diverged': diverged,
            'final_loss': round(float(np.mean(losses[-50:])), 4) if losses else None,
            'clip_fraction': round(clipped / max(1, step), 4),
            'update_ratio_median': float(f'{np.median(ratios):.3e}') if ratios else None}
    if not diverged:
        torch.save({'model': model.state_dict(), 'config': model_config().to_dict(), 'run': name, 'steps': step}, out / 'final.pt')
    ckpt.unlink(missing_ok=True)
    write_json(info_path, info)
    bus.post('trainer', 'decision', 'report', {'action': 'diverged' if diverged else 'run_done', 'run': name, **info})
    del model, cmodel, opt
    torch.cuda.empty_cache()
    return info


# ------------------------------------------------------------------ 评估智能体 ----------
def evaluate(name, tok_name, bus):
    import torch
    from tokenizers import Tokenizer
    from ..model import DecoderLM
    from .trainer_agent import autocast
    dev = torch.device('cuda')
    ck = torch.load(ABL / 'runs' / name / 'final.pt', map_location='cpu', weights_only=False)
    model = DecoderLM(model_config()).to(dev)
    model.load_state_dict(ck['model'])
    model.eval()
    tok = Tokenizer.from_file(str(tok_path(tok_name)))
    T, ln2 = CFG['seq_len'], math.log(2)
    by = {}
    with open(AB / 'data' / 'val_docs.jsonl', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                d = json.loads(line)
                by.setdefault(d['source'], []).append(d['text'])
    bpb = {}
    with torch.no_grad():
        for src, texts in by.items():
            heartbeat(f'eval {name} {src}')
            stream = []
            for e in tok.encode_batch(texts, add_special_tokens=False):
                stream += [BOS] + e.ids + [EOS]
            arr = torch.tensor(stream, dtype=torch.long)
            wins = [arr[s:s + T + 1] for s in range(0, len(arr) - 1, T)]
            full = [w for w in wins if len(w) == T + 1]
            groups = [torch.stack(full[i:i + 16]) for i in range(0, len(full), 16)] + \
                     [w.unsqueeze(0) for w in wins if 2 <= len(w) < T + 1]
            nll = 0.0
            for b in groups:
                b = b.to(dev)
                with autocast(dev):
                    logits = model(b[:, :-1]).logits
                tgt = b[:, 1:]
                tok_nll = -torch.log_softmax(logits.float(), -1).gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
                nll += float((tok_nll * ((tgt != BOS) & (tgt != EOS))).sum())
            bpb[src] = round(nll / ln2 / sum(len(t.encode('utf-8')) for t in texts), 4)
        ev = read_json(AB / 'data' / 'arith_eval.json')
        arith = {}
        for cat, probs in ev['problems'].items():
            heartbeat(f'eval {name} arith {cat}')
            probs = probs[:CFG['arith_limit']] if CFG['arith_limit'] else probs
            shots = ''.join(f'{a} {o} {b} = {c}\n' for a, o, b, c in ev['shots'][cat])
            correct = 0
            for a, o, b, c in probs:
                ids = [BOS] + tok.encode(shots + f'{a} {o} {b} = ', add_special_tokens=False).ids
                x = torch.tensor([ids], device=dev)
                gen = []
                for _ in range(12):
                    with autocast(dev):
                        nxt = int(model(x, last_logits_only=True).logits[0, -1].argmax())
                    if nxt == EOS:
                        break
                    gen.append(nxt)
                    if '\n' in tok.decode(gen):
                        break
                    x = torch.cat([x, torch.tensor([[nxt]], device=dev)], 1)
                m = re.match(r'\s*(-?\d+)', tok.decode(gen))
                correct += int(bool(m) and int(m.group(1)) == c)
            arith[cat] = round(correct / len(probs), 4)
    w = EVAL_WEIGHTS
    weighted = sum(bpb[s] * w[s] for s in w if s in bpb) / sum(w[s] for s in w if s in bpb)
    res = {'bpb': bpb, 'bpb_weighted': round(weighted, 4), 'arith': arith, 'arith_mean': round(float(np.mean(list(arith.values()))), 4)}
    bus.post('evaluator', 'decision', 'report', {'action': 'eval', 'run': name, 'bpb_weighted': res['bpb_weighted'],
                                                 'arith_mean': res['arith_mean']})
    del model
    torch.cuda.empty_cache()
    return res


def phase_experiments(state, bus):
    for exp in state['queue']:
        check_control()
        name = exp['name']
        if name in state['results']:
            continue
        bus.post('decision', 'trainer', 'directive', {'task': 'train', **exp})
        info = train(exp, state['tok'], bus)
        rec = {**exp, 'train': info, 'diverged': info['diverged']}
        if not info['diverged']:
            rec.update(evaluate(name, state['tok'], bus))
            if not CFG['keep_weights']:
                (ABL / 'runs' / name / 'final.pt').unlink(missing_ok=True)
        state['results'][name] = rec
        write_json(ABL / 'state.json', state)


# ------------------------------------------------------------------ 结果分析 ----------
def mean_std(values):
    v = [x for x in values if x is not None]
    if not v:
        return None, None
    return float(np.mean(v)), (float(np.std(v, ddof=1)) if len(v) >= 2 else None)


def phase_report(state, bus):
    tok, base, R = state['tok'], state.get('base'), state['results']
    rows = ([base] if base else []) + [R[e['name']] for e in state['queue'] if e['name'] in R]
    ok = [r for r in rows if not r.get('diverged')]
    seeds = [r for r in ok if r['group'] in ('base', 'seed')]
    lrs = sorted([r for r in rows if r['group'] in ('base', 'lr')], key=lambda r: r['peak_lr'])
    m_bpb, s_bpb = mean_std([r['bpb_weighted'] for r in seeds])
    m_ar, s_ar = mean_std([r['arith_mean'] for r in seeds])
    src_sigma = {s: mean_std([r['bpb'].get(s) for r in seeds])[1] for s in EVAL_WEIGHTS}
    r2 = math.sqrt(2)
    L = ['# 第一批消融报告：学习率扫描 + 多种子方差', '', f'- 生成时间：{stamp()}',
         f'- 设置：{tok} tokenizer，d768_l16（约 1.25 亿参数），同一份数据完整过一遍，只改学习率或随机种子；基准组取自 tokenizer 对照实验',
         '', '## 1. 多种子方差（同一配置重复训练的噪声）', '',
         '| 组 | 种子 | 加权 bits/字节 | 运算平均准确率 | 最终训练 loss |', '|---|---|---|---|---|']
    for r in seeds:
        L.append(f'| {r["name"]} | {r["seed"]} | {r["bpb_weighted"]} | {r["arith_mean"]:.1%} | {(r.get("train") or {}).get("final_loss")} |')
    if s_bpb is not None:
        L += ['', f'- 加权 bits/字节：均值 {m_bpb:.4f}，σ = {s_bpb:.4f}（相对 {s_bpb / m_bpb:.2%}）；两次训练之差超过 3·√2·σ = {3 * r2 * s_bpb:.4f} 才算显著',
              f'- 运算准确率：均值 {m_ar:.1%}，σ = {s_ar:.1%}；差异超过 {3 * r2 * s_ar:.1%} 才算显著',
              '- 分来源 σ（bits/字节）：' + '，'.join(f'{s} {v:.4f}' for s, v in src_sigma.items() if v is not None),
              f'- 注意：σ 只由 {len(seeds)} 次训练估计，本身误差很大（3 个样本的标准差，真值可能相差 2 倍），用来看量级']
    else:
        L += ['', '- 完成的种子组不足 2 个，无法估计 σ']
    L += ['', '## 2. 学习率扫描', '',
          '| 峰值学习率 | 加权 bits/字节 | 与基准之差（√2·σ 的倍数） | 运算准确率 | 最终训练 loss | 裁剪生效步数占比 | update/weight 中位数 |',
          '|---|---|---|---|---|---|---|']
    base_bpb = base['bpb_weighted'] if base else None
    for r in lrs:
        tr = r.get('train') or {}
        if r.get('diverged'):
            L.append(f'| {r["peak_lr"]:g} | 发散（第 {tr.get("steps")} 步） | - | - | - | {tr.get("clip_fraction")} | {tr.get("update_ratio_median")} |')
            continue
        z = '-' if base_bpb is None or s_bpb in (None, 0) or r['group'] == 'base' else f'{(r["bpb_weighted"] - base_bpb) / (r2 * s_bpb):+.1f}'
        L.append(f'| {r["peak_lr"]:g}{"（基准）" if r["group"] == "base" else ""} | {r["bpb_weighted"]} | {z} | {r["arith_mean"]:.1%} | '
                 f'{tr.get("final_loss")} | {tr.get("clip_fraction", "-")} | {tr.get("update_ratio_median", "-")} |')
    lr_ok = [r for r in lrs if not r.get('diverged')]
    best = min(lr_ok, key=lambda r: r['bpb_weighted']) if lr_ok else None
    summary = {'tok': tok, 'sigma_bpb': s_bpb, 'sigma_arith': s_ar, 'mean_bpb': m_bpb, 'best_lr': best['peak_lr'] if best else None}
    if best:
        tried = [r['peak_lr'] for r in lr_ok]
        edge = best['peak_lr'] in (min(tried), max(tried)) and len(tried) > 1
        sig = None
        if base_bpb is not None and s_bpb:
            sig = (base_bpb - best['bpb_weighted']) >= 3 * r2 * s_bpb
        summary.update(best_lr_at_edge=edge, best_lr_significant=sig)
        L += ['', f'- 最优学习率：{best["peak_lr"]:g}（加权 bits/字节 {best["bpb_weighted"]}）'
              + ('；它就是基准值' if best['group'] == 'base' else ('；没有噪声估计或基准组，无法判断它与基准的差异是否显著' if sig is None else '；比基准的改进超过噪声，是真实差异' if sig else '；与基准的差异在噪声范围内，不能认为更好')),
              '- ' + ('最优值落在扫描范围的边界上，真正的最优可能还在范围之外，需要朝这个方向再扩一档' if edge else '最优值在扫描范围内部，两侧都更差，范围够用'),
              '- 发散或明显变差的那一侧给出了学习率的上限；裁剪生效占比高说明该学习率下训练主要靠裁剪稳住']
    ab = read_json(AB / 'results.json') or {}
    L += ['', '## 3. 回看 tokenizer 对照：v1 与 v2 的差异是否超过噪声', '']
    v2_sig = None
    if 'v1' in ab and 'v2' in ab:
        d_bpb = ab['v1']['bpb_weighted'] - ab['v2']['bpb_weighted']
        d_ar = ab['v2']['arith_mean'] - ab['v1']['arith_mean']
        L.append(f'- 加权 bits/字节：v1 {ab["v1"]["bpb_weighted"]} → v2 {ab["v2"]["bpb_weighted"]}，差 {d_bpb:+.4f}（{d_bpb / ab["v1"]["bpb_weighted"]:+.1%}）'
                 + (f'，是 √2·σ 的 {d_bpb / (r2 * s_bpb):.1f} 倍' if s_bpb else ''))
        L.append(f'- 运算准确率：v1 {ab["v1"]["arith_mean"]:.1%} → v2 {ab["v2"]["arith_mean"]:.1%}，差 {d_ar:+.1%}'
                 + (f'，是 √2·σ 的 {d_ar / (r2 * s_ar):.1f} 倍' if s_ar else ''))
        if s_bpb is not None:
            bpb_sig = d_bpb / ab['v1']['bpb_weighted'] >= 0.01 and d_bpb >= 3 * r2 * s_bpb
            ar_sig = d_ar >= 0.1 and d_ar >= 3 * r2 * (s_ar or 0)
            v2_sig = bool(bpb_sig or ar_sig)
            L.append(f'- 判定：v2 {"确实" if v2_sig else "没有被证明"}明显更好（σ 取自 {tok} 组，假定两个 tokenizer 的噪声相近）')
        summary.update(ab_bpb_diff=round(d_bpb, 4), ab_arith_diff=round(d_ar, 4))
    else:
        L.append('- 没有找到对照实验的结果文件')
    summary['v2_significant'] = v2_sig
    write_text(ABL / 'REPORT.md', '\n'.join(L) + '\n')
    write_json(ABL / 'stage1_summary.json', summary)
    bus.post('analyst', 'decision', 'report', {'action': 'report', **summary})
    state['summary'] = summary


PHASE_FN = {'wait': phase_wait, 'plan': phase_plan, 'experiments': phase_experiments, 'report': phase_report}


def run():
    ABL.mkdir(parents=True, exist_ok=True)
    bus = Bus(ABL)
    state = read_json(ABL / 'state.json', {}) or {'phase': 'wait', 'failures': 0, 'started': stamp()}
    try:
        while state['phase'] != 'done':
            check_control()
            phase = state['phase']
            heartbeat(phase)
            bus.post('decision', 'all', 'phase', {'phase': phase})
            started = time.time()
            PHASE_FN[phase](state, bus)
            state.setdefault('hours', {})[phase] = round(state.get('hours', {}).get(phase, 0) + (time.time() - started) / 3600, 3)
            state['phase'] = PHASES[PHASES.index(phase) + 1]
            state['failures'] = 0
            write_json(ABL / 'state.json', state)
        bus.post('decision', 'all', 'decision', {'decision': 'complete', 'why': '第一批消融完成，报告见 REPORT.md'})
        return 0
    except StopRequested:
        bus.post('decision', 'all', 'stopped', {})
        return 4
    except FatalError as exc:
        err = {'time': stamp(), 'phase': state.get('phase'), 'error': str(exc), 'traceback': traceback.format_exc()[-4000:]}
        write_json(ABL / 'FAILED.json', err)
        bus.post('decision', 'all', 'fatal', err)
        return 2
    except Exception as exc:  # noqa: BLE001
        tb = traceback.format_exc()
        print(tb, flush=True)
        state['failures'] = state.get('failures', 0) + 1
        state['last_error'] = {'time': stamp(), 'phase': state.get('phase'), 'error': f'{type(exc).__name__}: {exc}', 'traceback': tb[-4000:]}
        write_json(ABL / 'state.json', state)
        bus.post('decision', 'all', 'error', {'phase': state.get('phase'), 'failures': state['failures'],
                                              'error': f'{type(exc).__name__}: {str(exc)[:400]}'})
        if state['failures'] >= 4:
            write_json(ABL / 'FAILED.json', state['last_error'])
            return 2
        return 1


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'run':
        sys.exit(run())
    print(json.dumps(read_json(ABL / 'state.json', {}), ensure_ascii=False, indent=1)[:4000])
