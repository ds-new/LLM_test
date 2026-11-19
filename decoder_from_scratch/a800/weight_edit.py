"""参数空间编辑智能体：复现 Weight-Space Programming（rikiyat.github.io/blog/weight-space-programming）实验 8 的思路。

原文：不做梯度训练，只在有限的前向次数预算内诊断模型"复读/打转"（looping）的原因，发现它由少数几层 MLP
写入残差流的一个方向造成，于是从这些层 MLP 输出矩阵的 64 列里减去该方向，复读率大幅下降、各项指标提升。
原文由 LLM（Astra）读权重做诊断，没有公开代码；这里把同样的步骤写成可复现的自动流程：

1. 练习数据只来自训练集（原文用 GSM8K/MATH 的训练集），验证集 / 测试集 / 选择题只用于最后的对照评估；
2. 诊断：对练习提示做贪心续写，把"处在复读中"的位置和正常文本位置分开，逐层比较 MLP 输出（写入残差流的向量）
   的均值差，得到每层的"复读方向" d_l，并按偏移量相对该层输出幅度的大小给层排序；
3. 选列：down_proj.weight 的第 j 列就是第 j 个 FFN 神经元写入残差流的向量。
   打分 c_j = (复读时激活均值 − 正常时激活均值) × (第 j 列在 d 上的投影)，取得分最高的 k 列；
4. 编辑：W[:, j] ← W[:, j] − α · d̂ d̂ᵀ W[:, j]（α=1 即原文的"减去该方向"）。只改这 k 列，其余权重不动；
5. 预算内搜索：每个候选（层组合 × 列数 × α）在练习集上测一次复读率和 loss，只接受"复读明显下降且练习 loss
   不升高超过容差"的编辑；接受后在编辑后的模型上重新诊断，进入下一轮；
6. 留出评估：编辑前后分别测验证/测试 loss、选择题、12 个固定提示的生成重复度，以及测试集文档开头的续写复读率。

独立冒烟测试（CPU、小模型、几十秒）：python -m decoder_from_scratch.a800.weight_edit --smoke
"""

import argparse
import copy
import json
import math
import os
import sys
import time

import numpy as np
import torch

from ..generation import filter_logits
from .common import RUN_DIR, now, stamp, write_json, write_text
from .eval_agent import PROMPTS, repetition_stats
from .trainer_agent import autocast

NAME = 'weight_editor'
BOS, EOS = 2, 3


# ---------------------------------------------------------------- 数据
def sample_prompts(manifest, split, n, prompt_len, seed):
    """从各来源 split 分片里随机取文档开头（BOS 之后）的 prompt_len 个 token，各来源数量相同。"""
    rng = np.random.default_rng(seed)
    sources = [s for s in manifest['sources'] if manifest['sources'][s].get(split)]
    per = max(1, n // max(1, len(sources)))
    rows, names = [], []
    for name in sources:
        shards = [np.memmap(RUN_DIR / s['path'], dtype='<u2', mode='r') for s in manifest['sources'][name][split]]
        shards = [a for a in shards if len(a) > 4 * prompt_len]
        got, tries = 0, 0
        while shards and got < per and tries < per * 50:
            tries += 1
            arr = shards[int(rng.integers(len(shards)))]
            off = int(rng.integers(0, len(arr) - 2 * prompt_len))
            seg = np.asarray(arr[off:off + 8192])
            for p in np.flatnonzero(seg == BOS):
                cand = seg[p:p + prompt_len]
                if len(cand) == prompt_len and not (cand[1:] == EOS).any() and not (cand[1:] == BOS).any():
                    rows.append(cand.astype(np.int64))
                    names.append(name)
                    got += 1
                    break
    return torch.from_numpy(np.stack(rows)), names


def sample_chunks(manifest, split, n, seq_len, seed):
    rng = np.random.default_rng(seed)
    rows = []
    sources = [s for s in manifest['sources'] if manifest['sources'][s].get(split)]
    for i in range(n):
        entry = manifest['sources'][sources[i % len(sources)]][split]
        arr = np.memmap(RUN_DIR / entry[int(rng.integers(len(entry)))]['path'], dtype='<u2', mode='r')
        if len(arr) <= seq_len + 1:
            continue
        off = int(rng.integers(0, len(arr) - seq_len - 1))
        rows.append(np.asarray(arr[off:off + seq_len + 1]).astype(np.int64))
    return torch.from_numpy(np.stack(rows))


# ---------------------------------------------------------------- 生成与指标
@torch.no_grad()
def batch_generate(model, prompts, max_new, device, precision, greedy=True, temperature=0.8, top_p=0.9,
                   batch=64, seed=0):
    """等长 prompt 批量续写（KV Cache）。返回 [N, max_new] 的新 token（EOS 之后的内容在指标里被截掉）。"""
    model.eval()
    outs = []
    gen = torch.Generator(device=device).manual_seed(seed) if not greedy else None
    for i in range(0, len(prompts), batch):
        x = prompts[i:i + batch].to(device)
        new = []
        with autocast(device, precision):
            out = model(x, use_cache=True, last_logits_only=True)
            for _ in range(max_new):
                logits = out.logits[:, -1].float()
                if greedy:
                    nxt = logits.argmax(-1, keepdim=True)
                else:
                    probs = filter_logits(logits / temperature, 0, top_p).softmax(-1)
                    nxt = torch.multinomial(probs, 1, generator=gen)
                new.append(nxt)
                out = model(nxt, past_key_values=out.past_key_values, use_cache=True, last_logits_only=True)
        outs.append(torch.cat(new, 1).cpu())
    return torch.cat(outs, 0)


def truncate_eos(ids):
    ids = ids.tolist()
    return ids[:ids.index(EOS) + 1] if EOS in ids else ids


def loop_metrics(generated):
    """复读率：未提前结束、且最后 64 个 token 的 4-gram 重复比例 ≥ 0.5 的序列占比。"""
    loops, rep4s, distinct2, eos = [], [], [], []
    for row in generated:
        ids = truncate_eos(row)
        ended = bool(ids and ids[-1] == EOS)
        tail = ids[-64:]
        tail_rep = repetition_stats(tail)['rep4'] if len(tail) >= 16 else 0.0
        loops.append(float((not ended) and tail_rep >= 0.5))
        rep4s.append(repetition_stats(ids)['rep4'])
        bi = list(zip(ids, ids[1:]))
        distinct2.append(len(set(bi)) / max(1, len(bi)))
        eos.append(float(ended))
    m = lambda v: round(float(np.mean(v)), 4)  # noqa: E731
    return {'loop_rate': m(loops), 'rep4': m(rep4s), 'distinct2': m(distinct2), 'eos_rate': m(eos)}


@torch.no_grad()
def chunk_loss(model, chunks, device, precision, batch=8):
    model.eval()
    total = 0.0
    for i in range(0, len(chunks), batch):
        x = chunks[i:i + batch].to(device)
        with autocast(device, precision):
            total += float(model(x[:, :-1], targets=x[:, 1:]).loss) * x.size(0)
    return round(total / len(chunks), 5)


def loop_mask(full, prompt_len):
    """full: [N, T] prompt+续写。位置 t 处在复读中 ⇔ 以 t 结尾的 4-gram 在续写中更早出现过。
    返回 loop 掩码与"续写中第一次复读之前"的正常位置掩码。"""
    n, t = full.shape
    loop = torch.zeros(n, t, dtype=torch.bool)
    clean = torch.zeros(n, t, dtype=torch.bool)
    for i in range(n):
        ids = full[i].tolist()
        seen, first = set(), None
        stop = ids.index(EOS, prompt_len) if EOS in ids[prompt_len:] else t
        for p in range(prompt_len + 3, stop):
            g = tuple(ids[p - 3:p + 1])
            if g in seen:
                loop[i, p] = True
                first = p if first is None else first
            seen.add(g)
        end = first if first is not None else stop
        clean[i, prompt_len + 3:end] = True
    return loop, clean


# ---------------------------------------------------------------- 诊断
class Collector:
    """在每层 FFN 上挂钩子，按掩码累加 FFN 输出（写入残差流的向量）和 FFN 神经元激活（down_proj 的输入）。"""

    def __init__(self, model):
        self.model = model
        self.mask = None
        self.stats = {}
        self.handles = []
        for li, layer in enumerate(model.layers):
            self.handles.append(layer.ffn.register_forward_hook(self._out_hook(li)))
            self.handles.append(layer.ffn.down_proj.register_forward_pre_hook(self._act_hook(li)))

    def _acc(self, key, value):
        for tag, m in self.mask.items():
            if m.any():
                sel = value[m].float()
                s = self.stats.setdefault((key, tag), [0.0, 0, 0.0])
                s[0] = s[0] + sel.sum(0).double().cpu()
                s[1] += sel.size(0)
                s[2] += float((sel ** 2).sum())

    def _out_hook(self, li):
        def hook(module, inputs, output):
            self._acc(('out', li), output)
        return hook

    def _act_hook(self, li):
        def hook(module, inputs):
            self._acc(('act', li), inputs[0])
        return hook

    def run(self, x, masks, device, precision, batch=16):
        for i in range(0, len(x), batch):
            self.mask = {k: v[i:i + batch].to(device) for k, v in masks.items()}
            with torch.no_grad(), autocast(device, precision):
                self.model(x[i:i + batch].to(device))
        self.mask = None

    def mean(self, key, tag):
        s = self.stats.get((key, tag))
        return None if not s or s[1] == 0 else (s[0] / s[1]).float(), (0 if not s else s[1]), (0.0 if not s else s[2] / max(1, s[1]))

    def close(self):
        for h in self.handles:
            h.remove()


def diagnose(model, full, prompt_len, natural, device, precision, min_positions=2000):
    """返回每层的复读方向、层得分和神经元得分。参照组优先用"同一批续写中复读前的位置"，不够时用训练集自然文本。"""
    loop, clean = loop_mask(full, prompt_len)
    col = Collector(model)
    try:
        col.run(full, {'loop': loop, 'clean': clean}, device, precision)
        if int(clean.sum()) < min_positions:
            nat_mask = torch.zeros(natural.shape[0], natural.shape[1] - 1, dtype=torch.bool)
            nat_mask[:, 16:] = True
            col.run(natural[:, :-1], {'natural': nat_mask}, device, precision)
    finally:
        col.close()
    ref = 'clean' if int(clean.sum()) >= min_positions else 'natural'
    layers = []
    for li, layer in enumerate(model.layers):
        mu_loop, n_loop, sq = col.mean(('out', li), 'loop')
        mu_ref, n_ref, _ = col.mean(('out', li), ref)
        if mu_loop is None or mu_ref is None or n_loop < 200:
            continue
        d = mu_loop - mu_ref
        rms = math.sqrt(max(sq / mu_loop.numel(), 1e-12))
        a_loop = col.mean(('act', li), 'loop')[0]
        a_ref = col.mean(('act', li), ref)[0]
        layers.append({'layer': li, 'direction': d / (d.norm() + 1e-8), 'shift': float(d.norm()),
                       'score': float(d.norm()) / (rms * math.sqrt(mu_loop.numel()) + 1e-8),
                       'act_delta': a_loop - a_ref})
    layers.sort(key=lambda r: -r['score'])
    return {'layers': layers, 'reference': ref, 'loop_positions': int(loop.sum()), 'ref_positions':
            int(clean.sum()) if ref == 'clean' else int(natural.shape[0] * (natural.shape[1] - 17))}


def column_scores(model, info):
    w = model.layers[info['layer']].ffn.down_proj.weight.detach().float().cpu()  # [hidden, ffn]
    align = info['direction'] @ w                                               # 每列在复读方向上的投影
    return info['act_delta'] * align                                           # >0：在复读时把输出推向该方向


def apply_edit(model, layer, direction, cols, alpha):
    w = model.layers[layer].ffn.down_proj.weight
    d = direction.to(w.device, torch.float32)
    with torch.no_grad():
        sub = w[:, cols].float()
        sub -= alpha * torch.outer(d, d @ sub)
        w[:, cols] = sub.to(w.dtype)


# ---------------------------------------------------------------- 智能体
class WeightEditAgent:
    def __init__(self, config, bus, manifest, evaluator, plan, device):
        self.cfg = config.get('weight_edit', {})
        self.bus, self.manifest, self.evaluator, self.plan = bus, manifest, evaluator, plan
        self.device = torch.device(device)
        self.precision = plan.get('precision', 'bf16')
        self.passes = 0
        self.log = []

    def say(self, kind, content):
        if self.bus is not None:
            self.bus.post(NAME, 'decision', kind, content)

    def practice_measure(self, model, prompts, chunks):
        self.passes += 1
        gen = batch_generate(model, prompts, self.cfg['gen_tokens'], self.device, self.precision, greedy=True)
        m = loop_metrics(gen)
        m['loss'] = chunk_loss(model, chunks, self.device, self.precision)
        return m, gen

    @staticmethod
    def objective(m):
        return m['loop_rate'] + 0.5 * m['rep4']

    def candidates(self, diag):
        top = [r for r in diag['layers'][:self.cfg['max_layers']]]
        sets = [[top[0]]] if top else []
        if len(top) > 1:
            sets += [[top[1]], top[:2]]
        if len(top) > 2:
            sets += [top[:3]]
        out = []
        for layer_set in sets:
            for k in self.cfg['columns']:
                for alpha in self.cfg['alphas']:
                    out.append((layer_set, k, alpha))
        return out

    def search(self, model, prompts, chunks):
        base_m, base_gen = self.practice_measure(model, prompts, chunks)
        self.say('report', {'action': 'practice_baseline', **base_m})
        current, current_m, edits = base_m, base_m, []
        natural = chunks[:, :self.cfg.get('natural_len', 257)]
        full = torch.cat([prompts, base_gen], 1)
        for rnd in range(self.cfg['rounds']):
            if self.passes >= self.cfg['budget']:
                break
            diag = diagnose(model, full, prompts.shape[1], natural, self.device, self.precision)
            self.say('report', {'action': 'diagnosis', 'round': rnd + 1, 'reference': diag['reference'],
                                'loop_positions': diag['loop_positions'], 'ref_positions': diag['ref_positions'],
                                'top_layers': [(r['layer'], round(r['score'], 4)) for r in diag['layers'][:6]]})
            if not diag['layers']:
                break
            best = None
            snapshot = {li: model.layers[li].ffn.down_proj.weight.detach().clone() for li in range(len(model.layers))}
            for layer_set, k, alpha in self.candidates(diag):
                if self.passes >= self.cfg['budget']:
                    break
                planned = []
                for info in layer_set:
                    scores = column_scores(model, info)
                    cols = torch.topk(scores, k).indices
                    cols = cols[scores[cols] > 0]
                    if len(cols):
                        planned.append((info['layer'], info['direction'], cols.tolist()))
                if not planned:
                    continue
                for li, d, cols in planned:
                    apply_edit(model, li, d, cols, alpha)
                m, gen = self.practice_measure(model, prompts, chunks)
                for li, _, _ in planned:  # 恢复到本轮起点
                    model.layers[li].ffn.down_proj.weight.data.copy_(snapshot[li])
                ok = m['loss'] <= current_m['loss'] * (1 + self.cfg['loss_tolerance'])
                row = {'round': rnd + 1, 'layers': [p[0] for p in planned], 'columns': k, 'alpha': alpha,
                       'weights_changed': sum(len(p[2]) for p in planned) * model.config.hidden_size,
                       **m, 'loss_ok': ok, 'pass': self.passes}
                self.log.append(row)
                self.say('progress', {'action': 'candidate', **row})
                if ok and (best is None or self.objective(m) < self.objective(best[1])):
                    best = (planned, m, gen, alpha, row)
            if best is None or self.objective(best[1]) > self.objective(current_m) * (1 - self.cfg['min_gain']):
                self.say('decision', {'decision': 'stop_search', 'round': rnd + 1,
                                      'why': '本轮没有候选在练习集上把复读目标降低足够多且不损害 loss'})
                break
            planned, m, gen, alpha, row = best
            for li, d, cols in planned:
                apply_edit(model, li, d, cols, alpha)
                edits.append({'layer': li, 'columns': cols, 'alpha': alpha, 'direction': d.tolist()})
            current_m = m
            full = torch.cat([prompts, gen], 1)
            self.say('decision', {'decision': 'accept_edit', 'round': rnd + 1, 'layers': row['layers'],
                                  'columns': row['columns'], 'alpha': alpha, 'weights_changed': row['weights_changed'],
                                  'why': f'练习集复读率 {current["loop_rate"]}→{m["loop_rate"]}，rep4 {current["rep4"]}→{m["rep4"]}，'
                                         f'loss {current["loss"]}→{m["loss"]}'})
            current = m
        return base_m, current_m, edits

    def heldout(self, model, prompts):
        """留出评估：与训练时的评估同口径，外加测试集文档开头的续写复读率（贪心 + 采样）。"""
        ev, plan = self.evaluator, self.plan
        weights = plan['eval_weights']
        val = ev.loss_by_source(model, 'val', plan['seq_len'], self.precision)
        test = ev.loss_by_source(model, 'test', plan['seq_len'], self.precision)
        wsum = sum(weights.get(n, 0) for n in val) or 1.0
        cloze_acc, cloze_by_cat, _ = ev.cloze(model, self.precision)
        gen_summary, samples = ev.generation_probe(model)
        g = loop_metrics(batch_generate(model, prompts, self.cfg['gen_tokens'], self.device, self.precision, greedy=True))
        s = loop_metrics(batch_generate(model, prompts, self.cfg['gen_tokens'], self.device, self.precision,
                                        greedy=False, seed=7))
        return {'val_loss': round(sum(val[n] * weights.get(n, 0) for n in val) / wsum, 4),
                'test_loss': round(sum(test[n] * weights.get(n, 0) for n in test) / wsum, 4),
                'val_by_source': val, 'test_by_source': test, 'cloze_acc': cloze_acc, 'cloze_by_cat': cloze_by_cat,
                'probe_sample_rep4': gen_summary['sample_rep4'], 'probe_greedy_rep4': gen_summary['greedy_rep4'],
                'heldout_greedy': g, 'heldout_sample': s}, samples

    def run(self, model):
        started = now()
        c = self.cfg
        prompts, _ = sample_prompts(self.manifest, 'train', c['practice_prompts'], c['prompt_len'], seed=11)
        chunks = sample_chunks(self.manifest, 'train', c['practice_chunks'], self.plan['seq_len'], seed=12)
        held_prompts, _ = sample_prompts(self.manifest, 'test', c['heldout_prompts'], c['prompt_len'], seed=13)
        self.say('progress', {'action': 'start', 'practice_prompts': len(prompts), 'practice_chunks': len(chunks),
                              'heldout_prompts': len(held_prompts), 'budget': c['budget']})
        base_weights = {li: l.ffn.down_proj.weight.detach().clone() for li, l in enumerate(model.layers)}
        before, before_samples = self.heldout(model, held_prompts)
        practice_before, practice_after, edits = self.search(model, prompts, chunks)
        after, after_samples = (before, before_samples) if not edits else self.heldout(model, held_prompts)
        edited_state = None
        if edits:
            edited_state = copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})
        for li, w in base_weights.items():  # 训练好的原模型保持不变
            model.layers[li].ffn.down_proj.weight.data.copy_(w)
        result = {'time': stamp(), 'minutes': round((now() - started) / 60, 1), 'passes_used': self.passes,
                  'budget': c['budget'], 'practice_before': practice_before, 'practice_after': practice_after,
                  'heldout_before': before, 'heldout_after': after,
                  'edits': [{k: v for k, v in e.items() if k != 'direction'} | {'n_columns': len(e['columns'])} for e in edits],
                  'weights_changed': sum(len(e['columns']) for e in edits) * model.config.hidden_size,
                  'total_weights': sum(p.numel() for p in model.parameters()), 'candidates': self.log,
                  'verdict': verdict(before, after, bool(edits))}
        return result, edited_state, edits, before_samples, after_samples


def verdict(before, after, edited):
    if not edited:
        return 'no_edit'
    loop_down = after['heldout_greedy']['loop_rate'] < before['heldout_greedy']['loop_rate']
    loss_ok = after['test_loss'] <= before['test_loss'] * 1.005
    cloze_ok = after['cloze_acc'] >= before['cloze_acc'] - 0.03
    return 'improved' if loop_down and loss_ok and cloze_ok else ('mixed' if loop_down else 'not_improved')


def render_report(result, before_samples, after_samples):
    b, a = result['heldout_before'], result['heldout_after']
    row = lambda name, x, y, fmt='{:.4f}': f'| {name} | {fmt.format(x)} | {fmt.format(y)} | {y - x:+.4f} |'  # noqa: E731
    L = ['# 参数空间编辑实验（Weight-Space Programming 实验 8 的复现）', '',
         f'- 时间：{result["time"]}，用时 {result["minutes"]} 分钟，练习前向预算 {result["passes_used"]}/{result["budget"]}',
         f'- 修改权重：{result["weights_changed"]:,} / {result["total_weights"]:,}'
         f'（{result["weights_changed"] / result["total_weights"]:.4%}），结论：**{result["verdict"]}**', '',
         '## 编辑内容', '']
    if result['edits']:
        L += [f'- 第 {e["layer"]} 层 FFN down_proj，{e["n_columns"]} 列，α={e["alpha"]}' for e in result['edits']]
    else:
        L += ['- 没有找到在练习集上既减少复读又不损害 loss 的编辑，模型保持不变']
    L += ['', '## 留出评估（编辑前 → 编辑后）', '', '| 指标 | 编辑前 | 编辑后 | 变化 |', '|---|---|---|---|',
          row('验证 loss', b['val_loss'], a['val_loss']), row('测试 loss', b['test_loss'], a['test_loss']),
          row('选择题准确率', b['cloze_acc'], a['cloze_acc']),
          row('固定提示 采样 rep4', b['probe_sample_rep4'], a['probe_sample_rep4']),
          row('固定提示 贪心 rep4', b['probe_greedy_rep4'], a['probe_greedy_rep4'])]
    for mode in ('greedy', 'sample'):
        for k in ('loop_rate', 'rep4', 'distinct2', 'eos_rate'):
            L.append(row(f'测试集续写（{"贪心" if mode == "greedy" else "采样"}）{k}', b[f'heldout_{mode}'][k], a[f'heldout_{mode}'][k]))
    L += ['', '分来源测试 loss：', '', '| 来源 | 编辑前 | 编辑后 |', '|---|---|---|']
    L += [f'| {n} | {b["test_by_source"][n]} | {a["test_by_source"].get(n)} |' for n in b['test_by_source']]
    p0, p1 = result['practice_before'], result['practice_after']
    L += ['', f'练习集（训练集文档开头，贪心）：复读率 {p0["loop_rate"]} → {p1["loop_rate"]}，rep4 {p0["rep4"]} → {p1["rep4"]}，'
          f'loss {p0["loss"]} → {p1["loss"]}', '', '## 候选记录', '',
          '| 轮 | 层 | 列数 | α | 练习复读率 | rep4 | loss | loss 合格 |', '|---|---|---|---|---|---|---|---|']
    L += [f'| {c["round"]} | {c["layers"]} | {c["columns"]} | {c["alpha"]} | {c["loop_rate"]} | {c["rep4"]} | {c["loss"]} | '
          f'{"是" if c["loss_ok"] else "否"} |' for c in result['candidates']]
    L += ['', '## 贪心生成对照（12 个固定提示中的贪心项）', '']
    for x, y in zip(before_samples, after_samples):
        if x['mode'] == 'greedy':
            L += [f'- **{x["prompt"].strip()[:40]}**', f'  - 编辑前：{x["completion"][:160].strip()}',
                  f'  - 编辑后：{y["completion"][:160].strip()}']
    L += ['', '说明：练习数据只来自训练集；验证/测试/选择题/固定提示只用于这里的对照，从未参与编辑的选择。']
    return '\n'.join(L) + '\n'


def run_weight_edit(config, bus, manifest, evaluator, trainer, device):
    """决策智能体调用的入口：跑实验、保存 edited.pt（如有编辑）和报告，返回结果摘要。"""
    agent = WeightEditAgent(config, bus, manifest, evaluator, trainer.plan, device)
    result, edited_state, edits, bs, as_ = agent.run(trainer.model)
    out = RUN_DIR / 'weight_edit'
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / 'result.json', result)
    write_json(out / 'samples.json', {'before': bs, 'after': as_})
    write_text(out / 'REPORT.md', render_report(result, bs, as_))
    if edited_state is not None:
        torch.save({'model': edited_state, 'config': trainer.model.config.to_dict(), 'tokenizer': trainer.tokenizer_dict,
                    'step': trainer.step, 'global_step': trainer.step, 'tokens_seen': trainer.tokens_seen,
                    'normalize_punct': True, 'weight_edit': [{k: v for k, v in e.items()} for e in edits]},
                   RUN_DIR / 'model' / 'edited.pt')
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    return result


DEFAULTS = {'enabled': True, 'practice_prompts': 384, 'practice_chunks': 48, 'heldout_prompts': 256, 'prompt_len': 48,
            'gen_tokens': 128, 'budget': 50, 'rounds': 3, 'max_layers': 3, 'columns': [32, 64, 128], 'alphas': [1.0, 0.5],
            'loss_tolerance': 0.005, 'min_gain': 0.05}


# ---------------------------------------------------------------- 冒烟测试
def smoke():
    """CPU 上的小模型 + 合成数据，验证整条流程能跑通（不检验效果）。"""
    import tempfile
    from ..config import ModelConfig
    from ..model import DecoderLM
    from .eval_agent import EvalAgent

    class Tok:
        def encode(self, text, add_bos=False):
            return ([BOS] if add_bos else []) + [5 + (ord(ch) % 250) for ch in text[:40]]

        def decode(self, ids):
            return ''.join(chr(0x4e00 + int(i)) for i in ids if int(i) >= 5)

    rng = np.random.default_rng(0)
    root = RUN_DIR
    (root / 'tok').mkdir(parents=True, exist_ok=True)
    manifest = {'sources': {}}
    for name in ('a', 'b'):
        entry = {}
        for split, ndoc in (('train', 300), ('val', 60), ('test', 60)):
            docs = []
            for _ in range(ndoc):
                motif = rng.integers(5, 255, size=int(rng.integers(3, 8)))
                body = np.concatenate([np.tile(motif, 30), rng.integers(5, 255, size=60)])
                docs.append(np.concatenate([[BOS], body, [EOS]]))
            path = root / 'tok' / f'{name}.{split}.bin'
            np.concatenate(docs).astype('<u2').tofile(path)
            entry[split] = [{'path': str(path.relative_to(root)), 'tokens': int(sum(map(len, docs)))}]
        manifest['sources'][name] = entry
    cfg = ModelConfig(vocab_size=256, hidden_size=64, num_layers=3, num_heads=4, num_kv_heads=2,
                      intermediate_size=128, max_seq_len=256)
    model = DecoderLM(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    tr = np.memmap(root / manifest['sources']['a']['train'][0]['path'], dtype='<u2', mode='r')
    for step in range(60):  # 训练几十步，让模型学会复读 motif
        idx = rng.integers(0, len(tr) - 129, size=16)
        x = torch.from_numpy(np.stack([tr[i:i + 129] for i in idx]).astype(np.int64))
        loss = model(x[:, :-1], targets=x[:, 1:]).loss
        opt.zero_grad()
        loss.backward()
        opt.step()
    plan = {'precision': 'fp32', 'seq_len': 128, 'eval_weights': {'a': 0.5, 'b': 0.5}}
    config = {'eval': {'val_chunks_per_source': 4, 'max_new_tokens': 12, 'top_p': 0.9, 'temperature': 0.8},
              'weight_edit': dict(DEFAULTS, practice_prompts=32, practice_chunks=8, heldout_prompts=16, prompt_len=16,
                                  gen_tokens=32, budget=6, columns=[8, 16], alphas=[1.0], rounds=2, min_gain=0.0)}
    ev = EvalAgent(config, None, manifest, Tok(), 'cpu')

    class T:
        pass
    t = T()
    t.model, t.plan, t.step, t.tokens_seen, t.tokenizer_dict = model, plan, 60, 0, {'type': 'smoke'}
    (root / 'model').mkdir(exist_ok=True)
    started = time.time()
    result = run_weight_edit(config, None, manifest, ev, t, torch.device('cpu'))
    print(json.dumps({k: result[k] for k in ('passes_used', 'weights_changed', 'verdict', 'practice_before', 'practice_after')},
                     ensure_ascii=False, indent=1))
    print('candidates', len(result['candidates']), 'seconds', round(time.time() - started, 1))
    print((RUN_DIR / 'weight_edit' / 'REPORT.md').read_text()[:1500])


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--smoke', action='store_true')
    args = p.parse_args()
    if args.smoke:
        smoke()
