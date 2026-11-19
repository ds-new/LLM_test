"""Decoder 机理分析与推理侧实验（plan/08 第 2、5 部分），对象是 v1 final.pt（0.48B，40 层，16 头 / 4 组 KV）。

用法（工程根目录）：/root/a800_venv/bin/python -m decoder_from_scratch.a800.mech <任务>... [--quick]
  sinks      注意力汇（attention sink）与超大激活；滑动窗口去掉 / 保留开头 token 的 loss 对照
  lens       logit lens：每层残差流直接解码的 loss、与最终预测的一致率、KL，分来源
  layers     逐层消融：去掉整层 / 只去注意力 / 只去 MLP 的 loss 增量；按重要性从低到高累计剪层
  heads      逐头消融（输出置零）的 loss 增量
  induction  随机 token 重复序列：上下文学习、induction head 与前一 token head 打分、消融 induction head
  probes     线性探针：数据来源、绝对位置、两位数加法（个位 / 进位 / 十位）
  trace      因果追踪：事实性提示的主语加噪后，逐层逐位置恢复干净激活看答案概率回升（残差流 / MLP / 注意力）
  decode     解码策略对复读、多样性、困惑度的影响；重复片段的自我强化
  quant      仅权重量化（int8/4/3/2）与激活量化的 loss 变化；逐层、逐矩阵类型的敏感度
结果：/root/autodl-tmp/a800_mech/<任务>.json，图：figs/*.png（--quick 时写到 quick/ 子目录，只做冒烟）。
和路线 A 训练共用 GPU：本进程显存上限默认约 7GB（MECH_GPU_FRACTION），训练进程不受影响。
"""

import argparse
from collections import Counter, defaultdict
import json
import math
import os
import re
import time
import types
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from ..config import ModelConfig
from ..model import DecoderLM

OUT = Path(os.environ.get('MECH_OUT', '/root/autodl-tmp/a800_mech'))
FIGS = OUT / 'figs'
V1 = Path('/root/autodl-tmp/a800_pretrain_v1')
CKPTS = {'v1': V1 / 'model' / 'final.pt', 'cpt': Path('/root/a800_scratch/cpt_stage1/latest.pt')}
SOURCES = ['zh_cci3', 'zh_fwedu', 'zh_industry', 'en_fwedu', 'en_cosmo', 'code', 'math']
EVAL_WEIGHTS = {'zh_cci3': 0.28, 'zh_fwedu': 0.14, 'zh_industry': 0.13, 'en_fwedu': 0.18, 'en_cosmo': 0.07,
                'code': 0.12, 'math': 0.08}
LIN = ('attention.q_proj', 'attention.k_proj', 'attention.v_proj', 'attention.o_proj',
       'ffn.gate_proj', 'ffn.up_proj', 'ffn.down_proj')
BOS, EOS = 2, 3
DEV = torch.device('cuda')
QUICK = False


def log(msg):
    print(time.strftime('%H:%M:%S'), msg, flush=True)


def n_(full, quick):
    return quick if QUICK else full


def save(name, obj):
    (OUT / f'{name}.json').write_text(json.dumps(obj, ensure_ascii=False, indent=1), encoding='utf-8')


def plt():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as p
    return p


def mean(xs):
    xs = list(xs)
    return float(sum(xs) / len(xs)) if xs else float('nan')


# ---------------- 模型、分词器、数据 ----------------
def load_model(which='v1'):
    ck = torch.load(CKPTS[which], map_location='cpu', weights_only=False, mmap=True)
    cfg = dict(ck['config'])
    cfg['attention_backend'] = 'sdpa'
    model = DecoderLM(ModelConfig(**cfg))
    model.load_state_dict(ck['model'])
    del ck
    model = model.to(DEV, dtype=torch.bfloat16).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def load_tokenizer():
    from .data_agent import load_bbpe
    return load_bbpe(V1 / 'tokenizer' / 'bbpe.json')


def norm_text(s):
    from .cleaning import normalize_zh_punct
    return normalize_zh_punct(s)


_MANIFEST = {}


def manifest():
    if not _MANIFEST:
        _MANIFEST.update(json.loads((V1 / 'data' / 'tokens' / 'manifest.json').read_text()))
    return _MANIFEST


def val_rows(per_source, seq_len=1024, split='val'):
    """与评估智能体同样的取法：每个来源在 val 分片上均匀取 per_source 个固定块（per_source=32 即 v1 的验证口径）。"""
    out = {}
    for name in SOURCES:
        shards = [np.memmap(V1 / s['path'], dtype='<u2', mode='r') for s in manifest()['sources'][name][split]]
        total = sum((len(a) - 1) // seq_len for a in shards)
        wanted = set(np.unique(np.linspace(0, total - 1, num=min(per_source, total)).astype(np.int64)).tolist())
        rows, idx = [], 0
        for arr in shards:
            for c in range((len(arr) - 1) // seq_len):
                if idx in wanted:
                    rows.append(arr[c * seq_len:c * seq_len + seq_len + 1].astype(np.int64))
                idx += 1
        out[name] = torch.from_numpy(np.stack(rows))
    return out


def doc_starts(name, need, count, split='val'):
    """文档开头（BOS 起）长度为 need 的片段，中间不跨文档。"""
    res = []
    for s in manifest()['sources'][name][split]:
        arr = np.memmap(V1 / s['path'], dtype='<u2', mode='r')
        for p in np.flatnonzero(arr == BOS):
            seg = np.asarray(arr[p:p + need])
            if len(seg) == need and not np.isin(seg[1:], (BOS, EOS)).any():
                res.append(seg.astype(np.int64))
                if len(res) >= count:
                    return torch.from_numpy(np.stack(res))
    return torch.from_numpy(np.stack(res)) if res else None


@torch.no_grad()
def token_losses(model, x):
    """x [B, T+1]（CPU 或 GPU）→ 每个位置预测下一个 token 的 loss [B, T]（float32，GPU）。"""
    x = x.to(DEV)
    logits = model(x[:, :-1]).logits
    return F.cross_entropy(logits.float().transpose(1, 2), x[:, 1:], reduction='none')


def source_losses(model, rows, batch=4, positions=None):
    out = {}
    for name, x in rows.items():
        tot, n = 0.0, 0
        for i in range(0, len(x), batch):
            l = token_losses(model, x[i:i + batch])
            if positions is not None:
                l = l[:, positions]
            tot += float(l.sum())
            n += l.numel()
        out[name] = tot / n
    return out


def weighted(losses):
    names = [n for n in losses if n in EVAL_WEIGHTS]
    return sum(losses[n] * EVAL_WEIGHTS[n] for n in names) / sum(EVAL_WEIGHTS[n] for n in names)


def wloss(model, rows, **kw):
    l = source_losses(model, rows, **kw)
    l['weighted'] = weighted(l)
    return l


# ---------------- 补丁：在不改模型文件的前提下读写内部状态 ----------------
class Patch:
    """替换每层注意力和 block 的 forward。默认行为与原模型完全一致；按需：附加注意力掩码、抓注意力权重、
    把某些头的输出置零、跳过某层或子层、读写残差流和子层输出（logit lens、探针、因果追踪都靠它）。"""

    def __init__(self, model):
        self.model = model
        self.reset()
        for i, layer in enumerate(model.layers):
            layer.attention.forward = types.MethodType(self._attn(i), layer.attention)
            layer.forward = types.MethodType(self._block(i), layer)
        emb, patch = model.embedding, self
        orig = emb.forward

        def emb_forward(_self, ids):
            x = orig(ids)
            return patch.emb_hook(x) if patch.emb_hook is not None else x
        emb.forward = types.MethodType(emb_forward, emb)

    def reset(self):
        self.extra_mask = None   # fn(qpos, kpos) -> bool[T, S]，True 表示允许看
        self.capture = None      # fn(layer, weights[B, H, T, S] float32)
        self.head_mask = {}      # layer -> tensor[H]，0 表示把该头的输出置零
        self.skip = {}           # layer -> 'all' | 'attn' | 'mlp'
        self.hook = None         # fn(layer, kind in {'attn', 'mlp', 'resid'}, tensor) -> tensor
        self.emb_hook = None     # fn(tensor) -> tensor，作用在词嵌入输出上

    def _attn(self, i):
        patch = self

        def forward(self, x, past=None, use_cache=False):
            cfg = self.config
            b, t, _ = x.shape
            q = self.q_proj(x).view(b, t, cfg.num_heads, cfg.head_dim).transpose(1, 2)
            k = self.k_proj(x).view(b, t, cfg.num_kv_heads, cfg.head_dim).transpose(1, 2)
            v = self.v_proj(x).view(b, t, cfg.num_kv_heads, cfg.head_dim).transpose(1, 2)
            past_len = 0 if past is None else past[0].size(2)
            q, k = self.rope(q, past_len), self.rope(k, past_len)
            if past is not None:
                k = torch.cat((past[0], k), dim=2)
                v = torch.cat((past[1], v), dim=2)
            present = (k, v) if use_cache else None
            groups = cfg.num_heads // cfg.num_kv_heads
            k = k.repeat_interleave(groups, dim=1)
            v = v.repeat_interleave(groups, dim=1)
            special = patch.extra_mask is not None or patch.capture is not None
            if not special and past_len == 0:
                out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            elif not special and t == 1:
                out = F.scaled_dot_product_attention(q, k, v)
            else:
                qpos = past_len + torch.arange(t, device=x.device)
                kpos = torch.arange(k.size(2), device=x.device)
                mask = kpos[None, :] <= qpos[:, None]
                if patch.extra_mask is not None:
                    mask = mask & patch.extra_mask(qpos, kpos)
                if patch.capture is not None:
                    scores = (q.float() @ k.float().transpose(-1, -2)) / math.sqrt(cfg.head_dim)
                    w = scores.masked_fill(~mask, float('-inf')).softmax(dim=-1)
                    patch.capture(i, w)
                    out = (w @ v.float()).to(v.dtype)
                else:
                    out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
            hm = patch.head_mask.get(i)
            if hm is not None:
                out = out * hm.to(device=out.device, dtype=out.dtype).view(1, -1, 1, 1)
            out = out.transpose(1, 2).contiguous().view(b, t, cfg.hidden_size)
            return self.o_proj(out), present
        return forward

    def _block(self, i):
        patch = self

        def forward(self, x, past=None, use_cache=False):
            mode = patch.skip.get(i)
            if mode == 'all':
                if patch.hook is not None:
                    x = patch.hook(i, 'resid', x)
                return x, None
            attended, present = self.attention(self.attn_norm(x), past, use_cache)
            if patch.hook is not None:
                attended = patch.hook(i, 'attn', attended)
            if mode != 'attn':
                x = x + attended
            m = self.ffn(self.ffn_norm(x))
            if patch.hook is not None:
                m = patch.hook(i, 'mlp', m)
            if mode != 'mlp':
                x = x + m
            if patch.hook is not None:
                x = patch.hook(i, 'resid', x)
            return x, present
        return forward


def grab_states(patch):
    """返回列表并挂上钩子：之后一次前向会依次收集词嵌入输出和每层输出（共 L+1 个残差流状态）。"""
    states = []
    patch.emb_hook = lambda t: (states.append(t), t)[1]
    patch.hook = lambda i, kind, t: (states.append(t), t)[1] if kind == 'resid' else t
    return states


# ---------------- 第 2 部分：机理分析 ----------------
@torch.no_grad()
def task_sinks(model, patch, tok):
    cfg = model.config
    L, H = cfg.num_layers, cfg.num_heads
    seqs = torch.cat([r for r in val_rows(n_(4, 1)).values()])[:, :-1]
    acc = {k: torch.zeros(L, H) for k in ('first', 'prev', 'self', 'entropy')}

    def cap(i, w):
        idx = torch.arange(16, w.size(2), device=w.device)
        acc['first'][i] += w[:, :, 16:, 0].mean(2).sum(0).cpu()
        acc['prev'][i] += w[:, :, idx, idx - 1].mean(2).sum(0).cpu()
        acc['self'][i] += w[:, :, idx, idx].mean(2).sum(0).cpu()
        ws = w[:, :, 16:]
        acc['entropy'][i] += (-(ws * ws.clamp_min(1e-12).log()).sum(-1)).mean(2).sum(0).cpu()

    patch.capture = cap
    for i in range(0, len(seqs), 2):
        model(seqs[i:i + 2].to(DEV))
    patch.reset()
    attn = {k: (v / len(seqs)).tolist() for k, v in acc.items()}

    # 残差流里的超大激活：每层每条序列记录绝对值最大的那个元素（维度、位置、token）和中位数
    recs = [[] for _ in range(L + 1)]
    cur = {}

    def note(li, t):
        a = t.float().abs()
        for b in range(a.size(0)):
            flat = a[b].flatten()
            v, j = flat.max(0)
            pos, dim = divmod(int(j), a.size(-1))
            recs[li].append({'value': float(v), 'dim': dim, 'pos': pos, 'token': int(cur['x'][b, pos]),
                             'median': float(flat[::7].median())})
        return t

    patch.emb_hook = lambda t: note(0, t)
    patch.hook = lambda i, kind, t: note(i + 1, t) if kind == 'resid' else t
    for i in range(0, len(seqs), 4):
        cur['x'] = seqs[i:i + 4]
        model(cur['x'].to(DEV))
    patch.reset()
    massive = []
    for li, rs in enumerate(recs):
        massive.append({'layer': li, 'top_mean': mean(r['value'] for r in rs), 'top_max': max(r['value'] for r in rs),
                        'median_mean': mean(r['median'] for r in rs), 'at_pos0': mean(r['pos'] == 0 for r in rs),
                        'dims': Counter(r['dim'] for r in rs).most_common(3),
                        'tokens': [(t, tok.decode([t]), c) for t, c in Counter(r['token'] for r in rs).most_common(3)]})

    # 去掉开头 token 会怎样：在 512–1023 位置上比较不同的注意力范围（StreamingLLM 的对照）
    rows8 = val_rows(n_(8, 2))
    tail = slice(512, 1024)
    masks = {
        'full': None,
        'window256': lambda q, k: (q[:, None] - k[None, :]) < 256,
        'window256+first4': lambda q, k: ((q[:, None] - k[None, :]) < 256) | (k[None, :] < 4),
        'mask_first': lambda q, k: (k[None, :] > 0) | (q[:, None] == 0),
    }
    window = {}
    for name, fn in masks.items():
        patch.extra_mask = fn
        window[name] = wloss(model, rows8, positions=tail)
        log(f'sinks {name}: {window[name]["weighted"]:.4f}')
    patch.reset()
    save('sinks', {'attention': attn, 'massive': massive, 'window': window})

    p = plt()
    fig, ax = p.subplots(1, 2, figsize=(13, 5))
    im = ax[0].imshow(np.array(attn['first']), aspect='auto', cmap='viridis', vmin=0, vmax=1)
    ax[0].set_xlabel('head')
    ax[0].set_ylabel('layer')
    ax[0].set_title('attention mass on the first token')
    fig.colorbar(im, ax=ax[0])
    for k in ('first', 'prev', 'self'):
        ax[1].plot(np.array(attn[k]).mean(1), label=k)
    ax[1].set_xlabel('layer')
    ax[1].set_ylabel('mean attention mass (over heads)')
    ax[1].legend()
    fig.tight_layout()
    fig.savefig(FIGS / 'sinks_attention.png', dpi=120)
    p.close(fig)
    fig, ax = p.subplots(figsize=(7, 4))
    ax.semilogy([m['top_mean'] for m in massive], label='max |x| per sequence (mean)')
    ax.semilogy([m['median_mean'] for m in massive], label='median |x|')
    ax.set_xlabel('layer (0 = embedding)')
    ax.legend()
    ax.set_title('residual stream magnitude')
    fig.tight_layout()
    fig.savefig(FIGS / 'massive_activations.png', dpi=120)
    p.close(fig)


@torch.no_grad()
def task_lens(model, patch, tok):
    L = model.config.num_layers
    rows = val_rows(n_(12, 2))
    res, examples = {}, {}
    show = (0, 8, 16, 24, 30, 34, 37, 40)
    for name in SOURCES:
        loss_sum, agree, kl = torch.zeros(L + 1), torch.zeros(L + 1), torch.zeros(L + 1)
        n = 0
        for i in range(0, len(rows[name]), 2):
            x = rows[name][i:i + 2].to(DEV)
            states = grab_states(patch)
            final = model(x[:, :-1]).logits.float()
            patch.reset()
            tgt = x[:, 1:]
            flp = final.log_softmax(-1)
            fp = flp.exp()
            ftop = final.argmax(-1)
            del final
            tops = []
            for li, h in enumerate(states):
                lp = model.lm_head(model.final_norm(h)).float().log_softmax(-1)
                loss_sum[li] += float(F.nll_loss(lp.transpose(1, 2), tgt, reduction='sum'))
                top = lp.argmax(-1)
                agree[li] += float((top == ftop).sum())
                kl[li] += float((fp * (flp - lp)).sum())
                if i == 0:
                    tops.append(top[0, 200:212].tolist())
                del lp
            if tops:
                examples[name] = {'input': [tok.decode([t]) for t in x[0, 200:212].tolist()],
                                  'target': [tok.decode([t]) for t in x[0, 201:213].tolist()],
                                  'layers': {li: [tok.decode([t]) for t in tops[li]] for li in show if li <= L}}
            n += tgt.numel()
            del states, flp, fp
        res[name] = {'loss': (loss_sum / n).tolist(), 'agree': (agree / n).tolist(), 'kl': (kl / n).tolist()}
        log(f'lens {name}: final {res[name]["loss"][-1]:.4f}, mid {res[name]["loss"][L // 2]:.3f}')
    w = {k: [sum(res[s][k][li] * EVAL_WEIGHTS[s] for s in SOURCES) for li in range(L + 1)] for k in ('loss', 'agree', 'kl')}
    save('lens', {'by_source': res, 'weighted': w, 'examples': examples})
    p = plt()
    fig, ax = p.subplots(1, 2, figsize=(13, 5))
    for name in SOURCES:
        ax[0].plot(res[name]['loss'], label=name)
        ax[1].plot(res[name]['agree'], label=name)
    ax[0].set_xlabel('layer (0 = embedding)')
    ax[0].set_ylabel('cross-entropy when decoding this layer')
    ax[0].set_ylim(0, 12)
    ax[1].set_xlabel('layer')
    ax[1].set_ylabel('top-1 agreement with final layer')
    ax[0].legend()
    fig.tight_layout()
    fig.savefig(FIGS / 'logit_lens.png', dpi=120)
    p.close(fig)


@torch.no_grad()
def task_layers(model, patch, tok):
    L = model.config.num_layers
    rows = val_rows(n_(8, 1))
    res = {'base': wloss(model, rows), 'all': [], 'attn': [], 'mlp': []}
    for i in range(L):
        for mode in ('all', 'attn', 'mlp'):
            patch.skip = {i: mode}
            res[mode].append(wloss(model, rows))
        log(f'layer {i}: all +{res["all"][-1]["weighted"] - res["base"]["weighted"]:.4f}')
    patch.reset()
    order = sorted(range(L), key=lambda i: res['all'][i]['weighted'])
    prune = []
    for k in (1, 2, 4, 6, 8, 12):
        patch.skip = {i: 'all' for i in order[:k]}
        prune.append({'k': k, 'layers': order[:k], **wloss(model, rows)})
    patch.reset()
    res['prune'] = prune
    save('layers', res)
    p = plt()
    fig, ax = p.subplots(1, 2, figsize=(13, 4.5))
    b = res['base']['weighted']
    for mode in ('all', 'attn', 'mlp'):
        ax[0].semilogy([max(1e-4, r['weighted'] - b) for r in res[mode]], label=f'remove {mode}')
    ax[0].set_xlabel('layer')
    ax[0].set_ylabel('loss increase (log)')
    ax[0].legend()
    ax[1].plot([0] + [r['k'] for r in prune], [0] + [r['weighted'] - b for r in prune], marker='o')
    ax[1].set_xlabel('number of least-important layers removed')
    ax[1].set_ylabel('loss increase')
    fig.tight_layout()
    fig.savefig(FIGS / 'layer_ablation.png', dpi=120)
    p.close(fig)


@torch.no_grad()
def task_heads(model, patch, tok):
    L, H = model.config.num_layers, model.config.num_heads
    rows = val_rows(n_(2, 1))  # 每个来源 2 块：640 次前向和训练共用 GPU，取少一点以减少对训练的拖慢
    base = wloss(model, rows)['weighted']
    delta = torch.zeros(L, H)
    for i in range(L):
        for h in range(H):
            m = torch.ones(H)
            m[h] = 0
            patch.head_mask = {i: m}
            delta[i, h] = wloss(model, rows)['weighted'] - base
        log(f'heads layer {i}: max +{float(delta[i].max()):.4f}')
    patch.reset()
    order = delta.flatten().argsort(descending=True)[:20].tolist()
    save('heads', {'base': base, 'delta': delta.tolist(),
                   'top': [{'layer': j // H, 'head': j % H, 'delta': float(delta.flatten()[j])} for j in order]})
    p = plt()
    fig, ax = p.subplots(figsize=(7, 8))
    im = ax.imshow(delta.numpy(), aspect='auto', cmap='magma')
    ax.set_xlabel('head')
    ax.set_ylabel('layer')
    ax.set_title('loss increase when one head is zeroed')
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(FIGS / 'head_ablation.png', dpi=120)
    p.close(fig)


def pos_curve(model, rows, batch=4):
    tot, n = torch.zeros(1024, device=DEV), 0
    for x in rows.values():
        for i in range(0, len(x), batch):
            l = token_losses(model, x[i:i + batch])
            tot += l.sum(0)
            n += l.size(0)
    return (tot / n).cpu()


@torch.no_grad()
def task_induction(model, patch, tok):
    L, H = model.config.num_layers, model.config.num_heads
    g = torch.Generator().manual_seed(0)
    n, half = n_(32, 4), 256
    r = torch.randint(300, model.config.vocab_size, (n, half), generator=g)
    seq = torch.cat([torch.full((n, 1), BOS), r, r], 1)  # [n, 1 + 2*half]

    def rep_losses():
        return torch.cat([token_losses(model, seq[i:i + 8]).cpu() for i in range(0, n, 8)])  # [n, 2*half]

    def summary(l):
        return {'first_copy': float(l[:, 1:half].mean()), 'second_copy': float(l[:, half + 1:].mean())}

    ind, prev = torch.zeros(L, H), torch.zeros(L, H)
    qp = torch.arange(half + 1, 2 * half)  # 第二遍里的查询位置；它上一次出现之后的那个 token 在 qp - half + 1
    qp1 = torch.arange(2, half + 1)        # 第一遍里的查询位置

    def cap(i, w):
        a, b = qp.to(w.device), qp1.to(w.device)
        ind[i] += w[:, :, a, a - half + 1].mean(2).sum(0).cpu()
        prev[i] += w[:, :, b, b - 1].mean(2).sum(0).cpu()

    patch.capture = cap
    for i in range(0, n, 4):
        model(seq[i:i + 4, :-1].to(DEV))
    patch.reset()
    ind /= n
    prev /= n
    flat = ind.flatten()
    order = flat.argsort(descending=True).tolist()
    top = [(j // H, j % H) for j in order[:32]]
    rows = val_rows(n_(8, 1))
    docs = {name: seg for name in SOURCES if (seg := doc_starts(name, 1025, n_(8, 2))) is not None}
    base_rep = summary(rep_losses())
    base_val = wloss(model, rows)
    base_curve = pos_curve(model, docs)

    def icl(curve):
        return float(curve[480:520].mean() - curve[40:60].mean())

    def heads_off(heads):
        patch.head_mask = {}
        for li, h in heads:
            patch.head_mask.setdefault(li, torch.ones(H))[h] = 0

    gen = torch.Generator().manual_seed(1)
    pool = [(j // H, j % H) for j in order[32:]]
    ablate, curves = [], {'base': base_curve.tolist()}
    for k in n_((2, 4, 8, 16, 32), (2, 8)):
        heads_off(top[:k])
        c = pos_curve(model, docs)
        row = {'k': k, 'kind': 'induction', **summary(rep_losses()), 'val': wloss(model, rows)['weighted'], 'icl': icl(c)}
        curves[f'top{k}'] = c.tolist()
        ablate.append(row)
        rnd = []
        for s in range(n_(3, 1)):
            pick = torch.randperm(len(pool), generator=gen)[:k].tolist()
            heads_off([pool[j] for j in pick])
            c = pos_curve(model, docs)
            rnd.append({**summary(rep_losses()), 'val': wloss(model, rows)['weighted'], 'icl': icl(c)})
        ablate.append({'k': k, 'kind': 'random', **{key: mean(x[key] for x in rnd) for key in rnd[0]}})
        log(f'induction ablate {k}: second copy {row["second_copy"]:.3f} vs random {ablate[-1]["second_copy"]:.3f}')
    patch.reset()
    save('induction', {'n_docs': {k: len(v) for k, v in docs.items()}, 'base_rep': base_rep, 'base_val': base_val['weighted'], 'base_icl': icl(base_curve),
                       'induction_score': ind.tolist(), 'prev_score': prev.tolist(),
                       'top_induction': [{'layer': a, 'head': b, 'score': float(ind[a, b])} for a, b in top[:16]],
                       'top_prev': [{'layer': j // H, 'head': j % H, 'score': float(prev.flatten()[j])}
                                    for j in prev.flatten().argsort(descending=True)[:16].tolist()],
                       'ablate': ablate, 'curves': curves})
    p = plt()
    fig, ax = p.subplots(1, 3, figsize=(17, 5))
    for a, (mat, title) in enumerate(((ind, 'induction score'), (prev, 'previous-token score'))):
        im = ax[a].imshow(mat.numpy(), aspect='auto', cmap='viridis', vmin=0, vmax=max(0.3, float(mat.max())))
        ax[a].set_xlabel('head')
        ax[a].set_ylabel('layer')
        ax[a].set_title(title)
        fig.colorbar(im, ax=ax[a])
    k = np.arange(1024)
    sm = lambda c: np.convolve(np.array(c), np.ones(16) / 16, mode='same')
    for name, c in curves.items():
        ax[2].plot(k[8:-8], sm(c)[8:-8], label=name)
    ax[2].set_xscale('log')
    ax[2].set_xlabel('token position')
    ax[2].set_ylabel('loss (smoothed)')
    ax[2].set_title('loss vs position, induction heads removed')
    ax[2].legend()
    fig.tight_layout()
    fig.savefig(FIGS / 'induction.png', dpi=120)
    p.close(fig)


@torch.no_grad()
def collect(model, patch, x, pos):
    """x [B, T] 输入；pos [B, P] 要取的位置 → [L+1, B*P, d] float32（CPU）。"""
    states = grab_states(patch)
    model(x.to(DEV))
    patch.reset()
    b = torch.arange(x.size(0), device=DEV)[:, None].expand(-1, pos.size(1))
    p = pos.to(DEV)
    return torch.stack([s[b, p].reshape(-1, s.size(-1)).float().cpu() for s in states])


def fit_logreg(xtr, ytr, xte, yte, n_cls, wd=1e-3, iters=100):
    """多类逻辑回归探针（CPU 上用 L-BFGS，避免和训练抢 GPU）。返回测试集准确率。"""
    with torch.enable_grad():
        mu, sd = xtr.mean(0, keepdim=True), xtr.std(0, keepdim=True) + 1e-5
        xtr, xte = (xtr - mu) / sd, (xte - mu) / sd
        w = torch.zeros(xtr.size(1), n_cls, requires_grad=True)
        b = torch.zeros(n_cls, requires_grad=True)
        opt = torch.optim.LBFGS([w, b], lr=1, max_iter=iters, history_size=20, line_search_fn='strong_wolfe')

        def closure():
            opt.zero_grad()
            loss = F.cross_entropy(xtr @ w + b, ytr) + wd * w.square().sum()
            loss.backward()
            return loss
        opt.step(closure)
    with torch.no_grad():
        return float(((xte @ w + b).argmax(1) == yte).float().mean())


def fit_ridge(xtr, ytr, xte, yte, lams=(1.0, 10.0, 100.0, 1e3, 1e4, 1e5)):
    """岭回归探针：在训练集里随机留 20% 选正则强度，再用全部训练集拟合。返回测试集 R^2。"""
    xtr, xte, ytr, yte = xtr.double(), xte.double(), ytr.double(), yte.double()
    mu, sd = xtr.mean(0, keepdim=True), xtr.std(0, keepdim=True) + 1e-5
    xtr, xte = (xtr - mu) / sd, (xte - mu) / sd
    ym = ytr.mean()

    def solve(x, y, lam_list):
        s, v = torch.linalg.eigh(x.T @ x)
        b = v.T @ (x.T @ (y - ym))
        return {lam: v @ (b / (s + lam)) for lam in lam_list}

    def r2(x, y, w):
        return float(1 - ((y - (x @ w + ym)) ** 2).sum() / ((y - y.mean()) ** 2).sum())

    perm = torch.randperm(len(xtr), generator=torch.Generator().manual_seed(0))
    nv = max(1, len(xtr) // 5)
    va, tr = perm[:nv], perm[nv:]
    ws = solve(xtr[tr], ytr[tr], lams)
    best = max(lams, key=lambda lam: r2(xtr[va], ytr[va], ws[lam]))
    return r2(xte, yte, solve(xtr, ytr, (best,))[best])


@torch.no_grad()
def task_probes(model, patch, tok):
    L = model.config.num_layers
    per = n_(48, 8)
    rows = val_rows(per)
    g = torch.Generator().manual_seed(0)
    feats, ysrc, ypos, rid = [], [], [], []
    for si, name in enumerate(SOURCES):
        x = rows[name][:, :-1]
        for i in range(0, len(x), 4):
            xb = x[i:i + 4]
            pos = torch.randint(0, 1024, (len(xb), 16), generator=g)
            feats.append(collect(model, patch, xb, pos))
            ysrc += [si] * pos.numel()
            ypos += pos.flatten().tolist()
            rid += [i + r for r in range(len(xb)) for _ in range(16)]
    X = torch.cat(feats, dim=1)
    ysrc, rid = torch.tensor(ysrc), torch.tensor(rid)
    ypos = torch.log1p(torch.tensor(ypos, dtype=torch.float64))
    te = rid >= int(per * 0.75)
    src_acc, pos_r2 = [], []
    for li in range(L + 1):
        src_acc.append(fit_logreg(X[li][~te], ysrc[~te], X[li][te], ysrc[te], len(SOURCES)))
        pos_r2.append(fit_ridge(X[li][~te], ypos[~te], X[li][te], ypos[te]))
    log(f'probes source acc L0 {src_acc[0]:.3f} mid {src_acc[L // 2]:.3f} last {src_acc[-1]:.3f}; position R2 {pos_r2[0]:.3f} / {pos_r2[L // 2]:.3f}')
    del X

    # 两位数加法：在 "=" 处探测个位、进位、十位；同时看模型自己答对多少
    pairs = [(a, b) for a in range(10, 100) for b in range(10, 100)]
    if QUICK:
        pairs = pairs[::25]
    few = '23+14=37\n51+26=77\n35+48=83\n'  # 三个示例，让基座模型按格式直接给出和
    enc = [tok.encode(few + f'{a}+{b}=', add_bos=True) for a, b in pairs]
    by_len = defaultdict(list)
    for k, ids in enumerate(enc):
        by_len[len(ids)].append(k)
    feats, answers = [None] * len(pairs), [None] * len(pairs)
    for t, ks in by_len.items():
        for j in range(0, len(ks), 256):
            kk = ks[j:j + 256]
            x = torch.tensor([enc[k] for k in kk])
            f = collect(model, patch, x, torch.full((len(kk), 1), t - 1))
            cur = x.to(DEV)
            for _ in range(3):
                cur = torch.cat([cur, model(cur).logits[:, -1].argmax(-1)[:, None]], 1)
            for r, k in enumerate(kk):
                feats[k] = f[:, r]
                answers[k] = tok.decode(cur[r, t:].tolist())
    Xa = torch.stack(feats, dim=1)
    sums = [a + b for a, b in pairs]
    got = [int(m.group(1)) if (m := re.match(r'\s*(\d+)', s)) else None for s in answers]
    model_acc = mean(g_ == s for g_, s in zip(got, sums))
    labels = {'units': torch.tensor([s % 10 for s in sums]), 'carry': torch.tensor([int(a % 10 + b % 10 >= 10) for a, b in pairs]),
              'tens': torch.tensor([(s // 10) % 10 for s in sums])}
    perm = torch.randperm(len(pairs), generator=torch.Generator().manual_seed(0))
    te = torch.zeros(len(pairs), dtype=torch.bool)
    te[perm[:len(pairs) // 5]] = True
    arith = {}
    for key, y in labels.items():
        arith[key] = [fit_logreg(Xa[li][~te], y[~te], Xa[li][te], y[te], int(y.max()) + 1) for li in range(L + 1)]
    log(f'arith model acc {model_acc:.3f}; units probe best {max(arith["units"]):.3f}')
    save('probes', {'source_acc': src_acc, 'source_chance': 1 / len(SOURCES), 'position_r2': pos_r2,
                    'arith_model_acc': model_acc, 'arith': arith,
                    'arith_examples': [(f'{a}+{b}=', answers[k]) for k, (a, b) in enumerate(pairs[:12])]})
    p = plt()
    fig, ax = p.subplots(1, 2, figsize=(13, 4.5))
    ax[0].plot(src_acc, label='data source (7-way)')
    ax[0].plot(pos_r2, label='absolute position (R^2 of log pos)')
    ax[0].axhline(1 / len(SOURCES), ls=':', c='gray')
    ax[0].set_xlabel('layer (0 = embedding)')
    ax[0].legend()
    for key, v in arith.items():
        ax[1].plot(v, label=f'sum {key}')
    ax[1].axhline(model_acc, ls='--', c='k', label='model answers correctly')
    ax[1].set_xlabel('layer')
    ax[1].set_ylabel('probe accuracy at "="')
    ax[1].legend()
    fig.tight_layout()
    fig.savefig(FIGS / 'probes.png', dpi=120)
    p.close(fig)


FACTS = [
    ('法国的首都是', '法国', ['巴黎']), ('日本的首都是', '日本', ['东京']), ('英国的首都是', '英国', ['伦敦']),
    ('德国的首都是', '德国', ['柏林']), ('俄罗斯的首都是', '俄罗斯', ['莫斯科']), ('意大利的首都是', '意大利', ['罗马']),
    ('美国的首都是', '美国', ['华盛顿']), ('韩国的首都是', '韩国', ['首尔']), ('中国的首都是', '中国', ['北京']),
    ('埃及的首都是', '埃及', ['开罗']), ('西班牙的首都是', '西班牙', ['马德里']), ('泰国的首都是', '泰国', ['曼谷']),
    ('澳大利亚的首都是', '澳大利亚', ['堪培拉']), ('加拿大的首都是', '加拿大', ['渥太华']), ('印度的首都是', '印度', ['新德里']),
    ('《红楼梦》的作者是', '《红楼梦》', ['曹雪芹']), ('《西游记》的作者是', '《西游记》', ['吴承恩']),
    ('《三国演义》的作者是', '《三国演义》', ['罗贯中']), ('《水浒传》的作者是', '《水浒传》', ['施耐庵']),
    ('《狂人日记》的作者是', '《狂人日记》', ['鲁迅']), ('《哈姆雷特》的作者是', '《哈姆雷特》', ['莎士比亚']),
    ('相对论的提出者是', '相对论', ['爱因斯坦']), ('万有引力定律的发现者是', '万有引力定律', ['牛顿']),
    ('埃菲尔铁塔位于', '埃菲尔铁塔', ['法国', '巴黎']), ('金字塔位于', '金字塔', ['埃及']), ('长城位于', '长城', ['中国']),
    ('唐朝的都城是', '唐朝', ['长安']), ('李白是唐朝著名的', '李白', ['诗人']), ('杜甫是唐朝著名的', '杜甫', ['诗人']),
    ('苹果公司的创始人是', '苹果公司', ['乔布斯', '史蒂夫']), ('微软公司的创始人是', '微软公司', ['比尔']),
    ('太阳系中最大的行星是', '最大的行星', ['木星']), ('地球唯一的天然卫星是', '天然卫星', ['月球', '月亮']),
    ('东京是日本的', '东京', ['首都']), ('巴黎是法国的', '巴黎', ['首都']), ('北京是中国的', '北京', ['首都']),
    ('大熊猫是中国的', '大熊猫', ['国宝']), ('孔子是中国古代著名的', '孔子', ['思想家', '教育家']),
    ('The capital of France is ', 'France', ['Paris']), ('The capital of Japan is ', 'Japan', ['Tokyo']),
    ('The capital of Germany is ', 'Germany', ['Berlin']), ('The capital of Italy is ', 'Italy', ['Rome']),
    ('The capital of Spain is ', 'Spain', ['Madrid']), ('The capital of Russia is ', 'Russia', ['Moscow']),
    ('The capital of England is ', 'England', ['London']), ('The capital of China is ', 'China', ['Beijing']),
    ('The capital of Egypt is ', 'Egypt', ['Cairo']), ('The capital of Canada is ', 'Canada', ['Ottawa']),
    ('The capital of Australia is ', 'Australia', ['Canberra']), ('The capital of Greece is ', 'Greece', ['Athens']),
    ('The Eiffel Tower is located in ', 'Eiffel Tower', ['Paris', 'France']), ('The Colosseum is located in ', 'Colosseum', ['Rome', 'Italy']),
    ('Big Ben is located in ', 'Big Ben', ['London', 'England']), ('The Great Wall is located in ', 'Great Wall', ['China']),
    ('Romeo and Juliet was written by ', 'Romeo and Juliet', ['William', 'Shakespeare']),
    ('Hamlet was written by ', 'Hamlet', ['William', 'Shakespeare']),
    ('The theory of relativity was developed by ', 'theory of relativity', ['Albert', 'Einstein']),
    ('Microsoft was founded by ', 'Microsoft', ['Bill']), ('Apple was founded by ', 'Apple', ['Steve']),
    ('Facebook was founded by ', 'Facebook', ['Mark']), ('The Mona Lisa was painted by ', 'Mona Lisa', ['Leonardo']),
    ('Tokyo is the capital of ', 'Tokyo', ['Japan']), ('Paris is the capital of ', 'Paris', ['France']),
    ('Berlin is the capital of ', 'Berlin', ['Germany']), ('Moscow is the capital of ', 'Moscow', ['Russia']),
    ('Rome is the capital of ', 'Rome', ['Italy']), ('Madrid is the capital of ', 'Madrid', ['Spain']),
    ('Cairo is the capital of ', 'Cairo', ['Egypt']), ('Beijing is the capital of ', 'Beijing', ['China']),
    ('Charles Darwin is known for the theory of ', 'Charles Darwin', ['evolution', 'natural']),
    ('Neil Armstrong was the first person to walk on the ', 'Neil Armstrong', ['moon', 'Moon']),
    ('The largest planet in the solar system is ', 'largest planet', ['Jupiter']),
    ('The largest ocean on Earth is the ', 'largest ocean', ['Pacific']),
]


def subject_positions(tok, ids, prompt, subject):
    s0 = prompt.find(subject)
    if s0 < 0:
        return []
    s1, prev, pos = s0 + len(subject), 0, []
    for k in range(1, len(ids) + 1):
        cur = len(tok.decode(ids[:k]))
        if cur > prev and cur > s0 and prev < s1:
            pos.append(k - 1)
        prev = cur
    return pos


@torch.no_grad()
def task_trace(model, patch, tok):
    L, d = model.config.num_layers, model.config.hidden_size
    sigma = 3 * float(model.embedding.weight.float().std())
    n_noise, half_w = 5, 2
    cats = ('subj_first', 'subj_mid', 'subj_last', 'after_first', 'after_rest', 'last')
    sums = {k: {c: torch.zeros(L + (1 if k == 'resid' else 0)) for c in cats} for k in ('resid', 'mlp', 'attn')}
    nsum = {k: {c: torch.zeros(L + (1 if k == 'resid' else 0)) for c in cats} for k in ('resid', 'mlp', 'attn')}
    used, cnt = [], {c: 0 for c in cats}
    for prompt, subject, answers in (FACTS[:6] if QUICK else FACTS):
        prompt, subject = norm_text(prompt), norm_text(subject)
        ids = tok.encode(prompt, add_bos=True)
        x1 = torch.tensor([ids], device=DEV)
        clean = {'resid': [None] * (L + 1), 'mlp': [None] * L, 'attn': [None] * L}

        def rec(i, kind, t):
            if kind == 'resid':
                clean['resid'][i + 1] = t
            else:
                clean[kind][i] = t
            return t
        patch.emb_hook = lambda t: (clean['resid'].__setitem__(0, t), t)[1]
        patch.hook = rec
        probs = model(x1).logits[0, -1].float().softmax(-1)
        patch.reset()
        target = None
        for ans in answers:
            full = tok.encode(prompt + norm_text(ans), add_bos=True)
            if full[:len(ids)] == ids and len(full) > len(ids) and int(probs.argmax()) == full[len(ids)]:
                target = full[len(ids)]
                break
        subj = subject_positions(tok, ids, prompt, subject)
        if target is None or not subj or subj[-1] >= len(ids) - 1:
            continue
        T = len(ids)
        p_clean = float(probs[target])
        noise = (torch.randn(n_noise, len(subj), d, generator=torch.Generator().manual_seed(1)) * sigma).to(DEV, torch.bfloat16)
        subj_t = torch.tensor(subj, device=DEV)

        def run(kind, layer):
            # 每行：第 r 个噪声样本 × 恢复位置 j；layer=None 表示不恢复（只加噪）
            if layer is None:
                rows = [(r, -1) for r in range(n_noise)]
            else:
                rows = [(r, j) for r in range(n_noise) for j in range(T)]
            nidx = torch.tensor([a for a, _ in rows], device=DEV)
            rpos = torch.tensor([b for _, b in rows], device=DEV)
            ar = torch.arange(len(rows), device=DEV)

            def corrupt(t):
                t = t.clone()
                t[:, subj_t] += noise[nidx]
                if kind == 'resid' and layer == 0:
                    t[ar, rpos] = clean['resid'][0][0, rpos]
                return t

            def hook(i, k, t):
                if layer is None or k != kind:
                    return t
                if kind == 'resid':
                    if i + 1 != layer:
                        return t
                    src = clean['resid'][layer]
                elif abs(i - layer) > half_w:
                    return t
                else:
                    src = clean[kind][i]
                t = t.clone()
                t[ar, rpos] = src[0, rpos]
                return t
            patch.emb_hook, patch.hook = corrupt, hook
            pr = model(x1.expand(len(rows), T)).logits[:, -1].float().softmax(-1)[:, target]
            patch.reset()
            return pr.view(n_noise, -1).mean(0).cpu()

        p_corrupt = float(run('resid', None)[0])
        if p_clean - p_corrupt < 0.05:
            continue
        last = subj[-1]
        groups = {'subj_first': [subj[0]], 'subj_mid': subj[1:-1], 'subj_last': [last],
                  'after_first': [last + 1] if last + 1 < T - 1 else [], 'after_rest': list(range(last + 2, T - 1)),
                  'last': [T - 1]}
        for kind in ('resid', 'mlp', 'attn'):
            layers = range(L + 1) if kind == 'resid' else range(L)
            ie = torch.stack([run(kind, l) for l in layers]) - p_corrupt  # [layers, T]
            for c, ps in groups.items():
                if ps:
                    sums[kind][c] += ie[:, ps].mean(1)
                    nsum[kind][c] += ie[:, ps].mean(1) / (p_clean - p_corrupt)
        for c, ps in groups.items():
            cnt[c] += int(bool(ps))
        used.append({'prompt': prompt, 'answer': tok.decode([target]), 'p_clean': p_clean, 'p_corrupt': p_corrupt,
                     'subject_tokens': [tok.decode([ids[j]]) for j in subj]})
        log(f'trace {prompt!r} -> {tok.decode([target])!r}: clean {p_clean:.3f} corrupt {p_corrupt:.3f}')
    out = {'facts': used, 'n_facts': len(used), 'sigma': sigma, 'window': 2 * half_w + 1, 'ie': {}, 'ie_norm': {}}
    for kind in sums:
        out['ie'][kind] = {c: (v / max(1, cnt[c])).tolist() for c, v in sums[kind].items()}
        out['ie_norm'][kind] = {c: (v / max(1, cnt[c])).tolist() for c, v in nsum[kind].items()}
    out['category_counts'] = cnt
    save('trace', out)
    p = plt()
    fig, ax = p.subplots(3, 1, figsize=(12, 9))
    for a, kind in enumerate(('resid', 'mlp', 'attn')):
        mat = np.array([out['ie'][kind][c] for c in cats])
        im = ax[a].imshow(mat, aspect='auto', cmap='Purples' if kind == 'resid' else ('Greens' if kind == 'mlp' else 'Reds'))
        ax[a].set_yticks(range(len(cats)))
        ax[a].set_yticklabels(cats)
        ax[a].set_title(f'restore {kind} (window {2 * half_w + 1} for mlp/attn): mean increase of p(answer)')
        fig.colorbar(im, ax=ax[a])
    ax[-1].set_xlabel('layer')
    fig.tight_layout()
    fig.savefig(FIGS / 'causal_trace.png', dpi=120)
    p.close(fig)


# ---------------- 第 5 部分：推理侧 ----------------
STRATS = {
    'greedy': {},
    'greedy+rep_penalty1.2': {'rep': 1.2},
    'greedy+no_repeat_3gram': {'no_repeat': 3},
    'sample_t1.0': {'sample': True, 't': 1.0},
    'sample_t0.7': {'sample': True, 't': 0.7},
    'top_k40': {'sample': True, 't': 1.0, 'top_k': 40},
    'top_p0.9': {'sample': True, 't': 1.0, 'top_p': 0.9},
    'min_p0.05': {'sample': True, 't': 1.0, 'min_p': 0.05},
}


@torch.no_grad()
def generate_batch(model, prompts, new, cfg):
    B = prompts.size(0)
    x = prompts.to(DEV)
    out = model(x, use_cache=True)
    past, logits = out.past_key_values, out.logits[:, -1].float()
    hist = x.clone()
    n = cfg.get('no_repeat')
    table = None
    if n:
        table = [defaultdict(set) for _ in range(B)]
        for r in range(B):
            s = hist[r].tolist()
            for j in range(len(s) - n + 1):
                table[r][tuple(s[j:j + n - 1])].add(s[j + n - 1])
    gen = []
    for _ in range(new):
        lg = logits
        if cfg.get('rep'):
            sc = torch.gather(lg, 1, hist)
            sc = torch.where(sc > 0, sc / cfg['rep'], sc * cfg['rep'])
            lg = lg.scatter(1, hist, sc)
        if table is not None:
            ctx = hist[:, -(n - 1):].tolist()
            for r in range(B):
                banned = table[r].get(tuple(ctx[r]))
                if banned:
                    lg[r, list(banned)] = float('-inf')
        if cfg.get('sample'):
            lg = lg / cfg.get('t', 1.0)
            if cfg.get('top_k'):
                kth = lg.topk(cfg['top_k'], dim=-1).values[:, -1:]
                lg = lg.masked_fill(lg < kth, float('-inf'))
            probs = lg.softmax(-1)
            if cfg.get('top_p'):
                sp, si = probs.sort(-1, descending=True)
                sp = sp.masked_fill(sp.cumsum(-1) - sp > cfg['top_p'], 0)
                probs = torch.zeros_like(probs).scatter(1, si, sp)
            if cfg.get('min_p'):
                probs = probs.masked_fill(probs < cfg['min_p'] * probs.max(-1, keepdim=True).values, 0)
            nxt = torch.multinomial(probs / probs.sum(-1, keepdim=True), 1).squeeze(1)
        else:
            nxt = lg.argmax(-1)
        gen.append(nxt)
        if table is not None:
            ctx = hist[:, -(n - 1):].tolist()
            for r in range(B):
                table[r][tuple(ctx[r])].add(int(nxt[r]))
        hist = torch.cat([hist, nxt[:, None]], 1)
        out = model(nxt[:, None], past_key_values=past, use_cache=True)
        past, logits = out.past_key_values, out.logits[:, -1].float()
    return torch.stack(gen, 1).cpu()


def rep4(ids):
    if len(ids) < 8:
        return 0.0
    grams = [tuple(ids[i:i + 4]) for i in range(len(ids) - 3)]
    return 1 - len(set(grams)) / len(grams)


@torch.no_grad()
def continuation_nll(model, prompts, conts, lengths=None, batch=16):
    out = []
    for i in range(0, len(prompts), batch):
        x = torch.cat([prompts[i:i + batch], conts[i:i + batch]], 1)
        lc = token_losses(model, x)[:, prompts.size(1) - 1:]
        for r in range(lc.size(0)):
            k = lengths[i + r] if lengths is not None else lc.size(1)
            out.append(float(lc[r, :max(1, k)].mean()))
    return out


@torch.no_grad()
def task_decode(model, patch, tok):
    P, NEW = 48, 128
    prompts, refs, srcs = [], [], []
    for name in SOURCES:
        seg = doc_starts(name, 1 + P + NEW, n_(20, 3))
        if seg is None:
            continue
        prompts.append(seg[:, :1 + P])
        refs.append(seg[:, 1 + P:])
        srcs += [name] * len(seg)
    prompts, refs = torch.cat(prompts), torch.cat(refs)
    human = continuation_nll(model, prompts, refs)
    human_rep = [rep4(r.tolist()) for r in refs]
    res, examples = {'human_reference': {'nll': mean(human), 'rep4': mean(human_rep)}}, {}
    for sname, cfg in STRATS.items():
        torch.manual_seed(0)
        gen = generate_batch(model, prompts, NEW, cfg)
        lens = [(row.index(EOS) if EOS in row else len(row)) for row in gen.tolist()]
        reps = [rep4(gen[r, :lens[r]].tolist()) for r in range(len(gen))]
        bi, tot = Counter(), 0
        for r in range(len(gen)):
            s = gen[r, :lens[r]].tolist()
            bi.update(zip(s, s[1:]))
            tot += max(0, len(s) - 1)
        nll = continuation_nll(model, prompts, gen, lens)
        res[sname] = {'rep4': mean(reps), 'distinct2': len(bi) / max(1, tot), 'nll': mean(nll),
                      'eos_rate': mean(l < NEW for l in lens), 'mean_len': mean(lens),
                      'by_source': {s: {'rep4': mean(reps[i] for i in range(len(gen)) if srcs[i] == s),
                                        'nll': mean(nll[i] for i in range(len(gen)) if srcs[i] == s)} for s in SOURCES if s in srcs}}
        for s in ('zh_cci3', 'en_fwedu', 'code'):
            if s in srcs:
                i = srcs.index(s)
                examples.setdefault(s, {'prompt': tok.decode(prompts[i].tolist()), 'human': tok.decode(refs[i, :80].tolist())})
                examples[s][sname] = tok.decode(gen[i, :min(80, lens[i])].tolist())
        log(f'decode {sname}: rep4 {res[sname]["rep4"]:.3f} distinct2 {res[sname]["distinct2"]:.3f} nll {res[sname]["nll"]:.3f}')

    # 自我强化：同一片段重复多遍，看模型给下一遍的概率怎么涨
    rows = val_rows(n_(16, 2))
    spans = torch.cat([rows[s][:, 100:124] for s in ('zh_cci3', 'en_fwedu', 'code')])
    reps_n = 6
    x = torch.cat([torch.full((len(spans), 1), BOS), spans.repeat(1, reps_n)], 1)
    probs, hits = [], []
    for i in range(0, len(x), 16):
        xb = x[i:i + 16].to(DEV)
        lg = model(xb[:, :-1]).logits.float()
        probs.append(lg.log_softmax(-1).gather(-1, xb[:, 1:, None]).squeeze(-1).exp().cpu())
        hits.append((lg.argmax(-1) == xb[:, 1:]).float().cpu())
    probs, hits = torch.cat(probs), torch.cat(hits)
    w = spans.size(1)
    reinforce = [{'repeat': k + 1, 'mean_prob': float(probs[:, k * w + (1 if k == 0 else 0):(k + 1) * w].mean()),
                  'argmax_hit': float(hits[:, k * w + (1 if k == 0 else 0):(k + 1) * w].mean())} for k in range(reps_n)]
    log(f'reinforce: {[round(r["mean_prob"], 3) for r in reinforce]}')
    save('decode', {'strategies': res, 'examples': examples, 'reinforce': reinforce, 'n_prompts': len(prompts),
                    'prompt_tokens': P, 'new_tokens': NEW})


def fake_quant(w, bits, group=None):
    shape, qmax = w.shape, 2 ** (bits - 1) - 1
    x = w.float().reshape(-1, group) if group else w.float()
    scale = x.abs().amax(dim=1, keepdim=True).clamp(min=1e-8) / qmax
    return ((x / scale).round().clamp(-qmax, qmax) * scale).reshape(shape).to(w.dtype)


@torch.no_grad()
def task_quant(model, patch, tok):
    L = model.config.num_layers
    orig = {n: p.detach().clone() for n, p in model.named_parameters()}

    def restore():
        for n, p in model.named_parameters():
            p.copy_(orig[n])

    def apply(bits, group, layers=None, kinds=LIN, embed=False):
        for i, layer in enumerate(model.layers):
            if layers is None or i in layers:
                for kname in kinds:
                    mod = layer.get_submodule(kname)
                    mod.weight.copy_(fake_quant(mod.weight, bits, group))
        if embed:
            model.embedding.weight.copy_(fake_quant(model.embedding.weight, bits, group))

    def act_hooks(mode):
        hs = []
        for layer in model.layers:
            for kname in LIN:
                def pre(m, args, mode=mode):
                    x = args[0]
                    s = (x.abs().amax() if mode == 'tensor' else x.abs().amax(-1, keepdim=True)).clamp(min=1e-8) / 127
                    return ((x / s).round().clamp(-127, 127) * s,)
                hs.append(layer.get_submodule(kname).register_forward_pre_hook(pre))
        return hs

    rows = val_rows(n_(32, 2))
    main = {'bf16': wloss(model, rows)}
    configs = [('W8 per-channel', 8, None, False), ('W4 per-channel', 4, None, False), ('W4 g128', 4, 128, False),
               ('W3 g128', 3, 128, False), ('W3 per-channel', 3, None, False), ('W2 g64', 2, 64, False),
               ('W4 g128 + embedding', 4, 128, True)]
    for name, bits, group, emb in configs:
        apply(bits, group, embed=emb)
        main[name] = wloss(model, rows)
        restore()
        log(f'quant {name}: {main[name]["weighted"]:.4f}')
    for name, mode, wbits in (('A8 per-tensor', 'tensor', None), ('A8 per-token', 'token', None), ('W8A8 per-token', 'token', 8)):
        if wbits:
            apply(wbits, None)
        hs = act_hooks(mode)
        main[name] = wloss(model, rows)
        for h in hs:
            h.remove()
        restore()
        log(f'quant {name}: {main[name]["weighted"]:.4f}')

    # 激活离群值：每个线性层输入的 max|x| / median|x|
    outl = defaultdict(list)
    hs = []
    for i, layer in enumerate(model.layers):
        for kname in LIN:
            def pre(m, args, key=(i, kname)):
                a = args[0].float().abs().flatten()
                outl[key].append(float(a.max() / a[::97].median().clamp(min=1e-8)))
            hs.append(layer.get_submodule(kname).register_forward_pre_hook(pre))
    sample = torch.cat([r[:2] for r in val_rows(n_(2, 1)).values()])
    for i in range(0, len(sample), 2):
        model(sample[i:i + 2, :-1].to(DEV))
    for h in hs:
        h.remove()
    outliers = {k: [mean(outl[(i, k)]) for i in range(L)] for k in LIN}

    small = val_rows(n_(16, 1))
    base_small = wloss(model, small)['weighted']
    per_layer = []
    for i in range(L):
        apply(3, 128, layers={i})
        per_layer.append(wloss(model, small)['weighted'] - base_small)
        restore()
    per_kind = {}
    for kname in LIN:
        apply(3, 128, kinds=(kname,))
        per_kind[kname] = wloss(model, small)['weighted'] - base_small
        restore()
    log(f'quant per-kind W3: {per_kind}')
    save('quant', {'main': main, 'per_layer_w3g128': per_layer, 'per_kind_w3g128': per_kind,
                   'activation_outlier_ratio': outliers, 'base_small': base_small})
    p = plt()
    fig, ax = p.subplots(1, 2, figsize=(13, 4.5))
    ax[0].bar(range(L), per_layer)
    ax[0].set_xlabel('layer quantized to int3 (g128), others bf16')
    ax[0].set_ylabel('loss increase')
    for kname in LIN:
        ax[1].semilogy(outliers[kname], label=kname.split('.')[-1])
    ax[1].set_xlabel('layer')
    ax[1].set_ylabel('max|x| / median|x| of linear input')
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGS / 'quant.png', dpi=120)
    p.close(fig)


TASKS = {'sinks': task_sinks, 'lens': task_lens, 'layers': task_layers, 'induction': task_induction,
         'probes': task_probes, 'trace': task_trace, 'decode': task_decode, 'quant': task_quant, 'heads': task_heads}


def main(argv=None):
    global QUICK, OUT, FIGS
    ap = argparse.ArgumentParser()
    ap.add_argument('tasks', nargs='+')
    ap.add_argument('--model', default='v1')
    ap.add_argument('--quick', action='store_true')
    a = ap.parse_args(argv)
    QUICK = a.quick
    if QUICK:
        OUT = OUT / 'quick'
        FIGS = OUT / 'figs'
    OUT.mkdir(parents=True, exist_ok=True)
    FIGS.mkdir(exist_ok=True)
    torch.cuda.set_per_process_memory_fraction(float(os.environ.get('MECH_GPU_FRACTION', '0.087')), 0)
    torch.set_num_threads(4)
    torch.manual_seed(0)
    tok = load_tokenizer()
    model = load_model(a.model)
    patch = Patch(model)
    tasks = list(TASKS) if a.tasks == ['all'] else a.tasks
    for t in tasks:
        t0 = time.time()
        log(f'== {t} ==')
        try:
            TASKS[t](model, patch, tok)
            log(f'== {t} done in {time.time() - t0:.0f}s, peak mem {torch.cuda.max_memory_allocated() / 1e9:.1f}GB')
            (OUT / f'{t}.FAILED').unlink(missing_ok=True)
        except Exception:
            import traceback
            traceback.print_exc()
            patch.reset()
            (OUT / f'{t}.FAILED').write_text(traceback.format_exc())
        torch.cuda.empty_cache()
    log('all done')


if __name__ == '__main__':
    main()
