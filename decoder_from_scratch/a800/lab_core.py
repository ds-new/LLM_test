"""实验室（第二、三批消融）共用的训练部件：同一套代码覆盖 μP、scaling law、数据工程对比、学习率调度、新优化器，
实验之间只差配置（实验清单在 lab.py）。

  * 模型：DecoderLM 加两个前向乘子（嵌入输出 × input_mult，logits × output_mult）。SP 时都是 1，与原模型完全一致。
  * 参数化：sp（现行：所有矩阵 std 0.02，残差输出层再除以 √(2L)）；mup（m = 宽度 / 基准宽度 256：隐藏矩阵初始化方差 ÷m、
    学习率 ÷m，logits × α_out/m，嵌入与 RMSNorm 不变；head_dim 固定 64、只增加头数，所以注意力缩放不随宽度变）。
  * 优化器：adamw；muon（隐藏矩阵用 torch.optim.Muon，更新 RMS 对齐 AdamW；嵌入和 RMSNorm 用 AdamW）；
    soap（隐藏矩阵用 SOAP，其余同上）；sf_adamw（Schedule-Free AdamW）。所有参数组带 lr_mult，学习率 = 调度值 × lr_mult。
  * 调度：wsd（冷却形状 linear / cosine / 1-sqrt，终点比例可设）、cosine、linear、constant。
  * 主线 + 分支：主线在指定 token 处存"里程碑检查点"（权重、优化器、采样器、步数），分支从里程碑恢复后换调度继续。
  * 评估：v1 验证文档上的 bits/字节（与第一批消融同一份 val_docs）、多位数运算（3-shot 贪心）、各层最大注意力 logit。
  * 诊断：裁剪生效占比、更新量/权重比、各类参数的更新 RMS（除以该组学习率）、坐标检查。
"""

import contextlib
import json
import math
import os
from pathlib import Path
import re
import time

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from ..config import ModelConfig
from ..model import DecoderLM, LMOutput
from .common import heartbeat, read_json, stamp, write_json

LAB = Path(os.environ.get('LAB_RUN_DIR', '/root/autodl-tmp/a800_lab'))
LAB_SCRATCH = Path(os.environ.get('LAB_SCRATCH', '/root/a800_scratch/lab'))
# 里程碑检查点放内存盘：数据盘在路线 A 之后只剩约 10GB，放不下几份 1.5–4GB 的检查点；机器重启会丢失，调度器会重跑对应的主线
LAB_CKPT = Path(os.environ.get('LAB_CKPT', '/dev/shm/a800_lab_ckpt'))
AB = Path('/root/autodl-tmp/a800_tok_ab')
VAL_DOCS = AB / 'data' / 'val_docs.jsonl'
ARITH_EVAL = AB / 'data' / 'arith_eval.json'
V1_TOK_HF = Path('/root/autodl-tmp/a800_pretrain_v1/tokenizer/tokenizer_hf.json')
BOS, EOS, VOCAB = 2, 3, 32000
EVAL_WEIGHTS = {'zh_cci3': 0.28, 'zh_fwedu': 0.14, 'zh_industry': 0.13, 'en_fwedu': 0.18, 'en_cosmo': 0.07, 'code': 0.12, 'math': 0.08}
HIDDEN = ('q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj')
PROXY_SHAPE = {'width': 768, 'depth': 16, 'heads': 12, 'kv_heads': 4, 'ffn': 2048}


class StopJob(Exception):
    pass


# ------------------------------------------------------------------ 结构 ----------
def shape_config(shape, max_seq_len=2048):
    """shape: width, depth, 可选 heads / kv_heads / ffn。默认 head_dim 64、多头注意力、SwiGLU 中间维 ≈ 2.75 × 宽度（取 64 的倍数）。"""
    w, depth = int(shape['width']), int(shape['depth'])
    heads = int(shape.get('heads') or max(1, w // 64))
    kv = int(shape.get('kv_heads') or heads)
    ffn = int(shape.get('ffn') or max(64, int(round(2.75 * w / 64)) * 64))
    return ModelConfig(vocab_size=VOCAB, hidden_size=w, num_layers=depth, num_heads=heads, num_kv_heads=kv,
                       intermediate_size=ffn, max_seq_len=max_seq_len, dropout=0.0, tie_embeddings=True, attention_backend='sdpa')


def family_shape(width, depth=16):
    """scaling law 与迁移验证用的同一族结构：head_dim 64，KV 头 = 头数 // 3，SwiGLU ≈ 8/3 宽度。宽 768 时正好是代理模型 d768_l16。"""
    heads = width // 64
    return {'width': width, 'depth': depth, 'heads': heads, 'kv_heads': max(1, heads // 3), 'ffn': int(round(8 * width / 3 / 64)) * 64}


def param_count(shape):
    with torch.device('meta'):
        model = DecoderLM(shape_config(shape))
    return sum(p.numel() for p in model.parameters())


def param_kind(name, p):
    if p.dim() == 2 and name.endswith(tuple(h + '.weight' for h in HIDDEN)):
        return 'hidden'
    return 'embed' if p.dim() >= 2 else 'vector'


class LabLM(DecoderLM):
    """DecoderLM + 两个前向乘子。只用于训练和整段打分（不需要 KV cache）。"""

    def __init__(self, config, input_mult=1.0, output_mult=1.0):
        super().__init__(config)
        self.input_mult, self.output_mult = float(input_mult), float(output_mult)

    def forward(self, input_ids, targets=None, last_logits_only=False):
        x = self.embedding(input_ids)
        if self.input_mult != 1.0:
            x = x * self.input_mult
        for layer in self.layers:
            x, _ = layer(x)
        if last_logits_only:
            x = x[:, -1:]
        logits = self.lm_head(self.final_norm(x))
        if self.output_mult != 1.0:
            logits = logits * self.output_mult
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.float().reshape(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-100)
        return LMOutput(logits=logits, loss=loss)


def width_mult(run):
    if run.get('param', 'sp') != 'mup':
        return 1.0
    return shape_config(run['shape']).hidden_size / float((run.get('mup') or {}).get('base_width', 256))


def build_model(run):
    """按 run 的结构和参数化建模型（调用前先 torch.manual_seed）。返回 (model, m)。"""
    cfg = shape_config(run['shape'])
    mu = run.get('mup') or {}
    m = width_mult(run)
    is_mup = run.get('param', 'sp') == 'mup'
    model = LabLM(cfg, input_mult=float(mu.get('alpha_in', 1.0)) if is_mup else 1.0,
                  output_mult=float(mu.get('alpha_out', 1.0)) / m if is_mup else 1.0)
    if is_mup:
        sigma = float(mu.get('sigma', 0.02))
        if m != 1.0 or sigma != 0.02:  # m = 1 且 σ = 0.02 时与 SP 的初始化完全相同，不重抽
            with torch.no_grad():
                for name, p in model.named_parameters():
                    kind = param_kind(name, p)
                    if kind == 'hidden':
                        std = sigma / math.sqrt(m)
                        if name.endswith(('o_proj.weight', 'down_proj.weight')):
                            std /= math.sqrt(2 * cfg.num_layers)
                        p.normal_(0.0, std)
                    elif kind == 'embed':
                        p.normal_(0.0, sigma)
    return model, m


# ------------------------------------------------------------------ 学习率调度 ----------
def lr_factor(step, s):
    """学习率相对峰值的倍数。s 的字段以步数表示：warmup、total、kind（wsd/cosine/linear/constant）、decay_start、shape、final。"""
    w = int(s.get('warmup', 0))
    if step < w:
        return (step + 1) / w
    kind, final = s.get('kind', 'wsd'), float(s.get('final', 0.0))
    if kind == 'constant':
        return 1.0
    total = int(s['total'])
    if kind == 'wsd':
        start = int(s['decay_start'])
        if step < start:
            return 1.0
        p = min(1.0, (step - start) / max(1, total - start))
        shape = s.get('shape', 'linear')
    elif kind in ('cosine', 'linear'):
        p = min(1.0, (step - w) / max(1, total - w))
        shape = kind
    else:
        raise ValueError(f'unknown schedule {kind}')
    if shape == 'linear':
        q = 1.0 - p
    elif shape == 'cosine':
        q = 0.5 * (1.0 + math.cos(math.pi * p))
    elif shape == '1-sqrt':
        q = 1.0 - math.sqrt(p)
    else:
        raise ValueError(f'unknown decay shape {shape}')
    return final + (1.0 - final) * q


def tokens_per_step(run):
    return int(run.get('seq_len', 1024)) * int(run.get('micro', 64)) * int(run.get('accum', 4))


def sched_steps(run, end_step, tps):
    """把 run['sched']（token 表示）换成步数表示。默认：WSD，最后 20% 线性衰减到峰值的 10%（与 v1 和第一批消融相同）。"""
    s = dict(run.get('sched') or {'kind': 'wsd', 'decay_fraction': 0.2, 'final': 0.1})
    if 'warmup' not in s:
        s['warmup'] = min(300, max(50, end_step // 10))
    s['total'] = end_step
    if s.get('kind', 'wsd') == 'wsd':
        if 'decay_from' in s:
            s['decay_start'] = int(int(s['decay_from']) // tps)
        else:
            s['decay_start'] = end_step - int(round(float(s.get('decay_fraction', 0.2)) * end_step))
    return s


# ------------------------------------------------------------------ 优化器 ----------
class SOAP(torch.optim.Optimizer):
    """SOAP（Vyas et al., 2024）：在 Shampoo 预条件器的特征基里跑 Adam。按作者公开实现的结构移植，只保留二维矩阵需要的部分。
    每个维度维护 GG = EMA(G Gᵀ)（维度超过 max_precond_dim 的一侧不做预条件），每 precondition_frequency 步用一次
    幂迭代 + QR 更新特征基，并把二阶矩按新基的特征值顺序重排；一阶、二阶矩都存在旋转后的空间。第一步只初始化预条件器。"""

    def __init__(self, params, lr=3e-3, betas=(0.95, 0.95), shampoo_beta=-1.0, eps=1e-8, weight_decay=0.01,
                 precondition_frequency=10, max_precond_dim=10000, correct_bias=True):
        defaults = dict(lr=lr, betas=tuple(betas), shampoo_beta=shampoo_beta, eps=eps, weight_decay=weight_decay,
                        precondition_frequency=precondition_frequency, max_precond_dim=max_precond_dim, correct_bias=correct_bias)
        super().__init__(params, defaults)

    @staticmethod
    def _project(x, Q):
        for q in Q:
            x = torch.tensordot(x, q, dims=[[0], [0]]) if q is not None else x.permute(*range(1, x.dim()), 0)
        return x

    @staticmethod
    def _project_back(x, Q):
        for q in Q:
            x = torch.tensordot(x, q, dims=[[0], [1]]) if q is not None else x.permute(*range(1, x.dim()), 0)
        return x

    @staticmethod
    def _eig(GGs):
        out = []
        for m in GGs:
            if m is None:
                out.append(None)
                continue
            eye = torch.eye(m.shape[0], device=m.device, dtype=m.dtype)
            try:
                _, q = torch.linalg.eigh(m + 1e-30 * eye)
            except Exception:  # noqa: BLE001
                _, q = torch.linalg.eigh(m.double() + 1e-30 * eye.double())
                q = q.to(m.dtype)
            out.append(torch.flip(q, [1]))
        return out

    def _qr_update(self, st):
        eas = st['exp_avg_sq']
        out = []
        for i, (m, o) in enumerate(zip(st['GG'], st['Q'])):
            if m is None:
                out.append(None)
                continue
            est = torch.diag(o.T @ m @ o)
            order = torch.argsort(est, descending=True)
            eas = eas.index_select(i, order)
            q, _ = torch.linalg.qr(m @ o[:, order])
            out.append(q)
        st['exp_avg_sq'] = eas
        return out

    def _update_precond(self, g, st, sb, group):
        if st['Q'] is not None:
            st['exp_avg'] = self._project_back(st['exp_avg'], st['Q'])
        for i, GG in enumerate(st['GG']):
            if GG is not None:
                dims = [d for d in range(g.dim()) if d != i]
                GG.lerp_(torch.tensordot(g, g, dims=[dims, dims]), 1 - sb)
        if st['Q'] is None:
            st['Q'] = self._eig(st['GG'])
        elif st['step'] > 0 and st['step'] % group['precondition_frequency'] == 0:
            st['Q'] = self._qr_update(st)
        if st['step'] > 0:
            st['exp_avg'] = self._project(st['exp_avg'], st['Q'])

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            b1, b2 = group['betas']
            sb = group['shampoo_beta'] if group['shampoo_beta'] >= 0 else b2
            for p in group['params']:
                if p.grad is None:
                    continue
                g = p.grad.float()
                st = self.state[p]
                if 'step' not in st:
                    st['step'] = 0
                    st['exp_avg'] = torch.zeros_like(g)
                    st['exp_avg_sq'] = torch.zeros_like(g)
                    st['GG'] = [torch.zeros(s, s, device=g.device, dtype=torch.float32) if s <= group['max_precond_dim'] else None
                                for s in g.shape]
                    st['Q'] = None
                    self._update_precond(g, st, sb, group)
                    continue
                gp = self._project(g, st['Q'])
                st['step'] += 1
                ea, eas = st['exp_avg'], st['exp_avg_sq']
                ea.mul_(b1).add_(gp, alpha=1 - b1)
                eas.mul_(b2).add_(gp.square(), alpha=1 - b2)
                denom = eas.sqrt().add_(group['eps'])
                step_size = group['lr']
                if group['correct_bias']:
                    step_size = step_size * math.sqrt(1 - b2 ** st['step']) / (1 - b1 ** st['step'])
                upd = self._project_back(ea / denom, st['Q'])
                p.add_(upd.to(p.dtype), alpha=-step_size)
                if group['weight_decay'] > 0:
                    p.add_(p, alpha=-group['lr'] * group['weight_decay'])
                self._update_precond(g, st, sb, group)
        return loss


class AdamWScheduleFree(torch.optim.Optimizer):
    """Schedule-Free AdamW（Defazio et al., 2024），按作者公开实现移植。训练时参数是插值点 y，评估前 eval() 切到平均点 x，
    训练前 train() 切回。学习率每步由外部写入（warmup 也在外部做）；平均权重 c = lr_max² / Σ，与原实现相同。"""

    def __init__(self, params, lr=2.5e-3, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.0, r=0.0, weight_lr_power=2.0):
        defaults = dict(lr=lr, betas=tuple(betas), eps=eps, weight_decay=weight_decay, r=r, weight_lr_power=weight_lr_power,
                        k=0, train_mode=True, weight_sum=0.0, lr_max=-1.0)
        super().__init__(params, defaults)

    @torch.no_grad()
    def eval(self):
        for group in self.param_groups:
            if group['train_mode']:
                b1 = group['betas'][0]
                for p in group['params']:
                    z = self.state[p].get('z')
                    if z is not None:
                        p.lerp_(z, 1 - 1 / b1)
                group['train_mode'] = False

    @torch.no_grad()
    def train(self):
        for group in self.param_groups:
            if not group['train_mode']:
                b1 = group['betas'][0]
                for p in group['params']:
                    z = self.state[p].get('z')
                    if z is not None:
                        p.lerp_(z, 1 - b1)
                group['train_mode'] = True

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            if not group['train_mode']:
                raise RuntimeError('Schedule-Free 优化器要先 train() 再 step()')
            b1, b2 = group['betas']
            k, lr, decay = group['k'], group['lr'], group['weight_decay']
            group['lr_max'] = lr_max = max(lr, group['lr_max'])
            weight = ((k + 1) ** group['r']) * (lr_max ** group['weight_lr_power'])
            group['weight_sum'] += weight
            ckp1 = weight / group['weight_sum'] if group['weight_sum'] > 0 else 0.0
            bc2 = 1 - b2 ** (k + 1)
            for p in group['params']:
                if p.grad is None:
                    continue
                st = self.state[p]
                if 'z' not in st:
                    st['z'] = p.detach().clone()
                    st['exp_avg_sq'] = torch.zeros_like(p)
                z, v = st['z'], st['exp_avg_sq']
                g = p.grad
                v.mul_(b2).addcmul_(g, g, value=1 - b2)
                gn = g / v.div(bc2).sqrt_().add_(group['eps'])
                if decay != 0:
                    gn.add_(p, alpha=decay)
                p.lerp_(z, ckp1)
                p.add_(gn, alpha=lr * (b1 * (1 - ckp1) - 1))
                z.sub_(gn, alpha=lr)
            group['k'] = k + 1
        return loss


class Combo:
    """几个优化器合起来用：统一设学习率（× 各组 lr_mult）、step、存取状态；Schedule-Free 需要 train()/eval() 切换。"""

    def __init__(self, opts, schedule_free=False):
        self.opts, self.schedule_free = opts, schedule_free

    @property
    def param_groups(self):
        return [g for o in self.opts for g in o.param_groups]

    def set_lr(self, lr):
        for g in self.param_groups:
            g['lr'] = lr * g.get('lr_mult', 1.0)

    def step(self):
        for o in self.opts:
            o.step()

    def zero_grad(self):
        for o in self.opts:
            o.zero_grad(set_to_none=True)

    def state_dict(self):
        return [o.state_dict() for o in self.opts]

    def load_state_dict(self, states):
        if len(states) != len(self.opts):
            raise ValueError('优化器状态与当前优化器组合不一致')
        for o, s in zip(self.opts, states):
            o.load_state_dict(s)

    def train(self):
        if self.schedule_free:
            for o in self.opts:
                o.train()

    def eval(self):
        if self.schedule_free:
            for o in self.opts:
                o.eval()


def build_optimizer(model, ocfg, m, device):
    name = ocfg.get('name', 'adamw')
    lr, wd = float(ocfg['lr']), float(ocfg.get('wd', 0.1))
    b1, b2 = ocfg.get('betas', (0.9, 0.95))
    eps = float(ocfg.get('eps', 1e-8))
    parts = {'hidden': [], 'embed': [], 'vector': []}
    for n, p in model.named_parameters():
        parts[param_kind(n, p)].append(p)

    def group(kind, mult=1.0):
        return {'params': parts[kind], 'lr': lr * mult, 'lr_mult': mult, 'weight_decay': 0.0 if kind == 'vector' else wd, 'kind': kind}

    hid, rest = group('hidden', 1.0 / m), [group('embed'), group('vector')]
    fused = device.type == 'cuda'
    if name == 'adamw':
        return Combo([torch.optim.AdamW([hid] + rest, lr=lr, betas=(b1, b2), eps=eps, fused=fused)])
    if name == 'sf_adamw':
        return Combo([AdamWScheduleFree([hid] + rest, lr=lr, betas=(b1, b2), eps=eps, weight_decay=wd)], schedule_free=True)
    adam = torch.optim.AdamW(rest, lr=lr, betas=(b1, b2), eps=eps, fused=fused)
    if name == 'muon':
        mu = torch.optim.Muon([hid], lr=lr, weight_decay=wd, momentum=float(ocfg.get('momentum', 0.95)), nesterov=True,
                              adjust_lr_fn=ocfg.get('adjust', 'match_rms_adamw'))
        return Combo([mu, adam])
    if name == 'soap':
        so = SOAP([hid], lr=lr, betas=tuple(ocfg.get('soap_betas', (0.95, 0.95))), eps=eps, weight_decay=wd,
                  precondition_frequency=int(ocfg.get('freq', 10)), max_precond_dim=int(ocfg.get('max_precond_dim', 10000)))
        return Combo([so, adam])
    raise ValueError(f'unknown optimizer {name}')


# ------------------------------------------------------------------ 数据 ----------
def load_dataset(name):
    path = LAB / 'data' / name / 'manifest.json'
    d = read_json(path)
    if not d or not (LAB / 'data' / name / 'READY').exists():
        raise FileNotFoundError(f'数据集 {name} 还没有构建好（{path}）')
    return d


def amp(device):
    if device.type == 'cuda':
        return torch.autocast(device_type='cuda', dtype=torch.bfloat16)
    return contextlib.nullcontext()


# ------------------------------------------------------------------ 评估 ----------
_EVAL = {}


def eval_streams(seq_len=1024):
    if seq_len not in _EVAL:
        from tokenizers import Tokenizer
        tok = Tokenizer.from_file(str(V1_TOK_HF))
        by = {}
        with open(VAL_DOCS, encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    d = json.loads(line)
                    by.setdefault(d['source'], []).append(d['text'])
        out = {}
        for src, texts in by.items():
            stream = []
            for e in tok.encode_batch(texts, add_special_tokens=False):
                stream += [BOS] + e.ids + [EOS]
            arr = torch.tensor(stream, dtype=torch.long)
            out[src] = {'wins': [arr[s:s + seq_len + 1] for s in range(0, len(arr) - 1, seq_len)],
                        'bytes': sum(len(t.encode('utf-8')) for t in texts)}
        _EVAL[seq_len] = (tok, out)
    return _EVAL[seq_len]


@torch.no_grad()
def eval_arith(model, tok, device, limit=None):
    ev = read_json(ARITH_EVAL)
    arith = {}
    for cat, probs in ev['problems'].items():
        heartbeat(f'eval arith {cat}')
        probs = probs[:limit] if limit else probs
        shots = ''.join(f'{a} {o} {b} = {c}\n' for a, o, b, c in ev['shots'][cat])
        correct = 0
        for a, o, b, c in probs:
            ids = [BOS] + tok.encode(shots + f'{a} {o} {b} = ', add_special_tokens=False).ids
            x = torch.tensor([ids], device=device)
            gen = []
            for _ in range(12):
                with amp(device):
                    nxt = int(model(x, last_logits_only=True).logits[0, -1].argmax())
                if nxt == EOS:
                    break
                gen.append(nxt)
                if '\n' in tok.decode(gen):
                    break
                x = torch.cat([x, torch.tensor([[nxt]], device=device)], 1)
            mm = re.match(r'\s*(-?\d+)', tok.decode(gen))
            correct += int(bool(mm) and int(mm.group(1)) == c)
        arith[cat] = round(correct / max(1, len(probs)), 4)
    return {'arith': arith, 'arith_mean': round(float(np.mean(list(arith.values()))), 4)}


@torch.no_grad()
def evaluate(model, device, arith=False, seq_len=1024, limit=None, arith_limit=None):
    """bits/字节：与第一批消融相同的验证文档、1024 长度不重叠窗口、不计 BOS/EOS 目标。limit 只给自测用（每个来源最多几个窗口）。"""
    tok, streams = eval_streams(seq_len)
    was_training = model.training
    model.eval()
    bpb = {}
    for src, e in streams.items():
        heartbeat(f'eval bpb {src}')
        wins = e['wins'][:limit] if limit else e['wins']
        full = [w for w in wins if len(w) == seq_len + 1]
        groups = [torch.stack(full[i:i + 16]) for i in range(0, len(full), 16)] + [w.unsqueeze(0) for w in wins if 2 <= len(w) < seq_len + 1]
        nll, nbytes = 0.0, e['bytes']
        for b in groups:
            b = b.to(device)
            with amp(device):
                logits = model(b[:, :-1]).logits
            tgt = b[:, 1:]
            t_nll = -torch.log_softmax(logits.float(), -1).gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
            nll += float((t_nll * ((tgt != BOS) & (tgt != EOS))).sum())
        if limit:  # 自测时只算了部分窗口：按窗口比例折算字节数，数值只用来检查流程
            nbytes = nbytes * len(wins) / max(1, len(e['wins']))
        bpb[src] = round(nll / math.log(2) / max(1.0, nbytes), 4)
    w = EVAL_WEIGHTS
    res = {'bpb': bpb, 'bpb_weighted': round(sum(bpb[s] * w[s] for s in w if s in bpb) / sum(w[s] for s in w if s in bpb), 4)}
    if arith:
        res.update(eval_arith(model, tok, device, arith_limit))
    model.train(was_training)
    return res


@torch.no_grad()
def max_attention_logits(model, device, n=2, seq_len=1024):
    """各层注意力打分 q·k/√d 的最大绝对值（取验证集前 n 个窗口）。Kimi K2 报告 Muon 下这一项更容易爆炸。"""
    _, streams = eval_streams(seq_len)
    wins = [w for e in streams.values() for w in e['wins'] if len(w) == seq_len + 1][:n]
    x = torch.stack(wins)[:, :-1].to(device)
    caps = []
    hooks = [layer.attention.register_forward_pre_hook(lambda mod, args: caps.append(args[0].detach())) for layer in model.layers]
    try:
        with amp(device):
            model(x)
    finally:
        for h in hooks:
            h.remove()
    out = []
    for layer, h in zip(model.layers, caps):
        a, cfg = layer.attention, layer.attention.config
        h = h.float()
        b, t, _ = h.shape
        q = a.q_proj(h).view(b, t, cfg.num_heads, cfg.head_dim).transpose(1, 2)
        k = a.k_proj(h).view(b, t, cfg.num_kv_heads, cfg.head_dim).transpose(1, 2)
        q, k = a.rope(q), a.rope(k)
        k = k.repeat_interleave(cfg.num_heads // cfg.num_kv_heads, dim=1)
        s = (q @ k.transpose(-1, -2)) / math.sqrt(cfg.head_dim)
        mask = torch.ones(t, t, dtype=torch.bool, device=s.device).tril()
        out.append(round(float(s.masked_fill(~mask, 0.0).abs().amax()), 2))
    return out


# ------------------------------------------------------------------ 训练 ----------
def run_dir(name):
    return LAB / 'runs' / name


def milestone_path(name, tokens):
    return LAB_CKPT / name / f'{int(tokens)}.pt'


def free_gb(path):
    import shutil
    p = Path(path)
    while not p.exists():
        p = p.parent
    return shutil.disk_usage(p).free / 1e9


def _save(path, obj, need_gb):
    path.parent.mkdir(parents=True, exist_ok=True)
    if free_gb(path.parent) < need_gb:
        raise RuntimeError(f'{path.parent} 可用空间 {free_gb(path.parent):.1f}GB，不够保存检查点（约需 {need_gb:.1f}GB）')
    tmp = path.with_name(path.name + '.tmp')
    try:
        torch.save(obj, tmp)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    os.replace(tmp, path)


def train(run, device=None, stop=lambda: False, eval_limit=None, arith_limit=None):
    """训练（或从里程碑分支继续训练）一个实验组，结束后评估。结果写入 runs/<name>/result.json 并返回。

    run 字段：name, shape, param(sp|mup), mup{base_width, alpha_in, alpha_out, sigma}, opt{name, lr, wd, betas, eps, ...},
      sched{kind, warmup(步), decay_from(token) 或 decay_fraction, shape, final}, data（数据集名）, weights（可选配比）,
      tokens（训练到的总 token 数，分支也按全局 token 计）, seed, seq_len, micro, accum, clip,
      branch_from{run, tokens}, milestones[token...], eval(bool), eval_at_milestones(bool), arith(bool), compile(bool)。
    """
    from .trainer_agent import MixtureSampler
    dev = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    name = run['name']
    out = run_dir(name)
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'result.json').exists():
        return read_json(out / 'result.json')
    T, micro, accum = int(run.get('seq_len', 1024)), int(run.get('micro', 64)), int(run.get('accum', 4))
    tps = T * micro * accum
    end_step = int(int(run['tokens']) // tps)
    sched = sched_steps(run, end_step, tps)
    peak = float(run['opt']['lr'])
    clip = float(run.get('clip', 1.0))
    if dev.type == 'cuda':
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    torch.manual_seed(int(run['seed']))
    model, m = build_model(run)
    model.to(dev)
    opt = build_optimizer(model, run['opt'], m, dev)
    data = load_dataset(run['data'])
    sampler = MixtureSampler(data, run.get('weights') or data['weights'], T, int(run['seed']))
    st = {'step': 0, 'start_step': 0, 't': 0.0, 'clipped': 0, 'bad': 0, 'ratios': [], 'urms': [], 'losses': [], 'milestone_evals': {}}
    latest = LAB_SCRATCH / 'latest' / f'{name}.pt'
    src = latest if latest.exists() else None
    if src is None and run.get('branch_from'):
        bf = run['branch_from']
        src = milestone_path(bf['run'], bf['tokens'])
        if not src.exists():
            raise FileNotFoundError(f'分支起点不存在：{src}')
    if src is not None:
        ck = torch.load(src, map_location='cpu', weights_only=False)
        model.load_state_dict(ck['model'])
        opt.load_state_dict(ck['opt'])
        sampler.load_state_dict(ck['sampler'])
        if src == latest:
            st.update(ck['st'])
        else:
            st['step'] = st['start_step'] = int(ck['st']['step'])
            if st['step'] != int(int(run['branch_from']['tokens']) // tps):
                raise RuntimeError(f'里程碑步数 {st["step"]} 与分支起点 {run["branch_from"]["tokens"]} 不一致')
        del ck
    plist = list(model.named_parameters())
    kinds = {n: param_kind(n, p) for n, p in plist}
    nparams = model.count_parameters()
    ck_gb = nparams * 4 * {'soap': 8, 'muon': 2, 'sf_adamw': 3}.get(run['opt'].get('name'), 3) / 1e9 + 1.5  # 权重 + 优化器状态的粗估（SOAP 还有每个维度的 GG 和特征基）
    ms = {int(int(t) // tps): int(t) for t in run.get('milestones', [])}
    cmodel = torch.compile(model) if (run.get('compile', True) and dev.type == 'cuda') else model

    def snapshot():
        return {'model': model.state_dict(), 'opt': opt.state_dict(), 'sampler': sampler.state_dict(),
                'rng': torch.get_rng_state(), 'st': json.loads(json.dumps(st)), 'run': name}

    opt.train()
    model.train()
    logf = (out / 'train_log.jsonl').open('a', encoding='utf-8')
    last_save, diverged = time.time(), False
    try:
        while st['step'] < end_step:
            if stop():
                _save(latest, snapshot(), ck_gb)
                raise StopJob(name)
            t0 = time.perf_counter()
            lr = peak * lr_factor(st['step'], sched)
            opt.set_lr(lr)
            tot = torch.zeros((), device=dev)
            for _ in range(accum):
                b = sampler.batch(micro)
                b = b.pin_memory().to(dev, non_blocking=True) if dev.type == 'cuda' else b
                with amp(dev):
                    loss = cmodel(b[:, :-1], targets=b[:, 1:]).loss / accum
                loss.backward()
                tot += loss.detach()
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            lv, gv = float(tot), float(gn)
            if not (math.isfinite(lv) and math.isfinite(gv)):
                opt.zero_grad()
                st['bad'] += 1
                if st['bad'] >= 20:
                    diverged = True
                    break
                continue
            st['bad'] = 0
            st['clipped'] += int(gv > clip)
            measure = st['step'] % 200 == 0
            if measure:
                before = {n: p.detach().clone() for n, p in plist}
                glr = {id(p): g['lr'] for g in opt.param_groups for p in g['params']}
            opt.step()
            opt.zero_grad()
            ratio = None
            if measure:
                num = den = 0.0
                acc = {}
                for n, p in plist:
                    d = (p.detach() - before[n]).float()
                    if kinds[n] != 'vector':
                        num += float((d ** 2).sum())
                        den += float((before[n].float() ** 2).sum())
                    a = acc.setdefault(kinds[n], [0.0, 0, 0.0])
                    a[0] += float((d ** 2).sum())
                    a[1] += d.numel()
                    a[2] = glr.get(id(p), lr)
                ratio = math.sqrt(num) / max(math.sqrt(den), 1e-12)
                st['ratios'].append(ratio)
                st['urms'].append({k: math.sqrt(v[0] / max(1, v[1])) / max(v[2], 1e-12) for k, v in acc.items()})
                del before
            st['step'] += 1
            dt = time.perf_counter() - t0
            st['t'] += dt
            st['losses'] = (st['losses'] + [lv])[-200:]
            if st['step'] % 50 == 0 or st['step'] == end_step or measure:
                rec = {'time': stamp(), 'step': st['step'], 'total': end_step, 'loss': round(lv, 4), 'lr': lr, 'grad_norm': round(gv, 3),
                       'tok_s': round(tps / dt)}
                if ratio is not None:
                    rec['update_ratio'] = float(f'{ratio:.3e}')
                    rec['update_rms'] = {k: round(v, 4) for k, v in st['urms'][-1].items()}
                logf.write(json.dumps(rec) + '\n')
                logf.flush()
                heartbeat(f'{name} {st["step"]}/{end_step}')
            if st['step'] in ms:
                if run.get('eval_at_milestones'):
                    opt.eval()
                    e = evaluate(model, dev, limit=eval_limit)
                    opt.train()
                    st['milestone_evals'][str(ms[st['step']])] = {'bpb_weighted': e['bpb_weighted'], 'bpb': e['bpb'],
                                                                  'train_loss': round(float(np.mean(st['losses'][-50:])), 4)}
                    logf.write(json.dumps({'time': stamp(), 'step': st['step'], 'milestone': ms[st['step']], 'eval': e}) + '\n')
                    logf.flush()
                _save(milestone_path(name, ms[st['step']]), snapshot(), ck_gb)
            if time.time() - last_save > 1800 and st['step'] < end_step:
                _save(latest, snapshot(), ck_gb)
                last_save = time.time()
    finally:
        logf.close()
    steps_here = st['step'] - st['start_step']
    info = {'steps': st['step'], 'end_step': end_step, 'tokens': st['step'] * tps, 'start_tokens': st['start_step'] * tps,
            'train_hours': round(st['t'] / 3600, 3), 'tok_s': round(steps_here * tps / max(1e-6, st['t'])), 'diverged': diverged,
            'final_loss': round(float(np.mean(st['losses'][-50:])), 4) if st['losses'] else None,
            'clip_fraction': round(st['clipped'] / max(1, steps_here), 4),
            'update_ratio_median': float(f'{np.median(st["ratios"]):.3e}') if st['ratios'] else None,
            'update_rms_median': {k: round(float(np.median([u[k] for u in st['urms'] if k in u])), 4)
                                  for k in ('hidden', 'embed', 'vector') if any(k in u for u in st['urms'])},
            'params': nparams, 'm': m, 'peak_lr': peak, 'milestone_evals': st['milestone_evals']}
    res = {'name': name, 'train': info, 'diverged': diverged}
    if not diverged and run.get('eval', True):
        opt.eval()
        res.update(evaluate(model, dev, arith=bool(run.get('arith')), limit=eval_limit, arith_limit=arith_limit))
        res['max_attn_logit'] = max_attention_logits(model, dev)
        opt.train()
    write_json(out / 'result.json', res)
    latest.unlink(missing_ok=True)
    return res


# ------------------------------------------------------------------ 坐标检查 ----------
def coord_check(widths, depth=2, params=('sp', 'mup'), steps=10, seeds=(0, 1, 2), lr=2 ** -7, data_windows=None, micro=8,
                seq_len=256, device=None, base_width=256):
    """μP 实现是否正确的第一道检查：几种宽度各训练 steps 步，记录每类层输出的平均绝对值。μP 下各宽度应基本重合。
    返回 {param: {width: {kind: [每步的值（各种子平均）]}}} 和每类层在最后一步对 log(宽度) 的斜率。"""
    dev = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    if data_windows is None:
        _, streams = eval_streams(1024)
        data_windows = torch.stack([w for e in streams.values() for w in e['wins'] if len(w) == 1025][:256])
    data = data_windows[:, :seq_len + 1]
    out = {}
    for param in params:
        out[param] = {}
        for width in widths:
            acc = {}
            for seed in seeds:
                run = {'shape': {'width': width, 'depth': depth}, 'param': param, 'mup': {'base_width': base_width}}
                torch.manual_seed(seed)
                model, m = build_model(run)
                model.to(dev).train()
                opt = build_optimizer(model, {'name': 'adamw', 'lr': lr, 'wd': 0.0}, m, dev)
                rec = {}
                hooks = [model.embedding.register_forward_hook(lambda mod, i, o: rec.setdefault('embed', []).append(float(o.detach().abs().mean())))]
                for layer in model.layers:
                    hooks.append(layer.attention.o_proj.register_forward_hook(lambda mod, i, o: rec.setdefault('attn', []).append(float(o.detach().abs().mean()))))
                    hooks.append(layer.ffn.down_proj.register_forward_hook(lambda mod, i, o: rec.setdefault('ffn', []).append(float(o.detach().abs().mean()))))
                    hooks.append(layer.register_forward_hook(lambda mod, i, o: rec.setdefault('block', []).append(float(o[0].detach().abs().mean()))))
                g = torch.Generator().manual_seed(1000 + seed)
                for t in range(steps):
                    rec.clear()
                    idx = torch.randint(0, data.size(0), (micro,), generator=g)
                    b = data[idx].to(dev)
                    with amp(dev):
                        o = model(b[:, :-1], targets=b[:, 1:])
                    rec['logits'] = [float(o.logits.detach().float().abs().mean())]
                    o.loss.backward()
                    opt.step()
                    opt.zero_grad()
                    for kind, vals in rec.items():
                        acc.setdefault(kind, np.zeros(steps))[t] += float(np.mean(vals)) / len(seeds)
                for h in hooks:
                    h.remove()
                del model, opt
            out[param][width] = {k: [round(float(x), 6) for x in v] for k, v in acc.items()}
    slopes = {}
    lw = np.log(np.array(widths, dtype=float))
    for param in params:
        slopes[param] = {}
        for kind in out[param][widths[0]]:
            y = np.log(np.array([out[param][w][kind][-1] for w in widths]) + 1e-12)
            slopes[param][kind] = round(float(np.polyfit(lw, y, 1)[0]), 3)
    return out, slopes
