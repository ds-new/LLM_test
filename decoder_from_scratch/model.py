"""Embedding → [Pre-RMSNorm + RoPE/GQA + residual + Pre-RMSNorm/SwiGLU + residual] × L.

记号：B=batch，T=本次输入长度，S=历史长度+T，Hq=Query 头数，Hkv=KV 头数，D=每头维度。
Cache 仅存旋转后的 K 和未旋转的 V，形状均为 [B, Hkv, S, D]。
"""

from dataclasses import dataclass
import math

import torch
from torch import nn
import torch.nn.functional as F

from .config import ModelConfig

KVCache = list[tuple[torch.Tensor, torch.Tensor]]


class RMSNorm(nn.Module):
    """RMSNorm(x) = weight * x / sqrt(mean(x²) + eps)，不减均值。"""

    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x):
        # 半精度下使用 float32 求均方，降低溢出风险。
        normalized = x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + self.eps)
        return (normalized * self.weight.float()).to(x.dtype)


class RotaryEmbedding(nn.Module):
    """相邻偶/奇维成对旋转；位置从 start_pos 开始，缓存解码不能每次从 0 开始。"""

    def __init__(self, head_dim, max_seq_len, theta):
        super().__init__()
        inv_freq = theta ** (-torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
        angles = torch.outer(torch.arange(max_seq_len, dtype=torch.float32), inv_freq)
        self.register_buffer('cos', angles.cos(), persistent=False)
        self.register_buffer('sin', angles.sin(), persistent=False)

    def forward(self, x, start_pos=0):
        end = start_pos + x.size(-2)
        cos = self.cos[start_pos:end].to(x.dtype)[None, None, :, :]
        sin = self.sin[start_pos:end].to(x.dtype)[None, None, :, :]
        even, odd = x[..., 0::2], x[..., 1::2]
        return torch.stack((even * cos - odd * sin, even * sin + odd * cos), dim=-1).flatten(-2)


class GroupedQueryAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        h, d = config.hidden_size, config.head_dim
        self.q_proj = nn.Linear(h, config.num_heads * d, bias=False)
        self.k_proj = nn.Linear(h, config.num_kv_heads * d, bias=False)
        self.v_proj = nn.Linear(h, config.num_kv_heads * d, bias=False)
        self.o_proj = nn.Linear(h, h, bias=False)
        self.rope = RotaryEmbedding(d, config.max_seq_len, config.rope_theta)

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

        # 每组 Query 共享一对 KV；仅在计算 Attention 时扩展，Cache 保持 Hkv 个头。
        groups = cfg.num_heads // cfg.num_kv_heads
        k = k.repeat_interleave(groups, dim=1)
        v = v.repeat_interleave(groups, dim=1)
        dropout = cfg.dropout if self.training else 0.0

        # 带历史时矩阵非方形，不能直接使用上左对齐的 is_causal=True。
        # 第 i 个新 Query 的绝对位置是 past_len+i，只允许看 <= 该位置的 Key。
        mask = None
        if cfg.attention_backend == 'manual' or (past_len > 0 and t > 1):
            q_positions = past_len + torch.arange(t, device=x.device)
            k_positions = torch.arange(k.size(2), device=x.device)
            mask = k_positions[None, :] <= q_positions[:, None]
        if cfg.attention_backend == 'manual':
            scores = (q.float() @ k.float().transpose(-1, -2)) / math.sqrt(cfg.head_dim)
            scores = scores.masked_fill(~mask, float('-inf'))
            weights = F.dropout(scores.softmax(dim=-1).to(v.dtype), dropout, self.training)
            out = weights @ v
        else:
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask,
                                                dropout_p=dropout, is_causal=(past_len == 0))
        out = out.transpose(1, 2).contiguous().view(b, t, cfg.hidden_size)
        return self.o_proj(out), present


class SwiGLU(nn.Module):
    """FFN(x) = W_down [SiLU(W_gate x) ⊙ W_up x]。"""

    def __init__(self, config):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class DecoderBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attn_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.attention = GroupedQueryAttention(config)
        self.ffn_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.ffn = SwiGLU(config)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x, past=None, use_cache=False):
        attended, present = self.attention(self.attn_norm(x), past, use_cache)
        x = x + self.dropout(attended)  # 第一条残差：Attention
        x = x + self.dropout(self.ffn(self.ffn_norm(x)))  # 第二条残差：FFN
        return x, present


@dataclass
class LMOutput:
    logits: torch.Tensor
    loss: torch.Tensor | None = None
    past_key_values: KVCache | None = None


class DecoderLM(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([DecoderBlock(config) for _ in range(config.num_layers)])
        self.final_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.apply(self._init_weights)
        if config.tie_embeddings:
            self.lm_head.weight = self.embedding.weight
        # 缩小残差分支初始方差，让多层累加更稳定。
        for layer in self.layers:
            nn.init.normal_(layer.attention.o_proj.weight, std=0.02 / math.sqrt(2 * config.num_layers))
            nn.init.normal_(layer.ffn.down_proj.weight, std=0.02 / math.sqrt(2 * config.num_layers))

    @staticmethod
    def _init_weights(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids, targets=None, past_key_values=None, use_cache=False,
                last_logits_only=False):
        """targets 与 input_ids 同形，已向右错开一位；-100 表示不计算该标签的 loss。

        此示例使用无 padding 的等长训练块；生成接口一次接收一个 prompt。
        Cache 用于 eval 推理；last_logits_only 避免 prefill 时计算无用的词表投影。
        """
        if input_ids.ndim != 2 or input_ids.size(1) == 0:
            raise ValueError('input_ids must have shape [batch, nonempty sequence]')
        if targets is not None and (targets.shape != input_ids.shape or last_logits_only):
            raise ValueError('targets must match input_ids; last_logits_only must be False')
        if (past_key_values is not None and not use_cache) or (use_cache and (self.training or targets is not None)):
            raise ValueError('KV cache requires eval mode, use_cache=True and no targets')
        past_len = 0
        if past_key_values is not None:
            if len(past_key_values) != self.config.num_layers:
                raise ValueError('cache length must equal num_layers')
            past_len = past_key_values[0][0].size(2)
            shape = (input_ids.size(0), self.config.num_kv_heads, past_len, self.config.head_dim)
            for k, v in past_key_values:
                if k.shape != shape or v.shape != shape or k.device != input_ids.device or v.device != input_ids.device:
                    raise ValueError('cache batch/head/sequence dimensions or device do not match')
        if past_len + input_ids.size(1) > self.config.max_seq_len:
            raise ValueError('sequence + cache exceeds max_seq_len')
        x = self.embedding(input_ids)
        cache = [] if use_cache else None
        for i, layer in enumerate(self.layers):
            past = None if past_key_values is None else past_key_values[i]
            x, present = layer(x, past, use_cache)
            if use_cache:
                cache.append(present)
        if last_logits_only:
            x = x[:, -1:]
        logits = self.lm_head(self.final_norm(x))
        loss = None if targets is None else F.cross_entropy(
            logits.float().reshape(-1, self.config.vocab_size), targets.reshape(-1), ignore_index=-100)
        return LMOutput(logits=logits, loss=loss, past_key_values=cache)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters())
