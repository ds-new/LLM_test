"""模型配置；head_n 对应 num_heads，即 Query 头数量。"""

from dataclasses import asdict, dataclass
import json
from pathlib import Path


@dataclass
class ModelConfig:
    vocab_size: int = 256  # 训练时由训练集字符词表覆盖
    hidden_size: int = 768
    num_layers: int = 8
    num_heads: int = 8
    num_kv_heads: int = 2
    intermediate_size: int = 2048
    max_seq_len: int = 2048
    rope_theta: float = 10000.0
    rms_norm_eps: float = 1e-6
    dropout: float = 0.0
    tie_embeddings: bool = True
    attention_backend: str = 'sdpa'

    def __post_init__(self):
        for name in ('vocab_size', 'hidden_size', 'num_layers', 'num_heads',
                     'num_kv_heads', 'intermediate_size', 'max_seq_len'):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f'{name} must be a positive integer')
        if self.hidden_size % self.num_heads:
            raise ValueError('hidden_size must be divisible by num_heads')
        if self.num_heads % self.num_kv_heads:
            raise ValueError('num_heads must be divisible by num_kv_heads')
        if self.head_dim % 2:
            raise ValueError('RoPE requires an even head_dim')
        if not 0 <= self.dropout < 1:
            raise ValueError('dropout must be in [0, 1)')
        if not self.rope_theta > 0 or not self.rms_norm_eps > 0:
            raise ValueError('rope_theta and rms_norm_eps must be positive')
        if self.attention_backend not in ('manual', 'sdpa'):
            raise ValueError('attention_backend must be manual or sdpa')

    @property
    def head_dim(self):
        return self.hidden_size // self.num_heads

    @classmethod
    def from_json(cls, path):
        return cls(**json.loads(Path(path).read_text(encoding='utf-8')))

    def to_dict(self):
        return asdict(self)
