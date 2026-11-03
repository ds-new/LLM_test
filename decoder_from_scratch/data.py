"""无外部依赖的字符级分词和随机连续块训练数据。"""

from dataclasses import dataclass
from pathlib import Path

import torch


@dataclass
class CharTokenizer:
    tokens: list[str]
    pad_id = 0
    unk_id = 1
    bos_id = 2
    eos_id = 3
    SPECIAL = ['<pad>', '<unk>', '<bos>', '<eos>']

    def __post_init__(self):
        if self.tokens[:4] != self.SPECIAL or len(set(self.tokens)) != len(self.tokens):
            raise ValueError('invalid character vocabulary')
        self.token_to_id = {token: i for i, token in enumerate(self.tokens)}

    @classmethod
    def from_text(cls, text):
        return cls(cls.SPECIAL + sorted(set(text)))

    def encode(self, text, add_bos=False, add_eos=False):
        ids = [self.token_to_id.get(char, self.unk_id) for char in text]
        return ([self.bos_id] if add_bos else []) + ids + ([self.eos_id] if add_eos else [])

    def decode(self, ids):
        # 未知字符显式显示，其他特殊符号不输出。
        return ''.join('�' if int(i) == self.unk_id else self.tokens[int(i)]
                       for i in ids if int(i) not in (self.pad_id, self.bos_id, self.eos_id))

    def to_dict(self):
        return {'type': 'character', 'tokens': self.tokens}

    @classmethod
    def from_dict(cls, data):
        if data['type'] != 'character':
            raise ValueError('unsupported tokenizer type')
        return cls(data['tokens'])


def prepare_corpus(path, val_fraction=0.1, val_path=None):
    """先切分原文，再仅用训练集建词表；每行是一个 BOS...EOS 文档。

    无独立验证文件时，按行保留最后一部分。仅一行时按字符位置切分。
    文档打包成一维 token 流，因此允许模型看到前一个文档的上下文。
    """
    if not 0 < val_fraction < 1:
        raise ValueError('val_fraction must be in (0, 1)')
    text = Path(path).read_text(encoding='utf-8')
    if val_path:
        train_text, val_text = text, Path(val_path).read_text(encoding='utf-8')
    else:
        lines = text.splitlines(keepends=True)
        if len(lines) >= 2:
            split = min(len(lines) - 1, max(1, int(len(lines) * (1 - val_fraction))))
            train_text, val_text = ''.join(lines[:split]), ''.join(lines[split:])
        else:
            split = int(len(text) * (1 - val_fraction))
            train_text, val_text = text[:split], text[split:]
    if not train_text.strip() or not val_text.strip():
        raise ValueError('both training and validation text must be nonempty')
    tokenizer = CharTokenizer.from_text(train_text)

    def pack(raw):
        ids = []
        for line in raw.splitlines(keepends=True):
            ids.extend(tokenizer.encode(line, add_bos=True, add_eos=True))
        return torch.tensor(ids, dtype=torch.long)

    return tokenizer, pack(train_text), pack(val_text)


def sample_batch(tokens, batch_size, seq_len, generator):
    if batch_size < 1 or seq_len < 1 or tokens.numel() <= seq_len:
        raise ValueError('need positive batch_size/seq_len and more than seq_len tokens in each split')
    starts = torch.randint(tokens.numel() - seq_len, (batch_size,), generator=generator)
    indices = starts[:, None] + torch.arange(seq_len + 1)[None, :]
    chunks = tokens[indices]
    return chunks[:, :-1].contiguous(), chunks[:, 1:].contiguous()
