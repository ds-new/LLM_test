"""本地 BPE 与 embedding_model_train/tokenizer_training 的 scratch-bbpe v1 适配。"""

import json
from pathlib import Path
import unicodedata

from .data import CharTokenizer


class ByteBPETokenizer:
    pad_id, unk_id, bos_id, eos_id = 0, 1, 2, 3

    def __init__(self, backend):
        self.backend = backend
        self.tokens = [backend.id_to_token(i) for i in range(backend.get_vocab_size())]
        if self.tokens[:4] != CharTokenizer.SPECIAL:
            raise ValueError('BPE special token IDs must be PAD=0, UNK=1, BOS=2, EOS=3')

    @classmethod
    def train(cls, texts, vocab_size=8192):
        from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders
        if not 260 <= vocab_size <= 65535:
            raise ValueError('byte BPE vocab_size must be in [260, 65535]')
        backend = Tokenizer(models.BPE(unk_token='<unk>'))
        backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        backend.decoder = decoders.ByteLevel()
        trainer = trainers.BpeTrainer(vocab_size=vocab_size, min_frequency=2,
                                      initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
                                      special_tokens=CharTokenizer.SPECIAL, show_progress=False)
        backend.train_from_iterator(texts, trainer=trainer)
        return cls(backend)

    def encode(self, text, add_bos=False, add_eos=False):
        return ([self.bos_id] if add_bos else []) + self.backend.encode(text).ids + ([self.eos_id] if add_eos else [])

    def decode(self, ids):
        return self.backend.decode([int(i) for i in ids], skip_special_tokens=True)

    def to_dict(self):
        return {'type': 'byte_bpe', 'backend_json': self.backend.to_str()}


class ScratchBBPETokenizer:
    """冻结自研 BBPE 的推理规则；checkpoint 内嵌完整规则，不依赖外部项目路径。

    对应上游 bbpe.py 的 unicode-category-v1 分段和按 rank 合并。
    [CLS]/[SEP] 分别充当 BOS/EOS；不使用 encoder 的 padding/truncation。
    """

    pad_id, unk_id, bos_id, eos_id = 0, 1, 2, 3
    SPECIAL = ['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]']

    def __init__(self, data):
        if (data.get('format') != 'scratch-bbpe' or type(data.get('version')) is not int
                or data['version'] != 1 or data.get('pretokenizer') != 'unicode-category-v1'
                or data.get('special_tokens') != self.SPECIAL or data.get('byte_offset') != 5
                or type(data.get('lowercase')) is not bool
                or type(data.get('model_max_length')) is not int or data['model_max_length'] < 3
                or not isinstance(data.get('merges'), list)):
            raise ValueError('unsupported scratch-bbpe format; expected upstream v1')
        self.data = json.loads(json.dumps(data))
        self._bytes = [None] * 5 + [bytes([i]) for i in range(256)]
        self._ranks = {}
        for rank, pair in enumerate(data['merges']):
            if (not isinstance(pair, (list, tuple)) or len(pair) != 2
                    or any(type(i) is not int or not 5 <= i < len(self._bytes) for i in pair)
                    or tuple(pair) in self._ranks):
                raise ValueError('invalid scratch-bbpe merge rule')
            self._ranks[tuple(pair)] = rank
            self._bytes.append(self._bytes[pair[0]] + self._bytes[pair[1]])
        self.tokens = self.SPECIAL + ['bytes:' + b.hex() for b in self._bytes[5:]]

    @staticmethod
    def _pieces(text):
        start, previous = 0, None
        for i, char in enumerate(text):
            category = unicodedata.category(char)[0]
            kind = ('space' if char.isspace() else 'letter' if category in ('L', 'M')
                    else 'number' if category == 'N' else 'other')
            if previous is not None and kind != previous:
                yield text[start:i]
                start = i
            previous = kind
        if text:
            yield text[start:]

    def encode(self, text, add_bos=False, add_eos=False):
        ids = [self.bos_id] if add_bos else []
        for piece in self._pieces(text.lower() if self.data['lowercase'] else text):
            sequence = [5 + b for b in piece.encode('utf-8')]
            while len(sequence) > 1:
                candidates = [(self._ranks[p], p) for p in zip(sequence, sequence[1:]) if p in self._ranks]
                if not candidates:
                    break
                rank, pair = min(candidates)
                merged, i = [], 0
                while i < len(sequence):
                    if i + 1 < len(sequence) and (sequence[i], sequence[i + 1]) == pair:
                        merged.append(261 + rank)
                        i += 2
                    else:
                        merged.append(sequence[i])
                        i += 1
                sequence = merged
            ids.extend(sequence)
        return ids + ([self.eos_id] if add_eos else [])

    def decode(self, ids):
        return b''.join(self._bytes[int(i)] for i in ids if int(i) >= 5).decode('utf-8', errors='replace')

    def to_dict(self):
        return {'type': 'scratch_bbpe', 'model': self.data}


def load_tokenizer_file(path):
    path = Path(path)
    if path.is_dir():
        path = path / 'bbpe.json'
    return load_tokenizer(json.loads(path.read_text(encoding='utf-8')))


def load_tokenizer(data):
    if data.get('format') == 'scratch-bbpe':
        return ScratchBBPETokenizer(data)
    if data.get('type') == 'scratch_bbpe':
        return ScratchBBPETokenizer(data['model'])
    if data.get('type') == 'character':
        return CharTokenizer.from_dict(data)
    if data.get('type') == 'byte_bpe':
        from tokenizers import Tokenizer
        return ByteBPETokenizer(Tokenizer.from_str(data['backend_json']))
    raise ValueError(f'unsupported tokenizer format/type: {data.get("format", data.get("type"))}')
