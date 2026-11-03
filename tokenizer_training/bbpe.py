"""纯标准库 BBPE：UTF-8 字节、带权相邻对统计、增量合并及按合并优先级编码。"""

from collections import Counter, defaultdict
import heapq
import json
from pathlib import Path
import unicodedata

SPECIAL_TOKENS = ['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]']
BYTE_OFFSET = len(SPECIAL_TOKENS)
BASE_VOCAB_SIZE = BYTE_OFFSET + 256


def split_text(text):
    """无损按 Unicode 类别连续分段；不跨字母/数字/空白/其他类别合并。

    字母与组合标记归为一类（包括汉字）。这是明确的自定义边界规则，
    不声称复制 GPT-2 或 Hugging Face ByteLevel 的正则表达式。
    """
    start, previous = 0, None
    for i, char in enumerate(text):
        category = unicodedata.category(char)[0]
        kind = 'space' if char.isspace() else 'letter' if category in ('L', 'M') else 'number' if category == 'N' else 'other'
        if previous is not None and kind != previous:
            yield text[start:i]
            start = i
        previous = kind
    if text:
        yield text[start:]


def merge_pair(sequence, pair, new_id):
    """从左到右非重叠替换；例如 a a a -> aa a，而不是重复使用中间 a。"""
    out, i = [], 0
    while i < len(sequence):
        if i + 1 < len(sequence) and (sequence[i], sequence[i + 1]) == pair:
            out.append(new_id)
            i += 2
        else:
            out.append(sequence[i])
            i += 1
    return tuple(out)


def validate_settings(vocab_size, min_frequency, lowercase, model_max_length):
    for name, value, minimum in [('vocab_size', vocab_size, BASE_VOCAB_SIZE),
                                  ('min_frequency', min_frequency, 1), ('model_max_length', model_max_length, 3)]:
        if type(value) is not int or value < minimum:
            raise ValueError(f'{name} 必须为不小于 {minimum} 的整数')
    if type(lowercase) is not bool:
        raise ValueError('lowercase 必须为布尔值')


class BBPETokenizer:
    pad_token_id, unk_token_id, cls_token_id, sep_token_id, mask_token_id = range(5)
    is_fast = False

    def __init__(self, merges=(), lowercase=False, model_max_length=512):
        validate_settings(BASE_VOCAB_SIZE, 1, lowercase, model_max_length)
        self.lowercase, self.model_max_length = lowercase, model_max_length
        self.merges, self._ranks = [], {}
        self._bytes = [None] * BYTE_OFFSET + [bytes([b]) for b in range(256)]
        for pair in merges:
            if (not isinstance(pair, (list, tuple)) or len(pair) != 2
                    or any(type(i) is not int or not BYTE_OFFSET <= i < len(self._bytes) for i in pair)):
                raise ValueError('无效合并规则：只能引用字节词元或之前生成的词元')
            pair = tuple(pair)
            if pair in self._ranks:
                raise ValueError('重复合并规则')
            self._ranks[pair] = len(self.merges)
            self.merges.append(pair)
            self._bytes.append(self._bytes[pair[0]] + self._bytes[pair[1]])

    def __len__(self):
        return len(self._bytes)

    @property
    def vocab_size(self):
        return len(self)

    @classmethod
    def train(cls, texts, vocab_size=32000, min_frequency=2, lowercase=False, model_max_length=512, progress=None,
              backend='memory', work_dir=None, status=None):
        validate_settings(vocab_size, min_frequency, lowercase, model_max_length)
        if backend == 'disk':
            from tokenizer_training.disk_trainer import train_disk
            return train_disk(texts, vocab_size, min_frequency, lowercase, model_max_length,
                              progress=progress, status=status, work_dir=work_dir)
        if backend != 'memory':
            raise ValueError('backend 必须为 disk 或 memory')
        words = Counter()
        text_count = 0
        for text in texts:
            if not isinstance(text, str):
                raise ValueError('训练文本必须为字符串')
            text_count += 1
            for piece in split_text(text.lower() if lowercase else text):
                words[tuple(BYTE_OFFSET + b for b in piece.encode('utf-8'))] += 1
        if not words:
            raise ValueError('训练语料为空')
        sequences, weights = list(words), list(words.values())
        del words
        if status:
            status(f'唯一片段 {len(sequences)}；开始建立内存相邻对索引')
        counts, positions = Counter(), defaultdict(set)
        for word_id, sequence in enumerate(sequences):
            for pair, occurrences in Counter(zip(sequence, sequence[1:])).items():
                counts[pair] += occurrences * weights[word_id]
                positions[pair].add(word_id)
        # 频次高者优先；同频时按两个 token ID 排序，以保证确定性。
        heap = [(-count, pair) for pair, count in counts.items()]
        heapq.heapify(heap)
        if status:
            status(f'内存索引完成：唯一相邻对 {len(counts)}')
        merges = []
        if progress:
            progress(0, None, 0)
        while heap and BASE_VOCAB_SIZE + len(merges) < vocab_size:
            negative, pair = heapq.heappop(heap)
            frequency = -negative
            if counts.get(pair, 0) != frequency:
                continue  # 懒删除过期的优先队列项。
            if frequency < min_frequency:
                break
            new_id = BASE_VOCAB_SIZE + len(merges)
            changed = set()
            for word_id in sorted(positions[pair]):
                old = sequences[word_id]
                new = merge_pair(old, pair, new_id)
                before, after = Counter(zip(old, old[1:])), Counter(zip(new, new[1:]))
                for old_pair, n in before.items():
                    counts[old_pair] -= n * weights[word_id]
                    positions[old_pair].discard(word_id)
                    if not counts[old_pair]:
                        del counts[old_pair]
                    if not positions[old_pair]:
                        del positions[old_pair]
                for new_pair, n in after.items():
                    counts[new_pair] += n * weights[word_id]
                    positions[new_pair].add(word_id)
                changed.update(before)
                changed.update(after)
                sequences[word_id] = new
            merges.append(pair)
            for changed_pair in changed:
                if counts.get(changed_pair, 0) > 0:
                    heapq.heappush(heap, (-counts[changed_pair], changed_pair))
            # 限制懒删除积累的历史项，避免随合并次数无限增长。
            if len(heap) > 4 * len(counts) + 1024:
                heap = [(-count, candidate) for candidate, count in counts.items()]
                heapq.heapify(heap)
            if progress:
                progress(len(merges), pair, frequency)
        result = cls(merges, lowercase=lowercase, model_max_length=model_max_length)
        result.training_stats = {'texts': text_count, 'unique_pieces': len(sequences), 'merges': len(merges), 'backend': 'memory'}
        return result

    def _encode_text(self, text):
        if not isinstance(text, str):
            raise ValueError('输入必须为字符串')
        output = []
        for piece in split_text(text.lower() if self.lowercase else text):
            sequence = tuple(BYTE_OFFSET + b for b in piece.encode('utf-8'))
            while len(sequence) > 1:
                candidates = [(self._ranks[pair], pair) for pair in zip(sequence, sequence[1:]) if pair in self._ranks]
                if not candidates:
                    break
                rank, pair = min(candidates)
                sequence = merge_pair(sequence, pair, BASE_VOCAB_SIZE + rank)
            output.extend(sequence)
        return output

    def encode(self, text, add_special_tokens=True):
        ids = self._encode_text(text)
        return [self.cls_token_id] + ids + [self.sep_token_id] if add_special_tokens else ids

    def decode_bytes(self, ids, skip_special_tokens=False):
        chunks = []
        for token_id in ids:
            if type(token_id) is not int or not 0 <= token_id < len(self):
                raise ValueError(f'无效 token ID：{token_id}')
            if token_id < BYTE_OFFSET:
                if not skip_special_tokens:
                    chunks.append(SPECIAL_TOKENS[token_id].encode('utf-8'))
            else:
                chunks.append(self._bytes[token_id])
        return b''.join(chunks)

    def decode(self, ids, skip_special_tokens=False, errors='replace'):
        # 先拼完整字节再解码，单个 token 可能仅含多字节字符的一部分。
        return self.decode_bytes(ids, skip_special_tokens).decode('utf-8', errors=errors)

    def convert_ids_to_tokens(self, ids):
        tokens = []
        for token_id in ids:
            if type(token_id) is not int or not 0 <= token_id < len(self):
                raise ValueError(f'无效 token ID：{token_id}')
            tokens.append(SPECIAL_TOKENS[token_id] if token_id < BYTE_OFFSET else 'bytes:' + self._bytes[token_id].hex())
        return tokens

    def __call__(self, text, text_pair=None, add_special_tokens=True, padding=False, truncation=False,
                 max_length=None, verbose=False):
        if padding not in (False, True, 'max_length') or type(truncation) is not bool:
            raise ValueError('padding 支持 false/true/max_length；truncation 必须为布尔值')
        length = self.model_max_length if max_length is None else max_length
        if type(length) is not int or length < 1:
            raise ValueError('max_length 必须为正整数')
        single = isinstance(text, str)
        texts = [text] if single else text
        if not isinstance(texts, (list, tuple)) or any(not isinstance(t, str) for t in texts):
            raise ValueError('输入必须为字符串或字符串列表')
        pairs = [text_pair] if single else ([None] * len(texts) if text_pair is None else text_pair)
        if not isinstance(pairs, (list, tuple)) or len(pairs) != len(texts) or any(p is not None and not isinstance(p, str) for p in pairs):
            raise ValueError('句对必须与输入逐条对应')
        rows = []
        for a_text, b_text in zip(texts, pairs):
            a, b = self._encode_text(a_text), self._encode_text(b_text) if b_text is not None else []
            specials = (3 if b_text is not None else 2) if add_special_tokens else 0
            if truncation:
                if length < specials:
                    raise ValueError('max_length 容纳不下特殊词元')
                while len(a) + len(b) > length - specials:
                    (a if len(a) >= len(b) else b).pop()
            first = [2] + a + [3] if add_special_tokens else a
            second = (b + [3] if add_special_tokens else b) if b_text is not None else []
            ids = first + second
            rows.append({'input_ids': ids, 'token_type_ids': [0] * len(first) + [1] * len(second),
                         'attention_mask': [1] * len(ids)})
        target = length if padding == 'max_length' else max((len(r['input_ids']) for r in rows), default=0)
        if padding:
            for row in rows:
                n = max(0, target - len(row['input_ids']))
                for key in row:
                    row[key].extend([0] * n)
        return rows[0] if single else {key: [row[key] for row in rows] for key in ('input_ids', 'token_type_ids', 'attention_mask')}

    def save_pretrained(self, output_dir):
        root = Path(output_dir)
        root.mkdir(parents=True, exist_ok=True)
        data = {'format': 'scratch-bbpe', 'version': 1, 'pretokenizer': 'unicode-category-v1',
                'special_tokens': SPECIAL_TOKENS, 'byte_offset': BYTE_OFFSET,
                'lowercase': self.lowercase, 'model_max_length': self.model_max_length, 'merges': self.merges}
        (root / 'bbpe.json').write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
        vocab = {str(i): {'special': SPECIAL_TOKENS[i]} if i < BYTE_OFFSET else {'hex': self._bytes[i].hex()}
                 for i in range(len(self))}
        (root / 'vocab.json').write_text(json.dumps(vocab, ensure_ascii=False, indent=2), encoding='utf-8')

    @classmethod
    def from_pretrained(cls, model_dir):
        path = Path(model_dir) / 'bbpe.json'
        if not path.is_file():
            raise ValueError(f'{path} 不存在；请加载本工程从零实现生成的模型，旧版 tokenizer.json 不兼容')
        data = json.loads(path.read_text(encoding='utf-8'))
        if (not isinstance(data, dict) or data.get('format') != 'scratch-bbpe' or type(data.get('version')) is not int
                or data['version'] != 1 or data.get('pretokenizer') != 'unicode-category-v1'
                or data.get('special_tokens') != SPECIAL_TOKENS or data.get('byte_offset') != BYTE_OFFSET
                or not isinstance(data.get('merges'), list) or 'lowercase' not in data or 'model_max_length' not in data):
            raise ValueError('无效或不支持的 BBPE 模型格式')
        return cls(data['merges'], data['lowercase'], data['model_max_length'])
