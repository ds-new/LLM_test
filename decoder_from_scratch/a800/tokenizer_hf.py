"""把 tokenizer_training 训练出的 BBPE（bbpe.json）等价转换成 HuggingFace tokenizers 格式，用于快速编码。

为什么要转换：原实现是纯 Python，单核每秒约 10 万字；要编码约 250 亿字的语料需要几十个 CPU·小时。
tokenizers 是 Rust 实现，快 20-50 倍。但"等价"必须验证，验证不通过就退回纯 Python 编码（只是慢）。

等价性来自三点：
  1. 预切分：原实现按 Unicode 类别（空白 / 字母+组合标记 / 数字 / 其他）切成连续片段，这里用同义的正则 Split；
  2. 字节映射：原实现 token id = 5 + 字节值；HF 的 ByteLevel 把每个字节映射为一个可见字符（GPT-2 的 bytes_to_unicode），
     我们按同样的 id 把这些字符写进词表；
  3. 合并：两者都是"每次合并当前片段里 rank 最小的相邻对、从左到右不重叠替换"，合并规则按原顺序写入。
特殊 token 用不可能由字节拼出的名字（含 U+2400 字符），因此文本里出现字面量 "[PAD]" 也不会被当成特殊 token。
"""

import json
from pathlib import Path
import random

SPECIAL_NAMES = ['␀PAD', '␀UNK', '␀BOS', '␀EOS', '␀MASK']
# Python str.isspace() 的字符集合（比正则 \s 多 \x1c-\x1f）
SPACE = r'\t-\r\x{1c}-\x{20}\x{85}\x{a0}\x{1680}\x{2000}-\x{200a}\x{2028}\x{2029}\x{202f}\x{205f}\x{3000}'
PATTERN = rf'[{SPACE}]+|[\p{{L}}\p{{M}}]+|\p{{N}}+|[^{SPACE}\p{{L}}\p{{M}}\p{{N}}]+'


def bytes_to_unicode():
    bs = list(range(ord('!'), ord('~') + 1)) + list(range(ord('¡'), ord('¬') + 1)) + list(range(ord('®'), ord('ÿ') + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, map(chr, cs)))


def convert(bbpe_path, out_path):
    from tokenizers import Regex, Tokenizer, decoders, models, pre_tokenizers
    data = json.loads(Path(bbpe_path).read_text(encoding='utf-8'))
    if data.get('format') != 'scratch-bbpe' or data.get('byte_offset') != 5:
        raise ValueError('unsupported bbpe format')
    if data.get('lowercase'):
        raise ValueError('lowercase tokenizer not supported by the fast path')
    enc = bytes_to_unicode()
    strings = SPECIAL_NAMES + [enc[b] for b in range(256)]
    vocab = {s: i for i, s in enumerate(strings)}
    merges = []
    for a, b in data['merges']:
        s = strings[a] + strings[b]
        if s in vocab:
            raise ValueError(f'duplicate merged string {s!r}; fast path not equivalent')
        vocab[s] = len(strings)
        strings.append(s)
        merges.append((strings[a], strings[b]))
    model = models.BPE(vocab=vocab, merges=merges, fuse_unk=False, byte_fallback=False, ignore_merges=False)
    tok = Tokenizer(model)
    tok.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.Split(Regex(PATTERN), behavior='isolated'),
        pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
    ])
    tok.decoder = decoders.ByteLevel()
    tok.save(str(out_path))
    return len(strings)


def verify(bbpe_path, hf_path, texts, slow_tokenizer=None):
    """逐条比较两种实现的 token id；返回不一致的条数与示例。"""
    from tokenizers import Tokenizer
    if slow_tokenizer is None:
        from .data_agent import load_bbpe
        slow_tokenizer = load_bbpe(bbpe_path)
    fast = Tokenizer.from_file(str(hf_path))
    extra = ['[PAD] [CLS] 字面量特殊符号', 'emoji 🙂👍🏽 混合 é 组合字符', '\x1c\x1d\x1e 控制空白 　全角空格',
             'tabs\t\tand\r\nnewlines', '数字123４５６和Ⅷ罗马数字', 'def f(x):\n    return x**2  # 注释', '']
    texts = list(texts) + extra
    mismatches, examples = 0, []
    encodings = fast.encode_batch(texts, add_special_tokens=False)
    for text, enc in zip(texts, encodings):
        slow = slow_tokenizer.encode(text)
        if enc.ids != slow:
            mismatches += 1
            if len(examples) < 5:
                examples.append({'text': text[:200], 'fast': enc.ids[:20], 'slow': slow[:20]})
        elif slow_tokenizer.decode(enc.ids) != text:
            mismatches += 1
            examples.append({'text': text[:200], 'error': 'decode roundtrip mismatch'})
    return {'checked': len(texts), 'mismatches': mismatches, 'examples': examples}


def sample_texts(jsonl_path, n=20000, seed=0):
    rows = [json.loads(line)['text'] for line in Path(jsonl_path).read_text(encoding='utf-8').splitlines() if line]
    rng = random.Random(seed)
    rng.shuffle(rows)
    return rows[:n]
