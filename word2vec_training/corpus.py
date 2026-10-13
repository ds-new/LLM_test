"""读取本地 TXT / JSONL 并生成分词文件，无需工程其他目录。"""

import hashlib
import json
from pathlib import Path


def fingerprint(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return {"path": str(Path(path).resolve()), "sha256": digest.hexdigest()}


def iter_texts(paths, fields=("text",), max_texts=0):
    if not paths or not fields or any(not isinstance(f, str) or not f for f in fields):
        raise ValueError("corpus_files 和 fields 不能为空")
    if type(max_texts) is not int or max_texts < 0:
        raise ValueError("max_texts 必须为非负整数，0 表示不限")
    count = 0
    for value in paths:
        path = Path(value)
        if path.suffix.lower() not in (".txt", ".jsonl"):
            raise ValueError(f"{path}: 仅支持 .txt 或 .jsonl")
        with path.open(encoding="utf-8-sig") as stream:
            for number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                try:
                    if path.suffix.lower() == ".txt":
                        texts = [line.rstrip("\r\n")]
                    else:
                        row = json.loads(line)
                        if not isinstance(row, dict):
                            raise ValueError("每行必须为 JSON 对象")
                        texts = []
                        for field in fields:
                            if field not in row:
                                continue
                            values = row[field] if isinstance(row[field], list) else [row[field]]
                            if any(not isinstance(text, str) for text in values):
                                raise ValueError(f"字段 {field} 必须为字符串或字符串列表")
                            texts.extend(text for text in values if text.strip())
                        if not texts:
                            raise ValueError(f"指定字段 {list(fields)} 中没有有效文本")
                except ValueError as error:
                    raise ValueError(f"{path}:{number}: {error}") from error
                for text in texts:
                    yield text
                    count += 1
                    if max_texts and count >= max_texts:
                        return


class WordTokenizer:
    """训练和评估共用完全一致的分词与大小写规则。"""

    def __init__(self, tokenizer="jieba", lowercase=False):
        if tokenizer not in ("jieba", "whitespace"):
            raise ValueError("tokenizer 必须为 jieba 或 whitespace")
        if type(lowercase) is not bool:
            raise ValueError("lowercase 必须为布尔值")
        self.lowercase = lowercase
        self.segmenter = None
        if tokenizer == "jieba":
            import jieba

            self.segmenter = jieba.Tokenizer()

    def __call__(self, text):
        if self.lowercase:
            text = text.lower()
        words = self.segmenter.cut(text, HMM=False) if self.segmenter else text.split()
        return [word for word in words if word.strip() and
                (self.segmenter is None or any(char.isalnum() for char in word))]


def prepare_corpus(paths, destination, fields=("text",), tokenizer="jieba",
                   lowercase=False, max_texts=0):
    tokenize = WordTokenizer(tokenizer, lowercase)
    counts = {"texts": 0, "sentences": 0, "tokens": 0}
    with destination.open("w", encoding="utf-8") as stream:
        for text in iter_texts(paths, fields, max_texts):
            counts["texts"] += 1
            # JSONL 字段内的换行也作为上下文边界。
            for line in text.splitlines():
                tokens = tokenize(line)
                # Gensim 的单句上限为 10000 tokens；主动切块，避免长行尾部被丢弃。
                for start in range(0, len(tokens), 10000):
                    chunk = tokens[start:start + 10000]
                    stream.write(" ".join(chunk) + "\n")
                    counts["sentences"] += 1
                    counts["tokens"] += len(chunk)
    if not counts["tokens"]:
        raise ValueError("语料为空，或分词后没有有效词语")
    return counts
