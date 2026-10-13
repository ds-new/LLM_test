"""按行读取本地语料，避免一次把全部训练文本载入 Python 内存。"""

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
