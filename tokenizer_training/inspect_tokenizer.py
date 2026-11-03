"""离线加载训练产物，展示分词结果并可统计独立语料的未知词元比例。"""

import argparse
import json
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tokenizer_training.corpus import iter_texts


def corpus_metrics(tokenizer, texts):
    count = tokens = unknown = characters = utf8_bytes = 0
    for text in texts:
        ids = tokenizer(text, add_special_tokens=False, verbose=False)["input_ids"]
        count += 1
        characters += len(text)
        utf8_bytes += len(text.encode("utf-8"))
        tokens += len(ids)
        unknown += sum(token == tokenizer.unk_token_id for token in ids)
    if not count:
        raise ValueError("检查语料为空")
    return {"texts": count, "characters": characters, "tokens_without_special": tokens,
            "unknown_tokens": unknown, "unknown_token_rate": unknown / tokens if tokens else None,
            "tokens_per_text": tokens / count,
            "tokens_per_character": tokens / characters if characters else None,
            "utf8_bytes_per_token": utf8_bytes / tokens if tokens else None}


def main():
    from tokenizer_training.bbpe import BBPETokenizer

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer", required=True, help="本地 tokenizer 目录")
    parser.add_argument("--text", nargs="+", default=["你好世界🙂", "BBPE tokenizer training"])
    parser.add_argument("--corpus", nargs="+", help="可选：统计本地 TXT/JSONL 语料")
    parser.add_argument("--fields", nargs="+", default=["text"])
    parser.add_argument("--max-texts", type=int, default=0)
    args = parser.parse_args()
    root = Path(args.tokenizer).expanduser().resolve()
    if not root.is_dir():
        parser.error(f"本地 tokenizer 目录不存在：{root}")
    if args.max_texts < 0:
        parser.error("max-texts 必须为非负整数")
    tokenizer = BBPETokenizer.from_pretrained(root)
    samples = []
    for text in args.text:
        encoded = tokenizer(text)
        samples.append({"text": text, "tokens": tokenizer.convert_ids_to_tokens(encoded["input_ids"]),
                        **encoded, "decoded": tokenizer.decode(encoded["input_ids"], skip_special_tokens=True)})
    report = {"tokenizer": str(root), "vocab_size": len(tokenizer), "is_fast": tokenizer.is_fast,
              "model_max_length": tokenizer.model_max_length, "samples": samples}
    if args.corpus:
        report["corpus_metrics"] = corpus_metrics(tokenizer, iter_texts(args.corpus, args.fields, args.max_texts))
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError) as error:
        raise SystemExit(str(error)) from None
    except KeyboardInterrupt:
        raise SystemExit(130) from None
