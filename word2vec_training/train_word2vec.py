"""从本地 TXT / JSONL 训练 Word2Vec（CPU），保存模型、词向量与训练报告。"""

import argparse
from importlib.metadata import version
import json
import logging
import math
import os
from pathlib import Path
import tempfile
import time

if __package__ in (None, ""):
    from corpus import fingerprint, prepare_corpus
else:
    from .corpus import fingerprint, prepare_corpus


def train_word2vec(corpus_files, output_dir, fields=("text",), tokenizer="jieba",
                   algorithm="skipgram", vector_size=100, window=5, min_count=5,
                   epochs=10, workers=1, negative=5, sample=0.001, seed=42,
                   lowercase=False, max_texts=0):
    parameters = dict(tokenizer=tokenizer, algorithm=algorithm, vector_size=vector_size,
                      window=window, min_count=min_count, epochs=epochs, workers=workers,
                      negative=negative, sample=sample, seed=seed, lowercase=lowercase,
                      max_texts=max_texts)
    for key in ("vector_size", "window", "min_count", "epochs", "workers", "negative", "seed", "max_texts"):
        minimum = 0 if key in ("seed", "max_texts") else 1
        if type(parameters[key]) is not int or parameters[key] < minimum:
            raise ValueError(f"{key} 必须为不小于 {minimum} 的整数")
    if seed >= 2**32:
        raise ValueError("seed 必须小于 2**32")
    if type(sample) not in (int, float) or not math.isfinite(sample) or not 0 <= sample < 1:
        raise ValueError("sample 必须在 [0, 1) 内，0 表示关闭高频词下采样")
    if algorithm not in ("skipgram", "cbow") or tokenizer not in ("jieba", "whitespace"):
        raise ValueError("algorithm 必须为 skipgram/cbow，tokenizer 必须为 jieba/whitespace")
    if type(lowercase) is not bool:
        raise ValueError("lowercase 必须为布尔值")
    if not isinstance(fields, (list, tuple)) or not fields or any(not isinstance(f, str) or not f for f in fields):
        raise ValueError("fields 必须为非空字段名列表")
    if not isinstance(corpus_files, (list, tuple)) or not corpus_files:
        raise ValueError("corpus_files 必须为非空文件列表")
    if any(not isinstance(p, (str, Path)) or not str(p).strip() for p in corpus_files):
        raise ValueError("corpus_files 必须包含有效路径")
    if not isinstance(output_dir, (str, Path)) or not str(output_dir).strip():
        raise ValueError("output_dir 必须为有效路径")
    output = Path(output_dir).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"输出目录已存在：{output}；请使用新的 --output-dir")
    paths = [Path(path).expanduser().resolve() for path in corpus_files]
    if len(set(paths)) != len(paths):
        raise ValueError("corpus_files 含重复文件")
    sources = [fingerprint(path) for path in paths]

    from gensim.models import KeyedVectors, Word2Vec
    from gensim.models.word2vec import LineSentence
    import numpy as np

    started = time.monotonic()
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".word2vec-", dir=output.parent) as temporary:
        temporary = Path(temporary)
        token_file = temporary / "corpus.txt"
        print(f"准备语料：{len(paths)} 个文件，分词方式 {tokenizer}", flush=True)
        counts = prepare_corpus(paths, token_file, fields, tokenizer, lowercase, max_texts)
        if [fingerprint(path) for path in paths] != sources:
            raise ValueError("准备期间语料发生变化，请固定语料后重试")
        # 先分词落盘，再反复读取；各 epoch 使用同一份语料且不重复分词。
        sentences = LineSentence(str(token_file))
        model = Word2Vec(vector_size=vector_size, window=window, min_count=min_count,
                         workers=workers, sg=int(algorithm == "skipgram"), hs=0,
                         negative=negative, sample=sample, seed=seed, epochs=epochs)
        model.build_vocab(sentences)
        if len(model.wv) < 2:
            raise ValueError("有效词表少于 2 个词，请增加语料或降低 min_count")
        if not any(sum(word in model.wv for word in sentence) >= 2 for sentence in sentences):
            raise ValueError("没有包含至少两个词表内词语的句子，无法学习上下文")
        print(f"开始 {algorithm} 训练：{counts['sentences']} 句，词表 {len(model.wv)}，"
              f"{vector_size} 维，{epochs} 轮，CPU workers={workers}", flush=True)
        effective_words, raw_words = model.train(sentences, total_examples=model.corpus_count, epochs=epochs)
        if effective_words == 0:
            raise ValueError("没有有效训练词，请增加语料或设置 sample=0")
        staging = temporary / "model"
        staging.mkdir()
        model.save(str(staging / "word2vec.model"))
        model.wv.save(str(staging / "vectors.kv"))
        model.wv.save_word2vec_format(str(staging / "vectors.txt"), binary=False)
        loaded = KeyedVectors.load(str(staging / "vectors.kv"), mmap="r")
        if loaded.index_to_key != model.wv.index_to_key or not np.array_equal(loaded.vectors[:3], model.wv.vectors[:3]):
            raise ValueError("保存后的词向量重载校验失败")
        del loaded
        report = {"parameters": parameters, "fields": list(fields), "sources": sources,
                  **counts, "vocab_size": len(model.wv), "effective_words_all_epochs": effective_words,
                  "raw_words_all_epochs": raw_words, "elapsed_seconds": round(time.monotonic() - started, 3),
                  "pythonhashseed": os.environ.get("PYTHONHASHSEED"),
                  "versions": {name: version(name) for name in ("gensim", "jieba", "numpy", "scipy")}}
        (staging / "training_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        if output.exists():
            raise FileExistsError(f"输出目录已存在：{output}")
        staging.rename(output)
    print(f"Word2Vec 已保存：{output}", flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(Path(__file__).with_name("config.json")))
    parser.add_argument("--corpus", nargs="+", help="TXT/JSONL 路径，相对当前工作目录")
    parser.add_argument("--output-dir")
    parser.add_argument("--fields", nargs="+", help="JSONL 文本字段，例如 query pos neg")
    parser.add_argument("--tokenizer", choices=("jieba", "whitespace"))
    parser.add_argument("--algorithm", choices=("skipgram", "cbow"))
    for name in ("vector-size", "window", "min-count", "epochs", "workers", "negative", "seed", "max-texts"):
        parser.add_argument(f"--{name}", type=int)
    parser.add_argument("--sample", type=float)
    parser.add_argument("--lowercase", action=argparse.BooleanOptionalAction, default=None)
    args = parser.parse_args()
    config_path = Path(args.config).expanduser().resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    overrides = vars(args).copy()
    overrides.pop("config")
    corpus_override = overrides.pop("corpus")
    if not isinstance(config, dict) or set(config) - (set(overrides) | {"corpus_files"}):
        parser.error("配置必须为 JSON 对象，且不能包含未知参数")
    corpus = corpus_override if corpus_override is not None else config.get("corpus_files")
    if not isinstance(corpus, list) or not corpus or any(not isinstance(p, str) or not p.strip() for p in corpus):
        parser.error("corpus_files 必须为非空路径字符串列表")
    config["corpus_files"] = corpus if corpus_override is not None else [
        str(config_path.parent / Path(p).expanduser()) for p in corpus]
    output = args.output_dir if args.output_dir is not None else config.get("output_dir")
    if not isinstance(output, str) or not output.strip():
        parser.error("output_dir 必须为非空路径字符串")
    config["output_dir"] = output if args.output_dir is not None else str(config_path.parent / Path(output).expanduser())
    config.update({key: value for key, value in overrides.items() if value is not None})
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    train_word2vec(**config)


if __name__ == "__main__":
    try:
        main()
    except ImportError as error:
        raise SystemExit(f"依赖导入失败：{error}；请使用当前解释器安装 {Path(__file__).with_name('requirements.txt')}") from None
    except (OSError, ValueError) as error:
        raise SystemExit(str(error)) from None
    except KeyboardInterrupt:
        raise SystemExit(130) from None
