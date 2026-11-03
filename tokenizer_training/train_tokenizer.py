"""纯 Python 从零训练 BBPE：不调用第三方分词或训练库。"""

import argparse
import json
from pathlib import Path
import sys
import tempfile
import time
import shutil

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tokenizer_training.corpus import fingerprint, iter_texts
from tokenizer_training.bbpe import BBPETokenizer, SPECIAL_TOKENS
from tokenizer_training.runtime_info import peak_rss_mib, snapshot


def train_tokenizer(corpus_files, output_dir, fields=("text",), vocab_size=32000, min_frequency=2,
                    lowercase=False, model_max_length=512, max_texts=0, algorithm="bbpe", backend="disk", work_dir=None):
    for key, value, minimum in (("vocab_size", vocab_size, 256 + len(SPECIAL_TOKENS)),
                                ("min_frequency", min_frequency, 1), ("model_max_length", model_max_length, 3),
                                ("max_texts", max_texts, 0)):
        if type(value) is not int or value < minimum:
            raise ValueError(f"{key} 必须为不小于 {minimum} 的整数")
    if type(lowercase) is not bool:
        raise ValueError("lowercase 必须为布尔值")
    if algorithm != "bbpe":
        raise ValueError("algorithm 必须为 bbpe (Byte-level BPE)")
    if backend not in ("memory", "disk"):
        raise ValueError("backend 必须为 memory 或 disk")
    if not isinstance(corpus_files, (list, tuple)) or not corpus_files:
        raise ValueError("corpus_files 必须为非空文件列表")
    if not isinstance(fields, (list, tuple)) or not fields or any(not isinstance(f, str) or not f for f in fields):
        raise ValueError("fields 必须为非空字段名列表")
    output = Path(output_dir).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"输出目录已存在：{output}；请指定新目录，不覆盖现有 tokenizer")
    work = Path(work_dir).expanduser().resolve() if work_dir else output.parent
    if backend == "disk":
        work.mkdir(parents=True, exist_ok=True)
    paths = [Path(path).expanduser().resolve() for path in corpus_files]
    if len(set(paths)) != len(paths):
        raise ValueError("corpus_files 含重复文件")
    sources = [fingerprint(path) for path in paths]
    report = {"algorithm": "BBPE", "implementation": "python-standard-library",
              "pretokenizer": "unicode-category-v1", "byte_alphabet_size": 256,
              "texts": 0, "characters": 0, "sources": sources,
              "requested_vocab_size": vocab_size, "min_frequency": min_frequency,
              "lowercase": lowercase, "model_max_length": model_max_length, "max_texts": max_texts,
              "fields": list(fields), "python_version": sys.version.split()[0], "backend": backend,
              "compatible_with_existing_bge_weights": False}
    samples = []
    started = time.monotonic()

    def status(message):
        print(f"{message}；耗时 {time.monotonic() - started:.1f}s；峰值 RSS {peak_rss_mib()} MiB", flush=True)

    def training_texts():
        for text in iter_texts(paths, fields, max_texts):
            report["texts"] += 1
            report["characters"] += len(text)
            if len(samples) < 3:
                samples.append(text[:200])
            if report["texts"] % 100000 == 0:
                status(f"已读取 {report['texts']} 条文本")
            yield text

    def progress(step, pair, frequency):
        if step == 0 or step % 100 == 0 or step + 261 == vocab_size:
            status(f"合并 {step}/{vocab_size - 261}；当前词表 {261 + step}；候选对频次 {frequency}")

    print(f"开始纯 Python BBPE 训练：文件 {len(paths)} 个，目标词表 {vocab_size}，backend={backend}，CPU 运行", flush=True)
    report["runtime_start"] = snapshot()
    print(json.dumps(report["runtime_start"], ensure_ascii=False), flush=True)
    if backend == "disk":
        status(f"工作目录 {work}；可用磁盘 {shutil.disk_usage(work).free / 1024**3:.2f} GiB")
    tokenizer = BBPETokenizer.train(training_texts(), vocab_size=vocab_size, min_frequency=min_frequency,
                                    lowercase=lowercase, model_max_length=model_max_length, progress=progress,
                                    backend=backend, work_dir=work, status=status)
    if [fingerprint(path) for path in paths] != sources:
        raise ValueError("训练期间语料发生变化，请固定语料后重试")
    report["actual_vocab_size"] = len(tokenizer)
    report["special_token_ids"] = dict(zip(SPECIAL_TOKENS, range(len(SPECIAL_TOKENS))))
    report.update(tokenizer.training_stats)
    report["elapsed_seconds"] = round(time.monotonic() - started, 2)
    report["runtime_end"] = snapshot()
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".tokenizer-", dir=output.parent) as temporary:
        staging = Path(temporary) / "tokenizer"
        tokenizer.save_pretrained(staging)
        loaded = BBPETokenizer.from_pretrained(staging)
        if loaded(samples) != tokenizer(samples):
            raise ValueError("保存后重新加载的 tokenizer 编码结果不一致")
        (staging / "training_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        staging.rename(output)
    print(f"Tokenizer 已保存：{output}；文本 {report['texts']} 条，实际词表 {len(tokenizer)}", flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(Path(__file__).with_name("config.json")))
    parser.add_argument("--corpus", nargs="+", help="覆盖语料路径；相对当前工作目录")
    parser.add_argument("--fields", nargs="+", help="JSONL 文本字段，例如 query pos neg")
    parser.add_argument("--output-dir")
    parser.add_argument("--backend", choices=("disk", "memory"), help="disk 使用磁盘保存统计，默认 disk")
    parser.add_argument("--work-dir", help="磁盘工作目录，默认输出目录的父目录；应选择容量充足的磁盘")
    for name in ("vocab-size", "min-frequency", "model-max-length", "max-texts"):
        parser.add_argument(f"--{name}", type=int)
    parser.add_argument("--lowercase", action=argparse.BooleanOptionalAction, default=None)
    args = parser.parse_args()
    config_path = Path(args.config).expanduser().resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    allowed = {"algorithm", "backend", "work_dir", "corpus_files", "output_dir", "fields", "vocab_size", "min_frequency", "lowercase", "model_max_length", "max_texts"}
    if not isinstance(config, dict) or set(config) - allowed:
        parser.error("配置必须为 JSON 对象，且不能包含未知参数")
    if "corpus_files" not in config or "output_dir" not in config:
        parser.error("配置缺少 corpus_files 或 output_dir")
    if not isinstance(config["corpus_files"], list) or any(not isinstance(p, str) for p in config["corpus_files"]):
        parser.error("corpus_files 必须为路径字符串列表")
    if not isinstance(config["output_dir"], str) or not config["output_dir"].strip():
        parser.error("output_dir 必须为非空路径字符串")
    config["corpus_files"] = args.corpus or [str(config_path.parent / Path(p).expanduser()) for p in config["corpus_files"]]
    config["output_dir"] = args.output_dir or str(config_path.parent / Path(config["output_dir"]).expanduser())
    if config.get("work_dir") is not None:
        if not isinstance(config["work_dir"], str) or not config["work_dir"].strip():
            parser.error("work_dir 必须为非空路径字符串或 null")
        config["work_dir"] = str(config_path.parent / Path(config["work_dir"]).expanduser())
    for key in ("fields", "vocab_size", "min_frequency", "model_max_length", "max_texts", "lowercase", "backend", "work_dir"):
        value = getattr(args, key)
        if value is not None:
            config[key] = value
    train_tokenizer(**config)


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError) as error:
        raise SystemExit(str(error)) from None
    except KeyboardInterrupt:
        raise SystemExit(130) from None
