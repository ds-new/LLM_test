"""评估 Word2Vec 平均词向量的检索效果与词表覆盖率，CPU 离线运行。"""

import argparse
from importlib.metadata import version
import json
from pathlib import Path
import tempfile
import time

if __package__ in (None, ""):
    from corpus import WordTokenizer, fingerprint
    from evaluation import coverage_summary, encode_records, load_eval_data, ranked_batches, retrieval_metrics
else:
    from .corpus import WordTokenizer, fingerprint
    from .evaluation import coverage_summary, encode_records, load_eval_data, ranked_batches, retrieval_metrics


def evaluate(vectors_path, output_dir, eval_data=None, corpus=None, queries=None,
             tokenizer=None, lowercase=None, limit_queries=0, seed=42,
             query_batch_size=64, corpus_chunk_size=20000, diagnose_self_matches=False):
    from gensim.models import KeyedVectors
    import numpy as np

    for name, value, minimum in (("limit_queries", limit_queries, 0), ("seed", seed, 0),
                                  ("query_batch_size", query_batch_size, 1),
                                  ("corpus_chunk_size", corpus_chunk_size, 1)):
        if type(value) is not int or value < minimum:
            raise ValueError(f"{name} 必须为不小于 {minimum} 的整数")
    output = Path(output_dir).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"结果目录已存在：{output}；请指定新的 --output-dir")
    vector_path = Path(vectors_path).expanduser().resolve()
    report_path = vector_path.with_name("training_report.json")
    training = json.loads(report_path.read_text(encoding="utf-8")) if report_path.is_file() else {}
    if not isinstance(training, dict) or not isinstance(training.get("parameters", {}), dict):
        raise ValueError("training_report.json 格式不正确")
    parameters = training.get("parameters", {})
    tokenizer = tokenizer if tokenizer is not None else parameters.get("tokenizer")
    lowercase = lowercase if lowercase is not None else parameters.get("lowercase")
    if tokenizer is None or lowercase is None:
        raise ValueError("无法从 training_report.json 确定预处理；请显式指定 --tokenizer 和 --lowercase/--no-lowercase")
    tokenize = WordTokenizer(tokenizer, lowercase)
    input_paths = [Path(p).expanduser().resolve() for p in (eval_data, corpus, queries) if p is not None]
    sources = [fingerprint(path) for path in input_paths]
    model_paths = [vector_path, *sorted(vector_path.parent.glob(vector_path.name + ".*.npy"))]
    if report_path.is_file():
        model_paths.append(report_path)
    model_sources = [fingerprint(path) for path in model_paths]
    training_hashes = {source.get("sha256") for source in training.get("sources", [])}
    query_file = Path(eval_data if eval_data is not None else queries).expanduser().resolve() if (eval_data or queries) else None
    matches = [source["path"] for source in sources if source["sha256"] in training_hashes]
    if query_file is not None and str(query_file) in matches:
        raise ValueError("评估查询文件与训练输入完全一致；请使用留出的 eval.jsonl 或独立标注查询")
    query_rows, documents, qrels = load_eval_data(
        eval_data=Path(eval_data).expanduser() if eval_data is not None else None,
        corpus=Path(corpus).expanduser() if corpus is not None else None,
        queries=Path(queries).expanduser() if queries is not None else None)
    total_queries = len(query_rows)
    if limit_queries and limit_queries < total_queries:
        selected = sorted(np.random.default_rng(seed).choice(total_queries, limit_queries, replace=False).tolist())
        query_rows = [query_rows[i] for i in selected]
        qrels = [qrels[i] for i in selected]
    print(f"评估 {len(query_rows)}/{total_queries} 个查询，完整候选库 {len(documents)} 篇；CPU 平均词向量", flush=True)
    started = time.monotonic()
    vectors = KeyedVectors.load(str(vector_path), mmap="r")
    query_vectors, query_coverage = encode_records(query_rows, vectors, tokenize, label="查询")
    doc_vectors, doc_coverage = encode_records(documents, vectors, tokenize, label="候选")
    vector_size, vocab_size = vectors.vector_size, len(vectors)
    del vectors
    text_ids = {}
    for i, document in enumerate(documents):
        text_ids.setdefault(document["text"], set()).add(i)
    same_text_unlabelled = [text_ids.get(row["text"], set()) - relevant for row, relevant in zip(query_rows, qrels)]
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".word2vec-eval-", dir=output.parent) as temporary:
        staging = Path(temporary) / "report"
        staging.mkdir()

        def run_ranking(prefix="", exclusions=None):
            totals = {key: 0.0 for key in retrieval_metrics([], {0})}
            review_count = top1_self_count = top10_self_count = 0
            with (staging / f"{prefix}results.jsonl").open("w", encoding="utf-8") as results, \
                    (staging / f"{prefix}review.jsonl").open("w", encoding="utf-8") as review:
                for i, ranked, scores in ranked_batches(
                        query_vectors, doc_vectors, query_batch_size=query_batch_size,
                        corpus_chunk_size=corpus_chunk_size, excluded_indices=exclusions):
                    metrics = retrieval_metrics(ranked, qrels[i])
                    for key, value in metrics.items():
                        totals[key] += value
                    self_ranks = [rank for rank, index in enumerate(ranked, 1) if index in same_text_unlabelled[i]]
                    top1_self_count += int(1 in self_ranks)
                    top10_self_count += int(bool(self_ranks))
                    record = {"id": query_rows[i]["id"], "query": query_rows[i]["text"], "metrics": metrics,
                              "coverage": query_coverage[i],
                              "relevant_documents": [documents[index] for index in sorted(qrels[i])],
                              "unlabelled_self_match_ids": [documents[index]["id"] for index in sorted(same_text_unlabelled[i])],
                              "unlabelled_self_match_ranks": self_ranks,
                              "excluded_document_ids": [documents[index]["id"] for index in sorted(exclusions[i])] if exclusions is not None else [],
                              "results": [{"rank": rank, **documents[index], "score": float(score),
                                           "is_relevant": index in qrels[i]}
                                          for rank, (index, score) in enumerate(zip(ranked, scores), 1)]}
                    line = json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n"
                    results.write(line)
                    if metrics["Hit@1"] == 0 and review_count < 30:
                        review.write(line)
                        review_count += 1
                    if (i + 1) % 200 == 0 or i + 1 == len(query_rows):
                        print(f"{'对照' if exclusions is not None else '原始'}检索评估：{i + 1}/{len(query_rows)}", flush=True)
            return {"metrics": {key: value / len(query_rows) for key, value in totals.items()},
                    "review_cases": review_count, "top1_self_count": top1_self_count,
                    "top10_self_count": top10_self_count}

        raw = run_ranking()
        diagnostic = run_ranking("diagnostic_", same_text_unlabelled) if diagnose_self_matches else None
        if [fingerprint(path) for path in input_paths] != sources or [fingerprint(path) for path in model_paths] != model_sources:
            raise ValueError("评估期间模型或输入文件发生变化，请固定文件后重试")
        summary = {
            "method": "word2vec_mean_l2_cosine", "vocab_size": vocab_size, "vector_size": vector_size,
            "total_queries": total_queries, "evaluated_queries": len(query_rows), "corpus_size": len(documents),
            "query_sampling": "seeded_random" if len(query_rows) < total_queries else "all",
            "metrics": raw["metrics"],
            "query_coverage": coverage_summary(query_coverage), "corpus_coverage": coverage_summary(doc_coverage),
            "queries_with_unlabelled_self_match": sum(bool(indices) for indices in same_text_unlabelled),
            "raw_top1_unlabelled_self_match_queries": raw["top1_self_count"],
            "raw_top10_unlabelled_self_match_queries": raw["top10_self_count"],
            "review_cases": raw["review_cases"], "elapsed_seconds": round(time.monotonic() - started, 3),
        }
        if diagnostic is not None:
            summary["self_match_diagnostic"] = {
                "candidate_policy": "exclude_unlabelled_exact_text_per_query",
                "queries_with_exclusions": sum(bool(indices) for indices in same_text_unlabelled),
                "excluded_query_document_pairs": sum(len(indices) for indices in same_text_unlabelled),
                "metrics": diagnostic["metrics"],
                "metric_deltas_vs_raw": {key: value - raw["metrics"][key]
                                         for key, value in diagnostic["metrics"].items()},
                "review_cases": diagnostic["review_cases"],
            }
        audit = {"model_files": model_sources, "data_files": sources, "training_report": training,
                 "preprocessing": {"tokenizer": tokenizer, "lowercase": lowercase, "hmm": False},
                 "pooling": "mean of raw in-vocabulary word vectors, then L2 normalize; repeated tokens retained",
                 "oov_policy": "skip OOV tokens; zero-vector queries count as misses; zero-vector documents cannot rank",
                 "ranking": "full candidate corpus; cosine descending, ties by corpus order; top 10; no min-count query filter",
                 "label_policy": "binary relevance; raw metrics keep all candidates; no automatic label expansion",
                 "self_match_diagnostic": {
                     "enabled": bool(diagnose_self_matches),
                     "rule": "per-query exclusion of exact text matches without a positive label; strip outer whitespace only",
                     "ranking": "rerank the full remaining corpus with cached vectors; labelled positives and qrel denominators unchanged",
                     "interpretation": "candidate-protocol diagnostic, not a model improvement or replacement for raw metrics"},
                 "seed": seed, "limit_queries": limit_queries, "query_batch_size": query_batch_size,
                 "corpus_chunk_size": corpus_chunk_size, "inputs_matching_training_files": matches,
                 "leakage_check": "exact input-file hashes only; text overlap between splits is not audited",
                 "versions": {name: version(name) for name in ("gensim", "jieba", "numpy")}}
        for name, data in (("summary.json", summary), ("audit.json", audit)):
            (staging / name).write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
        lines = ["# Word2Vec 检索评估", "", f"查询 {len(query_rows)}/{total_queries}，完整候选库 {len(documents)} 篇。",
                 "", "方法：沿用训练分词规则，平均词表内原始词向量，L2 归一化后按余弦相似度检索。", "",
                 "| 指标 | 数值 |", "| --- | ---: |"]
        lines += [f"| {key} | {value:.6f} |" for key, value in summary["metrics"].items()]
        lines += ["", f"查询词次覆盖率：{summary['query_coverage']['token_coverage']:.2%}；"
                  f"候选词次覆盖率：{summary['corpus_coverage']['token_coverage']:.2%}。",
                  f"零向量查询：{summary['query_coverage']['zero_vector_texts']}；"
                  f"零向量候选：{summary['corpus_coverage']['zero_vector_texts']}。", "",
                  "Hit@K：至少命中一个标注正例的查询比例；Recall@K：每个查询召回正例比例的平均值。",
                  "MRR@10：首个正例在前十名内的倒数排名均值；NDCG@10：二值相关性的归一化折损累计增益。",
                  "零向量查询计为未命中；零向量正例仍保留在相关性分母中。", "",
                  f"有未标正例的同文候选的查询：{summary['queries_with_unlabelled_self_match']}；原始指标保留这些候选。",
                  f"其中这类候选实际占据原始第一名的查询：{raw['top1_self_count']}；进入前十名的查询：{raw['top10_self_count']}。",
                  "results.jsonl 包含所有查询的 top-10 和标注正例；review.jsonl 为前 30 个 top-1 未命中案例。",
                  "复核样本用于排错，不能代表随机抽样的人工准确率。缺少正例标注不等于语义不相关。", "",
                  "抽样时仅抽查询，候选库保持完整。不同候选库、查询样本或标注口径的分数不能直接比较。",
                  "这评估的是平均词向量检索方案，不能代替词义质量、上下文理解或生成回答质量的评估。",
                  "文件哈希相异不保证训练/评估文本无重叠；本入口尚未审计跨集文本重复。"]
        if diagnostic is not None:
            lines += ["", "## 同文候选对照", "",
                      "仅对当前查询排除原文完全相同且未标为正例的候选，保留已有正例与分母；从剩余完整候选库重新取前十名。",
                      "模型、分词、查询样本均不变。分数变化来自候选口径变化，不能视为模型提升，也不替代上面的原始指标。", "",
                      "| 指标 | 原始 | 对照 | 差值（对照减原始） |", "| --- | ---: | ---: | ---: |"]
            lines += [f"| {key} | {raw['metrics'][key]:.6f} | {value:.6f} | {value - raw['metrics'][key]:+.6f} |"
                      for key, value in diagnostic["metrics"].items()]
            lines += ["", "diagnostic_results.jsonl 保存对照排名与逐查询排除的文档 ID；diagnostic_review.jsonl 保存对照后前 30 个 top-1 未命中案例。"]
        (staging / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        if output.exists():
            raise FileExistsError(f"结果目录已存在：{output}")
        staging.rename(output)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    print(f"评估结果已保存：{output}", flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vectors", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--eval-data", help="query/pos/neg 格式的留出 JSONL")
    parser.add_argument("--corpus", help="候选库 JSONL：id/text")
    parser.add_argument("--queries", help="标注查询 JSONL：id/text/relevant_ids")
    parser.add_argument("--tokenizer", choices=("jieba", "whitespace"), help="默认读取训练报告")
    parser.add_argument("--lowercase", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--limit-queries", type=int, default=0, help="随机抽样查询数，0 为全部；候选库不缩减")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--query-batch-size", type=int, default=64)
    parser.add_argument("--corpus-chunk-size", type=int, default=20000)
    parser.add_argument("--diagnose-self-matches", action="store_true",
                        help="额外计算逐查询排除未标正例同文候选的对照；保留原始指标")
    args = parser.parse_args()
    if not ((args.eval_data and not args.corpus and not args.queries) or
            (not args.eval_data and args.corpus and args.queries)):
        parser.error("请选择 --eval-data，或 --corpus 和 --queries 的组合")
    values = vars(args)
    values["vectors_path"] = values.pop("vectors")
    evaluate(**values)


if __name__ == "__main__":
    try:
        main()
    except ImportError as error:
        raise SystemExit(f"依赖导入失败：{error}；请安装 {Path(__file__).with_name('requirements.txt')}") from None
    except (OSError, ValueError) as error:
        raise SystemExit(str(error)) from None
    except KeyboardInterrupt:
        raise SystemExit(130) from None
