"""复用 Word2Vec 原始评估报告中的查询，离线比较本地 BGE 基座。"""

import argparse
import json
import math
from pathlib import Path
import tempfile
import time
from importlib.metadata import version

import numpy as np

if __package__ in (None, ""):
    from corpus import fingerprint
    from evaluation import load_eval_data, read_rows, ranked_batches, retrieval_metrics
else:
    from .corpus import fingerprint
    from .evaluation import load_eval_data, read_rows, ranked_batches, retrieval_metrics


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_baseline(report_dir, corpus_path, queries_path):
    """按报告 ID 和顺序锁定查询；同时验证数据、标签及原始指标。"""
    report = Path(report_dir)
    paths = [report / name for name in ("summary.json", "audit.json", "results.jsonl")]
    paths += [Path(corpus_path), Path(queries_path)]
    sources = [fingerprint(path) for path in paths]
    summary, audit = read_json(paths[0]), read_json(paths[1])
    if summary.get("method") != "word2vec_mean_l2_cosine":
        raise ValueError("需要 Word2Vec 平均词向量原始评估报告")
    recorded = audit.get("data_files", [])
    if len(recorded) != 2 or [s.get("sha256") for s in recorded] != [s["sha256"] for s in sources[-2:]]:
        raise ValueError("候选库/查询文件哈希与 Word2Vec 报告不一致；仅支持 --corpus/--queries 的原始报告")
    queries, documents, all_qrels = load_eval_data(corpus=corpus_path, queries=queries_path)
    if summary.get("total_queries") != len(queries) or summary.get("corpus_size") != len(documents):
        raise ValueError("报告的查询总数或候选数不一致")
    query_ids = {row["id"]: i for i, row in enumerate(queries)}
    doc_ids = {row["id"]: i for i, row in enumerate(documents)}
    selected, qrels, baseline, seen = [], [], [], set()
    totals = {key: 0.0 for key in retrieval_metrics([], {0})}
    for _, row in read_rows(paths[2]):
        query_id = row.get("id")
        if query_id not in query_ids or query_id in seen:
            raise ValueError(f"原始报告中的查询 ID 缺失、重复或不存在：{query_id}")
        seen.add(query_id)
        index = query_ids[query_id]
        query, relevant = queries[index], all_qrels[index]
        positives = row.get("relevant_documents", [])
        if (row.get("query") != query["text"] or
                {d.get("id") for d in positives} != {documents[i]["id"] for i in relevant}):
            raise ValueError(f"查询 {query_id} 的文本或正例标签与原始报告不一致")
        if row.get("excluded_document_ids"):
            raise ValueError("只能复用未排除候选的原始 results.jsonl")
        ranked, scores = [], []
        hits = row.get("results", [])
        if not isinstance(hits, list) or len(hits) > 10:
            raise ValueError(f"查询 {query_id} 的原始排名必须是 top-10 列表")
        for rank, hit in enumerate(hits, 1):
            doc_index = doc_ids.get(hit.get("id"))
            score = hit.get("score")
            if (doc_index is None or doc_index in ranked or hit.get("rank") != rank or
                    hit.get("text") != documents[doc_index]["text"] or
                    hit.get("is_relevant") != (doc_index in relevant) or
                    type(score) not in (float, int) or not math.isfinite(score)):
                raise ValueError(f"查询 {query_id} 的原始排名内容无效")
            ranked.append(doc_index)
            scores.append(score)
        if any((-scores[i], ranked[i]) > (-scores[i + 1], ranked[i + 1]) for i in range(len(ranked) - 1)):
            raise ValueError(f"查询 {query_id} 的原始排名顺序无效")
        metrics = retrieval_metrics(ranked, relevant)
        if any(not math.isclose(row.get("metrics", {}).get(key, -1), value, abs_tol=1e-10)
               for key, value in metrics.items()):
            raise ValueError(f"查询 {query_id} 的原始指标无法由排名复现")
        for key, value in metrics.items():
            totals[key] += value
        selected.append(query)
        qrels.append(relevant)
        baseline.append(row)
    if not selected or len(selected) != summary.get("evaluated_queries"):
        raise ValueError("原始报告的已评估查询数与 results.jsonl 不一致")
    if any(not math.isclose(summary.get("metrics", {}).get(key, -1), value / len(selected), abs_tol=1e-10)
           for key, value in totals.items()):
        raise ValueError("原始汇总指标无法由逐查询排名复现")
    return selected, documents, qrels, baseline, summary, sources


class BGEEncoder:
    """本地 BGE 的 CLS + L2 编码，查询指令与长度设置显式记录。"""

    def __init__(self, model_path, device, query_instruction, query_max_len, passage_max_len):
        import torch
        from transformers import AutoModel, AutoTokenizer

        if device == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA 不可用；请使用已有 PyTorch GPU 环境，或显式指定 --device cpu")
        self.torch = torch
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True, trust_remote_code=False)
        self.model = AutoModel.from_pretrained(str(model_path), local_files_only=True, trust_remote_code=False).to(device).eval()
        maximum = getattr(self.model.config, "max_position_embeddings", None)
        if maximum is not None and max(query_max_len, passage_max_len) > maximum:
            raise ValueError(f"编码长度超过模型位置上限 {maximum}")
        self.settings = {"pooling": "cls", "normalize": True, "query_instruction": query_instruction,
                         "query_max_len": query_max_len, "passage_max_len": passage_max_len,
                         "dtype": "float32", "settings_source": "explicit comparison CLI; saved embedding_config.json is not applied"}
        self.model.float()

    def encode(self, records, batch_size, is_query=False):
        length = self.settings["query_max_len" if is_query else "passage_max_len"]
        prefix = self.settings["query_instruction"] if is_query else ""
        vectors = np.empty((len(records), self.model.config.hidden_size), dtype=np.float32)
        with self.torch.inference_mode():
            for start in range(0, len(records), batch_size):
                texts = [prefix + row["text"] for row in records[start:start + batch_size]]
                inputs = self.tokenizer(texts, padding=True, truncation=True,
                                        max_length=length, return_tensors="pt").to(self.device)
                embeddings = self.model(**inputs).last_hidden_state[:, 0]
                embeddings = self.torch.nn.functional.normalize(embeddings, p=2, dim=1)
                vectors[start:start + len(texts)] = embeddings.cpu().numpy()
                done = start + len(texts)
                if start == 0 or done // 2000 != start // 2000 or done == len(records):
                    print(f"BGE 编码{'查询' if is_query else '候选'}：{done}/{len(records)}", flush=True)
        if not np.isfinite(vectors).all():
            raise ValueError("BGE 编码出现非有限值")
        return vectors


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def compare(baseline_report, model_path, corpus, queries, output_dir, device="cuda", batch_size=32,
            query_instruction="为这个句子生成表示以用于检索相关文章：", query_max_len=64,
            passage_max_len=256, query_batch_size=64, corpus_chunk_size=20000):
    for name, value in (("batch_size", batch_size), ("query_max_len", query_max_len),
                        ("passage_max_len", passage_max_len), ("query_batch_size", query_batch_size),
                        ("corpus_chunk_size", corpus_chunk_size)):
        if type(value) is not int or value < 1:
            raise ValueError(f"{name} 必须为正整数")
    if device not in ("cuda", "cpu"):
        raise ValueError("device 必须为 cuda 或 cpu")
    output = Path(output_dir).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"结果目录已存在：{output}；请指定新的 --output-dir")
    model_path = Path(model_path).expanduser().resolve()
    if not (model_path / "config.json").is_file():
        raise ValueError(f"请指定包含 config.json 的本地 BGE 目录：{model_path}")
    selected, documents, qrels, baseline, original, sources = load_baseline(
        Path(baseline_report).expanduser(), Path(corpus).expanduser(), Path(queries).expanduser())
    model_files = [fingerprint(path) for path in sorted(model_path.iterdir())
                   if path.is_file() and path.suffix in (".json", ".txt", ".bin", ".safetensors", ".model")]
    print(f"已校验 Word2Vec 原始报告：复用 {len(selected)} 个查询 ID，完整候选库 {len(documents)} 篇", flush=True)
    started = time.monotonic()
    encoder = BGEEncoder(model_path, device, query_instruction, query_max_len, passage_max_len)
    query_vectors = encoder.encode(selected, batch_size, is_query=True)
    doc_vectors = encoder.encode(documents, batch_size)
    settings = encoder.settings
    del encoder
    totals = {key: 0.0 for key in original["metrics"]}
    paired = {f"{outcome}_top{k}": 0 for k in (1, 10)
              for outcome in ("both_hit", "word2vec_only", "bge_only", "neither_hit")}
    review_counts = {"bge_only_top10": 0, "word2vec_only_top10": 0}
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".bge-comparison-", dir=output.parent) as temporary:
        staging = Path(temporary) / "report"
        staging.mkdir()
        with (staging / "selected_queries.jsonl").open("w", encoding="utf-8") as stream:
            for row in selected:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        with (staging / "bge_results.jsonl").open("w", encoding="utf-8") as results, \
                (staging / "review.jsonl").open("w", encoding="utf-8") as review:
            for i, ranked, scores in ranked_batches(query_vectors, doc_vectors, query_batch_size=query_batch_size,
                                                     corpus_chunk_size=corpus_chunk_size):
                metrics = retrieval_metrics(ranked, qrels[i])
                for key, value in metrics.items():
                    totals[key] += value
                groups = []
                for k in (1, 10):
                    w2v, bge = bool(baseline[i]["metrics"][f"Hit@{k}"]), bool(metrics[f"Hit@{k}"])
                    outcome = "both_hit" if w2v and bge else "word2vec_only" if w2v else "bge_only" if bge else "neither_hit"
                    group = f"{outcome}_top{k}"
                    paired[group] += 1
                    if group in review_counts and review_counts[group] < 10:
                        groups.append(group)
                row = {"id": selected[i]["id"], "query": selected[i]["text"], "metrics": metrics,
                       "word2vec_metrics": baseline[i]["metrics"],
                       "relevant_documents": [documents[j] for j in sorted(qrels[i])],
                       "results": [{"rank": rank, **documents[j], "score": float(score), "is_relevant": j in qrels[i]}
                                   for rank, (j, score) in enumerate(zip(ranked, scores), 1)]}
                results.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                for group in groups:
                    review.write(json.dumps({"group": group, **row, "word2vec_results": baseline[i]["results"]},
                                            ensure_ascii=False, allow_nan=False) + "\n")
                    review_counts[group] += 1
                if (i + 1) % 200 == 0 or i + 1 == len(selected):
                    print(f"BGE 检索评估：{i + 1}/{len(selected)}", flush=True)
        if any(fingerprint(source["path"]) != source for source in [*sources, *model_files]):
            raise ValueError("评估期间模型、数据或基线报告发生变化，请固定文件后重试")
        metrics = {key: value / len(selected) for key, value in totals.items()}
        summary = {"query_selection": "exact_ids_and_order_from_word2vec_raw_results",
                   "total_queries": original["total_queries"], "evaluated_queries": len(selected),
                   "corpus_size": len(documents), "word2vec": original["metrics"], "bge": metrics,
                   "delta_bge_minus_word2vec": {key: value - original["metrics"][key] for key, value in metrics.items()},
                   "paired_outcomes": paired, "review_cases": review_counts,
                   "bge_zero_vector_queries": int(np.sum(~np.any(query_vectors, axis=1))),
                   "bge_zero_vector_documents": int(np.sum(~np.any(doc_vectors, axis=1))),
                   "bge_elapsed_seconds": round(time.monotonic() - started, 3)}
        audit = {"baseline_and_data_files": sources, "bge_model_files": model_files,
                 "selected_queries": fingerprint(staging / "selected_queries.jsonl"),
                 "bge_settings": settings, "device": device, "batch_size": batch_size,
                 "query_batch_size": query_batch_size, "corpus_chunk_size": corpus_chunk_size,
                 "baseline_metrics_recomputed": True,
                 "ranking": "shared Word2Vec ranked_batches and retrieval_metrics; full corpus; no self exclusions; cosine descending, ties by corpus order",
                 "zero_vector_policy": "zero queries count as misses; zero documents cannot rank; positive denominators unchanged",
                 "comparison_scope": "each method uses its own encoding; Word2Vec uses full text, BGE truncates at configured token limits",
                 "versions": {name: version(name) for name in ("torch", "transformers", "numpy")}}
        audit["selected_queries"]["path"] = str(output / "selected_queries.jsonl")
        write_json(staging / "comparison.json", summary)
        write_json(staging / "audit.json", audit)
        lines = ["# Word2Vec / BGE 同样本对照", "",
                 f"复用 Word2Vec 原始报告的 {len(selected)} 个查询及其顺序，完整候选库 {len(documents)} 篇。",
                 "数据文件 SHA-256、查询 ID、正例标签与原始指标均已核对；不重新抽样，不排除同文候选。", "",
                 "| 指标 | Word2Vec | BGE | BGE 减 Word2Vec |", "| --- | ---: | ---: | ---: |"]
        lines += [f"| {key} | {original['metrics'][key]:.6f} | {value:.6f} | {value - original['metrics'][key]:+.6f} |"
                  for key, value in metrics.items()]
        lines += ["", "差值为指标原始单位；Hit/Recall 的差值乘以 100 才是百分点。",
                  "Hit 表示至少一个标注正例命中；Recall 是每个查询召回正例比例的平均值。", "",
                  f"BGE：CLS + L2，查询最大 {query_max_len} tokens，候选最大 {passage_max_len} tokens（含特殊 token/查询指令）。",
                  f"查询指令：{query_instruction!r}。编码设置由本次参数显式指定，不读取 embedding_config.json。",
                  "Word2Vec 使用全文平均；本次比较完整检索方案，编码和截断规则不同。",
                  "公开集结果不等于业务效果，也不保证 BGE 的上游训练未见过该数据。", "",
                  "paired_outcomes 记录双方共同命中、各自独有命中及共同未命中的数量。",
                  "review.jsonl 为双方各自独有 top-10 命中的前 10 个案例，含两边排名；不是随机人工准确率样本。",
                  "BGE 用 GPU 编码、CPU 分块检索；耗时包含不同阶段，不能据此推断在线检索延迟。"]
        (staging / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        if output.exists():
            raise FileExistsError(f"结果目录已存在：{output}")
        staging.rename(output)
    print(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False), flush=True)
    print(f"对照结果已保存：{output}", flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-report", required=True, help="已有 Word2Vec 原始评估目录")
    parser.add_argument("--model", required=True, dest="model_path", help="本地 BGE 基座目录")
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--queries", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--query-instruction", default="为这个句子生成表示以用于检索相关文章：")
    parser.add_argument("--query-max-len", type=int, default=64)
    parser.add_argument("--passage-max-len", type=int, default=256)
    parser.add_argument("--query-batch-size", type=int, default=64)
    parser.add_argument("--corpus-chunk-size", type=int, default=20000)
    compare(**vars(parser.parse_args()))


if __name__ == "__main__":
    try:
        main()
    except ImportError as error:
        raise SystemExit(f"依赖导入失败：{error}；请用已有 BGE 环境的 Python（torch、transformers、numpy）运行") from None
    except (OSError, ValueError, RuntimeError) as error:
        raise SystemExit(str(error)) from None
    except KeyboardInterrupt:
        raise SystemExit(130) from None
