"""独立的语料加载、平均词向量编码和分块检索评估，不依赖 PyTorch。"""

import json
from pathlib import Path

import numpy as np


def read_rows(path):
    with Path(path).open(encoding="utf-8-sig") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError("每行必须为 JSON 对象")
            except ValueError as error:
                raise ValueError(f"{path}:{number}: {error}") from error
            yield number, row


def nonempty_text(value):
    return isinstance(value, str) and bool(value.strip())


def load_eval_data(eval_data=None, corpus=None, queries=None):
    """返回查询/文档记录及文档下标正例集合，保留完整候选库。"""
    if eval_data is not None:
        if corpus is not None or queries is not None:
            raise ValueError("--eval-data 与 --corpus/--queries 不能混用")
        documents, doc_ids, query_rows, relevant = [], {}, {}, {}
        for number, row in read_rows(eval_data):
            if not nonempty_text(row.get("query")):
                raise ValueError(f"{eval_data}:{number}: query 必须为非空字符串")
            query = row["query"].strip()
            for field in ("pos", "neg"):
                values = row.get(field, [])
                if not isinstance(values, list) or any(not nonempty_text(s) for s in values):
                    raise ValueError(f"{eval_data}:{number}: {field} 必须为非空字符串组成的列表")
                row[field] = [s.strip() for s in values]
            if not row["pos"]:
                raise ValueError(f"{eval_data}:{number}: pos 不能为空")
            if set(row["neg"]) & (set(row["pos"]) | {query}):
                raise ValueError(f"{eval_data}:{number}: neg 包含 query 或已知正例")
            if query not in query_rows:
                query_rows[query] = {"id": f"q{len(query_rows)}", "text": query}
                relevant[query] = set()
            for field in ("pos", "neg"):
                for text in row[field]:
                    if text not in doc_ids:
                        doc_ids[text] = len(documents)
                        documents.append({"id": f"d{len(documents)}", "text": text})
                    if field == "pos":
                        relevant[query].add(doc_ids[text])
        if not query_rows:
            raise ValueError(f"{eval_data}: 评估数据为空")
        return list(query_rows.values()), documents, list(relevant.values())
    if corpus is None or queries is None:
        raise ValueError("请提供 --eval-data，或同时提供 --corpus 和 --queries")

    def read_records(path):
        rows, seen = [], set()
        for number, row in read_rows(path):
            if any(not nonempty_text(row.get(key)) for key in ("id", "text")):
                raise ValueError(f"{path}:{number}: id 和 text 必须为非空字符串")
            row = {**row, "id": row["id"].strip(), "text": row["text"].strip()}
            if row["id"] in seen:
                raise ValueError(f"{path}:{number}: 重复 ID：{row['id']}")
            rows.append(row)
            seen.add(row["id"])
        if not rows:
            raise ValueError(f"{path}: 数据为空")
        return rows

    documents, query_rows = read_records(corpus), read_records(queries)
    doc_ids = {row["id"]: i for i, row in enumerate(documents)}
    relevant = []
    for row in query_rows:
        labels = row.get("relevant_ids")
        if not isinstance(labels, list) or not labels or any(not nonempty_text(s) for s in labels):
            raise ValueError(f"query {row['id']}: relevant_ids 必须为非空文档 ID 列表")
        labels = {s.strip() for s in labels}
        if labels - doc_ids.keys():
            raise ValueError(f"query {row['id']}: 标注文档不存在：{sorted(labels - doc_ids.keys())}")
        relevant.append({doc_ids[value] for value in labels})
    return query_rows, documents, relevant


def encode_records(records, vectors, tokenize, label="文本"):
    """平均原始词向量后 L2 归一化，保留词次权重；OOV 不参与平均。"""
    embeddings = np.zeros((len(records), vectors.vector_size), dtype=np.float32)
    coverage = []
    for i, row in enumerate(records):
        tokens = tokenize(row["text"])
        known = [token for token in tokens if token in vectors]
        if known:
            mean = vectors.get_mean_vector(known, pre_normalize=False, post_normalize=True)
            if not np.isfinite(mean).all():
                raise ValueError(f"{label} {row['id']}: 平均词向量含非有限值")
            embeddings[i] = mean
        coverage.append({"tokens": len(tokens), "known_tokens": len(known),
                         "has_vector": bool(np.any(embeddings[i]))})
        if (i + 1) % 2000 == 0:
            print(f"编码{label}：{i + 1}/{len(records)}", flush=True)
    return embeddings, coverage


def coverage_summary(rows):
    tokens = sum(row["tokens"] for row in rows)
    known = sum(row["known_tokens"] for row in rows)
    return {"texts": len(rows), "tokens": tokens, "known_tokens": known,
            "oov_tokens": tokens - known, "token_coverage": known / tokens if tokens else 0.0,
            "zero_vector_texts": sum(not row["has_vector"] for row in rows)}


def ranked_batches(queries, documents, top_k=10, query_batch_size=64, corpus_chunk_size=20000,
                   excluded_indices=None):
    """精确点积检索；同分按候选文件顺序，零向量查询/文档不产生结果。"""
    if any(type(n) is not int or n < 1 for n in (top_k, query_batch_size, corpus_chunk_size)):
        raise ValueError("top_k 和分块大小必须为正整数")
    if (queries.ndim != 2 or documents.ndim != 2 or not len(queries) or not len(documents)
            or queries.shape[1] != documents.shape[1]):
        raise ValueError("查询与候选向量必须为非空、同维度二维数组")
    if not np.isfinite(queries).all() or not np.isfinite(documents).all():
        raise ValueError("向量包含非有限值")
    if excluded_indices is not None:
        if len(excluded_indices) != len(queries):
            raise ValueError("每个查询必须对应一组排除候选")
        if any(type(index) is not int or not 0 <= index < len(documents)
               for indices in excluded_indices for index in indices):
            raise ValueError("排除候选必须是候选库范围内的整数索引")
    valid_documents = np.any(documents != 0, axis=1)
    top_k = min(top_k, len(documents))
    for start in range(0, len(queries), query_batch_size):
        batch = queries[start:start + query_batch_size]
        best_scores = np.empty((len(batch), 0), dtype=np.float32)
        best_ids = np.empty((len(batch), 0), dtype=np.int64)
        for offset in range(0, len(documents), corpus_chunk_size):
            scores = batch @ documents[offset:offset + corpus_chunk_size].T
            scores[:, ~valid_documents[offset:offset + scores.shape[1]]] = -np.inf
            if excluded_indices is not None:
                for row, indices in enumerate(excluded_indices[start:start + len(batch)]):
                    local = [index - offset for index in indices if offset <= index < offset + scores.shape[1]]
                    scores[row, local] = -np.inf
            local_ids = np.argsort(-scores, axis=1, kind="stable")[:, :top_k]
            candidates = np.concatenate([best_scores, np.take_along_axis(scores, local_ids, axis=1)], axis=1)
            candidate_ids = np.concatenate([best_ids, local_ids + offset], axis=1)
            order = np.lexsort((candidate_ids, -candidates), axis=1)[:, :top_k]
            best_scores = np.take_along_axis(candidates, order, axis=1)
            best_ids = np.take_along_axis(candidate_ids, order, axis=1)
        for row in range(len(batch)):
            valid = np.isfinite(best_scores[row]) & bool(np.any(batch[row]))
            yield start + row, best_ids[row][valid].tolist(), best_scores[row][valid].tolist()


def retrieval_metrics(ranked_ids, relevant):
    """多正例、二值相关性；MRR/NDCG 截断到 10，未召回正例仍在分母。"""
    if not relevant:
        raise ValueError("每个查询必须至少有一个标注正例")
    hits = [int(index in relevant) for index in ranked_ids[:10]]
    result = {}
    for k in (1, 5, 10):
        count = sum(hits[:k])
        result[f"Hit@{k}"] = float(count > 0)
        result[f"Recall@{k}"] = count / len(relevant)
    first = next((i + 1 for i, hit in enumerate(hits) if hit), None)
    result["MRR@10"] = 1.0 / first if first else 0.0
    dcg = sum(hit / np.log2(rank + 2) for rank, hit in enumerate(hits))
    ideal = sum(1.0 / np.log2(rank + 2) for rank in range(min(10, len(relevant))))
    result["NDCG@10"] = float(dcg / ideal)
    return result
