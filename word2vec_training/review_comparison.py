"""从已有 Word2Vec / BGE 报告导出全部共同失败与 Word2Vec 独有命中案例。"""

import argparse
from collections import Counter
import html
import json
import math
from pathlib import Path
import tempfile

if __package__ in (None, ""):
    from corpus import fingerprint
    from evaluation import read_rows, retrieval_metrics
else:
    from .corpus import fingerprint
    from .evaluation import read_rows, retrieval_metrics


def same_metrics(actual, expected):
    return isinstance(actual, dict) and all(
        type(actual.get(key)) in (int, float) and math.isfinite(actual[key])
        and math.isclose(actual[key], value, rel_tol=1e-9, abs_tol=1e-10)
        for key, value in expected.items())


def checked_metrics(row):
    """用保存的 top-10 与标注重算指标，拒绝重复排名或伪造命中标志。"""
    positives, hits = row.get("relevant_documents"), row.get("results")
    if not isinstance(positives, list) or not positives or not isinstance(hits, list) or len(hits) > 10:
        raise ValueError("逐查询报告需要非空正例列表和最多 10 个排名结果")
    for doc in [*positives, *hits]:
        if not isinstance(doc, dict) or any(not isinstance(doc.get(k), str) or not doc[k].strip() for k in ("id", "text")):
            raise ValueError("文档 id/text 必须为非空字符串")
    relevant = {doc["id"] for doc in positives}
    if len(relevant) != len(positives) or row.get("excluded_document_ids"):
        raise ValueError("仅支持正例 ID 不重复、未排除候选的原始报告")
    ranked, previous = [], math.inf
    for rank, hit in enumerate(hits, 1):
        score = hit.get("score")
        if (hit.get("rank") != rank or hit["id"] in ranked
                or type(score) not in (int, float) or not math.isfinite(score) or score > previous
                or type(hit.get("is_relevant")) is not bool or hit["is_relevant"] != (hit["id"] in relevant)):
            raise ValueError("排名序号、分数、文档 ID 或相关性标志无效")
        ranked.append(hit["id"])
        previous = score
    metrics = retrieval_metrics(ranked, relevant)
    if not same_metrics(row.get("metrics"), metrics):
        raise ValueError("逐查询指标无法由已保存排名复现")
    return metrics


def load_comparison(baseline_report, comparison_report):
    """校验报告配对与汇总；不访问 audit 中记录的原机器绝对路径。"""
    baseline, comparison = Path(baseline_report).expanduser(), Path(comparison_report).expanduser()
    paths = [baseline / name for name in ("summary.json", "audit.json", "results.jsonl")]
    paths += [comparison / name for name in ("comparison.json", "audit.json", "bge_results.jsonl", "selected_queries.jsonl")]
    sources = [fingerprint(path) for path in paths]
    original, original_audit, summary, audit = [json.loads(paths[i].read_text(encoding="utf-8")) for i in (0, 1, 3, 4)]
    if original.get("method") != "word2vec_mean_l2_cosine" or summary.get("query_selection") != "exact_ids_and_order_from_word2vec_raw_results":
        raise ValueError("需要 Word2Vec 原始报告及对应的 BGE 同样本对照报告")
    recorded = audit.get("baseline_and_data_files", [])
    if (len(recorded) != 5 or [s.get("sha256") for s in recorded[:3]] != [s["sha256"] for s in sources[:3]]
            or [s.get("sha256") for s in recorded[3:]] != [s.get("sha256") for s in original_audit.get("data_files", [])]
            or audit.get("selected_queries", {}).get("sha256") != sources[-1]["sha256"]):
        raise ValueError("报告哈希不匹配：请使用生成该 BGE 对照的原始 Word2Vec 报告")
    for key in ("total_queries", "evaluated_queries", "corpus_size"):
        if type(summary.get(key)) is not int or summary[key] < 1 or summary[key] != original.get(key):
            raise ValueError(f"双方报告的 {key} 不一致或无效")
    w2v_rows, bge_rows, selected = [[row for _, row in read_rows(paths[i])] for i in (2, 5, 6)]
    count = summary["evaluated_queries"]
    if any(len(rows) != count for rows in (w2v_rows, bge_rows, selected)):
        raise ValueError("逐查询报告数量与汇总不一致")
    seen, paired, totals = set(), Counter(), {name: Counter() for name in ("word2vec", "bge")}
    for w2v, bge, query in zip(w2v_rows, bge_rows, selected):
        query_id = w2v.get("id")
        if (not isinstance(query_id, str) or not query_id.strip() or query_id in seen
                or query_id != bge.get("id") or query_id != query.get("id")
                or not isinstance(w2v.get("query"), str) or not w2v["query"].strip()
                or w2v["query"] != bge.get("query") or w2v["query"] != query.get("text")):
            raise ValueError("查询 ID/文本/顺序不一致或重复")
        seen.add(query_id)
        wm, bm = checked_metrics(w2v), checked_metrics(bge)
        labels = [{d["id"]: d["text"] for d in row["relevant_documents"]} for row in (w2v, bge)]
        if labels[0] != labels[1] or not same_metrics(bge.get("word2vec_metrics"), wm):
            raise ValueError("双方正例标签或记录的 Word2Vec 指标不一致")
        texts = dict(labels[0])
        for doc in [*w2v["results"], *bge["results"]]:
            if texts.setdefault(doc["id"], doc["text"]) != doc["text"]:
                raise ValueError("同一文档 ID 的文本不一致")
        for name, metrics in (("word2vec", wm), ("bge", bm)):
            totals[name].update(metrics)
        for k in (1, 10):
            w, b = wm[f"Hit@{k}"], bm[f"Hit@{k}"]
            group = "both_hit" if w and b else "word2vec_only" if w else "bge_only" if b else "neither_hit"
            paired[f"{group}_top{k}"] += 1
    means = {name: {key: value / count for key, value in values.items()} for name, values in totals.items()}
    delta = {key: means["bge"][key] - value for key, value in means["word2vec"].items()}
    outcomes = {f"{group}_top{k}": paired[f"{group}_top{k}"] for k in (1, 10)
                for group in ("both_hit", "word2vec_only", "bge_only", "neither_hit")}
    if (any(not same_metrics(summary.get(name), metrics) for name, metrics in means.items())
            or not same_metrics(original.get("metrics"), means["word2vec"])
            or not same_metrics(summary.get("delta_bge_minus_word2vec"), delta)
            or summary.get("paired_outcomes") != outcomes):
        raise ValueError("汇总指标或配对计数无法由逐查询报告复现")
    return w2v_rows, bge_rows, summary, audit, sources


def review_comparison(baseline_report, comparison_report, output_dir):
    output = Path(output_dir).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"结果目录已存在：{output}；请指定新的 --output-dir")
    w2v_rows, bge_rows, comparison, audit, sources = load_comparison(baseline_report, comparison_report)
    cases = []
    counts = {"neither_hit_top10": 0, "word2vec_only_top10": 0}
    # 优先共同失败，其次 Word2Vec 独有命中；组内保持原查询顺序，不设截断数量。
    for group in counts:
        for w2v, bge in zip(w2v_rows, bge_rows):
            if bge["metrics"]["Hit@10"] or bool(w2v["metrics"]["Hit@10"]) != (group == "word2vec_only_top10"):
                continue
            cases.append({"group": group, "id": w2v["id"], "query": w2v["query"],
                          "relevant_documents": w2v["relevant_documents"],
                          "word2vec_metrics": w2v["metrics"], "bge_metrics": bge["metrics"],
                          "word2vec_results": w2v["results"], "bge_results": bge["results"],
                          "manual_review": {"category": "unreviewed", "notes": ""}})
            counts[group] += 1
    summary = {"evaluated_queries": comparison["evaluated_queries"], "corpus_size": comparison["corpus_size"],
               "case_counts": counts, "exported_cases": len(cases),
               "bge_settings": audit.get("bge_settings", {}), "device": audit.get("device"),
               "scope": "all neither_hit_top10 and word2vec_only_top10 cases; original labels unchanged"}
    lines = ["# Word2Vec / BGE 失败案例复核", "",
             f"查询 {summary['evaluated_queries']} 条，候选 {summary['corpus_size']} 篇。",
             f"共同前十未命中：{counts['neither_hit_top10']}；Word2Vec 独有前十命中：{counts['word2vec_only_top10']}。",
             "以上两组全部导出，不受原 review.jsonl 的抽样数量限制；原始指标及标签不变。", "",
             "## 复核方法", "",
             "先检查标注正例是否回答了查询，再比较两边的前十结果。未标正例不等于语义不相关。",
             "在 review.jsonl 的 manual_review 中记录 category 和 notes；初始均为 unreviewed。",
             "可记录 retrieval_miss（相关正例未召回）、possible_missing_label（疑似漏标）、",
             "possible_label_issue（疑似错标）、needs_truncation_check（需要检查截断）、",
             "domain_term（领域词问题）或 other，并注明依据及相关文档 ID。",
             "仅凭文本长度无法确认截断；需用本次 BGE tokenizer、指令及长度设置检查正例是否被截去关键部分。",
             "只有保存的 top-10，没有正例的全库排名；不能据此声称正例在第 11 名或给出确切名次。", "",
             "## 本次编码设置", "", "```json",
             json.dumps(summary['bge_settings'], ensure_ascii=True, indent=2), "```", ""]

    def quoted(text):
        # 使用转义后的 HTML pre 保留全文和换行，避免语料中的 Markdown 改写报告结构。
        return "<pre>" + html.escape(text) + "</pre>"

    for number, case in enumerate(cases, 1):
        lines += [f"## 案例 {number} · {case['group']}", "",
                  "查询 ID：", quoted(case["id"]), "查询：", quoted(case["query"]),
                  "人工复核：unreviewed；分类与依据：待填写。", "", "### 标注正例（全文）", ""]
        for doc in case["relevant_documents"]:
            lines += [quoted(doc["id"] + "\n" + doc["text"])]
        for name in ("word2vec", "bge"):
            lines += ["", f"### {name.upper()} Top-10（全文）", ""]
            for hit in case[f"{name}_results"]:
                lines += [f"第 {hit['rank']} 名 · 分数 {hit['score']:.6f} · 标注正例 {hit['is_relevant']}",
                          quoted(hit["id"] + "\n" + hit["text"])]
            if not case[f"{name}_results"]:
                lines += ["无排名结果。"]
    if not cases:
        lines += ["本次没有符合条件的案例。"]
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".comparison-review-", dir=output.parent) as temporary:
        staging = Path(temporary) / "report"
        staging.mkdir()
        (staging / "review.jsonl").write_text("".join(json.dumps(case, ensure_ascii=False, allow_nan=False) + "\n"
                                                   for case in cases), encoding="utf-8")
        (staging / "review.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        review_audit = {"source_reports": sources, "validation": "report hashes, query order, labels, per-query metrics and paired totals",
                        "limits": "no access to original corpus/model; does not rerun retrieval or audit training-data overlap"}
        for name, value in (("summary.json", summary), ("audit.json", review_audit)):
            (staging / name).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        if any(fingerprint(source["path"]) != source for source in sources):
            raise ValueError("复核期间输入报告发生变化，请固定文件后重试")
        if output.exists():
            raise FileExistsError(f"结果目录已存在：{output}")
        staging.rename(output)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    print(f"复核案例已保存：{output}", flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-report", required=True, help="对应的 Word2Vec 原始报告目录")
    parser.add_argument("--comparison-report", required=True, help="compare_bge.py 生成的报告目录")
    parser.add_argument("--output-dir", required=True)
    review_comparison(**vars(parser.parse_args()))


if __name__ == "__main__":
    try:
        main()
    except ImportError as error:
        raise SystemExit(f"依赖导入失败：{error}；可使用已有 Word2Vec 或 BGE 环境") from None
    except (OSError, ValueError) as error:
        raise SystemExit(str(error)) from None
    except KeyboardInterrupt:
        raise SystemExit(130) from None
