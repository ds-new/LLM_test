import copy
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

from word2vec_training.corpus import fingerprint
from word2vec_training.evaluation import retrieval_metrics
from word2vec_training.review_comparison import load_comparison, review_comparison


class ComparisonReviewTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.baseline, self.comparison = self.root / "baseline", self.root / "comparison"
        self.baseline.mkdir()
        self.comparison.mkdir()
        documents = [{"id": f"d{i}", "text": f"候选文档 {i}"} for i in range(12)]
        self.selected = [{"id": f"q{i}", "text": f"问题 {i}"} for i in range(4)]
        # 共同失败、Word2Vec 独有命中、BGE 独有命中、共同命中；第二条含两个正例。
        qrels = [{11}, {10, 11}, {11}, {0}]
        miss, hit = list(range(10)), [11, 10, *range(8)]
        rankings = {"word2vec": [miss, hit, miss, miss], "bge": [miss, miss, hit, miss]}
        rows = {}
        for name in rankings:
            rows[name] = [{"id": query["id"], "query": query["text"],
                           "relevant_documents": [documents[j] for j in sorted(qrels[i])],
                           "results": [{"rank": r, **documents[j], "score": 1 - r * 0.01,
                                        "is_relevant": j in qrels[i]} for r, j in enumerate(rankings[name][i], 1)],
                           "metrics": retrieval_metrics(rankings[name][i], qrels[i])}
                          for i, query in enumerate(self.selected)]
        for i, row in enumerate(rows["bge"]):
            row["word2vec_metrics"] = rows["word2vec"][i]["metrics"]
        self.rows = rows
        means = {name: {key: sum(r["metrics"][key] for r in values) / 4 for key in values[0]["metrics"]}
                 for name, values in rows.items()}
        original = {"method": "word2vec_mean_l2_cosine", "total_queries": 6,
                    "evaluated_queries": 4, "corpus_size": 12, "metrics": means["word2vec"]}
        # 原始候选/查询/模型不在本机；复核只依赖报告文件与其中记录的哈希。
        data = [{"path": "/unavailable/corpus.jsonl", "sha256": "corpus"},
                {"path": "/unavailable/queries.jsonl", "sha256": "queries"}]
        self.write_json(self.baseline / "summary.json", original)
        self.write_json(self.baseline / "audit.json", {"data_files": data})
        self.write_rows(self.baseline / "results.jsonl", rows["word2vec"])
        self.write_rows(self.comparison / "bge_results.jsonl", rows["bge"])
        self.write_rows(self.comparison / "selected_queries.jsonl", self.selected)
        self.summary = {"query_selection": "exact_ids_and_order_from_word2vec_raw_results",
                        "total_queries": 6, "evaluated_queries": 4, "corpus_size": 12, **means,
                        "delta_bge_minus_word2vec": {key: means["bge"][key] - val for key, val in means["word2vec"].items()},
                        "paired_outcomes": {f"{group}_top{k}": 1 for k in (1, 10)
                                            for group in ("both_hit", "word2vec_only", "bge_only", "neither_hit")}}
        self.write_json(self.comparison / "comparison.json", self.summary)
        self.write_json(self.comparison / "audit.json", {
            "baseline_and_data_files": [fingerprint(self.baseline / name)
                                        for name in ("summary.json", "audit.json", "results.jsonl")] + data,
            "selected_queries": fingerprint(self.comparison / "selected_queries.jsonl"),
            "bge_settings": {"query_max_len": 64, "passage_max_len": 256}, "device": "cpu"})

    def write_json(self, path, data):
        path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")

    def write_rows(self, path, rows):
        path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")

    def load(self):
        return load_comparison(self.baseline, self.comparison)

    def test_replay_moved_reports_without_original_models_or_data(self):
        w2v, bge, summary, audit, sources = self.load()
        self.assertEqual([row["id"] for row in w2v], ["q0", "q1", "q2", "q3"])
        self.assertEqual(summary["paired_outcomes"]["neither_hit_top10"], 1)
        self.assertEqual(bge[0]["metrics"]["Hit@10"], 0)
        self.assertEqual(w2v[1]["metrics"]["Recall@10"], 1)
        self.assertEqual(len(sources), 7)

    def test_rejects_mismatched_baseline_hash(self):
        path = self.baseline / "results.jsonl"
        path.write_text(path.read_text() + "\n")
        with self.assertRaisesRegex(ValueError, "哈希"):
            self.load()

    def test_rejects_reordered_or_missing_queries_and_corrupt_rankings(self):
        variants = [list(reversed(self.rows["bge"])), self.rows["bge"][:-1]]
        for field, value in (("rank", 3), ("is_relevant", True), ("score", float("nan")), ("id", "d1")):
            rows = copy.deepcopy(self.rows["bge"])
            rows[0]["results"][0][field] = value
            variants.append(rows)
        for rows in variants:
            with self.subTest(rows=rows[0]["id"]):
                self.write_rows(self.comparison / "bge_results.jsonl", rows)
                with self.assertRaises(ValueError):
                    self.load()

    def test_rejects_inconsistent_summary_or_labels(self):
        self.summary["paired_outcomes"]["neither_hit_top10"] = 24
        self.write_json(self.comparison / "comparison.json", self.summary)
        with self.assertRaisesRegex(ValueError, "配对计数"):
            self.load()
        self.summary["paired_outcomes"]["neither_hit_top10"] = 1
        self.write_json(self.comparison / "comparison.json", self.summary)
        self.rows["bge"][0]["relevant_documents"][0]["text"] = "改变正例"
        self.write_rows(self.comparison / "bge_results.jsonl", self.rows["bge"])
        with self.assertRaisesRegex(ValueError, "正例标签"):
            self.load()

    def test_standalone_cli_exports_full_text_and_preserves_existing_output(self):
        standalone = self.root / "standalone"
        standalone.mkdir()
        for name in ("review_comparison.py", "corpus.py", "evaluation.py"):
            shutil.copy2(Path(__file__).resolve().parents[1] / name, standalone / name)
        command = [sys.executable, "review_comparison.py", "--baseline-report", str(self.baseline),
                   "--comparison-report", str(self.comparison), "--output-dir", "review"]
        env = {key: value for key, value in os.environ.items() if key != "PYTHONPATH"}
        result = subprocess.run(command, cwd=standalone, env=env, capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        output = standalone / "review"
        rows = [json.loads(line) for line in (output / "review.jsonl").read_text().splitlines()]
        self.assertEqual([row["id"] for row in rows], ["q0", "q1"])
        self.assertEqual([row["group"] for row in rows], ["neither_hit_top10", "word2vec_only_top10"])
        self.assertEqual(rows[1]["relevant_documents"], self.rows["word2vec"][1]["relevant_documents"])
        self.assertEqual(rows[0]["bge_results"], self.rows["bge"][0]["results"])
        self.assertEqual(rows[0]["manual_review"], {"category": "unreviewed", "notes": ""})
        self.assertIn("候选文档 11", (output / "review.md").read_text())
        self.assertEqual(list(standalone.glob(".comparison-review-*")), [])
        before = (output / "review.jsonl").read_bytes()
        result = subprocess.run(command, cwd=standalone, env=env, capture_output=True, text=True, timeout=30)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("已存在", result.stderr)
        self.assertEqual(before, (output / "review.jsonl").read_bytes())

    def test_exports_more_than_ten_cases_per_group_and_escapes_markup(self):
        expanded = {name: [] for name in self.rows}
        selected = []
        for n in range(12):
            for i, query in enumerate(self.selected):
                query = {**query, "id": f"{n}-{query['id']}", "text": "<script>data</script>\n```"}
                selected.append(query)
                for name in expanded:
                    expanded[name].append({**self.rows[name][i], "id": query["id"], "query": query["text"]})
        self.write_rows(self.baseline / "results.jsonl", expanded["word2vec"])
        self.write_rows(self.comparison / "bge_results.jsonl", expanded["bge"])
        self.write_rows(self.comparison / "selected_queries.jsonl", selected)
        original = json.loads((self.baseline / "summary.json").read_text())
        for data in (self.summary, original):
            data.update({"evaluated_queries": 48, "total_queries": 72})
        self.summary["paired_outcomes"] = {key: 12 for key in self.summary["paired_outcomes"]}
        self.write_json(self.baseline / "summary.json", original)
        self.write_json(self.comparison / "comparison.json", self.summary)
        audit = json.loads((self.comparison / "audit.json").read_text())
        audit["baseline_and_data_files"][:3] = [fingerprint(self.baseline / name)
                                                for name in ("summary.json", "audit.json", "results.jsonl")]
        audit["selected_queries"] = fingerprint(self.comparison / "selected_queries.jsonl")
        self.write_json(self.comparison / "audit.json", audit)
        output = self.root / "export"
        summary = review_comparison(self.baseline, self.comparison, output)
        self.assertEqual(summary["case_counts"], {"neither_hit_top10": 12, "word2vec_only_top10": 12})
        self.assertEqual(summary["exported_cases"], 24)
        text = (output / "review.md").read_text()
        self.assertNotIn("<script>", text)
        self.assertIn("&lt;script&gt;", text)

    def test_failed_validation_does_not_publish_review(self):
        output = self.root / "failed"
        self.write_rows(self.comparison / "bge_results.jsonl", self.rows["bge"][:-1])
        with self.assertRaisesRegex(ValueError, "数量"):
            review_comparison(self.baseline, self.comparison, output)
        self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
