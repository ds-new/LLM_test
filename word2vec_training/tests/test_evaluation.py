import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

from gensim.models import KeyedVectors
import numpy as np

from word2vec_training.corpus import WordTokenizer, fingerprint
from word2vec_training.evaluate_word2vec import evaluate
from word2vec_training.evaluation import (
    coverage_summary, encode_records, load_eval_data, ranked_batches, retrieval_metrics,
)


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def write_rows(self, name, rows):
        path = self.root / name
        path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows), encoding="utf-8")
        return path

    def test_triplets_merge_queries_and_keep_full_candidate_pool(self):
        path = self.write_rows("eval.jsonl", [
            {"query": "a", "pos": ["one", "one"], "neg": ["two"]},
            {"query": "a", "pos": ["three"], "neg": ["two"]},
            {"query": "b", "pos": ["two"], "neg": ["four"]},
        ])
        queries, docs, qrels = load_eval_data(eval_data=path)
        self.assertEqual([q["text"] for q in queries], ["a", "b"])
        self.assertEqual([d["text"] for d in docs], ["one", "two", "three", "four"])
        self.assertEqual(qrels, [{0, 2}, {1}])

    def test_public_corpus_keeps_ids_even_for_identical_texts(self):
        corpus = self.write_rows("corpus.jsonl", [{"id": "d1", "text": "same"},
                                                  {"id": "d2", "text": "same"},
                                                  {"id": "d3", "text": "distractor"}])
        queries = self.write_rows("queries.jsonl", [{"id": "q", "text": "query", "relevant_ids": ["d2"]}])
        _, docs, qrels = load_eval_data(corpus=corpus, queries=queries)
        self.assertEqual(len(docs), 3)
        self.assertEqual(qrels, [{1}])
        queries = self.write_rows("queries.jsonl", [{"id": "q", "text": "query", "relevant_ids": ["missing"]}])
        with self.assertRaisesRegex(ValueError, "不存在"):
            load_eval_data(corpus=corpus, queries=queries)

    def test_bad_labels_are_rejected(self):
        for row in ({"query": "q", "pos": []}, {"query": "q", "pos": "p"},
                    {"query": "q", "pos": ["p"], "neg": ["p"]}):
            with self.subTest(row=row), self.assertRaises(ValueError):
                load_eval_data(eval_data=self.write_rows("bad.jsonl", [row]))
        self.write_rows("empty.jsonl", [])
        with self.assertRaisesRegex(ValueError, "为空"):
            load_eval_data(eval_data=self.root / "empty.jsonl")

    def test_mean_vector_normalization_and_oov_coverage(self):
        vectors = KeyedVectors(vector_size=2)
        vectors.add_vectors(["a", "b"], np.array([[4, 0], [0, 1]], dtype=np.float32))
        rows = [{"id": "q", "text": "A b missing"}, {"id": "oov", "text": "unknown"}]
        embeds, coverage = encode_records(rows, vectors, WordTokenizer("whitespace", lowercase=True))
        np.testing.assert_allclose(embeds[0], [4 / np.sqrt(17), 1 / np.sqrt(17)], rtol=1e-6)
        np.testing.assert_array_equal(embeds[1], [0, 0])
        summary = coverage_summary(coverage)
        self.assertEqual(summary["token_coverage"], 0.5)
        self.assertEqual(summary["zero_vector_texts"], 1)

    def test_chunking_matches_exact_rank_and_zero_vectors_never_hit(self):
        queries = np.array([[1, 0], [0, 0], [-1, 0]], dtype=np.float32)
        docs = np.array([[1, 0], [1, 0], [0, 0], [-1, 0], [0, 1]], dtype=np.float32)
        for chunk_size in (1, 2, 100):
            with self.subTest(chunk_size=chunk_size):
                results = list(ranked_batches(queries, docs, top_k=3, query_batch_size=2,
                                               corpus_chunk_size=chunk_size))
                self.assertEqual([r[1] for r in results], [[0, 1, 4], [], [3, 4, 0]])

    def test_multi_positive_metrics_use_all_positive_labels(self):
        result = retrieval_metrics([2, 1, 0], {1, 3})
        self.assertEqual(result["Hit@1"], 0)
        self.assertEqual(result["Hit@5"], 1)
        self.assertEqual(result["Recall@5"], 0.5)
        self.assertEqual(result["MRR@10"], 0.5)
        self.assertAlmostEqual(result["NDCG@10"], (1 / np.log2(3)) / (1 + 1 / np.log2(3)))
        self.assertTrue(all(value == 0 for value in retrieval_metrics([], {1, 3}).values()))

    def test_exclusions_rerank_full_corpus_per_query(self):
        queries = np.array([[1, 0], [1, 0], [0, 0]], dtype=np.float32)
        docs = np.array([[1, 0]] * 12, dtype=np.float32)
        for chunk_size in (1, 4, 100):
            results = list(ranked_batches(queries, docs, corpus_chunk_size=chunk_size,
                                          query_batch_size=2, excluded_indices=[{0}, {1, 2}, set()]))
            self.assertEqual(results[0][1], list(range(1, 11)))
            self.assertEqual(results[1][1], [0, *range(3, 12)])
            self.assertEqual(results[2][1], [])
            self.assertEqual(retrieval_metrics(results[0][1], {10})["Hit@10"], 1)
        empty = list(ranked_batches(queries[:1], docs, excluded_indices=[set(range(12))]))
        self.assertEqual(empty[0][1], [])

    def test_invalid_exclusions_are_rejected(self):
        vectors = np.array([[1, 0]], dtype=np.float32)
        for exclusions in ([], [{-1}], [{1}], [{0.5}], [{True}]):
            with self.subTest(exclusions=exclusions), self.assertRaises(ValueError):
                list(ranked_batches(vectors, vectors, excluded_indices=exclusions))

    def make_fixture(self):
        vectors = KeyedVectors(vector_size=2)
        vectors.add_vectors(["cat", "pet", "plant"], np.array([[1, 0], [1, 0], [0, 1]], dtype=np.float32))
        path = self.root / "vectors.kv"
        vectors.save(str(path), separately=["vectors"])
        (self.root / "training_report.json").write_text(json.dumps({
            "parameters": {"tokenizer": "whitespace", "lowercase": True}, "sources": []}))
        corpus = self.write_rows("corpus.jsonl", [
            {"id": "d1", "text": "pet"}, {"id": "d2", "text": "plant"}, {"id": "d3", "text": "unknown"},
        ])
        queries = self.write_rows("queries.jsonl", [
            {"id": "q1", "text": "CAT", "relevant_ids": ["d1"]},
            {"id": "q2", "text": "unknown", "relevant_ids": ["d3"]},
            {"id": "q3", "text": "plant", "relevant_ids": ["d2"]},
        ])
        return path, corpus, queries

    def test_standalone_cli_metrics_coverage_cases_and_overwrite(self):
        vector_path, corpus, queries = self.make_fixture()
        uploaded = self.root / "uploaded-evaluation"
        uploaded.mkdir()
        scripts = Path(__file__).resolve().parents[1]
        for name in ("corpus.py", "evaluation.py", "evaluate_word2vec.py"):
            shutil.copy2(scripts / name, uploaded / name)
        command = [sys.executable, "evaluate_word2vec.py", "--vectors", str(vector_path),
                   "--corpus", str(corpus), "--queries", str(queries), "--output-dir", "report",
                   "--corpus-chunk-size", "1", "--query-batch-size", "2", "--diagnose-self-matches"]
        environment = {key: value for key, value in os.environ.items() if key != "PYTHONPATH"}
        result = subprocess.run(command, cwd=uploaded, env=environment, capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        summary = json.loads((uploaded / "report/summary.json").read_text())
        self.assertEqual(summary["corpus_size"], 3)
        self.assertAlmostEqual(summary["metrics"]["Hit@1"], 2 / 3)
        self.assertAlmostEqual(summary["metrics"]["Recall@10"], 2 / 3)
        self.assertEqual(summary["query_coverage"]["zero_vector_texts"], 1)
        self.assertAlmostEqual(summary["query_coverage"]["token_coverage"], 2 / 3)
        cases = [json.loads(line) for line in (uploaded / "report/results.jsonl").read_text().splitlines()]
        self.assertEqual(cases[1]["results"], [])
        self.assertTrue(all(hit["id"] != "d3" for row in cases for hit in row["results"]))
        self.assertEqual(summary["review_cases"], 1)
        self.assertEqual(summary["self_match_diagnostic"]["metrics"], summary["metrics"])
        self.assertEqual(summary["self_match_diagnostic"]["excluded_query_document_pairs"], 0)
        audit = json.loads((uploaded / "report/audit.json").read_text())
        self.assertTrue(any(row["path"].endswith(".npy") for row in audit["model_files"]))
        self.assertTrue(audit["preprocessing"]["lowercase"])
        result = subprocess.run(command, cwd=uploaded, env=environment, capture_output=True, text=True, timeout=30)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("已存在", result.stderr)
        self.assertEqual(list(uploaded.glob(".word2vec-eval-*")), [])

    def test_query_sampling_keeps_complete_corpus_and_repeats_by_seed(self):
        vector_path, corpus, queries = self.make_fixture()
        selected = []
        for name in ("first", "second"):
            summary = evaluate(vector_path, self.root / name, corpus=corpus, queries=queries,
                               limit_queries=1, seed=3)
            self.assertEqual(summary["total_queries"], 3)
            self.assertEqual(summary["evaluated_queries"], 1)
            self.assertEqual(summary["corpus_size"], 3)
            selected.append(json.loads((self.root / name / "results.jsonl").read_text())["id"])
        self.assertEqual(selected[0], selected[1])

    def test_triplet_evaluation_reports_unlabelled_self_matches(self):
        vector_path, _, _ = self.make_fixture()
        path = self.write_rows("eval.jsonl", [
            {"query": "cat", "pos": ["pet"], "neg": ["plant"]},
            {"query": "plant", "pos": ["plant"], "neg": ["cat"]},
        ])
        summary = evaluate(vector_path, self.root / "triplets", eval_data=path)
        self.assertEqual(summary["corpus_size"], 3)
        self.assertEqual(summary["queries_with_unlabelled_self_match"], 1)
        self.assertEqual(summary["metrics"]["Hit@1"], 1)
        self.assertEqual(summary["raw_top1_unlabelled_self_match_queries"], 0)
        self.assertEqual(summary["raw_top10_unlabelled_self_match_queries"], 1)
        self.assertNotIn("self_match_diagnostic", summary)

    def test_diagnostic_preserves_raw_metrics_and_labelled_identical_documents(self):
        vector_path, _, _ = self.make_fixture()
        corpus = self.write_rows("diagnostic-corpus.jsonl", [
            {"id": "self1", "text": "cat"}, {"id": "self2", "text": "cat"},
            {"id": "positive", "text": "pet"}, {"id": "oov", "text": "unknown"},
        ])
        queries = self.write_rows("diagnostic-queries.jsonl", [
            {"id": "q1", "text": "cat", "relevant_ids": ["positive"]},
            {"id": "q2", "text": "cat", "relevant_ids": ["self1"]},
            {"id": "q3", "text": "unknown", "relevant_ids": ["oov"]},
        ])
        raw = evaluate(vector_path, self.root / "raw", corpus=corpus, queries=queries)
        summary = evaluate(vector_path, self.root / "diagnostic", corpus=corpus, queries=queries,
                           diagnose_self_matches=True, corpus_chunk_size=1)
        self.assertEqual(summary["metrics"], raw["metrics"])
        self.assertEqual((self.root / "raw/results.jsonl").read_bytes(),
                         (self.root / "diagnostic/results.jsonl").read_bytes())
        self.assertEqual(summary["queries_with_unlabelled_self_match"], 2)
        self.assertEqual(summary["raw_top1_unlabelled_self_match_queries"], 1)
        self.assertEqual(summary["raw_top10_unlabelled_self_match_queries"], 2)
        comparison = summary["self_match_diagnostic"]
        self.assertAlmostEqual(summary["metrics"]["Hit@1"], 1 / 3)
        self.assertAlmostEqual(comparison["metrics"]["Hit@1"], 2 / 3)
        self.assertAlmostEqual(comparison["metric_deltas_vs_raw"]["Hit@1"], 1 / 3)
        self.assertEqual(comparison["excluded_query_document_pairs"], 3)
        cases = [json.loads(line) for line in
                 (self.root / "diagnostic/diagnostic_results.jsonl").read_text().splitlines()]
        self.assertEqual([hit["id"] for hit in cases[0]["results"]], ["positive"])
        self.assertEqual([hit["id"] for hit in cases[1]["results"]], ["self1", "positive"])
        self.assertEqual(cases[2]["results"], [])
        self.assertEqual(cases[2]["excluded_document_ids"], [])
        self.assertEqual(cases[1]["relevant_documents"], [{"id": "self1", "text": "cat"}])
        self.assertEqual(comparison["review_cases"], 1)
        self.assertIn("对照", (self.root / "diagnostic/report.md").read_text())

    def test_training_file_leakage_and_unknown_preprocessing_are_rejected(self):
        vector_path, corpus, queries = self.make_fixture()
        report = self.root / "training_report.json"
        report.write_text(json.dumps({"parameters": {"tokenizer": "whitespace", "lowercase": True},
                                      "sources": [fingerprint(queries)]}))
        with self.assertRaisesRegex(ValueError, "训练输入完全一致"):
            evaluate(vector_path, self.root / "leaked", corpus=corpus, queries=queries)
        self.assertFalse((self.root / "leaked").exists())
        report.unlink()
        with self.assertRaisesRegex(ValueError, "预处理"):
            evaluate(vector_path, self.root / "unknown", corpus=corpus, queries=queries)
        summary = evaluate(vector_path, self.root / "explicit", corpus=corpus, queries=queries,
                           tokenizer="whitespace", lowercase=True)
        self.assertEqual(summary["evaluated_queries"], 3)


if __name__ == "__main__":
    unittest.main()
