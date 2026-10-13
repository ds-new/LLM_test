import json
import importlib.util
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

import numpy as np

from word2vec_training.compare_bge import BGEEncoder, compare, load_baseline
from word2vec_training.corpus import fingerprint
from word2vec_training.evaluation import retrieval_metrics


class BaselineTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.report = self.root / "word2vec-report"
        self.report.mkdir()
        self.documents = [{"id": "d0", "text": "猫"}, {"id": "d1", "text": "狗"},
                          {"id": "d2", "text": "猫"}]
        self.queries = [{"id": "q0", "text": "猫", "relevant_ids": ["d0", "d2"]},
                        {"id": "q1", "text": "问题", "relevant_ids": ["d1"]},
                        {"id": "q2", "text": "狗", "relevant_ids": ["d1"]}]
        self.corpus_path = self.write_rows(self.root / "corpus.jsonl", self.documents)
        self.queries_path = self.write_rows(self.root / "queries.jsonl", self.queries)
        rows = []
        for index in (2, 0):
            query = self.queries[index]
            relevant = {i for i, d in enumerate(self.documents) if d["id"] in query["relevant_ids"]}
            rows.append({"id": query["id"], "query": query["text"],
                         "relevant_documents": [self.documents[i] for i in sorted(relevant)],
                         "metrics": retrieval_metrics([0, 1, 2], relevant),
                         "results": [{"rank": i + 1, **doc, "score": 1.0 - i * 0.1,
                                      "is_relevant": i in relevant} for i, doc in enumerate(self.documents)]})
        self.rows = rows
        self.write_rows(self.report / "results.jsonl", rows)
        self.summary = {"method": "word2vec_mean_l2_cosine", "total_queries": 3,
                        "evaluated_queries": 2, "corpus_size": 3,
                        "metrics": {key: sum(row["metrics"][key] for row in rows) / 2 for key in rows[0]["metrics"]}}
        self.write_json(self.report / "summary.json", self.summary)
        self.write_json(self.report / "audit.json", {"data_files": [fingerprint(self.corpus_path), fingerprint(self.queries_path)]})

    def write_json(self, path, data):
        path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")

    def write_rows(self, path, rows):
        path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows), encoding="utf-8")
        return path

    def load(self):
        return load_baseline(self.report, self.corpus_path, self.queries_path)

    def test_keeps_exact_query_ids_order_full_corpus_and_multiple_labels(self):
        queries, docs, qrels, rows, summary, sources = self.load()
        self.assertEqual([row["id"] for row in queries], ["q2", "q0"])
        self.assertEqual(docs, self.documents)
        self.assertEqual(qrels, [{1}, {0, 2}])
        self.assertEqual(summary["metrics"]["Hit@1"], 0.5)
        self.assertEqual(len(sources), 5)

    def test_changed_data_is_rejected(self):
        self.corpus_path.write_text(self.corpus_path.read_text() + "\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "哈希"):
            self.load()

    def test_duplicate_and_unknown_query_ids_are_rejected(self):
        for replacement in ("q0", "missing"):
            self.rows[0]["id"] = replacement
            self.write_rows(self.report / "results.jsonl", self.rows)
            with self.assertRaises(ValueError):
                self.load()
        self.write_rows(self.report / "results.jsonl", [self.rows[1], self.rows[1]])
        with self.assertRaisesRegex(ValueError, "重复"):
            self.load()

    def test_wrong_labels_and_diagnostic_results_are_rejected(self):
        self.rows[0]["relevant_documents"] = [self.documents[0]]
        self.write_rows(self.report / "results.jsonl", self.rows)
        with self.assertRaisesRegex(ValueError, "正例标签"):
            self.load()
        self.rows[0]["relevant_documents"] = [self.documents[1]]
        self.rows[0]["excluded_document_ids"] = ["d0"]
        self.write_rows(self.report / "results.jsonl", self.rows)
        with self.assertRaisesRegex(ValueError, "未排除"):
            self.load()

    def test_corrupt_summary_and_incomplete_results_are_rejected(self):
        self.summary["metrics"]["Hit@1"] = 1.0
        self.write_json(self.report / "summary.json", self.summary)
        with self.assertRaisesRegex(ValueError, "汇总指标"):
            self.load()
        self.write_rows(self.report / "results.jsonl", self.rows[:1])
        with self.assertRaisesRegex(ValueError, "查询数"):
            self.load()

    def test_invalid_rank_and_per_query_metrics_are_rejected(self):
        self.rows[0]["results"][0]["is_relevant"] = True
        self.write_rows(self.report / "results.jsonl", self.rows)
        with self.assertRaisesRegex(ValueError, "排名内容"):
            self.load()
        self.rows[0]["results"][0]["is_relevant"] = False
        self.rows[0]["metrics"]["MRR@10"] = 0
        self.write_rows(self.report / "results.jsonl", self.rows)
        with self.assertRaisesRegex(ValueError, "原始指标"):
            self.load()

    def make_tiny_model(self):
        import torch
        from transformers import BertConfig, BertModel, BertTokenizerFast

        torch.set_num_threads(1)
        torch.manual_seed(42)
        path = self.root / "tiny-bge"
        path.mkdir()
        vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", *"猫狗动物检索天气"]
        (path / "vocab.txt").write_text("\n".join(vocab), encoding="utf-8")
        BertTokenizerFast(vocab_file=str(path / "vocab.txt")).save_pretrained(path)
        BertModel(BertConfig(vocab_size=len(vocab), hidden_size=16, num_hidden_layers=1,
                             num_attention_heads=2, intermediate_size=24,
                             hidden_dropout_prob=0, attention_probs_dropout_prob=0)).save_pretrained(path)
        return path

    @unittest.skipUnless(importlib.util.find_spec("torch") and importlib.util.find_spec("transformers"), "需要 BGE 环境")
    def test_local_encoder_cls_normalization_instruction_and_truncation(self):
        import torch
        from transformers import BertModel, BertTokenizerFast

        path = self.make_tiny_model()
        records = [{"id": "short", "text": "猫"}, {"id": "long", "text": "狗动物天气" * 30}]
        encoder = BGEEncoder(path, "cpu", "检索", 8, 12)
        tokenizer = BertTokenizerFast.from_pretrained(str(path), local_files_only=True)
        model = BertModel.from_pretrained(str(path), local_files_only=True).eval()
        texts = [row["text"] for row in records]
        for query in (True, False):
            actual = encoder.encode(records, 2, is_query=query)
            inputs = tokenizer([("检索" if query else "") + text for text in texts],
                               padding=True, truncation=True, max_length=8 if query else 12, return_tensors="pt")
            self.assertEqual(inputs["input_ids"].shape[1], 8 if query else 12)
            if query:
                self.assertEqual(inputs["input_ids"][0, 1:3].tolist(), tokenizer.convert_tokens_to_ids(["检", "索"]))
            with torch.no_grad():
                cls = model(**inputs).last_hidden_state[:, 0].numpy()
            expected = cls / np.linalg.norm(cls, axis=1, keepdims=True)
            np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)
            np.testing.assert_allclose(np.linalg.norm(actual, axis=1), 1, atol=1e-6)

    @unittest.skipUnless(importlib.util.find_spec("torch") and importlib.util.find_spec("transformers"), "需要 BGE 环境")
    def test_standalone_cli_real_encoder_outputs_and_no_overwrite(self):
        model = self.make_tiny_model()
        upload = self.root / "standalone"
        upload.mkdir()
        for name in ("compare_bge.py", "corpus.py", "evaluation.py"):
            shutil.copy2(Path(__file__).resolve().parents[1] / name, upload / name)
        command = [sys.executable, "compare_bge.py", "--model", str(model), "--baseline-report", str(self.report),
                   "--corpus", str(self.corpus_path), "--queries", str(self.queries_path),
                   "--output-dir", "comparison", "--device", "cpu", "--batch-size", "2",
                   "--query-instruction", "检索", "--query-max-len", "8", "--passage-max-len", "12",
                   "--corpus-chunk-size", "1", "--query-batch-size", "1"]
        environment = {key: value for key, value in os.environ.items() if key != "PYTHONPATH"}
        environment.update({"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "OMP_NUM_THREADS": "1"})
        result = subprocess.run(command, cwd=upload, env=environment, capture_output=True, text=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        output = upload / "comparison"
        summary = json.loads((output / "comparison.json").read_text())
        self.assertEqual(summary["word2vec"], self.summary["metrics"])
        self.assertEqual(summary["evaluated_queries"], 2)
        self.assertEqual(summary["corpus_size"], 3)
        self.assertEqual(summary["bge"]["Recall@10"], 1.0)
        self.assertEqual(summary["paired_outcomes"]["both_hit_top10"], 2)
        selected = [json.loads(line) for line in (output / "selected_queries.jsonl").read_text().splitlines()]
        self.assertEqual([row["id"] for row in selected], ["q2", "q0"])
        results = [json.loads(line) for line in (output / "bge_results.jsonl").read_text().splitlines()]
        self.assertEqual([row["id"] for row in results], ["q2", "q0"])
        self.assertEqual([d["id"] for d in results[1]["relevant_documents"]], ["d0", "d2"])
        for key in summary["bge"]:
            self.assertAlmostEqual(summary["bge"][key], sum(row["metrics"][key] for row in results) / 2)
            self.assertAlmostEqual(summary["delta_bge_minus_word2vec"][key],
                                   summary["bge"][key] - summary["word2vec"][key])
        audit = json.loads((output / "audit.json").read_text())
        self.assertEqual(audit["selected_queries"], fingerprint(output / "selected_queries.jsonl"))
        self.assertTrue(audit["baseline_metrics_recomputed"])
        self.assertEqual(audit["bge_settings"]["query_instruction"], "检索")
        self.assertTrue(any(row["path"].endswith(".safetensors") for row in audit["bge_model_files"]))
        self.assertEqual(list(upload.glob(".bge-comparison-*")), [])
        original = (output / "comparison.json").read_bytes()
        result = subprocess.run(command, cwd=upload, env=environment, capture_output=True, text=True, timeout=60)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("已存在", result.stderr)
        self.assertEqual((output / "comparison.json").read_bytes(), original)

    def test_bad_settings_and_missing_model_fail_before_encoding(self):
        for overrides in ({"batch_size": 0}, {"device": "auto"}, {}):
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                compare(self.report, self.root / "missing", self.corpus_path, self.queries_path,
                        self.root / "output", **overrides)
        self.assertFalse((self.root / "output").exists())


if __name__ == "__main__":
    unittest.main()
