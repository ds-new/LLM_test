import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

from word2vec_training.corpus import prepare_corpus
from word2vec_training.train_word2vec import train_word2vec


class Word2VecTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def test_jsonl_fields_and_sentence_boundaries(self):
        source = self.root / "train.jsonl"
        source.write_text(json.dumps({"id": "exclude_me", "query": "Cat Dog\nCat Pet",
                                      "pos": ["Dog Pet"], "neg": ["Tree Plant"]}), encoding="utf-8")
        target = self.root / "tokens.txt"
        counts = prepare_corpus([source], target, ["query", "pos", "neg"],
                                tokenizer="whitespace", lowercase=True)
        self.assertEqual(target.read_text().splitlines(),
                         ["cat dog", "cat pet", "dog pet", "tree plant"])
        self.assertEqual(counts, {"texts": 3, "sentences": 4, "tokens": 8})

    def test_long_lines_keep_all_tokens_and_max_texts(self):
        source = self.root / "long.txt"
        source.write_text("猫 狗 " * 6000 + "\n忽略 此行\n", encoding="utf-8")
        target = self.root / "tokens.txt"
        counts = prepare_corpus([source], target, tokenizer="whitespace", max_texts=1)
        self.assertEqual(counts, {"texts": 1, "sentences": 2, "tokens": 12000})
        self.assertEqual([len(line.split()) for line in target.read_text().splitlines()], [10000, 2000])

    def test_bad_jsonl_has_source_line(self):
        source = self.root / "bad.jsonl"
        source.write_text('\n{"text": 123}\n', encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "bad.jsonl:2"):
            prepare_corpus([source], self.root / "tokens.txt", tokenizer="whitespace")

    def test_chinese_segmentation_filters_punctuation(self):
        source = self.root / "chinese.txt"
        source.write_text("北京是中国的城市。Word2Vec！\n", encoding="utf-8")
        target = self.root / "tokens.txt"
        prepare_corpus([source], target, lowercase=True)
        tokens = target.read_text().split()
        self.assertIn("北京", tokens)
        self.assertIn("word2vec", tokens)
        self.assertNotIn("。", tokens)

    def test_both_algorithms_train_and_all_formats_reload(self):
        from gensim.models import KeyedVectors, Word2Vec
        import numpy as np

        source = self.root / "train.txt"
        source.write_text("cat dog pet\ndog animal pet\ntree plant leaf\n" * 10, encoding="utf-8")
        for algorithm, sg in (("skipgram", 1), ("cbow", 0)):
            with self.subTest(algorithm=algorithm):
                output = self.root / algorithm
                report = train_word2vec([source], output, tokenizer="whitespace", algorithm=algorithm,
                                        vector_size=12, min_count=1, epochs=2, sample=0)
                model = Word2Vec.load(str(output / "word2vec.model"))
                vectors = KeyedVectors.load(str(output / "vectors.kv"))
                exported = KeyedVectors.load_word2vec_format(str(output / "vectors.txt"))
                self.assertEqual(model.sg, sg)
                self.assertEqual(vectors.vector_size, 12)
                self.assertEqual(vectors.index_to_key, model.wv.index_to_key)
                np.testing.assert_array_equal(vectors.vectors, model.wv.vectors)
                np.testing.assert_allclose(exported.vectors, model.wv.vectors)
                self.assertTrue(np.isfinite(vectors.vectors).all())
                self.assertTrue(np.any(model.syn1neg != 0), "训练必须实际更新权重")
                self.assertEqual(report["raw_words_all_epochs"], 180)
                self.assertEqual(report["effective_words_all_epochs"], 180)
                self.assertEqual(len(report["sources"][0]["sha256"]), 64)
                self.assertEqual(json.loads((output / "training_report.json").read_text()), report)

    def test_untrainable_corpora_do_not_publish_outputs(self):
        source = self.root / "train.txt"
        for contents, minimum in (("\n", 1), ("cat cat", 1), ("cat\ndog\n", 1), ("cat dog", 10)):
            with self.subTest(contents=contents, minimum=minimum):
                source.write_text(contents, encoding="utf-8")
                output = self.root / "failed"
                with self.assertRaises(ValueError):
                    train_word2vec([source], output, tokenizer="whitespace", min_count=minimum)
                self.assertFalse(output.exists())
                self.assertEqual(list(self.root.glob(".word2vec-*")), [])

    def test_invalid_parameters_and_existing_output(self):
        source = self.root / "train.txt"
        source.write_text("cat dog pet\n", encoding="utf-8")
        for arguments in ({"epochs": 0}, {"vector_size": True}, {"sample": float("nan")},
                          {"seed": 2**32}, {"tokenizer": "unknown"}, {"fields": "text"}):
            with self.subTest(arguments=arguments), self.assertRaises(ValueError):
                train_word2vec([source], self.root / "invalid", **arguments)
        existing = self.root / "existing"
        existing.mkdir()
        marker = existing / "keep.txt"
        marker.write_text("keep")
        with self.assertRaises(FileExistsError):
            train_word2vec([source], existing)
        self.assertEqual(marker.read_text(), "keep")

    def run_cli(self, script, *args):
        scripts = Path(__file__).resolve().parents[1]
        return subprocess.run([sys.executable, str(scripts / script), *map(str, args)], cwd=self.root,
                              env={**os.environ, "PYTHONHASHSEED": "42"},
                              capture_output=True, text=True, timeout=30)

    def test_cli_default_config_overrides_and_inspection(self):
        output = self.root / "cli"
        result = self.run_cli("train_word2vec.py", "--output-dir", output, "--vector-size", 8,
                              "--epochs", 2, "--max-texts", 2, "--no-lowercase")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        report = json.loads((output / "training_report.json").read_text())
        self.assertEqual(report["texts"], 2)
        self.assertFalse(report["parameters"]["lowercase"])
        result = self.run_cli("inspect_word2vec.py", "--vectors", output / "vectors.kv",
                              "--words", "模型", "词表中不存在的词语", "--topn", 3)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        inspection = json.loads(result.stdout)
        self.assertEqual(inspection["vector_size"], 8)
        self.assertEqual(len(inspection["words"][0]["similar"]), 3)
        self.assertFalse(inspection["words"][1]["found"])

    def test_cli_custom_config_paths_and_unknown_options(self):
        directory = self.root / "config-directory"
        directory.mkdir()
        (directory / "corpus.jsonl").write_text(json.dumps({"id": "ignored", "query": "cat dog",
                                                            "pos": ["dog pet"]}), encoding="utf-8")
        config = {"corpus_files": ["corpus.jsonl"], "output_dir": "trained", "fields": ["query", "pos"],
                  "tokenizer": "whitespace", "min_count": 1, "sample": 0, "epochs": 1}
        path = directory / "config.json"
        path.write_text(json.dumps(config))
        result = self.run_cli("train_word2vec.py", "--config", path)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        report = json.loads((directory / "trained/training_report.json").read_text())
        self.assertEqual(report["vocab_size"], 3)
        config["invalid_option"] = 1
        path.write_text(json.dumps(config))
        result = self.run_cli("train_word2vec.py", "--config", path)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("未知参数", result.stderr)

    def test_inspection_filters_all_candidates_by_frequency(self):
        from gensim.models import KeyedVectors
        import numpy as np

        vectors = KeyedVectors(vector_size=2)
        # 故意不按词频排列：查询词低频，最近邻也低频，验证筛选后重新取 top-n。
        keys = ["查询", "罕见词", "常见词", "边界词", "另一个词"]
        vectors.add_vectors(keys, np.array([[1, 0], [1, 0.001], [0.8, 0.6], [0.9, 0.4], [0, 1]], dtype=np.float32))
        for word, count in zip(keys, [1, 5, 100, 50, 60]):
            vectors.set_vecattr(word, "count", count)
        path = self.root / "vectors.kv"
        vectors.save(str(path))
        result = self.run_cli("inspect_word2vec.py", "--vectors", path, "--words", "查询", "不存在",
                              "--topn", 2, "--min-count", 50)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        report = json.loads(result.stdout)
        self.assertEqual(report["candidate_vocab_size"], 3)
        self.assertEqual(report["words"][0]["count"], 1)
        self.assertEqual([row["word"] for row in report["words"][0]["similar"]], ["边界词", "常见词"])
        self.assertEqual([row["count"] for row in report["words"][0]["similar"]], [50, 100])
        self.assertFalse(report["words"][1]["found"])
        result = self.run_cli("inspect_word2vec.py", "--vectors", path, "--words", "查询", "--min-count", 1000)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(json.loads(result.stdout)["words"][0]["similar"], [])
        result = self.run_cli("inspect_word2vec.py", "--vectors", path, "--words", "查询", "--topn", 1)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(json.loads(result.stdout)["words"][0]["similar"][0]["word"], "罕见词")

    def test_inspection_requires_counts_only_when_filtering(self):
        from gensim.models import KeyedVectors
        import numpy as np

        vectors = KeyedVectors(vector_size=2)
        vectors.add_vectors(["查询", "其他"], np.array([[1, 0], [0, 1]], dtype=np.float32))
        path = self.root / "vectors.kv"
        vectors.save(str(path))
        result = self.run_cli("inspect_word2vec.py", "--vectors", path, "--words", "查询", "--min-count", 50)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("缺少词频", result.stderr)
        result = self.run_cli("inspect_word2vec.py", "--vectors", path, "--words", "查询")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIsNone(json.loads(result.stdout)["words"][0]["count"])

    def test_uploaded_folder_runs_standalone_from_inside_directory(self):
        scripts = Path(__file__).resolve().parents[1]
        uploaded = self.root / "uploaded-training"
        shutil.copytree(scripts, uploaded, ignore=shutil.ignore_patterns("output", "__pycache__", ".venv*"))
        environment = {key: value for key, value in os.environ.items() if key != "PYTHONPATH"}
        result = subprocess.run(
            [sys.executable, "train_word2vec.py", "--output-dir", "output/run1", "--epochs", "1"],
            cwd=uploaded, env=environment, capture_output=True, text=True, timeout=30,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        report = json.loads((uploaded / "output/run1/training_report.json").read_text())
        self.assertEqual(report["texts"], 18)
        result = subprocess.run(
            [sys.executable, "inspect_word2vec.py", "--vectors", "output/run1/vectors.kv", "--words", "模型"],
            cwd=uploaded, env=environment, capture_output=True, text=True, timeout=30,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertTrue(json.loads(result.stdout)["words"][0]["found"])


if __name__ == "__main__":
    unittest.main()
