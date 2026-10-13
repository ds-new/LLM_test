# AutoDL：沿用已有数据和模型

适用于 Word2Vec 位于 `/root/autodl-tmp/word2vec_training`，原工程数据位于 `/root/autodl-tmp/embedding_model_train` 的已有部署。若使用 Git 克隆后的目录，应将第一条 `cd` 改为实际的 `LLM_test/word2vec_training` 路径。

新机器先按 [README](../README.md) 创建 `.venv-word2vec` 并安装依赖。本页中的训练数据、公开评测集、已训练词向量和 BGE 权重需要已在服务器准备好，不由本仓库分发。

## 训练正式词向量

```bash
cd /root/autodl-tmp/word2vec_training
PYTHONHASHSEED=42 .venv-word2vec/bin/python train_word2vec.py \
  --corpus /root/autodl-tmp/embedding_model_train/data/processed/train.jsonl \
  --fields query pos neg \
  --vector-size 200 --window 5 --min-count 5 \
  --epochs 10 --sample 0.001 --workers 4 \
  --output-dir output/word2vec-real
```

已有 `output/word2vec-real` 时直接使用其中模型，无需再次训练。重新训练应指定新输出目录。

## 留出句对评估与同文诊断

```bash
.venv-word2vec/bin/python -u evaluate_word2vec.py \
  --vectors output/word2vec-real/vectors.kv \
  --eval-data /root/autodl-tmp/embedding_model_train/data/processed/eval.jsonl \
  --limit-queries 1000 --seed 42 \
  --diagnose-self-matches \
  --output-dir output/eval-heldout-selfcheck-1000
```

全量使用 `--limit-queries 0`，并换成新的输出目录。原始与诊断口径同时保留。

## DuRetrieval 独立候选库评估

```bash
.venv-word2vec/bin/python -u evaluate_word2vec.py \
  --vectors output/word2vec-real/vectors.kv \
  --corpus /root/autodl-tmp/embedding_model_train/data/public_eval/duretrieval/corpus.jsonl \
  --queries /root/autodl-tmp/embedding_model_train/data/public_eval/duretrieval/queries.jsonl \
  --limit-queries 1000 --seed 42 \
  --output-dir output/eval-duretrieval-word2vec-1000
```

这一报告将作为 BGE 对照的输入，保留 `summary.json`、`audit.json`、`results.jsonl` 及原始数据文件。

## BGE 同样本对照：CPU

使用已有 BGE/PyTorch 环境的 `python`。如果另建 `.venv-bge`，将下面的 `python` 换成 `.venv-bge/bin/python`。需要的依赖见 `requirements-bge.txt`。

```bash
python -u compare_bge.py \
  --baseline-report output/eval-duretrieval-word2vec-1000 \
  --model /root/autodl-tmp/embedding_model_train/models/BAAI/bge-base-zh-v1___5 \
  --corpus /root/autodl-tmp/embedding_model_train/data/public_eval/duretrieval/corpus.jsonl \
  --queries /root/autodl-tmp/embedding_model_train/data/public_eval/duretrieval/queries.jsonl \
  --device cpu --batch-size 32 \
  --output-dir output/compare-word2vec-bge-duretrieval-1000
```

模型目录使用既有部署的 `bge-base-zh-v1___5` 名称；其他部署请按实际路径填写。程序会打印编码和排名进度。十万篇段落的 CPU 编码耗时较长，当前入口中断后需要重新编码，尚不支持断点续跑。

CUDA 可用时可将设备改为 `--device cuda`；它与 Word2Vec 的 `--workers` 不是同一设置。若提示 CUDA 不可用，可以继续显式使用 CPU，也可先核对当前解释器与 GPU：

```bash
nvidia-smi
python -c "import sys, torch; print(sys.executable); print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

查看 `comparison.json` 的 `word2vec`、`bge`、`delta_bge_minus_word2vec` 与 `paired_outcomes`。结果目录已存在时应换名，避免覆盖已经完成的实验。
