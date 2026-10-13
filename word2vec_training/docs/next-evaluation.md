# 下一步：复核 25 条案例，再跑全量 CPU 对照

现有 DuRetrieval 同样本报告覆盖 1,000 / 2,000 个查询、100,001 篇候选。下一步先利用已保存排名检查 24 条共同失败和 1 条 Word2Vec 独有命中，再扩大到全部查询。全量结果尚未生成，不能用抽样分数代替。

## 1. 导出已有 1,000 查询中的全部重点案例

以下命令在现有 AutoDL 部署上执行；先把仓库的新脚本同步到该目录。若使用 Git 克隆的仓库，将 `cd` 改为实际的 `LLM_test/word2vec_training` 路径。

```bash
cd /root/autodl-tmp/word2vec_training
.venv-word2vec/bin/python review_comparison.py \
  --baseline-report output/eval-duretrieval-word2vec-1000 \
  --comparison-report output/compare-word2vec-bge-duretrieval-1000 \
  --output-dir output/review-duretrieval-1000
```

使用已有 Word2Vec 环境即可，无需 PyTorch/GPU、模型权重或候选库文件，不重新计算向量。工具会从已保存排名重算双方指标与配对统计，并核对原始报告哈希、查询 ID/顺序及正例标签。如果是上面这次实验，`summary.json` 应显示：

```json
{
  "case_counts": {"neither_hit_top10": 24, "word2vec_only_top10": 1},
  "exported_cases": 25
}
```

输出文件：

- `review.md`：可阅读的查询、全部正例和双方完整 Top-10，不截断文档正文。
- `review.jsonl`：同样的案例数据，附带 `manual_review.category` 和 `manual_review.notes`，初始值为 `unreviewed` 与空字符串。
- `summary.json`：实际导出数、查询数、候选数、原 BGE 编码设置和设备。
- `audit.json`：本次读取报告的 SHA-256 及校验范围。

两组案例全部导出，组内保持原查询顺序，不使用原 `compare_bge.py` 中每组最多 10 条的复核样本。复核不会更改原始指标或标签，也不会推断正例在前十以外的具体名次。

也可以把双方完整报告目录复制到本机的 `word2vec_training/output/`，再运行同一条命令。报告内记录的旧机器绝对路径不用修改；校验按本次指定目录中的文件内容哈希进行。需要保留：

```text
eval-duretrieval-word2vec-1000/
  summary.json  audit.json  results.jsonl
compare-word2vec-bge-duretrieval-1000/
  comparison.json  audit.json  bge_results.jsonl  selected_queries.jsonl
```

仅粘贴 `comparison.json` 汇总值不足以恢复案例正文和排名。

## 2. 人工复核顺序

逐条检查标注正例是否满足查询意图，再看 BGE 前十是否存在未标注但相关的文档，最后对照 Word2Vec 结果。建议在 `manual_review` 中记录分类、依据及相关文档 ID：

| 分类 | 何时记录 |
| --- | --- |
| `retrieval_miss` | 标注正例相关，但模型前十没有找回 |
| `possible_missing_label` | 前十存在疑似相关结果，但未被标为正例 |
| `possible_label_issue` | 标注正例本身似乎不符合查询意图 |
| `needs_truncation_check` | 怀疑正例中的关键内容被截断，尚待 tokenizer 核实 |
| `domain_term` | 领域术语、简称或专名可能影响理解，需要注明具体词及证据 |
| `other` | 其他原因或证据不足 |

字符长短不等于 token 长度，不能据此直接确认截断。需要使用原 BGE tokenizer、查询指令、`query_max_len` / `passage_max_len`，查看原文关键片段是否超出编码范围。对疑似漏标或错标先记录，不自动修改官方标签。此案例集用于排错，不是随机人工准确率样本。

## 3. 全量 Word2Vec：保留同一模型与候选库

仍在 AutoDL 的 `word2vec_training` 目录下执行。使用前一轮同一个 `vectors.kv`；如果它实际保存在其他目录，按原 Word2Vec `audit.json` 的 `model_files` 修改路径。

```bash
.venv-word2vec/bin/python -u evaluate_word2vec.py \
  --vectors output/word2vec-real/vectors.kv \
  --corpus /root/autodl-tmp/embedding_model_train/data/public_eval/duretrieval/corpus.jsonl \
  --queries /root/autodl-tmp/embedding_model_train/data/public_eval/duretrieval/queries.jsonl \
  --limit-queries 0 --seed 42 \
  --output-dir output/eval-duretrieval-word2vec-full
```

`--limit-queries 0` 表示读取文件中所有查询；对于当前数据，应打印 `评估 2000/2000 个查询`。候选库仍应为 100,001 篇。无需重新训练 Word2Vec。预处理沿用词向量目录中的训练报告；若上一轮手动指定过预处理参数，请按上一轮 `audit.json` 的 `preprocessing` 传入相同设置。

## 4. BGE 全量 CPU 对照

上一条评估成功后，再使用已有 BGE 环境的 `python` 执行。以下保持之前命令的查询指令及 64/256 长度；如果上一轮使用过自定义参数，先按其 `audit.json` 中 `bge_settings` 调整，避免同时改变实验条件。

```bash
python -u compare_bge.py \
  --baseline-report output/eval-duretrieval-word2vec-full \
  --model /root/autodl-tmp/embedding_model_train/models/BAAI/bge-base-zh-v1___5 \
  --corpus /root/autodl-tmp/embedding_model_train/data/public_eval/duretrieval/corpus.jsonl \
  --queries /root/autodl-tmp/embedding_model_train/data/public_eval/duretrieval/queries.jsonl \
  --device cpu --batch-size 32 \
  --query-instruction "为这个句子生成表示以用于检索相关文章：" \
  --query-max-len 64 --passage-max-len 256 \
  --output-dir output/compare-word2vec-bge-duretrieval-full
```

BGE 按全量 Word2Vec 报告复用全部查询 ID 与顺序。不要继续传入旧的 `...-1000` 基线，否则仍然只评估 1,000 条。当前脚本会重新编码完整候选库，不复用前一轮向量缓存。预计耗时应以实际机器进度为准。

## 5. 导出全量失败案例

```bash
.venv-word2vec/bin/python review_comparison.py \
  --baseline-report output/eval-duretrieval-word2vec-full \
  --comparison-report output/compare-word2vec-bge-duretrieval-full \
  --output-dir output/review-duretrieval-full
```

全量的失败数量由新报告计算，不能预设仍为 24 和 1。保留 `comparison.json`、双方 `audit.json` 和人工复核备注，用于决定是否需要增加段落长度、分块、微调或其他后续实验。所有脚本都拒绝覆盖已有结果；重复实验应换用新目录。
