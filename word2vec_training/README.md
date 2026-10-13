# Word2Vec 训练与检索评估

使用 Gensim 从本地语料训练中文/英文静态词向量，支持 Skip-gram、CBOW、相似词查询、平均词向量检索评估，以及与本地 BGE 基座的同样本对照。整个目录可独立使用，不依赖原工程的 `src/`、`tokenizer_training/` 或其他模块。

Word2Vec 训练和检索评估使用 CPU；可选的 BGE 对照支持 CPU/CUDA。仓库只包含源码、配置、测试和小型示例，真实语料、模型权重、虚拟环境及运行输出保存在本地。

## 目录

```text
word2vec_training/
├── train_word2vec.py       # 训练、模型保存和重载校验
├── inspect_word2vec.py     # 相似词与词频检查
├── evaluate_word2vec.py    # 检索指标、覆盖率和同文诊断
├── compare_bge.py          # 复用 Word2Vec 查询 ID 比较 BGE
├── review_comparison.py    # 从已保存报告导出共同失败与 Word2Vec 独有命中
├── corpus.py               # 文本读取、分词与文件哈希
├── evaluation.py           # 数据校验、向量编码和分块排名
├── config.json             # 小样本试跑配置
├── requirements.txt        # Word2Vec 依赖
├── requirements-bge.txt    # 可选 BGE 依赖
├── data/sample.txt         # 18 条示例训练文本
├── examples/               # JSONL 训练/检索格式示例
├── docs/                   # AutoDL 操作与评估说明
└── tests/                  # 离线单元与 CLI 测试
```

## 1. 安装与快速试跑

推荐 Python 3.11，脚本支持 Python 3.9+。以下运行命令均在 **`word2vec_training` 目录内**执行。首次从仓库根目录进入：

```bash
cd word2vec_training
python3 -m venv .venv-word2vec
.venv-word2vec/bin/python -m pip install -r requirements.txt

PYTHONHASHSEED=42 .venv-word2vec/bin/python train_word2vec.py

.venv-word2vec/bin/python inspect_word2vec.py \
  --vectors output/word2vec-demo/vectors.kv \
  --words 模型 学习 苹果 --topn 5
```

默认读取本目录的 `config.json`，使用自带语料训练到 `output/word2vec-demo`。这个小模型用于验证流程，不代表真实业务效果。再次运行时用新的 `--output-dir`，脚本拒绝覆盖已有结果。

配置文件里的语料和输出相对路径以配置文件所在目录为基准；命令行传入的相对路径以当前工作目录为基准。Windows 可将 `.venv-word2vec/bin/python` 替换为 `.venv-word2vec\Scripts\python.exe`。

## 2. 使用真实训练语料

### TXT

UTF-8 文本，每行一段话。先将自己的训练语料放到 `data/train.txt`，然后运行：

```bash
PYTHONHASHSEED=42 .venv-word2vec/bin/python train_word2vec.py \
  --corpus data/train.txt \
  --vector-size 200 --window 5 --min-count 5 \
  --epochs 10 --sample 0.001 --workers 4 \
  --output-dir output/word2vec-real
```

`data/train.txt` 需要自行准备，不是仓库自带文件。也可给 `--corpus` 传入真实文件的绝对路径或多个文件路径。中文默认使用 Jieba 精确分词（HMM 关闭），过滤空白及纯标点；含中文、字母或数字的词元保留。已经用空格分好词的语料使用 `--tokenizer whitespace`。

### JSONL

每行一个 JSON 对象，只读取 `--fields` 指定的字符串或字符串列表字段：

```json
{"query":"模型如何学习？","pos":["模型从训练数据中学习规律。"],"neg":["苹果和香蕉都是水果。"]}
```

准备好自己的 `data/train.jsonl` 后：

```bash
PYTHONHASHSEED=42 .venv-word2vec/bin/python train_word2vec.py \
  --corpus data/train.jsonl --fields query pos neg \
  --vector-size 200 --window 5 --min-count 5 \
  --epochs 10 --sample 0.001 --workers 4 \
  --output-dir output/word2vec-jsonl
```

`examples/train.jsonl` 提供格式示例。`pos`、`neg` 在 Word2Vec 训练中都只是文本，脚本不使用其相关性标签；参数 `--negative` 表示词预测负采样数。验证集和测试集应留作评估，不加入本轮训练。

### 常用参数

| 参数 | 含义 |
| --- | --- |
| `--algorithm skipgram/cbow` | 训练算法，示例配置使用 Skip-gram |
| `--vector-size` / `--window` | 向量维度 / 上下文窗口 |
| `--min-count` | 训练词表的最低出现次数 |
| `--epochs` / `--workers` | 训练轮数 / CPU 训练线程数 |
| `--negative` | 负采样数，本实现不使用层次 softmax |
| `--sample` | 高频词下采样阈值；0 关闭下采样 |
| `--lowercase` / `--no-lowercase` | 是否转小写；默认配置开启 |
| `--seed` / `--max-texts` | 随机种子 / 最多读取文本数，0 文本数表示不限 |
| `--config` | 自定义 JSON 配置文件，命令行参数覆盖配置值 |

默认配置专用于小样本：100 维、`min_count=1`、30 轮、`sample=0`、单线程。正式训练应显式设置合适参数。固定 `PYTHONHASHSEED`、`--seed` 和 `--workers 1` 有助于复现；多线程及不同运行环境可能带来差异。

### 输出

- `word2vec.model`：完整 Gensim 模型，可加载后继续训练。
- `vectors.kv`：相似词查询及检索评估使用的 KeyedVectors。
- `vectors.txt`：通用文本格式词向量。
- `training_report.json`：训练参数、词表/语料统计、输入 SHA-256 和依赖版本。

迁移模型时保留整个输出目录，包括可能生成的 `.npy` 附件和训练报告。训练会先分词到临时文件，各轮复用同一份分词结果；成功后发布完整结果，失败时清理临时文件。

## 3. 检查相似词

```bash
.venv-word2vec/bin/python inspect_word2vec.py \
  --vectors output/word2vec-real/vectors.kv \
  --words 模型 学习 苹果 --topn 10 --min-count 50
```

这里的 `--min-count 50` 仅筛选相似词候选，并在全部合格候选中重新排名；不修改模型、不限制查询词本身，也不影响后续句子检索。结果包含查询词与候选词频。查询词须与训练的分词及大小写一致，词表外词返回 `found: false`。

## 4. 检索评估

方法为：沿用训练分词/小写规则 → 平均词表内的原始词向量 → L2 归一化 → 余弦相似度。重复词保留词次权重，OOV 词跳过，使用完整词表。它评估的是平均词向量检索方案，不等于单词近邻质量。

### 先运行自带的小型示例

完成快速训练后即可执行：

```bash
.venv-word2vec/bin/python evaluate_word2vec.py \
  --vectors output/word2vec-demo/vectors.kv \
  --corpus examples/corpus.jsonl --queries examples/queries.jsonl \
  --output-dir output/eval-demo
```

示例只有 4 篇文档和 3 个查询，与训练示例有内容重叠，仅用于检查输入输出流程。

### 正式数据格式

候选库 `corpus.jsonl`：

```json
{"id":"d1","text":"文档或段落内容"}
```

查询 `queries.jsonl`：

```json
{"id":"q1","text":"查询内容","relevant_ids":["d1"]}
```

每个查询至少有一个有效正例 ID，支持多正例。不同 ID 的同文候选仍保留。将文件准备到 `data/eval/` 后：

```bash
.venv-word2vec/bin/python -u evaluate_word2vec.py \
  --vectors output/word2vec-real/vectors.kv \
  --corpus data/eval/corpus.jsonl --queries data/eval/queries.jsonl \
  --limit-queries 1000 --seed 42 \
  --output-dir output/eval-real-1000
```

也可用 `--eval-data data/eval.jsonl` 读取 `query/pos/neg` 格式：同一 query 的正例合并，整个文件的正负例文本去重并集组成候选库。两种输入模式不能混用。`--limit-queries 0` 表示全量；抽样只减少查询数，始终保留完整候选库。

默认从词向量旁的 `training_report.json` 读取预处理设置。缺少报告时须显式指定 `--tokenizer jieba --lowercase`（或 `--no-lowercase`）。默认查询批次 64、候选块 20000，可用 `--query-batch-size`、`--corpus-chunk-size` 调节打分内存；文本和全部句向量仍驻留内存。

### 指标与报告

| 文件 | 内容 |
| --- | --- |
| `summary.json` | Hit/Recall@1/5/10、MRR@10、NDCG@10、词次覆盖率和零向量计数 |
| `results.jsonl` | 每个查询的前十名、正例、分数及指标 |
| `review.jsonl` | 前 30 个首位未命中案例，供定位问题 |
| `audit.json` | 数据/模型哈希、预处理、随机种子及版本 |
| `report.md` | 可直接阅读的指标说明 |

Hit@K 是至少命中一个标注正例的查询比例；Recall@K 是逐查询召回正例比例的平均值。零向量查询计为未命中，零向量候选不排名，但正例分母不变。同分按候选文件顺序排名。覆盖率不等于准确率，复核文件也不是随机人工准确率样本。

### 同文候选诊断

加 `--diagnose-self-matches` 并指定新输出目录，可额外计算逐查询排除“完全同文、但未标为正例”候选后的指标。保留已有正例，重新检索剩余完整候选库。原始指标不变，对照保存到 `self_match_diagnostic`、`diagnostic_results.jsonl`、`diagnostic_review.jsonl`。

`queries_with_unlabelled_self_match` 表示存在这类候选的查询数，`raw_top1_unlabelled_self_match_queries` 才是实际被其占据第一名的查询数。诊断提升来自候选范围变化，不能解释为模型训练提升，也不自动修改标签。详见[评估口径与已有实验记录](docs/evaluation.md)。

## 5. 与本地 BGE 基座对照（可选）

使用已有包含 torch、transformers、numpy 的 BGE 环境，或另建环境：

```bash
python3 -m venv .venv-bge
.venv-bge/bin/python -m pip install -r requirements-bge.txt
```

将已有 BGE 基座完整目录放到 `models/bge-base-zh-v1.5/`（需包含 `config.json`、tokenizer 和权重），下面用小型检索示例验证 CPU 对照流程：

```bash
.venv-bge/bin/python -u compare_bge.py \
  --baseline-report output/eval-demo \
  --model models/bge-base-zh-v1.5 \
  --corpus examples/corpus.jsonl --queries examples/queries.jsonl \
  --device cpu --batch-size 32 \
  --output-dir output/compare-demo
```

正式对照时替换为已完成的 Word2Vec 原始报告及对应原始候选/查询文件。入口按报告中的 ID 和顺序复用查询，校验文件哈希、文本、标签及指标，不重新抽样；输入只支持 `--corpus/--queries` 模式的原始报告，不依赖微调后的 `best_model`。

BGE 使用 CLS + L2、FP32，查询指令默认 `为这个句子生成表示以用于检索相关文章：`，查询/段落最大长度为 64/256 tokenizer tokens，含特殊 token 和查询指令。可用 `--query-instruction`、`--query-max-len`、`--passage-max-len` 修改；编码设置由本次参数指定，不自动读取 `embedding_config.json`。有可用 CUDA 时可改为 `--device cuda`，无 GPU 时显式使用 CPU，十万篇候选的 CPU 编码耗时较长。

输出包括 `comparison.json`（双方指标、差值和独有命中计数）、`selected_queries.jsonl`、`bge_results.jsonl`、`review.jsonl`、`audit.json` 和 `report.md`。Word2Vec 使用全文平均，BGE 按自身 tokenizer 截断，因此比较的是两套完整检索方案，报告会记录编码差异。BGE 权重不由本入口下载。

### 对照完成后复核失败案例

```bash
.venv-word2vec/bin/python review_comparison.py \
  --baseline-report output/eval-demo \
  --comparison-report output/compare-demo \
  --output-dir output/review-demo
```

读取双方已有报告，校验哈希、查询顺序、标签和可重算的指标，导出**全部共同前十未命中**及**全部 Word2Vec 独有前十命中**。无需重新编码、加载模型或访问原候选库；需要 NumPy，可直接使用 Word2Vec 环境。

`review.md` 展示查询、全部标注正例和双方完整 Top-10；`review.jsonl` 保留机器可读内容及待填写的 `manual_review` 字段。它不自动推断未命中原因、不修改标签，也不把未标正例当作已确认的负例。`summary.json` 记录导出数量和 BGE 编码设置，`audit.json` 记录输入报告哈希及校验范围。

真实 DuRetrieval 的案例复核及全量 2,000 查询 CPU 命令见 [下一步评估](docs/next-evaluation.md)。

## 6. 测试

**从仓库根目录执行**，解释器路径对应前面在模块内建立的环境：

```bash
word2vec_training/.venv-word2vec/bin/python -m unittest discover \
  -s word2vec_training/tests -v

word2vec_training/.venv-bge/bin/python -m unittest \
  word2vec_training.tests.test_bge_comparison -v
```

Word2Vec 环境运行全部核心测试；未装 BGE 依赖时自动跳过两项实际编码测试。BGE 专项测试使用本地随机小模型，验证 CPU 编码、查询对齐及独立目录 CLI，无需下载权重。随机小模型的测试指标不代表 BGE 基座效果。

## 常见问题

- **Python 路径不存在**：先在运行机器创建虚拟环境；环境目录不随 Git 上传。
- **语料路径不存在**：`data/train.txt` 等真实数据文件需要自行准备，示例试跑使用 `data/sample.txt`。
- **有效词表不足**：小语料可降低 `--min-count`，或先按默认示例配置验证。
- **CUDA 不可用**：CPU 对照使用 `--device cpu`；GPU 运行需要当前解释器的 PyTorch 能看到 GPU。
- **输出目录已存在**：换一个新的输出目录，脚本不会覆盖之前结果。
- **标点不同却相似度为 1**：纯标点会被分词过滤；相同词元或相同词袋可能得到相同平均向量。
- **评估数据与训练相同**：入口拒绝查询文件与训练输入完全同哈希；不同文件之间的文本重复仍需单独审计。

服务器现有目录的完整命令见 [AutoDL 操作说明](docs/autodl.md)。
