# 从零训练 Tokenizer：纯 Python BBPE

本目录实现 Byte-level BPE（BBPE）的训练、编码、解码、保存加载和语料统计。算法使用 **Python 标准库**编写，支持 SQLite 磁盘后端和内存后端，不依赖 `tokenizers`、`transformers` 或 SentencePiece。**Python 3.9+，在 CPU 上运行，无需 GPU 或安装运行依赖。**

适合学习 BPE 算法、验证自定义分词方案，以及为从零训练的模型准备词表。保存格式为本项目的 `bbpe.json`，不能直接通过 `AutoTokenizer` 加载，也不能直接替换已有 BGE 模型的分词器。

## 快速开始

先取得仓库，下面所有 Shell 命令均在 **`LLM_test` 仓库根目录**执行：

```bash
git clone https://github.com/ds-new/LLM_test.git
cd LLM_test

python3 -S tokenizer_training/train_tokenizer.py

python3 -S tokenizer_training/inspect_tokenizer.py \
  --tokenizer tokenizer_training/output/bbpe-scratch-demo \
  --text "知识库检索" "你好🙂" "BBPE tokenizer training"
```

`-S` 禁止加载 site-packages，训练和检查依然可以运行。默认读取自带的 12 行小语料，目标词表为 512，使用磁盘后端。小语料仅用于验证流程，不能代表真实分词效果。

成功后出现“Tokenizer 已保存”，生成：

```text
tokenizer_training/output/bbpe-scratch-demo/
├── bbpe.json             # 加载所需的模型文件：预切分设置、特殊 ID 和合并规则
├── vocab.json            # 用于检查的词表：token ID 与字节十六进制
└── training_report.json  # 参数、语料 SHA-256、文本数、词表大小、耗时及资源快照
```

输出目录已存在时训练会报错，避免覆盖已有结果；再次运行请指定新的 `--output-dir`。如果已经进入 `tokenizer_training` 目录，可直接执行 `python3 -S train_tokenizer.py`，检查路径则写成 `--tokenizer output/bbpe-scratch-demo`。

## 目录结构

```text
tokenizer_training/
├── README.md
├── bbpe.py                  # 核心算法与 BBPETokenizer 接口
├── disk_trainer.py          # SQLite 磁盘训练后端
├── corpus.py                # TXT/JSONL 读取、校验与文件指纹
├── train_tokenizer.py       # 训练入口与产物校验
├── inspect_tokenizer.py     # 分词展示与独立语料统计
├── runtime_info.py          # RSS、Linux cgroup 内存快照
├── run_train.sh             # 保存日志与真实训练退出码
├── config.json              # 默认小语料配置
├── requirements.txt         # 运行依赖说明，无第三方运行依赖
├── requirements-dev.txt     # 可选测试依赖 pytest
├── data/sample.txt          # 可公开的小语料
├── examples/                # 可运行的 JSONL 训练与检查样例
└── tests/                   # 算法、磁盘后端和命令行回归测试
```

## 训练语料与配置

支持 UTF-8 的 `.txt` 和 `.jsonl` 文件，可在 `--corpus` 后指定多个文件。

- TXT 每行一段，去掉记录末尾的换行符，保留文本首尾空格。
- JSONL 每行是一个对象，使用 `--fields` 选择字符串或字符串列表字段，不会把 ID 或 JSON 结构本身送入训练。
- 空白行跳过；无效 JSON、字段类型错误或指定字段没有有效文本时，报错包含路径和行号。
- 重复文本保留，以实际出现次数参与训练。`--max-texts` 按提取出的字符串计数；一行中的 query 和每个 pos/neg 分别计数，`0` 表示读取全部。

JSONL 示例：

```json
{"id":"doc-001","text":"用于训练的文本。"}
{"id":"doc-002","text":["另一段文本。","还可以使用字符串列表。"]}
```

运行仓库自带的 query/pos/neg 示例：

```bash
python3 -S -u tokenizer_training/train_tokenizer.py \
  --corpus tokenizer_training/examples/train.jsonl \
  --fields query pos neg \
  --vocab-size 512 --min-frequency 1 \
  --output-dir tokenizer_training/output/bbpe-jsonl-demo
```

配置文件默认为脚本旁边的 [config.json](config.json)，可用 `--config` 更换。**配置文件中的路径相对配置文件所在目录，命令行路径相对当前工作目录。** CLI 参数覆盖对应配置。

| 配置 / CLI | 含义 |
| --- | --- |
| `algorithm` | 固定为 `bbpe`，无切换算法的 CLI 参数 |
| `backend` / `--backend` | `disk` 或 `memory`；训练脚本默认 `disk` |
| `corpus_files` / `--corpus` | 一个或多个 TXT/JSONL 文件 |
| `fields` / `--fields` | JSONL 文本字段，默认 `text` |
| `output_dir` / `--output-dir` | 新的产物目录，不覆盖已有目录 |
| `work_dir` / `--work-dir` | 磁盘后端工作目录；训练脚本默认用输出目录的父目录 |
| `vocab_size` / `--vocab-size` | 包含特殊词元的目标大小，最少 261；可合并的候选不足时实际词表更小 |
| `min_frequency` / `--min-frequency` | 相邻对合并所需的最小加权频次 |
| `lowercase` / `--lowercase` | 默认保留大小写；也支持 `--no-lowercase` |
| `model_max_length` / `--model-max-length` | 调用分词器时的默认长度，不截断训练语料；仅显式启用 `truncation=True` 才截断 |
| `max_texts` / `--max-texts` | 最多读取的文本数，`0` 表示全部 |

`config.json` 为演示设置了 `vocab_size=512`、`min_frequency=1`；真实语料需要显式调整参数。训练集与最终评测集应独立保存，不要将最终评测集加入词表训练。

## 在 AutoDL 使用已有训练集

以下路径对应已有工程的训练集，数据文件不包含在 GitHub 仓库中。先小规模检查资源占用和流程：

```bash
python3 -S -u tokenizer_training/train_tokenizer.py \
  --corpus /root/autodl-tmp/embedding_model_train/data/processed/train.jsonl \
  --fields query pos neg \
  --backend disk --vocab-size 2000 --min-frequency 2 --max-texts 10000 \
  --output-dir tokenizer_training/output/bbpe-scratch-small
```

全量训练使用包装脚本，保留完整日志和训练退出码：

```bash
bash tokenizer_training/run_train.sh \
  --corpus /root/autodl-tmp/embedding_model_train/data/processed/train.jsonl \
  --fields query pos neg \
  --backend disk --work-dir /root/autodl-tmp/bbpe-work \
  --vocab-size 32000 --min-frequency 2 --max-texts 0 \
  --output-dir tokenizer_training/output/bbpe-scratch-train
```

`run_train.sh` 会切换到 `tokenizer_training` 的父目录运行，参数中的相对路径也以该目录为准；日志保存在该目录的 `output/logs/`，可用 `PYTHON=/path/to/python3 bash tokenizer_training/run_train.sh ...` 指定解释器。训练每读取 100,000 条文本、每完成 100 次合并报告一次进度，并输出索引阶段、耗时和进程峰值 RSS。

磁盘后端将片段、权重、相邻对计数和倒排表放入 SQLite，每次分页读取最多 256 条记录、逐片段更新。SQLite 页面缓存为 8 MiB，禁用 mmap；**这不是进程总内存上限**。长片段、Python 对象、合并规则和系统页缓存仍会占用资源。该后端不会自动抽样或裁剪语料，但增加磁盘读写，训练时间取决于语料规模和磁盘性能。

## 检查分词与评估压缩效率

先用自带的独立检查样例验证流程：

```bash
python3 -S tokenizer_training/inspect_tokenizer.py \
  --tokenizer tokenizer_training/output/bbpe-jsonl-demo \
  --text "中文🙂" "a tokenizer example" \
  --corpus tokenizer_training/examples/eval.jsonl --fields text \
  > tokenizer_training/output/bbpe-jsonl-eval.json
```

真实训练完成后，在独立验证集上统计：

```bash
python3 -S tokenizer_training/inspect_tokenizer.py \
  --tokenizer tokenizer_training/output/bbpe-scratch-train \
  --corpus /root/autodl-tmp/embedding_model_train/data/processed/eval.jsonl \
  --fields query pos neg --max-texts 10000 \
  > tokenizer_training/output/bbpe-scratch-eval.json
```

检查报告包含每个示例的 token ID、`bytes:...` 十六进制表示、mask 和 `decoded`，以及语料的以下指标（不计自动添加的特殊词元）：

| 指标 | 用途 |
| --- | --- |
| `texts` / `characters` / `tokens_without_special` | 实际检查的文本、字符与词元数量 |
| `unknown_token_rate` | 未知词元比例；完整字节词表通常为零，不代表分词高效 |
| `tokens_per_text` | 平均每条文本占用的词元数量 |
| `tokens_per_character` | 每字符所需词元数，在相同语料上比较 |
| `utf8_bytes_per_token` | 每词元覆盖的 UTF-8 字节数，用于观察压缩程度 |

`is_fast=false` 是预期值，表示手写 Python 实现。比较不同词表时应固定验证集、字段、文本上限和大小写设置，并同时考虑词表大小、序列长度和后续模型效果；这些统计不能替代下游任务评估。启用小写化后，解码还原的是小写后的文本。

## Python 接口与保存格式

下面代码在仓库根目录运行，使用快速开始生成的模型：

```python
from tokenizer_training.bbpe import BBPETokenizer

model = BBPETokenizer.from_pretrained("tokenizer_training/output/bbpe-scratch-demo")
ids = model.encode("中文🙂", add_special_tokens=False)
assert model.decode(ids) == "中文🙂"

batch = model(["知识库检索", "文本向量"], padding=True, truncation=True, max_length=256)
pair = model("问题", "文档", truncation=True, max_length=256)

# 底层 API 默认使用内存后端；大语料可以显式传入 backend="disk" 和 work_dir。
tiny = BBPETokenizer.train(["abab", "abab", "aaaa"], vocab_size=263, min_frequency=1)
assert len(tiny) == 263
```

`__call__` 返回由 Python 列表组成的 `input_ids`、`token_type_ids`、`attention_mask`，支持单句、句对、批量输入、右侧 padding 和 longest-first 截断。不支持 `return_tensors`、offset mapping 或完整 Transformers API。

`encode` 不截断。`__call__` 截断后的字节 token 可能只保留某个 UTF-8 字符的一部分；这类不完整序列解码时默认使用替换字符，可传 `errors="strict"` 检查。`decode_bytes` 返回原始字节。

模型格式包含版本号、预切分版本、特殊 ID、大小写设置、最大长度和按训练顺序保存的合并规则。训练入口在临时目录保存后重新加载，核对示例编码结果，成功后才发布结果目录；报告还记录源文件 SHA-256。加载器拒绝无效格式、引用未来词元或特殊词元的规则，以及重复合并规则。

## 算法说明

1. **初始词表**：`[PAD]=0`、`[UNK]=1`、`[CLS]=2`、`[SEP]=3`、`[MASK]=4`；256 个字节映射为 `byte + 5`，初始大小共 261，新词元从 ID 261 开始。
2. **预切分**：按 Unicode 类别连续分段：字母/组合标记（含汉字）、数字、空白、其他字符。拼接片段可还原输入，合并不跨片段边界，空白也参与编码。默认保留大小写，不插入前导空格。这是自定义规则，与 GPT-2 或 Hugging Face ByteLevel 的规则和 token ID 不可直接互换。
3. **训练**：相同字节片段累计权重，统计加权相邻对频次；每轮选择最高频对，同频时按两个 ID 排序；从左到右非重叠替换，再更新受影响片段的统计。到达目标词表、频次低于阈值或候选耗尽即停止。
4. **编码**：按相同规则预切分，从字节 ID 开始，反复应用当前相邻对中训练顺序最早的合并规则，不重新统计词频，也不是最长字符串匹配。
5. **解码**：先拼接所有词元的字节串，再整体按 UTF-8 解码，避免分别解码部分中文字节。输入中的 `[CLS]` 等字面字符串按普通文本编码；特殊 ID 仅由模板或调用方显式插入。

例如 `abab` 两次、`aaaa` 一次：首轮 `(a,b)` 频次为 4，生成 ID 261；下一轮 `(a,a)` 频次为 3，生成 ID 262，`aaaa` 替换为 `[262,262]`。计数允许重叠位置，替换不重叠：`aaa` 中 `(a,a)` 的频次为 2，但替换结果是 `aa,a`。

内存后端使用 `Counter`、倒排集合和 `heapq`；磁盘后端使用 SQLite 存储和索引排序。两者执行相同的计数、平局与替换规则，测试用慢速全量重算参考算法核对结果。建议按 [bbpe.py](bbpe.py) → [disk_trainer.py](disk_trainer.py) → [train_tokenizer.py](train_tokenizer.py) 的顺序阅读。

## 运行测试

仅测试需要安装 pytest；训练、编码和检查无需安装依赖。在仓库根目录执行：

```bash
python3 -m venv .venv-tokenizer
.venv-tokenizer/bin/python -m pip install -r tokenizer_training/requirements-dev.txt
.venv-tokenizer/bin/python -m pytest tokenizer_training/tests -q
```

测试涵盖手算合并、权重与重叠位置、同频确定性、随机语料与参考算法对照、完整字节覆盖、Unicode/空白/特殊字符串还原、保存加载、非法规则、句对 padding/truncation、磁盘分页与临时文件清理、日志退出码，以及禁用 site-packages 的命令行端到端流程。

## 常见问题

**`No such file or directory`**：确认当前目录和实际语料路径。配置路径相对配置文件，CLI 路径相对工作目录；包装脚本会切换到组件的父目录。训练无需复用 Word2Vec 的虚拟环境，直接用 Python 3.9+ 即可。

**输出目录已存在**：为新实验更换 `--output-dir`。不要预先创建最终输出目录；训练入口在成功完成后创建它。

**出现 `Killed`，没有模型目录**：进程被外部终止，不能仅凭这一行断定 OOM。使用 `run_train.sh` 保留训练日志、真实退出码和前后 cgroup 计数。退出码 137 通常对应 SIGKILL；`memory.events` 的 `oom_kill` 增加说明该 cgroup 发生过 OOM kill，具体进程需结合系统日志。`free` 可能展示宿主机内存；训练后快照来自独立进程，训练进程峰值 RSS 应查看训练进度或报告。

**磁盘后端的工作文件**：放在工作目录的 `.bbpe-work-*` 内，正常完成或 Python 异常退出时清理；SIGKILL 后可能残留，确认相关进程停止后再清理。当前不支持断点续训。工作目录需要足够磁盘空间，避免使用容量小或基于内存的临时盘。

**何时运行检查**：训练出现“Tokenizer 已保存”且退出码为 0 后再检查；可使用 `&&` 连接训练和检查，避免失败后继续执行。

**已有 `tokenizer.json` / BGE 模型**：本组件加载的是自定义 `bbpe.json`。新词表的 token ID 与已有模型权重不匹配，不能直接替换其分词器；需要配套训练模型并调整相应输入流程。

仓库发布源码、测试和小型示例；真实训练数据、训练产物、日志、SQLite 工作文件与虚拟环境不上传。相关忽略规则见 [.gitignore](.gitignore)。
