# LLM_test

本仓库提供两个独立的文本处理实验组件，均可从自带小语料开始运行：

| 组件 | 功能 | 环境 |
| --- | --- | --- |
| [Word2Vec](word2vec_training/README.md) | 训练静态词向量、查询相似词、检索评估、与 BGE 做同样本对照 | 推荐 Python 3.11，基础训练使用 CPU |
| [Tokenizer / BBPE](tokenizer_training/README.md) | 从零实现字节级 BPE，训练词表、编码解码、统计分词效率 | Python 3.9+，纯标准库、CPU 运行 |

## Word2Vec 快速开始

```bash
git clone https://github.com/ds-new/LLM_test.git
cd LLM_test/word2vec_training
python3 -m venv .venv-word2vec
.venv-word2vec/bin/python -m pip install -r requirements.txt
PYTHONHASHSEED=42 .venv-word2vec/bin/python train_word2vec.py
.venv-word2vec/bin/python inspect_word2vec.py \
  --vectors output/word2vec-demo/vectors.kv \
  --words 模型 学习 苹果 --topn 5
```

推荐 Python 3.11。默认使用自带小语料，Word2Vec 全程在 CPU 运行。BGE 对照是可选功能，需要另行准备本地 BGE 权重与 PyTorch 环境，也支持显式 CPU 运行。

## Tokenizer 快速开始

在已经克隆的 `LLM_test` 仓库根目录执行：

```bash
python3 -S tokenizer_training/train_tokenizer.py
python3 -S tokenizer_training/inspect_tokenizer.py \
  --tokenizer tokenizer_training/output/bbpe-scratch-demo \
  --text "知识库检索" "你好🙂"
```

如果刚执行了上面的 Word2Vec 命令，先 `cd ..` 返回仓库根目录。Tokenizer 无需安装运行依赖，默认使用 SQLite 磁盘训练后端。保存格式为本项目的 `bbpe.json`，用于配套的编码接口，不能直接替换已有 BGE 模型的分词器。各实验再次运行时请指定新的输出目录。

## 使用说明

- [Word2Vec README：安装、训练、查询、评估、BGE 对照和测试](word2vec_training/README.md)
- [AutoDL：沿用现有数据与模型的运行命令](word2vec_training/docs/autodl.md)
- [评估口径、实验记录与结果解释](word2vec_training/docs/evaluation.md)
- [训练及检索数据格式示例](word2vec_training/examples/)
- [Tokenizer README：BBPE 原理、训练、检查、AutoDL 命令和测试](tokenizer_training/README.md)
- [Tokenizer 的 JSONL 训练与检查示例](tokenizer_training/examples/)

模型、真实数据、运行输出和虚拟环境不随源码发布。示例仅验证流程，不能作为模型效果证明；真实训练和评估请使用独立准备的数据，并保留报告中的参数、哈希和版本信息。
