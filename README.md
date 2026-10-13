# LLM_test

本仓库当前提供独立的 **Word2Vec 训练与检索评估工具**：从本地 TXT/JSONL 语料训练静态词向量，检查相似词，在完整候选库中计算检索指标，并与已有 BGE 基座做同样本对照。

## 快速开始

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

## 使用说明

- [完整 README：安装、训练、查询、评估、BGE 对照和测试](word2vec_training/README.md)
- [AutoDL：沿用现有数据与模型的运行命令](word2vec_training/docs/autodl.md)
- [评估口径、实验记录与结果解释](word2vec_training/docs/evaluation.md)
- [训练及检索数据格式示例](word2vec_training/examples/)

模型、真实数据、运行输出和虚拟环境不随源码发布。示例仅验证流程，不能作为模型效果证明；真实训练和评估请使用独立准备的数据，并保留报告中的参数、哈希和版本信息。
