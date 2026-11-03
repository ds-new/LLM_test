# LLM_test

本仓库提供四个独立的文本处理与语言模型实验组件：

| 组件 | 功能 | 环境 |
| --- | --- | --- |
| [Word2Vec](word2vec_training/README.md) | 训练静态词向量、查询相似词、检索评估、与 BGE 做同样本对照 | 推荐 Python 3.11，基础训练使用 CPU |
| [Tokenizer / BBPE](tokenizer_training/README.md) | 从零实现字节级 BPE，训练词表、编码解码、统计分词效率 | Python 3.9+，纯标准库、CPU 运行 |
| [Decoder 从零预训练](decoder_from_scratch/README.md) | 五智能体协作的预训练 loop：数据清洗 → 用上面的 BBPE 训练 32k 词表 → 结构选型 → 训练 → 评估；附一次 Mac 72 小时完整运行的结果 | PyTorch 2.3+，Apple Silicon（MPS）/ CUDA / CPU |
| [Qwen3-VL 几何看图解题 RL](qwen3vl_geo3k_grpo/README.md) | 五智能体协作的多模态 GRPO loop（verl + vLLM）：Qwen3-VL-4B 在 geometry3k 上训练、诊断、拒绝采样微调与多数投票；附一次 AutoDL 2×A800 完整运行的结果 | verl 0.8、vLLM 0.12、torch 2.9，NVIDIA GPU（80GB × 2） |

## Decoder 从零预训练：结果一览

在一台 Apple Silicon Mac 的 GPU 上用 72 小时，从零训练了一个 **63.5M 参数**的 Decoder-only 模型（RoPE、GQA、SwiGLU、Pre-RMSNorm），训练数据为 BAAI/IndustryCorpus 的 11 个中英文领域，共 **1.12B token**：

| 指标 | 结果 |
| --- | --- |
| 验证 loss / 困惑度 | 3.679 / 39.6 |
| 测试 loss | 3.635 |
| 吞吐 | 约 5,000 tok/s（MPS，bf16） |

![loss 曲线](results/figures/loss_curve.png)

完整说明（模型结构、数据、五个智能体的分工与决策规则、本次运行的决策记录、生成样例、经验与局限）见 [decoder_from_scratch/README.md](decoder_from_scratch/README.md)，运行的全部原始记录在 [results/multiagent_pretrain_v1/](results/multiagent_pretrain_v1/)。

## Qwen3-VL 几何看图解题：结果一览

在 AutoDL 的 2 × A800 80GB 上，用五智能体 loop 对 **Qwen3-VL-4B-Instruct** 在 [geometry3k](https://huggingface.co/datasets/hiyouga/geometry3k) 上做 GRPO，并做了诊断实验和两版拒绝采样微调：

| 方式 | test 准确率（601 题） |
| --- | --- |
| 基线（未训练） | 50.08% |
| GRPO 第 1 轮（最终模型） | 53.58%（逐题符号检验 z = 2.39，显著） |
| 第 2 轮 GRPO / 两版拒绝采样微调 | 未超过第 1 轮（单次回答停在 53–54%） |
| **最终模型 + 8 次采样多数投票** | **66.39%** |

![test 准确率](results/qwen3vl_geo3k_grpo_v1/figures/test_accuracy.png)

训练配置及原因、诊断结论（截断主要是看错图后的绕圈）、与公开参考曲线的对比和经验见 [qwen3vl_geo3k_grpo/README.md](qwen3vl_geo3k_grpo/README.md)，原始记录在 [results/qwen3vl_geo3k_grpo_v1/](results/qwen3vl_geo3k_grpo_v1/)。

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

## Decoder 快速开始

在仓库根目录执行（模型权重 `best.pt` 超过 GitHub 单文件限制，请从 Releases 下载后放到 `checkpoints/best.pt`）：

```bash
python3 -m venv .venv-decoder
.venv-decoder/bin/python -m pip install -r requirements.txt
.venv-decoder/bin/python -m decoder_from_scratch.infer \
  --checkpoint checkpoints/best.pt --device mps \
  --prompt "人工智能的发展" --method sample --top-p 0.9 --temperature 0.8 \
  --repetition-penalty 1.2 --no-repeat-ngram-size 4
```

提示词请用半角标点。Apple Silicon 用 `--device mps`，NVIDIA GPU 用 `--device cuda`。从头复现训练见 [decoder_from_scratch/README.md](decoder_from_scratch/README.md#复现训练)。

## Qwen3-VL 几何解题快速开始

在 GPU 实例上执行（需按 [requirements](qwen3vl_geo3k_grpo/requirements.txt) 准备 verl / vLLM 环境；模型权重不随仓库发布）：

```bash
cd LLM_test/qwen3vl_geo3k_grpo
export PYTHONPATH=$PWD VLLM_WORKER_MULTIPROC_METHOD=spawn GEO3K_PATCH_CONV3D=1
nohup bash multiagent/run_loop.sh > /dev/null 2>&1 &     # 完整多智能体 loop
python -m multiagent.loop status
MA_DRY_RUN=1 bash multiagent/run_loop.sh                  # 不需要 GPU 的干跑
```

测试模型、多数投票和拒绝采样微调的命令见 [qwen3vl_geo3k_grpo/README.md](qwen3vl_geo3k_grpo/README.md#快速开始)。

## 使用说明

- [Word2Vec README：安装、训练、查询、评估、BGE 对照和测试](word2vec_training/README.md)
- [AutoDL：沿用现有数据与模型的运行命令](word2vec_training/docs/autodl.md)
- [评估口径、实验记录与结果解释](word2vec_training/docs/evaluation.md)
- [训练及检索数据格式示例](word2vec_training/examples/)
- [Tokenizer README：BBPE 原理、训练、检查、AutoDL 命令和测试](tokenizer_training/README.md)
- [Tokenizer 的 JSONL 训练与检查示例](tokenizer_training/examples/)
- [Decoder README：模型结构、数据、多智能体 loop、评估结果与经验](decoder_from_scratch/README.md)
- [多智能体 loop 的阶段、决策规则与运维命令](decoder_from_scratch/MULTIAGENT_LOOP.md)
- [Qwen3-VL 几何解题 README：实验过程、训练配置与原因、诊断、多数投票与拒绝采样微调、五个智能体](qwen3vl_geo3k_grpo/README.md)

模型、真实数据、运行输出和虚拟环境不随源码发布。例外是 [results/multiagent_pretrain_v1/](results/multiagent_pretrain_v1/) 和 [results/qwen3vl_geo3k_grpo_v1/](results/qwen3vl_geo3k_grpo_v1/)：前者保存了一次 Decoder 预训练的报告、日志、评估记录和生成样例，后者保存了一次 Qwen3-VL 几何解题 RL 的报告、评估汇总、每步训练指标和图表（都不含数据和权重），用于复核实验结论。示例仅验证流程，不能作为模型效果证明；真实训练和评估请使用独立准备的数据，并保留报告中的参数、哈希和版本信息。
