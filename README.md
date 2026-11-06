# LLM_test

本仓库提供五个独立的文本处理与语言模型实验组件：

| 组件 | 功能 | 环境 |
| --- | --- | --- |
| [Word2Vec](word2vec_training/README.md) | 训练静态词向量、查询相似词、检索评估、与 BGE 做同样本对照 | 推荐 Python 3.11，基础训练使用 CPU |
| [Tokenizer / BBPE](tokenizer_training/README.md) | 从零实现字节级 BPE，训练词表、编码解码、统计分词效率 | Python 3.9+，纯标准库、CPU 运行 |
| [Decoder 从零预训练](decoder_from_scratch/README.md) | 五智能体协作的预训练 loop：数据清洗 → 用上面的 BBPE 训练 32k 词表 → 结构选型 → 训练 → 评估；附一次 Mac 72 小时完整运行的结果 | PyTorch 2.3+，Apple Silicon（MPS）/ CUDA / CPU |
| [Qwen3-VL 几何看图解题 RL](qwen3vl_geo3k_grpo/README.md) | 五智能体协作的多模态 GRPO loop（verl + vLLM）：Qwen3-VL-4B 在 geometry3k 上训练、诊断、拒绝采样微调与多数投票；附一次 AutoDL 2×A800 完整运行的结果 | verl 0.8、vLLM 0.12、torch 2.9，NVIDIA GPU（80GB × 2） |
| [Weight-Space Programming 复现](weight_space_programming_repro/README.md) | 复现 MIT CSAIL 的"大模型直接读写神经网络权重"：多智能体 loop 用本机 Codex 调 gpt-6-astra 重跑 CPU 实验 1/2/3/5；在 AutoDL 上逐位重建作者对 Qwen3.5-0.8B 的权重编辑并重测 MATH-500 | CPU 部分：Mac + Codex CLI，Python 3.11+；GPU 部分：vLLM 0.33，24GB 以上显存 |

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

## Weight-Space Programming 复现：结果一览

复现博客 [*Sparks of Weight-Space Programming*](https://rikiyat.github.io/blog/weight-space-programming/)（作者公开的日志：[RikiyaT/lm_understand-logs](https://github.com/RikiyaT/lm_understand-logs)）。核心结论：Astra 不经训练、只改 **131,072 个权重**（第 15、19 层 MLP 的 `down_proj` 各 64 列），就让 Qwen3.5-0.8B 不再陷入 "Wait…" 循环：

| | 论文 | 本复现（AutoDL 4080 SUPER 32GB） |
| --- | --- | --- |
| 原模型 MATH-500 | 7.9% | 8.2% ± 0.6 |
| 编辑后（p50c）MATH-500 | 48.1% | **50.5% ± 1.1** |
| 回答写到 32K 上限的比例：原模型 → 编辑后 | 93.0% → 33.1% | 92.3% → 32.4% |

CPU 实验同样基本复现：只看权重认出游戏 4/4、认出形状 3/3；找出 induction head 5/6 完全一致；从零手写的 Breakout 策略 502 分，与作者网络在同一评测器下的 503 分一致，是 PPO 5 亿步的 20 倍。判断训练语料（4/6）和 Seaquest（34 分，作者 54 分）低于论文。

编辑细节、对照实验设想和差异分析见 [weight_space_programming_repro/README.md](weight_space_programming_repro/README.md)，原始记录在 [results/weight_space_programming_repro_v1/](results/weight_space_programming_repro_v1/)。

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

## Weight-Space Programming 复现快速开始

CPU 部分在 Mac 上运行，需要已登录的 Codex CLI（账号能用 `gpt-6-astra`）：

```bash
cd LLM_test/weight_space_programming_repro/cpu_loop
bash setup.sh                       # 克隆作者日志、建环境、自检（沙箱是否读不到答案键、是否断网）
./tests/replay_test.sh              # 离线回放：验证 prompt 与作者逐字节一致、评分复现论文，不花额度
.venv/bin/python run.py             # 实验 1/2/3/5 全部；可随时中断，重跑从断点继续
```

GPU 部分（实验 8 验证）在 AutoDL 上运行，跑完或异常停止 10 分钟后自动关机：

```bash
mkdir -p /root/autodl-tmp/wsp_gpu && cp -r LLM_test/weight_space_programming_repro/gpu_m6 /root/autodl-tmp/wsp_gpu/m6
cd /root/autodl-tmp/wsp_gpu/m6 && setsid nohup bash run_m6.sh > /dev/null 2>&1 < /dev/null &
```

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
- [Weight-Space Programming 复现 README：结果、多智能体 loop、AutoDL 验证流程、踩过的坑与下一步](weight_space_programming_repro/README.md)

模型、真实数据、运行输出和虚拟环境不随源码发布。例外是 [results/multiagent_pretrain_v1/](results/multiagent_pretrain_v1/)、[results/qwen3vl_geo3k_grpo_v1/](results/qwen3vl_geo3k_grpo_v1/) 和 [results/weight_space_programming_repro_v1/](results/weight_space_programming_repro_v1/)：第一个保存了一次 Decoder 预训练的报告、日志、评估记录和生成样例，第二个保存了一次 Qwen3-VL 几何解题 RL 的报告、评估汇总、每步训练指标和图表，第三个保存了 Weight-Space Programming 复现的每段对话、代码运行、评分与审计，以及 Qwen3.5-0.8B 编辑的评测结果（都不含数据集和模型权重；实验 5 中 Astra 写出的三个 MinAtar 小网络除外），用于复核实验结论。示例仅验证流程，不能作为模型效果证明；真实训练和评估请使用独立准备的数据，并保留报告中的参数、哈希和版本信息。
