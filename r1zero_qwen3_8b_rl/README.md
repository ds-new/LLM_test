# R1-Zero 数学推理 RL（Qwen3-8B-Base）

不经过任何 SFT，直接在 **Qwen3-8B-Base** 上用规则奖励做强化学习（R1-Zero 路线），看纯 RL 能把数学推理推到多远，并用多智能体流水线（Claude 做决策智能体）在 AutoDL 的 **4 × A800 80GB PCIe** 上无人值守地连续训练、评测、判定和接续。

- 框架：verl 0.6.1（训练用 Megatron，生成用 SGLang 0.5.5，同卡分时复用）
- 算法：GRPO + DAPO 的几项改动（clip-higher、token 级平均、超长软惩罚、动态采样），不加 KL、不要参考模型、不要 critic、不要奖励模型
- 奖励：只看最终答案的规则判分（答对 +1，否则 −1），审计后修正为 v2（[ma/reward_v2.py](ma/reward_v2.py)）
- 每个阶段结束都按**事先定好的标准**做同题配对 bootstrap，给出"显著提升 / 无显著差异 / 显著下降"，再自动接续下一阶段

## 结果一览

完整评测（AMC23 / AIME 每题采样 16 次取平均，MATH-500 全部 500 题各 1 次；temperature 0.6、top_p 0.95；判分 v2）：

| 模型 | MATH-500 | AMC23 | AIME24 | AIME25 | AIME 2026（留出） |
|---|---|---|---|---|---|
| step 0（基座） | 50.6 | 28.0 | 4.6 | 4.4 | – |
| step 25（第一轮） | 75.2 | 54.7 | 14.4 | 15.0 | – |
| step 46（第二轮最佳） | 84.8 | 62.0 | 20.8 | 17.3 | 15.1 |
| step 76（第二轮终点） | 86.0 | 65.0 | 23.8 | 18.5 | 14.0 |
| step 96（阶段 A 终点） | 85.2 | 66.4 | 21.2 | 19.2 | 15.1 |
| step 116（阶段 B 终点） | 86.2 | 68.4 | 22.1 | 22.1 | 18.1 |
| step 134（阶段 C 评选最佳） | 85.6 | **72.8** | **27.3** | 21.9 | 16.6 |
| step 136（阶段 C 终点） | **86.0** | 71.6 | 25.0 | 20.8 | **19.6** |

AIME 2026 是训练和选模型都没碰过的留出集（29 题），用来检查有没有对原评测集过拟合。

纯 RL 把基座的 MATH-500 从 50.6% 提到 86%、AIME24 从 4.6% 提到约 25%，AIME 2026 留出集约 17–20%。第三轮从 step 76 到 136 的累计提升是显著的：四个原评测集等权平均 **+2.53 个百分点，95% 区间 [+0.72, +4.33]**，AIME 2026 **+5.6 [+1.7, +10.1]**。阶段 D（边界题）的结论见 [results/r1zero_qwen3_8b_rl_v1/](../results/r1zero_qwen3_8b_rl_v1/)（STAGE_frontier_RESULT.md，生成后随仓库更新）。

和"能直接用"的目标（MATH-500 ≥ 90%、AIME24/25 ≥ 40%）相比，纯 RL 在 8B 基座上已经接近平台期，所以下一步是蒸馏 SFT（见文末）。

## 做了什么

| 轮次 / 阶段 | 步数 | 改动 | 结论（同题配对 bootstrap，四集等权平均，95% 区间） |
|---|---|---|---|
| 第一轮 | 0 → 49 | GRPO 基线，回答上限 4K（后期截断率升到约 11%） | 基座 → step 25：MATH-500 50.6 → 75.2 |
| 第二轮 | 40 → 76 | 回答上限放到 8K + DAPO 超长软惩罚；每 2 步存盘、每 4 步验证 | 第 50 步存盘时数据盘写满崩溃、第 56 步存盘损坏，回滚到 46 续训；提前停在 76 |
| 审计 | – | 判分器审计（漏判）、训练/评测题泄露检查 | 判分改为 v2；训练集再去掉 6 道泄露题；加 AIME 2026 留出集 |
| 第三轮 A | 76 → 96 | 每批数据更新 8 次（原 1 次） | −0.32 [−1.84, +1.14]，无显著差异 |
| 第三轮 B | 96 → 116 | 加 DAPO 动态采样 | **+1.70 [+0.09, +3.32]，显著提升** |
| 第三轮 C | 116 → 136 | 学习率 1e-6 → 2e-6（保留动态采样） | +1.15 [−0.42, +2.75]，无显著差异 |
| 第三轮 D | 136 → 156 | 只练"边界题"（基座能做对一部分或全错的 1,055 题），n=32 | 见 results（自动生成） |

详细配置、每步曲线、判定过程见 [results/r1zero_qwen3_8b_rl_v1/](../results/r1zero_qwen3_8b_rl_v1/)。

## 训练配置（第三轮 D 的实际命令见 [run/command_r1zero_qwen3_8b_dapo.sh](run/command_r1zero_qwen3_8b_dapo.sh)）

| 项 | 设置 | 原因 |
|---|---|---|
| 数据 | DAPO-Math-17k 去重（179 万行 → 16,054 题）去掉 121 道与评测集重合的题 + X-R1-750；D 阶段换成 1,055 道边界题 | 去重防止同题刷分；去污染保证评测可信 |
| 提示 | "Solve the following math problem step by step. The last line of your response should be of the form Answer: $Answer …"，套 chat 模板 | 和 DAPO 一致，便于规则判分 |
| 每步 | 前三个阶段 64 题 × 16 条回答；D 阶段 32 题 × 32 条 | 边界题通过率低，组大一点才有足够的非零优势 |
| 长度 | 提示 ≤ 1024，回答 ≤ 8192；超过 6144 token 线性扣分到 −1 | 防止无限变长，同时不一刀切截断 |
| 优化 | lr 1e-6（C 为 2e-6），预热 3 步，AdamW、权重衰减 0.1；每批更新 8 次（mini batch = 1/8） | – |
| PPO 裁剪 | clip-higher：下界 0.2、上界 0.28；负优势双裁剪 c = 10；token 级平均 | DAPO 的做法，减缓熵塌缩 |
| 正则 | 不加 KL、不加熵奖励 | R1-Zero / DAPO 设定 |
| 动态采样 | 一组 32 条全对或全错就丢掉重采，最多 6 批 | 全对/全错的组优势为 0，不贡献梯度 |
| 并行 | 训练：Megatron TP=2 × DP=2、序列并行、全量重计算、参数/梯度/优化器在生成时卸到 CPU；生成：4 个 SGLang（TP=1），显存比例 0.65 | 见 [docs/02_memory_and_throughput.md](docs/02_memory_and_throughput.md) |
| 存盘 | 每 2 步存一次（只存模型权重，16 GB），保留最新 + 评选最佳 | 数据盘只有 50 GB |
| 验证 | 每 4 步在 400 条评选集上单次采样，选最佳存盘；阶段末做完整评测 | – |

## 多智能体流水线

[ma/](ma/) 里是流水线本身：

| 智能体 | 文件 | 职责 |
|---|---|---|
| 环境 | [env_agent.py](ma/env_agent.py) | 检查 GPU、驱动、依赖版本、磁盘 |
| 数据 | [data_agent.py](ma/data_agent.py)、[build_round3_data.py](ma/build_round3_data.py)、[build_frontier.py](ma/build_frontier.py) | 下载、去重、去污染、构造评选集和边界题集 |
| 框架 | [framework_agent.py](ma/framework_agent.py) | 生成 verl 命令；显存"档位"表（OOM 时换下一档） |
| 训练 | [trainer_agent.py](ma/trainer_agent.py) | 启动、盯日志、把崩溃分类（OOM / NCCL / 存盘损坏 …） |
| 评测 | [eval_agent.py](ma/eval_agent.py) | 评选打分、维护最佳存盘、完整评测 |
| 分析 | [analysis_agent.py](ma/analysis_agent.py) | 曲线、质量检查（中英混杂、重复、答案格式） |
| 决策 | [loop3.py](ma/loop3.py) + Claude | 事先定好的规则处理常规情况；规则覆盖不了的（OOM 换档、阶段结论之后怎么走、删什么）停下来写 WAITING_CLAUDE.json，由 Claude 判断 |

配套脚本：[watchdog3.sh](watchdog3.sh)（看门狗：训练停了 30 分钟没人修就关机；REPAIRING 文件 20 分钟内有更新表示正在修复）、[stage_watch.py](ma/stage_watch.py)（阶段结束后自动做配对比较、写结论、按事先定的规则接续下一阶段）、[check_r3.py](ma/check_r3.py)（巡检）、[quality.py](ma/quality.py)（生成质量）。Claude 每 20–25 分钟巡检一次，判断标准写在 [docs/03_ops_oom_monitoring.md](docs/03_ops_oom_monitoring.md)。

## 经验

1. **动态采样是唯一显著有效的改动。** 每批更新次数、学习率翻倍都没有显著差异；动态采样让每一步的梯度都来自"有对有错"的组。
2. **单次采样的评选分噪声很大。** D 阶段评选分 0.66 → 0.65 → 0.63 → 0.6725，中间那次"下降"在配对检验下不显著，下一次就反弹成新高。不能凭一两个评选点提前停训；结论只看阶段末的完整评测 + 配对 bootstrap。
3. **判分器要审计。** v1 判分漏判了大量 markdown/LaTeX 包装的正确答案；修正后各阶段的最佳存盘排序有变化。训练后期模型逐渐改用 \boxed 收尾（"没写 Answer 行"从 7% 升到约 50%），判分器必须两种格式都认。
4. **显存 OOM 要看清是谁占的。** nvidia-smi 里 SGLang 的数字包含 PyTorch 缓存（OOM 前会先释放）；真正挤掉空间的是训练进程在生成阶段残留的显存。详见 [docs/03_ops_oom_monitoring.md](docs/03_ops_oom_monitoring.md)。
5. **每次续训学习率都会重新预热 3 步。** 中途重启有隐性成本，能不重启就不重启。
6. **磁盘是硬约束。** 只存模型权重（不存优化器状态）、只留最新 + 最佳、存盘前先校验完整性再删旧的。

## 下一步：蒸馏 SFT（E）→ 在 SFT 模型上做 RL（F）

[sft/](sft/) 是第 E 阶段：用 [open-r1/OpenR1-Math-220k](https://huggingface.co/datasets/open-r1/OpenR1-Math-220k)（Apache-2.0，DeepSeek-R1 生成的推理过程）对 Qwen3-8B-Base 做全参数 SFT。

- 数据（[prep_openr1.py](sft/prep_openr1.py)）：每题只取一条判对、推理完整、最短的生成；去掉含中文的；和全部评测题做词级 13-gram 去污染（删 493 题）；套 Qwen3 chat 模板后 ≤ 16,000 token；得到 62,848 条训练 + 256 条验证，约 3.3 亿 token
- 训练（[run_sft.sh](sft/run_sft.sh)）：verl SFT trainer（FSDP2，按 token 数动态拼批），1 个 epoch，全局批 64 条，lr 2e-5、cosine、预热 3%，最长 16,384 token；存盘时把权重转成 bf16（[run_engine_sft.py](sft/run_engine_sft.py)）
- 评测（[eval_e.py](sft/eval_e.py)）：和 R1-Zero 同一套评测题、同一个判分函数，和 R1-Zero 的最终模型做同题配对 bootstrap
- 总控（[start_e.sh](sft/start_e.sh)）：冒烟测试 → 试跑评测 → 正式训练 → 完整评测 → 写结论 → 关机

## 目录

| 路径 | 内容 |
|---|---|
| [ma/](ma/) | 多智能体流水线、判分器、审计和统计脚本 |
| 根目录的 .sh、[run/](run/) | 启动（run_loop3.sh）、看门狗（watchdog3.sh）、阶段接续、下载脚本；实际训练命令和对 verl/SGLang 的补丁（run/） |
| [sft/](sft/) | 第 E 阶段（蒸馏 SFT）的数据准备、训练、评测、总控 |
| [docs/](docs/) | 原理和工程笔记（见下） |
| [../results/r1zero_qwen3_8b_rl_v1/](../results/r1zero_qwen3_8b_rl_v1/) | 报告、阶段结论、每步训练指标、决策日志、曲线、每题完整评测结果 |

文档：

- [docs/01_grpo_theory_to_code.md](docs/01_grpo_theory_to_code.md)：GRPO / DAPO 的公式在 verl 代码里分别对应哪里，一步训练的完整流程
- [docs/02_memory_and_throughput.md](docs/02_memory_and_throughput.md)：显存账（SGLang + Megatron 同卡分时）、为什么 8B 要 4 张 80GB、MFU
- [docs/03_ops_oom_monitoring.md](docs/03_ops_oom_monitoring.md)：OOM 排查实录、看门狗和巡检的判断标准、续训和磁盘的坑
- [docs/04_eval_and_stats.md](docs/04_eval_and_stats.md)：评测口径、评选集噪声、预注册的配对 bootstrap 判定、判分器审计和去污染

## 复现

环境（AutoDL 镜像 + pip）：torch 2.8.0+cu128、transformers 4.57.1、verl 0.6.1、sglang 0.5.5、megatron-core 0.13.1、transformer-engine 2.6.0、mbridge 0.15.1、flash-attn 2.8.1（完整列表 [pip_versions.txt](pip_versions.txt)）。代码默认放在 /root/autodl-tmp/r1zero（数据盘）。

    cd /root/autodl-tmp/r1zero
    bash download.sh                                   # 模型（ModelScope）和数据（HF 镜像）
    export PYTHONPATH=$PWD
    R1Z_STAGE=frontier R1Z_EXTRA_STEPS=20 setsid nohup bash run_loop3.sh > /dev/null 2>&1 < /dev/null &
    setsid nohup bash watchdog3.sh > /dev/null 2>&1 < /dev/null &
    python ma/check_r3.py 6                            # 巡检

蒸馏 SFT：

    python sft/prep_openr1.py
    bash sft/run_sft.sh smoke engine                   # 冒烟 3 步
    bash sft/run_sft.sh full engine
    bash sft/run_eval.sh <HF 模型目录> sft/eval_full 0 16384

模型权重、数据集和评测的逐条生成不随仓库发布。
