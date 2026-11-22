# R1-Zero 数学推理 RL（Qwen3-8B-Base）

不经过任何 SFT，直接在 **Qwen3-8B-Base** 上用规则奖励做强化学习（R1-Zero 路线），看纯 RL 能把数学推理推到多远，并用多智能体流水线（Claude 做决策智能体）在 AutoDL 的 **4 × A800 80GB PCIe** 上无人值守地连续训练、评测、判定和接续。

- 框架：verl 0.6.1（训练用 Megatron，生成用 SGLang 0.5.5，同卡分时复用）
- 算法：GRPO + DAPO 的几项改动（clip-higher、token 级平均、超长软惩罚、动态采样），不加 KL、不要参考模型、不要 critic、不要奖励模型
- 奖励：只看最终答案的规则判分（答对 +1，否则 −1），审计后修正为 v2（[ma/reward_v2.py](ma/reward_v2.py)）
- 每个阶段结束都按**事先定好的标准**做同题配对 bootstrap，给出"显著提升 / 无显著差异 / 显著下降"，再自动接续下一阶段
- 完整的训练总结（时间线、三轮配置、各阶段结果、问题与修复、提速、后续计划）：[docs/00_training_summary.md](docs/00_training_summary.md)

> **项目已于 2026-11-22 收尾。** 最终模型是 F 第 20 步：先蒸馏 SFT（E），再在 SFT 模型上做 20 步 RL。它的四集平均是 57.0%，比纯 RL 的 step 156 高 4.3 个百分点（显著），但没有达到"可以用"的目标（MATH-500 ≥ 90%、AIME ≥ 40%）。结论见下文"结论"一节，完整过程见 [docs/00_training_summary.md](docs/00_training_summary.md)。

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
| step 134（阶段 C 评选最佳） | 85.6 | 72.8 | 27.3 | 21.9 | 16.6 |
| step 136（阶段 C 终点） | 86.0 | 71.6 | 25.0 | 20.8 | 19.6 |
| step 148（阶段 D 评选最佳） | 86.6 | 73.0 | 26.7 | 21.9 | 19.8 |
| step 156（阶段 D 终点，R1-Zero 最终模型） | 87.4 | 73.8 | 27.5 | 22.3 | 21.6 |
| E（蒸馏 SFT，974 步） | 86.4 | 70.8 | 31.0 | 26.7 | 24.6 |
| **F 第 20 步（E 上做 RL，项目最终模型）** | **89.4** | **76.6** | **32.7** | **29.4** | **26.3** |

AIME 2026 是训练和选模型都没碰过的留出集（29 题），用来检查有没有对原评测集过拟合。

纯 RL 把基座的 MATH-500 从 50.6% 提到 87.4%、AIME24 从 4.6% 提到 27.5%，AIME 2026 留出集到 21.6%。第三轮从 step 76 到 156 的累计提升是显著的：四个原评测集等权平均 **+4.41 个百分点，95% 区间 [+2.57, +6.30]**，AIME 2026 **+7.5 [+2.2, +13.8]**；单个阶段里 B（动态采样，+1.70）和 D（边界题，+1.89）显著。AIME 每题 16 次里至少对一次的比例（pass@16）约是单次正确率的 2 倍，但仍有 40–53% 的 AIME 题 16 次全错（见 [docs/04](docs/04_eval_and_stats.md) 第 7 节）。

和"能直接用"的目标（MATH-500 ≥ 90%、AIME24/25 ≥ 40%）相比，纯 RL 在 8B 基座上已经接近平台期，所以 11/19 起转向蒸馏 SFT（E），再在 SFT 模型上做 RL（F）。

## 结论

最终模型 F 第 20 步：MATH-500 89.4%、AMC23 76.6%、AIME24 32.7%、AIME25 29.4%、留出的 AIME 2026 26.3%，四集平均 57.0%。它四项都超过了事先估计的"好"档，但没有达到"可以用"的目标。

主要发现：

1. **零起点 RL 前期涨得快、后期很慢。** 基座到第 25 步，四个评测集涨了 10–27 个百分点；第三轮 80 步一共只涨 4.4。检验过的四个改动里，只有动态采样（+1.70）和只训边界题（+1.89）显著有效。
2. **8B 零起点 RL 的 AIME 停在 22–28%。** 蒸馏 SFT（6.2 万条 DeepSeek-R1 解答）的总分和它持平，但会做的题更多：AIME24 的 pass@16 是 76.7%，step 156 是 60.0%。E 的短板是写不完，AIME 约一半回答到 16K 还没写完。
3. **在 SFT 模型上做 RL 效果最好。** 20 步（有效约 10 步）四集平均 +3.29，主要来自写完的比例（AIME 51% → 70%）。这是项目里唯一一次显著超过 R1-Zero 最终模型。
4. **写完不等于做对。** 推理时加 presence_penalty，E 的 AIME 正确率 +3.0，对 F 却没有用：F 多写完的回答只有 4–8% 是对的。F 剩下的错误主要是做不出来的题，而不是重复循环。
5. **评测要用完整评测加按题配对 bootstrap。** 判定标准在看结果前定好；每题只采 1 次的验证会误导（F 第 20 步被低估约 2.6 个点）；判分器本身也要审计。
6. **时间损失大多来自工程问题，而且都没有报错。** 包括 verl 续训不同步优化器主参数、各推理引擎种子相同、只存权重时学习率调度器从 0 开始，以及磁盘和存盘管理，都是靠指标异常才发现的。

为什么收尾：按 F 第一阶段的速度，AIME 每 20 步约 +2 个点。原计划的第二阶段（10 步、约 ¥450）预计只涨 +0.5 到 +1.5，小于完整评测的误差（约 ±1.7）。公开的 7B 蒸馏模型 AIME24 在 50% 以上，但用了 35 万条以上的解答和 32K 的生成上限（[OpenR1-Distill-7B](https://huggingface.co/open-r1/OpenR1-Distill-7B)、[DeepSeek-R1](https://arxiv.org/abs/2501.12948)）。两者的基座和评测方式不同，不能直接比；但差距主要在蒸馏数据量和生成长度，不在 RL 步数。要达到"可以用"，下一步应是重做更大规模的蒸馏（几十万条、32K），再接短 RL。

## 做了什么

| 轮次 / 阶段 | 步数 | 改动 | 结论（同题配对 bootstrap，四集等权平均，95% 区间） |
|---|---|---|---|
| 第一轮 | 0 → 49 | GRPO 基线，回答上限 4K（后期截断率升到约 11%） | 基座 → step 25：MATH-500 50.6 → 75.2 |
| 第二轮 | 40 → 76 | 回答上限放到 8K + DAPO 超长软惩罚；每 2 步存盘、每 4 步验证 | 第 50 步存盘时数据盘写满崩溃、第 56 步存盘损坏，回滚到 46 续训；提前停在 76 |
| 审计 | – | 判分器审计（漏判）、训练/评测题泄露检查 | 判分改为 v2；训练集再去掉 6 道泄露题；加 AIME 2026 留出集 |
| 第三轮 A | 76 → 96 | 每批数据更新 8 次（原 2 次） | −0.32 [−1.84, +1.14]，无显著差异 |
| 第三轮 B | 96 → 116 | 加 DAPO 动态采样 | **+1.70 [+0.09, +3.32]，显著提升** |
| 第三轮 C | 116 → 136 | 学习率 1e-6 → 2e-6（保留动态采样） | +1.15 [−0.42, +2.75]，无显著差异 |
| 第三轮 D | 136 → 156 | 只练"边界题"：用 step 136 对 2,047 道训练题各采样 8 次，取答对 1–5 次的 739 题 + 全错的 316 题（封顶 30%），共 1,055 题；每题 32 条，学习率回到 1e-6 | **+1.89 [+0.17, +3.74]，显著提升** |
| E 蒸馏 SFT | 974 步 | OpenR1-Math 的 6.2 万条 DeepSeek-R1 解答，全参数 SFT，上限 16K | 对 step 156 +1.0 [−1.2, +3.2]，无显著差异；AIME 约一半写不完 |
| F 第一阶段 | 20 步 | 在 E 上做 RL：上限 16K、12K 起扣分，1,065 道边界题，每步 32 题 × 16 条 | **对 E +3.29 [+1.63, +5.07]、对 step 156 +4.3 [+2.0, +6.7]，显著提升** |
| 推理试验 | – | 生成时加 presence_penalty（不训练） | E 的 AIME +3.0 [+0.5, +5.8]；F 四集平均 −0.6 / +0.2，无变化 |

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

1. **有效的是改变"训练信号"的两个改动：动态采样和边界题。** 每批更新次数、学习率翻倍都没有显著差异；动态采样（B，+1.70）丢掉全对/全错的组，边界题（D，+1.89）只训测出来 8 次里答对 1–5 次的题（外加封顶 30% 的全错题），两者都让每一步的梯度尽量来自"有对有错"的组。
2. **单次采样的评选分噪声很大。** D 阶段评选分 0.66 → 0.65 → 0.63 → 0.6725，中间那次"下降"在配对检验下不显著，下一次就反弹成新高；而评选分最高的 step 148 在完整评测上反而不如阶段终点 step 156（+1.18 不显著，对 +1.89 显著）。不能凭一两个评选点提前停训；结论只看阶段末的完整评测 + 配对 bootstrap。
3. **判分器要审计。** v1 判分漏判了大量 markdown/LaTeX 包装的正确答案；修正后各阶段的最佳存盘排序有变化。训练后期模型逐渐改用 \boxed 收尾（"没写 Answer 行"从 7% 升到约 50%），判分器必须两种格式都认。
4. **显存 OOM 要看清是谁占的。** nvidia-smi 里 SGLang 的数字包含 PyTorch 缓存（OOM 前会先释放）；真正挤掉空间的是训练进程在生成阶段残留的显存。详见 [docs/03_ops_oom_monitoring.md](docs/03_ops_oom_monitoring.md)。
5. **每次续训学习率都会重新预热 3 步。** 中途重启有隐性成本，能不重启就不重启。
6. **磁盘是硬约束。** 只存模型权重（不存优化器状态）、只留最新 + 最佳、存盘前先校验完整性再删旧的。
7. **容器里的分布式环境要单独验证，冒烟测试要包含存盘。** 同一台机器上 Ray（每个进程只看到一张卡）一直正常，换成 torchrun（每个进程看到全部 4 张卡）跑 SFT 时 NCCL P2P 卡死，要 `NCCL_P2P_DISABLE=1`；verl 0.6.1 的 engine 版 SFT 忽略 `checkpoint.save_contents`，默认存 fp32 模型 + 优化器分片（8B 约 100 GB），冒烟测试第一次存盘就把数据盘写满。详见 [docs/03](docs/03_ops_oom_monitoring.md) 第 7 节。
8. **多个推理引擎要各自设种子。** verl 在每张卡上用同一个种子初始化 Python 随机数，SGLang 又从里面取种子，4 个引擎因此完全同步；再加上请求分配相同，F 第 1–10 步生成的回答有 3/4 是重复的。这种情况没有任何报错，是从优势值的上下限恰好等于 ±1.677（每种结果至少出现 4 次时的上限）发现的。
9. **只存权重的续训会丢掉学习率调度器状态。** Megatron 每次重启都从 0 开始预热，重启后第一步等于没训。
10. **判分不要放进多进程池。** math-verify 判等会自己开子进程池，放进 multiprocessing 的守护进程里会静默失败（MATH-500 少判对 5 个点）；按条判分 2,564 条也只要约 10 秒。
11. **写完 ≠ 做对。** 惩罚重复、缩短长度都能让模型写完，但多写完的大多是错答案；真正的提升要靠 RL 提高走对路的概率。

## 蒸馏 SFT（E）→ 在 SFT 模型上做 RL（F）

[sft/](sft/) 是第 E 阶段：用 [open-r1/OpenR1-Math-220k](https://huggingface.co/datasets/open-r1/OpenR1-Math-220k)（Apache-2.0，DeepSeek-R1 生成的推理过程）对 Qwen3-8B-Base 做全参数 SFT。

- 数据（[prep_openr1.py](sft/prep_openr1.py)）：每题只取一条判对、推理完整、最短的生成；去掉含中文的、</think> 之后没有 \boxed 答案的、同一行重复 ≥ 5 次的生成；和全部评测题做词级 13-gram 去污染（连同题面含中文的共删 482 题）；套 Qwen3 chat 模板后 ≤ 16,000 token；得到 62,382 条训练 + 256 条验证，约 3.26 亿 token
- 训练（[run_sft.sh](sft/run_sft.sh)）：verl 的 engine 版 SFT trainer（FSDP2，按 token 数动态拼批，每卡每批最多 16,384 token），1 个 epoch（974 步），全局批 64 条，lr 2e-5、cosine、预热 3%，最长 16,384 token；[run_engine_sft.py](sft/run_engine_sft.py) 给 verl 打了两个补丁：存盘时把权重转成 bf16，并修正 verl 0.6.1 忽略 `checkpoint.save_contents` 的 bug（只存 HF 权重，约 16 GB）。环境变量 `NCCL_P2P_DISABLE=1`、`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`；融合 CE 核和 torch 后端不兼容，关掉
- 评测（[eval_e.py](sft/eval_e.py)）：和 R1-Zero 同一套评测题、同一个判分函数，和 R1-Zero 的最终模型做同题配对 bootstrap
- 总控（[start_e_resume.sh](sft/start_e_resume.sh)）：冒烟测试（训 3 步并存一次盘）→ 用冒烟存盘试跑评测 → 正式训练 → 完整评测 → 写结论 → 关机。最初的 [start_e.sh](sft/start_e.sh) 开头写了 `set -f`，路径通配全部失效，换成了这个续跑版
- 结果（2026-11-20 04:26，[完整结论](../results/r1zero_qwen3_8b_rl_v1/audit/STAGE_E_RESULT.md)）：974 步约 13 小时 15 分训完，四集平均比 R1-Zero step 156 高 1.0 个百分点（95% 区间 −1.2 到 +3.2），按事先定的标准无显著差异：

| 评测集 | E | step 156 | 差值（95% 区间） | 写完的比例 | 写完部分的正确率 |
| --- | --- | --- | --- | --- | --- |
| MATH-500 | 86.4% | 87.4% | −1.0（−4.2 到 +2.2） | 92% | 93.9% |
| AMC23 | 70.8% | 73.8% | −3.0（−7.5 到 +1.6） | 82% | 86.1% |
| AIME24 | 31.0% | 27.5% | +3.5（−0.6 到 +7.3） | 50% | 61.6% |
| AIME25 | 26.7% | 22.3% | +4.4（−0.6 到 +10.0） | 51% | 52.0% |
| AIME 2026（留出） | 24.6% | 21.6% | +3.0（−3.9 到 +9.3） | 52% | 47.7% |

E 的主要问题是写不完：AIME 约一半回答到 16K 还没写完，截断的全部算错，其中四成多是同一行反复出现的死循环；没有中文混杂；平均长度 MATH-500 约 4.0K token、AMC23 7.2K、AIME 约 12K。但 E 会做的题明显更多：pass@1/4/16 在 AIME24 是 31.0/53.5/76.7%（step 156 为 27.5/42.3/60.0%），AIME 2026 是 24.6/44.7/62.1%（156 为 21.6/33.5/48.3%）。所以 F 的重点是让模型写完、写短：思考没写完记 −1，超过 12K 开始扣分。

[f/](f/) 是第 F 阶段：从 E 的模型出发做 RL。提示和判分与 SFT、评测一致，只判 `</think>` 之后的答案，思考没写完记 −1；回答上限 16K、12K 起线性扣分。先用 E 的模型对 2,048 道训练题各采样 4 次选边界题（[profile_f.py](f/profile_f.py)、[select_frontier_f.py](f/select_frontier_f.py)），冒烟 1 步后第一阶段 20 步（每步 32 题 × 16 条，lr 1e-6，动态采样），阶段末和 E 同题配对比较（[compare_f.py](f/compare_f.py)），总控 [start_ef.sh](f/start_ef.sh)。停训规则原定“截断率连续 2 步 ≥ 20%”，因为 E 本身截断率就高，开跑前改成相对第 1 步的基线（[monitor_f.py](f/monitor_f.py)，原版 `monitor_f.py.orig`）。

F 第一阶段（2026-11-20 13:45 – 11-22 10:11，[完整结论](../results/r1zero_qwen3_8b_rl_v1/audit/STAGE_F1_RESULT.md)）：

| 评测集 | step 156 | E | F 第 20 步 | F − E（95% 区间） | F − step 156（95% 区间） | 正常写完 E → F |
| --- | --- | --- | --- | --- | --- | --- |
| MATH-500 | 87.4% | 86.4% | 89.4% | +3.0（+0.4 到 +5.6） | +2.0（−1.0 到 +5.0） | 92.0% → 96.0% |
| AMC23 | 73.8% | 70.8% | 76.6% | +5.8（+2.7 到 +9.1） | +2.8（−1.4 到 +7.5） | 82.2% → 88.1% |
| AIME24 | 27.5% | 31.0% | 32.7% | +1.7（−2.1 到 +6.0） | +5.2（+1.5 到 +9.4） | 50.4% → 66.9% |
| AIME25 | 22.3% | 26.7% | 29.4% | +2.7（−0.8 到 +6.7） | +7.1（+1.2 到 +13.8） | 51.2% → 72.3% |
| AIME 2026（留出） | 21.6% | 24.6% | 26.3% | +1.7（−1.3 到 +5.0） | +4.7（−2.2 到 +11.4） | 51.5% → 70.0% |
| 四集平均 | 52.8% | 53.7% | 57.0% | **+3.29（+1.63 到 +5.07）** | **+4.3（+2.0 到 +6.7）** | |

- 第 1–10 步其实只有约 1/4 的样本是不同的：4 个 SGLang 引擎的随机种子相同，又收到相同的请求，每题 16 条回答只有 4 种。修复办法是给每个引擎设不同的种子，见 [f/patches/fix_sglang_seed_1121.py](f/patches/fix_sglang_seed_1121.py)。第 1、3 步因为重启后学习率调度器从 0 开始，没有更新权重；修复见 [make_command_f.py](f/make_command_f.py)。所以第一阶段实际有效的约 10 步。
- 推理时加 presence_penalty（[eval_reppen.py](f/eval_reppen.py)、[eval_pp_F.py](f/eval_pp_F.py)）：
  - E 加 1.5 后，AIME 正确率 +3.0，写完比例 +13 个点。
  - F 加 1.5 / 2.0 后，四集平均 −0.6 / +0.2，都不显著；AIME 写完比例从 70% 升到 83–86%，但多写完的回答只有 4–8% 是对的。
  - 两者在"写完"上能叠加，在正确率上不叠加。详见 [E_PRESENCE_PENALTY_RESULT.md](../results/r1zero_qwen3_8b_rl_v1/audit/E_PRESENCE_PENALTY_RESULT.md) 和 [F_PRESENCE_PENALTY_RESULT.md](../results/r1zero_qwen3_8b_rl_v1/audit/F_PRESENCE_PENALTY_RESULT.md)。
- 每步约 104 分钟，是 R1-Zero 第三轮的 3 倍多：回答长 3.7 倍，每张卡能同时生成的条数只有约 1/3。原计划还有第二阶段（再跑 10 步），因为预计提升小于评测误差，没有做。

## 目录

| 路径 | 内容 |
|---|---|
| [ma/](ma/) | 多智能体流水线、判分器、审计和统计脚本 |
| 根目录的 .sh、[run/](run/) | 启动（run_loop3.sh）、看门狗（watchdog3.sh）、阶段接续、下载脚本；实际训练命令和对 verl/SGLang 的补丁（run/） |
| [sft/](sft/) | 第 E 阶段（蒸馏 SFT）的数据准备、训练、评测、总控 |
| [f/](f/) | 第 F 阶段（在 SFT 模型上做 RL）：选题采样、训练命令生成、停训规则、总控、续跑、配对比较、推理试验；补丁和分析脚本在 [f/patches/](f/patches/) |
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
