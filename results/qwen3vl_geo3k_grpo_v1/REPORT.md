# Qwen3-VL-4B geometry3k 多智能体 GRPO 报告

> 本文件由多智能体 loop 自动生成（决策智能体随时更新），原样保留。之后的诊断实验与拒绝采样微调（RFT）不在 loop 内，
> 结果见同目录的 `eval_summary.csv`、`rft_progress.log` 与 [`qwen3vl_geo3k_grpo/README.md`](../../qwen3vl_geo3k_grpo/README.md)。

- 更新：2026-11-02 19:25:26　开始：2026-11-01 11:43:52　截止：2026-11-02 20:50:38
- 当前阶段：**done**

- 硬件：2 × NVIDIA A800 80GB PCIe（80.0GB），数据盘剩余 53.9GB
- verl：0.8.0（配置模式 strict）

## 结果

| 模型 | val 准确率 | test 准确率 |
|---|---|---|
| 基线（未训练） | 50.0% | 50.08% |
| 第 1 轮 grpo | 52.67% | 53.58% |
| 第 2 轮 grpo | 50.67% | — |

test 逐题对比：修复 49 题、弄坏 28 题，符号检验 z = 2.39（95% 显著）。最佳模型：`/root/autodl-tmp/geo3k_runs/multiagent_v1/rounds/round_1/best_hf`

## 数据清洗

| 划分 | 输入 | 保留 | 问题 |
|---|---|---|---|
| val | 300 | 300 | {} |
| test | 601 | 601 | {'resized_large': 3} |
| train | 2101 | 2098 | {'resized_large': 14, 'leak_to_eval': 1, 'duplicate': 1, 'opaque_answer': 1} |

难度过滤：{'input': 2098, 'kept': 1507, 'dropped_all_correct': 591, 'no_rate': 0, 'all_wrong_kept': 596, 'pass_rate_hist': {'0.0': 596, '0.125': 168, '0.25': 123, '0.375': 109, '0.5': 107, '0.625': 95, '0.75': 114, '0.875': 140, '1.0': 55}, 'applied': True}

## 第 1 轮：grpo（failed）

- 方案理由：首轮用 grpo（标准 GRPO：组内标准化优势 + low_var_kl 正则（verl geo3k 官方配方））建立可复现基线
- 结束原因：failed；诊断：{'entropy_first': 0.4709, 'entropy_last': 0.4522, 'entropy_collapse': False, 'response_clip_ratio': 0.4102, 'score_first': 0.3316, 'score_last': 0.3043, 'step_seconds': 1161.4058}
- 验证曲线：0:50.67，0:49.0，0:48.33，0:49.33，0:47.67，0:50.0，8:53.0，8:54.0，8:52.0，8:51.67，8:52.67，8:50.67，8:52.33

## 第 2 轮：grpo（deadline）

- 方案理由：按用户要求继续用 grpo（标准 GRPO：组内标准化优势 + low_var_kl 正则（verl geo3k 官方配方））；从第 1 轮最佳模型（val 52.67%）继续训练
- 结束原因：deadline；诊断：{'entropy_first': 0.4433, 'entropy_last': 0.3778, 'entropy_collapse': False, 'response_clip_ratio': 0.2965, 'score_first': 0.3472, 'score_last': 0.4211, 'step_seconds': 1032.9877}
- 验证曲线：0:50.67，0:52.33，8:49.67，8:55.0，16:50.67，24:54.0

## 决策记录（最近 30 条）

- 2026-11-01 11:43:52 **start_data** → data：环境就绪：2×NVIDIA A800 80GB PCIe，数据盘剩余 53.9GB，模型 /root/autodl-tmp/qwen3_vl_4b（约 4.44B 参数），rollout 引擎 vllm，可用算法预设 ['grpo', 'dapo_lite', 'drgrpo']
- 2026-11-01 11:44:13 **set_max_prompt_length** → trainer：实测 prompt 最长 599 token（p99 463），max_prompt_length 设为 768
- 2026-11-01 12:02:02 **baseline_recorded** → all：基线模型 val 43.0%，test 45.42%（test 仅用于最终对照）
- 2026-11-01 14:58:54 **raise_generation_length** → all：基线在 val 上 44.0% 的回答在 2048 token 处被截断（平均 4214.8 字符），训练与评估的生成长度统一提高到 4096，并重新评估基线
- 2026-11-01 15:05:37 **note_truncation** → all：生成长度已到上限 4096，仍有 34.0% 截断；训练会自然压缩回答长度
- 2026-11-01 15:10:05 **baseline_recorded** → all：基线模型 val 50.0%，test 50.08%（生成长度 4096；test 仅用于最终对照）
- 2026-11-01 16:52:21 **use_filtered_train** → trainer：基线对 591 道题 8 次全对（组内优势恒为 0），剔除后训练集 1507 题；全错题 596 道保留（可能随训练变得可学）
- 2026-11-01 16:52:21 **approve_plan** → trainer：首轮用 grpo（标准 GRPO：组内标准化优势 + low_var_kl 正则（verl geo3k 官方配方））建立可复现基线；磁盘不足以保存优化器状态：崩溃续训时 Adam 状态会重置
- 2026-11-01 17:02:40 **abort** → all：训练因环境错误失败，重试无意义；修复后重新启动 run_loop.sh 会从本轮开头继续
- 2026-11-01 17:14:52 **use_sdpa** → trainer：flash-attn 的 CUDA 扩展不可用，训练改用 PyTorch SDPA 注意力并关闭 remove_padding（速度约慢 10%～30%；装好匹配的 flash-attn 后会自动切回）
- 2026-11-01 17:14:52 **approve_plan** → trainer：首轮用 grpo（标准 GRPO：组内标准化优势 + low_var_kl 正则（verl geo3k 官方配方））建立可复现基线；磁盘不足以保存优化器状态：崩溃续训时 Adam 状态会重置
- 2026-11-02 00:05:04 **pin_best** → trainer：第 8 步验证准确率 53.00%（acc）创新高，钉住该检查点
- 2026-11-02 00:11:27 **pin_best** → trainer：第 8 步验证准确率 54.00%（acc）创新高，钉住该检查点
- 2026-11-02 02:34:25 **round_result** → all：第 1 轮（grpo）最佳 step 8：val 52.67%，较基线 +2.67 个点（修复 34 / 弄坏 26，不显著）
- 2026-11-02 02:34:26 **final** → evaluator：已完成 1 轮（上限 1），进入最终评估
- 2026-11-02 02:38:20 **complete** → all：最佳为第 1 轮（grpo）：test 53.58%，基线 50.08%（修复 49 / 弄坏 28，z=2.39）；模型在 /root/autodl-tmp/geo3k_runs/multiagent_v1/rounds/round_1/best_hf
- 2026-11-02 02:38:20 **shutdown** → all：按配置在完成后关机以停止 AutoDL 计费
- 2026-11-02 08:50:51 **use_sdpa** → trainer：flash-attn 的 CUDA 扩展不可用，训练改用 PyTorch SDPA 注意力并关闭 remove_padding（速度约慢 10%～30%；装好匹配的 flash-attn 后会自动切回）
- 2026-11-02 08:50:51 **approve_plan** → trainer：按用户要求继续用 grpo（标准 GRPO：组内标准化优势 + low_var_kl 正则（verl geo3k 官方配方））；从第 1 轮最佳模型（val 52.67%）继续训练；磁盘不足以保存优化器状态：崩溃续训时 Adam 状态会重置
- 2026-11-02 13:00:05 **pin_best** → trainer：第 8 步验证准确率 55.00%（acc）创新高，钉住该检查点
- 2026-11-02 19:25:25 **round_result** → all：第 2 轮（grpo）最佳 step 8：val 50.67%，较基线 +0.67 个点（修复 29 / 弄坏 27，不显著）
- 2026-11-02 19:25:26 **final** → evaluator：已完成 2 轮（上限 2），进入最终评估
- 2026-11-02 19:25:26 **complete** → all：最佳为第 1 轮（grpo）：test 53.58%，基线 50.08%（修复 49 / 弄坏 28，z=2.39）；模型在 /root/autodl-tmp/geo3k_runs/multiagent_v1/rounds/round_1/best_hf
- 2026-11-02 19:25:26 **shutdown** → all：按配置在完成后关机以停止 AutoDL 计费
