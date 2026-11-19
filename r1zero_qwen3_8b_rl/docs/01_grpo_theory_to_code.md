# GRPO / DAPO 的公式在 verl 代码里对应哪里

以 verl 0.6.1 + recipe/dapo 为准（本项目 vendored 在 third_party/verl_v061）。行号可能随版本变化，以函数名为准。

## 1. 目标函数

对每道题 q，用旧策略 π_old 采样一组 G 条回答 o_1 … o_G（第三轮前三个阶段 G = 16，D 阶段 G = 32），规则判分 r_i ∈ {+1, −1}（超长再线性扣分）。

**优势（结果监督，整条回答共享一个值）**

    A_i = (r_i − mean(r_1..r_G)) / (std(r_1..r_G) + ε)

→ verl/trainer/ppo/core_algos.py 的 compute_grpo_outcome_advantage（algorithm.adv_estimator=grpo，norm_adv_by_std_in_grpo=True）。按 uid 把同一道题的回答分组，求组内均值和标准差，再把标量优势广播到这条回答的每个 token（乘 response_mask）。

**每个 token 的重要性比**

    ρ_t = π_θ(o_t | q, o_<t) / π_old(o_t | q, o_<t)

π_old 的 log-prob 在更新前由训练引擎（Megatron）重新算一遍（日志里的 old_log_prob 阶段），不直接用生成引擎给的概率——两边数值精度不同（SGLang 推理 vs Megatron 训练），直接用会引入偏差。生成端的概率只拿来监控两者的差异（rollout_corr/* 指标，KL 约 1e-3）。

**裁剪后的策略损失（DAPO 版本）**

    L_t = max( −A·ρ_t , −A·clip(ρ_t, 1 − ε_low, 1 + ε_high) )      ε_low = 0.2，ε_high = 0.28（clip-higher）
    A < 0 时再取 min(L_t, −A·c)，c = 10                               （dual-clip，防止 ρ 很大时负优势的损失爆掉）

→ core_algos.py 的 compute_policy_loss（actor.clip_ratio_low / clip_ratio_high / clip_ratio_c）。clip-higher 让低概率 token 被"提上来"的空间更大，缓解熵塌缩。

**聚合：token 级平均**

    L = Σ_i Σ_t L_{i,t} / Σ_i |o_i|

→ core_algos.py 的 agg_loss（actor.loss_agg_mode=token-mean）。原始 GRPO 是先在每条回答内平均再在回答间平均（seq-mean），长回答里每个 token 的权重被稀释；token-mean 让长、短回答的每个 token 权重相同。

**没有的项**：不加 KL 惩罚（algorithm.use_kl_in_reward=False，actor.use_kl_loss=False），所以不需要参考模型；GRPO 不需要 critic；奖励是规则函数，不需要奖励模型。entropy_coeff = 0。

## 2. DAPO 的另外两项

**超长软惩罚**：回答超过 max_response_length − overlong_buffer（8192 − 2048 = 6144）token 后，按超出的长度线性扣分，到 8192 扣满 1 分。→ reward_model.reward_manager=dapo（verl/workers/reward_manager/dapo.py）的 overlong_buffer 配置。

**动态采样（filter_groups）**：一组回答全对或全错时，组内标准差为 0，优势全是 0，这组对梯度没有贡献，却占了批次。DAPO 的做法是把这些组丢掉，继续生成新的题目，直到凑够 train_batch_size 个"有对有错"的组（最多生成 max_num_gen_batches = 6 批）。→ recipe/dapo/dapo_ray_trainer.py 的 fit()：生成之后、算优势之前，按 metric=acc 统计每个 uid 的标准差，只保留 std > 0 的组；不够就打印 "num_gen_batches=… Keep generating..." 再生成一批。D 阶段的边界题全错的比例高，几乎每一步都要生成 2 批。

动态采样是第三轮唯一在配对检验下显著有效的改动（B 阶段 +1.70 个百分点，95% 区间 [+0.09, +3.32]）。

## 3. 一步训练的完整流程（D 阶段，约 30 分钟）

| 顺序 | 做什么 | 代码位置 | 耗时（第 135 步） |
|---|---|---|---|
| 1 | 训练引擎把权重从 CPU 搬回 GPU，用 mbridge 转成 HF 格式，原卡传给 SGLang（update_weights），再卸回 CPU；SGLang 恢复 KV cache | megatron_workers.py 的 rollout_mode | 几十秒 |
| 2 | SGLang 对 32 题 × 32 条生成（温度 1.0），不够就再生成一批 | dapo_ray_trainer.py 的 generate_sequences | ~800 秒 |
| 3 | 规则判分 + 超长扣分；过滤全对/全错的组 | reward_v2.py；filter_groups | 秒级 |
| 4 | Megatron 重算 π_old 的 log-prob | compute_log_prob（old_log_prob） | ~230 秒 |
| 5 | 算 GRPO 优势 | compute_grpo_outcome_advantage | 秒级 |
| 6 | 加载参数和优化器，按 mini batch（1/8 批）更新 8 次，每次按 token 数动态拼 micro batch | megatron_workers.py 的 update_actor | ~750 秒 |
| 7 | 卸载：参数和优化器状态 pin 到 CPU 内存，GPU 上的存储释放（storage.resize_(0)）；梯度直接丢弃 | megatron_utils.py 的 offload_megatron_model_to_cpu | 秒级 |

第 6 步更新完的新权重就是下一步第 1 步要同步给 SGLang 的权重——CPU 上那份就是最新的，不需要另存"旧权重"：π_old 在第 4 步已经算成 log-prob 存起来了。

## 4. "一批数据更新 8 次"是什么意思

train_batch_size = 32 题（× 32 条），ppo_mini_batch_size = 4 题 → 一批切成 8 个 mini batch，按顺序各做一次梯度更新（ppo_epochs = 1）。第一次更新时 π_θ = π_old，ρ = 1；后面 7 次 π_θ 已经离开 π_old，裁剪开始起作用（pg_clipfrac 约 0.2%）。第三轮 A 阶段把它从 1 次改成 8 次，配对检验无显著差异，但样本利用率更高，之后一直沿用。
