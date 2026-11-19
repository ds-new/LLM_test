# 显存账、为什么 8B 要 4 张 80GB、MFU

机器：AutoDL，4 × A800 80GB PCIe（卡间没有 NVLink），56 核 CPU，480 GB 内存。模型 Qwen3-8B（82 亿参数，36 层，8 个 KV 头 × 128 维）。

## 1. 训练和生成同卡分时

verl 的 hybrid engine 让训练引擎（Megatron）和生成引擎（SGLang）共用同一张卡，轮流占用：

- **生成阶段**：Megatron 的参数、梯度、优化器状态都卸到 CPU 的 pinned 内存（GPU 上的存储 resize 成 0），SGLang 占着权重 + KV cache 生成。
- **训练阶段**：SGLang 释放权重和 KV cache（只剩约 1.4 GB 的上下文），Megatron 把参数和优化器搬回 GPU 更新。
- **切换时**：Megatron 的权重通过 mbridge 转成 HF 格式，在同一张卡上直接交给 SGLang（update_weights），不经过磁盘。梯度在卸载时直接丢弃（下一步会重新算），不拷贝。

所以 CPU 内存曲线在训练中几乎是平的：pinned 内存里一直放着参数和优化器状态，切换只是在 CPU ↔ GPU 之间搬，不是"从内存搬走"。

## 2. 生成阶段（每卡一个 SGLang，TP=1，显存比例 0.65）

| 项 | 大小 |
|---|---|
| 权重（bf16） | 15.41 GiB |
| KV cache：每个 token 36 层 × 8 头 × 128 维 × 2（K、V）× 2 字节 = 144 KiB；255,755 个 token | 35.12 GiB |
| CUDA graph | 0.75 GiB |
| 合计（静态） | 约 51.3 GiB（0.65 × 79.25 GiB） |
| 临时张量（logits、算 logprob、被抢占请求的重算） | 几到十几 GiB，进 PyTorch 缓存 |
| 同卡上 Megatron 进程的残留 | D 阶段约 9.3 GB；之前的阶段 12–19 GiB |

一步要生成 32 题 × 32 条 = 1,024 条（每卡 256 条），平均约 3,000 token，KV cache 装不下全部并发，SGLang 会边生成边抢占（retract）再重算，日志里的 "KV cache pool is full. Retract requests." 就是这个。

## 3. 训练阶段（Megatron，TP=2 × DP=2，分布式优化器，序列并行，全量重计算）

| 项（每卡） | 大小 |
|---|---|
| 参数（bf16，TP 切一半） | 7.6 GiB |
| 梯度（fp32 累加，grad_reduce_in_fp32） | 15.3 GiB |
| Adam 状态（fp32 主权重 + m + v，DP 再切一半） | 22.9 GiB |
| 小计 | 45.8 GiB |
| 激活 + logits（每卡每个 micro batch 最多 10,240 token） | 约 9 GiB |
| 实测峰值 | max_memory_allocated 54–55 GiB，reserved 74–76 GiB |

整个模型的训练状态约 150 GB（bf16 参数 16 GB + fp32 梯度 33 GB + Adam 的 fp32 主权重和两个动量 98 GB），单张 80GB 放不下；切成 4 份之后每卡 46 GiB，再加上 8K 长序列的激活，正好在 80GB 里。

## 4. 为什么 8B 的 GRPO 要 4 张卡

- **放得下**：全参数 + Adam 的训练状态约 150 GB，必须分到多张卡（或者大量卸载到 CPU，那会慢很多）。
- **生成量大**：每步 1,000 条左右、每条几千 token 的生成占了一步时间的一半，4 个生成引擎并行才能把一步压到 30 分钟左右。
- **长序列**：8K 回答的 logits（8K × 15 万词表）和激活很占显存，TP=2 + 序列并行把它们切开。

## 5. MFU

MFU（Model FLOPs Utilization）= 实际完成的有效计算量 ÷ 硬件峰值算力。训练按每个 token 约 6 × 参数量的 FLOPs 算（前向 2 倍、反向 4 倍；重计算的那次前向不算"有效"），A800 的 bf16 峰值 312 TFLOPS。

| 阶段 | MFU | 原因 |
|---|---|---|
| 参数更新 | 约 19% | 全量重计算多一次前向；PCIe 上做 TP 的通信；动态拼批后 micro batch 大小不一 |
| 生成 | 约 10% | 逐 token 解码受显存带宽限制，不是算力；长尾：最后几条长回答拖着整批 |
| 整步 | 约 14% | 两者加权 |

第 135 步的耗时：生成 806 秒，重算 old log-prob 229 秒，更新 746 秒。

## 6. 和 SFT 对比

第 E 阶段的 SFT 不需要生成引擎，也不用重算 log-prob：FSDP2 把参数、fp32 梯度和 Adam 状态平均切到 4 张卡（约 33 GB/卡），每个 micro batch 按 token 数动态拼到 24,576 个，用融合的线性层 + 交叉熵避免把 16K × 15 万的 logits 整个物化出来。
