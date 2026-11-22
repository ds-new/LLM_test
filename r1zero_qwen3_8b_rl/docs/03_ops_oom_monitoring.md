# OOM 排查、看门狗和巡检的判断标准、续训和磁盘的坑

## 1. 第 136 步的 OOM（第三轮 C 阶段）

**现象**：第 136 步生成阶段崩溃，loop 暂停等 Claude 判断。

**报错**（SGLang 的 logits_processor 在算 input logprob）：

    Tried to allocate 1.70 GiB. GPU 0 has a total capacity of 79.25 GiB of which 1.67 GiB is free.
    Process … has 19.29 GiB memory in use.   ← 同卡的 Megatron 训练进程
    Including non-PyTorch memory, this process has 58.18 GiB memory in use.
    Of the allocated memory 56.67 GiB is allocated by PyTorch, … and 646.49 MiB is reserved by PyTorch but unallocated.

**分析**：

- 只差 0.03 GiB。SGLang 自己的缓存只有 0.65 GiB（PyTorch 报 OOM 之前已经把能释放的缓存都释放了），真正挤掉空间的是同卡的训练进程在生成阶段还占着 19.29 GiB。
- 为什么之前没出现：从 C 阶段开始打开了 calculate_log_probs（记录生成端概率，用来监控训推不一致），SGLang 要为被抢占后重算的长请求算 input logprob，临时张量变大了。

**处理**：在显存档位表里加一档，只把 SGLang 的显存比例从 0.70 降到 0.65（KV cache 少约 4 GiB），其余不变；从第 134 步续训。截至第 151 步没有再 OOM。

## 2. 显存监控：看哪个数

D 阶段第 141 步生成时，有一张卡上 SGLang 进程在 nvidia-smi 里涨到 71.2 GB，整卡只剩 0.44 GB，超过了事先定的"合计接近 78 GB 就考虑降档"。但对照上面那次 OOM 的报错：

- nvidia-smi 显示的是进程占用（包含 PyTorch 缓存），PyTorch 遇到分配失败会先把缓存还回去再重试；
- 那次 OOM 时 SGLang 实际在用 56.67 GiB，缓存只有 0.65 GiB；
- D 阶段训练进程在生成期间只占约 9.3 GB，比那次少 10 GB。

所以没有重启（重启会丢掉当前这一步，还要重新预热 3 步学习率），而是把标准改成：**生成阶段训练进程的占用 ≥ 15 GB、并且同卡 SGLang 也在高位时**，才在下一个存盘步之后主动降到 0.60。截至第 151 步没有 OOM。

## 3. 看门狗（watchdog3.sh）

每分钟看一次 run/state.json 的 phase 和 loop 进程：

| 情况 | 动作 |
|---|---|
| phase = done | loop 自己 60 秒后关机，看门狗 10 分钟兜底 |
| phase = failed / waiting_claude，或 loop 进程没了 | 30 分钟后关机 |
| run/REPAIRING 存在且 20 分钟内更新过 | 不关机（有人在修，修的人每次操作都 touch 它） |
| run/KEEP_RUNNING 存在 | 不关机（例如 E 阶段由自己的总控负责关机） |

巡检（Claude 每 25 分钟一次）的规则写在每次巡检的提示里：熵 > 0.5 告警、连续 2 步 > 1.0 暂停；训练中途出错（不是阶段完成）就先 touch REPAIRING 再排查；机器关了或连不上就停止巡检。

## 4. 续训的隐性成本

- verl 每次启动都按 lr_warmup_steps=3 重新预热：续训后第 1 步学习率是 0，第 2、3 步是 1/3、2/3。每重启一次，相当于少训了约 2 步。
- 只在偶数步存盘，崩溃会丢掉最多 2 步。
- 所以出现"可能会 OOM"的迹象时，先判断概率和代价，不轻易主动重启。

## 5. 磁盘

数据盘只有 50 GB，系统盘 30 GB。

- 只存模型权重（save_contents=[model]，8B 约 16 GB），不存优化器状态（会到 100 GB 以上）。代价是续训时 Adam 状态从零开始。
- 只留最新存盘 + 评选最佳（最佳单独放在系统盘的 /root/r1zero_best）。
- 第二轮第 50 步存盘时数据盘写满，存盘损坏、训练崩溃；之后改成删旧存盘前先校验新存盘完整，verl 自己不删（max_actor_ckpt_to_keep=null），由评测智能体管理。
- 基座模型的一个分片挪到系统盘，用软链接接回来，给数据盘腾出 3.8 GB。

## 6. 其他记录

- 第二轮第 56 步存盘损坏（"is not a distributed checkpoint"），回滚到第 46 步。
- 有一次实例在 14:50 被外部关机（日志里没有任何报错、看门狗也没有关机记录），开机后从最新存盘续训。
- 阶段之间的自动接续由 ma/stage_watch.py 完成：阶段完成 → 完整评测 → 按事先定的标准写结论 → 选下一阶段的配置并启动。D 之后改成直接启动蒸馏 SFT 的总控 sft/start_e.sh。

## 7. 第 E 阶段（SFT）启动时踩的坑（2026-11-19）

RL 一直用 Ray 起进程，换成 torchrun 跑 SFT 后碰到四个之前没暴露的问题，从 13:09 自动启动到 13:34 正式训练，前后约 25 分钟：

1. **总控脚本 `set -f`**：`start_e.sh` 开头写了 `set -f`（照抄 `run_sft.sh`，那里是为了 hydra 的列表参数），结果总控里 `rm -rf .../global_step_*`、`ls -d .../global_step_*/huggingface` 这类路径通配全部失效：删除没执行，后面会找不到存盘。总控换成了不带 `set -f` 的 `start_e_resume.sh`。
2. **NCCL P2P 卡死**：冒烟测试卡在第一个 `barrier`，不报错，py-spy 看到卡在 NCCL 调用里。用 NCCL 小测试确认：默认、`NCCL_CUMEM_ENABLE=0`、`NCCL_IB_DISABLE=1` 都卡，只有 `NCCL_P2P_DISABLE=1` 正常。RL 阶段 Ray 给每个 worker 只露一张卡，所以没碰到；torchrun 下每个进程看得到全部 4 张卡。
3. **融合 CE 核**：`model.use_fused_kernels=True` 在 torch 后端报 `forward_with_torch_backend has to return_dict`，只能关掉。代价是 logits 要整块算：16K token × 15 万词表的 fp32 logits 和它的梯度要二三十 GB，冒烟时单卡显存缓存峰值 77.5 GiB。所以每卡每批上限设为 16,384 token（verl 要求不小于最长样本），并开 `expandable_segments` 减少碎片。
4. **存盘写满数据盘**：verl 0.6.1 的 FSDP engine 建 `FSDPCheckpointManager` 时传的是 `checkpoint_contents=...`，被 `**kwargs` 吞掉，`checkpoint.save_contents=[hf_model]` 不起作用，默认存 fp32 模型 + 优化器分片（8B 约 100 GB）。冒烟测试训完 3 步存盘，写了 22 GB 后磁盘满了报错。修法：在 `sft/run_engine_sft.py` 里把这个参数转给 `checkpoint_config`；存盘内容含分片就直接报错，不再静默写盘。

教训：冒烟测试要把正式流程里所有一次性的动作（存盘、评测、清理）都走一遍，不能只训几步。
