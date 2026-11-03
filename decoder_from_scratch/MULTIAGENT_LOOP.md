# 多智能体 Decoder 预训练 Loop（本机 MPS，72 小时）

代码：`decoder_from_scratch/multiagent/`　运行目录：`decoder_from_scratch/runs/multiagent_pretrain_v1/`

## 五个智能体与分工

| 智能体 | 文件 | 职责 |
|---|---|---|
| 决策（decision） | `decision_agent.py` | 编排阶段；审核其他智能体的报告；按明确策略做决策（重试/回退数据源、隔离领域、批准训练计划、缩短/延长计划、回滚、补数据、调配比、结束），每条决策写明理由 |
| 数据清洗（data） | `data_agent.py`、`cleaning.py` | 从 ModelScope（BAAI/IndustryCorpus，固定 revision）下载 12 个领域 → 规则清洗 → 全局精确/段落级近重复去重 → 按文档哈希划分 train/val/test → 调用 `embedding_model_train/tokenizer_training` 从头训练 32k BBPE → 多进程编码；训练中可在独立进程补数据 |
| 结构与算法（architect） | `architect_agent.py` | 复用 `model.py` 的 RoPE/GQA/Pre-RMSNorm/SwiGLU/残差 Decoder；启动时做因果性、KV Cache 一致性、过拟合自检；对 4 个候选结构做 fp32/bf16 吞吐基准；按“训练 token ≈ 20×参数”选出计算最优规模 |
| 训练配置/执行/监控（trainer） | `trainer_agent.py` | AdamW(β2=0.95, wd=0.1)、warmup + 余弦、梯度裁剪、梯度累积；按领域权重的不放回随机块采样（可精确恢复）；每 45 分钟一段、每 15 分钟检查点；监控 loss/梯度/吞吐/非有限值 |
| 测试评估（evaluator） | `eval_agent.py` | 固定验证块的分领域 loss（主指标按初始配比加权）、12 个固定提示的采样/贪心生成与重复度指标、结束时测试集评估 |

协作方式：`events.jsonl` 是消息总线（发送方 → 接收方、类型：directive / report / decision / proposal），`state.json` 是共享黑板（阶段、计划、最新报告）。`REPORT.md` 每段训练后自动更新。

## 阶段

`selftest → data_acquire → data_merge → tokenizer → encode → model_select → train → final_eval → done`

每个阶段都有回执，中断后从断点继续；截止时间从首次启动起固定 72 小时，重启不会延长。

## 训练中的决策策略

1. 连续 3 步非有限 loss，或验证 loss 高于最佳 5% → 回滚到上一个检查点，学习率 ×0.7；
2. 按实测吞吐预测：跑不完计划就缩短步数；明显富余且进度 < 60% 就延长；余弦终点随之对齐；
3. 预测某领域重复超过 4 轮 → 数据智能体在独立进程续读同一来源补数据（不中断训练）；补不到则降权；
4. 某领域验证 loss 连续两次上升而整体下降 → 该领域权重 ×1.2（上限为初始 1.5 倍）；
5. 到截止或放置 STOP → 保存、最终评估（含测试集）、写报告。

## 命令（在工程根目录 verl_megatron_sglang 执行）

```bash
# 启动（防睡眠；异常退出 2 分钟后用最新代码自动重启）
nohup caffeinate -is bash decoder_from_scratch/multiagent/run_loop.sh >/dev/null 2>&1 &

# 查看状态 / 最近的智能体消息
python -m decoder_from_scratch.multiagent.loop status

# 停止（当前步保存后退出）；之后删除 STOP 文件再执行启动命令即可续跑
python -m decoder_from_scratch.multiagent.loop stop

# 修改代码后让 loop 保存并用新代码重启
python -m decoder_from_scratch.multiagent.loop restart
```

日志：`runs/multiagent_pretrain_v1/logs/loop.log`（全部输出）、`events.jsonl`（智能体消息）、`train_log.jsonl`（每 20 步）、`eval_history.jsonl`、`samples/`（每次评估的生成样例）。

## 使用训练好的模型

`checkpoints/best.pt` 内嵌 tokenizer，可直接用原有推理脚本：

```bash
python -m decoder_from_scratch.infer \
  --checkpoint decoder_from_scratch/runs/multiagent_pretrain_v1/checkpoints/best.pt \
  --device mps --prompt '人工智能的发展' --method sample --top-k 40 --top-p 0.9 --temperature 0.8 --max-new-tokens 100
```

## 调整

在运行目录的 `config.json` 修改（如 `budget_hours`、`data.total_chars`、`data.weights`、`train.segment_minutes`），然后执行 restart。已完成的阶段不会重做；训练计划在 model_select 阶段确定后，只由决策智能体按上面的策略调整。
