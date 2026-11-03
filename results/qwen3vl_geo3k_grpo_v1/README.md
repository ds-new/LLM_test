# Qwen3-VL-4B geometry3k GRPO：运行记录

2026-11-01 至 2026-11-03 在 AutoDL 2 × A800 80GB 上的一次完整运行（多智能体 GRPO 两轮 + 诊断实验 + 两版拒绝采样微调）。说明见 [`qwen3vl_geo3k_grpo/README.md`](../../qwen3vl_geo3k_grpo/README.md)。不含数据和模型权重。

| 文件 | 内容 |
| --- | --- |
| `REPORT.md` | 多智能体 loop 自动生成的报告：结果、数据清洗统计、每轮诊断与验证曲线、决策记录 |
| `eval_summary.csv` | 全部 16 次离线评估的汇总：模型、数据划分、采样数、温度、生成长度、准确率、mean@k / maj@k / pass@k、boxed 率、截断率、失败类型 |
| `train_metrics_round1.csv` / `train_metrics_round2.csv` | 每步训练指标：训练奖励、熵、回答长度、截断率、梯度范数、每步耗时、训练中验证准确率（`attempt` 是第几次启动，断点续训后同一步会出现多次） |
| `plans.json` | 两轮的训练计划：数据、算法、奖励、显存档位 |
| `final.json` | 最终结论：最佳轮次、与基线的逐题对比、test 多数投票、用 verl 官方 geo3k 奖励重算的分数 |
| `rft_progress.log` | 两版拒绝采样微调流水线的日志（样本挑选统计、训练与评估结果） |
| `figures/` | test 准确率对比、第 2 轮训练曲线、第 2 轮训练中验证点 |
