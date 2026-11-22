# R1-Zero 数学推理 RL 的原始记录（Qwen3-8B-Base，2026 年 11 月）

对应的代码和说明见 [r1zero_qwen3_8b_rl/](../../r1zero_qwen3_8b_rl/README.md)。不含模型权重、数据集和模型的逐条输出。

| 文件 | 内容 |
|---|---|
| REPORT.md、REPORT_round2.md、REPORT_round3.md | 各轮自动生成的运行报告：环境、数据、每步指标、评选分、完整评测、曲线 |
| audit/STAGE_B_RESULT.md、STAGE_lr2dyn_RESULT.md、STAGE_frontier_RESULT.md | 第三轮 B / C / D 阶段的自动结论（同题配对 bootstrap，事先定好的判定标准） |
| audit/AUDIT.md、verifier_summary.json、contamination_summary.json、rescore_v2.json | 判分器审计、训练/评测泄露检查、v2 重新判分 |
| audit/frontier_stats.json | D 阶段边界题的筛选统计 |
| audit/STAGE_E_RESULT.md、E/ | 蒸馏 SFT（E）的完整评测和结论 |
| audit/STAGE_F1_RESULT.md、STAGE_F1_RESULT.json、F/ | 在 SFT 模型上做 RL（F 第一阶段，项目最终模型）的完整评测和结论 |
| audit/E_PRESENCE_PENALTY_RESULT.md、F_PRESENCE_PENALTY_RESULT.md | 推理时加 presence_penalty 的试验 |
| audit/f_frontier_stats.json、f_val_baseline_*.json | F 的选题统计、验证基线 |
| metrics/e_stage.log、f_stage.log | E、F 阶段的运行日志（启动、崩溃、修复、决策） |
| metrics/metrics_r1zero_qwen3_8b_dapo.jsonl | 每步训练指标（verl 原始输出：奖励、长度、截断率、熵、裁剪比例、梯度范数、各环节耗时、生成端与训练端概率差异） |
| metrics/metrics_*_rollback_*.jsonl、metrics_*_badrun_*.jsonl | 回滚前、崩溃前的那几段记录，保留用于复核 |
| metrics/metrics_final_eval_*.jsonl | 各次完整评测的汇总指标 |
| metrics/events.jsonl | 多智能体流水线的事件和决策日志（启动、崩溃分类、换显存档位、新最佳、等待 Claude 判断 …） |
| final_eval/*.csv | 每次完整评测里每道题的采样次数和答对次数（v2 判分），可以直接复现配对 bootstrap |
| figures/curves_round2.png、curves_round3.png | 第二、三轮训练曲线 |
