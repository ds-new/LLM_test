# A800 阶段的实验记录

每个目录是一次运行的完整记录（配置、智能体消息、训练与评估日志、报告、图）。没有上传的：模型权重和检查点（v1 的 final.pt 约 1.9GB）、token 分片、下载的原始数据。代码在 [decoder_from_scratch/a800/](../../decoder_from_scratch/a800/)，总体说明在 [decoder_from_scratch/README.md](../../decoder_from_scratch/README.md#a800-阶段)。

| 目录 | 内容 | 先看 |
|---|---|---|
| v1_pretrain/ | 0.48B、25B token 的 v1 基座：配置、47,683 步的训练日志、每次评估、智能体决策、生成样例、分词器，以及训练后的 pass@k、欠训练 token、参数编辑、profiler 探测 | REPORT.md、analysis/FINAL_ANALYSIS.md |
| tokenizer_ab/ | 分词器对照（v1 vs v2，1.25 亿参数模型）：两组训练日志、结果和 v2 分词器 | REPORT.md |
| ablation_lr_seed/ | 第一批消融：学习率 5e-4 / 1e-3 / 2e-3 / 4e-3 和 3 个种子 | REPORT.md、stage1_summary.json |
| data_eng/ | 数据工程统计：跨文件近似重复、评测污染、文档长度、第 2 段去重决定 | REPORT.md |
| route_a_cpt/ | 路线 A（在 v1 上继续预训练），上传时进行中：第 1 段训练与评估日志、两次退火探针、决策记录 | REPORT.md、events.jsonl 里的 midcheck 决策 |
| mech/ | v1 的机理分析与推理侧实验：sink、logit lens、逐层 / 逐头消融、induction head、探针、因果追踪、解码策略、量化 | REPORT.md、figs/ |
| lab_selftest/ | 实验室代码的 CPU / GPU 自测日志 | gpu2.log |

常见文件：config.json（运行配置）、state.json（黑板：阶段、计划、最新报告）、events.jsonl（智能体之间的全部消息和决策理由）、train_log.jsonl（每 20 步）、eval_history.jsonl（每次评估）、samples/（每次评估的生成样例）、logs/（守护脚本和 loop 的日志）。
