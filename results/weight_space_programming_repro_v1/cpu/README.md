# CPU 部分：实验 1/2/3/5 的运行记录

由 [`cpu_loop`](../../../weight_space_programming_repro/cpu_loop/) 在 Mac 上用本机 Codex（`gpt-6-astra`，作者的 Codex 配置，工具全关）产生。路径中的本机目录已替换为 `<wsp_repro>`。作者的原始首轮 prompt 不重复存放，见 [RikiyaT/lm_understand-logs](https://github.com/RikiyaT/lm_understand-logs)。

| 文件 | 内容 |
| --- | --- |
| `REPORT.md` | Reporter 智能体写的复现报告，末尾是决策智能体对实验 5 可比基线的更正 |
| `results.md` | 每个单元论文数字与本次结果的对照表（确定性生成） |
| `summary.json` | 同上的机器可读版本 |
| `lead.log` | Lead 的完整日志：每轮调用、每次代码运行、重试、评分 |
| `blackboard.json` | 每个单元的最终状态（轮数、代码运行次数、是否全程无工具调用） |
| `exp1/<游戏>/`、`exp2/<形状>/` | `final_answer.md`（Astra 的回答）、`grade.json`（评分与置信度）、`turns/t001/`（`run.json` 含命令、耗时、token 用量与工具审计，`events.jsonl` 含推理摘要） |
| `exp3/<语料>_r<0/1>/` | 多轮对话：`turns/tNNN/`（第 2 轮起的 `prompt.txt` 是沙箱返回的输出），`tool_calls.jsonl`（每次代码运行的代码与输出），`grade.json`（语料、copy head 与 50 次运行的审计结果），`helpers/`（Auditor 的逐条审计） |
| `exp5/<游戏>/` | `final_answer.md`（权重编辑语句），`network.npz`（由回答逐位重建的网络），`grade.json`（官方 MinAtar 1.0.15 下 1000 局的分数） |
