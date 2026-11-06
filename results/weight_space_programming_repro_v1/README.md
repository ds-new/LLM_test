# Weight-Space Programming 复现：运行记录

2026-11-05 至 2026-11-06 的一次完整复现，分两部分：

- **CPU 部分**：在 Mac 上用本机 Codex CLI（`gpt-6-astra`）重跑原文的实验 1、2、3、5，与作者的 prompt 逐字节相同，用作者的答案键评分。
- **M6**：在 AutoDL（1 × RTX 4080 SUPER 32GB）上逐位重建作者对 Qwen3.5-0.8B 的权重编辑，按原文协议重测 MATH-500。

说明见 [`weight_space_programming_repro/README.md`](../../weight_space_programming_repro/README.md)。不含数据集和模型权重（实验 5 中 Astra 写出的三个 MinAtar 小网络除外）。

| 目录 | 内容 |
| --- | --- |
| [`cpu/`](cpu/) | 实验 1/2/3/5 每个单元的对话、代码运行、最终回答、评分与审计；汇总报告 `REPORT.md`、对照表 `results.md`、`summary.json`、Lead 日志 |
| [`m6/`](m6/) | Qwen3.5-0.8B 编辑的评测结果：`RESULTS.md`（与论文对照）、`m6_summary.json`（版本、哈希校验、协议、耗时） |

## 结果

| 实验 | 论文 | 本次 |
| --- | --- | --- |
| 8：原模型 MATH-500 | 7.9%（截断 93.0%） | 8.2% ± 0.6（截断 92.3%） |
| 8：编辑 p50c 后 MATH-500 | 48.1%（截断 33.1%） | **50.5% ± 1.1**（截断 32.4%） |
| 1：只看权重认游戏 | 4/4 | 4/4 |
| 2：只看权重认形状 | 3/3 | 3/3（同样画成空心轮廓） |
| 3：判断训练语料 / 找 copy head | 6/6 / 6/6 | 4/6 / 5/6 |
| 5：从零写策略 Breakout / Seaquest / Asterix | 517.4 / 53.1 / 53.2 | 502.2 / 33.9 / 50.8（作者网络同一评测器下 502.9 / 54.3 / 53.9） |
