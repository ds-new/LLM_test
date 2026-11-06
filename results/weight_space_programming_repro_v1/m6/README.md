# M6：Qwen3.5-0.8B 权重编辑的评测结果

由 [`gpu_m6`](../../../weight_space_programming_repro/gpu_m6/) 在 AutoDL（1 × RTX 4080 SUPER 32GB，vLLM 0.33.0）上产生。模型为 Qwen/Qwen3.5-0.8B @ `2fc0636`；作者的三个编辑重建后每个张量的 sha256 都与作者 manifest 一致。

| 文件 | 内容 |
| --- | --- |
| `RESULTS.md` | 原模型与 p50c 在 MATH-500 上的准确率、截断率、平均长度，与论文对照 |
| `m6_summary.json` | 硬件、软件版本、模型来源与哈希校验、评测协议、详细数字、时间线、未跑的项目 |

逐题评分（`math500_per_problem.json`）和每题第一条回答的全文仍在 AutoDL 数据盘 `/root/autodl-tmp/wsp_m6/eval/` 上，未随仓库发布。
