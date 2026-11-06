# gpu_m6：验证作者对 Qwen3.5-0.8B 的权重编辑（实验 8，AutoDL）

总体说明和结果见上一级的 [README](../README.md)，本次结果见 [results/weight_space_programming_repro_v1/m6/](../../results/weight_space_programming_repro_v1/m6/)。不需要 Codex。

1. 下载 Qwen/Qwen3.5-0.8B（revision `2fc0636`，作者编辑时用的版本）。大文件优先从 ModelScope 下载，大小必须与 HF 上的这个版本一致，否则退回 hf-mirror。
2. 用作者 `apply_edit.py` 的逻辑逐位重建 p50c / p10c / p25c；每个张量的 base 和 edited sha256 都要与作者 manifest 一致，否则直接报错。
3. 按论文协议评测（thinking 开，temperature 0.6 / top_p 0.95 / top_k 20 / presence_penalty 1.5，32,768 tokens，MATH-500 每题 4 次、GSM8K 每题 2 次，math-verify 判分），与论文对照。

```bash
# 在 AutoDL 实例上（代码放数据盘）
mkdir -p /root/autodl-tmp/wsp_gpu && cp -r gpu_m6 /root/autodl-tmp/wsp_gpu/m6
cd /root/autodl-tmp/wsp_gpu/m6
setsid nohup bash run_m6.sh > /dev/null 2>&1 < /dev/null &
tail -f /root/autodl-tmp/wsp_m6/progress.log          # 进度
cat /root/autodl-tmp/wsp_m6/RESULTS.md                # 结果（核心两项跑完就会写出）
```

- 顺序：环境检查 → 下载模型与数据 → 重建 → base/MATH-500 → p50c/MATH-500 → base/GSM8K → p50c/GSM8K → p10c、p25c/MATH-500。每一步可断点续跑；`BUDGET_HOURS`（默认 6）只限制核心两项之后的可选步骤。
- vLLM 不支持 Qwen3.5 时会自动 `pip install -U vllm`（本次装上 0.33.0）。升级后 `huggingface_hub` 可能因 httpx2 版本冲突无法下载，所以下载全部用 curl。
- 对国内源（hf-mirror、ModelScope）不走 AutoDL 学术加速代理。
- **自动关机**：`m6_watchdog.sh` 在 `run_m6.sh` 停止 10 分钟后关机（正常结束、出错、崩溃、被杀都算），数据盘保留。不想关机：`touch /root/autodl-tmp/wsp_m6/KEEP_RUNNING`；完全关闭：启动时加 `SHUTDOWN_WHEN_DONE=0`。
- 干跑（不需要 GPU，检查评分与汇总）：`python3 eval35.py <ckpt> <out> --sets math500 --bench <dir> --dry`。
