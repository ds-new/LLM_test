# Weight-Space Programming 复现

复现 MIT CSAIL 的博客 [*Sparks of Weight-Space Programming*](https://rikiyat.github.io/blog/weight-space-programming/)（2026-11）：一个前沿大模型（GPT-6 Astra）不经过任何训练，直接读懂、改写、甚至从零写出神经网络的权重。

原文作者在 [RikiyaT/lm_understand-logs](https://github.com/RikiyaT/lm_understand-logs) 公开了全部 prompt、对话记录、权重和答案键。本复现分两部分：

- **CPU 部分**（实验 1、2、3、5）：用本机的 Codex CLI 调 `gpt-6-astra`（不用 API），由一个多智能体 loop 发起和原文逐字节相同的对话，再用作者的答案键评分。
- **GPU 部分 M6**（实验 8）：在 AutoDL 上逐位重建作者对 Qwen3.5-0.8B 的权重编辑，按原文协议重新评测 MATH-500。

## 结果一览

### 实验 8（M6）：13 万个权重让 Qwen3.5-0.8B 的 MATH-500 从 8% 涨到 50%

作者让 Astra 在 50 次前向预算内修改 Qwen3.5-0.8B 的权重（不允许梯度、不允许训练），得到编辑 p50c：只改了第 15 层和第 19 层 MLP 的 `down_proj`，每层 64 列（131,072 个值，占全模型约 0.016%），把这些神经元写入残差流的"重复方向"分量反转。

| | 论文 | 本复现 |
|---|---|---|
| 原模型 MATH-500 | 7.9% | **8.2% ± 0.6** |
| p50c MATH-500 | 48.1% | **50.5% ± 1.1** |
| 原模型回答写到 32K 上限的比例 | 93.0% | 92.3% |
| p50c 回答写到 32K 上限的比例 | 33.1% | 32.4% |
| p50c 平均回答长度 | 22,037 token | 21,997 token |

- 模型版本和作者完全相同（Qwen/Qwen3.5-0.8B @ `2fc0636`）。三个编辑（p10c、p25c、p50c）重建后，每个张量的 sha256 都与作者的 manifest 一致。
- 评测协议同原文：开 thinking，temperature 0.6 / top_p 0.95 / top_k 20 / presence penalty 1.5，最多 32,768 token，每题 4 次采样，math-verify 判分。
- 原模型的低分是真实的：抽查写到上限的回答，结尾都在原地打转（"Wait, I'm seeing 196 in the list." 重复几十遍）。社区质疑的"基线偏低"，在这个协议下并不是评测写错了。
- 这次复现**能确认的**：作者的编辑确实带来这个提升。**还不能确认的**：提升来自"会做数学"还是只是"不再打转"。Astra 自己的报告也指向后者：它救回的 579 道练习题里，576 道原本都被截断。下一步的对照实验（只改解码方式、不改权重）见文末。

### CPU 实验：读懂与从零写出权重

| 实验 | 内容 | 论文 | 本复现 |
|---|---|---|---|
| 1 | 只看 PPO 策略网络的 17,253 个权重，判断是哪个游戏 | 4/4 | **4/4**，置信度 95–97%（论文 96–99%） |
| 2 | 只看 DDPM 的权重，判断生成什么形状 | 3/3，画成空心轮廓 | **3/3**，同样画成空心轮廓（实际是实心），置信度 84/72/88%（论文 84/77/88%） |
| 3 | 只给字符级 LM 中间 4 个 block，判断训练语料 | 6/6 | 4/6（模型给的概率普遍只有 0.5 左右） |
| 3 | 同上，找出 copy（induction）head | 6/6 完全一致 | **5/6** 完全一致；另 1 个多报一个头，没有漏 |
| 5 | 从零写 MinAtar 策略网络：Breakout | 517.4 | **502.2 ± 5.7**（作者网络在同一评测器下 502.9） |
| 5 | Seaquest | 53.1 | 33.9 ± 1.0（作者网络同一评测器下 54.3；仍是 PPO 1 亿步的 6 倍） |
| 5 | Asterix | 53.2 | **50.8 ± 0.9**（作者网络同一评测器下 53.9） |

- 实验 5 的 517 / 53 / 53 是作者自己评测种子下的分数。为了公平比较，作者发布的网络也放进同一个评测器重新打了分（官方 MinAtar 1.0.15，1000 局，sticky 0.1）。
- 所有回答都没有调用工具。实验 3 的 50 次代码运行全部经过审计，没有一次偷跑前向。
- 每个单元只跑了一次，所以 Seaquest 偏低和实验 3 的语料误判，都可能是单次采样的波动。

完整报告：[results/weight_space_programming_repro_v1/cpu/REPORT.md](../results/weight_space_programming_repro_v1/cpu/REPORT.md)，M6 结果：[results/weight_space_programming_repro_v1/m6/](../results/weight_space_programming_repro_v1/m6/)。

## 目录

```
weight_space_programming_repro/
  cpu_loop/            实验 1/2/3/5 的多智能体复现 loop（在 Mac 上用本机 Codex 运行）
    run.py             入口：--check / --plan / --daemon / --exp / --stage
    config.toml        模型、推理强度、并发、沙箱、评测参数
    wsp/
      lead.py          Lead：规划、并发派发、黑板与断点续跑、三个阶段（作答 → 评分 → 报告）
      codex.py         Codex 会话封装：codex exec / resume、隔离的 CODEX_HOME、重试、取消、工具审计
      experiments.py   各实验的适配器：原文 prompt、对话流程、作者答案键评分、论文数字
      roles.py         辅助智能体：Auditor（审计代码运行）、Extractor（结构化抽取）、Reporter（写报告）
      sandbox.py       实验 3 的代码沙箱（macOS sandbox-exec / Linux bubblewrap：断网、读不到答案键）
      edits.py         实验 5 的权重编辑语言解析器（逐位重建作者网络）
      minatar_eval.py  实验 5 的 MinAtar 评测（官方 1.0.15，或 prompt 里引用的源码）
      daemon.py        决策智能体模式：通过 runs/control/inbox 接收指令（check/plan/run/set/probe/restart/stop）
    tests/             假 codex：回放作者记录的回答，离线验证 prompt 逐字节一致、评分复现论文
  gpu_m6/              实验 8 的验证（在 AutoDL 上运行）
    run_m6.sh          总流程：环境 → 下载 → 重建 → 评测 → RESULTS.md，可断点续跑
    m6_watchdog.sh     脚本停止 10 分钟后自动关机（正常结束、出错、崩溃、被杀都算）
    fetch.py           curl 下载模型（大文件优先 ModelScope）与 MATH-500 / GSM8K
    build.py           用作者的 apply_edit 逐位重建 p10c / p25c / p50c，并校验 sha256
    eval35.py          论文的评测协议（think35）
    compare.py         与论文数字对照
    authors/           作者的代码（MIT）、编辑文件与论文数字（CC BY 4.0），见下方许可
results/weight_space_programming_repro_v1/
  cpu/                 每个单元的对话、代码运行、最终回答、评分、审计；REPORT.md、summary.json、lead.log
  m6/                  RESULTS.md、m6_summary.json
```

## 多智能体 loop

```
        决策智能体（Claude，在对话里）
              │  写指令 runs/control/inbox/*.json，读 outbox/、heartbeat.json、lead.log
              ▼
   ┌──────── Lead（run.py --daemon，确定性 Python）────────┐
   │ 规划 → 并发派发 → 黑板/断点续跑 → 作答 → 评分 → 报告      │
   └──┬─────────────────────┬────────────────────┬─────────┘
      │                     │                    │
 Subject = gpt-6-astra   Grader（作者答案键）     Reporter（Codex）
 codex exec / resume     Auditor（Codex）逐条      写 REPORT.md
 工具全关、隔离 CODEX_HOME  审计实验 3 的代码运行
 空工作目录、中性系统指令   Extractor（Codex）在规则
      │ ```python           解析失败时结构化回答
   Sandbox（实验 3）
```

- **Lead 是确定性代码**：它的决策就是实验协议（发什么消息、什么算一次运行、什么时候结束），必须每次完全一样。需要判断的环节（审计、抽取、写报告）才交给 Codex 智能体。
- **Subject 与其他角色隔离**：只看得到作者当年发给 Astra 的原始消息。辅助智能体每次调用都是新会话，Subject 的上下文里不会混进别的内容。
- **决策智能体**：Claude 在对话里读每个里程碑的结果（M0 自检 → M1 单次调用 → M2 实验 1/2 → M3 实验 5 → M4 实验 3 → 报告），再决定下一步。daemon 只接受白名单动作，`set` 只能改白名单参数，不能执行任意命令。

## 快速开始

### CPU 部分（Mac，本机 Codex）

前提：Python ≥ 3.11、git，Codex CLI 已登录，且账号能用 `gpt-6-astra`。

```bash
cd weight_space_programming_repro/cpu_loop
bash setup.sh                                  # 克隆作者日志（约 370MB）、建两个 venv、自检
./tests/replay_test.sh                         # 离线回放测试，不花额度
.venv/bin/python run.py --exp exp1 --unit exp1=breakout   # 冒烟：一次真实调用
.venv/bin/python run.py                        # 全部（可随时 Ctrl-C，重跑同一条命令从断点继续）
.venv/bin/python run.py --daemon               # 或：交给决策智能体通过 inbox 驱动
```

额度参考（按作者日志）：实验 1 每个约 7.7 万输入 token，实验 2 约 10 万，实验 3 共约 160 万（多数命中缓存），实验 5 每个约 3 万。

### GPU 部分 M6（AutoDL）

```bash
cd /root/autodl-tmp && tar xzf wsp_gpu.tgz && cd wsp_gpu/m6     # 或把 gpu_m6/ 放到 /root/autodl-tmp/wsp_gpu/m6
setsid nohup bash run_m6.sh > /dev/null 2>&1 < /dev/null &
tail -f /root/autodl-tmp/wsp_m6/progress.log
cat /root/autodl-tmp/wsp_m6/RESULTS.md
```

- 显存：0.8B 模型，24GB 以上的卡都能跑；本次用的是 32GB 的 4080 SUPER，原模型 MATH-500 用了 8.5 小时，p50c 约 6.5 小时（原模型 92% 的回答写满 32K token，总共约 6100 万 token）。
- 跑完或异常停止 10 分钟后自动关机。不想关机：`touch /root/autodl-tmp/wsp_m6/KEEP_RUNNING`。

## 踩过的坑

| 现象 | 原因 | 处理 |
|---|---|---|
| Codex 每次请求先"Reconnecting 5/5 (request timed out)"再成功，大 prompt 直接失败 | 国内网络下 WebSocket 连接超时，要失败 5 次才退回 HTTPS | 隔离的 CODEX_HOME 里定义一个只走 HTTPS 的 ChatGPT provider（`supports_websockets = false`），单次调用从 120 秒降到 10 秒 |
| 长回答中途 "stream disconnected" | 网络不稳 | Codex 自动重连续上；但 token 计数只记最后一次请求，用量不能直接和论文比 |
| 沙箱里 `import numpy` 失败 | 把 venv 的 python 软链接 `resolve()` 成了 miniforge 本体，跳出了 venv | 改用 `abspath` |
| hf-mirror 下载停在 603MB | AutoDL 学术加速代理（环境变量里的 `http_proxy`）拖住了国内镜像 | 对国内源不走代理 |
| `huggingface_hub` / `datasets` 报 `process() takes no keyword arguments` | `pip install -U vllm` 连带升级的 httpx2 与解压库不兼容 | 改用 curl 下载 |
| hf-mirror 直连只有 0.5MB/s | — | 大文件优先 ModelScope（约 4MB/s），大小必须与 HF 版本一致，之后逐张量校验 sha256 |
| 官方 MinAtar 报 `unexpected keyword argument 'random_seed'` | 1.0.15 的接口是先构造再 `seed()` | 已改；官方实现与 prompt 里的源码在同样种子下分数逐位相同 |

## 未完成与下一步

- **对照实验（推荐，约 3–4 小时 GPU）**：不改权重，只改解码方式（关 thinking、更强的防重复惩罚），看原模型能到多少分。如果也接近 50%，p50c 本质上是写进权重的"防循环开关"；如果做不到，说明在权重里精确定位循环方向比解码时处理更有效。
- **M7：让 Astra 自己改一遍**（约 1–1.5 天）：本机 Codex 驱动，AutoDL 上跑前向服务，50 次前向预算。作者没有公开这部分 harness，需要按 `runs/p50c/turns/t001/prompt.txt` 的规则实现。
- 实验 8 的 GSM8K、p10c、p25c 和其他基准没有跑；实验 4（Kinetix）、6（围棋，pgx）、7（Gemma 谄媚）没有做。
- 每个单元只跑了一次。要估计复现率，应对 Seaquest 和实验 3 的误判条件重复 3–5 次。

## 许可与致谢

- 原文与全部原始数据：Takehi et al., *Sparks of Weight-Space Programming*，[RikiyaT/lm_understand-logs](https://github.com/RikiyaT/lm_understand-logs)。
- `gpu_m6/authors/` 下的 `mathqa.py`、`apply_edit.py`、`st_numpy.py` 是作者的代码（MIT）；`edits/` 和 `paper/` 是作者的编辑文件和评测结果（CC BY 4.0）；许可原文见 [gpu_m6/authors/LICENSE](gpu_m6/authors/LICENSE)。`mcqa.py` 是本复现写的替身（作者的原文件未公开，评分不经过它）。`Qwen3.5-0.8B-text/` 是 Qwen 模型的配置与 chat template。
- 实验 5 prompt 里引用的 MinAtar 源码（GPL-3.0）和 MATH-500、GSM8K 题目不随本仓库发布，运行时从原处下载。
- 本复现的结果记录（`results/weight_space_programming_repro_v1/cpu/`）包含 Astra 在相同 prompt 下的新回答；作者的原始 prompt 不重复存放，见作者仓库。
