# cpu_loop：Weight-Space Programming 复现的 CPU 部分（多智能体 loop）

总体说明和结果见上一级的 [README](../README.md)，本次运行的记录见 [results/weight_space_programming_repro_v1/cpu/](../../results/weight_space_programming_repro_v1/cpu/)。

本项目复现 *Sparks of Weight-Space Programming*（Takehi et al., MIT CSAIL, 2026-11）中不需要 GPU 的实验，用本机的 Codex CLI 调用 `gpt-6-astra`，不调用 API。

| 实验 | 内容 | 被测模型拿到什么 | 本项目状态 |
|---|---|---|---|
| exp1 | 只看 PPO 策略网络的权重，猜它是哪个游戏 | 17,253 个权重的文本，无工具 | ✅ |
| exp2 | 只看 DDPM 的权重，猜它生成什么形状 | 23,824 个权重的文本，无工具，禁止前向 | ✅ |
| exp3 | 只给字符级 LM 的中间 4 个 block，判断语料并找 induction head | Python 沙箱（只能做线性代数，禁止前向），最多 100 次运行 | ✅ |
| exp5 | 从零手写 MinAtar 策略网络的权重 | 游戏源码，无工具，一次回答 | ✅ |
| exp4 | 改权重修好 Kinetix 小车 | — | 下一轮（需要 Kinetix/JAX） |
| exp6 | 从零写 150 万参数的围棋网络 | — | 下一轮（需要 pgx/JAX） |
| exp7/8 | Gemma 谄媚检测 / Qwen 数学 | — | 需要 GPU，不在本项目 |

所有首轮 prompt 都直接取自作者的日志，逐字节一致。评分使用作者的答案键，exp5 用 1000 局 MinAtar 评测，协议与论文相同。

## 多智能体结构

```
                 ┌──────────────── Lead（run.py / wsp/lead.py，确定性 Python）────────────────┐
                 │  规划任务 → 并发派发 → 黑板(state.json/blackboard.json) → 断点续跑 → 分阶段推进  │
                 └───────┬──────────────────────┬──────────────────────────┬────────────────┘
       stage 1: subject  │      stage 2: grade  │                 stage 3: report
   ┌─────────────────────┴───┐   ┌──────────────┴───────────────┐   ┌──────┴───────┐
   │ Subject = gpt-6-astra    │   │ Grader（确定性，作者答案键）     │   │ Reporter     │
   │ codex exec / resume      │   │ Auditor（Codex）逐条审查 exp3    │   │ (Codex)      │
   │ 工具全关、隔离 CODEX_HOME │   │   的代码有没有偷跑前向          │   │ 写 REPORT.md │
   │ 空工作目录、中性系统指令   │   │ Extractor（Codex）关键词规则    │   └──────────────┘
   └───────┬─────────────────┘   │   解析失败时才把自由回答结构化   │
           │ ```python 块         │ MinAtar 评测（多进程）          │
   ┌───────┴────────┐            └──────────────────────────────┘
   │ Sandbox（exp3） │ macOS sandbox-exec：断网，读不到日志/答案键/Codex 登录，只能写临时目录
   └────────────────┘
```

**为什么 Lead 不用 LLM**：它的决策就是实验协议本身（发什么消息、什么时候算一次运行、什么时候结束），必须每次完全一样。会产生判断的环节（审计、解析、写报告）才交给 Codex agent。

**为什么 Subject 要和其他 agent 隔离**：Subject 只能看到作者发给 Astra 的原始消息。辅助 agent 各自开新会话，Subject 的上下文里不会混进任何别的东西。

**Auditor 的两种模式**（`config.toml` → `[helpers] audit_mode`）：
- `flag`（默认）：事后审计，和作者的做法一致（作者逐条读了每次运行）。
- `gate`：审计不通过就不执行，并告诉 Subject 原因。这会改变协议，论文里没有这样做。

## 决策智能体模式（Claude 在对话里当 Lead）

```bash
.venv/bin/python run.py --daemon     # 在 Mac 终端里启动一次，保持窗口开着
```

daemon 监视 `runs/control/inbox/`。Claude 往里写指令（只允许 check / plan / run / set / stop，`set` 只能改白名单里的参数），daemon 在 Mac 上用你的 Codex 执行，把结果写进 `runs/control/outbox/`，并定时更新 `heartbeat.json`。每个里程碑由 Claude 读完结果再决定下一步。

## 使用步骤（Mac）

```bash
# 0. 前提：Python ≥ 3.11、git，Codex CLI 已登录且账号能用 gpt-6-astra
codex login

# 1. 一次性安装：克隆作者日志（约 370MB），建两个 venv，然后自检
./setup.sh

# 2. 不花额度的离线测试：假 codex 回放 Astra 的原始回答，验证 prompt 逐字节一致、评分能复现论文数字
./tests/replay_test.sh

# 3. 冒烟测试：只跑一次真实调用（exp1 Breakout，约 7.4 万输入 token，1–2 分钟）
.venv/bin/python run.py --exp exp1 --unit exp1=breakout

# 4. 全部跑完（可以随时 Ctrl-C，重新运行同一条命令会从断点继续）
.venv/bin/python run.py
```

其他常用命令：

```bash
.venv/bin/python run.py --check                       # 检查 Codex、登录、数据、沙箱隔离是否生效
.venv/bin/python run.py --plan                        # 看计划和实际的 codex 命令行，不执行
.venv/bin/python run.py --exp exp3 --unit exp3=python_r0 exp3=romaji_r0
.venv/bin/python run.py --stage grade --regrade       # 只重新评分
.venv/bin/python run.py --stage report                # 只重写报告
```

想在每个阶段之间停下来确认：把 `config.toml` 里的 `gate_between_stages` 设为 `true`。

## 输出

```
runs/
  lead.log                  Lead 的完整日志（每轮、每次运行、每次重试）
  blackboard.json           每个单元的当前状态
  exp3/python_r0/
    turns/t001/             prompt.txt / events.jsonl / answer.md / run.json（含 tool_free 审计）
    tool_calls.jsonl        沙箱里每次运行的代码和输出
    final_answer.md
    grade.json              评分结果 + Auditor 的逐条审计
    helpers/                Auditor / Extractor 的原始对话
  results.md                论文数字和本次结果的对照表（确定性生成）
  REPORT.md                 Reporter 写的复现报告
```

## 时间和额度（按作者日志估算）

| 实验 | 会话数 | 推理强度 | 原始耗时 | 输入 token |
|---|---|---|---|---|
| exp1 | 4 | high | 每个 1–2 分钟 | 每个约 7.4 万 |
| exp2 | 3 | high | 每个 1–2 分钟 | 每个约 10 万 |
| exp3 | 6 | xhigh | 约 20 分钟（6 个并行） | 共约 160 万（大部分命中缓存） |
| exp5 | 3 | xhigh | 每个约 9 分钟 | 每个约 6 千 |

`concurrency` 默认是 2，以免撞到订阅额度上限。撞到上限时，loop 会按 `retry_waits` 等待后重试，在同一会话里继续，不会丢进度。

## 已知差异与注意事项

- **单次结果有随机性**：论文每个单元只跑 1–2 次。想得到可信的复现率，可以把同一单元多跑几次（复制 unit 目录名，或者以后加 `repeats` 参数）。
- **exp5 Breakout**：直接评测作者给出的网络，我们 1000 局得到 503 ± 7.7，论文是 517 ± 7.4，相差约 1.4 个标准误。Seaquest 和 Asterix 与论文一致。setup.sh 会尝试装官方 minatar 1.0.15；装上后会用官方实现，否则用 prompt 里引用的源码。
- **沙箱**：macOS 用 `sandbox-exec`，Linux 用 bubblewrap。两者都没有时，exp3 会拒绝运行，因为 Subject 的代码可能读到答案键。`run.py --check` 会实际验证"读不到答案键、连不上网络"。
- **网络**：`config.toml` 的 `[codex] http_only = true`（默认）让隔离的 CODEX_HOME 只走 HTTPS，避免 WebSocket 反复超时；你自己 `~/.codex/config.toml` 里的 provider / 代理设置（`inherit_network = true`）会被带过来。
- **Codex 版本**：作者的 `codex_config.toml` 里有一些 feature 开关，你的 Codex 版本未必都认识。不认识的键一般会被忽略。如果你的版本支持 `--strict-config`，可以加到 `[codex] extra_flags`。
- exp4、exp6 下一轮再加：需要 Kinetix 和 pgx（都基于 JAX，能在 CPU 上跑），涉及评测环境的重建。
