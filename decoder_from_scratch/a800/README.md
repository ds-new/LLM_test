# A800 阶段代码

单卡 A800（AutoDL）上的多智能体预训练流程和这一阶段的全部实验。设计、结果和经验见上一级 [README 的 A800 阶段](../README.md#a800-阶段)；每个实验的报告、日志和图在 [results/a800/](../../results/a800/)；设计文档和过程记录在 [docs/a800_plan/](../docs/a800_plan/)。

## 模块

| 文件 | 作用 |
|---|---|
| loop.py | 入口：run / status / stop / restart；按运行目录 config.json 的 mode 走预训练或继续预训练流程 |
| common.py | 路径、配置、原子读写、消息总线 events.jsonl、黑板 state.json、控制文件（STOP / RESTART）与心跳 |
| decision_agent.py | 决策智能体：编排另外五个智能体、审核报告、调整计划、回滚、调配比，每条决策写明理由 |
| data_agent.py、cleaning.py、sources.py | 数据智能体：ModelScope 数据源、采样、训练分词器、流式"下载-清洗-去重-划分-编码" |
| architect_agent.py、bench_worker.py | 结构智能体：结构自检、候选结构吞吐基准（每个候选在独立进程测）、选规模 |
| trainer_agent.py | 训练智能体：WSD 调度、AdamW、按来源权重的可恢复采样、检查点 |
| eval_agent.py | 评估智能体：固定验证块的分来源 loss、选择题、固定提示的生成与重复率 |
| analyst_agent.py | 结果分析智能体：显著性判定（improving / plateau / regressing）、剩余时间、最终分析和下一步建议 |
| smoke.py、selfcheck.py | 开训前的端到端冒烟测试、部署后的快速自检 |
| passk_probe.py、token_freq_probe.py、weight_edit.py、profile_probe.py、cloze_stats.py | 训练后的探测：pass@k、欠训练 token 与污染、参数空间编辑、profiler 与 MFU、选择题的置信区间 |
| tok_ab.py、run_tok_ab.sh | 分词器对照（v1 vs v2） |
| ablate.py、run_ablate.sh | 第一批消融：学习率扫描 + 多种子方差 |
| data_eng.py、global_dedup.py | 数据工程统计（MinHash 近似去重、评测污染、文档长度）和第 2 段去重的执行端 |
| cpt.py、cpt_launch.py | 路线 A：在 v1 上继续预训练、退火探针、扩上下文；各路线之间的衔接与冒烟 |
| lab.py、lab_core.py、lab_data.py、lab_selftest.py、run_lab.sh | 实验室：μP、scaling law、数据对比、学习率调度、Muon / SOAP / Schedule-Free |
| mech.py | 机理分析与推理侧实验（attention sink、logit lens、消融、induction head、探针、因果追踪、解码、量化） |
| tokenizer_hf.py | 把自训的 BBPE 等价转换成 HuggingFace tokenizers 格式，用于快速编码 |
| run_loop.sh | 守护脚本：可重试错误自动重启、卡死检测、RESTART 平滑重启、结束后按规则关机 |
| autodl/setup.sh | AutoDL 上一键建 venv、检查依赖、自检并启动 |

## 运行

所有命令在仓库根目录执行。默认路径是 AutoDL 实例上的位置，换机器时用环境变量覆盖：A8_RUN_DIR（运行目录）、A8_SCRATCH_DIR（检查点）、LAB_RUN_DIR / LAB_SCRATCH / LAB_CKPT（实验室）、MECH_OUT（机理分析输出）。

```bash
bash decoder_from_scratch/a800/autodl/setup.sh                    # 建环境、自检并启动 v1 预训练

# 预训练 / 路线 A：守护脚本后台运行（路线 A 的运行目录 config.json 里 mode 为 cpt）
A8_RUN_DIR=<运行目录> PYTHON=<venv 的 python> setsid nohup bash decoder_from_scratch/a800/run_loop.sh > /dev/null 2>&1 < /dev/null &
A8_RUN_DIR=<运行目录> python -m decoder_from_scratch.a800.loop status   # 进度和最近的智能体消息
# 控制：运行目录里放 STOP（存档后停下）、RESTART（存档后用新代码重启）、NO_SHUTDOWN（结束后不关机）

python -m decoder_from_scratch.a800.lab_selftest cpu               # 实验室自测（gpu 模式需要显卡）
LAB_RUN_DIR=<目录> PYTHON=<venv 的 python> bash decoder_from_scratch/a800/run_lab.sh

python -m decoder_from_scratch.a800.mech sinks lens layers induction probes trace decode quant heads
```
