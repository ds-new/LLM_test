# 判分器与数据泄露审计（2026-11-16）

数据：/root/r1zero_valgen 下 step 40–56 的验证生成，共 3,200 条（val_select，400 条/步）。
脚本：ma/audit_verifier.py、ma/audit_rescore.py、ma/audit_contam.py、ma/audit_leak_impact.py；修正版判分 ma/reward_v2.py（训练仍在用 ma/reward.py）。

## 1. 判分器（ma/reward.py）

**误判（错误答案判成对）：未发现。**
- 102 条只靠符号等价判对的，逐类核对，都是等价写法：5.5 对 11/2，C 对 \text{(C)}，2516 对 2516_8。
- 356 条结尾有多个 Answer 行的，都是 "**Final Answer:**" 标题加一行 "Answer: X"，没有押多个答案。

**漏判（正确答案判成错）：76 / 3,200（2.4%），v2 全部修正，没有引入新误判。**

| 原因 | 例子 |
|---|---|
| 答案行带 markdown/LaTeX 包装 | `**Answer:** \(\frac{11}{2}\)`、`**Answer: 1 + 2i**` |
| "Answer:" 后换行写 \boxed，提取到空串 | AIME24 的 116、33；AMC23 的 7、9、13、194、265 |
| 文字答案被当成变量 | east、Navin、ellipse |
| 区间、复数、带分数格式 | `(5, \infty)`、`137½` |

**对分数的影响（挑选分数，v1 → v2）：**

| step | v1 | v2 |
|---|---|---|
| 40 | 0.5825 | 0.6125 |
| 42 | 0.580 | 0.605 |
| 44 | 0.5825 | 0.605 |
| 46 | 0.600 | 0.6225 |
| 48 | 0.5925 | 0.6175 |
| 50 | 0.5875 | 0.6025 |
| 52 | 0.550 | 0.575 |
| 56 | 0.5725 | 0.5975 |

- MATH-500 被低估约 3–5 个百分点，AMC/AIME 被低估 0–3 个百分点。
- 各步排名不变：step 46 仍是最佳。

**对训练的影响：**
- 训练集答案都是整数，只有"Answer: 后换行写 \boxed"这种格式会被误罚。按 AMC/AIME 验证样本估算，约占 0.7%，属于轻微噪声。
- 但它会推着模型改用严格的 "Answer: X" 格式，这本来就是题目要求的格式。

## 2. 数据泄露（训练集 16,689 题 vs 评测集 600 道不同题）

方法：
- 规范化后精确匹配；
- 13-gram 重叠；
- 8-gram 召回后计算编辑相似度；
- 对相似度高的逐条人工确认。

**确认重复的题：**
- **AMC23：3/40**（都在挑选集里）
  - 8^5·5^10·15^5 的位数；
  - log_{7x}2023 方程；
  - u⊗v 复数运算。训练版的问法不同，标准答案也不同。
- **MATH-500：3/500**（都不在挑选集里）
  - 平行四边形面积；
  - mod_5 递归函数；
  - F(z)=(z+i)/(z−i) 迭代。训练版问 a+b。
- **AIME24 / AIME25：0**

其余高相似度的都是"同题型换数字"，13-gram 命中的都是 "where m and n are relatively prime positive integers" 这类套话。

**影响：**
- AMC23 泄露题的正确率：step 46 是 6/6，step 48 是 5/6；其余题约 62%。
- 挑选分数虚高约 0.003–0.006。去掉泄露题后，step 46 是 0.6168，step 48 是 0.6142，排名不变。

## 3. 方法上的问题

1. **挑选集和测试集重叠**：val_select 是 val 的子集，AMC/AIME 的题完全重叠，用同一批题选最佳又报告成绩，会偏乐观。只有 MATH-500 剩 300 题可以当留出集。
2. **噪声**：400 条样本的标准误约 ±0.024。step 46 和 48 只差 0.005，在噪声范围内，"最佳"其实分不出来。

## 4. 建议

- 评测和挑选改用 reward_v2。只影响打分，不影响训练。
- 最终报告同时给出"全部"和"去掉泄露题"两个口径，AIME 用 avg@32。
- 第三轮开始前：
  - 训练改用 reward_v2；
  - 从训练集删掉这 6 道重复题（生成新的 parquet，原文件保留）；
  - 加一个留出测试集，比如 AIME 2026。

## 5. 已执行（2026-11-16 15:50，用户同意）

**第二轮（不重启、不打断）：**
- 训练继续用原判分 ma/reward.py 和原训练集。
- 旁路进程 ma/v2_watch.py（只读）：每出一批验证生成就用 v2 打分，写 run/select_history_v2.jsonl。
- 若 v2 下的最佳和现行最佳不是同一步，写 run/v2_flag.json，由 Claude 在每小时检查时决定。

**代码（第三轮或任何重启后生效；修改前的版本在 ma/round1_backup/*_before_v2.py）：**
- ma/rescore.py：v2 重打分，同时给出"去掉泄露题"的口径。
- ma/eval_agent.py：
  - 挑选最佳改用 v2 分数，验证生成文件不可用时才退回原判分；
  - 沿用的旧最佳会先换算成 v2（step 46：0.600 → 0.6225）；
  - 最终评测输出 v1、v2、v2_clean 三个口径。
- ma/framework_agent.py：
  - formal2（第二轮）不变；
  - formal2_dyn 和 stage16k（第三轮）改用 reward_v2.py 和 train_dapo_dedup.parquet；
  - eval/eval16k 改用 val.parquet + test_aime26.parquet，验证生成存到 /root/r1zero_valgen/final_<exp>/，判分用 v2。
  - 三种配置都通过了 Hydra --cfg job 校验。
- ma/loop3.py：第三轮开始训练前，先用 v2 对第二轮最佳做一次完整评测（含 AIME 2026），结果记在 board 的 r2_best_eval_v2。

**数据（ma/build_round3_data.py）：**
- run/data/train_dapo_dedup.parquet：15,933 题，去掉了 6 道重复题。
- run/data/test_aime26.parquet：AIME 2026，来自 MathArena/aime_2026（hf-mirror）。
  - 第 1 题（Patrick/Tanya/Jose 行程题，答案 277）和 DAPO 训练集逐字相同，已排除；
  - 剩 29 题 × 16 = 464 条。
