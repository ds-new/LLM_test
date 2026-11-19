"""第三轮数据：
1. train_dapo_dedup.parquet：去掉与评测集重复的 6 道题（审计 run/audit/AUDIT.md 第 2 节），原文件不动；
2. test_aime26.parquet：AIME 2026（MathArena/aime_2026），去掉训练集里已有的原题后 x16 次采样，作为从未参与挑选的留出测试集；
   并检查它和训练集有没有重复。"""
import json, re, sys
import pandas as pd
sys.path.insert(0, 'ma')
from audit_contam import norm, grams  # noqa: E402  (导入会重跑一遍泄露检查，约 30 秒)

D = 'run/data/'
PROMPT = ('Solve the following math problem step by step. The last line of your response should be of the form '
          'Answer: $Answer (without quotes) where $Answer is the answer to the problem.\n\n{q}\n\n'
          'Remember to put your answer on its own line after "Answer:".')
tr = pd.read_parquet(D + 'train_dapo.parquet')
# 用审计确认的 6 道重复题的验证集题干定位训练集里的对应行（编辑相似度 >= 0.7 且命中同一题）
LEAK_VAL = ['8^5 \\cdot 5^{10} \\cdot 15^5', '\\log_{7x}2023', 'u \\otimes v = ac + bdi', 'y=ax+c$, $y=ax+d$', '\\text{mod}_5 (n)', 'F(z)=\\frac{z+i}{z-i}']
con = [json.loads(l) for l in open('run/audit/contamination.jsonl')]
bad_q = {norm(c['train_q']) for c in con if c['train_src'] == 'train_dapo' and any(k in c['val_q'] for k in LEAK_VAL)}
tq = tr['extra_info'].map(lambda e: norm(e['question']))
# train_q 只存了前 300 字符，用前缀匹配
mask = tq.map(lambda q: any(q.startswith(b[:200]) for b in bad_q))
print('leaked val problems matched:', len(bad_q), '-> train rows removed:', int(mask.sum()))
tr[~mask].reset_index(drop=True).to_parquet(D + 'train_dapo_dedup.parquet')
print('train_dapo_dedup rows', int((~mask).sum()))

a = pd.read_parquet(D + 'raw/aime26/data/train-00000-of-00001.parquet')
print('aime26 cols', list(a.columns), len(a))
qcol = next(c for c in a.columns if c.lower() in ('problem', 'question'))
acol = next(c for c in a.columns if c.lower() in ('answer', 'gold_answer', 'solution_answer'))
tset13 = set()
for q in tq: tset13 |= grams(q.split(), 13)
rows = []; dropped = []
for i, r in a.iterrows():
    q = str(r[qcol]).strip(); ans = str(r[acol]).strip()
    g = grams(norm(q).split(), 13)
    if len([x for x in g if x in tset13]) / max(1, len(g)) >= 0.8:
        dropped.append(int(r['problem_idx']) if 'problem_idx' in r else int(i)); continue  # 训练集里已有原题（第 1 题与 DAPO 训练集逐字相同，答案 277）
    for rep in range(16):
        rows.append(dict(data_source='aime26', prompt=[{'role': 'user', 'content': PROMPT.format(q=q)}], ability='math',
                         reward_model={'ground_truth': ans, 'style': 'rule'},
                         extra_info={'index': int(i), 'question': q, 'rep': rep, 'split': 'test'}))
pd.DataFrame(rows).to_parquet(D + 'test_aime26.parquet')
print('aime26 dropped (already in train):', dropped)
print('test_aime26 rows', len(rows), 'sample answer', a[acol].tolist()[:5])

# 和训练集查重
import difflib
hits = []
for i, r in a.iterrows():
    t = norm(str(r[qcol])).split(); g = grams(t, 13)
    f = len([x for x in g if x in tset13]) / max(1, len(g))
    if f > 0.3: hits.append((i, round(f, 2), str(r[qcol])[:120]))
print('aime26 problems with >30% 13-gram overlap vs train:', hits)
