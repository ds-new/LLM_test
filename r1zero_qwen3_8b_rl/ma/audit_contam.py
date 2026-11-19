"""训练集 vs 评测集泄露检查：精确匹配（规范化后）、13-gram 重叠（GPT-3/OpenAI 口径）、8-gram 召回 + 编辑相似度。"""
import re, json, collections, difflib
import pandas as pd

def q_of(r):
    ei = r['extra_info']
    q = ei.get('question') if isinstance(ei, dict) else None
    if not q:
        q = r['prompt'][0]['content']
    return q

def norm(q):
    q = q.lower()
    q = re.sub(r'\\(left|right|displaystyle|dfrac|tfrac)', lambda m: 'frac' if 'frac' in m.group(1) else '', q)
    q = re.sub(r'[^a-z0-9]+', ' ', q)
    return ' '.join(q.split())

def grams(toks, n):
    return {' '.join(toks[i:i + n]) for i in range(len(toks) - n + 1)}

train = []
for f in ['train_dapo', 'train_xr750']:
    df = pd.read_parquet('run/data/%s.parquet' % f)
    for _, r in df.iterrows():
        train.append((f, q_of(r), str(r['reward_model']['ground_truth'])))
val = pd.read_parquet('run/data/val.parquet')
vq = {}
for _, r in val.iterrows():
    q = q_of(r); vq.setdefault((r['data_source'], q), str(r['reward_model']['ground_truth']))
sel = pd.read_parquet('run/data/val_select.parquet')
selq = {q_of(r) for _, r in sel.iterrows()}

tn = [(f, q, a, norm(q).split()) for f, q, a in train]
exact = collections.defaultdict(list)
for i, (f, q, a, t) in enumerate(tn): exact[' '.join(t)].append(i)
g13 = collections.defaultdict(set); g8 = collections.defaultdict(set)
for i, (f, q, a, t) in enumerate(tn):
    for g in grams(t, 13): g13[g].add(i)
    for g in grams(t, 8): g8[g].add(i)

res = []; C = collections.Counter()
for (src, q), gt in vq.items():
    t = norm(q).split(); key = ' '.join(t)
    G13 = grams(t, 13); G8 = grams(t, 8)
    hit13 = [g for g in G13 if g in g13]
    frac13 = len(hit13) / max(1, len(G13))
    cand = collections.Counter()
    for g in G8:
        for i in g8.get(g, ()): cand[i] += 1
    best = None
    for i, _ in cand.most_common(5):
        r = difflib.SequenceMatcher(None, key, ' '.join(tn[i][3])).ratio()
        if not best or r > best[0]: best = (r, i)
    ex = exact.get(key, [])
    sim = best[0] if best else 0.0
    bi = best[1] if best else None
    same_ans = bi is not None and re.sub(r'\.0$', '', gt.strip()) == re.sub(r'\.0$', '', tn[bi][2].strip())
    level = 'exact' if ex else ('near_dup' if sim >= 0.85 else ('high_overlap' if frac13 >= 0.5 or sim >= 0.7 else ('some_13gram' if hit13 else 'clean')))
    C[(src, level)] += 1; C[(src, 'n')] += 1
    if level != 'clean':
        res.append(dict(src=src, level=level, in_select=q in selq, sim=round(sim, 3), frac13=round(frac13, 3), same_answer=same_ans,
                        val_q=q[:300], train_src=tn[bi][0] if bi is not None else None, train_q=tn[bi][1][:300] if bi is not None else None,
                        gt=gt, train_gt=tn[bi][2] if bi is not None else None, hit13=hit13[:3]))
with open('run/audit/contamination.jsonl', 'w') as fo:
    for x in sorted(res, key=lambda x: -x['sim']): fo.write(json.dumps(x, ensure_ascii=False) + '\n')
summ = collections.defaultdict(dict)
for (s, k), v in C.items(): summ[s][k] = v
json.dump(summ, open('run/audit/contamination_summary.json', 'w'), indent=1)
print('train', len(train), 'val unique', len(vq)); print(json.dumps(summ, indent=1))
