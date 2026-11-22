"""E 阶段（蒸馏 SFT）数据准备：open-r1/OpenR1-Math-220k default（Apache-2.0，DeepSeek-R1 生成的推理过程）。
规则（2026-11-19 定）：
  * 每题只取一条：math_verify 判对、推理完整、finish_reason=stop 的生成里最短的一条；
  * 去掉含中文字符的生成（避免中英混杂）；
  * 去掉 </think> 之后没有 \\boxed{} 最终答案的生成、同一行（>20 字符）重复 ≥5 次的生成（2026-11-19 11:35 加）；
  * 去污染：和所有评测题（MATH-500 / AMC23 / AIME24 / AIME25 / AIME 2026 留出集）按词级 13-gram 比对，有重合就删；短题按完整包含比对；
  * 套用 Qwen3 的 chat 模板后总长 <= 16000 token；
  * 固定种子打乱，留 256 条做验证集（只看 loss）。
输出：sft/data/train.parquet、sft/data/val.parquet（messages 列）、sft/data/stats.json
"""
import glob, json, re, random, collections, sys
from multiprocessing import Pool
import pyarrow as pa, pyarrow.parquet as pq
import numpy as np

R = '/root/autodl-tmp/r1zero'
MODEL = '/root/autodl-tmp/models/Qwen3-8B-Base'
MAXTOK = 16000
INSTR = "Please reason step by step, and put your final answer within \\boxed{}."
CJK = re.compile(r'[\u3400-\u9fff]')

def clean(g):
    tail = g.split('</think>')[-1]
    if '\\boxed' not in tail:
        return False
    lines = [l.strip() for l in g.split('\n') if len(l.strip()) > 20]
    return not (lines and collections.Counter(lines).most_common(1)[0][1] >= 5)


def words(s):
    return re.sub(r'[^a-z0-9]+', ' ', s.lower()).split()

def grams(ws, n=13):
    return {' '.join(ws[i:i + n]) for i in range(len(ws) - n + 1)}

# 评测题
evalq = set()
for f in ['val.parquet', 'test_aime26.parquet']:
    for r in pq.read_table(R + '/run/data/' + f, columns=['extra_info']).to_pylist():
        evalq.add(r['extra_info']['question'])
EG = set(); SHORT = []
for q in evalq:
    w = words(q)
    if len(w) >= 13: EG |= grams(w)
    else: SHORT.append(' '.join(w))
print('eval problems', len(evalq), '13-grams', len(EG), 'short', len(SHORT), flush=True)

rows = []
stat = collections.Counter()
for f in sorted(glob.glob(R + '/raw/open-r1_OpenR1-Math-220k/data/*.parquet')):
    t = pq.read_table(f, columns=['problem', 'answer', 'source', 'uuid', 'generations', 'correctness_math_verify', 'is_reasoning_complete', 'finish_reasons'])
    for r in t.to_pylist():
        stat['rows'] += 1
        g = r['generations'] or []
        cv = r['correctness_math_verify'] or []; rc = r['is_reasoning_complete'] or []; fr = r['finish_reasons'] or [None] * len(g)
        ok = [i for i in range(len(g)) if i < len(cv) and cv[i] and i < len(rc) and rc[i] and (fr[i] in (None, 'stop')) and not CJK.search(g[i]) and '</think>' in g[i] and clean(g[i])]
        if not ok:
            stat['no_valid_gen'] += 1; continue
        w = words(r['problem'])
        nw = ' '.join(w)
        if (len(w) >= 13 and grams(w) & EG) or any(s and s in nw for s in SHORT) or CJK.search(r['problem']):
            stat['contaminated_or_cjk_problem'] += 1; continue
        i = min(ok, key=lambda j: len(g[j]))
        rows.append(dict(uuid=r['uuid'], source=r['source'], answer=r['answer'], messages=[
            {'role': 'user', 'content': r['problem'].strip() + '\n\n' + INSTR},
            {'role': 'assistant', 'content': g[i].strip()}]))
print(dict(stat), 'kept before length filter', len(rows), flush=True)

from transformers import AutoTokenizer
tok = None
def ntok(m):
    global tok
    if tok is None:
        tok = AutoTokenizer.from_pretrained(MODEL)
    return len(tok.apply_chat_template(m, tokenize=True))
with Pool(48) as p:
    lens = p.map(ntok, [r['messages'] for r in rows], chunksize=64)
for r, n in zip(rows, lens):
    r['n_tokens'] = n
keep = [r for r in rows if r['n_tokens'] <= MAXTOK]
print('length filter: kept', len(keep), 'dropped', len(rows) - len(keep), flush=True)
random.Random(20261119).shuffle(keep)
val, train = keep[:256], keep[256:]
import os
os.makedirs(R + '/sft/data', exist_ok=True)
for name, part in (('train', train), ('val', val)):
    pq.write_table(pa.Table.from_pylist(part), R + '/sft/data/%s.parquet' % name)
L = np.array([r['n_tokens'] for r in train])
st = dict(stat, kept=len(keep), train=len(train), val=len(val), train_tokens=int(L.sum()), mean_tokens=float(L.mean()),
          pct={p: int(np.percentile(L, p)) for p in (10, 50, 90, 99)}, sources=collections.Counter(r['source'] for r in train).most_common())
json.dump(st, open(R + '/sft/data/stats.json', 'w'), indent=1, ensure_ascii=False)
print(json.dumps(st, ensure_ascii=False))
print('PREP_DONE')
