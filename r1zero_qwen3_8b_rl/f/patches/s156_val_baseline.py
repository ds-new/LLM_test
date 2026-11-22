"""Step 156 (R1-Zero final) on the F validation set: expected accuracy on the same 400 val rows, from its full eval
(/root/r1zero_valgen/final_final_eval_r3_frontier_last/156.jsonl, own prompt format + v2 scoring, 'acc' field), and the 95% band
for F's single-sample validation score if F were as good as step 156. Also E on the same rows, and response lengths
(156: chars / (E chars per token) as an estimate; E: n_tok). Writes run/audit/f_val_baseline_156.json."""
import json, math, re
from collections import defaultdict
import pyarrow.parquet as pq
R = '/root/autodl-tmp/r1zero'
def norm(s):
    return re.sub(r'\s+', ' ', (s or '')).strip()
HEAD = 'where $Answer is the answer to the problem.'
TAIL = 'Remember to put your answer on its own line'
acc156 = defaultdict(list); ch156 = defaultdict(list)
for l in open('/root/r1zero_valgen/final_final_eval_r3_frontier_last/156.jsonl'):
    r = json.loads(l)
    inp = r['input']
    i = inp.find(HEAD); j = inp.rfind(TAIL)
    q = norm(inp[i + len(HEAD):j] if (i >= 0 and j > i) else inp)
    acc156[q].append(float(r['acc'])); ch156[q].append(len(r['output']))
accE = defaultdict(list); tokE = defaultdict(list); chE = defaultdict(list)
import sys
sys.path.insert(0, R + '/ma')
from reward_f import compute_score
for k in range(4):
    for l in open(R + '/sft/eval_full/gen_%d.jsonl' % k):
        r = json.loads(l)
        q = norm(r['q'])
        accE[q].append(float(compute_score(r['ds'], r['output'], r['gt'])['acc'])); tokE[q].append(int(r['n_tok'])); chE[q].append(len(r['output']))
cpt = sum(sum(v) for v in chE.values()) / sum(sum(v) for v in tokE.values())
va = pq.read_table(R + '/run/data/val_select_f.parquet').to_pylist()
by = defaultdict(list); miss = defaultdict(int)
for r in va:
    q = norm((r.get('extra_info') or {}).get('question'))
    if q not in acc156:
        c = [k for k in acc156 if k[:80] == q[:80]]
        if len(c) == 1:
            q = c[0]
        else:
            miss[r['data_source']] += 1; continue
    by[r['data_source']].append(q)
print('unmatched', dict(miss), 'E chars/token %.2f' % cpt)
F = {'math500': 0.87, 'amc23': 0.725, 'aime24': 0.2833333333333333, 'aime25': 0.3}
out = {}
for ds in ['math500', 'amc23', 'aime24', 'aime25']:
    keys = by[ds]; N = len(keys)
    cnt = defaultdict(int)
    for k in keys:
        cnt[k] += 1
    p = {k: sum(acc156[k]) / len(acc156[k]) for k in cnt}; m = {k: len(acc156[k]) for k in cnt}
    base = sum(p[k] * c for k, c in cnt.items()) / N
    if all(v == 1 for v in m.values()):
        varF = base * (1 - base) / N; varB = sum(c * c for c in cnt.values()) * base * (1 - base) / N / N
    else:
        varF = sum(c * p[k] * (1 - p[k]) for k, c in cnt.items()) / N / N
        varB = sum(c * c * p[k] * (1 - p[k]) / m[k] for k, c in cnt.items()) / N / N
    sd = math.sqrt(varF + varB)
    e = sum(sum(accE[k]) / len(accE[k]) * c for k, c in cnt.items()) / N
    t156 = sum(sum(ch156[k]) / len(ch156[k]) * c for k, c in cnt.items()) / N / cpt
    tE = sum(sum(tokE[k]) / len(tokE[k]) * c for k, c in cnt.items()) / N
    out[ds] = {'N': N, 's156': round(base, 4), 'E': round(e, 4), 'F10': F[ds], 'band95_if_F_eq_156': [round(base - 1.96 * sd, 4), round(base + 1.96 * sd, 4)], 'sd': round(sd, 4),
               'len_tok_156_est': round(t156), 'len_tok_E': round(tE)}
    print(ds, out[ds])
d = ['math500', 'amc23', 'aime24', 'aime25']
for name, sel in [('macro4', d), ('aime24_25', ['aime24', 'aime25'])]:
    b = sum(out[x]['s156'] for x in sel) / len(sel); f = sum(out[x]['F10'] for x in sel) / len(sel); e = sum(out[x]['E'] for x in sel) / len(sel)
    sd = math.sqrt(sum(out[x]['sd'] ** 2 for x in sel)) / len(sel)
    out[name] = {'s156': round(b, 4), 'E': round(e, 4), 'F10': round(f, 4), 'diff_F_minus_156': round(f - b, 4), 'band95_if_F_eq_156': [round(b - 1.96 * sd, 4), round(b + 1.96 * sd, 4)]}
    print(name, out[name])
json.dump(out, open(R + '/run/audit/f_val_baseline_156.json', 'w'), indent=1)
