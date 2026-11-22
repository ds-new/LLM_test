"""E baseline on the F validation set (run/data/val_select_f.parquet, 400 rows) from E's full eval outputs
(sft/eval_full/gen_*.jsonl), scored with the same reward function F uses (ma/reward_f.py).
Also gives the band in which F's step-10 validation score would fall if F were no better or worse than E
(validation = 1 sample per row): expected, SD of F's single-sample estimate, SD of the baseline, 95% band of the difference.
Writes run/audit/f_val_baseline_E.json. Usage: python f/patches/e_val_baseline.py"""
import json, math, os, re, sys, time
from collections import defaultdict
import pyarrow.parquet as pq
R = '/root/autodl-tmp/r1zero'
sys.path.insert(0, R + '/ma')
from reward_f import compute_score

t0 = time.time()
def norm(s):
    return re.sub(r'\s+', ' ', (s or '')).strip()

rows = []
for k in range(4):
    for l in open(R + '/sft/eval_full/gen_%d.jsonl' % k):
        rows.append(json.loads(l))
acc_q = defaultdict(list)
tot = defaultdict(list)
for r in rows:
    try:
        a = float(compute_score(r['ds'], r['output'], r['gt'])['acc'])
    except Exception as e:
        a = 0.0
    acc_q[(r['ds'], norm(r['q']))].append(a)
    tot[r['ds']].append(a)
recomp = {k: sum(v) / len(v) for k, v in tot.items()}
summ = json.load(open(R + '/sft/eval_full/summary.json'))['summary']
print('E acc recomputed vs summary:', {k: (round(recomp[k], 4), round(summ[k]['acc'], 4)) for k in recomp})

va = pq.read_table(R + '/run/data/val_select_f.parquet').to_pylist()
by_ds = defaultdict(list)
miss = defaultdict(int)
for r in va:
    ds = r['data_source']
    q = norm((r.get('extra_info') or {}).get('question'))
    key = (ds, q)
    if key not in acc_q:
        cand = [k for k in acc_q if k[0] == ds and (k[1][:80] == q[:80] or q[:80] in k[1] or k[1][:80] in q)]
        if len(cand) == 1:
            key = cand[0]
        else:
            miss[ds] += 1
            continue
    by_ds[ds].append(key)
print('unmatched val rows:', dict(miss))

out = {}
for ds, keys in sorted(by_ds.items()):
    N = len(keys)
    cnt = defaultdict(int)
    for k in keys:
        cnt[k] += 1
    p = {k: sum(acc_q[k]) / len(acc_q[k]) for k in cnt}
    m = {k: len(acc_q[k]) for k in cnt}
    base = sum(p[k] * c for k, c in cnt.items()) / N
    single = all(m[k] == 1 for k in cnt)
    if single:
        varF = base * (1 - base) / N
        varB = sum(c * c for c in cnt.values()) * base * (1 - base) / N / N
    else:
        varF = sum(c * p[k] * (1 - p[k]) for k, c in cnt.items()) / N / N
        varB = sum(c * c * p[k] * (1 - p[k]) / m[k] for k, c in cnt.items()) / N / N
    sd = math.sqrt(varF + varB)
    out[ds] = {'rows': N, 'problems': len(cnt), 'E_samples_per_problem': sorted(set(m.values())),
               'E_baseline': round(base, 4), 'sd_F_single_sample': round(math.sqrt(varF), 4),
               'sd_baseline': round(math.sqrt(varB), 4), 'band95_diff': round(1.96 * sd, 4),
               'band95': [round(base - 1.96 * sd, 4), round(base + 1.96 * sd, 4)]}
dss = [d for d in ['math500', 'amc23', 'aime24', 'aime25'] if d in out]
mac = sum(out[d]['E_baseline'] for d in dss) / len(dss)
macsd = math.sqrt(sum((out[d]['band95_diff'] / 1.96) ** 2 for d in dss)) / len(dss)
out['macro4'] = {'E_baseline': round(mac, 4), 'band95_diff': round(1.96 * macsd, 4), 'band95': [round(mac - 1.96 * macsd, 4), round(mac + 1.96 * macsd, 4)]}
aime = [d for d in ['aime24', 'aime25'] if d in out]
am = sum(out[d]['E_baseline'] for d in aime) / len(aime)
amsd = math.sqrt(sum((out[d]['band95_diff'] / 1.96) ** 2 for d in aime)) / len(aime)
out['aime24_25'] = {'E_baseline': round(am, 4), 'band95_diff': round(1.96 * amsd, 4), 'band95': [round(am - 1.96 * amsd, 4), round(am + 1.96 * amsd, 4)]}
out['_note'] = 'F step-10 val-core/<ds>/acc/mean@1 outside band95 = a change vs E beyond sampling noise (approx., normal approximation)'
os.makedirs(R + '/run/audit', exist_ok=True)
json.dump(out, open(R + '/run/audit/f_val_baseline_E.json', 'w'), indent=1, ensure_ascii=False)
for k, v in out.items():
    print(k, v)
print('done in %.0f s' % (time.time() - t0))
