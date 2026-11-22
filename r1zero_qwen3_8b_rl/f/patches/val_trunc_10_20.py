"""F 第 10 / 20 步验证（/root/r1zero_valgen/f/{10,20}.jsonl，每题 1 次）和 E（sft/eval_full，同题 16 次）在 400 道验证题上的对比：
正确率、超长扣分（reward - score）、写到 16384 上限的比例（扣分 = -1）、没有 </think> 的比例。"""
import json, re
from collections import defaultdict
import pyarrow.parquet as pq
R = '/root/autodl-tmp/r1zero'
SETS = ['math500', 'amc23', 'aime24', 'aime25']
def norm(s):
    return re.sub(r'\s+', ' ', (s or '')).strip()
QDS = {}
for _r in pq.read_table(R + '/run/data/val_select_f.parquet').to_pylist():
    QDS[norm((_r.get('extra_info') or {}).get('question'))] = _r['data_source']
def load_f(step):
    out = defaultdict(list)
    miss = 0
    for l in open('/root/r1zero_valgen/f/%d.jsonl' % step):
        d = json.loads(l)
        pen = float(d['reward']) - float(d['score'])
        q = d['input'].split('user\n', 1)[-1].rsplit('\n\nPlease reason step by step', 1)[0]
        ds = QDS.get(norm(q))
        if ds is None:
            miss += 1
            continue
        out[ds].append(dict(acc=float(d['acc']), pen=pen, cap=1.0 if pen <= -0.999 else 0.0, nothink=0.0 if '</think>' in d['output'] else 1.0))
    print('step', step, 'unmatched', miss, {k: len(v) for k, v in out.items()})
    return out
eq = defaultdict(list)
for k in range(4):
    for l in open(R + '/sft/eval_full/gen_%d.jsonl' % k):
        r = json.loads(l)
        n = int(r['n_tok'] or 0)
        eq[(r['ds'], norm(r['q']))].append(dict(pen=-min(max((n - 12288) / 4096.0, 0.0), 1.0), cap=1.0 if (n >= 16384 or r.get('finish') == 'length') else 0.0,
                                               nothink=0.0 if '</think>' in r['output'] else 1.0))
va = pq.read_table(R + '/run/data/val_select_f.parquet').to_pylist()
E = defaultdict(list)
for r in va:
    key = (r['data_source'], norm((r.get('extra_info') or {}).get('question')))
    if key in eq:
        xs = eq[key]
        E[r['data_source']].append({f: sum(x[f] for x in xs) / len(xs) for f in ('pen', 'cap', 'nothink')})
F10, F20 = load_f(10), load_f(20)
print('%-8s | %-34s | %-34s | %-22s' % ('set', 'F step10 acc/pen/cap/nothink', 'F step20 acc/pen/cap/nothink', 'E pen/cap/nothink'))
agg = defaultdict(list)
for ds in SETS:
    row = []
    for name, F in (('F10', F10), ('F20', F20)):
        xs = F[ds]; n = len(xs)
        m = {f: sum(x[f] for x in xs) / n for f in ('acc', 'pen', 'cap', 'nothink')}
        row.append('%5.1f%% %6.3f %5.1f%% %5.1f%% (n=%d)' % (100 * m['acc'], m['pen'], 100 * m['cap'], 100 * m['nothink'], n))
        if ds.startswith('aime'):
            agg[name].append(m)
    xs = E[ds]; n = len(xs)
    me = {f: sum(x[f] for x in xs) / n for f in ('pen', 'cap', 'nothink')}
    if ds.startswith('aime'):
        agg['E'].append(me)
    print('%-8s | %s | %s | %6.3f %5.1f%% %5.1f%% (n=%d)' % (ds, row[0], row[1], me['pen'], 100 * me['cap'], 100 * me['nothink'], n))
for name in ('F10', 'F20', 'E'):
    ms = agg[name]
    print('AIME mean %s: pen %.3f cap %.1f%% nothink %.1f%%' % (name, sum(m['pen'] for m in ms) / 2, 100 * sum(m['cap'] for m in ms) / 2, 100 * sum(m['nothink'] for m in ms) / 2))
