"""What do E's unfinished answers look like, and how much accuracy is there to gain from finishing?
From sft/eval_full/gen_*.jsonl (E, 16 samples per AMC/AIME problem, cap 16384 tokens), scored with ma/reward_f.py."""
import json, re, sys, zlib, collections
R = '/root/autodl-tmp/r1zero'
sys.path.insert(0, R + '/ma')
from reward_f import compute_score
def norm(s):
    return re.sub(r'\s+', ' ', (s or '')).strip()
def loop(o):
    lines = [l.strip() for l in o.split('\n') if len(l.strip()) > 20]
    c = collections.Counter(lines)
    top = c.most_common(1)[0][1] if c else 0
    tail = o[-8000:].encode('utf-8', 'ignore')
    ratio = len(zlib.compress(tail, 6)) / max(1, len(tail))
    return top, ratio
rows = []
for k in range(4):
    for l in open(R + '/sft/eval_full/gen_%d.jsonl' % k):
        r = json.loads(l)
        o = r['output']
        r['acc'] = float(compute_score(r['ds'], o, r['gt'])['acc'])
        r['complete'] = r.get('finish') == 'stop' and '</think>' in o
        r['think_done'] = '</think>' in o
        r['top'], r['ratio'] = loop(o)
        r['qk'] = norm(r['q'])
        rows.append(r)
print('ds        n   complete  trunc  | of truncated: think_unfinished  rep>=5  hard_loop(top>=20 or zlib<0.12)  mild(zlib<0.2) | acc_all  acc_complete')
res = {}
for ds in ['math500', 'amc23', 'aime24', 'aime25', 'aime26']:
    rs = [r for r in rows if r['ds'] == ds]
    tr = [r for r in rs if not r['complete']]
    co = [r for r in rs if r['complete']]
    hard = [r for r in tr if r['top'] >= 20 or r['ratio'] < 0.12]
    mild = [r for r in tr if r['ratio'] < 0.2]
    rep5 = [r for r in tr if r['top'] >= 5]
    nt = max(1, len(tr))
    print('%-8s %4d   %5.1f%%  %5.1f%%  |  %5.1f%%  %5.1f%%  %5.1f%%  %5.1f%%  |  %5.1f%%  %5.1f%%' % (ds, len(rs), 100 * len(co) / len(rs), 100 * len(tr) / len(rs),
          100 * sum(not r['think_done'] for r in tr) / nt, 100 * len(rep5) / nt, 100 * len(hard) / nt, 100 * len(mild) / nt,
          100 * sum(r['acc'] for r in rs) / len(rs), 100 * sum(r['acc'] for r in co) / max(1, len(co))))
    # per problem
    byq = collections.defaultdict(list)
    for r in rs:
        byq[r['qk']].append(r)
    nq = len(byq)
    solved = [q for q, v in byq.items() if any(x['acc'] for x in v)]
    never = [q for q in byq if q not in solved]
    def tr_rate(qs):
        v = [x for q in qs for x in byq[q]]
        return 100 * sum(not x['complete'] for x in v) / max(1, len(v))
    # counterfactual: truncated samples get that problem's completed-sample accuracy
    cf = 0.0; cf_noloop = 0.0
    for q, v in byq.items():
        c = [x for x in v if x['complete']]
        pc = sum(x['acc'] for x in c) / len(c) if c else 0.0
        cf += pc
        # only non-loop truncated samples are rescued; hard loops stay wrong
        n = len(v)
        resc = sum(1 for x in v if (not x['complete']) and not (x['top'] >= 20 or x['ratio'] < 0.12))
        cf_noloop += (sum(x['acc'] for x in v) + resc * pc) / n
    res[ds] = dict(nq=nq, solved=len(solved), tr_solved=tr_rate(solved), tr_never=tr_rate(never), acc=100 * sum(r['acc'] for r in rs) / len(rs), cf=100 * cf / nq, cf_noloop=100 * cf_noloop / nq)
print()
print('ds        problems  solved>=1  trunc rate on solved / never-solved problems | acc now -> if truncated samples scored like the same problem\'s completed ones (all / only non-loop rescued)')
for ds, v in res.items():
    print('%-8s  %3d  %3d   %5.1f%% / %5.1f%%  |  %5.1f%% -> %5.1f%% / %5.1f%%' % (ds, v['nq'], v['solved'], v['tr_solved'], v['tr_never'], v['acc'], v['cf'], v['cf_noloop']))
# length of completed vs token position
import statistics
for ds in ['aime24', 'aime25']:
    rs = [r for r in rows if r['ds'] == ds and r['complete']]
    ok = [int(r['n_tok']) for r in rs if r['acc']]; bad = [int(r['n_tok']) for r in rs if not r['acc']]
    print(ds, 'completed: median tokens correct %d (n=%d), wrong %d (n=%d)' % (statistics.median(ok), len(ok), statistics.median(bad) if bad else 0, len(bad)))
