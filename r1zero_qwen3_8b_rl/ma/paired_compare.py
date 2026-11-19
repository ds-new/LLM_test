"""两个 checkpoint 的完整评测做同题配对比较。
用法：python ma/paired_compare.py <valgen目录或文件A> <valgen目录或文件B>
每道题先对它的多次采样取平均正确率，再按题目做配对 bootstrap（重采样题目 10000 次），给出 B-A 的差值、95% 区间和 B>A 的比例。"""
import json, sys, collections
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ma import rescore

def load(p):
    p = Path(p)
    f = p if p.is_file() else rescore.newest_file(p)
    sm = rescore._src_map(); per = collections.defaultdict(list); src_of = {}
    for line in open(f, encoding='utf-8'):
        d = json.loads(line); key = d['input']
        per[key].append(float(d.get('acc', 0)) > 0.5)
        if key not in src_of:
            src_of[key] = next((s for k, s in sm.items() if k in key), 'unknown')
    return {k: (src_of[k], float(np.mean(v)), len(v)) for k, v in per.items()}, f

A, fa = load(sys.argv[1]); B, fb = load(sys.argv[2])
keys = sorted(set(A) & set(B))
print('A:', fa); print('B:', fb); print('paired problems:', len(keys), '(A only %d, B only %d)' % (len(set(A) - set(B)), len(set(B) - set(A))))
rng = np.random.default_rng(0)
W = {'math500': 200, 'amc23': 80, 'aime24': 60, 'aime25': 60}
def boot(idx_keys):
    a = np.array([A[k][1] for k in idx_keys]); b = np.array([B[k][1] for k in idx_keys]); d = b - a
    n = len(d); bs = d[rng.integers(0, n, size=(10000, n))].mean(1)
    return a.mean(), b.mean(), d.mean(), np.percentile(bs, 2.5), np.percentile(bs, 97.5), (bs > 0).mean(), n
by = collections.defaultdict(list)
for k in keys: by[A[k][0]].append(k)
for s in ['math500', 'amc23', 'aime24', 'aime25', 'aime26']:
    if by.get(s):
        a, b, d, lo, hi, p, n = boot(by[s])
        print('%-8s n=%3d  A %.1f%%  B %.1f%%  diff %+.1f  95%% [%+.1f, %+.1f]  P(B>A)=%.2f' % (s, n, 100 * a, 100 * b, 100 * d, 100 * lo, 100 * hi, p))
# 汇总：每个数据集等权（宏平均），含与不含 AIME26
for name, sets in (('macro(4 sets)', ['math500', 'amc23', 'aime24', 'aime25']), ('macro(5 sets)', ['math500', 'amc23', 'aime24', 'aime25', 'aime26'])):
    sets = [s for s in sets if by.get(s)]
    ds = {s: np.array([B[k][1] - A[k][1] for k in by[s]]) for s in sets}
    bs = np.mean([ds[s][rng.integers(0, len(ds[s]), size=(10000, len(ds[s])))].mean(1) for s in sets], axis=0)
    print('%-14s diff %+.2f  95%% [%+.2f, %+.2f]  P(B>A)=%.3f' % (name, 100 * np.mean([ds[s].mean() for s in sets]), 100 * np.percentile(bs, 2.5), 100 * np.percentile(bs, 97.5), (bs > 0).mean()))
