"""E + presence_penalty 试验的细分析（11/22）：按 E 原来每题的通过率分档看收益来自哪里；写完的回答里有多少是对的；
每题变好/变差的题数；新解出 / 丢掉的题。打分和 f/eval_reppen.py 完全一样（annotate）。"""
import collections
import importlib.util
import json
import sys

R = '/root/autodl-tmp/r1zero'
spec = importlib.util.spec_from_file_location('er', R + '/f/eval_reppen.py')
er = importlib.util.module_from_spec(spec)
sys.argv = [sys.argv[0]]
spec.loader.exec_module(er)

base = er.annotate(er.load(R + '/sft/eval_full'))
new = er.annotate(er.load(R + '/f/eval_reppen_E'))
B = collections.defaultdict(list)
for r in base:
    B[(r['ds'], r['q'])].append(r)
A = {'pp0.5': collections.defaultdict(list), 'pp1.5': collections.defaultdict(list)}
for r in new:
    A[r['arm']][(r['ds'], r['q'])].append(r)
keys = sorted(B)
m = lambda xs, f: sum(x[f] for x in xs) / len(xs)
print('problems', len(keys), 'base rows', len(base), 'new rows', len(new))

def bucket(p):
    if p == 0:
        return '0 (never)'
    if p < 0.25:
        return '(0,0.25)'
    if p < 0.75:
        return '[0.25,0.75)'
    if p < 1:
        return '[0.75,1)'
    return '1 (always)'

for arm in ('pp0.5', 'pp1.5'):
    X = A[arm]
    print('==', arm)
    agg = collections.defaultdict(lambda: [0, 0.0, 0.0, 0.0, 0.0])
    up = down = same = 0
    newly = lost = 0
    for k in keys:
        pb, pa = m(B[k], 'acc'), m(X[k], 'acc')
        cb, ca = m(B[k], 'complete'), m(X[k], 'complete')
        b = bucket(pb)
        a = agg[b]
        a[0] += 1; a[1] += pa - pb; a[2] += ca - cb; a[3] += pb; a[4] += cb
        if pa > pb + 1e-9: up += 1
        elif pa < pb - 1e-9: down += 1
        else: same += 1
        if pb == 0 and pa > 0: newly += 1
        if pb > 0 and pa == 0: lost += 1
    for b in ['0 (never)', '(0,0.25)', '[0.25,0.75)', '[0.75,1)', '1 (always)']:
        if b in agg:
            n, da, dc, sa, sc = agg[b]
            print('  E pass-rate bucket %-12s n=%2d  E acc %.1f%%  E complete %.1f%%  acc change %+.1f  complete change %+.1f' % (b, n, 100 * sa / n, 100 * sc / n, 100 * da / n, 100 * dc / n))
    print('  problems acc up %d, down %d, same %d; newly solved (E 0/16 -> >0) %d, lost (E >0 -> 0/8) %d' % (up, down, same, newly, lost))
    rows = [r for k in keys for r in X[k]]
    comp = [r for r in rows if r['complete']]
    bcomp = [r for r in base if r['complete']]
    print('  acc among completed: E %.1f%% (n=%d)  %s %.1f%% (n=%d)' % (100 * m(bcomp, 'acc'), len(bcomp), arm, 100 * m(comp, 'acc'), len(comp)))
    print('  mean tokens completed: E %.0f  %s %.0f; unfinished: E %.0f  %s %.0f' % (m(bcomp, 'tok'), arm, m(comp, 'tok'), m([r for r in base if not r['complete']], 'tok'), arm, m([r for r in rows if not r['complete']], 'tok')))
    unf_b = [r for r in base if not r['complete']]
    unf_a = [r for r in rows if not r['complete']]
    print('  loop share among unfinished: E %.1f%%  %s %.1f%%' % (100 * m(unf_b, 'loop'), arm, 100 * m(unf_a, 'loop')))
    nobox = sum(1 for r in comp if '\\boxed' not in r['output'].split('</think>')[-1])
    print('  completed answers without \\boxed after </think>: %d / %d' % (nobox, len(comp)))
print('ANALYSIS_DONE')
