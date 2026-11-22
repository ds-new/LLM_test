"""F 阶段结论：F 的完整评测对 E 的完整评测做同题配对 bootstrap（和前面各阶段同一套事先定好的标准：
四个原评测集等权平均，95% 区间下限 > 0 为显著提升，上限 < 0 为显著下降；AIME 2026 单独报告）。
用法：python f/compare_f.py E_EVAL_DIR F_EVAL_DIR 名称 OUT_MD"""
import json
import os
import sys

import numpy as np

R = '/root/autodl-tmp/r1zero'
sys.path.insert(0, R + '/sft')
sys.path.insert(0, R + '/ma')
import eval_e as E  # noqa: E402

base, new, name, out = sys.argv[1:5]


def load(d):
    rows = E.load_e(d)
    D = {}
    for r in rows:
        D.setdefault((r['ds'], r['q']), []).append(r['acc'])
    by = {}
    for r in rows:
        by.setdefault(r['ds'], []).append(r)
    summ = {ds: dict(acc=float(np.mean([r['acc'] for r in rs])), complete=float(np.mean([r['complete'] for r in rs])),
                     cjk=float(np.mean([r['cjk'] for r in rs])), rep=float(np.mean([r['rep'] for r in rs])),
                     tok=float(np.mean([r.get('n_tok') or 0 for r in rs]))) for ds, rs in by.items()}
    return D, summ


B, sb = load(base)
N, sn = load(new)
P = E.paired(N, B)
m = P['macro4']
L = ['# %s 结论' % name, '', '自动生成（f/compare_f.py）。基线：%s；本阶段：%s。' % (base, new), '',
     '**%s：%s。** 四个原评测集平均 %+.2f 个百分点，95%% 区间 [%+.2f, %+.2f]；AIME 2026 留出集 %+.1f，区间 [%+.1f, %+.1f]。' % (
         name, E.verdict(m), 100 * m['diff'], 100 * m['lo'], 100 * m['hi'],
         100 * P['aime26']['diff'], 100 * P['aime26']['lo'], 100 * P['aime26']['hi']) if 'aime26' in P else '',
     '', '| 评测集 | E | 本阶段 | 差值 | 95% 区间 | 题数 |', '|---|---|---|---|---|---|']
for ds in E.ALL5:
    if ds in P:
        p = P[ds]
        L.append('| %s | %.1f%% | %.1f%% | %+.1f | [%+.1f, %+.1f] | %d |' % (E.NAMES[ds], 100 * sb[ds]['acc'], 100 * sn[ds]['acc'],
                                                                          100 * p['diff'], 100 * p['lo'], 100 * p['hi'], p['n']))
L.append('| 四集平均 | | | %+.2f | [%+.2f, %+.2f] | |' % (100 * m['diff'], 100 * m['lo'], 100 * m['hi']))
L += ['', '可用性（E → 本阶段）：', '', '| 评测集 | 正常结束 | 含中文 | 行重复 | 平均长度（token） |', '|---|---|---|---|---|']
for ds in E.ALL5:
    if ds in sb and ds in sn:
        a, b = sb[ds], sn[ds]
        L.append('| %s | %.1f%% → %.1f%% | %.1f%% → %.1f%% | %.1f%% → %.1f%% | %d → %d |' % (
            E.NAMES[ds], 100 * a['complete'], 100 * b['complete'], 100 * a['cjk'], 100 * b['cjk'], 100 * a['rep'], 100 * b['rep'], a['tok'], b['tok']))
open(out, 'w').write('\n'.join(L) + '\n')
json.dump(dict(paired=P, base=sb, new=sn), open(os.path.splitext(out)[0] + '.json', 'w'), indent=1, ensure_ascii=False)
print('\n'.join(L[:5]))
