"""完整评测的 mean@16 / maj@16（多数投票）/ pass@16（16 次里至少对一次）。用法：python ma/passk.py <valgen 目录> [...]"""
import collections, json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import stage_watch as S
import reward_v2 as R
for p in sys.argv[1:]:
    D, f = S.load(p)
    sm = S.src_map(); per = collections.defaultdict(list)
    for line in open(f, encoding='utf-8'):
        d = json.loads(line)
        a, _ = R.final_answer(d['output'])
        per[d['input']].append((R._norm_txt(a) if a else None, float(d.get('acc', 0)) > 0.5))
    res = collections.defaultdict(lambda: collections.defaultdict(list))
    for k, v in per.items():
        src = next((s for t, s in sm.items() if t in k), '?')
        if len(v) < 16:
            continue
        votes = collections.defaultdict(list)
        for ans, ok in v:
            if ans:
                votes[ans].append(ok)
        maj = max(votes.values(), key=len) if votes else [False]
        res[src]['mean'].append(np.mean([ok for _, ok in v])); res[src]['maj'].append(np.mean(maj) > 0.5); res[src]['pass'].append(any(ok for _, ok in v))
    print(Path(f).parent.name, Path(f).name)
    for s in ['amc23', 'aime24', 'aime25', 'aime26']:
        r = res[s]
        print('  %-7s n=%2d  mean@16 %5.1f%%   maj@16 %5.1f%%   pass@16 %5.1f%%' % (s, len(r['mean']), 100 * np.mean(r['mean']), 100 * np.mean(r['maj']), 100 * np.mean(r['pass'])))
