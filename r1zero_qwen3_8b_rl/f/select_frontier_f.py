"""按 E 模型的通过率挑 F 的训练题（沿用 D 的规则）：k 次里答对 1..k-1 次的是边界题，全错的最多占 30%，全对的去掉。
另外挑 4 道平均回答最长的边界题做冒烟测试（尽量碰到 16K 长度，测显存）。用法：python f/select_frontier_f.py PROFILE_DIR"""
import collections
import glob
import json
import os
import random
import sys

import pyarrow as pa
import pyarrow.parquet as pq

R = '/root/autodl-tmp/r1zero'
out = sys.argv[1]
rows = pq.read_table(R + '/run/data/f_profile.parquet').to_pylist()
acc, ntok, trunc, nothink = (collections.defaultdict(list) for _ in range(4))
for f in sorted(glob.glob(os.path.join(out, 'prof_*.jsonl'))):
    for line in open(f):
        d = json.loads(line)
        acc[d['i']].append(d['acc'])
        ntok[d['i']].append(d['n_tok'] or 0)
        trunc[d['i']].append(d['finish'] == 'length')
        nothink[d['i']].append(not d['think_end'])
cnt = {i: int(round(sum(v))) for i, v in acc.items()}
dist = collections.Counter('%d/%d' % (c, len(acc[i])) for i, c in cnt.items())
front = sorted(i for i, c in cnt.items() if 0 < c < len(acc[i]))
hard = sorted(i for i, c in cnt.items() if c == 0)
easy = [i for i, c in cnt.items() if c == len(acc[i])]
random.Random(0).shuffle(hard)
hard_used = hard[:int(len(front) * 0.3 / 0.7)]
sel = sorted(front + hard_used)
pq.write_table(pa.Table.from_pylist([rows[i] for i in sel]), R + '/run/data/train_f.parquet')
mlen = {i: sum(v) / len(v) for i, v in ntok.items()}
smoke = sorted(front, key=lambda i: -mlen[i])[:4]
pq.write_table(pa.Table.from_pylist([rows[i] for i in smoke]), R + '/run/data/f_smoke.parquet')
alln = [x for v in ntok.values() for x in v]
st = dict(problems=len(cnt), samples=len(alln), dist=dict(sorted(dist.items())), easy=len(easy), frontier=len(front), hard=len(hard),
          hard_used=len(hard_used), n_train=len(sel), mean_pass=sum(sum(v) / len(v) for v in acc.values()) / max(1, len(acc)),
          mean_tokens=sum(alln) / max(1, len(alln)), trunc_rate=sum(x for v in trunc.values() for x in v) / max(1, len(alln)),
          no_think_end_rate=sum(x for v in nothink.values() for x in v) / max(1, len(alln)),
          smoke=[[i, round(mlen[i])] for i in smoke])
json.dump(st, open(R + '/run/audit/f_frontier_stats.json', 'w'), indent=1)
print(json.dumps(st))
