import glob
import json
import os
import time

for d in ['f/eval_reppen_E', 'f/eval_s20']:
    n = t = 0
    for f in sorted(glob.glob(d + '/gen_*.jsonl')):
        rows = [json.loads(l) for l in open(f)]
        n += len(rows)
        t += sum(x.get('n_tok') or 0 for x in rows)
        print(f, len(rows), sum(x.get('n_tok') or 0 for x in rows), time.strftime('%H:%M:%S', time.localtime(os.path.getmtime(f))))
    print(d, 'rows', n, 'tokens', t)
