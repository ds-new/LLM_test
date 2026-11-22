import sys, json
import pyarrow.parquet as pq
R = '/root/autodl-tmp/r1zero'
sys.path.insert(0, R + '/ma')
from reward_f import compute_score
tr = pq.read_table(R + '/run/data/train_f.parquet').to_pylist()
va = pq.read_table(R + '/run/data/val_select_f.parquet').to_pylist()
print('train', len(tr), sorted(tr[0].keys()))
print('val', len(va), sorted(va[0].keys()))
print('train extra', sorted((tr[0].get('extra_info') or {}).keys()), 'rm', sorted((tr[0].get('reward_model') or {}).keys()))
print('val extra', sorted((va[0].get('extra_info') or {}).keys()), 'rm', sorted((va[0].get('reward_model') or {}).keys()))
from collections import Counter
print('val sources', Counter(r['data_source'] for r in va).most_common(5))
print('train sources', Counter(r['data_source'] for r in tr).most_common(5))
p0, p1 = tr[0]['prompt'], va[0]['prompt']
print('prompt types', type(p0).__name__, type(p1).__name__)
def head(p):
    s = p if isinstance(p, str) else json.dumps(p, ensure_ascii=False)
    return s[:160].replace('\n', ' | '), s[-80:].replace('\n', ' | ')
print('train prompt', head(p0))
print('val prompt', head(p1))
bad = 0
for r in va:
    gt = r['reward_model']['ground_truth']
    try:
        a = compute_score(r['data_source'], 'think</think>\nThe answer is \\boxed{' + str(gt) + '}', gt, extra_info=r.get('extra_info'))
        b = compute_score(r['data_source'], 'no end of thinking', gt, extra_info=r.get('extra_info'))
        w = compute_score(r['data_source'], 'x</think>\n\\boxed{-98765}', gt, extra_info=r.get('extra_info'))
        if a['acc'] != 1.0 or b['score'] != -1.0:
            bad += 1
    except Exception as e:
        bad += 1
        print('EXC', repr(e)[:200]); break
print('val rows where gold answer not scored correct or exception:', bad, '/', len(va))
print('sample outputs', a, b, w)
