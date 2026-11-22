# F 管线的 CPU 自测（不占 GPU）：判分、选题、比较脚本、命令生成
import glob, json, os, random, subprocess, sys
R = '/root/autodl-tmp/r1zero'
sys.path.insert(0, R + '/ma'); sys.path.insert(0, R + '/sft')
from reward_f import compute_score as rf
import eval_e as E
# 1) 判分：和 E 的评测（eval_e.load_e）逐条一致
rows = [json.loads(l) for f in sorted(glob.glob(R + '/sft/eval_smoke/gen_*.jsonl')) for l in open(f)]
le = E.load_e(R + '/sft/eval_smoke')
mism = sum(1 for r, e in zip(rows, le) if ('</think>' in r['output']) and rf(r['ds'], r['output'], r['gt'])['acc'] != e['acc'])
print('reward_f vs eval_e on smoke outputs: n=%d mismatches(with think_end)=%d' % (len(rows), mism))
print(rf('math500', '<think>x</think>\n\nso \\boxed{42}', '42'), rf('math500', '<think>long thinking \\boxed{42}', '42'),
      rf('math500', '<think>a</think> \\boxed{41}<|im_end|>', '42'))
keys = {tuple(sorted(rf('math500', s, '42').keys())) for s in ['<think>x</think>\\boxed{42}', 'no think', '<think>x</think>nothing']}
print('key sets', keys)
# 2) 选题：用假的通过率数据跑一遍
tmp = '/tmp/f_prof_test'; os.makedirs(tmp, exist_ok=True)
rnd = random.Random(1)
with open(tmp + '/prof_0.jsonl', 'w') as fo:
    for i in range(2048):
        p = rnd.random()
        for j in range(4):
            fo.write(json.dumps(dict(i=i, j=j, acc=float(rnd.random() < p), src='x', finish='stop' if rnd.random() > 0.05 else 'length',
                                     n_tok=rnd.randint(500, 16384), think_end=True)) + '\n')
out = subprocess.run([sys.executable, R + '/f/select_frontier_f.py', tmp], capture_output=True, text=True)
print('select rc', out.returncode, out.stdout[:300], out.stderr[-300:])
import pyarrow.parquet as pq
t = pq.read_table(R + '/run/data/train_f.parquet'); s = pq.read_table(R + '/run/data/f_smoke.parquet')
print('train_f rows', t.num_rows, 'smoke rows', s.num_rows, t.schema.names)
print(t.slice(0, 1).to_pylist()[0]['prompt'])
for f in ['/run/data/train_f.parquet', '/run/data/f_smoke.parquet', '/run/audit/f_frontier_stats.json']:
    os.remove(R + f)
# 3) 比较脚本：评测试跑的结果和自己比，差值应为 0
out = subprocess.run([sys.executable, R + '/f/compare_f.py', R + '/sft/eval_smoke', R + '/sft/eval_smoke', 'selftest', '/tmp/cmp_selftest.md'], capture_output=True, text=True)
print('compare rc', out.returncode, out.stdout[:400].replace('\n', ' | '), out.stderr[-400:])
