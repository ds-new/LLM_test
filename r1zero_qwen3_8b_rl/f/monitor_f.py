"""F 训练中的停训条件。
事先定的原版：熵连续 2 步 > 1.0，或截断率连续 2 步 >= 0.2。
2026-11-20 04:42 在第一阶段开始前改成相对基线的版本：E 的模型本身截断率就高（完整评测里 AIME 约 49%、
AMC 18%、MATH-500 8%），按原版 0.2 的绝对阈值，第一阶段第 2 步就会误停。
现在的规则：截断率连续 2 步 >= max(0.2, 基线 + 0.15)，或熵连续 2 步 > max(1.0, 1.5 x 基线)；
基线 = 本模式最早一步（第 1 步，用初始模型采样）的值。
读 run/metrics_f_<mode>_*.jsonl（verl 的 file logger），触发时打印原因并以 1 退出。用法：python f/monitor_f.py stage1"""
import glob
import json
import sys

R = '/root/autodl-tmp/r1zero'
by = {}
for f in sorted(glob.glob(R + '/run/metrics_f_%s_*.jsonl' % sys.argv[1])):
    for line in open(f):
        try:
            rec = json.loads(line)
        except Exception:  # noqa: BLE001
            continue
        d = rec['data'] if isinstance(rec.get('data'), dict) else rec
        st = rec.get('step', d.get('training/global_step'))
        if st is not None and 'actor/entropy' in d:
            by[int(st)] = d
allsteps = sorted(by)
steps = allsteps[-2:]
if len(steps) == 2:
    base = by[allsteps[0]]
    tr_th = max(0.2, base.get('response_length/clip_ratio', 0) + 0.15)
    ent_th = max(1.0, 1.5 * base.get('actor/entropy', 0))
    ent = [by[s].get('actor/entropy', 0) for s in steps]
    tr = [by[s].get('response_length/clip_ratio', 0) for s in steps]
    if min(ent) > ent_th:
        print('entropy > %.3f (baseline step %d) at steps %s: %s' % (ent_th, allsteps[0], steps, ent))
        sys.exit(1)
    if min(tr) >= tr_th:
        print('truncation >= %.3f (baseline step %d) at steps %s: %s' % (tr_th, allsteps[0], steps, tr))
        sys.exit(1)
print('ok', steps[-1] if steps else None)
