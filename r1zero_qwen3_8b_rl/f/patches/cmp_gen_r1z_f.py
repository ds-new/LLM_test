"""用户 11/21 17:48 问：为什么 F 的生成比 R1-Zero（step 156）慢、为什么会整批重新生成。对比两次训练每步的生成批数、生成耗时、回答长度。"""
import json
import sys

def load(f):
    rows = {}
    for l in open(f):
        o = json.loads(l)
        d = o.get('data', o)
        s = o.get('step', d.get('step'))
        if s is None or 'timing_s/gen' not in d:
            continue
        rows[int(s)] = d
    return rows

def show(name, f, steps=None):
    rows = load(f)
    ks = sorted(rows)
    if steps:
        ks = [k for k in ks if steps[0] <= k <= steps[1]]
    print('==', name, f, 'steps', ks[0], '-', ks[-1], 'n', len(ks))
    tot_nb = tot_gen = tot_step = 0
    hist = {}
    for k in ks:
        d = rows[k]
        nb = int(d.get('train/num_gen_batches', 1))
        hist[nb] = hist.get(nb, 0) + 1
        tot_nb += nb
        tot_gen += d['timing_s/gen']
        tot_step += d['timing_s/step']
    print('  gen batches per step histogram', dict(sorted(hist.items())), 'mean %.2f' % (tot_nb / len(ks)))
    print('  mean per step: gen %.1f min, step %.1f min; gen per gen-batch %.1f min' % (tot_gen / len(ks) / 60, tot_step / len(ks) / 60, tot_gen / tot_nb / 60))
    rl = [rows[k]['response_length/mean'] for k in ks]
    cr = [rows[k]['response_length/clip_ratio'] for k in ks]
    print('  response length mean %.0f (first %.0f, last %.0f), clip ratio mean %.3f' % (sum(rl) / len(rl), rl[0], rl[-1], sum(cr) / len(cr)))
    if len(sys.argv) > 1:
        for k in ks:
            d = rows[k]
            print('   %4d nb=%d gen=%.1f step=%.1f len=%.0f clip=%.3f' % (k, int(d.get('train/num_gen_batches', 1)), d['timing_s/gen'] / 60, d['timing_s/step'] / 60, d['response_length/mean'], d['response_length/clip_ratio']))

R = '/root/autodl-tmp/r1zero/run/'
show('R1-Zero all', R + 'metrics_r1zero_qwen3_8b_dapo.jsonl')
show('R1-Zero 1-40', R + 'metrics_r1zero_qwen3_8b_dapo.jsonl', (1, 40))
show('R1-Zero 41-100', R + 'metrics_r1zero_qwen3_8b_dapo.jsonl', (41, 100))
show('R1-Zero 101-136', R + 'metrics_r1zero_qwen3_8b_dapo.jsonl', (101, 136))
show('R1-Zero 137-156', R + 'metrics_r1zero_qwen3_8b_dapo.jsonl', (137, 156))
show('F stage1 (gen batch 64)', R + 'metrics_f_stage1_1120_2113.jsonl')
show('F stage1 steps 1-2 (gen batch 32)', R + 'metrics_f_stage1_1120_1606.jsonl')
