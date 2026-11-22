"""SGLang 生成并发和吞吐：R1-Zero 第三轮（step 137-156）vs F 第一阶段。另外统计 R1-Zero 日志里第一批可用题数。"""
import re
import sys
import collections

pat = re.compile(rb'Decode batch, #running-req: (\d+), #token: (\d+), token usage: ([0-9.]+), cuda graph: \w+, gen throughput \(token/s\): ([0-9.]+)')
nb = re.compile(rb'num_prompt_in_batch=(\d+) < prompt_bsz=(\d+)')
for name, f in [('R1-Zero r3 (137-156)', '/root/autodl-tmp/r1zero/run/logs/train_r1zero_qwen3_8b_dapo_L2_1119_0033.log'),
                ('F stage1 (steps 3-11)', '/root/autodl-tmp/r1zero/run/logs/f_stage1_1120_2113.log')]:
    data = open(f, 'rb').read()
    rr, tt, gg = [], [], []
    for m in pat.finditer(data):
        rr.append(int(m.group(1))); tt.append(int(m.group(2))); gg.append(float(m.group(4)))
    rr.sort(); gs = sorted(gg)
    q = lambda a, p: a[int(p * (len(a) - 1))]
    pool = max(int(int(t) / float(u)) for t, u in [(m.group(2), m.group(3)) for m in pat.finditer(data)] if float(u) > 0.5)
    print('== %s: %d decode lines; running-req per GPU median %d (p10 %d, p90 %d, max %d); throughput tok/s per GPU median %.0f (p10 %.0f, p90 %.0f); KV pool ~%dK tokens per GPU' % (
        name, len(rr), q(rr, .5), q(rr, .1), q(rr, .9), rr[-1], q(gs, .5), q(gs, .1), q(gs, .9), pool // 1000))
    c = collections.Counter(int(m.group(1)) for m in nb.finditer(data))
    print('   num_prompt_in_batch < prompt_bsz lines:', sorted(c.items()))
