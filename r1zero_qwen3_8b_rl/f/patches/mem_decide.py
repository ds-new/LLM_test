"""Can SGLang gpu_memory_utilization go from 0.55 to 0.60? Uses run/logs/mem_samples.jsonl samples taken in [START, END]
during generation (SGLang process > 30 GB on that GPU). Exit 0 = raise is safe. Usage: python f/patches/mem_decide.py START [END]"""
import json, sys
start = int(sys.argv[1])
end = int(sys.argv[2]) if len(sys.argv) > 2 else 10 ** 12
TOTAL = 81152   # MiB, torch total_memory of an A800 80GB (79.25 GiB); SGLang sizes its static pool from this
DELTA = 4058    # 0.05 x TOTAL: what 0.55 -> 0.60 adds to the SGLang static pool
MARGIN = 4096   # keep >= 4 GiB free; with chunked_prefill_size 2048 the largest single SGLang allocation is about 1.2 GiB
best = n = tr = sg = 0
for line in open('/root/autodl-tmp/r1zero/run/logs/mem_samples.jsonl'):
    r = json.loads(line)
    if not (start <= r['t'] <= end):
        continue
    for g in r['gpus'].values():
        if g['sglang'] > 30000 and g['train'] <= 24000:  # generation phase; skip weight-sync transitions where both are large
            n += 1
            tot = g['sglang'] + g['train'] + g['other']
            if tot > best:
                best, tr, sg = tot, g['train'], g['sglang']
ok = n >= 20 and best + DELTA <= TOTAL - MARGIN
print('mem_decide: %d generation samples; peak per-GPU %d MiB (SGLang %d + training %d); at 0.60 about %d MiB vs limit %d -> %s'
      % (n, best, sg, tr, best + DELTA, TOTAL - MARGIN, 'RAISE to 0.60' if ok else 'KEEP 0.55'))
sys.exit(0 if ok else 1)
