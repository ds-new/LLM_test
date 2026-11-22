"""Append one GPU memory sample to run/logs/mem_samples.jsonl: per GPU bus id, MiB used by the SGLang scheduler,
by the training WorkerDict and by anything else (used by f/restart_after_step.sh)."""
import json, subprocess, time
out = subprocess.run(['nvidia-smi', '--query-compute-apps=gpu_bus_id,pid,used_memory', '--format=csv,noheader,nounits'],
                     capture_output=True, text=True).stdout
gpus = {}
for line in out.strip().splitlines():
    parts = [p.strip() for p in line.split(',')]
    if len(parts) != 3:
        continue
    bus, pid, mem = parts[0], parts[1], int(parts[2])
    try:
        cmd = open('/proc/%s/cmdline' % pid, 'rb').read().replace(b'\0', b' ').decode(errors='replace')
    except Exception:  # noqa: BLE001
        cmd = ''
    kind = 'sglang' if 'sglang' in cmd else ('train' if 'WorkerDict' in cmd else 'other')
    g = gpus.setdefault(bus, {'sglang': 0, 'train': 0, 'other': 0})
    g[kind] += mem
open('/root/autodl-tmp/r1zero/run/logs/mem_samples.jsonl', 'a').write(json.dumps({'t': int(time.time()), 'gpus': gpus}) + '\n')
