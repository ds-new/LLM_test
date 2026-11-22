"""Exit 0 if any run/metrics_f_stage1_*.jsonl has a record for the given step (used by f/restart_step5.sh)."""
import glob, json, sys
R = '/root/autodl-tmp/r1zero'
t = int(sys.argv[1])
for f in glob.glob(R + '/run/metrics_f_stage1_*.jsonl'):
    for line in open(f):
        try:
            rec = json.loads(line)
        except Exception:  # noqa: BLE001
            continue
        d = rec['data'] if isinstance(rec.get('data'), dict) else rec
        st = rec.get('step', d.get('training/global_step'))
        if st is not None and int(st) == t:
            print('yes', f)
            sys.exit(0)
print('no')
sys.exit(1)
