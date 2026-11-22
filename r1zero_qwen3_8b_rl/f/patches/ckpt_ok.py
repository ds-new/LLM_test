"""Exit 0 if an HF checkpoint dir is complete: every shard named in model.safetensors.index.json exists, total >= 15 GB."""
import json, os, sys
d = sys.argv[1]
if not os.path.isdir(d):
    print('missing dir', d)
    sys.exit(1)
idx = os.path.join(d, 'model.safetensors.index.json')
if os.path.exists(idx):
    need = sorted(set(json.load(open(idx))['weight_map'].values()))
else:
    need = sorted(f for f in os.listdir(d) if f.endswith('.safetensors'))
miss = [f for f in need if not os.path.exists(os.path.join(d, f))]
size = sum(os.path.getsize(os.path.join(d, f)) for f in need if os.path.exists(os.path.join(d, f)))
print('ckpt %s: %d shards, %d missing, %.1f GB, index %s' % (d, len(need), len(miss), size / 1e9, os.path.exists(idx)))
sys.exit(1 if (miss or not need or size < 15e9) else 0)
