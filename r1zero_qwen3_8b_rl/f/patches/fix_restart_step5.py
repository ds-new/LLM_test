P = '/root/autodl-tmp/r1zero/f/restart_step5.sh'
s = open(P).read()
a1 = '  flog "checkpoint step $TARGET complete (tracker $T)"\n'
b1 = a1 + '  flog "dataloader state: $(ls -la $CK/global_step_$TARGET/data.pt 2>&1 | cut -c1-120)"\n'
a2 = 'flog "after stop: GPU MiB'
b2 = ('used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1)\n'
      'if [ "$used" -ge 2000 ]; then\n'
      '  flog "GPUs still busy after stop; kill -9 GPU processes: $(nvidia-smi --query-compute-apps=pid --format=csv,noheader | tr \'\\n\' \' \')"\n'
      '  nvidia-smi --query-compute-apps=pid --format=csv,noheader | xargs -r kill -9\n'
      '  sleep 30\n'
      'fi\n') + a2
assert s.count(a1) == 1 and s.count(a2) == 1
s = s.replace(a1, b1).replace(a2, b2)
open(P, 'w').write(s)
print('updated')
