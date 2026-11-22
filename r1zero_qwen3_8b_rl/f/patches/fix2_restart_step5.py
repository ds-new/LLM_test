P = '/root/autodl-tmp/r1zero/f/restart_step5.sh'
s = open(P).read()
assert 'interleave safety net' not in s
s = s.rstrip('\n') + '\n' + '\n'.join([
    '# interleave safety net: if the resumed run dies with a traceback from the patch before the next step finishes,',
    '# put the original trainer back so that run_f.sh retries with the old code.',
    'NEXT=$((TARGET + 1))',
    'for t in $(seq 1 300); do',
    '  sleep 30',
    '  if python $R/f/patches/has_step.py $NEXT > /dev/null 2>&1; then flog "step $NEXT finished with the interleave trainer"; break; fi',
    '  L=$(ls -t $R/run/logs/f_stage1_*.log | head -1)',
    '  if grep -a -q Traceback $L && grep -a -q "_r1z_\\|in reorder" $L; then',
    '    cp -p $LIVE.orig_interleave $LIVE',
    '    flog "traceback involving the interleave patch in $L: original trainer restored, the run_f.sh retry will use it"',
    '    break',
    '  fi',
    'done',
    'flog "watcher done"',
    ''])
open(P, 'w').write(s)
print('appended')
