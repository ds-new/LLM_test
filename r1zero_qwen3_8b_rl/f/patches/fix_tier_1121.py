import shutil, re
R = '/root/autodl-tmp/r1zero'
p = R + '/f/make_command_f.py'
src = open(p).read()
shutil.copy(p, R + '/f/patches/make_command_f.py.bak1121a')
old = "tier = max(tier, int(os.environ.get('F_MIN_TIER', '0')))"
assert src.count(old) == 1, 'pattern'
new = "tier = tier + int(os.environ.get('F_MIN_TIER', '0'))  # 11/21 01:4x Claude: was max(); with F_MIN_TIER=1 the first OOM retry (raw tier 1) stayed at tier 1, so escalation was a no-op; now raw tier is an offset on top of F_MIN_TIER"
src = src.replace(old, new)
open(p, 'w').write(src)
# test copy writing to /tmp
t = src.replace("open(R + '/run/command_f_%s.sh' % mode, 'w')", "open('/tmp/command_f_test_%s_' % mode + sys.argv[3] + '.sh', 'w')")
assert t != src
open('/tmp/mcf_test.py', 'w').write(t)
print('ok')
