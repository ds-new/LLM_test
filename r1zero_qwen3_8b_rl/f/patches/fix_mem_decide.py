P = '/root/autodl-tmp/r1zero/f/patches/mem_decide.py'
s = open(P).read()
a = "        if g['sglang'] > 30000:\n"
b = "        if g['sglang'] > 30000 and g['train'] <= 24000:  # generation phase; skip weight-sync transitions where both are large\n"
assert s.count(a) == 1
open(P, 'w').write(s.replace(a, b))
print('mem_decide: transition filter added')
