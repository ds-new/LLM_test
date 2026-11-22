import ast
SRC = '/root/autodl-tmp/r1zero/third_party/verl_v061/recipe/dapo/dapo_ray_trainer.py'
DST = '/root/autodl-tmp/r1zero/f/patches/dapo_ray_trainer.interleave.py'
src = open(SRC).read()
anchor = 'gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch_output)'
lines = src.split('\n')
idx = [i for i, l in enumerate(lines) if l.strip() == anchor]
assert len(idx) == 1, idx
i = idx[0]
ind = lines[i][:len(lines[i]) - len(lines[i].lstrip())]
new = [
    ind + '# r1zero patch (11/20): spread the n samples of every prompt over the rollout workers (worker k gets',
    ind + '# samples k, k+W, ...), so every SGLang engine gets the same mix of easy and hard prompts and the',
    ind + '# engines finish together; the permutation is undone right after generation.',
    ind + '_r1z_w = self.actor_rollout_wg.world_size',
    ind + '_r1z_n = len(gen_batch_output)',
    ind + 'if _r1z_w > 1 and _r1z_n % _r1z_w == 0:',
    ind + '    _r1z_perm = torch.arange(_r1z_n).view(-1, _r1z_w).t().reshape(-1)',
    ind + '    gen_batch_output.reorder(_r1z_perm)',
    ind + '    gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch_output)',
    ind + '    gen_batch_output.reorder(torch.argsort(_r1z_perm))',
    ind + 'else:',
    ind + '    gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch_output)',
]
lines[i:i + 1] = new
out = '\n'.join(lines)
ast.parse(out)
open(DST, 'w').write(out)
print('wrote', DST, 'replacing line', i + 1)
