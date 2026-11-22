import shutil, sys, ast
P = '/root/miniconda3/lib/python3.12/site-packages/verl/utils/checkpoint/megatron_checkpoint_manager.py'
src = open(P).read()
MARK = 'r1zero patch (11/20): loaded checkpoint counts as previously saved'
if MARK in src:
    print('already patched'); sys.exit(0)
anchor = '    def save_checkpoint(self, local_path: str, hdfs_path: str = None, global_step: int = 0, max_ckpt_to_keep=None):'
assert src.count(anchor) == 1, 'anchor not unique'
add = ('        # ' + MARK + ', so max_ckpt_to_keep also removes it at the next save after a resume\n'
       '        # (otherwise the old 16 GB checkpoint stays and the next save overflows the data disk)\n'
       '        if local_path is not None and local_path not in self.previous_saved_paths:\n'
       '            self.previous_saved_paths.append(local_path)\n'
       '\n')
shutil.copy(P, P + '.orig_resumeclean')
new = src.replace(anchor, add + anchor)
ast.parse(new)
open(P, 'w').write(new)
print('patched')
