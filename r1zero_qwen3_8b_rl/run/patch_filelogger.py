# 2026-11-20 r1zero 补丁：verl FileLogger 写指标时遇到 torch.Tensor / numpy 值会 json.dumps 报错（F 冒烟第 1 步存完盘后崩在这里）。
# 改成把张量和 numpy 转成 Python 数，其他转成字符串；幂等，原文件备份为 tracking.py.orig_filelogger。
import shutil, sys
p = '/root/miniconda3/lib/python3.12/site-packages/verl/utils/tracking.py'
s = open(p).read()
old = '        self.fp.write(json.dumps(data) + "\\n")'
new = '        self.fp.write(json.dumps(data, default=_r1z_jsonable) + "\\n")\n        self.fp.flush()'
helper = '''

def _r1z_jsonable(o):  # r1zero patch: tensors / numpy scalars in metrics
    try:
        if hasattr(o, "numel") and o.numel() == 1:
            return o.item()
        if hasattr(o, "item") and getattr(o, "size", 2) == 1:
            return o.item()
        if hasattr(o, "tolist"):
            return o.tolist()
    except Exception:
        pass
    return str(o)
'''
if '_r1z_jsonable' in s:
    print('already patched'); sys.exit(0)
assert s.count(old) == 1, s.count(old)
shutil.copy(p, p + '.orig_filelogger')
s = s.replace(old, new)
s = s.replace('\nclass FileLogger:', helper + '\n\nclass FileLogger:', 1)
open(p, 'w').write(s)
print('patched')
