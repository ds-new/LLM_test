"""2026-11-21 r1zero 补丁：给每个 SGLang 生成引擎各自的采样种子。
原因：verl 的 megatron worker 在每个 rank 上都用同一个种子调用 random.seed(megatron.seed)，SGLang 没收到 random_seed 时用
random.randint 取一个，所以 4 个引擎的采样种子相同；11/20 加的"把每题的 n 个回答分散到各引擎"补丁又让 4 个引擎收到完全相同的请求列表，
结果 4 个引擎生成完全相同的回答：F 第 1-10 步每题 16 个回答其实是 4 个不同回答各重复 4 次（第 10 步验证给各引擎不同的题，随机状态
从此分开，第 11 步起正常）。修法：random_seed = R1Z_SGLANG_SEED（默认 20261121）+ 1009 * rank。幂等；原文件备份为 .orig_seed。
用法：python f/patches/fix_sglang_seed_1121.py [目标文件，默认 site-packages 里的 sglang_rollout.py]"""
import shutil
import sys

P = sys.argv[1] if len(sys.argv) > 1 else '/root/miniconda3/lib/python3.12/site-packages/verl/workers/rollout/sglang_rollout/sglang_rollout.py'
MARK = 'r1zero patch (11/21): per-rank SGLang sampling seed'
s = open(P).read()
if MARK in s:
    print('already patched')
    sys.exit(0)
anchor = '                "port": sglang_port,\n'
assert s.count(anchor) == 1, s.count(anchor)
new = ('                # ' + MARK + ' (same seed on every rank + identical request lists gave identical samples)\n'
       '                "random_seed": int(os.getenv("R1Z_SGLANG_SEED", "20261121")) + 1009 * dist.get_rank(),\n' + anchor)
shutil.copy(P, P + '.orig_seed')
open(P, 'w').write(s.replace(anchor, new))
print('patched', P)
