"""F 阶段的数据：题目和 R1-Zero 第三轮相同，提示改成 SFT / 评测时的格式
（题目 + 空行 + "Please reason step by step, and put your final answer within \\boxed{}."，chat 模板由 verl 套）。
输出 run/data/f_pool.parquet（第三轮训练题池）、run/data/f_profile.parquet（随机 2,048 题，用 E 的模型测通过率）、
run/data/val_select_f.parquet（400 条评选集，训练中每 10 步看一次）。"""
import random
import re

import pyarrow as pa
import pyarrow.parquet as pq

R = '/root/autodl-tmp/r1zero'
INSTR = 'Please reason step by step, and put your final answer within \\boxed{}.'


CJK = re.compile(r'[㐀-鿿]')


def conv(src, drop_cjk=False):
    rows = pq.read_table(src).to_pylist()
    if drop_cjk:  # SFT 数据去掉了题面含中文的题，F 也去掉，免得 RL 把回答带成中文
        rows = [r for r in rows if not CJK.search(r['extra_info']['question'])]
    for r in rows:
        q = r['extra_info']['question']
        assert isinstance(q, str) and q.strip(), r
        r['prompt'] = [{'role': 'user', 'content': q + '\n\n' + INSTR}]
    return rows


if __name__ == '__main__':
    pool = conv(R + '/run/data/train_dapo_r3.parquet', drop_cjk=True)
    pq.write_table(pa.Table.from_pylist(pool), R + '/run/data/f_pool.parquet')
    prof = list(pool)
    random.Random(20261120).shuffle(prof)
    pq.write_table(pa.Table.from_pylist(prof[:2048]), R + '/run/data/f_profile.parquet')
    val = conv(R + '/run/data/val_select.parquet')
    pq.write_table(pa.Table.from_pylist(val), R + '/run/data/val_select_f.parquet')
    print('pool', len(pool), 'profile', min(2048, len(prof)), 'val', len(val))
    print(repr(pool[0]['prompt'][0]['content'][-120:]))
