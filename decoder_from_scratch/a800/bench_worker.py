"""在独立进程中测一个 (结构, micro batch) 组合的训练吞吐，输出一行 BENCH <json>。"""

import json
import sys

from .architect_agent import bench


def main():
    a = json.loads(sys.argv[1])
    try:
        res = bench(a['name'], a['micro'], a['seq_len'], a['vocab'], a['compile'])
    except Exception as exc:  # 显存不足等
        res = {'ok': False, 'error': f'{type(exc).__name__}: {str(exc)[:300]}'}
    print('BENCH ' + json.dumps(res), flush=True)


if __name__ == '__main__':
    main()
