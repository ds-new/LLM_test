"""在独立进程中对单个(结构, 精度, micro batch)测速，结果以一行 BENCH_RESULT <json> 输出。"""

import json
import sys

import torch

from .architect_agent import ArchitectAgent


def main():
    args = json.loads(sys.argv[1])
    agent = ArchitectAgent({}, None)
    cfg = agent.model_config(args['name'], args['vocab'], args['seq_len'])
    try:
        res = agent._bench_one(torch.device(args['device']), cfg, args['precision'], args['micro'], args['seq_len'])
        res['ok'] = True
    except Exception as exc:
        res = {'ok': False, 'error': f'{type(exc).__name__}: {str(exc)[:300]}'}
    print('BENCH_RESULT ' + json.dumps(res), flush=True)


if __name__ == '__main__':
    main()
