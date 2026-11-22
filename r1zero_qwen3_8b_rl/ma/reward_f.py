"""F 阶段（在 E 的 SFT 模型上做 RL）的判分：和 E 的评测（sft/eval_e.py）一致，只判 </think> 之后的部分；
没有 </think>（思考没写完，多半是被截断）记 -1。返回的键和 reward_v2 完全相同（DAPO 的 reward manager 要求每条一致）。"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from reward_v2 import compute_score as _v2  # noqa: E402


def compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    s = solution_str.replace('<|im_end|>', '').replace('<|endoftext|>', '')
    if '</think>' not in s:
        return {'score': -1.0, 'acc': 0.0, 'pred': '', 'src': 'no_think_end'}
    return _v2(data_source, s.split('</think>')[-1], ground_truth, extra_info=extra_info, **kwargs)
