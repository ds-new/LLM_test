r"""R1-Zero 奖励 v2（审计后修正版）。判分原则不变：只看"最终答案"这一处，答对 +1，否则 -1。
相对 v1 修了审计发现的漏判：
  1. 答案行外面的 markdown/LaTeX 包装（**、\( \)、\[ \]、$、重复的 "Answer:"）先剥掉再比较；
  2. "**Answer:**" 后换行再写 \boxed{..}：答案行为空时取它后面的 \boxed / 下一行；
  3. 最后一个 \boxed 出现在最后一个 Answer 行之后时，以 \boxed 为最终答案；
  4. 文字答案（\text{east}、Navin、ellipse）按不区分大小写的字符串比较；
  5. Unicode ½ π 等转成 LaTeX；带分数按字符串规范化比较。
仍然只取一个最终答案，不会因为前文出现过正确答案而判对（防止押多个答案）。"""
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # verl 按文件路径加载本模块时，保证能 import 同目录的 reward.py
from verl.utils.reward_score import math_dapo  # noqa: E402
import reward as V1  # noqa: E402

_UNI = {'½': '\\frac{1}{2}', '⅓': '\\frac{1}{3}', '¼': '\\frac{1}{4}', '¾': '\\frac{3}{4}', 'π': '\\pi', '√': '\\sqrt', '−': '-', '∞': '\\infty'}

def _strip(s):
    s = s.strip()
    for _ in range(4):
        s0 = s
        s = re.sub(r'^(\*\*|__|Answer\s*:|answer\s*:|\\\(|\\\[|\$)\s*', '', s).strip()
        s = re.sub(r'\s*(\*\*|__|\\\)|\\\]|\$|\.)$', '', s).strip()
        if s == s0: break
    s = re.sub(r'\\[()\[\]]', '', s).strip()
    s = re.sub(r'\s*(miles?|units?|cm|meters?|inches|feet|dollars?|cents?|degrees?|square units)\.?$', '', s).strip()
    for k, v in _UNI.items(): s = s.replace(k, v)
    return s

def _boxed_spans(s):
    out = []
    for m in re.finditer(r'\\boxed\s*\{', s):
        j, d = m.end(), 1
        while j < len(s) and d:
            d += {'{': 1, '}': -1}.get(s[j], 0); j += 1
        if d == 0: out.append((m.start(), s[m.end():j - 1]))
    return out

def final_answer(s):
    """返回 (答案, 来源)。只取一个。"""
    tail = s[-800:]
    off = len(s) - len(tail)
    ms = list(re.finditer(r'Answer\s*:\s*\**[ \t]*(.*)', tail))
    bx = _boxed_spans(s)
    if ms:
        m = ms[-1]; pos = off + m.start()
        cand = _strip(m.group(1))
        after = [b for b in bx if b[0] > pos]
        if after:  # Answer 行之后还有 \boxed，以最后一个 boxed 为准
            return _strip(after[-1][1]), 'boxed_after_answer'
        if cand and cand not in ('**',):
            if '\\boxed' in cand:
                b = _boxed_spans(cand)
                if b: return _strip(b[-1][1]), 'answer_line_boxed'
            return cand, 'answer_line'
        nxt = [l for l in tail[m.end():].split('\n') if l.strip()]
        if nxt: return _strip(nxt[0]), 'answer_next_line'
        return None, 'empty_answer'
    if bx and bx[-1][0] >= len(s) - 800:
        return _strip(bx[-1][1]), 'boxed_only'
    return None, 'none'

def _norm_txt(s):
    s = re.sub(r'\\(text|mathrm|textbf|mbox|operatorname)\s*\{([^}]*)\}', r'\2', s)
    s = s.replace('\\dfrac', '\\frac').replace('\\tfrac', '\\frac').replace('\\left', '').replace('\\right', '')
    s = s.replace('^\\circ', '').replace('^{\\circ}', '').replace('\\!', '').replace('\\,', '').replace('\\ ', '').replace('\\%', '').replace('%', '')
    s = re.sub(r'\s+', '', s).strip('.').lower()
    s = re.sub(r'^\(?([a-z])\)?$', r'\1', s)
    s = re.sub(r'^(-?\d+)\.0+$', r'\1', s)
    return s

def compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    gt = str(ground_truth)
    r = math_dapo.compute_score(solution_str, gt)
    if r.get('acc'):
        return {'score': 1.0, 'acc': 1.0, 'pred': str(r.get('pred'))[:100], 'src': 'math_dapo'}
    pred, how = final_answer(solution_str)
    ok = False
    if pred:
        ok = _norm_txt(pred) == _norm_txt(gt) or V1._mv_equal(pred, gt)
    return {'score': 1.0 if ok else -1.0, 'acc': float(ok), 'pred': str(pred)[:100], 'src': how}
