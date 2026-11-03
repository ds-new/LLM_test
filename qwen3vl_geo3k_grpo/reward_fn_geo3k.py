"""
geometry3k 多模态几何题奖励函数（verl custom reward）。

    custom_reward_function.path=reward_fn_geo3k.py
    custom_reward_function.name=compute_score

相对旧版本的修改（都是会直接影响 GRPO 信号质量的问题）：
  1. 只从最后一个 \\boxed{} 取答案。旧版在没有 \\boxed 时用正则匹配推理过程里的第一个
     "= 数字"，推理中任何中间结果碰巧等于答案都会得分，等于奖励"不写最终答案"。
     严格模式（默认）下没有 \\boxed{} 直接 0 分。
  2. math-verify 只解析 \\boxed{} 里的内容，不再解析整段回答（避免匹配到推理中的式子）。
  3. 规范化 geometry3k 常见写法：度数符号、\\text{单位}、\\dfrac、\\left/\\right、"x=3"。
  4. 自带 LaTeX 数值求值（\\frac、\\sqrt、\\pi、乘方、隐式乘法），2\\sqrt{221} 与 \\sqrt{884}
     判为相等；保留至少 2 位小数且四舍五入正确的小数也算对（GEO3K_ACCEPT_ROUNDED=0 可关闭）。
  5. 返回 dict：score 之外单独给出 acc / format / boxed，verl 会把它们作为
     val-core/<data_source>/acc/mean@1 等指标记录，训练监控直接看准确率而不是混了格式分的 score。

可用环境变量（多智能体 loop 按算法预设设置）：
  GEO3K_FORMAT_BONUS   格式分系数，默认 0.1；0 表示纯准确率奖励
  GEO3K_STRICT_BOXED   1（默认）没有 \\boxed{} 不得分
  GEO3K_ACCEPT_ROUNDED 1（默认）接受正确四舍五入到 >=2 位小数的数值
"""

import ast
import math
import operator
import os
import re

FORMAT_BONUS = 0.1  # 兼容旧代码引用；实际值以环境变量为准


def _cfg():
    return {
        "format_bonus": float(os.environ.get("GEO3K_FORMAT_BONUS", str(FORMAT_BONUS))),
        "strict": os.environ.get("GEO3K_STRICT_BOXED", "1") != "0",
        "rounded": os.environ.get("GEO3K_ACCEPT_ROUNDED", "1") != "0",
    }


# ---------------------------------------------------------------------------
# \boxed{} 提取
# ---------------------------------------------------------------------------
def _last_boxed(s: str):
    idx = max(s.rfind("\\boxed"), s.rfind("\\fbox"))
    if idx < 0:
        return None
    i = s.find("{", idx)
    if i < 0:
        return None
    depth = 0
    for j in range(i, len(s)):
        if s[j] == "{":
            depth += 1
        elif s[j] == "}":
            depth -= 1
            if depth == 0:
                return s[i + 1: j]
    return None


def _extract_boxed_answer(text: str) -> str:
    return (_last_boxed(text or "") or "").strip()


# ---------------------------------------------------------------------------
# 规范化
# ---------------------------------------------------------------------------
_UNIT_WORDS = r"(?:units?|cm|mm|m|km|in|inches|ft|feet|yd|yards|degrees?|deg|sq|square)"


def _strip_text_cmds(s):
    # 数字后面的 \text{ cm} 之类视为单位去掉；整体被 \text{} 包住时保留内容
    s = re.sub(r"(?<=[0-9}\s])\\(?:text|mathrm|mbox|textbf)\s*\{\s*" + _UNIT_WORDS + r"[^{}]*\}", "", s)
    return re.sub(r"\\(?:text|mathrm|mbox|textbf|operatorname)\s*\{([^{}]*)\}", r"\1", s)


def normalize_answer(s: str) -> str:
    if s is None:
        return ""
    s = str(s).strip()
    s = s.replace("$", "").replace("\\!", "").replace("\\,", "").replace("\\;", "").replace("\\ ", "")
    s = re.sub(r"\\(?:left|right|displaystyle)", "", s)
    s = re.sub(r"\\[dt]frac", r"\\frac", s)
    s = re.sub(r"\^\s*\{?\s*\\circ\s*\}?", "", s)
    s = s.replace("°", "").replace("\\degree", "").replace("\\circ", "")
    s = _strip_text_cmds(s)
    # "x = 3" / "m\angle 1 = 30" -> 取最后一个等号右侧
    if "=" in s and not re.search(r"[<>]=|=[<>]", s):
        s = s.split("=")[-1]
    s = re.sub(r"\s+" + _UNIT_WORDS + r"\s*$", "", s.strip())
    s = s.replace(" ", "").replace(",", "").lower().rstrip(".")
    return s


# ---------------------------------------------------------------------------
# LaTeX 数值求值（受限语法，安全 eval）
# ---------------------------------------------------------------------------
def _brace(s, i):
    """s[i] == '{'，返回 (内容, 结束位置后一位)。"""
    depth = 0
    for j in range(i, len(s)):
        if s[j] == "{":
            depth += 1
        elif s[j] == "}":
            depth -= 1
            if depth == 0:
                return s[i + 1: j], j + 1
    raise ValueError("unbalanced")


def _latex_to_py(s: str) -> str:
    out, i = [], 0
    while i < len(s):
        if s.startswith("\\frac", i):
            j = i + 5
            num, j = _brace(s, j)
            den, j = _brace(s, j)
            out.append(f"(({_latex_to_py(num)})/({_latex_to_py(den)}))")
            i = j
        elif s.startswith("\\sqrt", i):
            j = i + 5
            root = None
            if j < len(s) and s[j] == "[":
                k = s.index("]", j)
                root, j = s[j + 1: k], k + 1
            if j < len(s) and s[j] == "{":
                body, j = _brace(s, j)
            else:  # \sqrt3
                m = re.match(r"[0-9.]+", s[j:])
                if not m:
                    raise ValueError("bad sqrt")
                body, j = m.group(0), j + len(m.group(0))
            body = _latex_to_py(body)
            out.append(f"(({body})**(1/({_latex_to_py(root)})))" if root else f"sqrt({body})")
            i = j
        elif s.startswith("\\pi", i):
            out.append("pi")
            i += 3
        elif s.startswith("\\cdot", i) or s.startswith("\\times", i):
            out.append("*")
            i += 5 if s.startswith("\\cdot", i) else 6
        elif s[i] == "{":
            body, i = _brace(s, i)
            out.append(f"({_latex_to_py(body)})")
        elif s[i] == "^":
            out.append("**")
            i += 1
        elif s[i] == "\\":
            raise ValueError("unsupported command")
        else:
            out.append(s[i])
            i += 1
    expr = "".join(out)
    # 隐式乘法：2sqrt(3) / 2pi / )( / 3(…)
    expr = re.sub(r"(?<=[0-9)])(?=(sqrt|pi|\())", "*", expr)
    expr = re.sub(r"(?<=pi)(?=[0-9(]|sqrt)", "*", expr)
    return expr


_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
        ast.Pow: operator.pow, ast.USub: operator.neg, ast.UAdd: operator.pos}


def _safe_eval(node):
    if isinstance(node, ast.Expression):
        return _safe_eval(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if isinstance(node, ast.Name) and node.id == "pi":
        return math.pi
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        a, b = _safe_eval(node.left), _safe_eval(node.right)
        if isinstance(node.op, ast.Pow) and abs(b) > 16:
            raise ValueError("power too large")
        return _OPS[type(node.op)](a, b)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_safe_eval(node.operand))
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "sqrt" and len(node.args) == 1:
        v = _safe_eval(node.args[0])
        if v < 0:
            raise ValueError("sqrt of negative")
        return math.sqrt(v)
    raise ValueError("disallowed expression")


def latex_value(s: str):
    """把规范化后的答案求成 float；无法求值返回 None。"""
    try:
        s = normalize_answer(s)
        if not s or len(s) > 80 or re.search(r"[a-oq-z]", s.replace("sqrt", "").replace("pi", "").replace("frac", "")):
            return None
        expr = _latex_to_py(s)
        v = _safe_eval(ast.parse(expr, mode="eval"))
        return v if math.isfinite(v) else None
    except Exception:
        return None


def _decimals(s: str):
    m = re.fullmatch(r"-?\d+\.(\d+)", normalize_answer(s))
    return len(m.group(1)) if m else 0


def _numeric_equal(pred: str, gt: str, accept_rounded: bool) -> bool:
    pv, gv = latex_value(pred), latex_value(gt)
    if pv is None or gv is None:
        return False
    if abs(pv - gv) <= 1e-6 * max(1.0, abs(gv)):
        return True
    d = _decimals(pred)
    # 正确的四舍五入（>=2 位小数），且标准答案本身不是整数
    if accept_rounded and d >= 2 and abs(gv - round(gv)) > 1e-9:
        return abs(pv - gv) <= 0.5 * 10 ** (-d) * 1.0001
    return False


# ---------------------------------------------------------------------------
# math-verify（可选）
# ---------------------------------------------------------------------------
try:
    from math_verify import parse as _mv_parse, verify as _mv_verify
    from math_verify.metric import LatexExtractionConfig
    _MATH_VERIFY_AVAILABLE = True
except Exception:  # noqa
    _MATH_VERIFY_AVAILABLE = False


def _math_verify_equal(pred: str, gt: str) -> bool:
    if not _MATH_VERIFY_AVAILABLE:
        return False
    try:
        cfg = [LatexExtractionConfig()]
        try:  # 新版支持关闭基于 signal 的超时（Ray 非主线程中 signal 不可用）
            g = _mv_parse("\\boxed{" + gt + "}", extraction_config=cfg, parsing_timeout=None)
            p = _mv_parse("\\boxed{" + pred + "}", extraction_config=cfg, parsing_timeout=None)
            return bool(_mv_verify(g, p, timeout_seconds=None))
        except TypeError:
            g = _mv_parse("\\boxed{" + gt + "}", extraction_config=cfg)
            p = _mv_parse("\\boxed{" + pred + "}", extraction_config=cfg)
            return bool(_mv_verify(g, p))
    except Exception:
        return False


# ---------------------------------------------------------------------------
# 判分
# ---------------------------------------------------------------------------
def answers_match(pred: str, gt: str, accept_rounded: bool = True) -> bool:
    if not pred or gt is None or str(gt).strip() == "":
        return False
    if normalize_answer(pred) == normalize_answer(gt):
        return True
    if _numeric_equal(pred, gt, accept_rounded):
        return True
    return _math_verify_equal(pred, str(gt))


def _check_format(solution_str: str) -> bool:
    """推理在 </think> 之前、\\boxed{} 在 </think> 之后。

    接受两种写法：完整的 <think>…</think>；或只有 </think>（Thinking 版模型的 chat template 已在 prompt
    末尾放了 <think>，生成内容从推理正文开始）。"""
    idx = solution_str.find("</think>")
    if idx < 0:
        return False
    head = solution_str[:idx]
    if head.count("<think>") > 1 or not head.replace("<think>", "").strip():
        return False
    return _last_boxed(solution_str[idx + len("</think>"):]) is not None


def grade(solution_str: str, ground_truth: str):
    """完整判分细节（评估脚本和清洗智能体也用这个）。"""
    cfg = _cfg()
    text = solution_str or ""
    pred = _extract_boxed_answer(text)
    boxed = _last_boxed(text) is not None
    if not boxed and not cfg["strict"]:
        tail = text.split("</think>")[-1]
        m = re.findall(r"(?:answer\s+is|=)\s*\$?([^\s$,]+)", tail, re.IGNORECASE)
        pred = m[-1] if m else ""
    acc = 1.0 if answers_match(pred, ground_truth, cfg["rounded"]) else 0.0
    fmt = 1.0 if _check_format(text) else 0.0
    return {"score": acc + cfg["format_bonus"] * fmt, "acc": acc, "format": fmt,
            "boxed": 1.0 if boxed else 0.0, "pred": pred}


def _check_accuracy(solution_str: str, ground_truth: str) -> bool:  # 兼容旧接口
    return grade(solution_str, ground_truth)["acc"] == 1.0


def compute_score(data_source=None, solution_str="", ground_truth="", extra_info=None, **kwargs):
    """verl 接口。返回 dict（score 为奖励；其余数值字段作为额外指标记录）。"""
    if "format_bonus" in kwargs:  # custom_reward_function.reward_kwargs 透传
        os.environ["GEO3K_FORMAT_BONUS"] = str(kwargs["format_bonus"])
    if not solution_str or ground_truth is None:
        return {"score": 0.0, "acc": 0.0, "format": 0.0, "boxed": 0.0}
    g = grade(solution_str, str(ground_truth))
    g.pop("pred")
    return g


def self_check_ground_truth(ground_truth: str) -> bool:
    """标准答案自身能否被判分器识别为正确（不能的话该题奖励永远为 0，应剔除）。"""
    return grade("<think>.</think> \\boxed{" + str(ground_truth) + "}", ground_truth)["acc"] == 1.0
