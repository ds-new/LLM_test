"""R1-Zero 奖励：答对 +1，答错/没写答案/截断 -1（DAPO 的做法）。

先用 verl 自带 math_dapo（取最后一行 "Answer: ..."，minerva 规范化）判分；判错时再用 math-verify 对
"Answer:" 后的内容或最后一个 \\boxed{} 与标准答案做符号等价比较（MATH-500 的 LaTeX 答案需要）。

math-verify 底层是 sympy，遇到模型写出的 "1000000!"、巨大幂次之类会算到天荒地老（第一次正式训练就在训练前验证里
卡在 factorial 上）。所以：
  1. 输入保护：答案过长、含阶乘、超长数字、大指数时不做符号比较（真正的标准答案用不到这些）；
  2. 硬超时：parse+verify 放到 fork 出来的小进程池里做，2 秒没结果就杀掉进程池重建，按答错处理。
     （不用信号：像 (10^{10})^{(10^{10})} 这种会在一次 C 层大整数运算里卡住，Python 信号打断不了，只能杀进程。）
"""
import multiprocessing as mp
import re

from verl.utils.reward_score import math_dapo

try:
    from math_verify import parse as _mv_parse, verify as _mv_verify
    _MV = True
except Exception:  # noqa: BLE001
    _MV = False

MV_TIMEOUT = 2.0
# 阶乘、超长数字、4 位以上的指数、指数塔（^ 的指数里还有 ^）
_RISKY = re.compile(r"!|\d{41,}|\^\s*\{?\s*\(?\s*\d{4,}|\^\s*\{[^}]*\^|\^\s*\([^)]*\^|\\(?:binom|choose)\s*\{?\s*\d{4,}")
_POOL = None


def _mv_job(pred, gt):
    g = _mv_parse("$" + gt + "$", parsing_timeout=None)
    p = _mv_parse("$" + pred + "$", parsing_timeout=None)
    return bool(_mv_verify(g, p, timeout_seconds=None))


def _pool():
    global _POOL
    if _POOL is None:
        _POOL = mp.get_context("fork").Pool(4, maxtasksperchild=500)
    return _POOL


def _kill_pool():
    global _POOL
    if _POOL is not None:
        try:
            _POOL.terminate()
        except Exception:  # noqa: BLE001
            pass
    _POOL = None


def _extract(s: str):
    m = re.findall(r"Answer\s*:\s*(.+)", s[-600:])
    if m:
        return m[-1].strip().strip("$").strip()
    i = s.rfind("\\boxed{")
    if i >= 0:
        j, d = i + 7, 1
        while j < len(s) and d:
            d += {"{": 1, "}": -1}.get(s[j], 0)
            j += 1
        if d == 0:
            return s[i + 7:j - 1]
    return None


def _mv_equal(pred, gt):
    if not _MV or not pred:
        return False
    if len(pred) > 200 or _RISKY.search(pred):
        return False
    try:
        return bool(_pool().apply_async(_mv_job, (pred, gt)).get(timeout=MV_TIMEOUT))
    except mp.TimeoutError:
        _kill_pool()  # 卡住的子进程连同进程池一起杀掉，下次重建
        return False
    except Exception:  # noqa: BLE001
        return False


def compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    r = math_dapo.compute_score(solution_str, str(ground_truth))
    acc = bool(r.get("acc"))
    pred = r.get("pred")
    if not acc:
        p2 = _extract(solution_str)
        if p2 and _mv_equal(p2, str(ground_truth)):
            acc, pred = True, p2
    return {"score": 1.0 if acc else -1.0, "acc": float(acc), "pred": str(pred)[:100]}
