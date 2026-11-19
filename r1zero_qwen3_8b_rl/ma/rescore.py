"""用修正版判分 reward_v2 给验证生成（/root/r1zero_valgen/*.jsonl）重新打分。
审计（run/audit/AUDIT.md）发现原判分 reward.py 会把约 2.4% 的正确答案判错，所以挑选最佳、最终报告都以 v2 为准；
同时给出"去掉泄露题"的口径（训练集里有与评测集重复的 6 道题）。"""
import json, sys, time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
import reward_v2  # noqa: E402

DATA = _HERE.parent / "run" / "data"
VALGEN_DIR = Path("/root/r1zero_valgen")
WEIGHTS = {"math500": 200, "amc23": 80, "aime24": 60, "aime25": 60}
LEAK_KEYS = ["8^5 \\cdot 5^{10} \\cdot 15^5", "\\log_{7x}2023", "u \\otimes v = ac + bdi",
             "y=ax+c$, $y=ax+d$", "\\text{mod}_5 (n)", "F(z)=\\frac{z+i}{z-i}"]
_SRC = None


def _src_map():
    global _SRC
    if _SRC is None:
        import pandas as pd
        _SRC = {}
        for f in ("val_select.parquet", "val.parquet", "test_aime26.parquet"):
            if (DATA / f).exists():
                for _, r in pd.read_parquet(DATA / f).iterrows():
                    _SRC[r["prompt"][0]["content"][-200:]] = r["data_source"]
    return _SRC


def score_file(path):
    """见 _score_file。判分用的子进程池用完就关：留着的话，loop 被强杀后这些子进程会变成孤儿，
    看门狗按进程名会误以为 loop 还在跑。"""
    try:
        return _score_file(path)
    finally:
        reward_v2.V1._kill_pool()


def _score_file(path):
    """返回 {'n':{src:n}, 'v1':{src:acc}, 'v2':{src:acc}, 'v2_clean':{src:acc}, 'score_v1', 'score_v2', 'score_v2_clean'}；
    score_* 按 WEIGHTS 里的数据集样本数加权（与评选集口径一致）。"""
    sm = _src_map()
    n, a1, a2, nc, a2c = {}, {}, {}, {}, {}
    for line in open(path, encoding="utf-8"):
        d = json.loads(line)
        inp = d["input"]
        src = next((s for k, s in sm.items() if k in inp), "unknown")
        acc1 = float(d.get("acc", 0)) > 0.5
        acc2 = reward_v2.compute_score(src, d["output"], d["gts"])["acc"] > 0.5
        leak = any(k in inp for k in LEAK_KEYS)
        n[src] = n.get(src, 0) + 1; a1[src] = a1.get(src, 0) + acc1; a2[src] = a2.get(src, 0) + acc2
        if not leak:
            nc[src] = nc.get(src, 0) + 1; a2c[src] = a2c.get(src, 0) + acc2

    def agg(num, den):
        t = sum(num.get(s, 0) for s in WEIGHTS if s in den); w = sum(den[s] for s in WEIGHTS if s in den)
        return round(t / w, 4) if w else None
    return {"n": n, "v1": {s: round(a1[s] / n[s], 4) for s in n}, "v2": {s: round(a2[s] / n[s], 4) for s in n},
            "v2_clean": {s: round(a2c.get(s, 0) / nc[s], 4) for s in nc},
            "score_v1": agg(a1, n), "score_v2": agg(a2, n), "score_v2_clean": agg(a2c, nc)}


def step_file(step, since_ts=None, d=VALGEN_DIR):
    """第 step 步的验证生成文件；since_ts（unix 时间）之前写的视为上一条训练线的旧文件，不用。"""
    p = Path(d) / f"{step}.jsonl"
    if not p.exists() or (since_ts and p.stat().st_mtime < since_ts):
        return None
    if time.time() - p.stat().st_mtime < 30:
        return None  # 可能还在写
    return p


def newest_file(d):
    fs = sorted(Path(d).glob("*.jsonl"), key=lambda p: p.stat().st_mtime)
    return fs[-1] if fs else None
