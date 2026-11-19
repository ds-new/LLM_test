"""数据智能体：把 DAPO-Math-17k / X-R1-750 / AIME25 / AIME24 / MATH-500 / AMC23 转成 verl 格式。

* 提示词统一用 DAPO 模板（最后一行写 "Answer: ..."），奖励用 verl 自带 math_dapo 判分 + math-verify 兜底；
* DAPO-Math-17k 原文件同一题重复约 100 次：按题面去重；
* 训练集与所有评测集去重：评测题面里任意 64 字符的规范化片段出现在训练题面中即剔除；
* 验证集：AIME25 / AIME24 / AMC23 每题复制 16 份（n=1 采样即等价于 avg@16），MATH-500 1 份。
"""
import random
import re
from pathlib import Path

from .common import DATA, RAW

NAME = "data"
TEMPLATE = ("Solve the following math problem step by step. The last line of your response should be of the form "
            "Answer: $Answer (without quotes) where $Answer is the answer to the problem.\n\n{q}\n\n"
            "Remember to put your answer on its own line after \"Answer:\".")


def _read(p):
    import pandas as pd
    p = Path(p)
    return pd.read_parquet(p) if p.suffix == ".parquet" else pd.read_json(p, lines=True)


def _pick(row, keys):
    for k in keys:
        if k in row and row[k] is not None and str(row[k]).strip():
            return str(row[k]).strip()
    return None


def _last_boxed(s):
    i = s.rfind("\\boxed{")
    if i < 0:
        return None
    j, depth = i + 7, 1
    while j < len(s) and depth:
        depth += {"{": 1, "}": -1}.get(s[j], 0)
        j += 1
    return s[i + 7:j - 1].strip() if depth == 0 else None


def _norm(s):
    return re.sub(r"[^a-z0-9]", "", s.lower())


def _rec(q, ans, source, split, idx, rep=0):
    return {"data_source": source, "prompt": [{"role": "user", "content": TEMPLATE.format(q=q)}],
            "ability": "math", "reward_model": {"style": "rule", "ground_truth": str(ans)},
            "extra_info": {"split": split, "index": int(idx), "rep": int(rep), "question": q}}


def _write(rows, path):
    import pandas as pd
    pd.DataFrame(rows).to_parquet(path, index=False)


class DataAgent:
    def __init__(self, board):
        self.board = board

    def eval_sets(self):
        specs = [("aime25", RAW / "math-ai_aime25/test.jsonl", 16), ("aime24", RAW / "math-ai_aime24/test-00000-of-00001.parquet", 16),
                 ("amc23", RAW / "math-ai_amc23/test-00000-of-00001.parquet", 16), ("math500", RAW / "HuggingFaceH4_MATH-500/test.jsonl", 1)]
        out = {}
        for name, p, rep in specs:
            items = []
            for i, r in _read(p).iterrows():
                r = r.to_dict()
                q = _pick(r, ["problem", "question", "Problem", "prompt"])
                a = _pick(r, ["answer", "Answer", "final_answer"])
                if a is None:
                    sol = _pick(r, ["solution", "Solution"])
                    a = _last_boxed(sol) if sol else None
                if q and a:
                    items.append((q, a))
            out[name] = (items, rep)
        return out

    def build(self):
        evals = self.eval_sets()
        # 评测题面的 64 字符片段，用于训练集去重
        shingles = set()
        for items, _ in evals.values():
            for q, _a in items:
                n = _norm(q)
                for k in range(0, max(1, len(n) - 63), 8):
                    shingles.add(n[k:k + 64])

        def leaked(q):
            n = _norm(q)
            return any(n[k:k + 64] in shingles for k in range(0, max(1, len(n) - 63)))

        stats = {}
        # DAPO-Math-17k
        d = _read(RAW / "BytedTsinghua-SIA_DAPO-Math-17k/data/dapo-math-17k.parquet")
        seen, train, drop_leak = set(), [], 0
        for i, r in d.iterrows():
            content = r["prompt"][0]["content"]
            m = re.search(r"to the problem\.\n\n(.*)\n\nRemember to put", content, flags=re.S)
            q = m.group(1).strip() if m else content.strip()
            key = _norm(q)
            if key in seen:
                continue
            seen.add(key)
            if leaked(q):
                drop_leak += 1
                continue
            train.append(_rec(q, r["reward_model"]["ground_truth"], "dapo_math", "train", len(train)))
        random.Random(0).shuffle(train)
        _write(train, DATA / "train_dapo.parquet")
        stats["dapo_rows"], stats["dapo_unique"], stats["dapo_leak_dropped"], stats["dapo_train"] = len(d), len(seen), drop_leak, len(train)
        # X-R1-750（冒烟测试）
        xr = []
        for i, r in _read(RAW / "xiaodongguaAIGC_X-R1-750/data/train-00000-of-00001.parquet").iterrows():
            r = r.to_dict()
            q = _pick(r, ["problem", "question", "prompt"])
            a = _pick(r, ["answer"]) or (_last_boxed(_pick(r, ["solution"]) or "") if _pick(r, ["solution"]) else None)
            if q and a and not leaked(q):
                xr.append(_rec(q, a, "xr750", "train", i))
        _write(xr, DATA / "train_xr750.parquet")
        stats["xr750_train"] = len(xr)
        # 验证集
        val, small = [], []
        for name, (items, rep) in evals.items():
            stats[f"{name}_n"] = len(items)
            for i, (q, a) in enumerate(items):
                for k in range(rep):
                    val.append(_rec(q, a, name, "test", i, k))
                if name == "aime25":
                    small.append(_rec(q, a, name, "test", i))
        _write(val, DATA / "val.parquet")
        _write(small, DATA / "val_small.parquet")
        stats["val_rows"] = len(val)
        self.board.set(data=stats)
        self.board.post(NAME, "built", stats)
        ok = stats["dapo_train"] > 10000 and stats["xr750_train"] > 500 and all(stats[f"{n}_n"] > 0 for n in evals)
        return ok, stats


    # ---------------- 第二轮：评选集 ----------------
    SELECT = {"math500": 200, "amc23": 80, "aime24": 60, "aime25": 60}

    def build_select(self, seed=0):
        """从 val.parquet 抽评选集：MATH-500 随机 200 题；AMC23/AIME24/AIME25 每题取前 2 份复制（avg@2）。共 400 条。
        每 2 步用它验证一次挑最佳 checkpoint；完整的 val.parquet 留到训练结束对最佳/最终模型各跑一次。"""
        import collections
        import pyarrow as pa
        import pyarrow.parquet as pq
        rows = pq.read_table(DATA / "val.parquet").to_pylist()
        by = collections.defaultdict(list)
        for r in rows:
            by[r["data_source"]].append(r)
        out = random.Random(seed).sample(by["math500"], self.SELECT["math500"])
        for name in ("amc23", "aime24", "aime25"):
            out += [r for r in by[name] if int(r["extra_info"]["rep"]) < 2]
        pq.write_table(pa.Table.from_pylist(out), DATA / "val_select.parquet")
        stats = dict(collections.Counter(r["data_source"] for r in out))
        self.board.set(select_set=stats)
        self.board.post(NAME, "select_built", stats)
        return stats == self.SELECT, stats
