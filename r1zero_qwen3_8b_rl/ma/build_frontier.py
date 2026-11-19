"""第三轮 D（边界题实验）的数据准备（2026-11-17 晚写，2026-11-18 由 loop3 调用）。
  make_profile()：从 train_dapo_r3.parquet 随机抽 2,048 道不同的题，每题重复 8 行 -> run/data/profile_r3.parquet（离线测通过率用）。
  select(valgen 文件)：按每题 8 次里做对几次分组 -> run/data/train_frontier.parquet：
      * 边界题：做对 1–5 次（通过率 12.5%–62.5%），全部保留；
      * 难题：8 次全错，随机保留一部分，最多占训练集 30%（n=32 时这类题有机会出现正确样本）；
      * 简单题：做对 6 次及以上，去掉。
  结果统计写 run/audit/frontier_stats.json。"""
import collections, json, sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "run" / "data"
PROFILE = DATA / "profile_r3.parquet"
OUT = DATA / "train_frontier.parquet"
STATS = ROOT / "run" / "audit" / "frontier_stats.json"
N_PROBLEMS, REPS, SEED = 2048, 8, 0
HARD_SHARE = 0.30
START, END = "the answer to the problem.\n\n", "\n\nRemember to put your answer"


def qkey(text):
    """从提示词里取出题目正文（parquet 的 content 和验证生成里的 input 都适用）。"""
    i = text.find(START)
    j = text.rfind(END)
    return text[i + len(START):j].strip() if i >= 0 and j > i else None


def make_profile():
    tr = pd.read_parquet(DATA / "train_dapo_r3.parquet")
    tr = tr[~tr.extra_info.map(lambda e: e["question"]).duplicated()].reset_index(drop=True)
    idx = np.random.default_rng(SEED).choice(len(tr), size=N_PROBLEMS, replace=False)
    rows = []
    for i in idx:
        r = tr.iloc[int(i)]
        for rep in range(REPS):
            ei = dict(r["extra_info"]); ei.update(rep=rep, profile_row=int(i))
            rows.append(dict(data_source=r["data_source"], prompt=r["prompt"], ability=r["ability"], reward_model=r["reward_model"], extra_info=ei))
    pd.DataFrame(rows).to_parquet(PROFILE)
    keys = {qkey(r["prompt"][0]["content"]) for r in rows}
    return dict(problems=N_PROBLEMS, rows=len(rows), unique_keys=len(keys), none_keys=int(None in keys))


def select(valgen_file):
    prof = pd.read_parquet(PROFILE)
    first = prof[prof.extra_info.map(lambda e: e["rep"] == 0)].reset_index(drop=True)
    by_key = {qkey(r["prompt"][0]["content"]): r for _, r in first.iterrows()}
    cnt = collections.defaultdict(lambda: [0, 0])
    unmatched = 0
    for line in open(valgen_file, encoding="utf-8"):
        d = json.loads(line)
        k = qkey(d["input"])
        if k not in by_key:
            unmatched += 1
            continue
        cnt[k][0] += float(d.get("acc", 0)) > 0.5
        cnt[k][1] += 1
    dist = collections.Counter()
    frontier, hard, easy = [], [], []
    for k, (c, n) in cnt.items():
        dist["%d/%d" % (c, n)] += 1
        rate = c / n
        (hard if c == 0 else (frontier if rate <= 0.625 else easy)).append(k)
    rng = np.random.default_rng(SEED)
    n_hard = min(len(hard), int(len(frontier) * HARD_SHARE / (1 - HARD_SHARE)))
    hard_used = list(rng.choice(sorted(hard), size=n_hard, replace=False)) if n_hard else []
    keys = frontier + hard_used
    rng.shuffle(keys)
    rows = []
    for k in keys:
        r = by_key[k]
        ei = dict(r["extra_info"]); ei.pop("rep", None); ei["profile_pass"] = "%d/%d" % tuple(cnt[k])
        rows.append(dict(data_source=r["data_source"], prompt=r["prompt"], ability=r["ability"], reward_model=r["reward_model"], extra_info=ei))
    pd.DataFrame(rows).to_parquet(OUT)
    stats = dict(valgen=str(valgen_file), problems_scored=len(cnt), unmatched_lines=unmatched,
                 easy=len(easy), frontier=len(frontier), hard=len(hard), hard_used=len(hard_used), n_train=len(rows),
                 mean_pass=float(np.mean([c / n for c, n in cnt.values()])) if cnt else None,
                 dist=dict(sorted(dist.items(), key=lambda x: (int(x[0].split("/")[0]), x[0]))))
    STATS.parent.mkdir(parents=True, exist_ok=True)
    STATS.write_text(json.dumps(stats, ensure_ascii=False, indent=1), encoding="utf-8")
    return stats


if __name__ == "__main__":
    if sys.argv[1:2] == ["profile"]:
        print(make_profile())
    elif sys.argv[1:2] == ["select"]:
        print(json.dumps(select(sys.argv[2]), ensure_ascii=False, indent=1))
