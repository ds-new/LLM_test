"""拒绝采样微调（RFT）的数据构造：从模型自己的采样里挑「答对、没被截断」的解法。

输入：evaluate_geo3k.py --save_samples 写出的 jsonl（每题 n 条完整输出 + 对错）与对应的 verl 格式题目文件。
输出：SFT 用 jsonl，每行 {"index", "question", "response"}；图片由 sft_qwen3vl.py 按 index 从题目文件取。

挑选规则：
  * 只要答对、写出了 \\boxed{} 且没被截断的样本（绕圈被截断的一律不要）；
  * 同一题去重后按长度从短到长，最多保留 --per_item 条（默认 2）：偏好简洁的解法，顺带教模型别绕；
  * --max_chars 过滤极长的正确解（默认 12000 字符，约 3500 token），避免学到啰嗦写法。
  * --balanced：按难度平衡（第一版只取最短解，偏向简单题，实测准确率没涨）。n 次采样全对的题最多留
    --per_item_easy 条（默认 1），有对有错的题最多留 --per_item_mixed 条（默认 4），在正确解里随机挑。

用法：
  python rft_build.py --samples rft/train_samples.jsonl --data data/clean/train.parquet --out rft/sft.jsonl
"""

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from geo3k_data import read_split  # noqa: E402


def build(samples_path, data_path, out_path, per_item=2, max_chars=12000, balanced=False, per_item_easy=1,
          per_item_mixed=4, seed=0):
    rng = random.Random(seed)
    rows = read_split(data_path)
    by_index = {r["extra_info"].get("index", i): r for i, r in enumerate(rows)}
    stats = Counter()
    out = []
    with open(samples_path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            s = json.loads(line)
            stats["items"] += 1
            r = by_index.get(s["index"])
            if r is None:
                stats["missing_item"] += 1
                continue
            good, seen = [], set()
            for t, ok, tr, bx in zip(s["texts"], s["correct"], s["truncated"], s["boxed"]):
                stats["samples"] += 1
                if not ok:
                    stats["wrong"] += 1
                    continue
                if tr or not bx:
                    stats["truncated_or_unboxed"] += 1
                    continue
                if len(t) > max_chars:
                    stats["too_long"] += 1
                    continue
                key = " ".join(t.split())
                if key in seen:
                    continue
                seen.add(key)
                good.append(t)
            if balanced:
                n_ok = sum(bool(c) for c in s["correct"])
                cap = per_item_easy if n_ok == len(s["correct"]) else per_item_mixed
                stats["easy_items" if n_ok == len(s["correct"]) else "mixed_items"] += bool(good)
                rng.shuffle(good)
                kept = good[:cap]
            else:
                good.sort(key=len)
                kept = good[:per_item]
            stats["items_with_kept"] += bool(kept)
            for t in kept:
                out.append({"index": s["index"], "question": r["extra_info"]["question"], "response": t.strip()})
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for o in out:
            f.write(json.dumps(o, ensure_ascii=False) + "\n")
    lens = sorted(len(o["response"]) for o in out)
    stats["examples"] = len(out)
    stats["median_chars"] = lens[len(lens) // 2] if lens else 0
    return dict(stats)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--samples", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--per_item", type=int, default=2)
    ap.add_argument("--max_chars", type=int, default=12000)
    ap.add_argument("--balanced", action="store_true", help="按难度平衡挑选（见文件头说明）")
    ap.add_argument("--per_item_easy", type=int, default=1)
    ap.add_argument("--per_item_mixed", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    print(json.dumps(build(args.samples, args.data, args.out, args.per_item, args.max_chars, args.balanced,
                           args.per_item_easy, args.per_item_mixed, args.seed), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
