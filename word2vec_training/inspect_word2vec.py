"""加载本地 vectors.kv，查看词表规模并查询相似词。"""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vectors", default=str(Path(__file__).parent / "output/word2vec-demo/vectors.kv"))
    parser.add_argument("--words", nargs="+", required=True, help="词表中的完整词语，不做额外分词或大小写转换")
    parser.add_argument("--topn", type=int, default=10)
    parser.add_argument("--min-count", type=int, default=0,
                        help="相似词候选的最低词频，0 表示不筛选；仅影响查询，不修改模型")
    args = parser.parse_args()
    if args.topn < 1:
        parser.error("topn 必须为正整数")
    if args.min_count < 0:
        parser.error("min-count 必须为非负整数")
    from gensim.models import KeyedVectors
    import numpy as np

    vectors = KeyedVectors.load(str(Path(args.vectors).expanduser()), mmap="r")
    counts = []
    for key in vectors.index_to_key:
        try:
            counts.append(int(vectors.get_vecattr(key, "count")))
        except KeyError:
            counts.append(None)
    if args.min_count and any(count is None for count in counts):
        raise ValueError("词向量缺少词频属性，请使用本工程训练输出的 vectors.kv，或取消 --min-count")
    eligible = np.array([args.min_count == 0 or count >= args.min_count for count in counts], dtype=bool)
    results = []
    for word in args.words:
        found = word in vectors
        neighbors = []
        if found:
            # 检索全部合格候选，不对原来的 top-n 直接过滤；不假设词表按词频排序。
            scores = vectors.most_similar(word, topn=None)
            indices = np.flatnonzero(eligible & np.isfinite(scores))
            indices = indices[indices != vectors.get_index(word)]
            indices = indices[np.argsort(-scores[indices], kind="stable")[:args.topn]]
            neighbors = [{"word": vectors.index_to_key[index], "score": float(scores[index]),
                          "count": counts[index]} for index in indices]
        results.append({"word": word, "found": found,
                        "count": counts[vectors.get_index(word)] if found else None,
                        "similar": neighbors})
    print(json.dumps({"vocab_size": len(vectors), "vector_size": vectors.vector_size,
                      "candidate_min_count": args.min_count, "candidate_vocab_size": int(eligible.sum()),
                      "words": results}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    try:
        main()
    except ImportError as error:
        raise SystemExit(f"依赖导入失败：{error}；请安装 word2vec_training/requirements.txt") from None
    except (OSError, ValueError) as error:
        raise SystemExit(str(error)) from None
    except KeyboardInterrupt:
        raise SystemExit(130) from None
