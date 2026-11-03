"""
把 hiyouga/geometry3k 转成 verl 多模态训练格式（train / val / test 三个文件）。

与旧版相比的关键修正：旧版把 validation 写成 test.parquet，真正的 test（601 题）从未使用，
训练期间的验证集和最终评估用的是同一份数据。现在：
  train.parquet —— 训练（2101 题）
  val.parquet   —— 训练中验证、挑选 checkpoint（300 题）
  test.parquet  —— 只在最终评估时使用（601 题），不参与任何选择

数据清洗（去重、跨划分泄漏、坏图、标准答案可判分性检查、难度过滤）由多智能体 loop 中的
数据清洗智能体完成；这里只做格式转换，便于单独使用。

用法：
  python3 preprocess_geo3k.py --output_dir ./data/geo3k                    # 从 Hub（可设 HF_ENDPOINT 镜像）
  python3 preprocess_geo3k.py --data_dir /root/autodl-tmp/geometry3k --output_dir ./data/geo3k
"""

import argparse
import os
import random

from geo3k_data import DATA_SOURCE, load_raw, split_ext, to_verl_record, write_split


def convert(raw, data_source=DATA_SOURCE, val_fraction=0.1, seed=42):
    train = list(raw.get("train") or [])
    val = list(raw.get("validation") or [])
    test = list(raw.get("test") or [])
    if not train:
        raise RuntimeError("数据集中缺少 train split")
    if not val:  # 没有 validation 时从训练集切出一部分，test 保持独立
        rng = random.Random(seed)
        idx = list(range(len(train)))
        rng.shuffle(idx)
        k = max(1, int(len(train) * val_fraction))
        val = [train[i] for i in idx[:k]]
        train = [train[i] for i in idx[k:]]
    out = {}
    for name, rows in (("train", train), ("val", val), ("test", test)):
        out[name] = [to_verl_record(r["problem"], r["answer"], r["image"], name, i, data_source)
                     for i, r in enumerate(rows)]
    return out


def main():
    parser = argparse.ArgumentParser(description="预处理 geometry3k 为 verl 多模态格式")
    parser.add_argument("--data_dir", default="hiyouga/geometry3k", help="Hub 名称或本地目录")
    parser.add_argument("--output_dir", default="./data/geo3k")
    parser.add_argument("--data_source_name", default=DATA_SOURCE)
    parser.add_argument("--format", default=None, choices=[None, ".parquet", ".jsonl"])
    args = parser.parse_args()

    raw = load_raw(args.data_dir)
    print(f"[INFO] 原始 split: { {k: len(v) for k, v in raw.items()} }")
    splits = convert(raw, args.data_source_name)
    ext = args.format or split_ext()
    os.makedirs(args.output_dir, exist_ok=True)
    for name, rows in splits.items():
        if rows:
            path = write_split(rows, os.path.join(args.output_dir, name + ext))
            print(f"[OK] {name:5s}: {len(rows):5d} 条 -> {path}")
    print("\n训练脚本使用:  DATA_DIR=" + os.path.abspath(args.output_dir) +
          "  （train" + ext + " 训练，val" + ext + " 验证，test" + ext + " 仅最终评估）")


if __name__ == "__main__":
    main()
