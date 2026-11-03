"""用训练好的模型做几何题：自己的图片 + 题目，或 geometry3k 里的某一题；可同时对比两个模型。

用法（GPU 实例，工程根目录）：
  # 自己的图片和题目
  python infer_geo3k.py --image my_figure.png --question "Find x. Round to the nearest tenth."
  # 数据集中的第 12 题（带标准答案，自动判分），并与原始模型对比
  python infer_geo3k.py --index 12 --compare_with /root/autodl-tmp/qwen3_vl_4b

默认模型是多智能体 loop 选出的最终模型（第 1 轮最佳）。提示词、图像处理、判分与训练和评估完全一致。
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
os.environ.setdefault("GEO3K_PATCH_CONV3D", "1")

from evaluate_geo3k import _processor  # noqa: E402
from geo3k_data import build_eval_messages, clean_question, open_image, png_bytes, read_split, record_image_bytes  # noqa: E402,E501
from reward_fn_geo3k import grade  # noqa: E402

RUN = Path(os.environ.get("MA_RUN_DIR", "/root/autodl-tmp/geo3k_runs/multiagent_v1"))
DEFAULT_MODEL = RUN / "rounds" / "round_1" / "best_hf"
DEFAULT_DATA = RUN / "data" / "clean" / "test.parquet"


def load_item(args):
    if args.image:
        from PIL import Image
        img = Image.open(args.image).convert("RGB")
        if max(img.size) > 1024:  # 与数据清洗一致：长边缩到 1024
            img.thumbnail((1024, 1024))
        return {"image": png_bytes(img), "question": clean_question(args.question), "answer": args.answer}
    r = read_split(args.input_file)[args.index]
    return {"image": record_image_bytes(r), "question": r["extra_info"]["question"],
            "answer": r["reward_model"]["ground_truth"]}


def run(model_path, item, max_new_tokens, temperature, n):
    from vllm import LLM, SamplingParams
    processor = _processor(str(model_path))
    llm = LLM(model=str(model_path), trust_remote_code=True, max_model_len=max_new_tokens + 4096,
              limit_mm_per_prompt={"image": 1}, gpu_memory_utilization=0.85)
    prompt = processor.apply_chat_template(build_eval_messages(item["question"]), tokenize=False,
                                           add_generation_prompt=True)
    sp = SamplingParams(n=n, max_tokens=max_new_tokens, temperature=temperature,
                        top_p=1.0 if temperature == 0 else 0.95)
    out = llm.generate([{"prompt": prompt, "multi_modal_data": {"image": open_image(item["image"])}}], sp)[0]
    del llm
    return [(c.text, c.finish_reason == "length") for c in out.outputs]


def show(tag, results, answer, tail_chars):
    print(f"\n========== {tag} ==========")
    for i, (text, trunc) in enumerate(results):
        verdict = ""
        if answer:
            g = grade(text, str(answer))
            verdict = f"  判分：{'对' if g['acc'] else '错'}（boxed={'有' if g.get('boxed') else '无'}）"
        print(f"--- 样本 {i + 1}：{len(text)} 字符{'，被截断' if trunc else ''}{verdict}")
        print(("…" + text[-tail_chars:]) if tail_chars and len(text) > tail_chars else text)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model_path", default=str(DEFAULT_MODEL))
    ap.add_argument("--compare_with", default=None, help="再跑一个模型对比（如原始模型目录）")
    ap.add_argument("--image", default=None, help="自己的题图路径")
    ap.add_argument("--question", default=None, help="与 --image 一起用：题目文字")
    ap.add_argument("--answer", default=None, help="可选：标准答案，给了就自动判分")
    ap.add_argument("--input_file", default=str(DEFAULT_DATA))
    ap.add_argument("--index", type=int, default=0)
    ap.add_argument("--max_new_tokens", type=int, default=4096)
    ap.add_argument("--temperature", type=float, default=0.0, help="0 = 贪心（与评估一致）")
    ap.add_argument("--n", type=int, default=1, help="采样条数（temperature>0 时有意义）")
    ap.add_argument("--tail_chars", type=int, default=1500, help="只打印回答末尾若干字符；0 = 全部")
    args = ap.parse_args()
    if args.image and not args.question:
        ap.error("--image 需要配合 --question")
    item = load_item(args)
    print(f"题目：{item['question']}\n标准答案：{item.get('answer') or '（未提供）'}")
    show(f"模型 {args.model_path}", run(args.model_path, item, args.max_new_tokens, args.temperature, args.n),
         item.get("answer"), args.tail_chars)
    if args.compare_with:
        # 另起进程跑对比模型：同一进程里第二个 vLLM 引擎拿不回第一个占的显存
        import subprocess
        argv, skip = [], False
        for a in sys.argv[1:]:
            if skip:
                skip = False
                continue
            if a in ("--compare_with", "--model_path"):
                skip = True
                continue
            if a.startswith(("--compare_with=", "--model_path=")):
                continue
            argv.append(a)
        subprocess.run([sys.executable, str(Path(__file__).resolve()), "--model_path", args.compare_with, *argv],
                       check=False)


if __name__ == "__main__":
    main()
