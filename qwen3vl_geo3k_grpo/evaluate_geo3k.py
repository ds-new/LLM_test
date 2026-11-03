"""
Qwen3-VL-4B geometry3k 测试 / 评估。

后端：
  sglang —— 离线 Engine 一次性批量推理（比逐条 HF generate 快一到两个数量级）
  vllm   —— 离线 LLM 批量推理（未装 sglang 时使用）
  hf     —— transformers 左填充批量生成（sglang 不可用时的回退）
  fake   —— 不加载模型，按 checkpoint 中的 fake_quality.json 生成伪结果（多智能体 loop 干跑用）
  auto   —— sglang → vllm → hf，取第一个能正常导入的

指标：greedy 准确率（pass@1），n_samples>1 时给出 mean@k 与 pass@k、每题通过率（数据智能体据此
做难度过滤）；另外统计无 \\boxed、截断、格式命中等失败类型。--output_json 写出全部结果。

用法：
  python3 evaluate_geo3k.py --model_path /root/autodl-tmp/models/Qwen3-VL-4B-Instruct \\
      --input_file data/geo3k/test.parquet --output_json out/base_test.json
  python3 evaluate_geo3k.py --model_path <ckpt_hf> --input_file data/geo3k/train.parquet \\
      --n_samples 8 --temperature 1.0 --output_json out/train_passrate.json
  python3 evaluate_geo3k.py --model_path <path> --mode single --index 0 --input_file ...
"""

import argparse
import base64
import hashlib
import io
import json
import math
import os
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from geo3k_data import (DATA_SOURCE, build_eval_messages, clean_question, open_image, read_split,  # noqa: E402
                        record_image_bytes)
from reward_fn_geo3k import grade  # noqa: E402


def load_samples(args):
    if args.input_file:
        rows = read_split(args.input_file)
        samples = [{"index": r["extra_info"].get("index", i), "question": r["extra_info"]["question"],
                    "gt": r["reward_model"]["ground_truth"], "image": record_image_bytes(r)}
                   for i, r in enumerate(rows)]
    else:
        from geo3k_data import load_raw
        raw = load_raw(args.dataset)[args.split]
        samples = [{"index": i, "question": clean_question(r["problem"]), "gt": r["answer"], "image": r["image"]}
                   for i, r in enumerate(raw)]
    if args.limit > 0:
        samples = samples[: args.limit]
    if getattr(args, "instruction", "default") == "describe":
        for s in samples:
            s["question"] = s["question"] + " " + DESCRIBE_HINT
    if getattr(args, "upscale", 1.0) and args.upscale != 1.0:
        for s in samples:
            s["image"] = upscale_image(s["image"], args.upscale)
    return samples


# 诊断用：先读图再解题的提示（追加在题干后、标准指令前）
DESCRIBE_HINT = ("Before solving, carefully read the figure: list every labeled point, segment, angle and number "
                 "shown in the diagram, together with the given conditions. If your reasoning reaches a "
                 "contradiction, re-examine the figure instead of repeating the same calculation.")


def upscale_image(data, factor):
    """诊断用：把题图放大 factor 倍（LANCZOS），看分辨率是否限制读图。"""
    from PIL import Image
    img = open_image(data)
    img = img.resize((max(1, round(img.width * factor)), max(1, round(img.height * factor))), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# ---------------------------------------------------------------------------
# 后端
# ---------------------------------------------------------------------------
def _processor(model_path):
    from transformers import AutoProcessor
    return AutoProcessor.from_pretrained(model_path, trust_remote_code=True)


def run_sglang(args, samples):
    import sglang as sgl
    processor = _processor(args.model_path)
    prompts = [processor.apply_chat_template(build_eval_messages(s["question"]), tokenize=False,
                                             add_generation_prompt=True) for s in samples]
    images = ["data:image/png;base64," + base64.b64encode(s["image"]).decode() for s in samples]
    k = args.n_samples
    engine = sgl.Engine(model_path=args.model_path, tp_size=args.tp, mem_fraction_static=args.mem_fraction,
                        trust_remote_code=True, log_level="error")
    sp = {"max_new_tokens": args.max_new_tokens, "temperature": args.temperature if args.temperature > 0 else 0.0}
    if args.temperature > 0:
        sp["top_p"] = args.top_p
    try:
        outs = engine.generate(prompt=[p for p in prompts for _ in range(k)],
                               image_data=[im for im in images for _ in range(k)], sampling_params=sp)
    finally:
        engine.shutdown()
    texts = [o["text"] for o in outs]
    finish = [str((o.get("meta_info") or {}).get("finish_reason", {})) for o in outs]
    return [texts[i * k:(i + 1) * k] for i in range(len(samples))], [
        ["length" in f for f in finish[i * k:(i + 1) * k]] for i in range(len(samples))]


def run_vllm(args, samples):
    """vLLM 离线批量推理（verl 0.8 的 vllm extra 要求 vllm<=0.12，与 torch 2.9.0 匹配）。"""
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")  # 避免 fork 后子进程 CUDA 初始化失败
    from vllm import LLM, SamplingParams
    processor = _processor(args.model_path)
    llm = LLM(model=args.model_path, tensor_parallel_size=args.tp, gpu_memory_utilization=args.mem_fraction,
              trust_remote_code=True, max_model_len=args.max_new_tokens + 4096, limit_mm_per_prompt={"image": 1},
              seed=args.seed)
    reqs = [{"prompt": processor.apply_chat_template(build_eval_messages(s["question"]), tokenize=False,
                                                     add_generation_prompt=True),
             "multi_modal_data": {"image": open_image(s["image"])}} for s in samples]
    sp = SamplingParams(n=args.n_samples, max_tokens=args.max_new_tokens,
                        temperature=args.temperature if args.temperature > 0 else 0.0,
                        top_p=args.top_p if args.temperature > 0 else 1.0, seed=args.seed)
    outs = llm.generate(reqs, sp)
    texts = [[c.text for c in o.outputs] for o in outs]
    truncated = [[c.finish_reason == "length" for c in o.outputs] for o in outs]
    return texts, truncated


def available_backend():
    import importlib.util
    for name in ("sglang", "vllm"):
        if importlib.util.find_spec(name) is not None:
            try:
                __import__(name)
                return name
            except Exception as e:  # noqa  装了但导入失败（版本冲突），继续尝试下一个
                print(f"[WARN] {name} 导入失败: {type(e).__name__}: {str(e)[:200]}", flush=True)
    return "hf"


def run_hf(args, samples):
    import torch
    try:
        from transformers import AutoModelForImageTextToText as VLModel
    except ImportError:
        from transformers import AutoModelForVision2Seq as VLModel
    processor = _processor(args.model_path)
    processor.tokenizer.padding_side = "left"
    model = VLModel.from_pretrained(args.model_path, torch_dtype="auto", device_map="auto", trust_remote_code=True)
    model.eval()
    k = args.n_samples
    texts, truncated = [[] for _ in samples], [[] for _ in samples]
    flat = [(i, s) for i, s in enumerate(samples) for _ in range(k)]
    gen = {"max_new_tokens": args.max_new_tokens, "do_sample": args.temperature > 0}
    if args.temperature > 0:
        gen.update(temperature=args.temperature, top_p=args.top_p)
    eos = processor.tokenizer.eos_token_id
    for b in range(0, len(flat), args.batch_size):
        chunk = flat[b: b + args.batch_size]
        prompts = [processor.apply_chat_template(build_eval_messages(s["question"]), tokenize=False,
                                                 add_generation_prompt=True) for _, s in chunk]
        imgs = [open_image(s["image"]) for _, s in chunk]
        inputs = processor(text=prompts, images=imgs, padding=True, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, **gen)
        new = out[:, inputs["input_ids"].shape[1]:]
        decoded = processor.batch_decode(new, skip_special_tokens=True)
        for (i, _), seq, txt in zip(chunk, new, decoded):
            texts[i].append(txt)
            truncated[i].append(bool(len(seq) >= args.max_new_tokens and eos not in seq.tolist()))
        print(f"  ... {min(b + args.batch_size, len(flat))}/{len(flat)}", flush=True)
    return texts, truncated


def run_fake(args, samples):
    """干跑：正确率取自模型目录中的 fake_quality.json（由模拟训练写入），否则 0.35。"""
    q = 0.35
    qf = Path(args.model_path) / "fake_quality.json"
    if qf.exists():
        q = json.loads(qf.read_text())["acc"]
    texts, truncated = [], []
    for s in samples:
        row, tr = [], []
        for j in range(args.n_samples):
            h = int(hashlib.md5(f"{args.model_path}|{s['index']}|{s['question']}|{j}".encode()).hexdigest(), 16)
            difficulty = (int(hashlib.md5(s["question"].encode()).hexdigest(), 16) % 1000) / 1000
            p = min(0.98, max(0.02, q + 0.5 * (0.5 - difficulty)))
            u = (h % 10000) / 10000
            trunc_p = float(os.environ.get("FAKE_TRUNC_BELOW_4096", "0")) if args.max_new_tokens < 4096 else 0.0
            if ((h >> 20) % 1000) / 1000 < trunc_p:
                ans = None
            elif u < p:
                ans = s["gt"]
            elif u < p + 0.05:
                ans = None
            else:
                ans = "123456"
            row.append("<think>fake</think> " + ("\\boxed{" + ans + "}" if ans is not None else "I am not sure"))
            tr.append(ans is None)
        texts.append(row)
        truncated.append(tr)
    return texts, truncated


def generate_all(args, samples):
    backend = args.backend
    if backend == "auto":
        backend = available_backend()
    fn = {"sglang": run_sglang, "vllm": run_vllm, "hf": run_hf, "fake": run_fake}[backend]
    t0 = time.time()
    texts, truncated = fn(args, samples)
    return backend, texts, truncated, time.time() - t0


def visible_gpus():
    """不 import torch（避免父进程初始化 CUDA），从 CUDA_VISIBLE_DEVICES 或 nvidia-smi 取 GPU 列表。"""
    env = os.environ.get("CUDA_VISIBLE_DEVICES")
    if env is not None:
        return [g.strip() for g in env.split(",") if g.strip()]
    try:
        import subprocess
        out = subprocess.run(["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=30).stdout
        return [l.strip() for l in out.splitlines() if l.strip()]
    except Exception:
        return []


def generate_parallel(args, samples):
    """sglang/vllm 单卡即可装下 4B 模型：每张卡起一个独立进程处理一片数据，吞吐随卡数线性增加。"""
    import subprocess
    import tempfile
    backend = args.backend if args.backend != "auto" else available_backend()
    gpus = visible_gpus()
    if backend == "fake":
        gpus = [str(i) for i in range(max(1, args.dp))]
    dp = args.dp or len(gpus)
    dp = min(dp, len(gpus) or 1, max(1, len(samples) // 8))
    if backend not in ("sglang", "vllm", "fake") or args.tp != 1 or dp <= 1 or args.mode == "single":
        args.backend = backend
        return generate_all(args, samples)
    t0 = time.time()
    tmp = Path(tempfile.mkdtemp(prefix="geo3k_eval_"))
    procs = []
    for i in range(dp):
        raw = tmp / f"shard{i}.json"
        cmd = [sys.executable, os.path.abspath(__file__)] + sys.argv[1:] + [
            "--backend", backend, "--num_shards", str(dp), "--shard_id", str(i), "--raw_out", str(raw)]
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpus[i])
        procs.append((raw, subprocess.Popen(cmd, env=env)))
    print(f"[INFO] {backend} 数据并行 {dp} 进程，GPU {gpus[:dp]}", flush=True)
    codes = [p.wait() for _, p in procs]
    if any(codes):
        raise RuntimeError(f"数据并行子进程失败，退出码 {codes}")
    texts, truncated = [None] * len(samples), [None] * len(samples)
    for i, (raw, _) in enumerate(procs):
        part = json.loads(raw.read_text(encoding="utf-8"))
        for j, pos in enumerate(range(i, len(samples), dp)):
            texts[pos], truncated[pos] = part["texts"][j], part["truncated"][j]
    import shutil
    shutil.rmtree(tmp, ignore_errors=True)
    assert all(t is not None for t in texts)
    return backend + f"x{dp}", texts, truncated, time.time() - t0


# ---------------------------------------------------------------------------
# 统计
# ---------------------------------------------------------------------------
def score(samples, texts, truncated):
    per_item = []
    tot = {"acc": 0.0, "format": 0.0, "boxed": 0.0, "truncated": 0.0, "len": 0.0}
    fail = {"no_boxed": 0, "truncated": 0, "wrong_answer": 0}
    n = 0
    for s, outs, trs in zip(samples, texts, truncated):
        accs, votes = [], {}
        for j, (t, tr) in enumerate(zip(outs, trs)):
            g = grade(t, s["gt"])
            accs.append(g["acc"])
            if g["boxed"] and not tr:  # 多数投票只数给出了答案的样本
                key = "".join(str(g["pred"]).split())
                cnt, ok = votes.get(key, (0, False))
                votes[key] = (cnt + 1, ok or bool(g["acc"]))
            n += 1
            tot["acc"] += g["acc"]
            tot["format"] += g["format"]
            tot["boxed"] += g["boxed"]
            tot["truncated"] += float(tr)
            tot["len"] += len(t)
            if g["acc"] < 1:
                fail["truncated" if tr else ("no_boxed" if not g["boxed"] else "wrong_answer")] += 1
            if j == 0:
                first = {"pred": g["pred"], "correct": bool(g["acc"]), "boxed": bool(g["boxed"]),
                         "truncated": bool(tr), "output": t[-600:]}
        maj = max(votes.values(), key=lambda v: v[0])[1] if votes else False
        per_item.append({"index": s["index"], "gt": s["gt"], "pass_rate": sum(accs) / len(accs),
                         "maj_correct": bool(maj), **first})
    k = len(texts[0]) if texts else 1
    m = len(per_item)
    summary = {
        "n_items": m, "n_samples": k,
        "accuracy": round(100 * sum(p["correct"] for p in per_item) / max(1, m), 2),  # 第一个样本
        f"mean@{k}": round(100 * tot["acc"] / max(1, n), 2),
        f"pass@{k}": round(100 * sum(p["pass_rate"] > 0 for p in per_item) / max(1, m), 2),
        f"maj@{k}": round(100 * sum(p["maj_correct"] for p in per_item) / max(1, m), 2),
        "format_rate": round(100 * tot["format"] / max(1, n), 2),
        "boxed_rate": round(100 * tot["boxed"] / max(1, n), 2),
        "truncated_rate": round(100 * tot["truncated"] / max(1, n), 2),
        "mean_chars": round(tot["len"] / max(1, n), 1),
        "failure_types": fail,
        "pass_rate_hist": {"0": sum(p["pass_rate"] == 0 for p in per_item),
                           "(0,1)": sum(0 < p["pass_rate"] < 1 for p in per_item),
                           "1": sum(p["pass_rate"] == 1 for p in per_item)},
    }
    return summary, per_item


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model_path", default="Qwen/Qwen3-VL-4B-Instruct")
    p.add_argument("--mode", default="batch", choices=["batch", "single"])
    p.add_argument("--backend", default="auto", choices=["auto", "sglang", "vllm", "hf", "fake"])
    p.add_argument("--dataset", default=DATA_SOURCE)
    p.add_argument("--split", default="test")
    p.add_argument("--input_file", default=None, help="verl 格式 parquet/jsonl（优先于 --dataset）")
    p.add_argument("--index", type=int, default=0)
    p.add_argument("--limit", type=int, default=0, help="<=0 表示全部")
    p.add_argument("--n_samples", type=int, default=1)
    p.add_argument("--max_new_tokens", type=int, default=2048, help="与训练 max_response_length 对齐")
    p.add_argument("--temperature", type=float, default=0.0, help="0 = 贪心")
    p.add_argument("--top_p", type=float, default=1.0)
    p.add_argument("--batch_size", type=int, default=16, help="hf 后端批大小")
    p.add_argument("--tp", type=int, default=1, help="sglang 张量并行")
    p.add_argument("--mem_fraction", type=float, default=0.8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--detail", action="store_true")
    p.add_argument("--instruction", default="default", choices=["default", "describe"],
                   help="describe：题干后追加『先读图、遇矛盾回看图』的提示（诊断用）")
    p.add_argument("--upscale", type=float, default=1.0, help="题图放大倍数（诊断分辨率影响）")
    p.add_argument("--save_samples", default=None,
                   help="把每题全部采样的完整输出写成 jsonl（拒绝采样微调 rft_build.py 用）")
    p.add_argument("--output_json", default=None)
    p.add_argument("--dp", type=int, default=0,
                   help="数据并行进程数（每个进程独占一张卡）；0 = sglang/vllm 且 tp=1 时自动用全部可见 GPU")
    p.add_argument("--num_shards", type=int, default=1, help=argparse.SUPPRESS)  # 内部：子进程分片
    p.add_argument("--shard_id", type=int, default=0, help=argparse.SUPPRESS)
    p.add_argument("--raw_out", default=None, help=argparse.SUPPRESS)
    args = p.parse_args()
    random.seed(args.seed)

    samples = load_samples(args)
    if args.mode == "single":
        samples = [samples[max(0, min(args.index, len(samples) - 1))]]
    if args.num_shards > 1:  # 子进程：只生成自己那一片，原样写出文本，由父进程统一判分
        mine = samples[args.shard_id::args.num_shards]
        print(f"[shard {args.shard_id}/{args.num_shards}] {len(mine)} 题 × {args.n_samples}", flush=True)
        backend, texts, truncated, secs = generate_all(args, mine)
        Path(args.raw_out).write_text(json.dumps({"backend": backend, "texts": texts, "truncated": truncated,
                                                  "seconds": secs}, ensure_ascii=False), encoding="utf-8")
        return 0
    print(f"[INFO] {len(samples)} 题 × {args.n_samples} 次采样，模型 {args.model_path}", flush=True)
    backend, texts, truncated, secs = generate_parallel(args, samples)
    summary, per_item = score(samples, texts, truncated)
    summary.update({"model_path": args.model_path, "input_file": args.input_file, "backend": backend,
                    "temperature": args.temperature, "max_new_tokens": args.max_new_tokens,
                    "instruction": args.instruction, "upscale": args.upscale,
                    "seconds": round(secs, 1), "time": time.strftime("%Y-%m-%d %H:%M:%S")})

    if args.mode == "single":
        s, it = samples[0], per_item[0]
        print(f"[题目] {s['question']}\n[标准答案] {s['gt']}\n[模型输出]\n{texts[0][0]}\n"
              f"[提取答案] {it['pred']}  ({'正确' if it['correct'] else '错误'})")
    elif args.detail:
        for it in per_item:
            print(f"  [{'PASS' if it['correct'] else 'FAIL'}] #{it['index']} GT={it['gt']} Pred={it['pred']}"
                  f" pass_rate={it['pass_rate']:.2f}")
    print("=" * 60)
    print(json.dumps({k: v for k, v in summary.items() if k not in ("model_path", "input_file")}, ensure_ascii=False))
    print(f"  geometry3k 准确率: {summary['accuracy']:.2f}%  ({backend}, {secs:.0f}s)")
    print("=" * 60)
    if args.save_samples:
        sp = Path(args.save_samples)
        sp.parent.mkdir(parents=True, exist_ok=True)
        with sp.open("w", encoding="utf-8") as f:
            for smp, outs, trs in zip(samples, texts, truncated):
                gs = [grade(t, smp["gt"]) for t in outs]
                f.write(json.dumps({"index": smp["index"], "gt": smp["gt"], "texts": outs, "truncated": trs,
                                    "correct": [bool(g["acc"]) for g in gs], "boxed": [bool(g["boxed"]) for g in gs]},
                                   ensure_ascii=False) + "\n")
        print(f"[INFO] 全部采样已写入 {sp}", flush=True)
    if args.output_json:
        out = Path(args.output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_name(out.name + ".tmp")
        tmp.write_text(json.dumps({"summary": summary, "items": per_item}, ensure_ascii=False, indent=1), encoding="utf-8")
        os.replace(tmp, out)
    return 0 if all(not math.isnan(v) for v in [summary["accuracy"]]) else 1


if __name__ == "__main__":
    sys.exit(main())
