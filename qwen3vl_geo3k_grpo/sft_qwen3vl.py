"""Qwen3-VL 全参数 SFT（拒绝采样微调用），PyTorch FSDP，多卡 torchrun。

只训练语言模型部分（视觉编码器冻结），loss 只算回答部分（含结尾的 <|im_end|>，教模型按时收尾）。
提示词与训练 / 评估完全一致（geo3k_data.build_eval_messages）。

用法（GPU 实例，工程根目录）：
  PYTHONPATH=$PWD GEO3K_PATCH_CONV3D=1 torchrun --nproc_per_node 2 sft_qwen3vl.py \\
      --model_path $RUN/rounds/round_1/best_hf --data $RUN/rft/sft.jsonl \\
      --images $RUN/data/clean/train.parquet --out $RUN/rft/sft_hf
  --max_steps 2 --no_save   # 冒烟测试
"""

import argparse
import functools
import json
import math
import os
import random
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402


def log(msg):
    if int(os.environ.get("RANK", "0")) == 0:
        print(f"[{time.strftime('%F %T')}] {msg}", flush=True)


def load_examples(data_path, images_path):
    from geo3k_data import read_split, record_image_bytes
    rows = read_split(images_path)
    imgs = {r["extra_info"].get("index", i): record_image_bytes(r) for i, r in enumerate(rows)}
    out = []
    with open(data_path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                e = json.loads(line)
                if e["index"] in imgs:
                    e["image"] = imgs[e["index"]]
                    out.append(e)
    return out


def encode(processor, ex, max_len):
    """→ dict(input_ids, attention_mask, pixel_values, image_grid_thw, labels)；超长返回 None。"""
    from geo3k_data import build_eval_messages, open_image
    user = build_eval_messages(ex["question"])
    full = user + [{"role": "assistant", "content": [{"type": "text", "text": ex["response"]}]}]
    img = open_image(ex["image"])
    prompt_text = processor.apply_chat_template(user, tokenize=False, add_generation_prompt=True)
    full_text = processor.apply_chat_template(full, tokenize=False)
    if not full_text.startswith(prompt_text):
        raise ValueError("对话模板的前缀不一致，无法定位回答起点")
    p = processor(text=[prompt_text], images=[img], return_tensors="pt")
    b = processor(text=[full_text], images=[img], return_tensors="pt")
    ids = b["input_ids"][0]
    if ids.numel() > max_len:
        return None
    plen = p["input_ids"].shape[1]
    labels = ids.clone()
    labels[:plen] = -100
    # 模板在 <|im_end|> 后的换行不计 loss
    if labels.numel() and int(ids[-1]) != processor.tokenizer.convert_tokens_to_ids("<|im_end|>"):
        labels[-1] = -100
    return {"input_ids": ids[None], "attention_mask": b["attention_mask"], "pixel_values": b["pixel_values"],
            "image_grid_thw": b["image_grid_thw"], "labels": labels[None]}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--data", required=True, help="rft_build.py 的输出 jsonl")
    ap.add_argument("--images", required=True, help="题目文件（verl 格式），按 index 取图")
    ap.add_argument("--out", required=True)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--global_batch", type=int, default=32, help="每次参数更新的样本数（全部卡合计）")
    ap.add_argument("--warmup_ratio", type=float, default=0.05)
    ap.add_argument("--max_len", type=int, default=6144)
    ap.add_argument("--max_steps", type=int, default=0, help=">0 时只跑这么多次更新（冒烟测试）")
    ap.add_argument("--no_save", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    local = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local)
    torch.manual_seed(args.seed)

    from torch.distributed.fsdp import FullStateDictConfig, MixedPrecision, ShardingStrategy, StateDictType
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
    from transformers.models.qwen3_vl import modeling_qwen3_vl as mq

    import geo3k_patches
    geo3k_patches.patch_module(mq)  # bf16 Conv3d 慢一万倍的问题

    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
    examples = load_examples(args.data, args.images)
    random.Random(args.seed).shuffle(examples)
    log(f"{len(examples)} 条样本，{world} 卡")

    model = Qwen3VLForConditionalGeneration.from_pretrained(args.model_path, torch_dtype=torch.float32,
                                                            low_cpu_mem_usage=True)
    model.config.use_cache = False
    n_frozen = 0
    for name, prm in model.named_parameters():
        if "visual" in name:
            prm.requires_grad_(False)
            n_frozen += prm.numel()
    n_total = sum(p.numel() for p in model.parameters())
    log(f"参数 {n_total / 1e9:.2f}B，冻结视觉部分 {n_frozen / 1e9:.2f}B")
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    wrap = functools.partial(transformer_auto_wrap_policy,
                             transformer_layer_cls={mq.Qwen3VLTextDecoderLayer, mq.Qwen3VLVisionBlock})
    model = FSDP(model, auto_wrap_policy=wrap, sharding_strategy=ShardingStrategy.FULL_SHARD,
                 mixed_precision=MixedPrecision(param_dtype=torch.bfloat16, reduce_dtype=torch.float32,
                                                buffer_dtype=torch.bfloat16),
                 device_id=local, use_orig_params=True, limit_all_gathers=True)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, betas=(0.9, 0.95), weight_decay=0.0)

    mine = examples[rank::world]
    per_rank = max(1, args.global_batch // world)
    # 步数必须各卡一致：样本数不能被卡数整除时各卡分到的条数差 1，按各自条数算会多跑一步、在集合通信处死锁
    steps = max(1, math.floor((len(examples) // world) * args.epochs / per_rank))
    if args.max_steps:
        steps = min(steps, args.max_steps)
    warm = max(1, int(steps * args.warmup_ratio))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, steps - warm))))
    log(f"{steps} 次更新，每卡每次 {per_rank} 条，学习率 {args.lr}，预热 {warm} 步")

    model.train()
    it, skipped, t0 = 0, 0, time.time()
    stream = (mine * (int(args.epochs) + 1))
    for step in range(steps):
        tot, n = 0.0, 0
        for _ in range(per_rank):
            batch = None
            while batch is None:
                ex = stream[it % len(stream)]
                it += 1
                batch = encode(processor, ex, args.max_len)
                if batch is None:
                    skipped += 1
            batch = {k: v.cuda(non_blocking=True) for k, v in batch.items()}
            loss = model(**batch).loss / per_rank
            loss.backward()
            tot += float(loss) * per_rank
            n += 1
        gn = model.clip_grad_norm_(1.0)
        opt.step()
        sched.step()
        opt.zero_grad(set_to_none=True)
        stat = torch.tensor([tot, n], device="cuda")
        dist.all_reduce(stat)
        if step % 5 == 0 or step == steps - 1:
            el = time.time() - t0
            log(f"step {step + 1}/{steps} loss {stat[0].item() / stat[1].item():.4f} grad_norm {float(gn):.3f} "
                f"lr {sched.get_last_lr()[0]:.2e} 用时 {el / 60:.1f} 分钟，预计还需 {el / (step + 1) * (steps - step - 1) / 60:.1f} 分钟")
    log(f"训练结束，跳过超长样本 {skipped} 条")

    if not args.no_save:
        cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, cfg):
            sd = model.state_dict()
        if rank == 0:
            out = Path(args.out)
            if out.exists():
                shutil.rmtree(out)
            sd = {k: v.to(torch.bfloat16) for k, v in sd.items()}
            model.module.save_pretrained(out, state_dict=sd, safe_serialization=True)
            processor.save_pretrained(out)
            for f in ("chat_template.json", "chat_template.jinja", "preprocessor_config.json",
                      "video_preprocessor_config.json", "generation_config.json"):
                src = Path(args.model_path) / f
                if src.exists() and not (out / f).exists():
                    shutil.copy2(src, out / f)
            log(f"已保存到 {out}")
        dist.barrier()
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    sys.exit(main())
