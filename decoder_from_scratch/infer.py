"""从训练检查点续写文本；支持 Top-k、Top-p、组合采样、贪心和 Beam Search。"""

import argparse
import json
from pathlib import Path
import time

import torch

from .generation import beam_search, generate
from .runtime import choose_device, load_model, synchronize


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--prompt', default='人工智能')
    p.add_argument('--method', choices=['greedy', 'topk', 'topp', 'sample', 'beam'], default='sample')
    p.add_argument('--max-new-tokens', type=int, default=100)
    p.add_argument('--temperature', type=float, default=0.8)
    p.add_argument('--top-k', type=int, default=50)
    p.add_argument('--top-p', type=float, default=0.9)
    p.add_argument('--num-beams', type=int, default=4)
    p.add_argument('--length-penalty', type=float, default=1.0)
    p.add_argument('--repetition-penalty', type=float, default=1.0,
                   help='已出现 token 的惩罚系数，>=1；1 关闭，历史含 prompt')
    p.add_argument('--no-repeat-ngram-size', type=int, default=0,
                   help='禁止重复的 token n-gram 大小；0 关闭，历史含 prompt')
    p.add_argument('--no-cache', action='store_true')
    p.add_argument('--ignore-eos', action='store_true', help='生成固定长度，便于观察或比较')
    p.add_argument('--device', default='auto')
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--attention-backend', choices=['manual', 'sdpa'])
    p.add_argument('--output-json', type=Path)
    args = p.parse_args(argv)
    torch.set_num_threads(args.threads)
    device = choose_device(args.device)
    model, tokenizer, checkpoint = load_model(args.checkpoint, device)
    if args.attention_backend:
        model.config.attention_backend = args.attention_backend
    # 加载模型的随机初始化不影响采样种子。
    torch.manual_seed(args.seed)
    if checkpoint.get('normalize_punct'):
        # A800 流程训练的模型：训练数据中文标点已统一为全角，提示词做同样的规范化
        from decoder_from_scratch.a800.cleaning import normalize_zh_punct
        args.prompt = normalize_zh_punct(args.prompt)
    ids = tokenizer.encode(args.prompt, add_bos=True)
    inputs = torch.tensor([ids], dtype=torch.long, device=device)
    common = dict(max_new_tokens=args.max_new_tokens, use_cache=not args.no_cache,
                  repetition_penalty=args.repetition_penalty,
                  no_repeat_ngram_size=args.no_repeat_ngram_size,
                  eos_token_id=None if args.ignore_eos else tokenizer.eos_id)
    synchronize(device)
    started = time.perf_counter()
    if args.method == 'beam':
        output = beam_search(model, inputs, num_beams=args.num_beams,
                             length_penalty=args.length_penalty, **common)
    else:
        output = generate(model, inputs, temperature=args.temperature,
                          top_k=args.top_k if args.method in ('topk', 'sample') else 0,
                          top_p=args.top_p if args.method in ('topp', 'sample') else 1.0,
                          do_sample=args.method != 'greedy', **common)
    synchronize(device)
    elapsed = time.perf_counter() - started
    generated = output[0, len(ids):].tolist()
    result = {'prompt': args.prompt, 'completion': tokenizer.decode(generated),
              'generated_ids': generated, 'method': args.method,
              'repetition_penalty': args.repetition_penalty,
              'no_repeat_ngram_size': args.no_repeat_ngram_size,
              'use_cache': not args.no_cache, 'checkpoint_step': checkpoint['step'],
              'checkpoint_global_step': checkpoint.get('global_step', checkpoint['step']),
              'prompt_unknown_tokens': ids.count(tokenizer.unk_id),
              'new_tokens': len(generated), 'seconds': elapsed,
              'tokens_per_second': len(generated) / elapsed}
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    return result


if __name__ == '__main__':
    main()
