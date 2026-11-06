"""M6 step 3: the paper's Qwen3.5-0.8B evaluation protocol ("think35" in the authors' evaluate.py), on one checkpoint.

  python eval35.py <ckpt_dir> <out_dir> --sets math500,gsm8k [--bench DIR] [--dry]

Protocol (authors' code/evaluate.py + code/profiles.py, profile qwen35think): chat template with enable_thinking=True;
user turn = problem + "\\n" + "Please reason step by step, and put your final answer within \\boxed{}."; vLLM sampling
temperature 0.6, top_p 0.95, top_k 20, presence_penalty 1.5, max 32,768 new tokens, seed 0; samples per problem:
MATH-500 4, GSM8K 2; score = mean accuracy over all samples, graded by the last \\boxed{} answer with math-verify
(authors' mathqa.extract / is_correct). Also reported: cut_share (answers that hit 32,768 tokens), mean_tokens.
Resumable: a finished set is kept in scores.json and skipped.
"""
import argparse, gzip, json, os, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "authors"))
import mathqa as Q  # noqa: E402

SAMPLES = {"math500": 4, "gsm8k": 2}
SAMPLING = {"temperature": 0.6, "top_p": 0.95, "top_k": 20, "presence_penalty": 1.5, "max_tokens": 32768, "seed": 0}


def load_set(bench, name):
    return [json.loads(l) for l in open(os.path.join(bench, f"{name}.jsonl"))]


class _Out:
    def __init__(self, text, n, reason):
        self.text, self.token_ids, self.finish_reason = text, list(range(n)), reason


class _Req:
    def __init__(self, outs):
        self.outputs = outs


def _dry_generate(rows, n):
    """--dry: no GPU. Every 3rd sample answers right, every 5th loops to the cap, the rest answer wrong."""
    res = []
    for i, r in enumerate(rows):
        outs = []
        for k in range(n):
            j = i * n + k
            if j % 5 == 0:
                outs.append(_Out("Wait, let me check. " * 100, 32768, "length"))
            elif j % 3 == 0:
                outs.append(_Out(f"<think>ok</think> The answer is \\boxed{{{r['answer']}}}.", 400, "stop"))
            else:
                outs.append(_Out("<think>hmm</think> \\boxed{-12345}", 300, "stop"))
        res.append(_Req(outs))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("out")
    ap.add_argument("--sets", default="math500")
    ap.add_argument("--bench", default=os.path.join(HERE, "bench"))
    ap.add_argument("--gpu-mem", type=float, default=float(os.environ.get("EVAL_GPU_MEM", 0.90)))
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    sp_path = os.path.join(a.out, "scores.json")
    res = json.load(open(sp_path)) if os.path.exists(sp_path) else {}
    res["_protocol"] = {"name": "think35", "sampling": SAMPLING, "samples": SAMPLES, "thinking": True,
                        "prompt": Q.INSTRUCTION, "ckpt": os.path.abspath(a.ckpt)}
    todo = [s for s in a.sets.split(",") if s and s not in res]
    if not todo:
        print("nothing to do:", sp_path)
        return
    if a.dry:
        tok, llm = None, None
    else:
        from transformers import AutoTokenizer
        from vllm import LLM, SamplingParams
        tok = AutoTokenizer.from_pretrained(a.ckpt)
        cfg = json.load(open(os.path.join(a.ckpt, "config.json")))
        kw = {}
        if any("ConditionalGeneration" in x for x in cfg.get("architectures", [])):
            kw["limit_mm_per_prompt"] = {"image": 0, "video": 0}    # text-only prompts; skip the vision memory profile
        llm = LLM(model=a.ckpt, dtype="bfloat16", gpu_memory_utilization=a.gpu_mem, seed=0,
                  max_model_len=SAMPLING["max_tokens"] + 2048, **kw)
        import vllm
        res["_protocol"]["vllm"] = vllm.__version__
    for name in todo:
        t0 = time.time()
        rows = load_set(a.bench, name)
        if a.dry:
            outs = _dry_generate(rows, SAMPLES[name])
        else:
            prompts = [tok.apply_chat_template([{"role": "user", "content": Q.build_prompt(r["problem"])}],
                                               tokenize=False, add_generation_prompt=True, enable_thinking=True)
                       for r in rows]
            outs = llm.generate(prompts, SamplingParams(n=SAMPLES[name], **SAMPLING))
            bad = sum(1 for o in outs for x in o.outputs if x.finish_reason not in ("stop", "length"))
            if bad:
                raise RuntimeError(f"{bad} generations ended without finishing or reaching max_tokens; not saved")
        per, ok, tot, cut, lens, noans = [], 0, 0, 0, [], 0
        for r, o in zip(rows, outs):
            c = []
            for s in o.outputs:
                pred, _ = Q.extract(s.text)
                good = Q.is_correct(pred, r["answer"])
                c.append(int(good))
                cut += s.finish_reason == "length"
                noans += pred is None and s.finish_reason != "length"
                lens.append(len(s.token_ids))
            per.append({"id": r["id"], "correct": c, "pred_first": Q.extract(o.outputs[0].text)[0]})
            ok += sum(c)
            tot += len(c)
        res[name] = {"acc": ok / tot, "problems": len(rows), "samples": SAMPLES[name], "cut_share": cut / tot,
                     "mean_tokens": sum(lens) / len(lens), "tokens_sum": sum(lens), "ok": ok, "total": tot, "cut": cut,
                     "no_answer_finished": noans, "s": round(time.time() - t0)}
        json.dump(per, open(os.path.join(a.out, f"{name}_per_problem.json"), "w"))
        with gzip.open(os.path.join(a.out, f"{name}_first_samples.jsonl.gz"), "wt") as f:
            for r, o in zip(rows, outs):
                f.write(json.dumps({"id": r["id"], "answer": r["answer"], "finish": o.outputs[0].finish_reason,
                                    "text": o.outputs[0].text}, ensure_ascii=False) + "\n")
        json.dump(res, open(sp_path, "w"), indent=1)
        print(name, json.dumps(res[name]), flush=True)


if __name__ == "__main__":
    main()
