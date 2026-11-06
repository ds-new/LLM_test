"""M6 step 4: our numbers next to the paper's.   python compare.py <eval_root> > RESULTS.md"""
import json, math, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
NAMES = {"base": "unedited", "p10c": "p10c", "p25c": "p25c", "p50c": "p50c"}


def se(acc, n):
    return 100 * math.sqrt(max(acc * (1 - acc), 1e-12) / max(n, 1))


def main(root):
    lines = ["# M6: the authors' Qwen3.5-0.8B edits, re-evaluated", "",
             "Accuracy in %, ± one standard error over samples (samples are not independent within a problem, so this is "
             "a lower bound). cut = share of answers that hit 32,768 tokens.", "",
             "| checkpoint | set | paper acc | ours acc | paper cut | ours cut | ours mean tokens |", "|---|---|---|---|---|---|---|"]
    out = {}
    for ck, pname in NAMES.items():
        p = os.path.join(root, ck, "scores.json")
        if not os.path.exists(p):
            continue
        ours = json.load(open(p))
        paper = json.load(open(os.path.join(HERE, "authors", "paper", f"{pname}.json")))
        for s in ("math500", "gsm8k"):
            if s not in ours:
                continue
            o, q = ours[s], paper.get(s, {})
            lines.append(f"| {ck} | {s} | {100 * q.get('acc', float('nan')):.1f} | {100 * o['acc']:.1f} ± "
                         f"{se(o['acc'], o['total']):.1f} | {100 * q.get('cut_share', float('nan')):.1f} | "
                         f"{100 * o['cut_share']:.1f} | {o['mean_tokens']:.0f} |")
            out.setdefault(ck, {})[s] = {"paper": q.get("acc"), "ours": o["acc"], "paper_cut": q.get("cut_share"),
                                         "ours_cut": o["cut_share"]}
    print("\n".join(lines))
    os.makedirs(root, exist_ok=True)
    json.dump(out, open(os.path.join(root, "compare.json"), "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1])
