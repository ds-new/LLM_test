"""Experiment adapters. Each one knows its first message (the authors' exact prompt from the logs), how a
conversation runs, how to grade the result with the authors' answer key, and the post's numbers to compare with.
"""
import json, re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np

from .edits import apply_edits
from .sandbox import Sandbox


def _first_line(text, label):
    """The answer to a labelled field: 'Game: X', '**Game:** X', or a heading ('## Game', '**Images:**') with the
    answer on the next non-empty line. The last such field in the reply wins."""
    lines, found = text.splitlines(), ""
    pat = re.compile(rf"^[\W_]*{label}\b[^\w\n]*(.*)$", re.I)
    for i, line in enumerate(lines):
        m = pat.match(line)
        if not m:
            continue
        rest = m.group(1).strip()
        if not rest.strip("*_ "):
            rest = next((l.strip() for l in lines[i + 1:] if l.strip()), "")
        found = rest
    return found


def _keyword_label(line, table):
    low, best = line.lower(), (None, 10 ** 9)
    for label, words in table.items():
        for w in words:
            i = low.find(w)
            if i != -1 and i < best[1]:
                best = (label, i)
    return best[0]


def _confidence(text):
    m = re.search(r"(\d+(?:\.\d+)?)", _first_line(text, "confidence"))
    return float(m.group(1)) if m else None


class Experiment:
    key, folder, effort, summary, units = "", "", "high", "detailed", []

    def __init__(self, lead):
        self.lead = lead
        self.src = Path(lead.cfg["paths"]["logs_dir"]) / self.folder

    # --- subject stage -------------------------------------------------------------------------------------
    def first_message(self, unit):
        raise NotImplementedError

    def run_subject(self, unit, udir, st):
        """Default: one message, one reply (exp1, 2, 5)."""
        agent = self.lead.subject(self, unit)
        reply, rec = agent.send(self.first_message(unit), udir / "turns" / "t001")
        (udir / "final_answer.md").write_text(reply, encoding="utf-8")
        st.update(agent.state(), done=True, tool_free=rec["audit"]["tool_free"])
        return st

    # --- grading stage -------------------------------------------------------------------------------------
    def grade(self, unit, udir):
        raise NotImplementedError

    def paper(self, unit):
        return {}


# ------------------------------------------------------------------------------------------------- Experiment 1
class Exp1(Experiment):
    key, folder, effort = "exp1", "exp1_which_game", "high"
    units = ["breakout", "pong", "pacman", "freeway"]
    KEYWORDS = {"breakout": ["breakout", "brick", "arkanoid"], "pong": ["pong"],
                "pacman": ["pac-man", "pacman", "pac man", "ms. pac"],
                "freeway": ["freeway", "frogger", "road-crossing", "road crossing", "traffic-crossing",
                            "traffic crossing", "chicken"]}
    PAPER_CONF = {"breakout": 99, "pong": 96, "pacman": 96, "freeway": 97}

    def first_message(self, unit):
        return (self.src / "runs" / unit / "prompt.txt").read_text(encoding="utf-8")

    def grade(self, unit, udir):
        ans = (udir / "final_answer.md").read_text(encoding="utf-8")
        line = _first_line(ans, "game")
        guess, via = _keyword_label(line, self.KEYWORDS), "keywords"
        if guess is None:
            q = "Which game was this neural network trained on? (open question, no list of games given)"
            ex = self.lead.extractor().extract(q, ans, '{"game": "the game the answer names, lowercase"}',
                                               udir / "helpers" / "extractor")
            guess, via = _keyword_label(str(ex.get("game", "")), self.KEYWORDS) or "other", "extractor"
        return {"answer_line": line, "guess": guess, "true": unit, "correct": guess == unit, "via": via,
                "confidence": _confidence(ans)}

    def paper(self, unit):
        return {"correct": True, "confidence": self.PAPER_CONF[unit]}


# ------------------------------------------------------------------------------------------------- Experiment 2
class Exp2(Exp1):
    key, folder, effort = "exp2", "exp2_diffusion", "high"
    units = ["circle", "square", "triangle"]
    KEYWORDS = {"circle": ["circle", "circular", "disc", "disk", "round", "oval", "ellipse", "ring"],
                "square": ["square", "rectang", "box"],
                "triangle": ["triang", "pyramid", "wedge"]}
    PAPER_CONF = {"circle": 84, "square": 77, "triangle": 88}

    def grade(self, unit, udir):
        ans = (udir / "final_answer.md").read_text(encoding="utf-8")
        line = _first_line(ans, "images")
        guess, via = _keyword_label(line, self.KEYWORDS), "keywords"
        if guess is None:
            q = "What does this diffusion model generate? (open question, no list of shapes given)"
            ex = self.lead.extractor().extract(q, ans, '{"shape": "the shape the answer says is drawn"}',
                                               udir / "helpers" / "extractor")
            guess, via = _keyword_label(str(ex.get("shape", "")), self.KEYWORDS) or "other", "extractor"
        outline = bool(re.search(r"outline|hollow|thin|one-pixel|1-pixel|ring", line, re.I))
        return {"answer_line": line[:300], "guess": guess, "true": unit, "correct": guess == unit, "via": via,
                "confidence": _confidence(ans), "drawn_as_outline": outline}

    def paper(self, unit):
        return {"correct": True, "confidence": self.PAPER_CONF[unit], "drawn_as_outline": True}


# ------------------------------------------------------------------------------------------------- Experiment 3
class Exp3(Experiment):
    key, folder, effort, summary = "exp3", "exp3_language", "xhigh", None
    units = ["python_r0", "python_r1", "romaji_r0", "romaji_r1", "english_r0", "english_r1"]
    PRELUDE = "import numpy as np\nW = dict(np.load('{INPUTS}/blocks.npz'))\n"
    PAPER = {"python_r0": 0.56, "python_r1": 0.52, "romaji_r0": 0.55, "romaji_r1": 0.62,
             "english_r0": 0.58, "english_r1": 0.48}
    CODE = re.compile(r"```python[ \t]*\n(.*?)```", re.S)
    LABELS = ("previous", "local", "broad", "copy", "unclear")

    def first_message(self, unit):
        return (self.src / "runs" / unit / "message.txt").read_text(encoding="utf-8")

    def rules(self, unit):
        msg = self.first_message(unit)
        i = msg.find("WHAT YOU CAN USE")
        j = msg.find("HOW YOU COMPUTE")
        return msg[i:j].strip() if i != -1 and j != -1 else msg

    def run_subject(self, unit, udir, st):
        cfg = self.lead.cfg["sandbox"]
        lang = unit.split("_")[0]
        sbx = Sandbox(cfg["python"], self.PRELUDE, {"blocks.npz": self.src / "networks" / lang / "blocks.npz"},
                      deny_read=self.lead.deny_read(), timeout=cfg["timeout_s"], max_out=cfg["max_output_chars"],
                      log=udir / "tool_calls.jsonl")
        if sbx.kind == "none" and not cfg.get("allow_unisolated"):
            sbx.close()
            raise RuntimeError("no sandbox isolation on this machine (needs macOS sandbox-exec or Linux bubblewrap): "
                               "the subject's code could read the answer keys. Set [sandbox] allow_unisolated = true "
                               "only for offline tests.")
        agent = self.lead.subject(self, unit)
        agent.restore(st)
        runs, limit = st.get("runs", 0), cfg["max_runs"]
        msg = st.get("next") or self.first_message(unit)
        try:
            while True:
                if st.get("pending_reply") is not None:
                    # Resumed after an interruption: this reply was already received; do not send its message again.
                    reply, tdir = st["pending_reply"], udir / "turns" / f"t{agent.turn:03d}"
                    self.lead.log(f"{self.key}/{unit}: resuming from the saved reply of turn {agent.turn}")
                else:
                    tdir = udir / "turns" / f"t{agent.turn + 1:03d}"
                    reply, rec = agent.send(msg, tdir)
                    st.update(agent.state(), pending_reply=reply,
                              tool_free=st.get("tool_free", True) and rec["audit"]["tool_free"])
                    self.lead.save_state(self.key, unit, st)
                blocks = self.CODE.findall(reply)
                if not blocks:
                    (udir / "final_answer.md").write_text(reply, encoding="utf-8")
                    st.update(done=True, runs=runs, next=None, pending_reply=None)
                    break
                if len(blocks) > 1:
                    msg = "Reply with exactly one ```python code block (or, when you are done, your final answer " \
                          "with no code block). Nothing was run."
                elif runs >= limit:
                    msg = f"You have used all {limit} runs. Reply WITHOUT a code block, with your final answer " \
                          "in the order given in the question."
                else:
                    code = blocks[0]
                    verdict = None
                    if self.lead.cfg["helpers"]["audit_mode"] == "gate":
                        verdict = self.lead.auditor().check(self.rules(unit), code, tdir / "audit")
                    if verdict and verdict.get("verdict") == "violation":
                        msg = (f"This code was not run: it breaks a rule ({verdict.get('rule', '')}: "
                               f"{verdict.get('reason', '')}). It does not count as a run.")
                    else:
                        runs += 1
                        out, secs = sbx.run(code)
                        self.lead.log(f"{self.key}/{unit} turn {agent.turn}: run {runs} ({len(code):,} chars of "
                                      f"code, {len(out):,} chars out, {secs} s, isolation {sbx.kind})")
                        msg = f"OUTPUT of run {runs} of at most {limit}:\n```\n{out}\n```"
                st.update(runs=runs, next=msg, pending_reply=None)
                self.lead.save_state(self.key, unit, st)
        finally:
            sbx.close()
        return st

    def grade(self, unit, udir):
        ans = (udir / "final_answer.md").read_text(encoding="utf-8")
        key = json.loads((self.src / "eval" / "key.json").read_text())
        lang, rep = unit.split("_")
        net = key[f"{lang}_L4_H6_d384_s0_{rep}"]["net"]
        truth_corpus = key[f"{lang}_L4_H6_d384_s0_{rep}"]["corpus"]
        heads = json.loads((self.src / "eval" / "heads.json").read_text())[net]
        true_copy = sorted(h for h, v in heads.items() if v["induct"] > 0.2)
        corpus = (re.findall(r"^[\W_]*corpus[\W_]*:\s*\**\s*(\w+)", ans, re.M | re.I) or [""])[-1].lower()
        order = (re.findall(r"^[\W_]*order[\W_]*:\s*\**\s*(as is|shuffled)", ans, re.M | re.I) or [""])[-1].lower()
        probs = {k: float(v) for k, v in re.findall(r"(python|romaji|english)\s*=\s*([0-9.]+)", ans, re.I)}
        starts = [m.start() for m in re.finditer(r"^[#*\s]*heads\b", ans, re.M | re.I)]
        section = ans[starts[-1]:] if starts else ans
        labels = {}
        for h, lab in re.findall(r"(L[0-3]H[0-5])\W*[:\-]\s*\**\s*'?(previous|local|broad|copy|unclear)", section, re.I):
            labels.setdefault(h.upper(), lab.lower())
        via = "regex"
        if not corpus or len(labels) < 24:
            ex = self.lead.extractor().extract(
                "Which corpus was the network trained on, and what does each of the 24 heads do?", ans,
                '{"corpus": "python|romaji|english", "order": "as is|shuffled", '
                '"probabilities": {"python": 0, "romaji": 0, "english": 0}, '
                '"heads": {"L0H0": "previous|local|broad|copy|unclear", "...": "all 24 heads"}}',
                udir / "helpers" / "extractor")
            corpus = corpus or str(ex.get("corpus", "")).lower()
            labels = labels if len(labels) >= 24 else {k.upper(): str(v).lower() for k, v in ex.get("heads", {}).items()}
            probs = probs or {k: float(v) for k, v in ex.get("probabilities", {}).items()}
            via = "extractor"
        named = sorted(h for h, v in labels.items() if v == "copy")
        state = json.loads((udir / "state.json").read_text()) if (udir / "state.json").exists() else {}
        res = {"corpus_guess": corpus, "true": truth_corpus, "corpus_correct": corpus == truth_corpus,
               "order_guess": order, "probability_of_guess": probs.get(corpus), "probabilities": probs,
               "copy_named": named, "copy_true": true_copy, "copy_exact_match": named == true_copy,
               "copy_false_pos": sorted(set(named) - set(true_copy)), "copy_missed": sorted(set(true_copy) - set(named)),
               "runs": state.get("runs"), "via": via}
        res["audit"] = self.audit(unit, udir)
        return res

    def audit(self, unit, udir):
        """Post-hoc audit (the authors read every code run): one auditor call per run, in parallel."""
        p = udir / "tool_calls.jsonl"
        if not p.exists():
            return {"runs": 0}
        codes = [json.loads(l)["code"] for l in p.read_text().splitlines() if l.strip()]
        rules = self.rules(unit)
        aud = self.lead.auditor()

        def one(i):
            out = udir / "helpers" / f"audit_run{i + 1:03d}"
            cached = out / "verdict.json"
            if cached.exists():
                return json.loads(cached.read_text())
            v = aud.check(rules, codes[i], out)
            cached.write_text(json.dumps(v, ensure_ascii=False, indent=1))
            return v

        with ThreadPoolExecutor(4) as ex:
            verdicts = list(ex.map(one, range(len(codes))))
        flagged = [{"run": i + 1, **v} for i, v in enumerate(verdicts) if v.get("verdict") != "ok"]
        return {"runs": len(codes), "violations": sum(v.get("verdict") == "violation" for v in verdicts),
                "unsure": sum(v.get("verdict") == "unsure" for v in verdicts), "flagged": flagged}

    def paper(self, unit):
        truth = {"python": ["L2H1", "L3H1", "L3H5"], "romaji": ["L3H3"], "english": ["L3H1"]}[unit.split("_")[0]]
        return {"corpus_correct": True, "probability_of_guess": self.PAPER[unit], "copy_named": truth,
                "copy_exact_match": True}


# ------------------------------------------------------------------------------------------------- Experiment 5
class Exp5(Experiment):
    key, folder, effort = "exp5", "exp5_write_from_scratch", "xhigh"
    units = ["breakout", "seaquest", "asterix"]
    SHAPES = {"W1": (600, 32), "b1": (32,), "W2": (32, 32), "b2": (32,), "W3": (32, 6), "b3": (6,)}
    PAPER = {"breakout": (517.4, 15.8, 25.3), "seaquest": (53.1, 5.6, 88.9), "asterix": (53.2, 25.8, 55.2)}

    def first_message(self, unit):
        return (self.src / "runs" / unit / "message.txt").read_text(encoding="utf-8")

    def grade(self, unit, udir):
        from .minatar_eval import evaluate
        ans = (udir / "final_answer.md").read_text(encoding="utf-8")
        W = {k: np.zeros(s, np.float32) for k, s in self.SHAPES.items()}
        rep = apply_edits(ans, W)
        np.savez(udir / "network.npz", **W)
        src = re.findall(r"```python\n(.*?)```", self.first_message(unit), re.S)
        src = next(s for s in src if "class Env" in s)
        env_p = udir / f"env_{unit}.py"
        env_p.write_text(src)
        ev = evaluate(unit, W, str(env_p), self.lead.cfg["eval"]["minatar_episodes"], self.lead.cfg["eval"]["workers"])
        nz = int(sum(np.count_nonzero(v) for v in W.values()))
        return {"score": round(ev["mean"], 1), "se": round(ev["se"], 1), "episodes": ev["episodes"],
                "mean_length": round(ev["mean_length"]), "env": ev["env"], "statements": rep["statements"],
                "applied": rep["applied"], "edit_errors": rep["errors"][:10], "nonzero_weights": nz,
                "eval_seconds": ev["seconds"]}

    def paper(self, unit):
        a, p100, p500 = self.PAPER[unit]
        return {"score": a, "ppo_100M": p100, "ppo_500M": p500}


class NotYet(Experiment):
    why = ""

    def run_subject(self, unit, udir, st):
        raise NotImplementedError(self.why)


class Exp4(NotYet):
    key, folder = "exp4", "exp4_fix_car"
    units = ["oneshot_repair"]
    why = "exp4 needs the Kinetix simulator (JAX) to score the edited car policy; added in the next round."


class Exp6(NotYet):
    key, folder = "exp6", "exp6_go_from_scratch"
    units = ["budget_0", "budget_10", "budget_25", "budget_50"]
    why = "exp6 needs pgx (JAX Go) for the games/positions passes and scoring; added in the next round."


REGISTRY = {c.key: c for c in (Exp1, Exp2, Exp3, Exp4, Exp5, Exp6)}
