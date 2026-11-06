"""Helper agents around the subject. Each call is a fresh, tool-free Codex conversation that only sees the text it is
given (never the answer keys unless it is grading), so the helpers cannot leak anything into the subject's runs.

  Auditor    reads each code run (exp3) against the experiment's rules and says whether it broke one
  Extractor  turns a free-form answer into the structured fields the grader needs, when the keyword rules fail
  Reporter   writes the final reproduction report from the grades and the paper's numbers
"""
import json, re, threading
from pathlib import Path
from .codex import CodexAgent

_counter = threading.Lock()


def parse_json(text):
    m = re.search(r"```json\s*(.*?)```", text, re.S)
    cand = m.group(1) if m else text[text.find("{"): text.rfind("}") + 1]
    try:
        return json.loads(cand)
    except Exception:
        return {"_unparsed": text[:2000]}


class Helper:
    role = "helper"

    def __init__(self, lead, home):
        self.lead, self.home = lead, home
        h = lead.cfg["helpers"]
        self.model, self.effort = h.get("model") or None, h.get("effort") or None

    def ask(self, prompt, out_dir):
        agent = CodexAgent(self.lead.cfg, self.role, self.home, self.model, self.effort, log=self.lead.log)
        reply, rec = agent.send(prompt, out_dir)
        if not rec["audit"]["tool_free"]:
            self.lead.log(f"{self.role}: used tools {rec['audit']['tool_items']} (ignored, answer kept)")
        return reply


class Auditor(Helper):
    role = "auditor"

    def check(self, rules, code, out_dir):
        prompt = (
            "You audit one code run from an experiment. The experimenter's rules for the code are:\n\n"
            f"{rules}\n\nThe code that was run:\n```python\n{code}\n```\n\n"
            "Does this code break any of the rules? Judge only what the code does, not its comments. "
            "Reply with one JSON object in a ```json block: "
            '{"verdict": "ok" | "violation" | "unsure", "rule": "<which rule, or empty>", '
            '"evidence": "<the lines that show it>", "reason": "<one sentence>"}')
        return parse_json(self.ask(prompt, out_dir))


class Extractor(Helper):
    role = "extractor"

    def extract(self, question, answer, fields, out_dir):
        prompt = (
            "Read the answer below and fill in the fields. Copy what the answer says; do not judge whether it is "
            f"right and do not add knowledge of your own.\n\nTHE QUESTION THAT WAS ASKED:\n{question}\n\n"
            f"THE ANSWER:\n{answer}\n\nFIELDS:\n{fields}\n\nReply with one JSON object in a ```json block.")
        return parse_json(self.ask(prompt, out_dir))


class Reporter(Helper):
    role = "reporter"

    def write(self, summary, out_dir, language="zh"):
        lang = {"zh": "Write in Chinese (简体中文).", "en": "Write in English."}.get(language, "")
        prompt = (
            "You are writing the report of a reproduction of the blog post 'Sparks of Weight-Space Programming' "
            "(Takehi et al., MIT CSAIL, Nov 2026): a frontier LLM read and wrote neural network weights with no "
            "training. We re-ran the CPU-only experiments with the same prompts, model and settings, and graded the "
            "answers with the authors' answer keys.\n\n"
            f"RESULTS (JSON; 'paper' = the post's number, 'ours' = this reproduction):\n```json\n"
            f"{json.dumps(summary, indent=1, ensure_ascii=False)}\n```\n\n"
            "Write a concise Markdown report: a results table per experiment (paper vs ours), what reproduced and "
            "what did not, plausible reasons for any difference (sampling, number of runs, protocol details), "
            "any audit findings, and what to run next. Do not invent numbers that are not in the JSON. " + lang)
        return self.ask(prompt, out_dir)
