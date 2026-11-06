#!/usr/bin/env python3
"""Weight-Space Programming reproduction loop (CPU experiments 1, 2, 3, 5), run with your local Codex.

  python run.py                         # all stages for the experiments in config.toml
  python run.py --exp exp1 exp5         # only these experiments
  python run.py --exp exp3 --unit exp3=python_r0
  python run.py --stage grade --regrade # re-grade what is already answered
  python run.py --plan                  # show what would run, and the Codex command, without running
  python run.py --check                 # check Codex login, data, sandbox packages and isolation
  python run.py --daemon                # let the decision agent drive the loop through runs/control/inbox
"""
import argparse, json, sys, tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from wsp.lead import Lead  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "config.toml"))
    ap.add_argument("--exp", nargs="*")
    ap.add_argument("--unit", nargs="*", default=[], help="exp=unit, e.g. exp3=python_r0 (repeatable)")
    ap.add_argument("--stage", choices=["all", "subject", "grade", "report"], default="all")
    ap.add_argument("--regrade", action="store_true")
    ap.add_argument("--no-reporter", action="store_true", help="write results.md only, no reporter agent")
    ap.add_argument("--plan", action="store_true")
    ap.add_argument("--check", action="store_true", help="check Codex, login, data and sandbox isolation")
    ap.add_argument("--daemon", action="store_true",
                    help="wait for instructions from the decision agent in runs/control/inbox (see wsp/daemon.py)")
    a = ap.parse_args()

    cfg = tomllib.loads(Path(a.config).read_text())
    lead = Lead(cfg, ROOT)
    if a.check:
        sys.exit(0 if lead.check() else 1)
    if a.daemon:
        from wsp.daemon import serve
        serve(lead)
        return
    units = {}
    for s in a.unit:
        k, u = s.split("=", 1)
        units.setdefault(k, []).append(u)
    tasks = lead.plan(a.exp, units or None)

    if a.plan:
        for exp, u in tasks:
            st = lead.load_state(exp.key, u)
            print(f"{exp.key}/{u:12s} effort={exp.effort:6s} status={st.get('status', 'new')}")
        exp, u = tasks[0]
        print("\nsubject command (first turn):", " ".join(lead.subject(exp, u)._cmd(Path("answer.md"))))
        print("CODEX_HOME =", lead.subject_home)
        return

    if a.stage in ("all", "subject"):
        lead.stage_subject(tasks)
        if a.stage == "all":
            lead.gate("grade")
    if a.stage in ("all", "grade"):
        lead.stage_grade(tasks, a.regrade)
        if a.stage == "all":
            lead.gate("report")
    if a.stage in ("all", "report"):
        lead.stage_report(tasks, use_reporter=not a.no_reporter)
    print(json.dumps(json.loads((lead.runs / "blackboard.json").read_text()) if (lead.runs / "blackboard.json").exists()
                     else {}, indent=1))


if __name__ == "__main__":
    main()
