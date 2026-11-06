"""Daemon mode: the decision agent (Claude, in chat) drives the loop through files.

You start it once in a Mac terminal:   .venv/bin/python run.py --daemon
Then the decision agent writes instructions into  runs/control/inbox/<id>.json  and reads the results from
runs/control/outbox/<id>.json, runs/control/heartbeat.json, runs/lead.log and the run folders.

Only these actions exist (nothing else can be executed through the inbox):
  {"action": "check"}
  {"action": "run",    "args": {"exp": ["exp1"], "units": {"exp1": ["breakout"]}, "stage": "subject|grade|report|all",
                                "regrade": false, "no_reporter": false}}
  {"action": "set",    "args": {"run.concurrency": 1, "helpers.audit_mode": "flag", ...}}   # whitelisted keys only
  {"action": "plan",   "args": {"exp": [...]}}
  {"action": "probe",  "args": {"model": "gpt-6-astra", "variants": ["own_default", "own_model", "isolated"]}}   # tiny codex calls: your own CODEX_HOME vs the isolated one
  {"action": "restart"}   # re-exec the daemon with the same command line (picks up code changes)
  {"action": "stop"}
Actions run one at a time, in file-name order. A finished inbox file is moved to inbox/done/.
To cancel the running action: create runs/control/CANCEL (any content). The daemon removes it when the action ends.
"""
import contextlib, io, json, os, threading, time, traceback
from pathlib import Path

SETTABLE = {"run.concurrency": int, "helpers.audit_mode": str, "helpers.model": str, "helpers.effort": str,
            "helpers.report_language": str, "eval.minatar_episodes": int, "eval.workers": int,
            "codex.max_wait_hours": float, "sandbox.max_runs": int}


class _Tee(io.TextIOBase):
    def __init__(self, *streams):
        self.streams = streams

    def write(self, s):
        for st in self.streams:
            st.write(s)
            st.flush()
        return len(s)


def serve(lead, poll_s=5):
    ctl = lead.runs / "control"
    inbox, outbox, done = ctl / "inbox", ctl / "outbox", ctl / "inbox" / "done"
    for d in (inbox, outbox, done):
        d.mkdir(parents=True, exist_ok=True)
    state = {"pid": os.getpid(), "started": time.time(), "current": None, "processed": 0}

    def beat():
        while True:
            state["ts"] = time.time()
            state["ts_local"] = time.strftime("%Y-%m-%d %H:%M:%S")
            (ctl / "heartbeat.json").write_text(json.dumps(state, indent=1))
            time.sleep(poll_s)

    threading.Thread(target=beat, daemon=True).start()
    cancel = Path(lead.cfg["_cancel_file"])
    cancel.unlink(missing_ok=True)
    lead.log(f"daemon: watching {inbox} (pid {os.getpid()}); network keys carried from your Codex config: "
             f"{getattr(lead, 'network_keys', [])}")
    while True:
        files = sorted(p for p in inbox.glob("*.json"))
        if not files:
            time.sleep(poll_s)
            continue
        f = files[0]
        try:
            msg = json.loads(f.read_text())
        except Exception as e:
            msg = {"action": "invalid", "error": str(e)}
        mid = msg.get("id") or f.stem
        state["current"] = {"id": mid, "action": msg.get("action"), "since": time.strftime("%H:%M:%S")}
        out = {"id": mid, "action": msg.get("action"), "status": "running", "started": time.time()}
        (outbox / f"{mid}.json").write_text(json.dumps(out, indent=1))
        f.rename(done / f.name)
        lead.log(f"daemon: {mid} {msg.get('action')} {json.dumps(msg.get('args', {}), ensure_ascii=False)}")
        buf = io.StringIO()
        try:
            with contextlib.redirect_stdout(_Tee(buf, __import__('sys').__stdout__)):
                result = handle(lead, msg)
            out.update(status="done", result=result)
        except SystemExit as e:
            out.update(status="done" if not e.code else "failed", result={"exit": e.code})
        except Exception as e:
            out.update(status="failed", error=f"{type(e).__name__}: {e}", traceback=traceback.format_exc()[-4000:])
        if cancel.exists():
            out["cancelled"] = True
            cancel.unlink(missing_ok=True)
        out.update(finished=time.time(), output_tail=buf.getvalue()[-6000:])
        (outbox / f"{mid}.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
        state["current"], state["processed"] = None, state["processed"] + 1
        lead.log(f"daemon: {mid} {out['status']}")
        if msg.get("action") == "stop":
            lead.log("daemon: stopped by the decision agent")
            return
        if msg.get("action") == "restart":
            import sys
            lead.log("daemon: restarting to load new code")
            os.execv(sys.executable, [sys.executable] + sys.argv)


def handle(lead, msg):
    a, args = msg.get("action"), msg.get("args", {}) or {}
    if a == "check":
        return {"ok": lead.check()}
    if a == "plan":
        tasks = lead.plan(args.get("exp"), args.get("units"))
        return [{"unit": f"{e.key}/{u}", "status": lead.load_state(e.key, u).get("status", "new")} for e, u in tasks]
    if a == "set":
        changed = {}
        for k, v in args.items():
            if k not in SETTABLE:
                raise ValueError(f"{k} cannot be set through the inbox; settable: {sorted(SETTABLE)}")
            sec, key = k.split(".")
            lead.cfg[sec][key] = SETTABLE[k](v)
            changed[k] = lead.cfg[sec][key]
        return changed
    if a == "run":
        tasks = lead.plan(args.get("exp"), args.get("units"))
        stage = args.get("stage", "all")
        if stage in ("all", "subject"):
            lead.stage_subject(tasks)
        if stage in ("all", "grade"):
            lead.stage_grade(tasks, bool(args.get("regrade")))
        if stage in ("all", "report"):
            lead.stage_report(tasks, use_reporter=not args.get("no_reporter"))
        bb = lead.runs / "blackboard.json"
        return json.loads(bb.read_text()) if bb.exists() else {}
    if a == "probe":
        return probe(lead, args.get("model", lead.cfg["subject"]["model"]), args.get("variants"))
    if a == "stop":
        return {"stopping": True}
    if a == "restart":
        return {"restarting": True}
    raise ValueError(f"unknown action {a!r}")


def probe(lead, model, only=None):
    """Three tiny codex calls, to tell a network/provider problem from a model-access problem."""
    import subprocess, tempfile
    from .codex import parse_events, cancel_requested
    user_home = Path(os.environ.get("CODEX_HOME", "~/.codex")).expanduser()
    variants = [("your CODEX_HOME, default model", user_home, None),
                (f"your CODEX_HOME, {model}", user_home, model),
                (f"isolated subject home, {model}", lead.subject_home, model)]
    keys = {"own_default": 0, "own_model": 1, "isolated": 2}
    if only:
        variants = [variants[keys[k]] for k in only]
    res = []
    for name, home, m in variants:
        d = Path(tempfile.mkdtemp(prefix="wsp_probe_"))
        cmd = [lead.cfg["codex"]["binary"], "exec", "--skip-git-repo-check", "--json", "-o", str(d / "a.md")]
        if m:
            cmd += ["-m", m]
        cmd += ["-c", 'model_reasoning_effort="low"', "-"]
        env = dict(os.environ, CODEX_HOME=str(home))
        t0 = time.time()
        (d / "prompt.txt").write_text("Reply with exactly the word OK and nothing else.")
        with open(d / "ev.jsonl", "w") as ev, open(d / "err.log", "w") as er, open(d / "prompt.txt") as stdin:
            p = subprocess.Popen(cmd, stdin=stdin, stdout=ev, stderr=er, cwd=d, env=env)
            while p.poll() is None and time.time() - t0 < 240 and not cancel_requested(lead.cfg):
                time.sleep(1)
            if p.poll() is None:
                p.kill()
                p.wait()
        info = parse_events(d / "ev.jsonl")
        reply = (d / "a.md").read_text() if (d / "a.md").exists() else ""
        res.append({"variant": name, "rc": p.returncode, "seconds": round(time.time() - t0, 1),
                    "reply": reply.strip()[:80], "errors": info["errors"][:3], "notices": info["notices"][-3:],
                    "stderr_tail": (d / "err.log").read_text(errors="replace")[-600:]})
        lead.log(f"probe: {name}: rc {p.returncode}, {res[-1]['seconds']} s, reply {reply.strip()[:30]!r}")
    return res
