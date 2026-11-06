"""The lead agent: plans the work, dispatches the subject conversations in parallel, keeps the blackboard (per-unit
state on disk, so an interrupted run resumes where it stopped), sends finished work to grading, and has the
reporter write up the result. It is plain Python on purpose: its decisions are the experiment protocol, and they must
be the same every time.

Stages:  subject (Astra answers)  ->  grade (answer keys + auditor + extractor)  ->  report (reporter)
"""
import json, os, re, threading, time, traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .codex import Cancelled, CodexAgent, network_overrides, prepare_home
from .experiments import REGISTRY
from .roles import Auditor, Extractor, Reporter

NEUTRAL = "Answer the user's request.\n"


class Lead:
    def __init__(self, cfg, root):
        self.cfg, self.root = cfg, Path(root)
        p = cfg["paths"]
        p["logs_dir"] = str((self.root / p["logs_dir"]).resolve())
        p["runs_dir"] = str((self.root / p["runs_dir"]).resolve())
        cx = cfg["codex"]
        if "/" in cx["binary"] and not os.path.isabs(cx["binary"]):
            cx["binary"] = os.path.abspath(self.root / cx["binary"])
        sb = cfg["sandbox"]
        if "/" in sb["python"] and not os.path.isabs(sb["python"]):
            sb["python"] = os.path.abspath(self.root / sb["python"])  # not resolve(): keep the venv
        self.runs = Path(p["runs_dir"])
        self.runs.mkdir(parents=True, exist_ok=True)
        (self.runs / "control").mkdir(exist_ok=True)
        cfg["_cancel_file"] = str(self.runs / "control" / "CANCEL")
        self._lock = threading.Lock()
        self.homes = self.runs / "_codex_homes"
        base_cfg = (Path(p["logs_dir"]) / "harness" / "codex_config.toml").read_text()
        auth = Path(os.environ.get("CODEX_HOME", "~/.codex")).expanduser() / "auth.json"
        self.subject_home = prepare_home(self.homes / "subject", self._config(base_cfg, "subject"), NEUTRAL, auth)
        self.helper_home = prepare_home(self.homes / "helpers", self._config(base_cfg, "helpers"), NEUTRAL, auth)

    def _config(self, text, role):
        home = self.homes / role
        if self.cfg["codex"].get("inherit_network", True):
            user_cfg = Path(os.environ.get("CODEX_HOME", "~/.codex")).expanduser() / "config.toml"
            top, tables, used = network_overrides(user_cfg)
            self.network_keys = used
            if self.cfg["codex"].get("http_only") and "model_provider" not in used:
                # Same ChatGPT-login backend as the built-in provider, but HTTPS/SSE only: no WebSocket attempts
                # (which time out 5x on some networks before Codex falls back), and a long idle timeout for
                # high-effort reasoning.
                top += 'model_provider = "openai_http"\n'
                tables += ('\n[model_providers.openai_http]\nname = "OpenAI (ChatGPT login, HTTPS only)"\n'
                           f'base_url = "{self.cfg["codex"].get("chatgpt_base_url", "https://chatgpt.com/backend-api/codex")}"\n'
                           'wire_api = "responses"\nrequires_openai_auth = true\nsupports_websockets = false\n'
                           f'stream_idle_timeout_ms = {int(self.cfg["codex"].get("stream_idle_timeout_ms", 1800000))}\n'
                           'request_max_retries = 4\nstream_max_retries = 10\n')
                self.network_keys = used + ["openai_http (HTTPS only)"]
            text = top + text + ("\n" + tables if tables else "")
        model = self.cfg[role].get("model") or ""
        text = re.sub(r'^model = ".*"$', f'model = "{model}"' if model else "", text, flags=re.M)
        return re.sub(r"^model_instructions_file = .*$", f'model_instructions_file = "{home / "instructions.md"}"',
                      text, flags=re.M)

    # --- shared services ------------------------------------------------------------------------------------
    def log(self, msg):
        line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
        with self._lock:
            print(line, flush=True)
            with open(self.runs / "lead.log", "a") as f:
                f.write(line + "\n")

    def subject(self, exp, unit):
        return CodexAgent(self.cfg, "subject", self.subject_home, self.cfg["subject"]["model"], exp.effort,
                          summary=exp.summary and self.cfg["subject"].get("reasoning_summary"), log=self.log,
                          env_extra={"WSP_TASK": f"{exp.key}/{unit}"})

    def auditor(self):
        return Auditor(self, self.helper_home)

    def extractor(self):
        return Extractor(self, self.helper_home)

    def deny_read(self):
        codex_home = Path(os.environ.get("CODEX_HOME", "~/.codex")).expanduser()
        return [self.cfg["paths"]["logs_dir"], str(self.runs), str(codex_home)]

    def udir(self, exp_key, unit):
        d = self.runs / exp_key / unit
        d.mkdir(parents=True, exist_ok=True)
        return d

    def load_state(self, exp_key, unit):
        p = self.udir(exp_key, unit) / "state.json"
        return json.loads(p.read_text()) if p.exists() else {}

    def save_state(self, exp_key, unit, st):
        with self._lock:
            p = self.udir(exp_key, unit) / "state.json"
            tmp = p.with_suffix(".tmp")
            tmp.write_text(json.dumps(st, indent=1, ensure_ascii=False))
            tmp.replace(p)
            bb = self.runs / "blackboard.json"
            board = json.loads(bb.read_text()) if bb.exists() else {}
            board.setdefault(exp_key, {})[unit] = {k: st.get(k) for k in ("status", "done", "turn", "runs",
                                                                          "tool_free", "error", "updated")}
            bb.write_text(json.dumps(board, indent=1))

    # --- plan -----------------------------------------------------------------------------------------------
    def plan(self, exps=None, units=None):
        exps = exps or self.cfg["run"]["experiments"]
        sel = units or self.cfg["run"].get("units") or {}
        tasks = []
        for k in exps:
            exp = REGISTRY[k](self)
            for u in sel.get(k, exp.units):
                if u not in exp.units:
                    raise SystemExit(f"{k}: unknown unit {u}; units are {exp.units}")
                tasks.append((exp, u))
        return tasks

    def gate(self, next_stage):
        if self.cfg["run"].get("gate_between_stages"):
            input(f"\n=== Lead: ready for stage '{next_stage}'. Press Enter to continue (Ctrl-C to stop). ")

    # --- stage 1: the subject ---------------------------------------------------------------------------------
    def stage_subject(self, tasks):
        todo = []
        for exp, u in tasks:
            st = self.load_state(exp.key, u)
            if st.get("done"):
                self.log(f"{exp.key}/{u}: already answered, skipped")
            else:
                todo.append((exp, u, st))
        self.log(f"stage subject: {len(todo)} conversations, {self.cfg['run']['concurrency']} at a time")

        def work(item):
            exp, u, st = item
            udir = self.udir(exp.key, u)
            st.update(status="running", updated=time.time())
            self.save_state(exp.key, u, st)
            self.log(f"{exp.key}/{u}: start ({'resume turn ' + str(st['turn'] + 1) if st.get('turn') else 'new'})")
            if os.path.exists(self.cfg["_cancel_file"]):
                st.update(status="cancelled")
                self.save_state(exp.key, u, st)
                return
            try:
                st = exp.run_subject(u, udir, st)
                st.update(status="answered", updated=time.time(), error=None)
                self.log(f"{exp.key}/{u}: answered (turns {st.get('turn')}, tool-free {st.get('tool_free')})")
            except Cancelled as e:
                st = {**self.load_state(exp.key, u), "status": "cancelled", "error": str(e), "updated": time.time()}
                self.log(f"{exp.key}/{u}: cancelled ({e})")
            except NotImplementedError as e:
                st.update(status="skipped", error=str(e))
                self.log(f"{exp.key}/{u}: skipped: {e}")
            except Exception as e:
                st.update(status="failed", error=f"{type(e).__name__}: {e}", updated=time.time())
                self.log(f"{exp.key}/{u}: FAILED {e}\n{traceback.format_exc()}")
            self.save_state(exp.key, u, st)

        with ThreadPoolExecutor(max(1, self.cfg["run"]["concurrency"])) as ex:
            list(ex.map(work, todo))

    # --- stage 2: grading -------------------------------------------------------------------------------------
    def stage_grade(self, tasks, regrade=False):
        for exp, u in tasks:
            udir = self.udir(exp.key, u)
            st = self.load_state(exp.key, u)
            gp = udir / "grade.json"
            if not st.get("done"):
                self.log(f"{exp.key}/{u}: not answered yet, not graded")
                continue
            if gp.exists() and not regrade:
                continue
            self.log(f"{exp.key}/{u}: grading")
            try:
                g = exp.grade(u, udir)
                g["tool_free"] = st.get("tool_free")
                gp.write_text(json.dumps(g, indent=1, ensure_ascii=False))
                self.log(f"{exp.key}/{u}: graded {json.dumps({k: g[k] for k in list(g)[:4]}, ensure_ascii=False)}")
            except Exception as e:
                self.log(f"{exp.key}/{u}: grading FAILED {e}\n{traceback.format_exc()}")

    # --- stage 3: report --------------------------------------------------------------------------------------
    def summary(self, tasks):
        out = {}
        for exp, u in tasks:
            gp = self.udir(exp.key, u) / "grade.json"
            ours = json.loads(gp.read_text()) if gp.exists() else None
            out.setdefault(exp.key, {})[u] = {"paper": exp.paper(u), "ours": ours}
        return out

    def stage_report(self, tasks, use_reporter=True):
        summary = self.summary(tasks)
        (self.runs / "summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False))
        lines = ["# Reproduction results (CPU experiments)", ""]
        for k, units in summary.items():
            lines += [f"## {k}", "", "| unit | paper | ours |", "|---|---|---|"]
            for u, r in units.items():
                ours = r["ours"]
                short = "not graded" if ours is None else ", ".join(
                    f"{a}={ours.get(a)}" for a in r["paper"] if a in ours)
                paper = ", ".join(f"{a}={b}" for a, b in r["paper"].items())
                lines.append(f"| {u} | {paper} | {short} |")
            lines.append("")
        (self.runs / "results.md").write_text("\n".join(lines))
        self.log(f"wrote {self.runs / 'results.md'}")
        if use_reporter and any(r["ours"] for units in summary.values() for r in units.values()):
            rep = Reporter(self, self.helper_home).write(summary, self.runs / "_reporter",
                                                         self.cfg["helpers"].get("report_language", "zh"))
            (self.runs / "REPORT.md").write_text(rep, encoding="utf-8")
            self.log(f"wrote {self.runs / 'REPORT.md'}")

    # --- environment check ------------------------------------------------------------------------------------
    def check(self):
        """Run before spending any quota: Codex, login, logs repo, sandbox Python, and that isolation really holds."""
        import shutil, subprocess
        from .sandbox import Sandbox
        ok = True

        def say(good, what, hint=""):
            nonlocal ok
            ok &= bool(good)
            print(f"[{'ok' if good else 'FAIL'}] {what}" + (f"  -> {hint}" if hint and not good else ""))

        b = self.cfg["codex"]["binary"]
        path = shutil.which(b) or (b if os.path.exists(b) else None)
        ver = subprocess.run([path, "--version"], capture_output=True, text=True, stdin=subprocess.DEVNULL, timeout=30).stdout.strip() if path else ""
        say(path, f"codex binary: {path} {ver}", "install Codex CLI, or set [codex] binary")
        say((self.subject_home / "auth.json").exists(), "Codex login linked into the isolated CODEX_HOME",
            "run `codex login` first (needs ~/.codex/auth.json), then rerun --check")
        print(f"[info] network/provider keys carried from your Codex config: {getattr(self, 'network_keys', [])}")
        logs = Path(self.cfg["paths"]["logs_dir"])
        say((logs / "exp3_language" / "eval" / "heads.json").exists(), f"logs repo at {logs}", "run setup.sh")
        py = self.cfg["sandbox"]["python"]
        r = subprocess.run([py, "-c", "import numpy, scipy, sklearn, torch, pandas, sympy, networkx, threadpoolctl"],
                           capture_output=True, text=True) if os.path.exists(py) or shutil.which(py) else None
        say(r is not None and r.returncode == 0, f"sandbox python {py} has the exp3 packages",
            (r.stderr.strip().splitlines() or ["missing"])[-1] if r else "run setup.sh")
        sbx = Sandbox(py, "import numpy as np\nW = dict(np.load('{INPUTS}/blocks.npz'))\n",
                      {"blocks.npz": logs / "exp3_language" / "networks" / "python" / "blocks.npz"},
                      deny_read=self.deny_read(), timeout=60)
        out, _ = sbx.run("print('weights', len(W))")
        say("weights 40" in out, "sandbox reads the weights", out[-300:])
        key = logs / "exp3_language" / "eval" / "heads.json"
        out, _ = sbx.run(f"print(len(open({str(key)!r}).read()))")
        say("Error" in out or "denied" in out.lower(), f"sandbox cannot read the answer keys ({sbx.kind})",
            "the subject's code could read the answer key: install bubblewrap (Linux) or run on macOS")
        out, _ = sbx.run("import socket\nsocket.create_connection(('1.1.1.1', 443), timeout=5)\nprint('NET-OPEN')")
        say("NET-OPEN" not in out, f"sandbox has no network ({sbx.kind})", "network is reachable from the sandbox")
        sbx.close()
        print("\nall checks passed" if ok else "\nfix the FAIL lines before a real run")
        return ok
