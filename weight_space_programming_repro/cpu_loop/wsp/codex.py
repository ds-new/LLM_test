"""One Codex conversation: `codex exec` for the first turn, `codex exec resume <thread>` after that.

Every turn is saved under <turn_dir>: prompt.txt, events.jsonl, answer.md, stderr.log, run.json (with the tool audit).
Each agent role gets its own CODEX_HOME (config + neutral instructions + a link to your login), and Codex is started in
an empty working folder, so no AGENTS.md, project file or global instruction reaches the model.
"""
import json, os, subprocess, time
from pathlib import Path

# Item types that are not tool calls. Anything else in the event stream (command_execution, file_change,
# mcp_tool_call, web_search, ...) means the model used a tool and the turn is not tool-free.
NON_TOOL_ITEMS = {"agent_message", "reasoning", "error"}

RATE_WORDS = ("usage limit", "rate limit", "too many requests", "429", "quota", "try again")


class CodexError(RuntimeError):
    pass


class Cancelled(RuntimeError):
    """The decision agent put a CANCEL file in runs/control/."""


def cancel_requested(cfg):
    f = cfg.get("_cancel_file")
    return bool(f) and os.path.exists(f)


def wait_or_cancel(cfg, seconds):
    end = time.time() + seconds
    while time.time() < end:
        if cancel_requested(cfg):
            raise Cancelled("cancelled while waiting to retry")
        time.sleep(min(2, max(0, end - time.time())))


def parse_events(path):
    """Read a `codex exec --json` stream: thread id, final message, usage, items, failures."""
    out = {"thread_id": None, "messages": [], "usage": None, "item_types": {}, "errors": [], "notices": []}
    if not Path(path).exists():
        return out
    for line in open(path, encoding="utf-8", errors="replace"):
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            e = json.loads(line)
        except json.JSONDecodeError:
            continue
        t = e.get("type", "")
        if t == "thread.started":
            out["thread_id"] = e.get("thread_id")
        elif t == "turn.completed":
            out["usage"] = e.get("usage")
        elif t == "turn.failed":
            out["errors"].append(e.get("error", e))
        elif t == "error":   # stream notices such as "Reconnecting... 2/5"; fatal only if no reply follows
            out["notices"].append(e.get("message", e))
        elif t == "item.completed":
            item = e.get("item", {})
            it = item.get("type", item.get("item_type", "?"))
            out["item_types"][it] = out["item_types"].get(it, 0) + 1
            if it == "agent_message":
                out["messages"].append(item.get("text", ""))
    return out


class CodexAgent:
    """A role (subject, auditor, extractor, reporter) talking to Codex in one conversation."""

    def __init__(self, cfg, role, home, model, effort, summary=None, log=None, env_extra=None):
        self.cfg, self.role, self.home = cfg, role, Path(home)
        self.model, self.effort, self.summary = model, effort, summary
        self.thread_id, self.turn = None, 0
        self.log = log or (lambda msg: None)
        self.env_extra = env_extra or {}
        self.workdir = self.home / "empty_cwd"
        self.workdir.mkdir(parents=True, exist_ok=True)

    def _cmd(self, answer_path):
        c = self.cfg["codex"]
        cmd = [c["binary"], "exec"]
        if self.thread_id:
            cmd.append("resume")
        cmd += list(c.get("extra_flags", [])) + ["--skip-git-repo-check", "--json"]
        if self.model:
            cmd += ["-m", self.model]
        if self.effort:
            cmd += ["-c", f'model_reasoning_effort="{self.effort}"']
        if self.summary:
            cmd += ["-c", f'model_reasoning_summary="{self.summary}"']
        cmd += ["-c", f'model_instructions_file="{self.home / "instructions.md"}"', "-o", str(answer_path)]
        if self.thread_id:
            cmd.append(self.thread_id)
        cmd.append("-")
        return cmd

    def send(self, message, turn_dir):
        """Send one message, wait for the reply (retrying through usage limits), return the reply text."""
        turn_dir = Path(turn_dir)
        turn_dir.mkdir(parents=True, exist_ok=True)
        (turn_dir / "prompt.txt").write_text(message, encoding="utf-8")
        env = dict(os.environ, CODEX_HOME=str(self.home), WSP_ROLE=self.role, **self.env_extra)
        waits = self.cfg["codex"]["retry_waits"]
        deadline = time.time() + 3600 * self.cfg["codex"]["max_wait_hours"]
        attempt = 0
        while True:
            attempt += 1
            ans_p, ev_p = turn_dir / "answer.md", turn_dir / "events.jsonl"
            for p in (ans_p, ev_p):
                p.unlink(missing_ok=True)
            cmd = self._cmd(ans_p)
            t0 = time.time()
            with open(ev_p, "w") as ev, open(turn_dir / "stderr.log", "w") as er, \
                    open(turn_dir / "prompt.txt", encoding="utf-8") as stdin:
                proc = subprocess.Popen(cmd, stdin=stdin, stdout=ev, stderr=er, cwd=self.workdir, env=env)
                while proc.poll() is None:
                    if cancel_requested(self.cfg):
                        proc.kill()
                        proc.wait()
                        raise Cancelled(f"{self.role}: codex call cancelled")
                    time.sleep(2)
                rc = proc.returncode
            wall = time.time() - t0
            info = parse_events(ev_p)
            reply = ans_p.read_text(encoding="utf-8") if ans_p.exists() else ""
            if not reply.strip() and info["messages"]:
                reply = info["messages"][-1]
            ok = rc == 0 and reply.strip() and not info["errors"]
            if ok:
                break
            err = ((turn_dir / "stderr.log").read_text(errors="replace")[-2000:] + " "
                   + json.dumps(info["errors"] + info["notices"][-4:], ensure_ascii=False)[:2000])
            wait = waits[min(attempt - 1, len(waits) - 1)]
            limited = any(w in err.lower() for w in RATE_WORDS)
            self.log(f"{self.role} turn {self.turn + 1} attempt {attempt}: codex failed (rc {rc}"
                     f"{', usage limit' if limited else ''}); waiting {wait} s. {err.strip()[:300]}")
            (turn_dir / f"fail_{attempt}.log").write_text(err)
            if time.time() + wait > deadline:
                raise CodexError(f"{self.role}: gave up after {attempt} attempts: {err[:500]}")
            wait_or_cancel(self.cfg, wait)
        if info["thread_id"] and not self.thread_id:
            self.thread_id = info["thread_id"]
        self.turn += 1
        tools = {k: v for k, v in info["item_types"].items() if k not in NON_TOOL_ITEMS}
        rec = {"ok": True, "role": self.role, "model": self.model, "effort": self.effort, "turn": self.turn,
               "thread": self.thread_id, "cmd": cmd, "rc": rc, "wall_s": round(wall, 1),
               "attempts": attempt, "usage": info["usage"], "notices": info["notices"][-10:],
               "audit": {"tool_free": not tools, "tool_items": tools, "item_types": info["item_types"]}}
        (turn_dir / "run.json").write_text(json.dumps(rec, indent=1))
        ans_p.write_text(reply, encoding="utf-8")
        return reply, rec

    def state(self):
        return {"thread_id": self.thread_id, "turn": self.turn}

    def restore(self, st):
        self.thread_id, self.turn = st.get("thread_id"), st.get("turn", 0)


# Keys of your own ~/.codex/config.toml that decide HOW Codex reaches the model (provider, endpoint, proxy, timeouts).
# They are carried into the isolated homes; nothing else of yours is (no instructions, MCP servers, profiles, tools).
NETWORK_KEYS = ("model_provider", "model_providers", "openai_base_url", "chatgpt_base_url", "forced_login_method",
                "preferred_auth_method", "stream_idle_timeout_ms", "request_max_retries", "stream_max_retries",
                "network_proxy", "http_proxy", "https_proxy")


def _toml_value(v):
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return repr(v)
    if isinstance(v, str):
        return json.dumps(v, ensure_ascii=False)
    if isinstance(v, list):
        return "[" + ", ".join(_toml_value(x) for x in v) + "]"
    if isinstance(v, dict):
        return "{ " + ", ".join(f"{json.dumps(k)} = {_toml_value(x)}" for k, x in v.items()) + " }"
    raise TypeError(type(v))


def network_overrides(user_config_path):
    """The network/provider part of your Codex config, as TOML text to append to an isolated config."""
    import tomllib
    p = Path(user_config_path).expanduser()
    if not p.exists():
        return "", "", []
    try:
        user = tomllib.loads(p.read_text())
    except Exception:
        return "", "", []
    top, tables, used = [], [], []
    for k in NETWORK_KEYS:
        if k not in user:
            continue
        used.append(k)
        v = user[k]
        if k == "model_providers" and isinstance(v, dict):
            for name, prov in v.items():
                tables.append(f"\n[model_providers.{json.dumps(name)}]")
                tables += [f"{json.dumps(a)} = {_toml_value(b)}" for a, b in prov.items()]
        else:
            top.append(f"{k} = {_toml_value(v)}")
    head = "# --- carried over from your own Codex config (network/provider only) ---\n" + "\n".join(top) + "\n"
    return (head if top else ""), ("\n".join(tables) + "\n" if tables else ""), used


def prepare_home(home, config_text, instructions, auth_source):
    """Create an isolated CODEX_HOME: config.toml, instructions.md, and a link to your Codex login."""
    home = Path(home)
    home.mkdir(parents=True, exist_ok=True)
    (home / "config.toml").write_text(config_text)
    (home / "instructions.md").write_text(instructions)
    auth = Path(auth_source).expanduser()
    link = home / "auth.json"
    if auth.exists() and not link.exists():
        link.symlink_to(auth)
    return home
