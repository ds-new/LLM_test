"""Runs the subject's Python code (exp3): a fresh process per run, no network, a time limit, output capped.

Isolation, best available on the machine:
  macOS  sandbox-exec: no network, no reading of the logs repo / runs folder / Codex logins, writes only in the work dir
  Linux  bubblewrap if installed (as in the authors' harness), else plain subprocess (with a warning)
The weights are copied into a work folder outside the logs repo, so the answer keys stay unreadable.
"""
import json, os, platform, shutil, subprocess, sys, tempfile, time
from pathlib import Path

MAC_PROFILE = """(version 1)
(allow default)
(deny network*)
{deny_reads}
(deny file-write*)
(allow file-write* (subpath "{work}") (subpath "/private/var/folders") (subpath "/private/tmp") (literal "/dev/null"))
"""


def _isolated_cmd(py, script, work, deny):
    sysname = platform.system()
    if sysname == "Darwin" and shutil.which("sandbox-exec"):
        deny_reads = "\n".join(f'(deny file-read* (subpath "{os.path.realpath(d)}"))' for d in deny if d)
        prof = MAC_PROFILE.format(deny_reads=deny_reads, work=os.path.realpath(work))
        return ["sandbox-exec", "-p", prof, py, "-I", script], "sandbox-exec"
    if sysname == "Linux" and shutil.which("bwrap"):
        cmd = ["bwrap", "--unshare-net", "--die-with-parent", "--new-session", "--dev-bind", "/", "/"]
        for d in deny:
            if d and os.path.exists(d):
                cmd += ["--tmpfs", os.path.realpath(d)]
        return cmd + ["--chdir", str(work), py, "-I", script], "bwrap"
    return [py, "-I", script], "none"


class Sandbox:
    def __init__(self, python, prelude, files, deny_read, timeout=600, max_out=20000, log=None):
        self.python = python if os.path.isabs(python) or "/" not in python else os.path.abspath(python)
        if not shutil.which(self.python) and not os.path.exists(self.python):
            self.python = sys.executable
        self.prelude, self.timeout, self.max_out = prelude, timeout, max_out
        self.deny = [str(Path(d).resolve()) for d in deny_read]
        self.base = Path(tempfile.mkdtemp(prefix="wsp_sbx_"))
        self.inputs = self.base / "inputs"
        self.inputs.mkdir()
        for name, src in files.items():
            shutil.copy(src, self.inputs / name)
        self.log = log
        self.kind = _isolated_cmd(self.python, "x.py", self.base, self.deny)[1]

    def run(self, code):
        work = Path(tempfile.mkdtemp(prefix="run_", dir=self.base))   # fresh folder: nothing kept between runs
        script = work / "main.py"
        script.write_text(self.prelude.replace("{INPUTS}", str(self.inputs)) + "\n" + code)
        cmd, self.kind = _isolated_cmd(self.python, str(script), work, self.deny)
        env = {"PATH": "/usr/bin:/bin", "HOME": str(work), "OMP_NUM_THREADS": "8", "CUDA_VISIBLE_DEVICES": "",
               "MPLBACKEND": "Agg", "PYTHONDONTWRITEBYTECODE": "1"}
        t0 = time.time()
        try:
            p = subprocess.run(cmd, capture_output=True, text=True, timeout=self.timeout, cwd=work, env=env)
            out = p.stdout + (("\n[stderr]\n" + p.stderr[-6000:]) if p.stderr.strip() else "")
        except subprocess.TimeoutExpired:
            out = f"[timed out after {self.timeout} s]"
        if len(out) > self.max_out:
            out = out[:self.max_out] + f"\n[output cut at {self.max_out:,} characters]"
        secs = round(time.time() - t0, 2)
        if self.log:
            with open(self.log, "a") as f:
                f.write(json.dumps({"t": time.time(), "kind": "python", "s": secs, "isolation": self.kind,
                                    "code": code, "out": out}) + "\n")
        shutil.rmtree(work, ignore_errors=True)
        return out or "(no output)", secs

    def close(self):
        shutil.rmtree(self.base, ignore_errors=True)
