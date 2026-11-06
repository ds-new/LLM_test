#!/usr/bin/env python3
"""A stand-in for the `codex` binary, for testing the loop offline: it replays Astra's recorded replies from the logs
repo (subject) and gives canned replies for the helper roles. It speaks the same CLI and --json event stream:
    codex exec [resume] ... -o <answer> [<thread>] -      (prompt on stdin)
Every subject prompt it receives is checked against the prompt the authors actually sent (fake_checks.jsonl).
"""
import hashlib, json, os, sys
from pathlib import Path

args = sys.argv[1:]
if "--version" in args:
    print("fake-codex 0.0 (replays the authors' logs)")
    sys.exit(0)
prompt = sys.stdin.read()
home = Path(os.environ["CODEX_HOME"])
logs = Path(os.environ["WSP_REPLAY_LOGS"])
role, task = os.environ.get("WSP_ROLE", "?"), os.environ.get("WSP_TASK", "")
out_path = args[args.index("-o") + 1]
resume = "resume" in args
threads = home / "fake_threads"
threads.mkdir(exist_ok=True)
if resume:
    thread = args[args.index("-") - 1]
else:
    thread = "fake-" + hashlib.sha1((task + role + str(os.getpid())).encode()).hexdigest()[:8]
tf = threads / f"{thread}.json"
n = json.loads(tf.read_text())["turn"] + 1 if tf.exists() else 1
tf.write_text(json.dumps({"turn": n, "task": task}))

check = None
if role == "subject":
    exp, unit = task.split("/")
    folder = {"exp1": "exp1_which_game", "exp2": "exp2_diffusion", "exp3": "exp3_language",
              "exp5": "exp5_write_from_scratch"}[exp]
    run = logs / folder / "runs" / unit
    if exp in ("exp1", "exp2"):
        reply, recorded = (run / "answer.md").read_text(), (run / "prompt.txt").read_text()
    elif exp == "exp5":
        reply, recorded = (run / "reply" / "answer.md").read_text(), (run / "reply" / "prompt.txt").read_text()
    else:
        t = run / "turns" / f"t{n:03d}"
        reply, recorded = (t / "answer.md").read_text(), (t / "prompt.txt").read_text()
    if n == 1:
        check = {"task": task, "turn": n, "prompt_identical": prompt == recorded}
    else:  # later exp3 turns carry our own sandbox output, so compare the protocol, not the numbers
        check = {"task": task, "turn": n, "header_identical": prompt.split("\n", 1)[0] == recorded.split("\n", 1)[0],
                 "frame_ok": prompt.split("\n", 2)[1] == "```" and prompt.rstrip().endswith("```")}
elif role == "auditor":
    reply = '```json\n{"verdict": "ok", "rule": "", "evidence": "", "reason": "fake auditor"}\n```'
elif role == "extractor":
    reply = '```json\n{}\n```'
else:
    reply = "# Report (fake reporter)\n\nThe offline replay finished."

if check:
    with open(home / "fake_checks.jsonl", "a") as f:
        f.write(json.dumps(check) + "\n")
Path(out_path).write_text(reply)
events = []
if not resume:
    events.append({"type": "thread.started", "thread_id": thread})
events += [{"type": "turn.started"},
           {"type": "item.completed", "item": {"id": "item_0", "type": "reasoning", "text": "**replay**"}},
           {"type": "item.completed", "item": {"id": "item_1", "type": "agent_message", "text": reply}},
           {"type": "turn.completed", "usage": {"input_tokens": len(prompt) // 4, "cached_input_tokens": 0,
                                                "output_tokens": len(reply) // 4}}]
for e in events:
    print(json.dumps(e), flush=True)
