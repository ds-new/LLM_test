#!/usr/bin/env bash
# Offline test of the whole loop, no Codex quota used: a fake `codex` replays Astra's recorded replies from the logs.
# Checks that (1) every first prompt is byte-identical to what the authors sent, (2) exp3's multi-turn protocol
# lines up with theirs, (3) the graders reproduce the post's numbers from the recorded answers.
set -euo pipefail
cd "$(dirname "$0")/.."
export WSP_REPLAY_LOGS="$PWD/data/lm_understand-logs"
rm -rf tests/runs_replay
PYBIN=${PYBIN:-.venv/bin/python}
$PYBIN run.py --config tests/config.replay.toml --exp exp1 exp2 exp3 exp5 --unit \
  exp1=breakout exp1=pong exp1=pacman exp1=freeway exp2=circle exp2=square exp2=triangle \
  exp3=python_r0 exp5=breakout exp5=seaquest exp5=asterix > tests/replay.log 2>&1
$PYBIN - <<'EOF'
import json
rows = [json.loads(l) for l in open("tests/runs_replay/_codex_homes/subject/fake_checks.jsonl")]
first = [r for r in rows if r["turn"] == 1]
later = [r for r in rows if r["turn"] > 1]
print(f"first prompts identical to the authors': {sum(r['prompt_identical'] for r in first)}/{len(first)}")
print(f"exp3 later turns in the authors' format:  {sum(r['header_identical'] and r['frame_ok'] for r in later)}/{len(later)}")
EOF
cat tests/runs_replay/results.md
