#!/usr/bin/env bash
# One-time setup on your Mac: data, two Python environments, and a check of Codex + sandbox isolation.
set -euo pipefail
cd "$(dirname "$0")"
PY=${PYTHON:-python3}
"$PY" -c 'import sys; assert sys.version_info >= (3, 11), "need Python 3.11+ (tomllib)"; print("python", sys.version.split()[0])'

mkdir -p data
if [ ! -d data/lm_understand-logs/.git ]; then
  echo "== cloning the authors' logs (about 370 MB)"
  git clone --depth 1 https://github.com/RikiyaT/lm_understand-logs.git data/lm_understand-logs
fi

echo "== .venv: the loop itself (numpy; the official MinAtar 1.0.15 if pip has it)"
[ -d .venv ] || "$PY" -m venv .venv
.venv/bin/pip install -q -U pip
.venv/bin/pip install -q numpy
.venv/bin/pip install -q "minatar==1.0.15" || echo "   minatar 1.0.15 not installed: exp5 will use the game source quoted in the prompt"

echo "== .venv-sandbox: what the subject's code may import in exp3 (as in the authors' sandbox)"
[ -d .venv-sandbox ] || "$PY" -m venv .venv-sandbox
.venv-sandbox/bin/pip install -q -U pip
.venv-sandbox/bin/pip install -q numpy scipy scikit-learn torch pandas sympy networkx threadpoolctl

command -v codex >/dev/null || echo "!! codex not found: install the Codex CLI (npm i -g @openai/codex, or brew install codex), then: codex login"
chmod +x tests/fake_codex.py tests/replay_test.sh run.py
echo "== checking"
.venv/bin/python run.py --check || true
