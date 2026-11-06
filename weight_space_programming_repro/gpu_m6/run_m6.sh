#!/usr/bin/env bash
# M6 on AutoDL: rebuild the authors' Qwen3.5-0.8B weight edits bit for bit and re-run the paper's evaluation.
#   cd /root/autodl-tmp/wsp_gpu/m6 && setsid nohup bash run_m6.sh > /dev/null 2>&1 < /dev/null &
#   tail -f /root/autodl-tmp/wsp_m6/progress.log
# Stages, in priority order (each is resumable; rerunning the script skips what is done):
#   setup -> fetch -> build -> base/math500 -> p50c/math500 -> base/gsm8k -> p50c/gsm8k -> p10c,p25c/math500 -> RESULTS.md
# Shutdown (AutoDL stops GPU billing on shutdown; the data disk is kept): m6_watchdog.sh shuts the instance down 10 min
# after this script stops, whether it finished, failed, crashed or was killed, unless $W/KEEP_RUNNING exists.
# Set SHUTDOWN_WHEN_DONE=0 to never shut down. BUDGET_HOURS limits the optional stages.
set -u
cd "$(dirname "$0")"
HERE=$PWD
W=${M6_DIR:-/root/autodl-tmp/wsp_m6}
BUDGET_HOURS=${BUDGET_HOURS:-6}
SHUTDOWN_WHEN_DONE=${SHUTDOWN_WHEN_DONE:-1}
mkdir -p "$W"
P=$W/progress.log
START=$(date +%s)
export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com} VLLM_WORKER_MULTIPROC_METHOD=spawn PYTHONUNBUFFERED=1
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY   # hf-mirror and ModelScope are domestic: no proxy
say() { echo "$(date '+%F %T') $*" | tee -a "$P"; }
hours() { echo $(( ($(date +%s) - START) / 3600 )); }
finish() {
  say "END: $1"
  cp "$P" "$W/progress_final.log" 2>/dev/null
  [ "$SHUTDOWN_WHEN_DONE" = "1" ] && say "the watchdog shuts the instance down in 10 min (touch $W/KEEP_RUNNING to cancel)"
  exit 0
}
fail() { say "FAIL $1"; echo "$1" > "$W/FAILED"; finish "failed at $1"; }
echo $$ > "$W/run.pid"
rm -f "$W/FAILED"
say "start (budget ${BUDGET_HOURS} h, shutdown_when_done=${SHUTDOWN_WHEN_DONE})"
if [ "$SHUTDOWN_WHEN_DONE" = "1" ]; then
  (M6_DIR=$W setsid nohup bash "$HERE/m6_watchdog.sh" > /dev/null 2>&1 < /dev/null &)
fi

# ---- setup ------------------------------------------------------------------------------------------------------------
nvidia-smi -L >> "$P" 2>&1 || fail "no GPU (start the instance with a GPU, not in no-GPU mode)"
free=$(df -BG --output=avail /root/autodl-tmp 2>/dev/null | tail -1 | tr -dc 0-9); say "data disk free ${free:-?}G"
[ -n "$free" ] && [ "$free" -lt 15 ] && fail "data disk has < 15G free"
python3 -c "import math_verify" 2>/dev/null || pip install -q "math-verify[antlr4_13_2]" >> "$W/pip.log" 2>&1 || fail "pip install math-verify"
python3 -c "import huggingface_hub, datasets" 2>/dev/null || pip install -q huggingface_hub datasets >> "$W/pip.log" 2>&1
arch_ok() { python3 - <<'EOF' 2>>"$W/pip.log"
import vllm
from vllm import ModelRegistry
a = set(ModelRegistry.get_supported_archs())
print(vllm.__version__, "Qwen3_5ForConditionalGeneration" in a, "Qwen3_5ForCausalLM" in a)
EOF
}
v=$(arch_ok); say "vllm: ${v:-not installed} (version, supports Qwen3_5ForConditionalGeneration, Qwen3_5ForCausalLM)"
if ! echo "$v" | grep -q " True"; then
  say "vLLM missing or too old for Qwen3.5: pip install -U vllm (several minutes)"
  pip install -U vllm >> "$W/pip.log" 2>&1 || fail "pip install -U vllm (see $W/pip.log)"
  v=$(arch_ok); say "vllm after upgrade: $v"
  echo "$v" | grep -q " True" || fail "this vLLM cannot load Qwen3.5 (see $W/pip.log)"
fi

# ---- fetch + build ----------------------------------------------------------------------------------------------------
python3 fetch.py model "$W/release" >> "$W/fetch.log" 2>&1 || fail "model download (see $W/fetch.log)"
say "model: $(cat $W/release/_source.json 2>/dev/null)"
python3 fetch.py bench "$HERE/bench" >> "$W/fetch.log" 2>&1 || fail "benchmark download (see $W/fetch.log)"
say "bench: $(wc -l < bench/math500.jsonl) MATH-500, $(wc -l < bench/gsm8k.jsonl) GSM8K"
if [ ! -f "$W/ckpt/build_report.json" ]; then
  python3 build.py "$W/release" "$W/ckpt" p50c p10c p25c >> "$W/build.log" 2>&1 || fail "build (see $W/build.log)"
fi
say "build: $(tail -3 $W/build.log | tr '\n' ' ')"

# ---- evaluation, most important first ---------------------------------------------------------------------------------
ev() {  # ev <checkpoint> <sets>
  say "eval $1 $2: start"
  python3 eval35.py "$W/ckpt/$1" "$W/eval/$1" --sets "$2" --bench "$HERE/bench" >> "$W/eval_$1.log" 2>&1 \
    || fail "eval $1 $2 (see $W/eval_$1.log)"
  say "eval $1 $2: $(python3 -c "
import json; d=json.load(open('$W/eval/$1/scores.json'))
print({s: (round(100*d[s]['acc'],1), 'cut', round(100*d[s]['cut_share'],1), str(d[s]['s'])+'s') for s in '$2'.split(',') if s in d})")"
}
ev base math500
ev p50c math500
python3 compare.py "$W/eval" > "$W/RESULTS.md"; say "core result written: $W/RESULTS.md"
for step in "base gsm8k" "p50c gsm8k" "p10c math500" "p25c math500"; do
  if [ "$(hours)" -ge "$BUDGET_HOURS" ]; then say "budget reached: skipping $step"; continue; fi
  ev $step
  python3 compare.py "$W/eval" > "$W/RESULTS.md"
done
say "$(cat $W/RESULTS.md | tail -12)"
touch "$W/DONE"
finish "all stages done in $(( ($(date +%s) - START) / 60 )) min"
