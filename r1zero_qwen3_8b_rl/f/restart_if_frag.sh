#!/usr/bin/env bash
# 2026-11-21 01:4x Claude (early version of the OOM handling; told the user at 01:05): during step 4 generation GPU 0/1 reached 81,030 of 81,920 MiB
# (training WorkerDict keeps 25.6 GiB of fragmented reserved memory after training + SGLang process 52.8 GB).
# At every generation start read the four "[r1z mem] rollout_mode start" lines; if the worst rank's reserved is above the threshold AND we are right after
# a checkpoint (no finished step lost), stop the current run and resume from the latest checkpoint with f/resume_f7.sh (F_MIN_TIER=2, SGLang 0.50),
# instead of generating for 1-2 h and then OOMing. One-shot. Also samples GPU memory every 20 s (f/patches/mem_sample.py -> run/logs/mem_samples.jsonl).
# usage: setsid nohup bash f/restart_if_frag.sh 26.5 0 > run/logs/restart_if_frag.out 2>&1 < /dev/null &
#   arg 2 = 1: dry run (judge the latest four lines at once, never stop anything, log to /tmp)
R=/root/autodl-tmp/r1zero
cd $R
TH=$1; [ -z "$TH" ] && TH=26.5
DRY=$2; [ -z "$DRY" ] && DRY=0
CK=$R/run/ckpt/f_sftrl
FLOG=$R/run/logs/f_stage.log
[ "$DRY" = 1 ] && FLOG=/tmp/restart_if_frag_dry.log
flog() { echo "$(date '+%F %T') [Claude restart_if_frag] $*" >> $FLOG; }
L=$(ls -t $R/run/logs/f_stage1_*.log | head -1)
MAIN=$(pgrep -f recipe.dapo.main_dapo | head -1)
N0=$(grep -a -c "rollout_mode start" $L)
[ "$DRY" = 1 ] && N0=$((N0 - 4))
[ "$DRY" = 1 ] || echo $$ > $R/run/restart_if_frag.pid
flog "watcher started (pid $$, threshold $TH GiB, dry=$DRY, log $L, main_dapo $MAIN, rollout_mode start lines so far $N0)"
DEADLINE=$(( $(date +%s) + 40 * 3600 ))
while true; do
  if [ $(date +%s) -gt $DEADLINE ]; then flog "40 h passed; exit"; exit 0; fi
  st=$(cat $R/f/F_STATUS 2>/dev/null)
  if [ "$st" != "stage1" ]; then flog "F_STATUS is '$st'; exit"; exit 0; fi
  if [ -f $R/run/F_STOPPED ]; then flog "stop rule fired; exit"; exit 0; fi
  cur=$(pgrep -f recipe.dapo.main_dapo | head -1)
  if [ "$cur" != "$MAIN" ]; then flog "main_dapo changed ('$MAIN' -> '$cur'): run_f.sh restarted by itself; exit"; exit 0; fi
  [ "$DRY" = 1 ] || python $R/f/patches/mem_sample.py 2>/dev/null
  N=$(grep -a -c "rollout_mode start" $L)
  if [ $N -ge $((N0 + 4)) ]; then
    [ "$DRY" = 1 ] || sleep 5
    N=$(grep -a -c "rollout_mode start" $L)
    VALS=$(grep -a "rollout_mode start" $L | tail -4 | grep -o "reserved [0-9.]*" | awk '{print $2}' | tr '\n' ' ')
    MAX=$(echo $VALS | tr ' ' '\n' | sort -g | tail -1)
    N0=$N
    T=$(cat $CK/latest_checkpointed_iteration.txt 2>/dev/null)
    NEXT=$((T + 1))
    if python $R/f/patches/has_step.py $NEXT > /dev/null 2>&1; then AFTER_SAVE=0; else AFTER_SAVE=1; fi
    flog "rollout start: reserved per rank [$VALS] GiB, max $MAX; latest checkpoint $T; right after that checkpoint: $AFTER_SAVE"
    if [ -n "$MAX" ] && awk "BEGIN{exit !($MAX > $TH)}"; then
      if [ "$AFTER_SAVE" != 1 ]; then
        flog "over threshold, but step $NEXT is already finished after checkpoint $T; leave it (an OOM is handled by run_f.sh, now escalating to tier 2)"
        [ "$DRY" = 1 ] && exit 0
        sleep 20; continue
      fi
      if ! python $R/f/patches/ckpt_ok.py $CK/global_step_$T/actor/huggingface >> $FLOG 2>&1; then
        flog "checkpoint $T incomplete; no restart"
        [ "$DRY" = 1 ] && exit 0
        sleep 20; continue
      fi
      flog "max reserved $MAX > $TH GiB: this generation would likely OOM at SGLang 0.55; restart from checkpoint $T at memory tier 2 (SGLang 0.50)"
      if [ "$DRY" = 1 ]; then flog "DRY: would stop resume_f6, run_f.sh, command_f_stage1.sh, main_dapo, ray, sglang and start f/resume_f7.sh"; exit 0; fi
      touch $R/run/REPAIRING
      P6=$(pgrep -f "f/resume_f6.sh" | head -1)
      flog "stopping: resume_f6 '$P6'; run_f.sh '$(pgrep -f f/run_f.sh | tr '\n' ' ')'; main_dapo '$(pgrep -f recipe.dapo.main_dapo | tr '\n' ' ')'"
      [ -n "$P6" ] && kill $P6
      sleep 3
      pkill -f "f/run_f.sh"
      sleep 2
      pkill -f "run/command_f_stage1.sh"
      sleep 1
      pkill -f recipe.dapo.main_dapo
      sleep 15
      ray stop --force > /dev/null 2>&1
      pkill -f sglang
      sleep 10
      for t in $(seq 1 60); do
        used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1)
        [ "$used" -lt 2000 ] && break
        touch $R/run/REPAIRING
        sleep 10
      done
      used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1)
      if [ "$used" -ge 2000 ]; then
        flog "GPUs still busy after stop; kill -9 GPU processes: $(nvidia-smi --query-compute-apps=pid --format=csv,noheader | tr '\n' ' ')"
        nvidia-smi --query-compute-apps=pid --format=csv,noheader | xargs -r kill -9
        sleep 30
      fi
      flog "after stop: GPU MiB $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | tr '\n' ' '); left: $(pgrep -f 'resume_f6|f/run_f.sh|main_dapo' | wc -l) processes"
      touch $R/run/KEEP_RUNNING
      setsid nohup bash $R/f/resume_f7.sh > $R/run/logs/resume_f7.out 2>&1 < /dev/null &
      sleep 5
      flog "started f/resume_f7.sh (pid $(pgrep -f f/resume_f7.sh | head -1)); stage1 resumes from checkpoint $T at memory tier 2"
      for t in $(seq 1 40); do
        sleep 30
        touch $R/run/REPAIRING
        L2=$(ls -t $R/run/logs/f_stage1_*.log | head -1)
        if [ "$L2" != "$L" ] && grep -a -q "Decode batch" $L2 2>/dev/null; then flog "generation running after restart ($L2): $(grep -a -o 'mem_fraction_static=[0-9.]*' $L2 | sort -u | tr '\n' ' ')"; break; fi
      done
      rm -f $R/run/REPAIRING
      flog "watcher done"
      exit 0
    fi
    if [ "$DRY" = 1 ]; then flog "DRY: under threshold, nothing to do"; exit 0; fi
  fi
  sleep 20
done
