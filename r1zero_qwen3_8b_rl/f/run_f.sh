#!/usr/bin/env bash
# F 训练：bash f/run_f.sh smoke|stage1 MODEL_HF
#   失败自动从最近的存盘续训（最多 3 次；日志里有显存不足就降一档）；
#   后台守护 1：存盘目录只留最新一份 global_step_*（数据盘只够放一份 16 GB；续训后 verl 不知道旧存盘，不会自己删）；
#   后台守护 2（stage1）：每分钟按事先定好的停训条件看指标，触发就停训，退出码 3。
R=/root/autodl-tmp/r1zero
cd $R
MODE=$1; MODEL=$2
LOG=$R/run/logs/f_stage.log
log() { echo "$(date '+%F %T') [run_f $MODE] $*" >> $LOG; }
if [ "$MODE" = smoke ]; then CK=$R/run/ckpt/f_smoke; TOTAL=1; else CK=$R/run/ckpt/f_sftrl; TOTAL=20; fi
rm -f $R/run/F_STOPPED
stop_all() { pkill -f recipe.dapo.main_dapo; sleep 15; ray stop --force > /dev/null 2>&1; pkill -f sglang; sleep 10; }
wait_gpu_free() {
  for t in $(seq 1 60); do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1)
    [ "$used" -lt 2000 ] && return 0
    sleep 10
  done
  log "GPUs still busy after 10 min: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader | tr '\n' ' ')"
}
( while sleep 30; do
    ls -d $CK/global_step_* 2>/dev/null | sort -V | head -n -1 | while read d; do log "guard: remove older checkpoint $d"; rm -rf "$d"; done
  done ) &
GUARD=$!
MON=
if [ "$MODE" = stage1 ]; then
  ( while sleep 60; do
      msg=$(python $R/f/monitor_f.py $MODE 2>&1)
      if [ $? -eq 1 ]; then echo "$msg" > $R/run/F_STOPPED; log "STOP RULE: $msg"; pkill -f recipe.dapo.main_dapo; break; fi
    done ) &
  MON=$!
fi
trap 'kill $GUARD $MON 2> /dev/null' EXIT
attempt=0; tier=0
while [ $attempt -lt 3 ]; do
  attempt=$((attempt + 1))
  wait_gpu_free
  python $R/f/make_command_f.py $MODE $MODEL $tier >> $LOG 2>&1 || { log "make_command failed"; exit 9; }
  L=$R/run/logs/f_$MODE"_"$(date +%m%d_%H%M).log
  ln -sf $L $R/run/logs/f_$MODE"_latest.log"
  log "attempt $attempt (memory tier $tier): $L"
  bash $R/run/command_f_$MODE.sh > $L 2>&1
  rc=$?
  log "attempt $attempt rc=$rc"
  if [ -f $R/run/F_STOPPED ]; then log "stopped by rule: $(cat $R/run/F_STOPPED)"; stop_all; exit 3; fi
  if [ $rc -eq 0 ] && ls $CK/global_step_$TOTAL/actor/huggingface/*.safetensors > /dev/null 2>&1; then
    log "done: $CK/global_step_$TOTAL"; exit 0
  fi
  if grep -a -q -i "out of memory\|OutOfMemoryError" $L; then tier=$((tier + 1)); log "OOM seen -> memory tier $tier"; fi
  stop_all
  T=$CK/latest_checkpointed_iteration.txt
  if [ -f $T ] && [ ! -d $CK/global_step_$(cat $T) ]; then log "tracker points to missing checkpoint $(cat $T): removing tracker"; rm -f $T; fi
done
log "giving up after $attempt attempts"
exit 1
