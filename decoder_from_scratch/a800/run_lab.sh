#!/usr/bin/env bash
# 实验室（第二、三批消融）守护脚本：断点重启、卡死检测；全部完成或无法恢复时按统一规则关机。
# 由路线 A 的 NEXT_ON_DONE.sh 启动；手动启动：cd /root/autodl-tmp/verl_megatron_sglang && setsid nohup bash decoder_from_scratch/a800/run_lab.sh >/dev/null 2>&1 &
# 关机规则：运行目录有 NO_SHUTDOWN、或 同机的 SAT 求解任务在跑时不关机；运行目录有 NEXT_ON_DONE.sh 时完成后执行它而不关机。启动后不要再编辑本文件。
cd "$(dirname "$0")/../.." || exit 1
PY="$PYTHON"; [ -z "$PY" ] && PY=/root/a800_venv/bin/python
[ -z "$LAB_RUN_DIR" ] && LAB_RUN_DIR=/root/autodl-tmp/a800_lab
[ -z "$LAB_SCRATCH" ] && LAB_SCRATCH=/root/a800_scratch/lab
export LAB_RUN_DIR LAB_SCRATCH
export A8_RUN_DIR="$LAB_RUN_DIR"
export PYTHON="$PY"
unset OMP_NUM_THREADS
RUN="$LAB_RUN_DIR"
mkdir -p "$RUN/logs"
echo $$ > "$RUN/supervisor.pid"
log() { echo "$(date '+%F %T') $*" >> "$RUN/logs/supervisor.log"; }
failures=0
final=""
while true; do
  if [ -f "$RUN/STOP" ]; then log "STOP 文件存在，退出（不关机）"; final="stopped"; break; fi
  log "启动（累计失败 $failures 次）"
  date +%s > "$RUN/heartbeat"
  "$PY" -u -m decoder_from_scratch.a800.lab run >> "$RUN/logs/run.log" 2>&1 &
  pid=$!
  while kill -0 $pid 2>/dev/null; do
    sleep 60
    hb=$(cut -d" " -f1 "$RUN/heartbeat" 2>/dev/null | cut -d. -f1)
    [ -z "$hb" ] && hb=0
    age=$(( $(date +%s) - hb ))
    if [ "$age" -gt 2700 ]; then
      log "心跳 $age 秒未更新，判定卡死，终止（pid $pid 及其子进程）"
      pkill -TERM -P $pid 2>/dev/null; kill $pid; sleep 30; pkill -KILL -P $pid 2>/dev/null; kill -9 $pid 2>/dev/null
    fi
  done
  wait $pid; code=$?
  log "退出，退出码 $code"
  case "$code" in
    0) final="done"; break ;;
    2) final="fatal"; break ;;
    4) final="stopped"; break ;;
  esac
  failures=$((failures + 1))
  if [ "$failures" -ge 10 ]; then final="too_many_failures"; break; fi
  sleep 120
done
rm -f "$RUN/supervisor.pid"
if [ "$final" = "done" ] && [ -f "$RUN/NEXT_ON_DONE.sh" ]; then
  log "发现 NEXT_ON_DONE.sh，完成后执行它而不关机"
  mv "$RUN/NEXT_ON_DONE.sh" "$RUN/NEXT_ON_DONE.sh.started"
  bash "$RUN/NEXT_ON_DONE.sh.started" >> "$RUN/logs/supervisor.log" 2>&1
  exit 0
fi
case "$final" in
  done|fatal|too_many_failures)
    if [ -f "$RUN/NO_SHUTDOWN" ]; then
      log "结束（$final），存在 NO_SHUTDOWN，不关机"
    elif pgrep -x cadical >/dev/null || pgrep -x kissat >/dev/null || pgrep -x lrat-trim >/dev/null; then
      log "结束（$final），同机的 SAT 求解任务仍在运行，为保护该任务不关机"
    else
      log "结束（$final），2 分钟后关机"
      sync; sleep 120; sync
      shutdown -h now || /usr/bin/shutdown || poweroff
    fi ;;
  *) log "守护脚本退出（$final）" ;;
esac
