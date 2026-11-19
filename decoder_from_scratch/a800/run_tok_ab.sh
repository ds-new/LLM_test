#!/usr/bin/env bash
# tokenizer 对照实验的守护脚本：断点重启、卡死检测（心跳 45 分钟）、结束或致命错误后关机（与 run_loop.sh 相同的约定）。
#   启动（工程根目录）：nohup bash decoder_from_scratch/a800/run_tok_ab.sh >/dev/null 2>&1 &
#   STOP 文件：退出且不关机；NO_SHUTDOWN 文件：结束后不关机。
cd "$(dirname "$0")/../.." || exit 1
PY="${PYTHON:-/root/a800_venv/bin/python}"
export AB_RUN_DIR="${AB_RUN_DIR:-/root/autodl-tmp/a800_tok_ab}"
export A8_RUN_DIR="$AB_RUN_DIR"
export AB_SCRATCH="${AB_SCRATCH:-$AB_RUN_DIR/scratch}"
unset OMP_NUM_THREADS
RUN="$AB_RUN_DIR"
mkdir -p "$RUN/logs"
echo $$ > "$RUN/supervisor.pid"
log() { echo "$(date '+%F %T') $*" >> "$RUN/logs/supervisor.log"; }
failures=0
final=""
while true; do
  if [ -f "$RUN/STOP" ]; then log "STOP 文件存在，退出（不关机）"; final="stopped"; break; fi
  log "启动（累计失败 $failures 次）"
  date +%s > "$RUN/heartbeat"
  "$PY" -u -m decoder_from_scratch.a800.tok_ab run >> "$RUN/logs/run.log" 2>&1 &
  pid=$!
  while kill -0 $pid 2>/dev/null; do
    sleep 60
    hb=$(cut -d' ' -f1 "$RUN/heartbeat" 2>/dev/null || echo 0)
    age=$(( $(date +%s) - ${hb%.*} ))
    if [ "$age" -gt 2700 ]; then
      log "心跳 ${age} 秒未更新，判定卡死，终止（pid $pid）"
      kill $pid; sleep 30; kill -9 $pid 2>/dev/null
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
  if [ "$failures" -ge 8 ]; then final="too_many_failures"; break; fi
  sleep 120
done
rm -f "$RUN/supervisor.pid"
if [ "$final" = "done" ] && [ -f "$RUN/CHAIN" ]; then
  log "对照完成，衔接路线 A（冒烟测试 → 正式启动）"
  if "$PY" -m decoder_from_scratch.a800.cpt_launch after_ab >> "$RUN/logs/chain.log" 2>&1; then
    log "路线 A 已启动，本守护脚本退出（不关机）"
    exit 0
  fi
  log "衔接失败，见 logs/chain.log"
fi
case "$final" in
  done|fatal|too_many_failures)
    if [ -f "$RUN/NO_SHUTDOWN" ]; then
      log "结束（$final），存在 NO_SHUTDOWN，不关机"
    else
      log "结束（$final），2 分钟后关机"
      sync; sleep 120; sync
      if pgrep -x cadical >/dev/null || pgrep -x kissat >/dev/null || pgrep -x lrat-trim >/dev/null; then
        log "同机的 SAT 求解任务仍在运行（cadical/kissat/lrat-trim），为保护该任务不关机"
      else
        shutdown -h now || /usr/bin/shutdown || poweroff
      fi
    fi ;;
  *) log "守护脚本退出（$final）" ;;
esac
