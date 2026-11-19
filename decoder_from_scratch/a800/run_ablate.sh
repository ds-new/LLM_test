#!/usr/bin/env bash
# 第一批消融（学习率扫描 + 多种子）的守护脚本：先等 tokenizer 对照实验结束，再依次训练；断点重启、卡死检测。
# 全部完成后，如果运行目录里有 CHAIN_ROUTE_A 文件，就衔接路线 A（清理数据 → 冒烟测试 → 正式启动）。
# 结束或失败时的关机规则与 run_loop.sh 相同：NO_SHUTDOWN 文件或 同机的 SAT 求解任务在跑时不关机。启动后不要再编辑本文件。
cd "$(dirname "$0")/../.." || exit 1
PY="${PYTHON:-/root/a800_venv/bin/python}"
export ABLATE_RUN_DIR="${ABLATE_RUN_DIR:-/root/autodl-tmp/a800_ablate}"
export A8_RUN_DIR="$ABLATE_RUN_DIR"
unset OMP_NUM_THREADS
RUN="$ABLATE_RUN_DIR"
mkdir -p "$RUN/logs"
echo $$ > "$RUN/supervisor.pid"
log() { echo "$(date '+%F %T') $*" >> "$RUN/logs/supervisor.log"; }
failures=0
final=""
while true; do
  if [ -f "$RUN/STOP" ]; then log "STOP 文件存在，退出（不关机）"; final="stopped"; break; fi
  log "启动（累计失败 $failures 次）"
  date +%s > "$RUN/heartbeat"
  "$PY" -u -m decoder_from_scratch.a800.ablate run >> "$RUN/logs/run.log" 2>&1 &
  pid=$!
  while kill -0 $pid 2>/dev/null; do
    sleep 60
    hb=$(cut -d' ' -f1 "$RUN/heartbeat" 2>/dev/null || echo 0)
    age=$(( $(date +%s) - ${hb%.*} ))
    if [ "$age" -gt 2700 ]; then
      log "心跳 $age 秒未更新，判定卡死，终止（pid $pid）"
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
if [ "$final" = "done" ] && [ -f "$RUN/CHAIN_ROUTE_A" ]; then
  log "第一批消融完成，衔接路线 A（冒烟测试 → 正式启动）"
  if "$PY" -m decoder_from_scratch.a800.cpt_launch after_ab >> "$RUN/logs/chain.log" 2>&1; then
    log "路线 A 已启动，本守护脚本退出（不关机）"
    exit 0
  fi
  log "衔接失败，见 logs/chain.log"
  final="chain_failed"
fi
case "$final" in
  done|fatal|too_many_failures|chain_failed)
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
