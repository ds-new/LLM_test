#!/usr/bin/env bash
# 守护脚本：运行多智能体 loop，负责重启、卡死检测和结束后关机。
#   启动（工程根目录）：nohup bash decoder_from_scratch/a800/run_loop.sh >/dev/null 2>&1 &
# 行为：
#   * loop 可重试错误退出（码 1）→ 2 分钟后用最新代码重启，最多 20 次；
#   * 心跳文件超过 45 分钟没有更新 → 认为卡死，杀掉 loop 后重启（计一次失败）；
#   * 训练完成（码 0）、致命错误（码 2）或重试次数用尽 → 写入说明，2 分钟后执行 shutdown 关机（AutoDL 关机即停止计费）；
#   * 用户 stop（码 4）→ 守护脚本退出，不关机；
#   * 运行目录里放 NO_SHUTDOWN 文件可禁止自动关机（调试用）。
cd "$(dirname "$0")/../.." || exit 1
PY="${PYTHON:-python}"
RUN="${A8_RUN_DIR:-/root/autodl-tmp/a800_pretrain_v1}"
mkdir -p "$RUN/logs"
echo $$ > "$RUN/supervisor.pid"
log() { echo "$(date '+%F %T') $*" >> "$RUN/logs/supervisor.log"; }
failures=0
final=""
while true; do
  if [ -f "$RUN/STOP" ]; then log "STOP 文件存在，守护脚本退出（不关机）"; final="stopped"; break; fi
  log "启动 loop（累计失败 $failures 次）"
  date +%s > "$RUN/heartbeat"
  "$PY" -u -m decoder_from_scratch.a800.loop run >> "$RUN/logs/loop.log" 2>&1 &
  pid=$!
  while kill -0 $pid 2>/dev/null; do
    sleep 60
    hb=$(cut -d' ' -f1 "$RUN/heartbeat" 2>/dev/null || echo 0)
    age=$(( $(date +%s) - ${hb%.*} ))
    if [ "$age" -gt 2700 ]; then
      log "心跳已 ${age} 秒未更新，判定卡死，终止 loop（pid $pid）"
      kill $pid; sleep 30; kill -9 $pid 2>/dev/null
    fi
  done
  wait $pid; code=$?
  log "loop 退出，退出码 $code"
  case "$code" in
    0) final="done"; break ;;
    2) final="fatal"; break ;;
    3) final="duplicate"; break ;;
    4) final="stopped"; break ;;
    75) sleep 5; continue ;;
  esac
  failures=$((failures + 1))
  if [ "$failures" -ge 20 ]; then final="too_many_failures"; break; fi
  sleep 120
done
rm -f "$RUN/supervisor.pid"
if [ -f "$RUN/NEXT_ON_DONE.sh" ] && [ "$final" = "done" ]; then
  log "发现 NEXT_ON_DONE.sh，全部完成后执行后续任务而不关机"
  mv "$RUN/NEXT_ON_DONE.sh" "$RUN/NEXT_ON_DONE.sh.started"
  bash "$RUN/NEXT_ON_DONE.sh.started" >> "$RUN/logs/supervisor.log" 2>&1
  exit 0
fi
if [ -f "$RUN/NEXT.sh" ] && [ "$final" != "fatal" ] && [ "$final" != "too_many_failures" ]; then
  log "发现 NEXT.sh（$final），执行后续任务而不关机"
  mv "$RUN/NEXT.sh" "$RUN/NEXT.sh.started"
  bash "$RUN/NEXT.sh.started" >> "$RUN/logs/supervisor.log" 2>&1
  exit 0
fi
case "$final" in
  done|fatal|too_many_failures)
    "$PY" -m decoder_from_scratch.a800.loop status >> "$RUN/logs/supervisor.log" 2>&1
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
