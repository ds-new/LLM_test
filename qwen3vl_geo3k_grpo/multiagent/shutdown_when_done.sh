#!/usr/bin/env bash
# 自动关机看守（AutoDL 关机即停止 GPU 计费，数据盘保留）：
#   1. 训练全部完成（state.json 的 phase == done）→ 等 2 分钟后关机；
#   2. 异常停止：loop 与守护脚本都不在运行、且 phase != done，持续 30 分钟未恢复 → 关机。
# 任何时候存在 $MA_RUN_DIR/KEEP_RUNNING 文件都不会关机。
# 用法（工程根目录）：setsid nohup bash multiagent/shutdown_when_done.sh > /dev/null 2>&1 < /dev/null &
R="${MA_RUN_DIR:-/root/autodl-tmp/geo3k_runs/multiagent_v1}"
IDLE_LIMIT="${IDLE_LIMIT_SECONDS:-1800}"
LOG="$R/logs/shutdown_watcher.log"
mkdir -p "$R/logs"
echo $$ > "$R/shutdown_watcher.pid"
log() { echo "$(date '+%F %T') $*" >> "$LOG"; }
do_shutdown() {
  if [ -f "$R/KEEP_RUNNING" ]; then log "KEEP_RUNNING present; shutdown cancelled ($1)"; return 1; fi
  log "shutdown now: $1"
  sync
  shutdown -h now || shutdown
  exit 0
}
log "watcher started (pid $$), idle limit ${IDLE_LIMIT}s"
idle_since=""
while true; do
  phase=$(python3 -c "import json;print(json.load(open('$R/state.json')).get('phase',''))" 2>/dev/null)
  if [ "$phase" = "done" ]; then
    log "phase=done; shutting down in 120s (touch $R/KEEP_RUNNING to cancel)"
    sleep 120
    do_shutdown "training finished" || { sleep 600; continue; }
  fi
  if pgrep -f "multiagent.loop run" >/dev/null || pgrep -f "multiagent/run_loop.sh" >/dev/null; then
    [ -n "$idle_since" ] && log "loop running again; idle timer reset"
    idle_since=""
  else
    now=$(date +%s)
    if [ -z "$idle_since" ]; then
      idle_since=$now
      log "loop not running (phase=$phase); will shut down if not resumed within ${IDLE_LIMIT}s"
    elif [ $((now - idle_since)) -ge "$IDLE_LIMIT" ]; then
      err=$(python3 -c "import json;print((json.load(open('$R/state.json')).get('last_error') or {}).get('error','')[:200])" 2>/dev/null)
      do_shutdown "abnormal stop for $(( (now - idle_since) / 60 )) min, phase=$phase, last_error=$err" || idle_since=$now
    fi
  fi
  sleep 60
done
