#!/usr/bin/env bash
# M6 shutdown watchdog (AutoDL stops GPU billing on shutdown; the data disk is kept).
# Shuts the instance down once run_m6.sh has not been running for IDLE_LIMIT seconds (default 600 = 10 min):
# after it finished normally, failed, crashed or was killed. Any time $W/KEEP_RUNNING exists, nothing is shut down
# and the idle timer is reset. Started by run_m6.sh (one copy only); log in $W/watchdog.log.
W=${M6_DIR:-/root/autodl-tmp/wsp_m6}
IDLE_LIMIT=${IDLE_LIMIT_SECONDS:-600}
LOG=$W/watchdog.log
mkdir -p "$W"
if [ -f "$W/watchdog.pid" ] && kill -0 "$(cat "$W/watchdog.pid")" 2>/dev/null; then exit 0; fi
echo $$ > "$W/watchdog.pid"
log() { echo "$(date '+%F %T') $*" >> "$LOG"; }
log "watchdog started (pid $$), shuts down ${IDLE_LIMIT}s after run_m6.sh stops"
idle=0
while true; do
  if [ -f "$W/KEEP_RUNNING" ]; then
    [ "$idle" -gt 0 ] && log "KEEP_RUNNING present: idle timer reset"
    idle=0
  elif pgrep -f "bash run_m6.sh" > /dev/null; then
    [ "$idle" -gt 0 ] && log "run_m6.sh running again: idle timer reset"
    idle=0
  else
    [ "$idle" -eq 0 ] && log "run_m6.sh not running ($( [ -f "$W/DONE" ] && echo finished || echo stopped abnormally )); shutting down in ${IDLE_LIMIT}s unless it restarts or KEEP_RUNNING appears"
    idle=$((idle + 30))
    if [ "$idle" -ge "$IDLE_LIMIT" ]; then
      log "shutdown now"
      sync
      shutdown -h now || shutdown
      exit 0
    fi
  fi
  sleep 30
done
