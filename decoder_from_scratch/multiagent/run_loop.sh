#!/usr/bin/env bash
# 守护脚本：运行多智能体 loop；异常退出后等待 2 分钟用最新代码自动重启，最多 30 次。
# 用法（工程根目录）：nohup caffeinate -is bash decoder_from_scratch/multiagent/run_loop.sh >/dev/null 2>&1 &
cd "$(dirname "$0")/../.." || exit 1
PY="${PYTHON:-python}"
RUN="${MA_RUN_DIR:-decoder_from_scratch/runs/multiagent_pretrain_v1}"
mkdir -p "$RUN/logs"
echo $$ > "$RUN/supervisor.pid"
failures=0
while true; do
  if [ -f "$RUN/STOP" ]; then
    echo "$(date '+%F %T') STOP file present; supervisor exits" >> "$RUN/logs/supervisor.log"
    break
  fi
  echo "$(date '+%F %T') starting loop" >> "$RUN/logs/supervisor.log"
  "$PY" -u -m decoder_from_scratch.multiagent.loop run >> "$RUN/logs/loop.log" 2>&1
  code=$?
  echo "$(date '+%F %T') loop exited with $code" >> "$RUN/logs/supervisor.log"
  case "$code" in
    0|2|3) break ;;
    75) failures=0; sleep 5; continue ;;
  esac
  failures=$((failures + 1))
  if [ "$failures" -ge 30 ]; then
    echo "$(date '+%F %T') too many failures; supervisor exits" >> "$RUN/logs/supervisor.log"
    break
  fi
  sleep 120
done
rm -f "$RUN/supervisor.pid"
