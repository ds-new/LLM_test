#!/usr/bin/env bash
# 守护脚本（AutoDL）：运行多智能体 loop；异常退出后等待 2 分钟用最新代码自动重启，最多 30 次。
#
# 用法（工程根目录）：
#   nohup bash multiagent/run_loop.sh > /dev/null 2>&1 &
#   python -m multiagent.loop status          # 查看进度；tail -f $MA_RUN_DIR/logs/loop.log 看实时日志
#   python -m multiagent.loop stop            # 停止
#   wandb login                               # 可选：登录后训练曲线自动同步到 wandb（见 multiagent/wandb_sync.py）
# 关闭 SSH / JupyterLab 不影响运行；建议放在 tmux 或 nohup 下。
cd "$(dirname "$0")/.." || exit 1
PY="${PYTHON:-python3}"
if [ -z "${MA_RUN_DIR}" ]; then
  if [ "${MA_DRY_RUN}" = "1" ]; then MA_RUN_DIR="$PWD/runs/dryrun";
  elif [ -d /root/autodl-tmp ]; then MA_RUN_DIR=/root/autodl-tmp/geo3k_runs/multiagent_v1;
  else MA_RUN_DIR="$PWD/runs/multiagent_v1"; fi
fi
export MA_RUN_DIR
# AutoDL 学术资源加速（huggingface / github）；HF 默认走镜像
[ -f /etc/network_turbo ] && source /etc/network_turbo >/dev/null 2>&1
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
mkdir -p "$MA_RUN_DIR/logs"
# 手动启动守护脚本即表示要运行：清除上一次 `loop stop` 留下的 STOP 文件
if [ -f "$MA_RUN_DIR/STOP" ]; then
  rm -f "$MA_RUN_DIR/STOP"
  echo "$(date '+%F %T') removed stale STOP file at supervisor start" >> "$MA_RUN_DIR/logs/supervisor.log"
fi
echo $$ > "$MA_RUN_DIR/supervisor.pid"
# 旁路监控：把训练过程同步到 wandb / TensorBoard（只读 loop 的产物，不影响训练；已在运行则不重复启动）
start_monitor() {
  [ "${MA_MONITOR:-1}" = "0" ] && return
  local pidf="$MA_RUN_DIR/monitor/wandb_sync.pid"
  if [ -f "$pidf" ] && kill -0 "$(cat "$pidf")" 2>/dev/null; then return; fi
  mkdir -p "$MA_RUN_DIR/monitor"
  setsid nohup "$PY" -u -m multiagent.wandb_sync >> "$MA_RUN_DIR/logs/wandb_sync.log" 2>&1 < /dev/null &
}
failures=0
while true; do
  start_monitor
  if [ -f "$MA_RUN_DIR/STOP" ]; then
    echo "$(date '+%F %T') STOP file present; supervisor exits" >> "$MA_RUN_DIR/logs/supervisor.log"
    break
  fi
  echo "$(date '+%F %T') starting loop" >> "$MA_RUN_DIR/logs/supervisor.log"
  "$PY" -u -m multiagent.loop run >> "$MA_RUN_DIR/logs/loop.log" 2>&1
  code=$?
  echo "$(date '+%F %T') loop exited with $code" >> "$MA_RUN_DIR/logs/supervisor.log"
  case "$code" in
    0|2|3) break ;;
    75) failures=0; sleep 5; continue ;;
  esac
  failures=$((failures + 1))
  if [ "$failures" -ge 30 ]; then
    echo "$(date '+%F %T') too many failures; supervisor exits" >> "$MA_RUN_DIR/logs/supervisor.log"
    break
  fi
  if [ "${MA_DRY_RUN}" = "1" ]; then sleep 1; else sleep 120; fi
done
rm -f "$MA_RUN_DIR/supervisor.pid"
