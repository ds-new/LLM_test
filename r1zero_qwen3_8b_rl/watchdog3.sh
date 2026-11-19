#!/usr/bin/env bash
# 看门狗（关机的唯一决策者，每分钟检查一次）：
#   - 训练正常结束（phase=done）：loop 自己 60 秒后关机，这里 10 分钟兜底；
#   - 训练报错停止（phase=failed）、或 loop 进程消失且没收尾：10 分钟后关机；
#   - 正在修复（run/REPAIRING 存在且 20 分钟内更新过，修复者每次操作都 touch 它）：不关机，计时清零；
#     REPAIRING 超过 20 分钟没更新 = 修复中断/没修好，按上面的 10 分钟规则关机；
#   - 运行目录 150 分钟没有任何文件更新（卡死；第二轮每步可能超过 50 分钟，所以放宽）：结束训练进程、标记 failed，再走 10 分钟规则；
#   - run/KEEP_RUNNING：用户手动保留开机（只在用户自己创建时使用）。
R=/root/autodl-tmp/r1zero/run
L=$R/logs/watchdog.log
REPAIR_TTL=${REPAIR_TTL:-1200}
GRACE=${GRACE:-1800}  # 2026-11-16：异常暂停后等 Claude 30 分钟（Claude 每 20 分钟检查一次）
log() { echo "$(date '+%F %T') $*" >> $L; }
off() { log "shutdown: $*"; sync; /usr/bin/shutdown; exit 0; }
log "watchdog start (pid $$, repair_ttl=${REPAIR_TTL}s, grace=${GRACE}s)"
t=0; note=""
while true; do
  sleep 60
  if [ -f $R/KEEP_RUNNING ]; then
    [ "$note" != keep ] && log "KEEP_RUNNING present: shutdown disabled until removed"; note=keep; t=0; continue
  fi
  if [ -f $R/REPAIRING ]; then
    age=$(( $(date +%s) - $(stat -c %Y $R/REPAIRING) ))
    if [ $age -lt $REPAIR_TTL ]; then
      [ "$note" != repair ] && log "repair in progress (REPAIRING age ${age}s): holding"; note=repair; t=0; continue
    fi
    [ "$note" != stale ] && log "REPAIRING stale (${age}s without update): treat as unresolved"; note=stale
  else
    note=""
  fi
  phase=$(python3 -c "import json;print(json.load(open('$R/state.json')).get('phase',''))" 2>/dev/null)
  running=0; pgrep -f 'python -u -m ma.loop3' >/dev/null && running=1
  if [ "$phase" = "done" ]; then
    t=$((t+60)); [ $t -ge $GRACE ] && off "phase=done (backup)"; continue
  fi
  if [ "$phase" = "failed" ] || [ "$phase" = "waiting_claude" ] || [ $running -eq 0 ]; then
    [ $t -eq 0 ] && log "training stopped (phase=$phase, loop running=$running): shutdown in $((GRACE/60)) min unless repair starts"
    t=$((t+60)); [ $t -ge $GRACE ] && off "phase=$phase, loop running=$running, not repaired within $((GRACE/60)) min"
    continue
  fi
  t=0
  newest=$(find $R -type f -newermt '-150 minutes' 2>/dev/null | head -1)
  if [ -z "$newest" ]; then
    log "no file update for 150 min: stopping training and marking failed"
    pkill -f 'bash run_loop3.sh'; pkill -f 'python -u -m ma.loop3'; ray stop --force >/dev/null 2>&1
    pkill -9 -f 'verl.trainer.main_ppo'; pkill -9 -f 'sglang::'
    cd /root/autodl-tmp/r1zero && PYTHONPATH=. python3 -c "from ma.common import Board; b=Board(); b.set(phase='failed', final_status='failed', final_reason='stall: 150 min without file updates'); b.post('watchdog','stall',{})" 2>/dev/null
  fi
done
