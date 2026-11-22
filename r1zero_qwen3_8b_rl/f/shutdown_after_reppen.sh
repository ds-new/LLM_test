#!/usr/bin/env bash
# 2026-11-22 10:40 用户要求："跑完后机器立刻关机"。
# 等 F 第一阶段完整评测（f/resume_f6.sh）和 presence_penalty 试验（f/after_f1_reppen.sh）都结束后立刻关机，不再等看门狗的 30 分钟。
# 例外（交给看门狗的 30 分钟规则，留出接手修复的时间）：F 完整评测失败、没有 run/audit/STAGE_F1_RESULT.md；或 run/REPAIRING 存在（有人在修）。
# DRY=1：只写日志不关机（测试用）。
R=/root/autodl-tmp/r1zero
FLOG=$R/run/logs/f_stage.log
[ -z "$DRY" ] && DRY=0
log() { echo "$(date '+%F %T') [shutdown_after] $*" >> $FLOG; }
busy() { pgrep -f "f/resume_f6.sh" >/dev/null || pgrep -f "f/after_f1_reppen.sh" >/dev/null || pgrep -f "f/eval_reppen.py" >/dev/null || pgrep -f "sft/eval_e.py" >/dev/null; }
log "waiting for f/resume_f6.sh and f/after_f1_reppen.sh to finish, then shut down at once (pid $$, DRY=$DRY)"
while busy; do sleep 30; done
sleep 20
while busy; do sleep 30; done
if [ ! -s $R/run/audit/STAGE_F1_RESULT.md ]; then
  log "no run/audit/STAGE_F1_RESULT.md (F eval failed or not run): not shutting down here; watchdog 30-min rule applies"
  exit 0
fi
if [ -f $R/run/REPAIRING ]; then
  log "run/REPAIRING present: not shutting down here"
  exit 0
fi
if [ "$DRY" = 1 ]; then log "DRY: would shut down now"; exit 0; fi
log "F stage1 eval and presence_penalty test finished: shutting down now"
sync
/usr/bin/shutdown
