#!/usr/bin/env bash
# 等第 N 步（含）之后的某次存盘完成、评选打完分、最佳复制结束，再重启第二轮以载入新配置（提速改动）。
# 用法：setsid nohup bash restart_after.sh 52 &
R=/root/autodl-tmp/r1zero/run; N=$1; LOG=$R/logs/restart.log
echo "$(date '+%F %T') waiting for step >= $N saved+scored" >> $LOG
while true; do
  L=$(cat $R/ckpt/r1zero_qwen3_8b_dapo/latest_checkpointed_iteration.txt 2>/dev/null || echo -1)
  if [ "$L" -ge "$N" ] && python3 -c "import json,sys;s=json.load(open('$R/state.json'));sys.exit(0 if $L in s.get('eval_seen',[]) else 1)" \
     && [ -z "$(ls -A /root/r1zero_best 2>/dev/null | grep '^.tmp_')" ]; then
    sleep 60   # 留时间给最佳 checkpoint 的复制线程收尾
    [ -z "$(ls -A /root/r1zero_best 2>/dev/null | grep '^.tmp_')" ] && break
  fi
  sleep 30
done
touch $R/REPAIRING
echo "$(date '+%F %T') restarting at latest=$L to load new config" >> $LOG
pkill -f "bash run_loop2.sh"; pkill -f "python -u -m ma.loop2"; sleep 3
pkill -f "verl.trainer.main_ppo"; ray stop --force >/dev/null 2>&1; pkill -9 -f "sglang::"; sleep 15
cd /root/autodl-tmp/r1zero && PYTHONPATH=. python3 -c "from ma.common import Board; Board().post('decision','restart_for_speedup',{'at_step': $L, 'why': '载入新配置（见 ma/framework_agent.py 的 LEVELS 注释）；只改执行效率'})"
setsid nohup bash run_loop2.sh > /dev/null 2>&1 < /dev/null &
sleep 90
pgrep -f "python -u -m ma.loop2" >/dev/null && rm -f $R/REPAIRING
echo "$(date '+%F %T') restarted; loop2 running=$(pgrep -f 'python -u -m ma.loop2' >/dev/null && echo yes || echo no)" >> $LOG
