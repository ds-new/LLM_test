#!/usr/bin/env bash
# 第三轮多智能体 loop：R1Z_STAGE=16k|dyn8k [R1Z_EXTRA_STEPS=100] setsid nohup bash run_loop3.sh &   （再启动 watchdog3.sh）
cd /root/autodl-tmp/r1zero || exit 1
export PYTHONPATH=/root/autodl-tmp/r1zero
python -u -m ma.loop3 >> run/logs/loop3_stdout.log 2>&1
echo "$(date '+%F %T') loop3 exited $?" >> run/logs/loop3_stdout.log
