#!/usr/bin/env bash
# 第三轮：上一阶段完成（phase=waiting_claude 且 stage_done）后，启动下一阶段和该阶段的收尾脚本（2026-11-17）。
# 用法：bash start_next_stage.sh lr2|lr2dyn|frontier [步数，默认 20]
cd /root/autodl-tmp/r1zero || exit 1
STAGE=$1; STEPS=${2:-20}
[ -n "$STAGE" ] || { echo "usage: $0 <stage> [steps]"; exit 1; }
nvidia-smi -L 2>/dev/null | grep -q GPU || { echo "没有 GPU（无卡模式？），不启动"; exit 1; }
python3 - <<'EOF' || { echo "上一阶段没有正常结束（不是 stage_done），不启动"; exit 1; }
import json, sys
d = json.load(open('run/state.json')); w = d.get('waiting') or {}
print('phase', d.get('phase'), '| stage', d.get('stage'), '| stage_done', w.get('stage_done'))
sys.exit(0 if d.get('phase') == 'waiting_claude' and w.get('stage_done') else 1)
EOF
touch run/REPAIRING
[ -f run/WAITING_CLAUDE.json ] && mv run/WAITING_CLAUDE.json run/logs/WAITING_CLAUDE_$(date +%Y%m%d_%H%M)_before_$STAGE.json
pkill -f "ma/stage_watch.py"
(R1Z_STAGE=$STAGE R1Z_EXTRA_STEPS=$STEPS R1Z_AFTER=pause setsid nohup bash run_loop3.sh > /dev/null 2>&1 < /dev/null &)
pgrep -f "bash watchdog3.sh" > /dev/null || (setsid nohup bash watchdog3.sh > /dev/null 2>&1 < /dev/null &)
for i in $(seq 1 24); do  # 等 loop3 把 state.json 切到新阶段，再启动收尾脚本（否则它读到旧阶段名会直接退出）
  sleep 5
  python3 -c "import json,sys; sys.exit(0 if json.load(open('run/state.json')).get('stage') == '$STAGE' else 1)" && break
done
THEN=""; case $STAGE in lr2|lr2dyn) THEN="--then auto";; esac  # C 结束后自动接着做 D（用户 2026-11-17 同意）；D 结束后关机
(setsid nohup nice python ma/stage_watch.py --stage $STAGE $THEN >> run/logs/stage_watch.log 2>&1 < /dev/null &)
sleep 30
pgrep -f "python -u -m ma.loop3" > /dev/null && rm -f run/REPAIRING
echo "started $STAGE: loop3=$(pgrep -f 'python -u -m ma.loop3' > /dev/null && echo yes || echo no) stage_watch=$(pgrep -f 'ma/stage_watch.py' > /dev/null && echo yes || echo no) watchdog3=$(pgrep -f 'bash watchdog3.sh' > /dev/null && echo yes || echo no)"
