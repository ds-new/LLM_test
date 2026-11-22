#!/usr/bin/env bash
# 2026-11-22 15:17 用户要求"进行 F 加惩罚的评测"：F 第一阶段 step 20 + presence_penalty 1.5 / 2.0（f/eval_pp_F.py，4,064 条）。
# 跑完后：结果写好 -> 等 GRACE_OK 秒（默认 900 秒，给 Claude 读结果）再关机；失败 -> 等 GRACE_FAIL 秒（默认 1800 秒，留出修复时间）再关机。
# run/KEEP_RUNNING 或 run/REPAIRING 存在时暂停计时、不关机（每分钟检查一次）。
R=/root/autodl-tmp/r1zero
M=$R/run/ckpt/f_sftrl/global_step_20/actor/huggingface
O=$R/f/eval_pp_F
FLOG=$R/run/logs/f_stage.log
[ -z "$GRACE_OK" ] && GRACE_OK=900
[ -z "$GRACE_FAIL" ] && GRACE_FAIL=1800
log() { echo "$(date '+%F %T') [pp_F] $*" >> $FLOG; }
cd $R
export PYTHONPATH=$R:$PYTHONPATH OMP_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false
mkdir -p $O
log "start: F step 20 + presence_penalty 1.5 / 2.0 (4,064 samples) -> $O (f/run_eval_pp_F.sh pid $$)"
for i in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$i timeout 10800 python $R/f/eval_pp_F.py gen --model $M --out $O --shard $i --nshards 4 > $O/gen_$i.log 2>&1 &
done
wait
ok=1
for i in 0 1 2 3; do [ -s $O/gen_$i.jsonl ] || ok=0; done
if [ $ok = 1 ]; then
  log "generation done; scoring"
  python $R/f/eval_pp_F.py score --out $O > $O/score.log 2>&1 || ok=0
fi
if [ $ok = 1 ]; then
  log "done: run/audit/F_PRESENCE_PENALTY_RESULT.md; shutting down in $((GRACE_OK/60)) min unless run/KEEP_RUNNING or run/REPAIRING"
  wait_s=$GRACE_OK
else
  log "FAILED (see $O/gen_*.log, $O/score.log); shutting down in $((GRACE_FAIL/60)) min unless run/KEEP_RUNNING or run/REPAIRING"
  wait_s=$GRACE_FAIL
fi
t=0
while [ $t -lt $wait_s ]; do
  sleep 60
  if [ -f $R/run/KEEP_RUNNING ] || [ -f $R/run/REPAIRING ]; then t=0; continue; fi
  t=$((t+60))
done
log "shutting down now"
sync
/usr/bin/shutdown
