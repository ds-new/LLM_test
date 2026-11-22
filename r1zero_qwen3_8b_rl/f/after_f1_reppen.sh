#!/usr/bin/env bash
# 2026-11-21 16:39 用户同意（"按照建议的进行"）：F 第一阶段的评测完成后，给 E 加 presence_penalty 再评一次 AIME（约 30 元，f/eval_reppen.py）。
# 这个脚本只等待、不碰训练：f/F_STATUS 变成 "stage1 done" 之后才开始；F 失败、编排脚本消失、或显卡没空出来，就什么都不做直接退出。
# 试验期间用 run/KEEP_RUNNING 保持开机（生成最多 MAXSEC 秒），结束时（成功或失败）删掉它，看门狗 30 分钟后关机。
# DRY=1：只测流程（不占显卡、不动 KEEP_RUNNING、不写正式结果），配合 STATUS_FILE 指向一个测试文件。
R=/root/autodl-tmp/r1zero
ST=${STATUS_FILE:-$R/f/F_STATUS}
FLOG=$R/run/logs/f_stage.log
M=/root/autodl-tmp/sft_ckpt/full_engine/global_step_974/huggingface
O=$R/f/eval_reppen_E
MAXSEC=${MAXSEC:-10800}
DRY=${DRY:-0}
[ "$DRY" = 1 ] && { FLOG=/tmp/reppen_dry.log; O=/tmp/reppen_dry; }
flog() { echo "$(date '+%F %T') [reppen] $*" >> $FLOG; }
cd $R
export PYTHONPATH=$R:$PYTHONPATH OMP_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false
flog "waiting for F stage1 to finish (f/after_f1_reppen.sh pid $$, status file $ST)"
while true; do
  s=$(cat $ST 2>/dev/null)
  case "$s" in
    "stage1 done"*) break ;;
    failed*) flog "F status is '$s': test not run"; exit 0 ;;
  esac
  if [ "$DRY" != 1 ] && ! pgrep -f "resume_f6.sh" > /dev/null; then
    sleep 20
    s=$(cat $ST 2>/dev/null)
    case "$s" in "stage1 done"*) break ;; esac
    flog "resume_f6.sh is gone and F status is '$s': test not run"
    exit 0
  fi
  sleep 30
done
sleep 10
if [ "$DRY" != 1 ]; then
  touch $R/run/KEEP_RUNNING
  trap 'rm -f /root/autodl-tmp/r1zero/run/KEEP_RUNNING' EXIT
  ok=0
  for i in $(seq 1 24); do
    mx=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1)
    [ "$mx" -lt 3000 ] && { ok=1; break; }
    sleep 15
  done
  [ $ok -eq 1 ] || { flog "GPUs still in use ($mx MiB): test not run; KEEP_RUNNING removed"; exit 0; }
fi
mkdir -p $O
flog "start: E + presence_penalty 0.5 / 1.5 on AIME24 / AIME25 / AIME 2026 (1,424 samples) -> $O"
EXTRA=""; [ "$DRY" = 1 ] && EXTRA="--dry"
for i in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$i timeout $MAXSEC python $R/f/eval_reppen.py gen --model $M --out $O --shard $i --nshards 4 $EXTRA > $O/gen_$i.log 2>&1 &
done
wait
if [ "$DRY" = 1 ]; then
  python $R/f/eval_reppen.py score --out $O --selftest > $O/score.log 2>&1 || { flog "FAILED: score"; exit 1; }
  flog "dry run done"; exit 0
fi
n=$(ls $O/gen_[0-3].jsonl 2>/dev/null | wc -l)
[ "$n" -eq 4 ] || { flog "FAILED: only $n/4 shards finished (see $O/gen_*.log); KEEP_RUNNING removed"; exit 1; }
python $R/f/eval_reppen.py score --out $O > $O/score.log 2>&1 || { flog "FAILED: score (see $O/score.log); KEEP_RUNNING removed"; exit 1; }
flog "done: run/audit/E_PRESENCE_PENALTY_RESULT.md; KEEP_RUNNING removed, watchdog shuts down in 30 min unless continued"
