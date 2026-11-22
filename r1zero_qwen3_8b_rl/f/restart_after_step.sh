#!/usr/bin/env bash
# 2026-11-20 17:14 用户同意（"按照建议进行"）：第 2 步生成时实测显存，余量够就在第 2 步存盘后重启，把 SGLang 显存比例从 0.55 提到 0.60。
# 17:2x 加：第 1 步 48 题只有 29 组可用要再生成一整批，重启时生成批一并改成 64；显存检查不通过也照样重启（只改生成批，保持 0.55）。
# 用法：setsid nohup bash f/restart_after_step.sh 2 0 > run/logs/restart_after_step.out 2>&1 < /dev/null &
#   参数 1 = 在第几步存盘后重启（默认 2）；参数 2 = 1 时只演练（不停进程，日志写 /tmp）。
# 流程：每 20 秒采一次显存（f/patches/mem_sample.py -> run/logs/mem_samples.jsonl）；第 STEP-1 步指标出现时记下第 STEP 步生成开始
#   -> 等第 STEP 步指标（verl 先存盘再写指标）-> 再等 75 秒让停训监控先看一遍 -> 停训规则触发 / F_STATUS 不是 stage1 / 存盘不完整都不重启
#   -> f/patches/mem_decide.py 用第 STEP 步生成阶段的显存峰值算：提到 0.60 后余量 >= 4 GiB 才重启，否则保持 0.55
#   -> 先停 resume_f5.sh，再停 run_f.sh、command、main_dapo、ray、sglang -> 等显存释放 -> 启动 f/resume_f6.sh（F_GMU=0.60，从最新存盘续训；
#      再 OOM 时 run_f.sh 升档，make_command_f.py 在 raw tier >= 1 时不用 F_GMU，回到 0.55）。
R=/root/autodl-tmp/r1zero
cd $R
STEP=$1; [ -z "$STEP" ] && STEP=2
DRY=$2; [ -z "$DRY" ] && DRY=0
PREV=$((STEP - 1))
CK=$R/run/ckpt/f_sftrl
FLOG=$R/run/logs/f_stage.log
[ "$DRY" = 1 ] && FLOG=/tmp/restart_after_step_dry.log
flog() { echo "$(date '+%F %T') [Claude restart_after_step] $*" >> $FLOG; }
[ "$DRY" = 1 ] || echo $$ > $R/run/restart_after_step.pid
flog "watcher started (pid $$, restart after step $STEP if memory allows, dry=$DRY)"
GEN_START=0
if python $R/f/patches/has_step.py $PREV > /dev/null 2>&1; then GEN_START=$(date +%s); flog "step $PREV already done; memory samples from now count for step $STEP generation"; fi
DEADLINE=$(( $(date +%s) + 12 * 3600 ))
while true; do
  if [ $(date +%s) -gt $DEADLINE ]; then flog "no step $STEP within 12 h; exit without restart"; exit 2; fi
  st=$(cat $R/f/F_STATUS 2>/dev/null)
  if [ "$st" != "stage1" ]; then flog "F_STATUS is '$st', not stage1; exit without restart"; exit 2; fi
  python $R/f/patches/mem_sample.py 2>/dev/null
  if [ "$GEN_START" = 0 ] && python $R/f/patches/has_step.py $PREV > /dev/null 2>&1; then GEN_START=$(date +%s); flog "step $PREV metrics seen; memory samples from now count for step $STEP generation"; fi
  if python $R/f/patches/has_step.py $STEP > /dev/null 2>&1; then break; fi
  if [ "$DRY" = 1 ]; then flog "DRY: step $STEP not in metrics yet; going through the rest anyway"; break; fi
  sleep 20
done
GEN_END=$(date +%s)
flog "step $STEP metrics seen (or dry); waiting 75 s so the stop-rule monitor checks it first"
[ "$DRY" = 1 ] || sleep 75
if [ -f $R/run/F_STOPPED ]; then flog "stop rule fired ($(cat $R/run/F_STOPPED)); no restart"; exit 2; fi
if [ "$(cat $R/f/F_STATUS 2>/dev/null)" != "stage1" ]; then flog "F_STATUS changed to '$(cat $R/f/F_STATUS)'; no restart"; exit 2; fi
T=$(cat $CK/latest_checkpointed_iteration.txt 2>/dev/null)
if [ "$T" = "$STEP" ] && python $R/f/patches/ckpt_ok.py $CK/global_step_$STEP/actor/huggingface >> $FLOG 2>&1; then
  flog "checkpoint step $STEP complete (tracker $T); dataloader state: $(ls -la $CK/global_step_$STEP/data.pt 2>&1 | cut -c1-100)"
else
  flog "checkpoint step $STEP not complete (tracker '$T'); no restart"
  [ "$DRY" = 1 ] || exit 2
fi
if python $R/f/patches/mem_decide.py $GEN_START $GEN_END >> $FLOG 2>&1; then
  GMU_ENV="F_GMU=0.60"; flog "memory check passed: restart with gen_batch_size 64 and SGLang gpu_memory_utilization 0.60"
else
  GMU_ENV=""; flog "memory check failed: restart with gen_batch_size 64 only, SGLang stays at 0.55"
fi
P5=$(pgrep -f "f/resume_f5.sh" | head -1)
flog "stopping: resume_f5 '$P5'; run_f.sh '$(pgrep -f f/run_f.sh | tr '\n' ' ')'; main_dapo '$(pgrep -f recipe.dapo.main_dapo | tr '\n' ' ')'"
if [ "$DRY" = 1 ]; then flog "DRY: would kill resume_f5, run_f.sh, command_f_stage1.sh, main_dapo, ray stop --force, pkill sglang, wait for free GPUs, start f/resume_f6.sh"; exit 0; fi
[ -n "$P5" ] && kill $P5
sleep 3
pkill -f "f/run_f.sh"
sleep 2
pkill -f "run/command_f_stage1.sh"
sleep 1
pkill -f recipe.dapo.main_dapo
sleep 15
ray stop --force > /dev/null 2>&1
pkill -f sglang
sleep 10
for t in $(seq 1 60); do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1)
  [ "$used" -lt 2000 ] && break
  sleep 10
done
used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1)
if [ "$used" -ge 2000 ]; then
  flog "GPUs still busy after stop; kill -9 GPU processes: $(nvidia-smi --query-compute-apps=pid --format=csv,noheader | tr '\n' ' ')"
  nvidia-smi --query-compute-apps=pid --format=csv,noheader | xargs -r kill -9
  sleep 30
fi
flog "after stop: GPU MiB $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | tr '\n' ' '); left: $(pgrep -f 'resume_f5|f/run_f.sh|main_dapo' | wc -l) processes"
touch $R/run/KEEP_RUNNING
env $GMU_ENV setsid nohup bash $R/f/resume_f6.sh > $R/run/logs/resume_f6.out 2>&1 < /dev/null &
sleep 5
flog "started f/resume_f6.sh (pid $(pgrep -f f/resume_f6.sh | head -1)); stage1 resumes from step $STEP with gen_batch_size 64, [$GMU_ENV]"
for t in $(seq 1 40); do
  sleep 30
  L=$(ls -t $R/run/logs/f_stage1_*.log | head -1)
  if grep -a -q "Decode batch" $L 2>/dev/null; then flog "generation running after restart ($L): $(grep -a -o 'mem_fraction_static=[0-9.]*' $L | sort -u | tr '\n' ' ')"; break; fi
done
flog "watcher done"
