#!/usr/bin/env bash
# 2026-11-20 15:25 用户同意：第一阶段第 5 步存盘后重启一次，换上"每道题的 16 条回答拆成每卡 4 条"的生成补丁。
# 用法：setsid nohup bash f/restart_step5.sh 5 0 > run/logs/restart_step5.out 2>&1 < /dev/null &
#   参数 1 = 第几步之后重启（默认 5）；参数 2 = 1 时只演练（不改文件、不停进程，日志写 /tmp）。
# 流程：等指标文件出现第 5 步（verl 先存盘、再写指标）→ 再等 75 秒，让停训监控至少看一次 → 停训规则触发了就不重启
#   → 确认存盘完整（tracker = 5、HF 分片齐全、约 16 GB）→ 生成并换上补丁版 dapo_ray_trainer.py（原文件 .orig_interleave）
#   → 按顺序停：resume_f4.sh（先停它，免得它的 fail() 删 KEEP_RUNNING）→ run_f.sh → main_dapo → ray stop → sglang
#   → 等显存释放 → 启动 f/resume_f5.sh（run_f.sh stage1 从第 5 步存盘续训，重试次数重新算；之后评测、比较）。
R=/root/autodl-tmp/r1zero
cd $R
TARGET=$1; [ -z "$TARGET" ] && TARGET=5
DRY=$2; [ -z "$DRY" ] && DRY=0
CK=$R/run/ckpt/f_sftrl
LIVE=$R/third_party/verl_v061/recipe/dapo/dapo_ray_trainer.py
PATCHED=$R/f/patches/dapo_ray_trainer.interleave.py
FLOG=$R/run/logs/f_stage.log
[ "$DRY" = 1 ] && FLOG=/tmp/restart_step5_dry.log
flog() { echo "$(date '+%F %T') [Claude restart_step5] $*" >> $FLOG; }
echo $$ > $R/run/restart_step5.pid
flog "watcher started (pid $$, after step $TARGET, dry=$DRY)"
DEADLINE=$(( $(date +%s) + 20 * 3600 ))
while true; do
  if [ $(date +%s) -gt $DEADLINE ]; then flog "no step $TARGET within 20 h; exit without restart"; exit 2; fi
  st=$(cat $R/f/F_STATUS 2>/dev/null)
  if [ "$st" != "stage1" ]; then flog "F_STATUS is '$st', not stage1; exit without restart"; exit 2; fi
  if python $R/f/patches/has_step.py $TARGET > /dev/null 2>&1; then break; fi
  if [ "$DRY" = 1 ]; then flog "DRY: step $TARGET not in metrics yet"; exit 0; fi
  sleep 20
done
flog "step $TARGET metrics seen; waiting 75 s so the stop-rule monitor checks it first"
[ "$DRY" = 1 ] || sleep 75
if [ -f $R/run/F_STOPPED ]; then flog "stop rule fired ($(cat $R/run/F_STOPPED)); no restart"; exit 2; fi
if [ "$(cat $R/f/F_STATUS 2>/dev/null)" != "stage1" ]; then flog "F_STATUS changed to '$(cat $R/f/F_STATUS)'; no restart"; exit 2; fi
T=$(cat $CK/latest_checkpointed_iteration.txt 2>/dev/null)
if [ "$T" = "$TARGET" ] && python $R/f/patches/ckpt_ok.py $CK/global_step_$TARGET/actor/huggingface >> $FLOG 2>&1; then
  flog "checkpoint step $TARGET complete (tracker $T)"
  flog "dataloader state: $(ls -la $CK/global_step_$TARGET/data.pt 2>&1 | cut -c1-120)"
else
  flog "checkpoint step $TARGET not complete (tracker '$T'); no restart"
  [ "$DRY" = 1 ] || exit 2
fi
if grep -q "r1zero patch (11/20): spread" $LIVE; then
  flog "live trainer already has the interleave patch"
elif [ "$DRY" = 1 ]; then
  flog "DRY: would regenerate $PATCHED from $LIVE, back up $LIVE.orig_interleave and copy the patched file over"
else
  python $R/f/patches/make_interleave_patch.py >> $FLOG 2>&1 || { flog "make_interleave_patch failed; no restart"; exit 2; }
  cp -p $LIVE $LIVE.orig_interleave && cp $PATCHED $LIVE || { flog "swap failed; no restart"; exit 2; }
  if ! python -c "import ast, sys; ast.parse(open(sys.argv[1]).read())" $LIVE; then
    cp -p $LIVE.orig_interleave $LIVE; flog "patched trainer does not parse; original restored; no restart"; exit 2
  fi
  flog "trainer swapped to the interleave version (backup $LIVE.orig_interleave)"
fi
P4=$(pgrep -f "f/resume_f4.sh" | head -1)
flog "stopping: resume_f4 '$P4'; run_f.sh '$(pgrep -f f/run_f.sh | tr '\n' ' ')'; main_dapo '$(pgrep -f recipe.dapo.main_dapo | tr '\n' ' ')'"
if [ "$DRY" = 1 ]; then flog "DRY: would kill resume_f4, run_f.sh, main_dapo, ray stop --force, pkill sglang, wait for free GPUs, start f/resume_f5.sh"; exit 0; fi
[ -n "$P4" ] && kill $P4
sleep 3
pkill -f "f/run_f.sh"
sleep 3
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
flog "after stop: GPU MiB $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | tr '\n' ' '); left: $(pgrep -f 'resume_f4|f/run_f.sh|main_dapo' | wc -l) processes"
touch $R/run/KEEP_RUNNING
setsid nohup bash $R/f/resume_f5.sh > $R/run/logs/resume_f5.out 2>&1 < /dev/null &
sleep 5
flog "started f/resume_f5.sh (pid $(pgrep -f f/resume_f5.sh | head -1)); stage1 resumes from step $TARGET with balanced rollout"
# interleave safety net: if the resumed run dies with a traceback from the patch before the next step finishes,
# put the original trainer back so that run_f.sh retries with the old code.
NEXT=$((TARGET + 1))
for t in $(seq 1 300); do
  sleep 30
  if python $R/f/patches/has_step.py $NEXT > /dev/null 2>&1; then flog "step $NEXT finished with the interleave trainer"; break; fi
  L=$(ls -t $R/run/logs/f_stage1_*.log | head -1)
  if grep -a -q Traceback $L && grep -a -q "_r1z_\|in reorder" $L; then
    cp -p $LIVE.orig_interleave $LIVE
    flog "traceback involving the interleave patch in $L: original trainer restored, the run_f.sh retry will use it"
    break
  fi
done
flog "watcher done"
