#!/usr/bin/env bash
# 2026-11-20 13:1x：冒烟第 3 次（12:48，带 ma/common.py 的整套 ENV）已经跑完生成、训练和 HF 存盘（16 GB，13:07:22），
# 只在最后写指标文件时崩（verl FileLogger 遇到 Tensor 不能 json.dumps，已打补丁 run/patch_filelogger.py 并自测）。
# 冒烟的目的（16K 下显存、判分、HF 存盘）已经验证，不再重跑，直接从第一阶段开始；内容和 start_ef.sh 第 6-7 步相同。
R=/root/autodl-tmp/r1zero
cd $R
export PYTHONUNBUFFERED=1 NCCL_P2P_DISABLE=1 VERL_DISABLE_EXPANDABLE_SEGMENTS=1 MEGATRON_CI_DISABLE_EXPANDABLE_SEGMENTS=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False OMP_NUM_THREADS=8 CUDA_DEVICE_MAX_CONNECTIONS=1 HYDRA_FULL_ERROR=1 RAY_DEDUP_LOGS=0 HF_ENDPOINT=https://hf-mirror.com TOKENIZERS_PARALLELISM=false PYTHONPATH=/root/autodl-tmp/r1zero:$PYTHONPATH
FLOG=$R/run/logs/f_stage.log
HF=/root/autodl-tmp/sft_ckpt/full_engine/global_step_974/huggingface
flog() { echo "$(date '+%F %T') $*" >> $FLOG; }
fail() { flog "FAILED: $*"; echo "failed: $*" > $R/f/F_STATUS; rm -f $R/run/KEEP_RUNNING; exit 1; }
touch $R/run/KEEP_RUNNING
flog "F resume 4 (f/resume_f4.sh pid $$): smoke validated (gen 12:51-13:00, train+save to 13:07, HF 16G), FileLogger patched; starting stage1"
rm -rf $R/run/ckpt/f_smoke

# ---- 6. 第一阶段 20 步 ----
echo "stage1" > $R/f/F_STATUS
bash $R/f/run_f.sh stage1 $HF
rc=$?
[ $rc -eq 0 ] || [ $rc -eq 3 ] || fail "F stage1 failed rc=$rc (run/logs/f_stage1_latest.log)"
FHF=$(ls -d $R/run/ckpt/f_sftrl/global_step_*/actor/huggingface 2>/dev/null | sort -V | tail -1)
[ -n "$FHF" ] || fail "no F checkpoint"
STEP=$(echo $FHF | grep -o "global_step_[0-9]*" | cut -d_ -f3)
flog "stage1 end (rc=$rc): $FHF"

# ---- 7. F 完整评测、和 E 比较 ----
echo "eval" > $R/f/F_STATUS
bash $R/sft/run_eval.sh $FHF $R/f/eval_s$STEP 0 16384 || fail "F eval failed (f/eval_s$STEP/*.log)"
python $R/f/compare_f.py $R/sft/eval_full $R/f/eval_s$STEP "F 第一阶段（step $STEP）" $R/run/audit/STAGE_F1_RESULT.md >> $FLOG 2>&1 || fail "compare failed"
echo "stage1 done: run/audit/STAGE_F1_RESULT.md" > $R/f/F_STATUS
flog "F stage1 done (run/audit/STAGE_F1_RESULT.md); waiting for Claude: KEEP_RUNNING removed, watchdog shuts down in 30 min unless continued"
rm -f $R/run/KEEP_RUNNING
