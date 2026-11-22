#!/usr/bin/env bash
# 2026-11-20 17:14 用户同意第 2 步实测显存后把 SGLang 显存比例提到 0.60；17:2x 第 1 步 48 题只有 29 组可用、要再生成一整批，生成批一并改成 64。
# 由 f/restart_after_step.sh 在第 2 步存盘后启动：显存检查通过时它在环境里设 F_GMU=0.60，否则不设（0.55）；run_f.sh 从最新存盘续训，再 OOM 回到 0.55。
R=/root/autodl-tmp/r1zero
cd $R
export PYTHONUNBUFFERED=1 NCCL_P2P_DISABLE=1 VERL_DISABLE_EXPANDABLE_SEGMENTS=1 MEGATRON_CI_DISABLE_EXPANDABLE_SEGMENTS=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False OMP_NUM_THREADS=8 CUDA_DEVICE_MAX_CONNECTIONS=1 HYDRA_FULL_ERROR=1 RAY_DEDUP_LOGS=0 HF_ENDPOINT=https://hf-mirror.com TOKENIZERS_PARALLELISM=false PYTHONPATH=/root/autodl-tmp/r1zero:$PYTHONPATH
export F_MIN_TIER=1
export F_GEN_BATCH=64
FLOG=$R/run/logs/f_stage.log
HF=/root/autodl-tmp/sft_ckpt/full_engine/global_step_974/huggingface
flog() { echo "$(date '+%F %T') $*" >> $FLOG; }
fail() { flog "FAILED: $*"; echo "failed: $*" > $R/f/F_STATUS; rm -f $R/run/KEEP_RUNNING; exit 1; }
touch $R/run/KEEP_RUNNING
flog "F resume 6 (f/resume_f6.sh pid $$): after the step-2 checkpoint; gen_batch_size $F_GEN_BATCH, SGLang gpu_memory_utilization override [$F_GMU] (empty = tier default 0.55; back to 0.55 on OOM); balanced rollout, min memory tier 1, save every 2 steps, chunked_prefill_size 2048; stage1 resumes from the latest checkpoint"

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
