#!/usr/bin/env bash
# 2026-11-20 12:4x：冒烟卡在 NCCL（P2P 卡死，12:41 用 run/nccl_test.py 复测：默认环境超时，NCCL_P2P_DISABLE=1 正常）。
# 从冒烟开始续跑，内容和 f/start_ef.sh 的第 5-7 步一样，只多了 export NCCL_P2P_DISABLE=1（选题采样 12:09 已完成）。
R=/root/autodl-tmp/r1zero
cd $R
# 和 R1-Zero 的多智能体 loop（ma/common.py 的 ENV）用同一套环境：P2P 卡死、容器内核不支持 pidfd_open（expandable_segments 下 CUDA IPC 权重同步失败）等
export PYTHONUNBUFFERED=1 NCCL_P2P_DISABLE=1 VERL_DISABLE_EXPANDABLE_SEGMENTS=1 MEGATRON_CI_DISABLE_EXPANDABLE_SEGMENTS=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False OMP_NUM_THREADS=8 CUDA_DEVICE_MAX_CONNECTIONS=1 HYDRA_FULL_ERROR=1 RAY_DEDUP_LOGS=0 HF_ENDPOINT=https://hf-mirror.com TOKENIZERS_PARALLELISM=false PYTHONPATH=/root/autodl-tmp/r1zero:$PYTHONPATH
FLOG=$R/run/logs/f_stage.log
HF=/root/autodl-tmp/sft_ckpt/full_engine/global_step_974/huggingface
flog() { echo "$(date '+%F %T') $*" >> $FLOG; }
fail() { flog "FAILED: $*"; echo "failed: $*" > $R/f/F_STATUS; rm -f $R/run/KEEP_RUNNING; exit 1; }
touch $R/run/KEEP_RUNNING
flog "F resume 3 (f/resume_f3.sh pid $$): restart from smoke with the ma/common.py training env"
[ -s $R/run/data/train_f.parquet ] && [ -s $R/run/data/f_smoke.parquet ] || fail "F data missing"
rm -rf $R/run/ckpt/f_smoke

# ---- 5. 冒烟 ----
echo "smoke" > $R/f/F_STATUS
bash $R/f/run_f.sh smoke $HF || fail "F smoke failed (run/logs/f_smoke_latest.log)"
flog "smoke ok: $(du -sh $R/run/ckpt/f_smoke/global_step_1/actor/huggingface | cut -f1); $(grep -a -o 'timing_s/step:[0-9.]*' $R/run/logs/f_smoke_latest.log | tail -1)"
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
