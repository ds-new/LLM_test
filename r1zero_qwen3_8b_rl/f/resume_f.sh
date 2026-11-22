#!/usr/bin/env bash
# 2026-11-20 07:3x：AutoDL 账户欠费、06:19 自动关机，07:25 用户充值开机后续跑 F。
# 从选题采样开始，内容和 f/start_ef.sh 的第 4-7 步完全一样（E 已完成、基座已删，不再重复）。
R=/root/autodl-tmp/r1zero
cd $R
FLOG=$R/run/logs/f_stage.log
HF=/root/autodl-tmp/sft_ckpt/full_engine/global_step_974/huggingface
flog() { echo "$(date '+%F %T') $*" >> $FLOG; }
fail() { flog "FAILED: $*"; echo "failed: $*" > $R/f/F_STATUS; rm -f $R/run/KEEP_RUNNING; exit 1; }
touch $R/run/KEEP_RUNNING
flog "F resume (f/resume_f.sh pid $$): instance was shut down at 06:19 (account balance ran out), restarted 07:25; restarting from profiling"
[ -s $HF/model.safetensors.index.json ] || fail "E model missing: $HF"
df -h /root/autodl-tmp / | tail -2 >> $FLOG
FREE=$(df --output=avail -B1G /root/autodl-tmp | tail -1 | tr -d ' ')
[ "$FREE" -ge 17 ] || fail "data disk has $FREE GB free; F needs >= 17 GB for one checkpoint"

# ---- 4. 测通过率、选边界题 ----
echo "profiling" > $R/f/F_STATUS
rm -f $R/f/profile/prof_*.jsonl $R/f/profile/prof_*.jsonl.tmp
[ -s $R/run/data/f_profile.parquet ] || python $R/f/build_f_data.py >> $FLOG 2>&1 || fail "build_f_data failed"
flog "profile start (2048 problems x 4 samples, temperature 1.0, max 16384)"
bash $R/f/run_profile_f.sh $HF $R/f/profile 4 16384 >> $FLOG 2>&1 || fail "profiling failed (f/profile/prof_*.log)"
NT=$(python -c "import json; print(json.load(open('$R/run/audit/f_frontier_stats.json'))['n_train'])")
flog "profile done: n_train=$NT"
[ "$NT" -ge 300 ] || fail "only $NT training problems selected (run/audit/f_frontier_stats.json)"

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
