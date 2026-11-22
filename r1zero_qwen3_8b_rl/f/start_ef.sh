#!/usr/bin/env bash
# E 收尾 + F（在 E 的 SFT 模型上做 RL）总控。2026-11-19 17:50 起接替 sft/start_e_resume.sh（用户 17:16：接着做 E 和 F）。
#   等 E 正式训练结束（run_sft.sh 在日志末尾写 rc=）-> E 完整评测、写 STAGE_E_RESULT.md
#   -> 删基座权重腾磁盘（ModelScope 可重新下载，F 用不到；run/F_KEEP_BASE 存在则跳过）
#   -> 用 E 的模型对 2,048 道训练题各采样 4 次测通过率、按 D 的规则选边界题
#   -> F 冒烟（1 步、4 题 x 16 条、存一次盘）-> F 第一阶段 20 步（每 5 步存盘，只留最新一份）
#   -> F 完整评测（和 E 同一套 sft/run_eval.sh），和 E 同题配对比较 -> run/audit/STAGE_F1_RESULT.md
#   -> 删 KEEP_RUNNING，等 Claude 决定是否继续（看门狗 30 分钟后关机）
R=/root/autodl-tmp/r1zero
cd $R
LOG=$R/run/logs/e_stage.log
FLOG=$R/run/logs/f_stage.log
log() { echo "$(date '+%F %T') $*" >> $LOG; }
flog() { echo "$(date '+%F %T') $*" >> $FLOG; }
fail() { log "FAILED: $*"; flog "FAILED: $*"; echo "failed: $*" > $R/f/F_STATUS; rm -f $R/run/KEEP_RUNNING; exit 1; }
touch $R/run/KEEP_RUNNING
echo "waiting for E" > $R/f/F_STATUS
log "E+F orchestrator start (pid $$, f/start_ef.sh), replacing start_e_resume.sh; waiting for the full SFT run"
flog "E+F orchestrator start (pid $$)"

# ---- 1. 等 E 正式训练结束 ----
SL=$(readlink -f $R/run/logs/sft_full_latest.log)
while ! grep -a -q "rc=" $SL; do
  if ! pgrep -f "run_sft.sh full" > /dev/null; then
    sleep 30
    grep -a -q "rc=" $SL || { echo "failed: full SFT process gone" > $R/sft/E_STATUS; fail "full SFT process gone without rc line ($SL)"; }
  fi
  sleep 60
done
RC=$(grep -a -o "rc=[0-9]*" $SL | tail -1 | cut -d= -f2)
[ "$RC" = 0 ] || { echo "failed: full SFT rc=$RC" > $R/sft/E_STATUS; fail "full SFT failed rc=$RC ($SL)"; }
HF=$(ls -d /root/autodl-tmp/sft_ckpt/full_engine/global_step_*/huggingface 2>/dev/null | sort -V | tail -1)
[ -n "$HF" ] || fail "no HF checkpoint after full SFT"
log "full SFT done: $HF"
python - "$HF" >> $LOG 2>&1 <<'EOF'
import json, os, shutil, sys
d = sys.argv[1]
src = '/root/autodl-tmp/models/Qwen3-8B-Base'
for f in ['tokenizer.json', 'tokenizer_config.json', 'vocab.json', 'merges.txt']:
    if not os.path.exists(os.path.join(d, f)) and os.path.exists(os.path.join(src, f)):
        shutil.copy(os.path.join(src, f), d)
p = os.path.join(d, 'generation_config.json')
g = json.load(open(p)) if os.path.exists(p) else {}
g.update(bos_token_id=151643, eos_token_id=[151645, 151643], pad_token_id=151643, do_sample=True, temperature=0.6, top_p=0.95)
json.dump(g, open(p, 'w'), indent=2)
print('generation_config set', g)
EOF

# ---- 2. E 完整评测、结论 ----
bash $R/sft/run_eval.sh $HF $R/sft/eval_full 0 16384 || { echo "failed: full eval" > $R/sft/E_STATUS; fail "E full eval failed (sft/eval_full/*.log)"; }
{ echo "# E 阶段（蒸馏 SFT）结论"; echo; echo "自动生成于 $(date '+%F %T')（f/start_ef.sh）。模型：$HF"; echo; cat $R/sft/eval_full/summary.md; } > $R/run/audit/STAGE_E_RESULT.md
echo done > $R/sft/E_STATUS
log "E done: run/audit/STAGE_E_RESULT.md; continuing with F (f/start_ef.sh, log run/logs/f_stage.log)"
flog "E done; F starts from $HF"

# ---- 3. 腾磁盘 ----
if [ -f $R/run/F_KEEP_BASE ]; then
  flog "run/F_KEEP_BASE present: keep base model"
else
  rm -rf /root/autodl-tmp/models/Qwen3-8B-Base /root/r1zero_base_shards && flog "deleted base model weights (Qwen3-8B-Base, re-downloadable from ModelScope)"
fi
df -h /root/autodl-tmp / | tail -2 >> $FLOG
FREE=$(df --output=avail -B1G /root/autodl-tmp | tail -1 | tr -d ' ')
[ "$FREE" -ge 17 ] || fail "data disk has $FREE GB free; F needs >= 17 GB for one checkpoint"

# ---- 4. 测通过率、选边界题 ----
echo "profiling" > $R/f/F_STATUS
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
