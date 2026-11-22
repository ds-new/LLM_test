#!/usr/bin/env bash
# E 阶段总控（续跑版）。13:09 由 stage_watch 启动的 sft/start_e.sh 开头写了 set -f（关掉了通配符），
# 导致"删 D 的最新存盘""找冒烟存盘""找正式训练存盘"几处路径都不展开（删除没有执行，后面会找不到存盘而失败）。
# 13:14 停掉了原总控（它启动的冒烟测试照常在跑），由本脚本接着做：
#   等冒烟测试结束 -> 判断结果（失败就换 fsdp 版再冒烟）-> 用冒烟存盘试跑评测 -> 正式 SFT -> 完整评测 -> 写结论 -> 关机。
# D 的最终模型 global_step_156 按正式结论（显著提升）保留，不再删除。
R=/root/autodl-tmp/r1zero
cd $R
LOG=$R/run/logs/e_stage.log
log() { echo "$(date '+%F %T') $*" >> $LOG; }
fail() { log "FAILED: $*"; echo "failed: $*" > $R/sft/E_STATUS; rm -f $R/run/KEEP_RUNNING; exit 1; }
touch $R/run/KEEP_RUNNING
echo running > $R/sft/E_STATUS
log "E resume orchestrator start (pid $$); 重新冒烟（13:15 修 NCCL P2P；13:21 关融合 CE 核；13:28 修存盘内容 bug，只存 bf16 HF 权重）"
TR=engine
if ! bash $R/sft/run_sft.sh smoke engine; then
  log "engine smoke failed: $(tail -c 3000 $R/run/logs/sft_smoke_latest.log | tr '\n' ' ' | tail -c 800)"
  TR=fsdp
  bash $R/sft/run_sft.sh smoke fsdp || fail "both SFT smoke tests failed (run/logs/sft_smoke_*.log)"
fi
log "SFT smoke ok with $TR trainer"
SM=$(ls -d /root/autodl-tmp/sft_ckpt/smoke_$TR/global_step_*/huggingface 2>/dev/null | tail -1)
if [ -n "$SM" ]; then
  log "smoke HF checkpoint: $SM ($(du -sh $SM | cut -f1))"
  if bash $R/sft/run_eval.sh $SM $R/sft/eval_smoke 4 2048; then log "eval smoke ok"; else log "eval smoke FAILED (sft/eval_smoke/*.log) - SFT continues, fix eval meanwhile"; fi
else
  log "no smoke HF checkpoint found - eval smoke skipped"
fi
rm -rf /root/autodl-tmp/sft_ckpt/smoke_engine /root/autodl-tmp/sft_ckpt/smoke_fsdp
df -h /root/autodl-tmp | tail -1 >> $LOG
log "full SFT start ($TR)"
bash $R/sft/run_sft.sh full $TR || fail "full SFT failed (run/logs/sft_full_latest.log)"
HF=$(ls -d /root/autodl-tmp/sft_ckpt/full_$TR/global_step_*/huggingface 2>/dev/null | sort -V | tail -1)
[ -n "$HF" ] || fail "no HF checkpoint after full SFT"
log "full SFT done: $HF"
python - "$HF" >> $LOG 2>&1 <<'EOF'
import json, os, shutil, sys
d = sys.argv[1]
src = '/root/autodl-tmp/models/Qwen3-8B-Base'
for f in ['tokenizer.json', 'tokenizer_config.json', 'vocab.json', 'merges.txt']:
    if not os.path.exists(os.path.join(d, f)):
        shutil.copy(os.path.join(src, f), d)
p = os.path.join(d, 'generation_config.json')
g = json.load(open(p)) if os.path.exists(p) else {}
g.update(bos_token_id=151643, eos_token_id=[151645, 151643], pad_token_id=151643, do_sample=True, temperature=0.6, top_p=0.95)
json.dump(g, open(p, 'w'), indent=2)
print('generation_config set', g)
EOF
bash $R/sft/run_eval.sh $HF $R/sft/eval_full 0 16384 || fail "full eval failed (sft/eval_full/*.log)"
{ echo "# E 阶段（蒸馏 SFT）结论"; echo; echo "自动生成于 $(date '+%F %T')（sft/start_e_resume.sh）。模型：$HF"; echo; cat $R/sft/eval_full/summary.md; } > $R/run/audit/STAGE_E_RESULT.md
echo done > $R/sft/E_STATUS
log "E done: run/audit/STAGE_E_RESULT.md; shutting down"
rm -f $R/run/KEEP_RUNNING
sync
/usr/bin/shutdown
