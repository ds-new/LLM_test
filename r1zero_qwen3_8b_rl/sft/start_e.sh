#!/usr/bin/env bash
# E 阶段（蒸馏 SFT）总控。D 出结论后由 ma/stage_watch.py（--stage frontier --then sft）自动启动；
# 依据：用户 2026-11-19 09:03 指示"D 之后的蒸馏 SFT 不需要我同意直接开始，直接删除其中不需要的历史内容进行"。
# 手动启动：cd /root/autodl-tmp/r1zero && setsid nohup bash sft/start_e.sh > /dev/null 2>&1 < /dev/null &
# 流程：暂停看门狗关机 -> 清理 D 的进程、最新训练存盘（最佳存盘保留在 /root/r1zero_best）和 Ray 旧日志
#      -> SFT 冒烟测试 3 步（engine 版，不行换 fsdp 版）-> 用冒烟存盘试跑评测（每个评测集 4 条）
#      -> 正式 SFT（1 个 epoch）-> 完整评测 + 和 R1-Zero 配对比较 -> 写 run/audit/STAGE_E_RESULT.md -> 关机
# 出错：sft/E_STATUS 写 failed，删掉 run/KEEP_RUNNING，看门狗 30 分钟后关机（期间可以 touch run/REPAIRING 接手修复）。
set -f
R=/root/autodl-tmp/r1zero
cd $R
LOG=$R/run/logs/e_stage.log
log() { echo "$(date '+%F %T') $*" >> $LOG; }
fail() { log "FAILED: $*"; echo "failed: $*" > $R/sft/E_STATUS; rm -f $R/run/KEEP_RUNNING; exit 1; }
touch $R/run/KEEP_RUNNING
echo running > $R/sft/E_STATUS
log "E start (pid $$)"
ray stop --force >> $LOG 2>&1
sleep 10
pkill -9 -f "sglang::" ; pkill -9 -f "ray::" ; sleep 5
for i in $(seq 1 36); do
  busy=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1 > 2000' | wc -l)
  [ "$busy" = 0 ] && break
  sleep 5
done
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | tr '\n' ' ' >> $LOG; echo >> $LOG
if [ -f /root/r1zero_best/best.json ] && ls -d /root/r1zero_best/global_step_* > /dev/null 2>&1; then
  for d in $R/run/ckpt/r1zero_qwen3_8b_dapo/global_step_*; do
    [ -d "$d" ] && rm -rf "$d" && log "deleted D latest training ckpt $d (best kept: $(ls -d /root/r1zero_best/global_step_*))"
  done
fi
rm -rf /tmp/ray/session_* 2>/dev/null && log "deleted old Ray session logs in /tmp/ray"
df -h / /root/autodl-tmp | tr '\n' ' ' >> $LOG; echo >> $LOG
TR=engine
if ! bash $R/sft/run_sft.sh smoke engine; then
  log "engine smoke failed: $(tail -c 3000 $R/run/logs/sft_smoke_latest.log | tr '\n' ' ' | tail -c 800)"
  TR=fsdp
  bash $R/sft/run_sft.sh smoke fsdp || fail "both SFT smoke tests failed (run/logs/sft_smoke_*.log)"
fi
log "SFT smoke ok with $TR trainer"
SM=$(ls -d /root/autodl-tmp/sft_ckpt/smoke_$TR/global_step_*/huggingface 2>/dev/null | tail -1)
if [ -n "$SM" ]; then
  if bash $R/sft/run_eval.sh $SM $R/sft/eval_smoke 4 2048; then log "eval smoke ok"; else log "eval smoke FAILED (sft/eval_smoke/*.log) - SFT continues, fix eval meanwhile"; fi
else
  log "no smoke HF checkpoint found - eval smoke skipped"
fi
rm -rf /root/autodl-tmp/sft_ckpt/smoke_engine /root/autodl-tmp/sft_ckpt/smoke_fsdp
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
{ echo "# E 阶段（蒸馏 SFT）结论"; echo; echo "自动生成于 $(date '+%F %T')（sft/start_e.sh）。模型：$HF"; echo; cat $R/sft/eval_full/summary.md; } > $R/run/audit/STAGE_E_RESULT.md
echo done > $R/sft/E_STATUS
log "E done: run/audit/STAGE_E_RESULT.md; shutting down"
rm -f $R/run/KEEP_RUNNING
sync
/usr/bin/shutdown
