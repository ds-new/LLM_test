#!/usr/bin/env bash
# R1-Zero 准备：下载模型与数据（无卡模式即可）
export HF_ENDPOINT=https://hf-mirror.com
D=/root/autodl-tmp/r1zero
mkdir -p $D/raw /root/autodl-tmp/models
log() { echo "$(date '+%F %T') $*" >> $D/download.log; }
log "start"
modelscope download --model Qwen/Qwen3-8B-Base --local_dir /root/autodl-tmp/models/Qwen3-8B-Base > $D/dl_model.log 2>&1; log "Qwen3-8B-Base rc=$?"
for ds in BytedTsinghua-SIA/DAPO-Math-17k xiaodongguaAIGC/X-R1-750 math-ai/aime25 math-ai/aime24 HuggingFaceH4/MATH-500 math-ai/amc23; do
  n=$(echo $ds | tr '/' '_')
  huggingface-cli download --repo-type dataset $ds --local-dir $D/raw/$n > $D/dl_$n.log 2>&1; log "$ds rc=$? $(du -sh $D/raw/$n 2>/dev/null | cut -f1)"
done
log DOWNLOAD_DONE
