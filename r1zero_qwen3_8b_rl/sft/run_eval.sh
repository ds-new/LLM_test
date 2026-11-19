#!/usr/bin/env bash
# E 阶段评测：bash sft/run_eval.sh MODEL_DIR OUT_DIR [每个评测集只取前 N 题(0=全部)] [最长生成 token]
# 4 张卡各起一个 SGLang（TP=1），把 2,564 条评测请求分成 4 份并行生成，然后统一打分、和 R1-Zero 做配对比较。
set -f
M=$1; O=$2; LIM=$3; MAXNEW=$4
[ -z "$LIM" ] && LIM=0
[ -z "$MAXNEW" ] && MAXNEW=16384
R=/root/autodl-tmp/r1zero
export PYTHONPATH=$R:$PYTHONPATH OMP_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false
mkdir -p $O
for i in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$i python $R/sft/eval_e.py gen --model $M --out $O --shard $i --nshards 4 --limit $LIM --max-new $MAXNEW > $O/gen_$i.log 2>&1 &
done
wait
for i in 0 1 2 3; do
  [ -s $O/gen_$i.jsonl ] || { echo "shard $i missing"; exit 2; }
done
python $R/sft/eval_e.py score --out $O > $O/score.log 2>&1
