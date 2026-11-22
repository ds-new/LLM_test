#!/usr/bin/env bash
# 用 E 的模型测 F 训练题的通过率（4 张卡各一个 SGLang），然后按 D 的规则选边界题。
# 用法：bash f/run_profile_f.sh MODEL OUT [k] [max_new]
M=$1; O=$2; K=$3; MAXNEW=$4
[ -z "$K" ] && K=4
[ -z "$MAXNEW" ] && MAXNEW=16384
R=/root/autodl-tmp/r1zero
export PYTHONPATH=$R:$PYTHONPATH OMP_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false
mkdir -p $O
for i in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$i python $R/f/profile_f.py --model $M --out $O --shard $i --nshards 4 --k $K --max-new $MAXNEW > $O/prof_$i.log 2>&1 &
done
wait
for i in 0 1 2 3; do
  [ -s $O/prof_$i.jsonl ] || { echo "shard $i missing"; exit 2; }
done
python $R/f/select_frontier_f.py $O
