#!/usr/bin/env bash
# E 阶段 SFT：bash sft/run_sft.sh smoke|full engine|fsdp
#   engine = verl 新版 SFT trainer（按 token 数动态拼批，效率高；融合 CE 核在 torch 后端会报 "has to return_dict"，所以关掉，每卡每批 16384 token）；fsdp = 老版 fsdp_sft_trainer（每卡每次 1 条，作备选）
# 配置（2026-11-19 定）：Qwen3-8B-Base 全参数，OpenR1-Math 62,848 条，1 个 epoch，全局批 64 条，
#   lr 2e-5、cosine、预热 3%、AdamW(0.9, 0.95)、权重衰减 0.01、梯度裁剪 1.0，最长 16384 token，只存 bf16 的 HF 权重。
set -f
MODE=$1; TR=$2
R=/root/autodl-tmp/r1zero
MODEL=/root/autodl-tmp/models/Qwen3-8B-Base
OUT=/root/autodl-tmp/sft_ckpt/$MODE"_"$TR
# 这台机器（A800 PCIe、容器）里 NCCL 的 P2P 会卡死（2026-11-19 13:15 测过：默认、NCCL_CUMEM_ENABLE=0、NCCL_IB_DISABLE=1 都卡；NCCL_P2P_DISABLE=1 正常）。
# RL 训练是 Ray 每个进程只看见一张卡，所以没碰到。
# 显存：不用融合 CE 核时 16k token 的 logits（fp32）要占二三十 GB，冒烟时单卡缓存峰值到 77.5 GiB；
# 动态拼批每批 token 数不同，大块显存反复申请容易碎片化，所以开 expandable_segments（2026-11-19 13:27 加）。
export PYTHONPATH=$R:$PYTHONPATH OMP_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false HF_HOME=/root/autodl-tmp/hf_home NCCL_DEBUG=WARN NCCL_P2P_DISABLE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
if [ "$MODE" = smoke ]; then
  EXTRA="trainer.total_training_steps=3 trainer.save_freq=100000 trainer.test_freq=100000 data.train_max_samples=640"
else
  EXTRA="trainer.save_freq=250 trainer.test_freq=250"
fi
OPT="optim.lr=2e-5 optim.betas=[0.9,0.95] optim.weight_decay=0.01 optim.lr_warmup_steps_ratio=0.03 optim.clip_grad=1.0"
TRN="trainer.default_local_dir=$OUT trainer.project_name=r1zero_e trainer.experiment_name=openr1_sft_$MODE trainer.total_epochs=1 trainer.logger=[console] trainer.max_ckpt_to_keep=1 trainer.resume_mode=disable trainer.seed=1"
DAT="data.train_files=$R/sft/data/train.parquet data.val_files=$R/sft/data/val.parquet data.train_batch_size=64 data.max_length=16384 data.truncation=right"
if [ "$TR" = engine ]; then
  ARGS="$DAT data.messages_key=messages data.micro_batch_size_per_gpu=2 data.use_dynamic_bsz=True data.max_token_len_per_gpu=16384 data.pad_mode=no_padding model.path=$MODEL model.enable_gradient_checkpointing=True model.use_remove_padding=True model.use_fused_kernels=False engine.strategy=fsdp2 engine.model_dtype=fp32 engine.dtype=bfloat16 engine.use_torch_compile=False $OPT optim.lr_scheduler_type=cosine optim.min_lr_ratio=0.0 checkpoint.save_contents=[hf_model] $TRN"
else
  ARGS="$DAT data.multiturn.enable=true data.multiturn.messages_key=messages data.micro_batch_size_per_gpu=1 model.partial_pretrain=$MODEL model.enable_gradient_checkpointing=True model.strategy=fsdp2 model.fsdp_config.model_dtype=fp32 use_remove_padding=True ulysses_sequence_parallel_size=1 $OPT optim.lr_scheduler=cosine trainer.checkpoint.save_contents=[hf_model] trainer.n_gpus_per_node=4 $TRN"
fi
[ "$DRY" = 1 ] && { echo $ARGS $EXTRA; exit 0; }
LOG=$R/run/logs/sft_$MODE"_"$TR"_"$(date +%m%d_%H%M).log
ln -sf $LOG $R/run/logs/sft_$MODE"_latest.log"
echo "$(date '+%F %T') start $MODE $TR" > $LOG
torchrun --standalone --nnodes=1 --nproc_per_node=4 $R/sft/run_engine_sft.py $TR $ARGS $EXTRA >> $LOG 2>&1
rc=$?
echo "$(date '+%F %T') rc=$rc" >> $LOG
exit $rc
