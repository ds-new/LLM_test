#!/bin/bash
###############################################################################
# Qwen3-VL-4B GRPO 单次训练（不经过多智能体 loop 时手动使用）
# 框架：verl + FSDP（训练）+ SGLang（rollout）；硬件：AutoDL NVIDIA GPU，单机
#
#   bash run_grpo_qwen3vl_4b_fsdp_sglang_gpu.sh
#   MODEL_PATH=... DATA_DIR=... N_SAMPLES=8 bash run_grpo_qwen3vl_4b_fsdp_sglang_gpu.sh
#
# 相对旧版的修改：
#   - 所有路径 / 超参都可用环境变量覆盖（旧版 MODEL_PATH、DATA_DIR 写死，README 中的 export 不生效）
#   - GPU 数量自动检测（旧版默认 8 卡、GEN_TP=2，在 AutoDL 单卡 / 双卡实例上直接报错）
#   - 验证用 val 文件，test 只留给最终评估（旧版两者是同一份数据）
#   - 键名不再用 ++ 前缀：拼错或当前 verl 不支持的键会直接报错，而不是静默不生效
#   - free_cache_engine=True：训练阶段释放 SGLang KV cache（旧版 False，显存紧张时容易 OOM）
#   - 动态 batch（按 token 数打包）、验证用贪心解码、限制保留的检查点数量（每个约 50GB）
#   - 每题采样 8 次（旧版 5），save_freq == test_freq（每个检查点都有验证分数）
# 需要按显存自动选择 offload、OOM 自动降级、提前停止、多轮算法对比时，请用多智能体 loop：
#   nohup bash multiagent/run_loop.sh > /dev/null 2>&1 &
###############################################################################
set -xeo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
[ -f /etc/network_turbo ] && source /etc/network_turbo >/dev/null 2>&1 || true

# ------------------------------- 路径 -------------------------------
MODEL_PATH="${MODEL_PATH:-/root/autodl-tmp/qwen3_vl_4}"
DATA_DIR="${DATA_DIR:-/root/autodl-tmp/data/geo3k}"           # preprocess_geo3k.py 的输出目录
SAVE_DIR="${SAVE_DIR:-/root/autodl-tmp/checkpoints/qwen3vl_4b_geo3k_grpo}"
REWARD_FN_PATH="${SCRIPT_DIR}/reward_fn_geo3k.py"

# ------------------------------- GPU -------------------------------
if [ -z "${NUM_GPU}" ]; then
    if [ -n "${CUDA_VISIBLE_DEVICES}" ]; then
        NUM_GPU=$(echo "${CUDA_VISIBLE_DEVICES}" | tr ',' '\n' | grep -c .)
    else
        NUM_GPU=$(nvidia-smi -L | wc -l)
    fi
fi
GPU_MEM_GB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1 | awk '{print int($1/1024)}')

# ------------------------------- 超参 -------------------------------
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-64}
PPO_MINI_BATCH=${PPO_MINI_BATCH:-32}
N_SAMPLES=${N_SAMPLES:-8}
MAX_PROMPT_LEN=${MAX_PROMPT_LEN:-1024}      # geometry3k 图较小，实测 prompt 远低于 1024（多智能体 loop 会实测）
MAX_RESPONSE_LEN=${MAX_RESPONSE_LEN:-2048}
LR=${LR:-1e-6}
KL_LOSS_COEF=${KL_LOSS_COEF:-0.01}
USE_KL_LOSS=${USE_KL_LOSS:-True}
CLIP_HIGH=${CLIP_HIGH:-0.2}                 # 0.28 即 DAPO clip-higher
TOTAL_EPOCHS=${TOTAL_EPOCHS:-12}
SAVE_FREQ=${SAVE_FREQ:-8}
KEEP_CKPT=${KEEP_CKPT:-2}
export GEO3K_FORMAT_BONUS=${GEO3K_FORMAT_BONUS:-0.1}

# 显存：FSDP 训练状态约 16 字节/参数（4.4B → 约 70GB），按卡数分片
STATE_PER_GPU=$(( 70 / NUM_GPU ))
BUDGET=$(( GPU_MEM_GB * 55 / 100 ))
OPTIMIZER_OFFLOAD=${OPTIMIZER_OFFLOAD:-$([ ${STATE_PER_GPU} -gt ${BUDGET} ] && echo True || echo False)}
PARAM_OFFLOAD=${PARAM_OFFLOAD:-$([ $(( 35 / NUM_GPU )) -gt ${BUDGET} ] && echo True || echo False)}
if [ ${GPU_MEM_GB} -ge 70 ]; then GMU=0.55; TOK_K=4; elif [ ${GPU_MEM_GB} -ge 40 ]; then GMU=0.45; TOK_K=2; else GMU=0.35; TOK_K=1; fi
GMU=${GPU_MEMORY_UTILIZATION:-$GMU}
MAX_TOK=$(( (MAX_PROMPT_LEN + MAX_RESPONSE_LEN) * TOK_K ))
GEN_TP=${GEN_TP:-1}

mkdir -p "${SAVE_DIR}"
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false NCCL_DEBUG=WARN HYDRA_FULL_ERROR=1
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
LOGGER="['console']"; [ -n "${WANDB_API_KEY}" ] && LOGGER="['console','wandb']"

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    data.train_files="${DATA_DIR}/train.parquet" \
    data.val_files="${DATA_DIR}/val.parquet" \
    data.train_batch_size=${TRAIN_BATCH_SIZE} \
    data.max_prompt_length=${MAX_PROMPT_LEN} \
    data.max_response_length=${MAX_RESPONSE_LEN} \
    data.filter_overlong_prompts=True \
    data.truncation=error \
    data.image_key=images \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=${LR} \
    actor_rollout_ref.actor.ppo_epochs=1 \
    actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BATCH} \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${MAX_TOK} \
    actor_rollout_ref.actor.use_kl_loss=${USE_KL_LOSS} \
    actor_rollout_ref.actor.kl_loss_coef=${KL_LOSS_COEF} \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.clip_ratio=0.2 \
    actor_rollout_ref.actor.clip_ratio_high=${CLIP_HIGH} \
    actor_rollout_ref.actor.loss_agg_mode=token-mean \
    actor_rollout_ref.actor.fsdp_config.param_offload=${PARAM_OFFLOAD} \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=${OPTIMIZER_OFFLOAD} \
    actor_rollout_ref.rollout.name=sglang \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${GEN_TP} \
    actor_rollout_ref.rollout.gpu_memory_utilization=${GMU} \
    actor_rollout_ref.rollout.n=${N_SAMPLES} \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=$(( MAX_TOK * 2 )) \
    actor_rollout_ref.rollout.val_kwargs.do_sample=False \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=$(( MAX_TOK * 2 )) \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    custom_reward_function.path="${REWARD_FN_PATH}" \
    custom_reward_function.name=compute_score \
    trainer.critic_warmup=0 \
    trainer.logger="${LOGGER}" \
    trainer.project_name=qwen3vl_4b_geo3k_grpo \
    trainer.experiment_name=qwen3vl_4b_geo3k_${NUM_GPU}gpu \
    trainer.n_gpus_per_node=${NUM_GPU} \
    trainer.nnodes=1 \
    trainer.save_freq=${SAVE_FREQ} \
    trainer.test_freq=${SAVE_FREQ} \
    trainer.val_before_train=True \
    trainer.resume_mode=auto \
    trainer.max_actor_ckpt_to_keep=${KEEP_CKPT} \
    trainer.total_epochs=${TOTAL_EPOCHS} \
    trainer.default_local_dir="${SAVE_DIR}" \
    "$@" 2>&1 | tee "${SAVE_DIR}/training_$(date +%Y%m%d_%H%M%S).log"

echo "[DONE] 检查点: ${SAVE_DIR}"
echo "  导出 HF:  python3 -m verl.model_merger merge --backend fsdp --local_dir ${SAVE_DIR}/global_step_N/actor --target_dir ${SAVE_DIR}/hf_step_N"
echo "  评估:     python3 evaluate_geo3k.py --model_path ${SAVE_DIR}/hf_step_N --input_file ${DATA_DIR}/test.parquet"
