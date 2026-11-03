#!/bin/bash
###############################################################################
# 单次流水线（不经过多智能体 loop）：环境检查 → 预处理 → GRPO 训练 → 导出 → test 评估
#   bash run_all.sh             # 全部
#   bash run_all.sh --step 3    # 从训练开始
# 自动调参 / OOM 降级 / 提前停止 / 多轮对比请改用：nohup bash multiagent/run_loop.sh > /dev/null 2>&1 &
###############################################################################
set -eo pipefail

START_STEP=1
while [[ $# -gt 0 ]]; do
    case $1 in
        --step) START_STEP="$2"; shift 2;;
        *) echo "未知参数: $1"; exit 1;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "${SCRIPT_DIR}"
[ -f /etc/network_turbo ] && source /etc/network_turbo >/dev/null 2>&1 || true
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export MODEL_PATH="${MODEL_PATH:-/root/autodl-tmp/qwen3_vl_4}"
export DATA_DIR="${DATA_DIR:-/root/autodl-tmp/data/geo3k}"
export SAVE_DIR="${SAVE_DIR:-/root/autodl-tmp/checkpoints/qwen3vl_4b_geo3k_grpo}"
GEO3K_RAW="${GEO3K_RAW:-hiyouga/geometry3k}"

echo "# 模型 ${MODEL_PATH} | 数据 ${DATA_DIR} | 输出 ${SAVE_DIR} | 起始步骤 ${START_STEP}"

if [ ${START_STEP} -le 1 ]; then
    echo "========== 1/5 环境检查 =========="
    bash setup_env_gpu.sh
fi

if [ ${START_STEP} -le 2 ]; then
    echo "========== 2/5 预处理 (train / val / test) =========="
    if [ -f "${DATA_DIR}/train.parquet" ] && [ -f "${DATA_DIR}/val.parquet" ] && [ -f "${DATA_DIR}/test.parquet" ]; then
        echo "[INFO] 已存在: ${DATA_DIR}"
    else
        python3 preprocess_geo3k.py --data_dir "${GEO3K_RAW}" --output_dir "${DATA_DIR}"
    fi
fi

if [ ${START_STEP} -le 3 ]; then
    echo "========== 3/5 GRPO 训练 =========="
    bash run_grpo_qwen3vl_4b_fsdp_sglang_gpu.sh
fi

HF_DIR=""
if [ ${START_STEP} -le 4 ]; then
    echo "========== 4/5 导出最新检查点为 HF 格式 =========="
    LATEST=$(cat "${SAVE_DIR}/latest_checkpointed_iteration.txt" 2>/dev/null || true)
    if [ -n "${LATEST}" ]; then
        ACTOR="${SAVE_DIR}/global_step_${LATEST}/actor"
        HF_DIR="${SAVE_DIR}/hf_step_${LATEST}"
        if ls "${ACTOR}/huggingface/"*.safetensors >/dev/null 2>&1; then
            HF_DIR="${ACTOR}/huggingface"
        elif [ ! -f "${HF_DIR}/config.json" ]; then
            python3 -m verl.model_merger merge --backend fsdp --local_dir "${ACTOR}" --target_dir "${HF_DIR}"
            for f in preprocessor_config.json chat_template.json chat_template.jinja tokenizer.json tokenizer_config.json vocab.json merges.txt generation_config.json; do
                [ -f "${MODEL_PATH}/$f" ] && [ ! -f "${HF_DIR}/$f" ] && cp "${MODEL_PATH}/$f" "${HF_DIR}/"
            done
        fi
    else
        echo "[WARN] 未找到检查点"
    fi
fi

if [ ${START_STEP} -le 5 ]; then
    echo "========== 5/5 test 评估（基线 vs 训练后）=========="
    mkdir -p "${SAVE_DIR}/eval"
    python3 evaluate_geo3k.py --model_path "${MODEL_PATH}" --input_file "${DATA_DIR}/test.parquet" \
        --output_json "${SAVE_DIR}/eval/base_test.json"
    if [ -n "${HF_DIR}" ]; then
        python3 evaluate_geo3k.py --model_path "${HF_DIR}" --input_file "${DATA_DIR}/test.parquet" \
            --output_json "${SAVE_DIR}/eval/trained_test.json"
    fi
fi
echo "# 完成。注意：这里评估的是最新检查点；按 val 挑选最佳检查点请用多智能体 loop。"
