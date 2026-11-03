#!/bin/bash
###############################################################################
# AutoDL NVIDIA GPU 环境检查（verl + FSDP + SGLang，Qwen3-VL-4B）
#   bash setup_env_gpu.sh            # 只检查
#   bash setup_env_gpu.sh --install  # 缺失的轻量依赖（math-verify、datasets、qwen-vl-utils、modelscope）自动安装
# torch / verl / sglang 版本强相关，请按 verl 官方文档安装匹配版本，这里不自动装。
###############################################################################
INSTALL=0; [ "$1" = "--install" ] && INSTALL=1
[ -f /etc/network_turbo ] && source /etc/network_turbo >/dev/null 2>&1

echo "== Python / GPU =="
python3 --version
nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv,noheader || echo "[ERROR] nvidia-smi 不可用"

echo "== 关键依赖 =="
python3 - <<'EOF'
import importlib.util
for m in ["torch", "verl", "sglang", "vllm", "transformers", "huggingface_hub", "ray", "datasets", "math_verify", "qwen_vl_utils", "modelscope", "yaml"]:
    try:
        mod = importlib.import_module(m)
        print(f"  {m:14s} {getattr(mod, '__version__', 'installed')}")
    except Exception as e:
        print(f"  {m:14s} [缺失] {type(e).__name__}: {str(e)[:160]}")
try:
    import torch
    print(f"  CUDA 可用: {torch.cuda.is_available()}  CUDA {torch.version.cuda}  GPU 数 {torch.cuda.device_count()}")
    from transformers import AutoModelForImageTextToText  # noqa
    print("  transformers ImageTextToText 接口: OK（Qwen3-VL 需要 >= 4.57）")
except Exception as e:
    print(f"  [WARN] {e}")
EOF

if [ ${INSTALL} -eq 1 ]; then
    # 只安装缺失的包，且不升级已有包：-U 会把 huggingface-hub 拉到 1.x/2.x，
    # 与 transformers 4.57（要求 huggingface-hub<1.0）冲突
    MISSING=$(python3 - <<'PY'
import importlib.util
pkgs = {"math_verify": "math-verify", "datasets": "datasets", "qwen_vl_utils": "qwen-vl-utils",
        "modelscope": "modelscope", "yaml": "pyyaml"}
print(" ".join(p for m, p in pkgs.items() if importlib.util.find_spec(m) is None))
PY
)
    if [ -n "${MISSING}" ]; then
        pip install ${MISSING} "huggingface-hub<1.0"
    else
        echo "  轻量依赖已齐全，不做任何安装"
    fi
fi

echo "== 磁盘（检查点放数据盘 /root/autodl-tmp：至少约 50GB，约 86GB 以上可保存含优化器的完整检查点）=="
df -h /root/autodl-tmp 2>/dev/null || df -h .

echo "== verl 配置键探测（多智能体 loop 会自动做）=="
CFG_OUT=$(python3 -m verl.trainer.main_ppo --cfg job 2>&1)
if echo "${CFG_OUT}" | grep -q "adv_estimator"; then
    for k in clip_ratio_high norm_adv_by_std_in_grpo max_actor_ckpt_to_keep free_cache_engine; do
        echo "${CFG_OUT}" | grep -q "${k}:" && echo "  ${k}: 支持" || echo "  ${k}: 不支持"
    done
else
    echo "  [WARN] verl --cfg job 失败，最后 15 行输出："; echo "${CFG_OUT}" | tail -15
fi
