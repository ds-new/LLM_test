"""E 阶段（蒸馏 SFT）训练入口：包一层 verl 的 SFT trainer（engine 版或 fsdp 版），
只改一处：存 Hugging Face 权重时转成 bf16（verl 默认按 fp32 主权重存，8B 模型会变成 33 GB）。
用法（由 sft/run_sft.sh 调用）：torchrun ... sft/run_engine_sft.py engine|fsdp <hydra overrides>
"""
import sys

import torch
import verl.utils.checkpoint.fsdp_checkpoint_manager as _m

_orig = _m.get_fsdp_full_state_dict


def _bf16_state_dict(*a, **k):
    sd = _orig(*a, **k)
    if not sd:
        return sd
    return {n: (v.to(torch.bfloat16) if torch.is_tensor(v) and v.is_floating_point() else v) for n, v in sd.items()}


_m.get_fsdp_full_state_dict = _bf16_state_dict

if __name__ == "__main__":
    which = sys.argv.pop(1)
    if which == "engine":
        from verl.trainer.sft_trainer import main
    else:
        from verl.trainer.fsdp_sft_trainer import main
    main()
