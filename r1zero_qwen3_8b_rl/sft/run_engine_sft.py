"""E 阶段（蒸馏 SFT）训练入口：包一层 verl 的 SFT trainer（engine 版或 fsdp 版），改两处：
1. 存 Hugging Face 权重时转成 bf16（verl 默认按 fp32 主权重存，8B 模型会变成 33 GB）。
2. 修 verl 0.6.1 的 bug：FSDP engine 建 FSDPCheckpointManager 时传的是 checkpoint_contents=...，
   被 **kwargs 吞掉，save_contents 永远是默认的 model+optimizer+extra（8B 的 fp32 分片约 100 GB）。
   2026-11-19 13:25 冒烟存盘因此把数据盘写满；这里把 checkpoint_contents 转给 checkpoint_config。
用法（由 sft/run_sft.sh 调用）：torchrun ... sft/run_engine_sft.py engine|fsdp <hydra overrides>
"""
import os
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

_CM = _m.FSDPCheckpointManager
_orig_init = _CM.__init__


def _init(self, *a, checkpoint_config=None, **k):
    if checkpoint_config is None and "checkpoint_contents" in k:
        checkpoint_config = k.pop("checkpoint_contents")
    _orig_init(self, *a, checkpoint_config=checkpoint_config, **k)
    if os.environ.get("RANK", "0") == "0":
        print(f"[run_engine_sft] checkpoint save_contents = {self.checkpoint_save_contents}", flush=True)
    bad = {"model", "optimizer"} & set(self.checkpoint_save_contents)
    if bad and os.environ.get("ALLOW_SHARDED_CKPT") != "1":
        raise RuntimeError(f"save_contents 含 {sorted(bad)}：8B 的分片存盘约 100 GB，数据盘放不下")


_CM.__init__ = _init

if __name__ == "__main__":
    which = sys.argv.pop(1)
    if which == "engine":
        from verl.trainer.sft_trainer import main
    else:
        from verl.trainer.fsdp_sft_trainer import main
    main()
