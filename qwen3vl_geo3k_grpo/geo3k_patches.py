"""运行时补丁（训练进程通过 sitecustomize 自动加载，或由 verl 的 model.external_lib 显式导入）。

Qwen3-VL 视觉塔的 patch embedding 是 kernel == stride 的 Conv3d，本质是对每个不重叠 patch 做一次线性投影。
torch 2.9.0 + cuDNN 9.10 下 bf16 Conv3d 有严重性能回退：实测 1000 个 patch 需 6.2 秒，等价的 F.linear 只要 0.6 毫秒，
使 FSDP 计算 log-prob / 反向时 GPU 长时间空转。这里把 forward 换成数学上等价的 F.linear（同一份权重，view 成二维）。
设置环境变量 GEO3K_PATCH_CONV3D=0 可关闭。
"""

import os

import torch.nn.functional as F

_PATCHED = False


def _patch_embed_forward(self, hidden_states):
    target_dtype = self.proj.weight.dtype
    k = self.in_channels * self.temporal_patch_size * self.patch_size * self.patch_size
    x = hidden_states.reshape(-1, k).to(dtype=target_dtype)
    return F.linear(x, self.proj.weight.reshape(self.embed_dim, k), self.proj.bias)


def patch_module(mod):
    """对已导入的 transformers.models.qwen3_vl.modeling_qwen3_vl 打补丁。"""
    global _PATCHED
    cls = getattr(mod, "Qwen3VLVisionPatchEmbed", None)
    if cls is None or getattr(cls, "_geo3k_linear", False):
        return False
    cls.forward = _patch_embed_forward
    cls._geo3k_linear = True
    _PATCHED = True
    if os.environ.get("GEO3K_PATCH_VERBOSE"):
        print(f"[geo3k_patches] Qwen3VLVisionPatchEmbed.forward -> F.linear (pid {os.getpid()})", flush=True)
    return True


def apply():
    if os.environ.get("GEO3K_PATCH_CONV3D", "1") == "0":
        return False
    try:
        from transformers.models.qwen3_vl import modeling_qwen3_vl
    except Exception:
        return False
    return patch_module(modeling_qwen3_vl)


apply()
