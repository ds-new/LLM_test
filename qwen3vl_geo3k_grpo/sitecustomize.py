"""工程目录在 PYTHONPATH 中时，Python 每次启动都会自动导入本文件（含 Ray 拉起的 verl 训练进程）。

只注册一个惰性导入钩子：当 transformers 的 Qwen3-VL 建模模块被导入时，立即应用 geo3k_patches 中的补丁；
其他进程（vLLM、评估脚本等）不受影响，也不会因此多导入 transformers。GEO3K_PATCH_CONV3D=0 可关闭。
"""

import os
import sys

_TARGET = "transformers.models.qwen3_vl.modeling_qwen3_vl"


def _install():
    if os.environ.get("GEO3K_PATCH_CONV3D", "1") == "0":
        return
    import importlib.abc
    import importlib.machinery

    class _Finder(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path, target=None):
            if fullname != _TARGET:
                return None
            spec = importlib.machinery.PathFinder.find_spec(fullname, path)
            if spec is None or spec.loader is None:
                return spec
            orig_exec = spec.loader.exec_module

            def exec_module(module):
                orig_exec(module)
                try:
                    import geo3k_patches
                    geo3k_patches.patch_module(module)
                except Exception as e:  # 补丁失败不影响原功能
                    print(f"[sitecustomize] geo3k patch failed: {e}", file=sys.stderr)

            spec.loader.exec_module = exec_module
            return spec

    sys.meta_path.insert(0, _Finder())


try:
    _install()
except Exception:
    pass
