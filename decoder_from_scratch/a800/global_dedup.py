"""路线 A 第 2 段全局去重的执行端：在正式构建（data_agent.process_file）的清洗之后，按数据工程给出的剔除清单丢掉文档。

剔除清单由 data_eng.py 生成，按（来源，文档哈希，文件编号）定位文档：同一份文本出现在多个文件里时，只丢清单指定文件里的那几份。
原因代码：1 = 与 v1 验证 / 测试文档近似重复（评测污染，总是剔除）；2–5 = 与更早的训练文档近似重复（只在该来源启用去重时剔除）。
清单加载出错时退回普通构建，附加功能不能拖垮主流程。
"""

import numpy as np

_CACHE = {}


def load_exclusion(path, source, exclude_dup):
    key = (str(path), source, bool(exclude_dup))
    if key not in _CACHE:
        z = np.load(path)
        h, f, r = z[f'{source}__hash'], z[f'{source}__file'], z[f'{source}__reason']
        keep = (r == 1) | ((r != 1) & bool(exclude_dup))
        table = {}
        for hh, ff, rr in zip(h[keep].tolist(), f[keep].tolist(), r[keep].tolist()):
            table.setdefault(hh, {})[ff] = rr
        _CACHE[key] = table
    return _CACHE[key]


def process_file_excl(job):
    from . import cleaning
    from .data_agent import process_file
    try:
        table = load_exclusion(job['exclude_path'], job['source'], job.get('exclude_dup'))
    except Exception:  # noqa: BLE001
        return process_file(job)
    idx = int(job['index'])
    orig = cleaning.clean_text

    def clean_text(raw, kind, *args, **kwargs):
        text, reason = orig(raw, kind, *args, **kwargs)
        if reason is None:
            hit = table.get(cleaning.doc_hash64(text))
            if hit is not None and idx in hit:
                return None, ('eval_contamination' if hit[idx] == 1 else 'global_duplicate')
        return text, reason

    cleaning.clean_text = clean_text
    try:
        return process_file(job)
    finally:
        cleaning.clean_text = orig
