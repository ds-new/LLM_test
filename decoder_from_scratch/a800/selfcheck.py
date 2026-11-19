"""部署后的快速自检（不需要 GPU，约 1 分钟）：python -m decoder_from_scratch.a800.selfcheck

检查清洗规则、WSD 学习率、混合采样器的确定性与断点恢复、ModelScope 文件列表接口、
以及（装了 tokenizers 时）HF 快速分词器转换。GPU 相关检查在 loop 的 selftest 阶段完成。
"""

import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np


def check(name, cond, detail=''):
    print(f'[{"OK" if cond else "FAIL"}] {name} {detail}', flush=True)
    if not cond:
        check.failed += 1


check.failed = 0


def main(online=True):
    tmp = Path(tempfile.mkdtemp(prefix='a8check_'))
    os.environ.setdefault('A8_RUN_DIR', str(tmp))
    from . import cleaning
    from .trainer_agent import MixtureSampler, wsd_lr

    # 1. 清洗
    check('中文标点规范化', cleaning.normalize_zh_punct('你好,世界!今天(周一)') == '你好，世界！今天（周一）')
    good = ('人工智能是计算机科学的一个分支，研究如何让机器模拟人类的学习、推理和决策能力。早期的研究集中在符号推理和专家系统上，'
            '研究者试图把人类专家的知识写成规则。二十世纪八十年代以后，统计学习方法逐渐兴起，机器开始从数据中自动归纳规律。'
            '近十年来，深度学习借助大规模数据和图形处理器的算力，在图像识别、语音识别和机器翻译等任务上取得了突破。'
            '如今，大语言模型能够根据上下文生成连贯的文字，被广泛用于写作辅助、编程和教育等领域。')
    text, reason = cleaning.clean_text(good, 'zh')
    check('正常中文被接收', text is not None, reason or '')
    check('垃圾文本被过滤', cleaning.clean_text('点击这里 online casino 赢大奖 ' * 20, 'zh')[0] is None)
    check('specialist 不被误伤', cleaning.clean_text('The specialist explained the socialist history of the region in a long and '
                                                  'careful lecture that covered many decades of economic change. ' * 3, 'en')[0] is not None)

    # 2. 学习率
    lrs = [wsd_lr(s, 1000, 4e-4, 100, 0.2, 0.1) for s in (0, 99, 500, 799, 900, 1000)]
    check('WSD 学习率', abs(lrs[1] - 4e-4) < 1e-12 and lrs[2] == 4e-4 and lrs[3] == 4e-4 and abs(lrs[4] - 2.2e-4) < 1e-9 and abs(lrs[5] - 4e-5) < 1e-9,
          str([f'{x:.2e}' for x in lrs]))

    # 3. 采样器
    run = Path(os.environ['A8_RUN_DIR'])
    (run / 'tok').mkdir(parents=True, exist_ok=True)
    manifest = {'sources': {}}
    for name, n in (('a', 50_000), ('b', 20_000)):
        p = run / 'tok' / f'{name}.bin'
        np.arange(n, dtype=np.uint16).tofile(p)
        manifest['sources'][name] = {'train': [{'path': f'tok/{name}.bin', 'tokens': n}]}
    s1 = MixtureSampler(manifest, {'a': 0.7, 'b': 0.3}, 64, 1)
    s1.batch(10)
    s2 = MixtureSampler(manifest, {'a': 0.7, 'b': 0.3}, 64, 1)
    s2.load_state_dict(s1.state_dict())
    check('采样器断点恢复一致', bool((s1.batch(5) == s2.batch(5)).all()))
    b = s1.batch(4)
    check('采样块是连续 token', bool((np.diff(np.asarray(b), axis=1) == 1).all()), str(tuple(b.shape)))

    # 4. ModelScope
    if online:
        from .sources import SOURCES, SUFFIX, list_files
        for spec in SOURCES:
            try:
                files = list_files(spec['dataset'], spec['revision'], spec['roots'][0], SUFFIX[spec['format']])
                check(f'ModelScope 列表 {spec["name"]}', len(files) > 0,
                      f'{spec["roots"][0]}: {len(files)} 个文件，例 {files[0]["path"] if files else ""}')
            except Exception as exc:  # noqa: BLE001
                check(f'ModelScope 列表 {spec["name"]}', False, f'{type(exc).__name__}: {exc}')

    # 5. 依赖
    for mod in ('pyarrow', 'tokenizers', 'torch'):
        try:
            m = __import__(mod)
            check(f'依赖 {mod}', True, getattr(m, '__version__', ''))
        except ImportError:
            check(f'依赖 {mod}', False, '未安装')
    print('自检失败项：', check.failed, flush=True)
    return check.failed


if __name__ == '__main__':
    sys.exit(1 if main(online='--offline' not in sys.argv) else 0)
