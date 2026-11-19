"""开训前的端到端冒烟测试：用合成 token 跑通"采样 → 训练（含 torch.compile）→ 检查点保存/恢复 → 评估"。

目的：在花几个小时准备数据之前，几分钟内暴露 GPU 环境或代码问题。产物在 RUN_DIR/smoke，不影响正式训练。
"""

import copy
import math
import shutil

import numpy as np
import torch

from .common import RUN_DIR, now
from .eval_agent import EvalAgent
from .trainer_agent import TrainerAgent


class _StubTokenizer:
    def encode(self, text, add_bos=False, add_eos=False):
        ids = [5 + (ord(c) % 290) for c in text[:32]]
        return ([2] if add_bos else []) + ids + ([3] if add_eos else [])

    def decode(self, ids):
        return ''.join(chr(0x4e00 + int(i)) for i in ids if int(i) >= 5)


def pipeline_smoke(config, device, compile_model):
    root = RUN_DIR / 'smoke'
    if root.exists():
        shutil.rmtree(root)
    (root / 'tokens').mkdir(parents=True)
    rng = np.random.default_rng(0)
    manifest = {'tokenizer': {'vocab_size': 300}, 'sources': {}}
    for name in ('alpha', 'beta'):
        entry = {}
        for split, n in (('train', 60000), ('val', 6000), ('test', 6000)):
            path = root / 'tokens' / f'{name}.{split}.bin'
            rng.integers(5, 300, size=n).astype('<u2').tofile(path)
            entry[split] = [{'path': str(path.relative_to(RUN_DIR)), 'tokens': n}]
        manifest['sources'][name] = entry
    weights = {'alpha': 0.6, 'beta': 0.4}
    plan = {'name': 'smoke', 'precision': 'bf16', 'micro_batch': 4, 'accum': 2, 'seq_len': 64, 'total_steps': 30,
            'warmup_steps': 5, 'decay_fraction': 0.2, 'peak_lr': 1e-3, 'compile': compile_model,
            'model': dict(hidden_size=128, num_layers=2, num_heads=4, num_kv_heads=2, intermediate_size=256),
            'weights': dict(weights), 'eval_weights': dict(weights)}
    cfg = copy.deepcopy(config)
    cfg['eval'].update(max_new_tokens=12, val_chunks_per_source=4)
    cfg['train']['checkpoint_minutes'] = 10_000

    def make():
        t = TrainerAgent(cfg, None, manifest, copy.deepcopy(plan), device, {'type': 'smoke'},
                         ckpt_dir=root / 'ckpt', log_path=root / 'train_log.jsonl')
        t.build()
        return t

    started = now()
    trainer = make()
    report = trainer.run_segment(seconds=600, hard_deadline=now() + 900)
    assert report['stop_reason'] == 'plan_complete' and report['segment_steps'] == 30, report
    assert math.isfinite(report['train_loss']), report
    resumed = make()
    assert resumed.step == 30 and resumed.tokens_seen == trainer.tokens_seen
    for (n1, p1), (n2, p2) in zip(trainer.model.named_parameters(), resumed.model.named_parameters()):
        assert n1 == n2 and torch.equal(p1.detach().cpu(), p2.detach().cpu()), n1
    assert torch.equal(trainer.sampler.batch(3), resumed.sampler.batch(3)), 'sampler state not restored'
    evaluator = EvalAgent(cfg, None, manifest, _StubTokenizer(), device)
    evaluator.out_dir = root
    record = evaluator.evaluate(trainer, final=True)
    assert math.isfinite(record['val_loss']) and math.isfinite(record['test_loss']), record
    trainer.save_weights(root / 'ckpt' / 'weights.pt')
    best = torch.load(root / 'ckpt' / 'weights.pt', map_location='cpu', weights_only=True)
    assert best['step'] == 30 and 'config' in best and 'tokenizer' in best
    return {'passed': True, 'compile': compile_model, 'train_loss': report['train_loss'], 'val_loss': record['val_loss'],
            'uniform_loss': round(math.log(295), 3), 'cloze_acc': record['cloze_acc'], 'seconds': round(now() - started, 1)}
