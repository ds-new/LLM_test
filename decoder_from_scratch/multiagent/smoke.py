"""启动时的端到端冒烟测试（结构算法智能体 + 训练智能体 + 评估智能体）。

在真正花数小时准备数据之前，用合成 token 跑通：混合采样 → 训练 30 步 → 检查点保存/恢复（参数与采样状态一致）
→ 验证 loss 与生成评估 → best.pt 可被 weights_only 加载（与工程 infer.py 兼容）。
产物写在 RUN_DIR/smoke，不影响正式训练目录。
"""

import copy
import math

import numpy as np
import torch

from .common import RUN_DIR, now
from .eval_agent import EvalAgent
from .trainer_agent import TrainerAgent


class _StubTokenizer:
    def encode(self, text, add_bos=False, add_eos=False):
        ids = [5 + (ord(c) % 290) for c in text[:24]]
        return ([2] if add_bos else []) + ids + ([3] if add_eos else [])

    def decode(self, ids):
        return ''.join(chr(0x4e00 + int(i)) for i in ids if int(i) >= 5)


def pipeline_smoke(config, device):
    root = RUN_DIR / 'smoke'
    (root / 'ckpt').mkdir(parents=True, exist_ok=True)
    for stale in (root / 'ckpt').glob('*.pt'):
        stale.replace(root / 'ckpt' / (stale.name + '.old'))
    rng = np.random.default_rng(0)
    manifest = {'tokenizer': {'vocab_size': 300}, 'bos_id': 2, 'eos_id': 3, 'domains': {}}
    for domain in ('alpha', 'beta'):
        entry = {}
        for split, n in (('train', 40000), ('val', 4000), ('test', 4000)):
            path = root / f'{domain}.{split}.bin'
            rng.integers(5, 300, size=n).astype('<u2').tofile(path)
            info = {'path': str(path.relative_to(RUN_DIR)), 'tokens': n, 'docs': 1}
            entry[split] = [info] if split == 'train' else info
        manifest['domains'][domain] = entry
    weights = {'alpha': 0.7, 'beta': 0.3}
    plan = {'name': 'smoke', 'params': 0, 'precision': 'fp32', 'micro_batch': 4, 'accum': 2, 'seq_len': 64,
            'tokens_per_step': 512, 'total_steps': 30, 'warmup_steps': 5, 'peak_lr': 1e-3,
            'model': dict(hidden_size=64, num_layers=2, num_heads=4, num_kv_heads=2, intermediate_size=128),
            'weights': dict(weights), 'eval_weights': dict(weights), 'initial_weights': dict(weights),
            'extra_shards': []}
    cfg = copy.deepcopy(config)
    cfg['eval']['max_new_tokens'] = 12
    cfg['train']['checkpoint_minutes'] = 10_000

    def make():
        t = TrainerAgent(cfg, None, manifest, copy.deepcopy(plan), device, {'type': 'smoke'})
        t.ckpt_dir = root / 'ckpt'
        t.log_path = root / 'train_log.jsonl'
        t.build()
        return t

    started = now()
    trainer = make()
    report = trainer.run_segment(seconds=300, hard_deadline=now() + 600)
    assert report['stop_reason'] == 'plan_complete' and report['segment_steps'] == 30, report
    assert report['train_loss'] is not None and math.isfinite(report['train_loss']), report
    resumed = make()
    assert resumed.step == 30 and resumed.tokens_seen == trainer.tokens_seen
    for (n1, p1), (n2, p2) in zip(trainer.model.named_parameters(), resumed.model.named_parameters()):
        assert n1 == n2 and torch.equal(p1.detach().cpu(), p2.detach().cpu()), n1
    assert torch.equal(trainer.sampler.batch(3), resumed.sampler.batch(3)), 'sampler state not restored'
    evaluator = EvalAgent(cfg, None, manifest, _StubTokenizer(), device)
    evaluator.out_dir = root
    evaluator.history_path = root / 'eval_history.jsonl'
    record, samples = evaluator.evaluate(trainer, final=True)
    assert math.isfinite(record['val_loss']) and math.isfinite(record['test_loss']), record
    assert len(samples) == 16
    trainer.save_best(record['val_loss'])
    best = torch.load(root / 'ckpt' / 'best.pt', map_location='cpu', weights_only=True)
    assert best['step'] == 30 and 'config' in best and 'tokenizer' in best
    first_loss = math.log(295)
    return {'passed': True, 'steps': 30, 'train_loss': report['train_loss'], 'val_loss': record['val_loss'],
            'uniform_loss_reference': round(first_loss, 3), 'seconds': round(now() - started, 1),
            'segment_tok_s': report['segment_tok_s']}
