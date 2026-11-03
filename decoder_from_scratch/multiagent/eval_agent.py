"""模型测试评估智能体：固定验证集分领域 loss、固定提示生成质量指标、最终测试集评估。

评估集合在首次评估时固定（固定的块位置、固定提示、固定随机种子），保证不同时刻的结果可比。
"""

from collections import Counter
import json
import math

import numpy as np
import torch

from ..generation import generate
from .architect_agent import autocast_ctx
from .common import RUN_DIR, stamp, write_json

NAME = 'evaluator'

PROMPTS = [
    ('zh_general', '人工智能的发展'),
    ('zh_narrative', '春天来了，小明和妈妈一起去公园，'),
    ('zh_science', '光合作用是指绿色植物'),
    ('zh_history', '中国古代的四大发明包括'),
    ('zh_geography', '长江是中国最长的河流，'),
    ('zh_education', '学习一门新语言最重要的是'),
    ('zh_math', '在直角三角形中，勾股定理指出'),
    ('zh_news', '记者从市教育局获悉，今年'),
    ('zh_travel', '杭州西湖是著名的旅游景点，'),
    ('code', 'def fibonacci(n):\n    '),
    ('en_education', 'The water cycle describes how water'),
    ('en_general', 'Learning to read is important because'),
]
GREEDY_PROMPTS = [0, 2, 6, 10]


def repetition_stats(ids):
    """生成部分的重复度：4-gram 重复比例、末尾 64 token 内最常见 token 占比。"""
    if len(ids) < 8:
        return {'rep4': 0.0, 'tail_top_share': 0.0}
    grams = [tuple(ids[i:i + 4]) for i in range(len(ids) - 3)]
    rep4 = 1 - len(set(grams)) / len(grams)
    tail = ids[-64:]
    top = Counter(tail).most_common(1)[0][1] / len(tail)
    return {'rep4': round(rep4, 4), 'tail_top_share': round(top, 4)}


class EvalAgent:
    def __init__(self, config, bus, manifest, tokenizer, device):
        self.config, self.bus, self.manifest, self.tok = config, bus, manifest, tokenizer
        self.ecfg = config['eval']
        self.device = torch.device(device)
        self.out_dir = RUN_DIR
        self.history_path = RUN_DIR / 'eval_history.jsonl'
        self._val_sets = {}

    def say(self, recipient, kind, content):
        if self.bus is not None:
            self.bus.post(NAME, recipient, kind, content)

    def fixed_chunks(self, split, seq_len):
        """每个领域在 val/test 上均匀取固定位置的块；结果缓存。"""
        key = (split, seq_len)
        if key in self._val_sets:
            return self._val_sets[key]
        k = self.ecfg['val_chunks_per_domain'] if split == 'val' else 4 * self.ecfg['val_chunks_per_domain']
        sets = {}
        for domain, entry in self.manifest['domains'].items():
            arr = np.memmap(RUN_DIR / entry[split]['path'], dtype='<u2', mode='r')
            n = (len(arr) - 1) // seq_len
            if n <= 0:
                continue
            idx = np.unique(np.linspace(0, n - 1, num=min(k, n)).astype(np.int64))
            sets[domain] = np.stack([arr[i * seq_len:i * seq_len + seq_len + 1].astype(np.int64) for i in idx])
        self._val_sets[key] = sets
        return sets

    @torch.no_grad()
    def loss_by_domain(self, model, split, seq_len, precision, batch=8):
        model.eval()
        out = {}
        for domain, rows in self.fixed_chunks(split, seq_len).items():
            total, count = 0.0, 0
            for i in range(0, len(rows), batch):
                x = torch.from_numpy(rows[i:i + batch]).to(self.device)
                with autocast_ctx(self.device, precision):
                    loss = model(x[:, :-1], targets=x[:, 1:]).loss
                total += float(loss) * x.size(0)
                count += x.size(0)
            out[domain] = round(total / count, 4)
        return out

    @torch.no_grad()
    def generation_probe(self, model):
        model.eval()
        rows = []
        for i, (tag, prompt) in enumerate(PROMPTS):
            ids = self.tok.encode(prompt, add_bos=True)
            x = torch.tensor([ids], device=self.device)
            modes = [('sample', True)] + ([('greedy', False)] if i in GREEDY_PROMPTS else [])
            for mode, do_sample in modes:
                torch.manual_seed(1234 + i)
                kwargs = dict(max_new_tokens=self.ecfg['max_new_tokens'], eos_token_id=3, use_cache=True)
                if do_sample:
                    kwargs.update(do_sample=True, temperature=self.ecfg['temperature'], top_p=self.ecfg['top_p'])
                else:
                    kwargs.update(do_sample=False)
                out = generate(model, x, **kwargs)[0, len(ids):].tolist()
                text = self.tok.decode(out)
                stats = repetition_stats(out)
                rows.append({'tag': tag, 'mode': mode, 'prompt': prompt, 'completion': text,
                             'new_tokens': len(out), 'ended_with_eos': bool(out and out[-1] == 3),
                             'replacement_chars': text.count('�'), **stats})
        if self.device.type == 'mps':
            torch.mps.empty_cache()

        def mean(key, mode):
            vals = [r[key] for r in rows if r['mode'] == mode]
            return round(sum(vals) / max(1, len(vals)), 4)

        summary = {'sample_rep4': mean('rep4', 'sample'), 'greedy_rep4': mean('rep4', 'greedy'),
                   'sample_tail_top_share': mean('tail_top_share', 'sample'),
                   'replacement_chars': sum(r['replacement_chars'] for r in rows)}
        return summary, rows

    def evaluate(self, trainer, final=False):
        plan = trainer.plan
        T, precision = plan['seq_len'], plan['precision']
        val = self.loss_by_domain(trainer.model, 'val', T, precision)
        # 主指标固定使用初始配比加权，配比调整后仍可与历史比较
        weights = plan.get('eval_weights') or trainer.sampler.weights
        wsum = sum(weights.get(d, 0) for d in val) or 1.0
        weighted = sum(val[d] * weights.get(d, 0) for d in val) / wsum
        summary, samples = self.generation_probe(trainer.model)
        record = {'time': stamp(), 'step': trainer.step, 'tokens_seen': trainer.tokens_seen,
                  'val_loss': round(weighted, 4), 'val_loss_mean': round(sum(val.values()) / len(val), 4),
                  'val_ppl': round(math.exp(weighted), 2), 'val_by_domain': val, **summary}
        if final:
            test = self.loss_by_domain(trainer.model, 'test', T, precision)
            record['test_by_domain'] = test
            record['test_loss'] = round(sum(test[d] * weights.get(d, 0) for d in test) / wsum, 4)
        trainer.model.train()
        with self.history_path.open('a', encoding='utf-8') as f:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
        write_json(self.out_dir / 'samples' / f'step_{trainer.step:07d}.json', samples)
        write_json(self.out_dir / 'samples' / 'latest.json', samples)
        self.say('decision', 'report', {'action': 'eval_final' if final else 'eval', **{
            k: v for k, v in record.items() if k not in ('val_by_domain', 'test_by_domain')}})
        return record, samples
