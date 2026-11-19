"""模型测试评估智能体。三类指标，口径在第一次评估时固定，保证不同时刻可比：

  1. 验证集 loss：每个数据来源从 val 分片里均匀取固定位置的 32 个块，分来源报告，主指标按初始配比加权；
  2. 选择题（cloze）准确率：38 道常识 / 数学 / 代码小题，比较 4 个候选续写的平均对数概率，看模型"知道"什么。
     0.5B 规模的预训练模型在 C-Eval/CMMLU 上接近随机（25%），这种简单题更能反映训练进展；
  3. 生成质量：12 个固定提示的采样与贪心续写，统计 4-gram 重复率、是否正常结束、替换字符数，并保存全文供人工阅读。
"""

from collections import Counter
import json
import math

import numpy as np
import torch

from ..generation import generate
from .cleaning import normalize_zh_punct
from .common import RUN_DIR, stamp, write_json
from .trainer_agent import autocast

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
    ('code_python', 'def fibonacci(n):\n    """返回第 n 个斐波那契数"""\n'),
    ('en_education', 'The water cycle describes how water'),
    ('en_story', 'Once upon a time, there was a little rabbit who'),
    ('math_en', 'To solve the equation 2x + 3 = 11, we first'),
]
GREEDY_TAGS = {'zh_general', 'zh_science', 'code_python', 'en_education'}

CLOZE = [  # (类别, 上下文, 选项, 正确选项下标)
    ('zh_知识', '中国的首都是', ['北京', '上海', '广州', '重庆'], 0),
    ('zh_知识', '在标准大气压下，水的沸点是', ['100摄氏度', '50摄氏度', '0摄氏度', '200摄氏度'], 0),
    ('zh_知识', '一年有', ['十二个月', '十个月', '八个月', '二十个月'], 0),
    ('zh_知识', '《红楼梦》的作者是', ['曹雪芹', '鲁迅', '李白', '莫言'], 0),
    ('zh_知识', '太阳每天从', ['东方升起', '西方升起', '南方升起', '北方升起'], 0),
    ('zh_知识', '植物进行光合作用的主要场所是', ['叶绿体', '线粒体', '细胞核', '液泡'], 0),
    ('zh_知识', '长江最终注入', ['东海', '南海', '渤海', '黄海'], 0),
    ('zh_知识', '中国古代四大发明是造纸术、印刷术、火药和', ['指南针', '蒸汽机', '电话', '望远镜'], 0),
    ('zh_知识', '李白是唐朝著名的', ['诗人', '皇帝', '将军', '科学家'], 0),
    ('zh_知识', '地球绕太阳公转一周大约需要', ['一年', '一天', '一个月', '一小时'], 0),
    ('zh_常识', '下雨了，出门最好带上', ['雨伞', '墨镜', '扇子', '泳衣'], 0),
    ('zh_常识', '冬天很冷，人们通常会穿上', ['羽绒服', '短袖', '凉鞋', '泳衣'], 0),
    ('zh_常识', '医生的主要工作是', ['给病人看病', '给学生上课', '驾驶飞机', '种植庄稼'], 0),
    ('zh_常识', '小明饿了，于是他去厨房', ['做饭', '睡觉', '洗澡', '跑步'], 0),
    ('zh_数学', '3加5等于', ['8', '7', '9', '15'], 0),
    ('zh_数学', '12除以4等于', ['3', '4', '6', '8'], 0),
    ('zh_数学', '三角形的内角和是', ['180度', '90度', '360度', '270度'], 0),
    ('zh_数学', '如果x+2=5，那么x等于', ['3', '2', '5', '7'], 0),
    ('en_knowledge', 'The capital of France is', [' Paris', ' London', ' Berlin', ' Madrid'], 0),
    ('en_knowledge', 'Water freezes at', [' 0 degrees Celsius', ' 100 degrees Celsius', ' 50 degrees Celsius', ' 30 degrees Celsius'], 0),
    ('en_knowledge', 'The largest planet in our solar system is', [' Jupiter', ' Mars', ' Earth', ' Mercury'], 0),
    ('en_knowledge', 'Plants make their own food through a process called', [' photosynthesis', ' digestion', ' evaporation', ' erosion'], 0),
    ('en_common', 'The opposite of hot is', [' cold', ' big', ' fast', ' happy'], 0),
    ('en_common', 'A baby cat is called a', [' kitten', ' puppy', ' calf', ' chick'], 0),
    ('en_common', 'She was very tired, so she went to', [' bed', ' work', ' the gym', ' a party'], 0),
    ('en_common', 'There are seven days in a', [' week', ' month', ' year', ' day'], 0),
    ('en_math', '2 + 2 =', [' 4', ' 5', ' 3', ' 22'], 0),
    ('en_math', '10 times 3 equals', [' 30', ' 13', ' 7', ' 300'], 0),
    ('en_math', 'The square root of 81 is', [' 9', ' 8', ' 7', ' 81'], 0),
    ('code', 'def add(a, b):\n    return a', [' + b', ' - b', ' * b', ' / b'], 0),
    ('code', 'for i in range(10):\n    print(', ['i)', 'x)', 'j)', 'k)'], 0),
    ('code', 'import numpy as', [' np', ' pd', ' plt', ' tf'], 0),
    ('code', 'if x > 0:\n    sign = 1\nelse:\n    sign =', [' -1', ' 1', ' 2', ' 10'], 0),
    ('code', 'def is_even(n):\n    return n %', [' 2 == 0', ' 3 == 0', ' 2 == 1', ' 5 == 0'], 0),
    ('code', 'const x = 5;\nconsole.', ['log(x);', 'print(x);', 'echo(x);', 'puts(x);'], 0),
    ('code', '#include <stdio.h>\nint main() {\n    printf("Hello\\n");\n    return', [' 0;', ' 1;', ' -1;', ' 2;'], 0),
    ('zh_知识', '人体最大的器官是', ['皮肤', '心脏', '眼睛', '牙齿'], 0),
    ('en_knowledge', 'The chemical symbol for water is', [' H2O', ' CO2', ' NaCl', ' O2'], 0),
]


def repetition_stats(ids):
    if len(ids) < 8:
        return {'rep4': 0.0}
    grams = [tuple(ids[i:i + 4]) for i in range(len(ids) - 3)]
    return {'rep4': round(1 - len(set(grams)) / len(grams), 4)}


class EvalAgent:
    def __init__(self, config, bus, manifest, tokenizer, device):
        self.config, self.bus, self.manifest, self.tok = config, bus, manifest, tokenizer
        self.ecfg = config['eval']
        self.device = torch.device(device)
        self.out_dir = RUN_DIR
        self._chunks = {}

    def say(self, recipient, kind, content):
        if self.bus is not None:
            self.bus.post(NAME, recipient, kind, content)

    def fixed_chunks(self, split, seq_len):
        key = (split, seq_len)
        if key not in self._chunks:
            k = self.ecfg['val_chunks_per_source'] * (2 if split == 'test' else 1)
            sets = {}
            for name, entry in self.manifest['sources'].items():
                rows = []
                shards = [np.memmap(RUN_DIR / s['path'], dtype='<u2', mode='r') for s in entry[split]]
                total = sum((len(a) - 1) // seq_len for a in shards)
                if total <= 0:
                    continue
                wanted = set(np.unique(np.linspace(0, total - 1, num=min(k, total)).astype(np.int64)).tolist())
                idx = 0
                for arr in shards:
                    for c in range((len(arr) - 1) // seq_len):
                        if idx in wanted:
                            rows.append(arr[c * seq_len:c * seq_len + seq_len + 1].astype(np.int64))
                        idx += 1
                sets[name] = np.stack(rows)
            self._chunks[key] = sets
        return self._chunks[key]

    @torch.no_grad()
    def loss_by_source(self, model, split, seq_len, precision, batch=8):
        model.eval()
        out = {}
        for name, rows in self.fixed_chunks(split, seq_len).items():
            total, count = 0.0, 0
            for i in range(0, len(rows), batch):
                x = torch.from_numpy(rows[i:i + batch]).to(self.device)
                with autocast(self.device, precision):
                    loss = model(x[:, :-1], targets=x[:, 1:]).loss
                total += float(loss) * x.size(0)
                count += x.size(0)
            out[name] = round(total / count, 4)
        return out

    @torch.no_grad()
    def cloze(self, model, precision):
        model.eval()
        by_cat, details = Counter(), []
        totals = Counter()
        for cat, context, options, answer in CLOZE:
            context = normalize_zh_punct(context)
            ctx_ids = self.tok.encode(context, add_bos=True)
            seqs, starts = [], []
            for opt in options:
                full = self.tok.encode(context + normalize_zh_punct(opt), add_bos=True)
                common = 0
                while common < min(len(full), len(ctx_ids)) and full[common] == ctx_ids[common]:
                    common += 1
                seqs.append(full)
                starts.append(max(1, common))
            width = max(map(len, seqs))
            x = torch.zeros(len(seqs), width, dtype=torch.long)
            for i, s in enumerate(seqs):
                x[i, :len(s)] = torch.tensor(s)
            x = x.to(self.device)
            with autocast(self.device, precision):
                logits = model(x).logits.float()
            logp = logits.log_softmax(-1)
            scores = []
            for i, (s, st) in enumerate(zip(seqs, starts)):
                lp = sum(float(logp[i, t - 1, s[t]]) for t in range(st, len(s)))
                scores.append(lp / max(1, len(options[i])))  # 按选项字符数归一化，避免偏向短选项
            pred = int(np.argmax(scores))
            totals[cat] += 1
            by_cat[cat] += int(pred == answer)
            details.append({'cat': cat, 'context': context, 'pred': options[pred], 'answer': options[answer]})
        acc = {c: round(by_cat[c] / totals[c], 3) for c in totals}
        return round(sum(by_cat.values()) / len(CLOZE), 4), acc, details

    @torch.no_grad()
    def generation_probe(self, model):
        model.eval()
        rows = []
        for i, (tag, prompt) in enumerate(PROMPTS):
            prompt = normalize_zh_punct(prompt)
            ids = self.tok.encode(prompt, add_bos=True)
            x = torch.tensor([ids], device=self.device)
            for mode in ['sample'] + (['greedy'] if tag in GREEDY_TAGS else []):
                torch.manual_seed(1234 + i)
                kw = dict(max_new_tokens=self.ecfg['max_new_tokens'], eos_token_id=3, use_cache=True)
                if mode == 'sample':
                    kw.update(do_sample=True, temperature=self.ecfg['temperature'], top_p=self.ecfg['top_p'])
                else:
                    kw.update(do_sample=False)
                with autocast(self.device, 'bf16'):
                    out = generate(model, x, **kw)[0, len(ids):].tolist()
                text = self.tok.decode(out)
                rows.append({'tag': tag, 'mode': mode, 'prompt': prompt, 'completion': text, 'new_tokens': len(out),
                             'ended_with_eos': bool(out and out[-1] == 3), 'replacement_chars': text.count('�'),
                             **repetition_stats(out)})

        def mean(mode):
            vals = [r['rep4'] for r in rows if r['mode'] == mode]
            return round(sum(vals) / max(1, len(vals)), 4)

        return {'sample_rep4': mean('sample'), 'greedy_rep4': mean('greedy'),
                'replacement_chars': sum(r['replacement_chars'] for r in rows)}, rows

    def evaluate(self, trainer, final=False):
        plan = trainer.plan
        model = trainer.model
        val = self.loss_by_source(model, 'val', plan['seq_len'], plan['precision'])
        weights = plan['eval_weights']
        wsum = sum(weights.get(n, 0) for n in val) or 1.0
        weighted = sum(val[n] * weights.get(n, 0) for n in val) / wsum
        cloze_acc, cloze_by_cat, cloze_details = self.cloze(model, plan['precision'])
        gen_summary, samples = self.generation_probe(model)
        record = {'time': stamp(), 'step': trainer.step, 'tokens_seen': trainer.tokens_seen,
                  'val_loss': round(weighted, 4), 'val_ppl': round(math.exp(weighted), 2), 'val_by_source': val,
                  'cloze_acc': cloze_acc, 'cloze_by_cat': cloze_by_cat, **gen_summary}
        if final:
            test = self.loss_by_source(model, 'test', plan['seq_len'], plan['precision'])
            record['test_by_source'] = test
            record['test_loss'] = round(sum(test[n] * weights.get(n, 0) for n in test) / wsum, 4)
        model.train()
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
        with (self.out_dir / 'eval_history.jsonl').open('a', encoding='utf-8') as f:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
        write_json(self.out_dir / 'samples' / f'step_{trainer.step:07d}.json', {'generation': samples, 'cloze': cloze_details})
        write_json(self.out_dir / 'samples' / 'latest.json', {'generation': samples, 'cloze': cloze_details})
        self.say('decision', 'report', {'action': 'eval_final' if final else 'eval',
                                        **{k: v for k, v in record.items() if k not in ('val_by_source', 'test_by_source')}})
        return record
