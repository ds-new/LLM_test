"""pass@k 能力探测：判断预训练后的模型是否适合直接做 RLVR（可验证奖励的强化学习）。

RLVR（GRPO 等）只能放大模型已经"偶尔做得对"的能力：一组采样全错或全对时组内优势为 0，没有梯度。
所以关键指标不只是 pass@1，而是：
  * pass@k（k = 1/4/8/16，用无偏估计 1 − C(n−c, k) / C(n, k)，n = 每题采样数，c = 答对次数）；
  * 有效分组比例：0 < c < n 的题目占比 —— 这些题在 GRPO 里才产生学习信号；
  * 贪心准确率与格式遵循率（输出里能否解析出答案）。

任务（全部可自动判分，少样本提示，基座模型不需要指令微调）：
  合成任务（固定随机种子，每个 100 题）：两位数加减、三位数加减、两位数×一位数、一元一次方程、
  数大小比较、中文应用题、英文应用题；
  GSM8K（AI-ModelScope/gsm8k test 前 200 题，3-shot 思维链，训练集样例做示范）。

对训练好的原模型和参数空间编辑后的模型（若有 model/edited.pt）各测一次，结果写入 RUN_DIR/passk/。
独立冒烟测试（CPU 小模型）：python -m decoder_from_scratch.a800.passk_probe --smoke
"""

import argparse
import io
import json
import math
import random
import re
import time
from urllib.request import Request, urlopen

import numpy as np
import torch

from .cleaning import normalize_zh_punct
from .common import RUN_DIR, now, stamp, write_json, write_text
from .weight_edit import batch_generate

NAME = 'passk_probe'
EOS = 3
NUM = re.compile(r'-?\d+(?:\.\d+)?')

DEFAULTS = {'enabled': True, 'n': 16, 'ks': [1, 4, 8, 16], 'problems': 100, 'gsm8k_problems': 200, 'gsm8k_shots': 3,
            'temperature': 0.8, 'top_p': 0.95, 'batch': 128}


# ---------------------------------------------------------------- 任务
def _fmt(q, a, lang):
    return (f'问：{q}\n答：{a}\n\n' if lang == 'zh' else f'Q: {q}\nA: {a}\n\n')


def synthetic_tasks(n_problems, seed=20261104):
    """每个任务返回 (名称, 题目列表[(prompt, answer)], 最大生成长度)。示范与测试题用不同随机数。"""
    rng = random.Random(seed)
    tasks = []

    def build(name, gen, lang, shots=4, max_new=16):
        demo = ''.join(_fmt(*gen(), lang) for _ in range(shots))
        probs = []
        seen = set()
        while len(probs) < n_problems:
            q, a = gen()
            if q in seen:
                continue
            seen.add(q)
            probs.append((demo + ('问：' if lang == 'zh' else 'Q: ') + q + ('\n答：' if lang == 'zh' else '\nA: '), str(a)))
        tasks.append({'name': name, 'problems': probs, 'max_new': max_new, 'kind': 'short'})

    def add2():
        a, b = rng.randint(10, 99), rng.randint(10, 99)
        if rng.random() < 0.5:
            return f'{a}+{b}=?', a + b
        a, b = max(a, b), min(a, b)
        return f'{a}-{b}=?', a - b

    def add3():
        a, b = rng.randint(100, 999), rng.randint(100, 999)
        if rng.random() < 0.5:
            return f'{a}+{b}=?', a + b
        a, b = max(a, b), min(a, b)
        return f'{a}-{b}=?', a - b

    def mul():
        a, b = rng.randint(11, 99), rng.randint(2, 9)
        return f'{a}×{b}=?', a * b

    def equation():
        x, a, b = rng.randint(1, 20), rng.randint(2, 9), rng.randint(1, 30)
        return f'解方程 {a}x+{b}={a * x + b}，x=?', x

    def compare():
        a, b = rng.sample(range(10, 1000), 2)
        return f'{a} 和 {b} 哪个大？', max(a, b)

    names = ['小明', '小红', '小刚', '小丽', '爸爸', '妈妈']
    things = ['苹果', '铅笔', '本子', '气球', '糖果', '贴纸']

    def zh_word():
        who, what = rng.choice(names), rng.choice(things)
        a, b = rng.randint(5, 40), rng.randint(2, 30)
        if rng.random() < 0.5:
            return f'{who}有{a}个{what}，又买了{b}个，现在一共有几个{what}？', a + b
        a, b = max(a, b), min(a, b)
        return f'{who}有{a}个{what}，送给同学{b}个，还剩几个{what}？', a - b

    en_names = ['Tom', 'Anna', 'Lily', 'Sam', 'Ben', 'Mia']
    en_things = ['apples', 'pencils', 'books', 'balloons', 'cookies', 'stickers']

    def en_word():
        who, what = rng.choice(en_names), rng.choice(en_things)
        a, b = rng.randint(5, 40), rng.randint(2, 30)
        if rng.random() < 0.5:
            return f'{who} has {a} {what} and buys {b} more. How many {what} does {who} have now?', a + b
        a, b = max(a, b), min(a, b)
        return f'{who} has {a} {what} and gives away {b}. How many {what} are left?', a - b

    build('两位数加减', add2, 'zh')
    build('三位数加减', add3, 'zh')
    build('两位数×一位数', mul, 'zh')
    build('一元一次方程', equation, 'zh')
    build('比较大小', compare, 'zh')
    build('中文应用题', zh_word, 'zh')
    build('英文应用题', en_word, 'en')
    return tasks


def _read_parquet(path_or_bytes):
    import pyarrow.parquet as pq
    src = io.BytesIO(path_or_bytes) if isinstance(path_or_bytes, bytes) else path_or_bytes
    t = pq.read_table(src)
    return [dict(zip(t.column_names, row)) for row in zip(*[t.column(c).to_pylist() for c in t.column_names])]


def load_gsm8k(cache_dir):
    """优先读本地缓存，否则从 ModelScope（AI-ModelScope/gsm8k）下载 test/train 并缓存。"""
    from .sources import download_url
    out = {}
    cache_dir.mkdir(parents=True, exist_ok=True)
    for split in ('test', 'train'):
        path = cache_dir / f'gsm8k_{split}.parquet'
        if not path.exists():
            url = download_url('AI-ModelScope/gsm8k', 'master', f'main/{split}-00000-of-00001.parquet')
            data = urlopen(Request(url, headers={'User-Agent': 'decoder-from-scratch'}), timeout=120).read()
            path.write_bytes(data)
        out[split] = _read_parquet(str(path))
    return out


def gsm8k_task(cache_dir, n_problems, shots):
    data = load_gsm8k(cache_dir)

    def clean(ans):
        reasoning, final = ans.split('####')
        reasoning = re.sub(r'<<[^>]*>>', '', reasoning).strip().replace('\n', ' ')
        return reasoning, final.strip().replace(',', '')

    demos = []
    for row in data['train'][:shots]:
        r, f = clean(row['answer'])
        demos.append(f"Question: {row['question']}\nAnswer: {r} The answer is {f}.\n\n")
    demo = ''.join(demos)
    probs = [(demo + f"Question: {row['question']}\nAnswer:", clean(row['answer'])[1])
             for row in data['test'][:n_problems]]
    return {'name': 'GSM8K（3-shot 思维链）', 'problems': probs, 'max_new': 160, 'kind': 'cot'}


# ---------------------------------------------------------------- 判分
def extract(text, kind):
    if kind == 'short':
        line = text.split('\n')[0]
        m = NUM.search(line)
        return m.group(0) if m else None
    text = re.split(r'\n\s*\n|Question:', text)[0]
    m = re.search(r'answer is\s*\$?\s*(-?[\d,]+(?:\.\d+)?)', text, re.I)
    if m:
        return m.group(1).replace(',', '')
    nums = NUM.findall(text.replace(',', ''))
    return nums[-1] if nums else None


def correct(pred, gold):
    if pred is None:
        return False
    try:
        return abs(float(pred) - float(gold)) < 1e-6
    except ValueError:
        return pred.strip() == gold.strip()


def pass_at_k(n, c, k):
    if n - c < k:
        return 1.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


# ---------------------------------------------------------------- 生成
def generate_grouped(model, items, max_new, device, precision, greedy, temperature, top_p, batch, seed):
    """items: [(key, token_ids)]，按 prompt 长度分组成等长 batch（模型不支持 padding）。返回 {key: completion_ids}。"""
    by_len = {}
    for key, ids in items:
        by_len.setdefault(len(ids), []).append((key, ids))
    out = {}
    for length, group in sorted(by_len.items()):
        for i in range(0, len(group), batch):
            chunk = group[i:i + batch]
            x = torch.tensor([ids for _, ids in chunk], dtype=torch.long)
            gen = batch_generate(model, x, max_new, device, precision, greedy=greedy, temperature=temperature,
                                 top_p=top_p, batch=batch, seed=seed + length * 1000 + i)
            for (key, _), row in zip(chunk, gen.tolist()):
                out[key] = row[:row.index(EOS)] if EOS in row else row
    return out


def probe_model(model, tok, tasks, cfg, device, precision, max_ctx):
    results = []
    for task in tasks:
        started = now()
        items_s, items_g, prompts = [], [], []
        for pi, (prompt, gold) in enumerate(task['problems']):
            ids = tok.encode(normalize_zh_punct(prompt), add_bos=True)
            ids = ids[-(max_ctx - task['max_new']):]           # 超长时保留结尾（题目本身），丢弃最早的示范
            prompts.append(ids)
            items_g.append(((pi, 0), ids))
            items_s += [((pi, j), ids) for j in range(cfg['n'])]
        greedy = generate_grouped(model, items_g, task['max_new'], device, precision, True, 1.0, 1.0, cfg['batch'], 0)
        sampled = generate_grouped(model, items_s, task['max_new'], device, precision, False, cfg['temperature'],
                                   cfg['top_p'], cfg['batch'], 17)
        rows, counts, parsed, greedy_ok = [], [], 0, 0
        for pi, (prompt, gold) in enumerate(task['problems']):
            g_text = tok.decode(greedy[(pi, 0)])
            g_ok = correct(extract(g_text, task['kind']), gold)
            greedy_ok += g_ok
            c = 0
            for j in range(cfg['n']):
                pred = extract(tok.decode(sampled[(pi, j)]), task['kind'])
                parsed += pred is not None
                c += correct(pred, gold)
            counts.append(c)
            if pi < 3:
                rows.append({'question': prompt.split('\n\n')[-1][:200], 'gold': gold, 'greedy': g_text[:200], 'greedy_ok': g_ok,
                             'correct_of_n': c})
        n = cfg['n']
        res = {'task': task['name'], 'problems': len(counts), 'n': n,
               'greedy_acc': round(greedy_ok / len(counts), 4),
               'pass_at_k': {str(k): round(float(np.mean([pass_at_k(n, c, k) for c in counts])), 4) for k in cfg['ks']},
               'mixed_groups': round(float(np.mean([0 < c < n for c in counts])), 4),
               'all_wrong': round(float(np.mean([c == 0 for c in counts])), 4),
               'all_right': round(float(np.mean([c == n for c in counts])), 4),
               'format_rate': round(parsed / (len(counts) * n), 4),
               'seconds': round(now() - started, 1), 'examples': rows}
        results.append(res)
    return results


# ---------------------------------------------------------------- 结论与报告
def rl_readiness(res):
    """按 GRPO 的学习信号判断：有效分组 ≥ 30% 适合直接 RL；pass@16 ≥ 10% 但有效分组少，先 SFT/课程再 RL；否则先 SFT。"""
    p16 = res['pass_at_k'].get('16', max(res['pass_at_k'].values()))
    if res['mixed_groups'] >= 0.3:
        return '适合直接 RLVR'
    if res['all_right'] >= 0.7:
        return '已基本掌握（太简单，RL 信号少）'
    if p16 >= 0.1:
        return '边缘：先 SFT 或降低难度后再 RL'
    return '不适合：先 SFT'


def render_report(result):
    L = ['# pass@k 能力探测（RLVR 适合度）', '',
         f'- 时间：{result["time"]}，用时 {result["minutes"]} 分钟；每题采样 n={result["config"]["n"]}，'
         f'温度 {result["config"]["temperature"]}，top-p {result["config"]["top_p"]}',
         '- 有效分组 = 0 < 答对次数 < n 的题目占比：GRPO 只从这些题得到梯度；全错 / 全对的组优势为 0。', '']
    for model_name, rows in result['models'].items():
        L += [f'## {model_name}', '',
              '| 任务 | 贪心 | ' + ' | '.join(f'pass@{k}' for k in result['config']['ks']) + ' | 有效分组 | 全错 | 全对 | 可解析 | 结论 |',
              '|' + '---|' * (len(result['config']['ks']) + 7)]
        for r in rows:
            L.append(f'| {r["task"]} | {r["greedy_acc"]:.0%} | ' + ' | '.join(f'{r["pass_at_k"][str(k)]:.0%}' for k in result['config']['ks'])
                     + f' | {r["mixed_groups"]:.0%} | {r["all_wrong"]:.0%} | {r["all_right"]:.0%} | {r["format_rate"]:.0%} | {rl_readiness(r)} |')
        L.append('')
    if result.get('errors'):
        L += ['## 未完成的任务', ''] + [f'- {e}' for e in result['errors']] + ['']
    first = next(iter(result['models'].values()))
    L += ['## 样例（原模型，每任务第 1 题，贪心）', '']
    for r in first:
        if r['examples']:
            e = r['examples'][0]
            L += [f'- **{r["task"]}**：{e["question"].strip()[:120]}', f'  - 标准答案 {e["gold"]}；模型：{e["greedy"].strip()[:160]}'
                  f'（{"对" if e["greedy_ok"] else "错"}，采样 {e["correct_of_n"]}/{result["config"]["n"]} 次答对）']
    L += ['', '判读：「适合直接 RLVR」的任务可直接作为 GRPO 的起点；「边缘」的任务先做少量 SFT（固定答案格式）或降低难度做课程；'
          '「不适合」说明模型几乎从不答对，RL 没有信号，需要先 SFT 注入能力。']
    return '\n'.join(L) + '\n'


def run_passk(config, bus, trainer, tok, device):
    cfg = dict(DEFAULTS, **config.get('passk', {}))
    out_dir = RUN_DIR / 'passk'
    out_dir.mkdir(parents=True, exist_ok=True)
    started = now()
    tasks, errors = synthetic_tasks(cfg['problems']), []
    try:
        tasks.append(gsm8k_task(out_dir, cfg['gsm8k_problems'], cfg['gsm8k_shots']))
    except Exception as exc:  # noqa: BLE001
        errors.append(f'GSM8K 加载失败：{type(exc).__name__}: {exc}')
    model = trainer.model
    precision = trainer.plan.get('precision', 'bf16')
    max_ctx = model.config.max_seq_len
    models = {}
    final = RUN_DIR / 'model' / 'final.pt'
    if final.exists():
        model.load_state_dict(torch.load(final, map_location='cpu', weights_only=True)['model'])
    if bus is not None:
        bus.post(NAME, 'decision', 'progress', {'action': 'start', 'tasks': [t['name'] for t in tasks], 'n': cfg['n']})
    models['原模型（final.pt）'] = probe_model(model, tok, tasks, cfg, device, precision, max_ctx)
    edited = RUN_DIR / 'model' / 'edited.pt'
    if edited.exists():
        model.load_state_dict(torch.load(edited, map_location='cpu', weights_only=True)['model'])
        models['参数空间编辑后（edited.pt）'] = probe_model(model, tok, tasks, cfg, device, precision, max_ctx)
        if final.exists():
            model.load_state_dict(torch.load(final, map_location='cpu', weights_only=True)['model'])
    result = {'time': stamp(), 'minutes': round((now() - started) / 60, 1), 'config': cfg, 'models': models, 'errors': errors}
    write_json(out_dir / 'result.json', result)
    write_text(out_dir / 'REPORT.md', render_report(result))
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    return result


# ---------------------------------------------------------------- 冒烟测试
def smoke():
    from ..config import ModelConfig
    from ..model import DecoderLM

    class Tok:
        def encode(self, text, add_bos=False):
            return ([2] if add_bos else []) + [5 + (ord(ch) % 250) for ch in text]

        def decode(self, ids):
            return ''.join(chr(48 + (int(i) % 10)) if int(i) % 7 else '\n' for i in ids)

    model = DecoderLM(ModelConfig(vocab_size=256, hidden_size=64, num_layers=2, num_heads=4, num_kv_heads=2,
                                  intermediate_size=128, max_seq_len=1024))

    class T:
        pass
    t = T()
    t.model, t.plan = model, {'precision': 'fp32'}
    cfg = {'passk': {'n': 4, 'ks': [1, 2, 4], 'problems': 5, 'gsm8k_problems': 3, 'batch': 32}}
    started = time.time()
    res = run_passk(cfg, None, t, Tok(), torch.device('cpu'))
    print('tasks', [r['task'] for r in next(iter(res['models'].values()))], 'errors', res['errors'], 'seconds', round(time.time() - started, 1))
    assert abs(pass_at_k(16, 4, 1) - 0.25) < 1e-9 and pass_at_k(16, 0, 16) == 0.0 and pass_at_k(16, 1, 16) == 1.0
    assert extract('82\n问：', 'short') == '82' and extract(' 5 + 3 = 8. The answer is 1,234.\n\nQuestion', 'cot') == '1234'
    print((RUN_DIR / 'passk' / 'REPORT.md').read_text()[:1800])


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--smoke', action='store_true')
    if p.parse_args().smoke:
        smoke()
