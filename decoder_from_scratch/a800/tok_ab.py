"""Tokenizer 对照实验（v1 vs v2）：同一份原文、同一模型结构，只换 tokenizer，比较每字节信息量与多位数运算能力。

角色分工沿用主流程的多智能体约定，消息写入 <RUN>/events.jsonl：
  * 数据智能体：从 ModelScope 取 v1 没用过的文件 → 与 v1 相同的清洗 / 去重 → 训练 tokenizer v2
    → 同一份文本分别用 v1、v2 编码；另加约 3% 合成算术文本（两组完全相同，评测题从训练里剔除）；
    验证集用 v1 的 val 文档（解码回原文后两边各自编码）。
  * 模型框架智能体：d768_l16（约 1.25 亿参数，GQA 12/4，SwiGLU 2048），两组结构、词表大小（32000）完全相同。
  * 训练智能体：两组各把自己那份 token 完整过一遍。原文相同，所以 v2 若压缩更好，token 和算力都更少。
  * 评估智能体：分来源 bits/字节（与分词方式无关，可直接比）、多位数运算贪心准确率、压缩率、罕见 token。
  * 结果分析：写 REPORT.md 与 results.json，给出"v2 是否值得"的结论。
阶段：sample → tokenizer → build → val → train_v1 → train_v2 → eval → report → done（可断点续跑）。
v2 tokenizer：Qwen2 / GPT-4 风格预切分（前导空格并入单词、数字逐位），字节级 BPE，32000 词表。
"""

import json
import math
import os
from array import array
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path
import random
import re
import sys
import threading
import time
import traceback

import numpy as np

from . import cleaning
from .common import (RUN_DIR, Bus, FatalError, StopRequested, control_signal, heartbeat, read_json, stamp, write_json,
                     write_text)
from .data_agent import _iter_texts
from .sources import SOURCES

RUN = Path(os.environ.get('AB_RUN_DIR', '/root/autodl-tmp/a800_tok_ab'))
SCRATCH = Path(os.environ.get('AB_SCRATCH', '/root/a800_scratch/tok_ab'))
V1_RUN = Path('/root/autodl-tmp/a800_pretrain_v1')
V1_TOK = V1_RUN / 'tokenizer' / 'tokenizer_hf.json'
V2_TOK = RUN / 'tokenizer_v2.json'
BOS, EOS = 2, 3
MODEL_VOCAB = 32000
SPECIAL = ['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]']
V2_PATTERN = r"""(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"""
PHASES = ['sample', 'tokenizer', 'build', 'val', 'train_v1', 'train_v2', 'eval', 'report', 'done']

CFG = {
    'weights': {'zh_cci3': 0.28, 'zh_fwedu': 0.14, 'zh_industry': 0.13, 'en_fwedu': 0.18, 'en_cosmo': 0.07,
                'code': 0.12, 'math': 0.08},
    'train_tokens_v1': 2_500_000_000,    # 按 v1 token 计的原文量；v2 编码同一份原文
    'arith_fraction': 0.03,
    'tok_sample_chars': 600_000_000,
    'vocab': 32000,
    'max_token_length': 32,
    'model': dict(hidden_size=768, num_layers=16, num_heads=12, num_kv_heads=4, intermediate_size=2048),
    'seq_len': 1024, 'micro': 64, 'accum': 4, 'peak_lr': 1e-3, 'warmup': 300, 'decay_fraction': 0.2,
    'min_ratio': 0.1, 'wd': 0.1, 'beta2': 0.95, 'clip': 1.0, 'seed': 1234, 'compile': True,
    'val_bytes_per_source': 1_500_000,
    'arith_eval_per_cat': 200,
    'workers': 12,
}
CFG.update(json.loads(os.environ.get('AB_CFG', '{}')))

if Path(RUN_DIR) != RUN:
    raise RuntimeError(f'common.RUN_DIR={RUN_DIR} 与 AB_RUN_DIR={RUN} 不一致：请用 run_tok_ab.sh 启动（会同时设置 A8_RUN_DIR）')


def read_jsonl(path):
    with open(path, encoding='utf-8') as f:
        return [json.loads(l) for l in f if l.strip()]


class keepalive:
    """CPU 长阶段（下载、tokenizer 训练）期间定时写心跳，避免被守护脚本当成卡死。"""

    def __init__(self, note, every=300):
        self.note, self.every, self.stop = note, every, threading.Event()

    def __enter__(self):
        heartbeat(self.note)

        def loop():
            while not self.stop.wait(self.every):
                heartbeat(self.note)
        threading.Thread(target=loop, daemon=True).start()
        return self

    def __exit__(self, *exc):
        self.stop.set()


def check_control():
    if control_signal() == 'stop':
        raise StopRequested('STOP file present')


# ------------------------------------------------------------------ 数据智能体：单文件任务（子进程） ----------
def make_job(spec, f, **extra):
    return {'source': spec['name'], 'kind': spec['kind'], 'format': spec['format'], 'fields': spec['fields'],
            'url': f['url'], 'path': f['path'], 'size': f['size'], 'index': f['index'],
            'download_path': str(SCRATCH / 'downloads' / spec['name'] / f'{f["index"]:05d}.parquet'), **extra}


def clean_iter(job, stats):
    seen_docs, seen_paras = set(), set()
    for raw in _iter_texts(job):
        stats['rows'] += 1
        text, reason = cleaning.clean_text(raw, job['kind'])
        if reason:
            stats['reject_' + reason] += 1
            continue
        h = cleaning.doc_hash64(text)
        if h in seen_docs:
            stats['reject_duplicate'] += 1
            continue
        paras = cleaning.paragraph_hashes(text)
        if len(paras) >= 3 and sum(p in seen_paras for p in paras) * 2 >= len(paras):
            stats['reject_near_duplicate'] += 1
            continue
        seen_docs.add(h)
        seen_paras.update(paras)
        yield text


def sample_job(job):
    out = Path(job['out'])
    out.parent.mkdir(parents=True, exist_ok=True)
    stats, chars = Counter(), 0
    tmp = Path(str(out) + '.partial')
    with tmp.open('w', encoding='utf-8') as f:
        for text in clean_iter(job, stats):
            f.write(json.dumps({'text': text}, ensure_ascii=False) + '\n')
            chars += len(text)
            if chars >= job['char_budget']:
                break
    os.replace(tmp, out)
    return {'source': job['source'], 'index': job['index'], 'chars': chars}


_ENC = {}


def encoders(v1_path, v2_path):
    if not _ENC:
        os.environ['RAYON_NUM_THREADS'] = '1'
        os.environ['TOKENIZERS_PARALLELISM'] = 'false'
        from tokenizers import Tokenizer
        _ENC['v1'] = Tokenizer.from_file(v1_path)
        _ENC['v2'] = Tokenizer.from_file(v2_path)
    return _ENC


def build_job(job):
    enc = encoders(job['v1_tok'], job['v2_tok'])
    out = Path(job['out_dir'])
    out.mkdir(parents=True, exist_ok=True)
    stem = f'{job["index"]:05d}'
    bufs = {k: array('H') for k in ('v1', 'v2')}
    handles = {k: (out / f'{stem}.{k}.bin.partial').open('wb') for k in bufs}
    stats, cnt, pending = Counter(), Counter(), []

    def flush():
        for k in bufs:
            for e in enc[k].encode_batch(pending, add_special_tokens=False):
                bufs[k].append(BOS)
                bufs[k].extend(e.ids)
                bufs[k].append(EOS)
                cnt['tokens_' + k] += len(e.ids) + 2
            if len(bufs[k]) > (1 << 20):
                bufs[k].tofile(handles[k])
                del bufs[k][:]
        cnt['docs'] += len(pending)
        cnt['chars'] += sum(len(t) for t in pending)
        cnt['bytes'] += sum(len(t.encode('utf-8')) for t in pending)
        pending.clear()

    try:
        for text in clean_iter(job, stats):
            pending.append(text)
            if len(pending) >= 256:
                flush()
                if cnt['tokens_v1'] >= job['token_cap']:
                    break
        if pending:
            flush()
        for k in bufs:
            bufs[k].tofile(handles[k])
    finally:
        for h in handles.values():
            h.close()
    for k in bufs:
        os.replace(out / f'{stem}.{k}.bin.partial', out / f'{stem}.{k}.bin')
    if job['format'] == 'parquet':
        Path(job['download_path']).unlink(missing_ok=True)
    result = {'source': job['source'], 'index': job['index'], **dict(cnt), 'rows': stats['rows'],
              'accept_rate': round(cnt['docs'] / max(1, stats['rows']), 3)}
    write_json(job['receipt'], result)
    return result


def run_pool(fn, jobs, workers, bus, phase):
    results, attempts, pending, futs = [], Counter(), list(jobs), {}
    with ProcessPoolExecutor(max_workers=workers) as ex:
        while pending or futs:
            check_control()
            while pending and len(futs) < workers:
                j = pending.pop(0)
                futs[ex.submit(fn, j)] = j
            done, _ = wait(list(futs), timeout=60, return_when=FIRST_COMPLETED)
            for fut in done:
                j = futs.pop(fut)
                try:
                    results.append(fut.result())
                except Exception as exc:  # noqa: BLE001
                    key = (j['source'], j['index'])
                    attempts[key] += 1
                    if attempts[key] < 3:
                        pending.append(j)
                    else:
                        bus.post('data', 'decision', 'report', {'action': 'file_skipped', 'phase': phase, 'source': j['source'],
                                                                'index': j['index'], 'error': f'{type(exc).__name__}: {str(exc)[:300]}'})
            heartbeat(f'{phase} {len(results)}/{len(jobs)}')
    return results


# ------------------------------------------------------------------ 合成算术 ----------
OPS = {'+': lambda a, b: a + b, '-': lambda a, b: a - b, '×': lambda a, b: a * b}
EVAL_CATS = {'add2': ('+', 2, 2), 'sub2': ('-', 2, 2), 'add3': ('+', 3, 3), 'sub3': ('-', 3, 3), 'add4': ('+', 4, 4),
             'mul2x1': ('×', 2, 1), 'mul3x1': ('×', 3, 1), 'mul2x2': ('×', 2, 2)}
ZH_WORD = {'+': '加', '-': '减', '×': '乘以'}
EN_WORD = {'+': 'plus', '-': 'minus', '×': 'times'}
ZH_TPL = ['{a}{o}{b}={c}', '计算：{a} {o} {b} = {c}', '{a}{w}{b}等于{c}。', '问：{a}{w}{b}是多少？答：{c}']
EN_TPL = ['What is {a} {we} {b}? The answer is {c}.', '{a} {we} {b} equals {c}.', '{a}{o}{b}={c}']


def rand_n(rng, digits):
    return rng.randint(0 if digits == 1 else 10 ** (digits - 1), 10 ** digits - 1)


def make_problem(rng, op, da, db):
    a, b = rand_n(rng, da), rand_n(rng, db)
    if op == '-' and b > a:
        a, b = b, a
    return (a, op, b, OPS[op](a, b))


def eval_set():
    rng = random.Random(777)
    problems, shots = {}, {}
    for cat, (op, da, db) in EVAL_CATS.items():
        seen = set()
        while len(seen) < CFG['arith_eval_per_cat'] + 3:
            seen.add(make_problem(rng, op, da, db))
        items = sorted(seen)
        rng.shuffle(items)
        shots[cat], problems[cat] = items[:3], items[3:]
    return {'problems': problems, 'shots': shots}


def arith_doc(rng, banned):
    lines = []
    for _ in range(rng.randint(15, 40)):
        op = rng.choices(['+', '-', '×'], [0.4, 0.3, 0.3])[0]
        da, db = (rng.randint(1, 4), rng.randint(1, 2)) if op == '×' else (rng.randint(1, 5), rng.randint(1, 5))
        a, o, b, c = make_problem(rng, op, da, db)
        if (a, o, b) in banned:
            continue
        r = rng.random()
        tpl = '{a} {o} {b} = {c}' if r < 0.5 else rng.choice(ZH_TPL if r < 0.75 else EN_TPL)
        lines.append(tpl.format(a=a, o=o, b=b, c=c, w=ZH_WORD[o], we=EN_WORD[o]))
    return '\n'.join(lines)


# ------------------------------------------------------------------ 阶段 ----------
def plan_files():
    fl = read_json(V1_RUN / 'data' / 'file_lists.json')
    done = {}
    for p in (V1_RUN / 'data' / 'receipts').glob('*/*.json'):
        r = read_json(p)
        done.setdefault(r['source'], {})[r['index']] = r['stats']
    specs = {s['name']: s for s in SOURCES}
    plan = {}
    for name, w in CFG['weights'].items():
        files = fl[name]
        size = {f['index']: f['size'] for f in files}
        d = done.get(name, {})
        base = max(1, sum(size[i] for i in d if i in size))
        tpb = sum(s.get('tokens_train', 0) for s in d.values()) / base
        cpb = sum(s.get('chars_train', 0) for s in d.values()) / base
        target = int(CFG['train_tokens_v1'] * w)
        chosen, est = [], 0.0
        for f in files:
            if f['index'] in d:
                continue
            chosen.append(f)
            est += f['size'] * tpb
            if est >= target * 1.1:
                break
        spec = {k: specs[name][k] for k in ('name', 'kind', 'format', 'fields')}
        plan[name] = {'spec': spec, 'files': chosen, 'target': target, 'tpb': tpb, 'cpb': cpb}
    return plan


def phase_sample(bus):
    plan = plan_files()
    write_json(RUN / 'data' / 'plan.json', plan)
    jobs = []
    for name, p in plan.items():
        remaining = int(CFG['tok_sample_chars'] * CFG['weights'][name])
        for f in p['files']:
            if remaining <= 0:
                break
            budget = int(min(remaining, f['size'] * p['cpb'] * 1.2 + 1))
            out = RUN / 'data' / 'tok_sample' / f'{name}_{f["index"]:05d}.jsonl'
            if not out.exists():
                jobs.append(make_job(p['spec'], f, out=str(out), char_budget=budget))
            remaining -= budget
    bus.post('decision', 'data', 'directive', {'task': 'tokenizer_sample', 'files': len(jobs), 'chars': CFG['tok_sample_chars']})
    with keepalive('sample'):
        run_pool(sample_job, jobs, CFG['workers'], bus, 'sample')
    chars = sum(sum(len(json.loads(l)['text']) for l in open(p, encoding='utf-8'))
                for p in (RUN / 'data' / 'tok_sample').glob('*.jsonl'))
    bus.post('data', 'decision', 'report', {'action': 'tokenizer_sample_done', 'chars': chars})
    return {'chars': chars, 'files': {n: len(p['files']) for n, p in plan.items()}}


def phase_tokenizer(bus):
    from tokenizers import Regex, Tokenizer, decoders, models, pre_tokenizers, trainers
    files = sorted((RUN / 'data' / 'tok_sample').glob('*.jsonl'))

    def texts():
        for p in files:
            with open(p, encoding='utf-8') as f:
                for line in f:
                    yield json.loads(line)['text']
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.Sequence([pre_tokenizers.Split(Regex(V2_PATTERN), behavior='isolated'),
                                                 pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)])
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=CFG['vocab'], min_frequency=2, special_tokens=SPECIAL, show_progress=False,
                                  initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
                                  max_token_length=CFG['max_token_length'])
    started = time.time()
    with keepalive('tokenizer'):
        tok.train_from_iterator(texts(), trainer=trainer)
    if tok.token_to_id('[CLS]') != BOS or tok.token_to_id('[SEP]') != EOS:
        raise FatalError('v2 tokenizer 特殊 token 编号与 v1 不一致')
    tok.save(str(V2_TOK))
    probes = ['12345 + 678 = 13023', 'Hello world, this is a test of the tokenizer.', '人工智能的发展非常迅速。',
              'def foo(x):\n    return x * 2\n']
    examples = {s: tok.encode(s).tokens for s in probes}
    roundtrip = all(tok.decode(tok.encode(s).ids) == s for s in probes)
    if not roundtrip:
        raise FatalError('v2 tokenizer 解码往返不一致')
    info = {'minutes': round((time.time() - started) / 60, 1), 'vocab': tok.get_vocab_size(), 'examples': examples}
    bus.post('data', 'decision', 'report', {'action': 'tokenizer_v2', **info})
    return info


def phase_build(bus):
    plan = read_json(RUN / 'data' / 'plan.json')
    rec = RUN / 'data' / 'receipts'
    jobs = []
    for name, p in plan.items():
        est = 0.0
        for f in p['files']:
            cap = int((p['target'] - est) * 1.1)
            est += f['size'] * p['tpb']
            receipt = rec / name / f'{f["index"]:05d}.json'
            if cap <= 0 or receipt.exists():
                continue
            jobs.append(make_job(p['spec'], f, out_dir=str(RUN / 'data' / 'tokens' / name), token_cap=cap,
                                 v1_tok=str(V1_TOK), v2_tok=str(V2_TOK), receipt=str(receipt)))
    bus.post('decision', 'data', 'directive', {'task': 'build', 'jobs': len(jobs)})
    with keepalive('build'):
        run_pool(build_job, jobs, CFG['workers'], bus, 'build')
        arith = build_arith()
    sources, totals = {}, {}
    for name in list(plan) + ['arith']:
        sources[name] = {}
        for k in ('v1', 'v2'):
            shards = []
            for p in sorted((RUN / 'data' / 'tokens' / name).glob(f'*.{k}.bin')):
                n = p.stat().st_size // 2
                if n > CFG['seq_len'] + 1:
                    shards.append({'path': str(p), 'tokens': n})
            sources[name][k] = shards
        recs = [read_json(p) for p in (rec / name).glob('*.json')] if name != 'arith' else [arith]
        totals[name] = {k: sum(r.get(k, 0) for r in recs) for k in ('tokens_v1', 'tokens_v2', 'docs', 'chars', 'bytes')}
    write_json(RUN / 'data' / 'manifest.json', {'sources': sources, 'totals': totals})
    t1 = sum(t['tokens_v1'] for t in totals.values())
    t2 = sum(t['tokens_v2'] for t in totals.values())
    bus.post('data', 'decision', 'report', {'action': 'build_done', 'tokens_v1': t1, 'tokens_v2': t2,
                                            'v2_over_v1': round(t2 / max(1, t1), 4)})
    return {'tokens_v1': t1, 'tokens_v2': t2, 'totals': totals}


def build_arith():
    out = RUN / 'data' / 'tokens' / 'arith'
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'stats.json').exists():
        return read_json(out / 'stats.json')
    from tokenizers import Tokenizer
    enc = {'v1': Tokenizer.from_file(str(V1_TOK)), 'v2': Tokenizer.from_file(str(V2_TOK))}
    ev = eval_set()
    banned = {(a, o, b) for group in (ev['problems'], ev['shots']) for items in group.values() for a, o, b, c in items}
    rng = random.Random(4242)
    target = int(CFG['train_tokens_v1'] * CFG['arith_fraction'])
    bufs, cnt = {k: array('H') for k in enc}, Counter()
    while cnt['tokens_v1'] < target:
        docs = [arith_doc(rng, banned) for _ in range(512)]
        for k, e in enc.items():
            for x in e.encode_batch(docs, add_special_tokens=False):
                bufs[k].append(BOS)
                bufs[k].extend(x.ids)
                bufs[k].append(EOS)
                cnt['tokens_' + k] += len(x.ids) + 2
        cnt['docs'] += len(docs)
        cnt['chars'] += sum(len(d) for d in docs)
        cnt['bytes'] += sum(len(d.encode('utf-8')) for d in docs)
    for k in bufs:
        with open(out / f'00000.{k}.bin', 'wb') as fh:
            bufs[k].tofile(fh)
    write_json(RUN / 'data' / 'arith_eval.json', ev)
    write_json(out / 'stats.json', dict(cnt))
    return dict(cnt)


def phase_val(bus):
    from tokenizers import Tokenizer
    v1 = Tokenizer.from_file(str(V1_TOK))
    man = read_json(V1_RUN / 'data' / 'tokens' / 'manifest.json')
    docs, mismatch, checked = [], 0, 0
    for name in CFG['weights']:
        got = 0
        for shard in man['sources'][name]['val']:
            arr = np.fromfile(V1_RUN / shard['path'], dtype='<u2')
            starts = np.where(arr == BOS)[0]
            for i, s in enumerate(starts):
                e = starts[i + 1] if i + 1 < len(starts) else len(arr)
                ids = arr[s + 1:e].tolist()
                if ids and ids[-1] == EOS:
                    ids = ids[:-1]
                if not ids:
                    continue
                text = v1.decode(ids, skip_special_tokens=True)
                if checked < 50:
                    checked += 1
                    mismatch += int(v1.encode(text, add_special_tokens=False).ids != ids)
                docs.append({'source': name, 'text': text})
                got += len(text.encode('utf-8'))
                if got >= CFG['val_bytes_per_source']:
                    break
            if got >= CFG['val_bytes_per_source']:
                break
    with open(RUN / 'data' / 'val_docs.jsonl', 'w', encoding='utf-8') as f:
        for d in docs:
            f.write(json.dumps(d, ensure_ascii=False) + '\n')
    info = {'docs': len(docs), 'roundtrip_checked': checked, 'roundtrip_mismatch': mismatch}
    bus.post('evaluator', 'decision', 'report', {'action': 'val_docs', **info})
    return info


def model_config():
    from ..config import ModelConfig
    return ModelConfig(vocab_size=MODEL_VOCAB, max_seq_len=2048, dropout=0.0, tie_embeddings=True,
                       attention_backend='sdpa', **CFG['model'])


def train_run(tag, bus):
    import torch
    from ..model import DecoderLM
    from .trainer_agent import MixtureSampler, autocast, wsd_lr
    man = read_json(RUN / 'data' / 'manifest.json')
    manifest = {'sources': {n: {'train': e[tag]} for n, e in man['sources'].items() if e[tag]}}
    tokens = {n: sum(s['tokens'] for s in e['train']) for n, e in manifest['sources'].items()}
    total_tokens = sum(tokens.values())
    weights = {n: t / total_tokens for n, t in tokens.items()}
    T, micro, accum = CFG['seq_len'], CFG['micro'], CFG['accum']
    tps = micro * accum * T
    total_steps = max(1, total_tokens // tps)
    out = RUN / 'runs' / tag
    out.mkdir(parents=True, exist_ok=True)
    SCRATCH.mkdir(parents=True, exist_ok=True)
    ckpt = SCRATCH / f'{tag}_latest.pt'
    dev = torch.device('cuda')
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.manual_seed(CFG['seed'])
    model = DecoderLM(model_config()).to(dev)
    decay = [p for p in model.parameters() if p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.dim() < 2]
    opt = torch.optim.AdamW([{'params': decay, 'weight_decay': CFG['wd']}, {'params': no_decay, 'weight_decay': 0.0}],
                            lr=CFG['peak_lr'], betas=(0.9, CFG['beta2']), eps=1e-8, fused=True)
    sampler = MixtureSampler(manifest, weights, T, CFG['seed'])
    step, t_train = 0, 0.0
    if ckpt.exists():
        ck = torch.load(ckpt, map_location='cpu', weights_only=False)
        model.load_state_dict(ck['model'])
        opt.load_state_dict(ck['opt'])
        sampler.load_state_dict(ck['sampler'])
        step, t_train = ck['step'], ck['t']
        torch.set_rng_state(ck['rng'])
    cmodel = torch.compile(model) if CFG['compile'] else model
    bus.post('trainer', 'decision', 'report', {'action': 'resumed' if step else 'initialized', 'run': tag, 'step': step,
                                               'total_steps': total_steps, 'tokens': total_tokens,
                                               'params': model.count_parameters()})

    def save():
        tmp = ckpt.with_name(ckpt.name + '.tmp')
        torch.save({'model': model.state_dict(), 'opt': opt.state_dict(), 'sampler': sampler.state_dict(), 'step': step,
                    't': t_train, 'rng': torch.get_rng_state()}, tmp)
        os.replace(tmp, ckpt)

    log = (out / 'train_log.jsonl').open('a', encoding='utf-8')
    last_ck, bad, losses = time.time(), 0, []
    prefetch = sampler.batch(micro).pin_memory()
    try:
        while step < total_steps:
            check_control()
            t0 = time.perf_counter()
            lr = wsd_lr(step, total_steps, CFG['peak_lr'], CFG['warmup'], CFG['decay_fraction'], CFG['min_ratio'])
            for g in opt.param_groups:
                g['lr'] = lr
            tot = torch.zeros((), device=dev)
            for _ in range(accum):
                b = prefetch.to(dev, non_blocking=True)
                prefetch = sampler.batch(micro).pin_memory()
                with autocast(dev):
                    loss = cmodel(b[:, :-1], targets=b[:, 1:]).loss / accum
                loss.backward()
                tot += loss.detach()
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), CFG['clip'])
            lv, gv = float(tot), float(gn)
            if not (math.isfinite(lv) and math.isfinite(gv)):
                opt.zero_grad(set_to_none=True)
                bad += 1
                if bad >= 20:
                    raise FatalError(f'{tag}: 连续出现非有限 loss/梯度')
                continue
            bad = 0
            opt.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            dt = time.perf_counter() - t0
            t_train += dt
            losses.append(lv)
            if step % 50 == 0 or step == total_steps:
                log.write(json.dumps({'time': stamp(), 'step': step, 'total': total_steps, 'loss': round(lv, 4), 'lr': lr,
                                      'grad_norm': round(gv, 3), 'tok_s': round(tps / dt)}) + '\n')
                log.flush()
                heartbeat(f'{tag} step {step}/{total_steps}')
            if time.time() - last_ck > 1800:
                save()
                last_ck = time.time()
    finally:
        log.close()
    save()
    torch.save({'model': model.state_dict(), 'config': model_config().to_dict(), 'tag': tag, 'steps': step,
                'tokens': step * tps}, out / 'final.pt')
    info = {'steps': step, 'tokens': step * tps, 'train_hours': round(t_train / 3600, 2),
            'tok_s': round(step * tps / max(1e-6, t_train)), 'final_loss': round(float(np.mean(losses[-50:])), 4) if losses else None}
    bus.post('trainer', 'decision', 'report', {'action': 'run_done', 'run': tag, **info})
    return info


def eval_model(tag, bus):
    import torch
    from ..model import DecoderLM
    from .trainer_agent import autocast
    from tokenizers import Tokenizer
    dev = torch.device('cuda')
    ck = torch.load(RUN / 'runs' / tag / 'final.pt', map_location='cpu', weights_only=False)
    model = DecoderLM(model_config()).to(dev)
    model.load_state_dict(ck['model'])
    model.eval()
    tok = Tokenizer.from_file(str(V1_TOK if tag == 'v1' else V2_TOK))
    T, ln2 = CFG['seq_len'], math.log(2)
    by = {}
    for d in read_jsonl(RUN / 'data' / 'val_docs.jsonl'):
        by.setdefault(d['source'], []).append(d['text'])
    bpb = {}
    with torch.no_grad():
        for src, texts in by.items():
            heartbeat(f'eval {tag} {src}')
            stream, text_tokens = [], 0
            for e in tok.encode_batch(texts, add_special_tokens=False):
                stream += [BOS] + e.ids + [EOS]
                text_tokens += len(e.ids)
            arr = torch.tensor(stream, dtype=torch.long)
            wins = [arr[s:s + T + 1] for s in range(0, len(arr) - 1, T)]
            full = [w for w in wins if len(w) == T + 1]
            groups = [torch.stack(full[i:i + 16]) for i in range(0, len(full), 16)] + \
                     [w.unsqueeze(0) for w in wins if 2 <= len(w) < T + 1]
            nll, counted = 0.0, 0
            for b in groups:
                b = b.to(dev)
                with autocast(dev):
                    logits = model(b[:, :-1]).logits
                tgt = b[:, 1:]
                tok_nll = -torch.log_softmax(logits.float(), -1).gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
                mask = (tgt != BOS) & (tgt != EOS)
                nll += float((tok_nll * mask).sum())
                counted += int(mask.sum())
            nbytes = sum(len(t.encode('utf-8')) for t in texts)
            bpb[src] = {'bpb': round(nll / ln2 / nbytes, 4), 'bytes': nbytes, 'tokens': text_tokens,
                        'bytes_per_token': round(nbytes / max(1, text_tokens), 3), 'nats_per_token': round(nll / max(1, counted), 4)}
        ev = read_json(RUN / 'data' / 'arith_eval.json')
        arith, samples = {}, {}
        for cat, probs in ev['problems'].items():
            heartbeat(f'eval {tag} arith {cat}')
            shots = ''.join(f'{a} {o} {b} = {c}\n' for a, o, b, c in ev['shots'][cat])
            correct = 0
            for i, (a, o, b, c) in enumerate(probs):
                ids = [BOS] + tok.encode(shots + f'{a} {o} {b} = ', add_special_tokens=False).ids
                x = torch.tensor([ids], device=dev)
                gen = []
                for _ in range(12):
                    with autocast(dev):
                        nxt = int(model(x, last_logits_only=True).logits[0, -1].argmax())
                    if nxt == EOS:
                        break
                    gen.append(nxt)
                    if '\n' in tok.decode(gen):
                        break
                    x = torch.cat([x, torch.tensor([[nxt]], device=dev)], 1)
                text = tok.decode(gen)
                m = re.match(r'\s*(-?\d+)', text)
                ok = bool(m) and int(m.group(1)) == c
                correct += int(ok)
                if i < 3:
                    samples.setdefault(cat, []).append(f'{a} {o} {b} = {text.strip()}（应为 {c}）')
            arith[cat] = round(correct / len(probs), 4)
    w = CFG['weights']
    weighted = sum(bpb[s]['bpb'] * w[s] for s in w if s in bpb) / sum(w[s] for s in w if s in bpb)
    counts = token_counts(tag)
    info = {'bpb': bpb, 'bpb_weighted': round(weighted, 4), 'arith': arith,
            'arith_mean': round(float(np.mean(list(arith.values()))), 4), 'arith_samples': samples, 'rare_tokens': counts}
    bus.post('evaluator', 'decision', 'report', {'action': 'eval', 'run': tag, 'bpb_weighted': info['bpb_weighted'],
                                                 'arith_mean': info['arith_mean'], 'rare_tokens': counts})
    return info


def token_counts(tag):
    man = read_json(RUN / 'data' / 'manifest.json')
    cnt = np.zeros(MODEL_VOCAB, dtype=np.int64)
    for e in man['sources'].values():
        for s in e[tag]:
            arr = np.memmap(s['path'], dtype='<u2', mode='r')
            for i in range(0, len(arr), 1 << 27):
                cnt += np.bincount(arr[i:i + (1 << 27)], minlength=MODEL_VOCAB)
    size = MODEL_VOCAB if tag == 'v1' else CFG['vocab']
    c = cnt[len(SPECIAL):size]
    return {'zero': int((c == 0).sum()), 'lt100': int((c < 100).sum()), 'lt1000': int((c < 1000).sum())}


def phase_eval(bus):
    res = {tag: eval_model(tag, bus) for tag in ('v1', 'v2')}
    write_json(RUN / 'results.json', res)
    return {tag: {'bpb_weighted': r['bpb_weighted'], 'arith_mean': r['arith_mean']} for tag, r in res.items()}


def phase_report(bus):
    st = read_json(RUN / 'state.json', {})
    res = read_json(RUN / 'results.json')
    man = read_json(RUN / 'data' / 'manifest.json')
    tok_info = (st.get('results') or {}).get('tokenizer', {})
    tr = {t: (st.get('results') or {}).get('train_' + t, {}) for t in ('v1', 'v2')}
    r1, r2 = res['v1'], res['v2']
    gain = (r1['bpb_weighted'] - r2['bpb_weighted']) / r1['bpb_weighted']
    L = ['# Tokenizer 对照实验报告（v1 vs v2）', '', f'- 生成时间：{stamp()}',
         '- 设计：同一份原文（来自 v1 没用过的文件，含约 3% 合成算术）、同一模型（d768_l16，约 1.25 亿参数，词表 32000），只换 tokenizer；'
         '每组把自己那份 token 完整过一遍，超参完全相同。', '',
         '## 结论', '']
    arith_d = r2['arith_mean'] - r1['arith_mean']
    if gain >= 0.01 or arith_d >= 0.1:
        verdict = '**v2 明显更好，建议正式预训练改用 v2（路线 B）**'
    elif gain <= -0.01 and arith_d <= 0:
        verdict = '**v2 没有优势，保留 v1，按路线 A 继续**'
    else:
        verdict = '**差异不大：路线 A 优先，v2 留作下一代模型时再换**'
    L += [f'- {verdict}',
          f'- 加权 bits/字节：v1 {r1["bpb_weighted"]} → v2 {r2["bpb_weighted"]}（{"低" if gain > 0 else "高"} {abs(gain):.1%}）',
          f'- 多位数运算平均准确率：v1 {r1["arith_mean"]:.1%} → v2 {r2["arith_mean"]:.1%}',
          f'- 训练 token：v1 {tr["v1"].get("tokens", 0) / 1e9:.2f}B → v2 {tr["v2"].get("tokens", 0) / 1e9:.2f}B；'
          f'训练耗时 {tr["v1"].get("train_hours")} h → {tr["v2"].get("train_hours")} h', '',
          '## 压缩率（同一份训练原文，每 token 平均字节数，越大越省）', '', '| 来源 | v1 | v2 | v2/v1 |', '|---|---|---|---|']
    for name, t in man['totals'].items():
        b1 = t['bytes'] / max(1, t['tokens_v1'] - 2 * t['docs'])
        b2 = t['bytes'] / max(1, t['tokens_v2'] - 2 * t['docs'])
        L.append(f'| {name} | {b1:.2f} | {b2:.2f} | {b2 / b1:.2f} |')
    L += ['', '## 分来源 bits/字节（验证集，越低越好）', '', '| 来源 | v1 | v2 | 变化 |', '|---|---|---|---|']
    for s in r1['bpb']:
        a, b = r1['bpb'][s]['bpb'], r2['bpb'][s]['bpb']
        L.append(f'| {s} | {a} | {b} | {(b - a) / a:+.1%} |')
    L += ['', '## 多位数运算（3-shot，贪心，每类 ' + str(CFG['arith_eval_per_cat']) + ' 题）', '',
          '| 类别 | v1 | v2 |', '|---|---|---|']
    for cat in r1['arith']:
        L.append(f'| {cat} | {r1["arith"][cat]:.1%} | {r2["arith"][cat]:.1%} |')
    L += ['', '## 罕见 token（训练数据中出现次数，不含 5 个特殊 token）', '', '| | 0 次 | <100 次 | <1000 次 |', '|---|---|---|---|',
          f'| v1 | {r1["rare_tokens"]["zero"]} | {r1["rare_tokens"]["lt100"]} | {r1["rare_tokens"]["lt1000"]} |',
          f'| v2 | {r2["rare_tokens"]["zero"]} | {r2["rare_tokens"]["lt100"]} | {r2["rare_tokens"]["lt1000"]} |', '',
          '## v2 切分示例', '']
    for s, toks in (tok_info.get('examples') or {}).items():
        L.append(f'- {s!r} → {len(toks)} 个 token')
    L += ['', '## 运算样例', '']
    for tag, r in (('v1', r1), ('v2', r2)):
        for cat, items in r['arith_samples'].items():
            L.append(f'- {tag} {cat}：' + '；'.join(items))
    write_text(RUN / 'REPORT.md', '\n'.join(L) + '\n')
    summary = {'verdict': verdict, 'bpb_gain': round(gain, 4), 'arith_v1': r1['arith_mean'], 'arith_v2': r2['arith_mean']}
    bus.post('analyst', 'decision', 'report', {'action': 'report', **summary})
    return summary


PHASE_FN = {'sample': phase_sample, 'tokenizer': phase_tokenizer, 'build': phase_build, 'val': phase_val,
            'train_v1': lambda bus: train_run('v1', bus), 'train_v2': lambda bus: train_run('v2', bus),
            'eval': phase_eval, 'report': phase_report}


def run():
    RUN.mkdir(parents=True, exist_ok=True)
    bus = Bus(RUN)
    state = read_json(RUN / 'state.json', {}) or {'phase': 'sample', 'failures': 0, 'started': stamp()}
    try:
        while state['phase'] != 'done':
            check_control()
            phase = state['phase']
            heartbeat(phase)
            bus.post('decision', 'all', 'phase', {'phase': phase})
            started = time.time()
            info = PHASE_FN[phase](bus)
            state.setdefault('hours', {})[phase] = round((time.time() - started) / 3600, 3)
            state.setdefault('results', {})[phase] = info
            state['phase'] = PHASES[PHASES.index(phase) + 1]
            state['failures'] = 0
            write_json(RUN / 'state.json', state)
        bus.post('decision', 'all', 'decision', {'decision': 'complete', 'why': '对照实验完成，报告见 REPORT.md；守护脚本将关机'})
        return 0
    except StopRequested:
        bus.post('decision', 'all', 'stopped', {})
        return 4
    except FatalError as exc:
        err = {'time': stamp(), 'phase': state.get('phase'), 'error': str(exc), 'traceback': traceback.format_exc()[-4000:]}
        write_json(RUN / 'FAILED.json', err)
        bus.post('decision', 'all', 'fatal', err)
        return 2
    except Exception as exc:  # noqa: BLE001
        tb = traceback.format_exc()
        print(tb, flush=True)
        state['failures'] = state.get('failures', 0) + 1
        state['last_error'] = {'time': stamp(), 'phase': state.get('phase'), 'error': f'{type(exc).__name__}: {exc}', 'traceback': tb[-4000:]}
        write_json(RUN / 'state.json', state)
        bus.post('decision', 'all', 'error', {'phase': state.get('phase'), 'failures': state['failures'],
                                              'error': f'{type(exc).__name__}: {str(exc)[:400]}'})
        if state['failures'] >= 4:
            write_json(RUN / 'FAILED.json', state['last_error'])
            return 2
        return 1


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'run':
        sys.exit(run())
    print(json.dumps(read_json(RUN / 'state.json', {}), ensure_ascii=False, indent=1)[:3000])
