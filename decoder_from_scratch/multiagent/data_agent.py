"""数据清洗智能体：下载 → 规则清洗 → 全局去重与划分 → 训练 tokenizer → 编码 → (训练中)按需补数据。

只负责数据；配比、是否补数据、领域是否隔离由决策智能体决定。所有产物写在 RUN_DIR 下，
已完成的步骤有回执文件，重启后直接复用。
"""

from array import array
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import gzip
import json
import multiprocessing
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time
import types
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from . import cleaning
from .common import (DECODER_DIR, RUN_DIR, TOKENIZER_PROJECT, file_sha256, now, python_executable,
                     read_json, write_json)

NAME = 'data'
LEGACY_RAW = DECODER_DIR / 'runs' / 'scale_144m_24h_v2' / 'raw'
BOS_ID, EOS_ID = 2, 3


# ----------------------------------------------------------------------------------------------
# 下载 + 清洗（在子进程中运行，必须是模块级函数）
# ----------------------------------------------------------------------------------------------

def source_url(dataset_id, revision, path):
    query = urlencode(dict(Source='SDK', Revision=revision, FilePath=path, View='False'))
    return f'https://www.modelscope.cn/api/v1/datasets/{dataset_id}/repo?{query}'


def acquire_domain(job):
    """流式读取 gzip JSONL，逐条清洗并写出；到达目标字符数或数据流结束即停止。

    job: domain, url, target_chars, output, deadline, skip_rows, local_file(可选，替代下载)
    返回回执 dict；异常时回执中 error 非空（已写出的部分仍然有效）。
    """
    domain, output = job['domain'], Path(job['output'])
    partial = output.with_name(output.name + '.partial')
    stats, seen = Counter(), set()
    error, finished_stream, last_row = None, False, job.get('skip_rows', 0) - 1
    started = time.time()
    try:
        if job.get('local_file'):
            stream_ctx = open(job['local_file'], 'rb')
            gz = None
        else:
            request = Request(job['url'], headers={'Accept-Encoding': 'identity',
                                                   'User-Agent': 'decoder-multiagent/1.0'})
            stream_ctx = urlopen(request, timeout=90)
            gz = True
        with stream_ctx as response, partial.open('w', encoding='utf-8') as dest:
            reader = gzip.GzipFile(fileobj=response) if gz else response
            row = -1
            while stats['chars'] < job['target_chars']:
                if time.time() > job['deadline']:
                    raise TimeoutError('data deadline reached')
                line = reader.readline(4 * 1024 * 1024)
                if not line:
                    finished_stream = True
                    break
                row += 1
                if row < job.get('skip_rows', 0):
                    continue
                last_row = row
                stats['rows'] += 1
                stats['raw_bytes'] += len(line)
                try:
                    record = json.loads(line)
                    raw_text = record.get('text') if isinstance(record, dict) else None
                except (ValueError, UnicodeError):
                    stats['reject_malformed'] += 1
                    continue
                text, reason = cleaning.clean(raw_text, domain)
                if reason:
                    stats['reject_' + reason] += 1
                    continue
                h = cleaning.doc_hash(text)
                if h in seen:
                    stats['reject_duplicate_local'] += 1
                    continue
                seen.add(h)
                dest.write(json.dumps({'text': text, 'h': h, 'row': row}, ensure_ascii=False) + '\n')
                stats['docs'] += 1
                stats['chars'] += len(text)
                if stats['rows'] % 20000 == 0:
                    print(json.dumps({'agent': NAME, 'domain': domain, 'rows': stats['rows'],
                                      'chars': stats['chars'], 'target': job['target_chars']}), flush=True)
    except Exception as exc:  # 网络/解压/超时：保留部分结果，交由决策智能体判断
        error = f'{type(exc).__name__}: {exc}'
    if partial.exists():
        os.replace(partial, output)
    receipt = {'domain': domain, 'url': job.get('url'), 'local_file': job.get('local_file'),
               'target_chars': job['target_chars'], 'skip_rows': job.get('skip_rows', 0),
               'last_row': last_row, 'finished_stream': finished_stream, 'error': error,
               'seconds': round(time.time() - started, 1), 'stats': dict(stats)}
    return receipt


# ----------------------------------------------------------------------------------------------
# tokenizer 加载与编码（子进程）
# ----------------------------------------------------------------------------------------------

def load_bbpe(path):
    """使用工程内 decoder_from_scratch.tokenizer.ScratchBBPETokenizer，保证与推理时完全一致。"""
    if str(DECODER_DIR.parent) not in sys.path:
        sys.path.insert(0, str(DECODER_DIR.parent))
    try:
        from decoder_from_scratch.tokenizer import load_tokenizer_file
    except ImportError:
        # 无 torch 的测试环境：用桩替代 .data 模块（CharTokenizer 在此不会被用到）。
        stub = types.ModuleType('decoder_from_scratch.data')
        stub.CharTokenizer = None
        sys.modules['decoder_from_scratch.data'] = stub
        from decoder_from_scratch.tokenizer import load_tokenizer_file
    return load_tokenizer_file(path)


_TOK = None


def _init_encoder(path):
    global _TOK
    _TOK = load_bbpe(path)


def _encode_doc(text):
    ids = _TOK.encode(text, add_bos=True, add_eos=True)
    if max(ids) > 65535:
        raise ValueError('token id exceeds uint16')
    return array('H', ids).tobytes()


def encode_jsonl(pool, source, target, chunksize=64):
    docs, tokens = 0, 0
    tmp = Path(str(target) + '.partial')

    def texts():
        with open(source, encoding='utf-8') as f:
            for line in f:
                yield json.loads(line)['text']

    with tmp.open('wb') as out:
        for blob in pool.imap(_encode_doc, texts(), chunksize=chunksize):
            out.write(blob)
            docs += 1
            tokens += len(blob) // 2
    os.replace(tmp, target)
    return {'docs': docs, 'tokens': tokens}


# ----------------------------------------------------------------------------------------------
# 补数据任务（训练期间由决策智能体在独立进程中启动）
# ----------------------------------------------------------------------------------------------

def topup_job(run_dir, domain, extra_chars, deadline):
    os.environ['MA_RUN_DIR'] = str(run_dir)
    agent = DataAgent(config=read_json(Path(run_dir) / 'config.json'), bus=None)
    try:
        result = agent.topup(domain, extra_chars, deadline)
    except Exception as exc:
        result = {'domain': domain, 'error': f'{type(exc).__name__}: {exc}'}
    write_json(Path(run_dir) / 'topup' / f'{domain}.result.json', result)


# ----------------------------------------------------------------------------------------------
# 智能体
# ----------------------------------------------------------------------------------------------

class DataAgent:
    def __init__(self, config, bus):
        self.config, self.bus = config, bus
        self.cfg = config['data']
        self.raw_dir = RUN_DIR / 'data' / 'raw_clean'
        self.clean_dir = RUN_DIR / 'data' / 'clean'
        self.token_dir = RUN_DIR / 'data' / 'tokens'
        self.tokenizer_dir = RUN_DIR / 'tokenizer'
        for folder in (self.raw_dir, self.clean_dir, self.token_dir, RUN_DIR / 'logs', RUN_DIR / 'topup'):
            folder.mkdir(parents=True, exist_ok=True)

    def say(self, recipient, kind, content):
        if self.bus is not None:
            self.bus.post(NAME, recipient, kind, content)

    # ---------------- 来源 ----------------
    def sources(self):
        spec = read_json(DECODER_DIR / self.cfg['source_config'])
        out = {}
        for source in spec['sources']:
            if source['domain'] in self.cfg['weights']:
                out[source['domain']] = {'url': source_url(spec['dataset_id'], spec['revision'], source['path']),
                                         'path': source['path'], 'dataset': spec['dataset_id'],
                                         'revision': spec['revision'], 'language': source.get('language')}
        return out

    # ---------------- 阶段 1：下载与清洗 ----------------
    def acquire(self, deadline, domains=None, use_local=None):
        """并行下载并清洗各领域；返回 {domain: receipt}。已有完整回执的领域直接复用。"""
        weights = self.cfg['weights']
        total = self.cfg['total_chars']
        sources = self.sources()
        use_local = use_local or {}
        jobs = []
        receipts = {}
        for domain in (domains or list(weights)):
            receipt_path = self.raw_dir / f'{domain}.receipt.json'
            saved = read_json(receipt_path)
            if saved and not saved.get('error') and not use_local.get(domain):
                receipts[domain] = saved
                continue
            job = {'domain': domain, 'target_chars': int(total * weights[domain]),
                   'output': str(self.raw_dir / f'{domain}.jsonl'), 'deadline': deadline}
            if use_local.get(domain):
                job['local_file'] = use_local[domain]
            elif domain in sources:
                job['url'] = sources[domain]['url']
            else:
                receipts[domain] = {'domain': domain, 'error': 'no source configured', 'stats': {}}
                continue
            jobs.append(job)
        if jobs:
            self.say('decision', 'progress', {'action': 'download_start',
                                              'domains': [j['domain'] for j in jobs],
                                              'target_chars': {j['domain']: j['target_chars'] for j in jobs}})
            ctx = multiprocessing.get_context('spawn')
            with ProcessPoolExecutor(max_workers=self.cfg['download_workers'], mp_context=ctx) as pool:
                futures = {pool.submit(acquire_domain, job): job for job in jobs}
                for future in as_completed(futures):
                    job = futures[future]
                    try:
                        receipt = future.result()
                    except Exception as exc:
                        receipt = {'domain': job['domain'], 'error': f'worker crashed: {exc}', 'stats': {}}
                    write_json(self.raw_dir / f'{job["domain"]}.receipt.json', receipt)
                    receipts[job['domain']] = receipt
                    s = receipt.get('stats', {})
                    self.say('decision', 'report', {
                        'action': 'domain_acquired', 'domain': job['domain'], 'docs': s.get('docs', 0),
                        'chars': s.get('chars', 0), 'target': job['target_chars'], 'rows': s.get('rows', 0),
                        'accept_rate': round(s.get('docs', 0) / max(1, s.get('rows', 0)), 3),
                        'error': receipt.get('error'), 'seconds': receipt.get('seconds')})
        return receipts

    def legacy_file(self, domain):
        path = LEGACY_RAW / f'{domain}.jsonl'
        return str(path) if path.exists() else None

    # ---------------- 阶段 2：全局去重 + 划分 ----------------
    def merge(self, active_domains):
        """跨领域精确去重 + 段落级近重复去重，按文档哈希稳定划分 train/val/test。"""
        report_path = RUN_DIR / 'data' / 'merge_report.json'
        if report_path.exists():
            return read_json(report_path)
        weights = self.cfg['weights']
        order = sorted(active_domains, key=lambda d: -weights[d])
        doc_seen, para_seen = set(), set()
        stats = {}
        caps = {'val': self.cfg['val_docs_per_domain'], 'test': self.cfg['test_docs_per_domain']}
        samples = {}
        rng = random.Random(7)
        for domain in order:
            counts = Counter()
            outs = {s: (self.clean_dir / f'{domain}.{s}.jsonl.partial').open('w', encoding='utf-8')
                    for s in ('train', 'val', 'test')}
            picked = []
            with (self.raw_dir / f'{domain}.jsonl').open(encoding='utf-8') as f:
                for line in f:
                    rec = json.loads(line)
                    text, h = rec['text'], rec['h']
                    if h in doc_seen:
                        counts['dup_exact'] += 1
                        continue
                    paras = cleaning.paragraph_hashes(text)
                    if len(paras) >= 3 and sum(p in para_seen for p in paras) >= 0.5 * len(paras):
                        counts['dup_paragraph'] += 1
                        continue
                    doc_seen.add(h)
                    para_seen.update(paras)
                    u = int(h[:8], 16) / 2 ** 32
                    split = 'train'
                    if u < 0.01 and counts['docs_val'] < caps['val']:
                        split = 'val'
                    elif 0.01 <= u < 0.02 and counts['docs_test'] < caps['test']:
                        split = 'test'
                    outs[split].write(json.dumps({'text': text, 'h': h}, ensure_ascii=False) + '\n')
                    counts['docs_' + split] += 1
                    counts['chars_' + split] += len(text)
                    if split == 'train' and len(picked) < 200 and rng.random() < 0.01:
                        picked.append(text[:600])
            for split, handle in outs.items():
                handle.close()
                os.replace(self.clean_dir / f'{domain}.{split}.jsonl.partial', self.clean_dir / f'{domain}.{split}.jsonl')
            stats[domain] = dict(counts)
            samples[domain] = rng.sample(picked, min(5, len(picked)))
            self.say('decision', 'report', {'action': 'domain_merged', 'domain': domain, **dict(counts)})
        # 持久化哈希，供训练中补数据时继续去重
        array('Q', [int(h[:16], 16) for h in doc_seen]).tofile(open(RUN_DIR / 'data' / 'doc_hashes.u64', 'wb'))
        array('Q', para_seen).tofile(open(RUN_DIR / 'data' / 'para_hashes.u64', 'wb'))
        write_json(RUN_DIR / 'data' / 'data_samples.json', samples)
        report = {'domains': stats, 'total_train_chars': sum(s.get('chars_train', 0) for s in stats.values()),
                  'policy': 'rule-based cleaning + global exact/paragraph dedup; val/test split by doc hash before training'}
        write_json(report_path, report)
        return report

    # ---------------- 阶段 3：tokenizer（调用外部 tokenizer_training 工程） ----------------
    def build_tokenizer_corpus(self, active_domains, chars_budget):
        path = RUN_DIR / 'data' / f'tokenizer_corpus_{chars_budget}.jsonl'
        if path.exists():
            return path
        weights = self.cfg['weights']
        total_w = sum(weights[d] for d in active_domains)
        tmp = Path(str(path) + '.partial')
        with tmp.open('w', encoding='utf-8') as out:
            for domain in active_domains:
                budget = chars_budget * weights[domain] / total_w
                train_file = self.clean_dir / f'{domain}.train.jsonl'
                size = sum(1 for _ in train_file.open(encoding='utf-8'))
                stride = max(1, size // 4000)  # 跨文件均匀抽样，避免只取开头
                used = 0
                with train_file.open(encoding='utf-8') as f:
                    for i, line in enumerate(f):
                        if i % stride:
                            continue
                        text = json.loads(line)['text']
                        out.write(json.dumps({'text': text}, ensure_ascii=False) + '\n')
                        used += len(text)
                        if used >= budget:
                            break
        os.replace(tmp, path)
        return path

    def train_tokenizer(self, active_domains, deadline):
        target = self.tokenizer_dir / 'bbpe.json'
        if target.exists():
            return self.tokenizer_report()
        trainer = TOKENIZER_PROJECT / 'train_tokenizer.py'
        if not trainer.exists():
            raise FileNotFoundError(f'tokenizer project not found: {trainer}')
        budget = int(self.cfg['tokenizer_chars'])
        timeout = self.cfg['tokenizer_timeout_hours'] * 3600
        for attempt in range(3):
            corpus = self.build_tokenizer_corpus(active_domains, budget)
            out_dir = RUN_DIR / f'tokenizer_attempt{attempt}'
            if out_dir.exists():
                shutil.move(str(out_dir), str(RUN_DIR / f'tokenizer_attempt{attempt}_stale_{int(now())}'))
            cmd = [python_executable(), '-u', str(trainer), '--corpus', str(corpus), '--fields', 'text',
                   '--backend', 'memory', '--vocab-size', str(self.cfg['tokenizer_vocab']),
                   '--min-frequency', str(self.cfg['tokenizer_min_frequency']), '--output-dir', str(out_dir)]
            self.say('decision', 'progress', {'action': 'tokenizer_start', 'attempt': attempt,
                                              'corpus_chars': budget, 'vocab': self.cfg['tokenizer_vocab'],
                                              'trainer': str(trainer)})
            log = (RUN_DIR / 'logs' / f'tokenizer_attempt{attempt}.log').open('w')
            limit = min(timeout, max(600, deadline - now()))
            start = now()
            try:
                result = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, timeout=limit,
                                        cwd=str(TOKENIZER_PROJECT.parent))
                ok = result.returncode == 0 and (out_dir / 'bbpe.json').exists()
                reason = f'exit {result.returncode}'
            except subprocess.TimeoutExpired:
                ok, reason = False, f'timeout after {limit:.0f}s'
            finally:
                log.close()
            if ok:
                self.tokenizer_dir.mkdir(parents=True, exist_ok=True)
                for name in ('bbpe.json', 'vocab.json', 'training_report.json'):
                    if (out_dir / name).exists():
                        shutil.copyfile(out_dir / name, self.tokenizer_dir / name)
                report = self.tokenizer_report(seconds=now() - start)
                self.say('decision', 'report', {'action': 'tokenizer_done', **report})
                return report
            self.say('decision', 'report', {'action': 'tokenizer_failed', 'attempt': attempt, 'reason': reason})
            budget //= 2  # 失败/超时：语料减半重试
        raise RuntimeError('tokenizer training failed 3 times')

    def tokenizer_report(self, seconds=None):
        tok = load_bbpe(self.tokenizer_dir)
        stats = {}
        for path in sorted(self.clean_dir.glob('*.val.jsonl'))[:20]:
            chars = tokens = 0
            with path.open(encoding='utf-8') as f:
                for i, line in enumerate(f):
                    if i >= 30:
                        break
                    text = json.loads(line)['text']
                    ids = tok.encode(text)
                    if tok.decode(ids) != text:
                        raise ValueError(f'tokenizer round-trip mismatch in {path.name}')
                    chars += len(text)
                    tokens += len(ids)
            stats[path.name.split('.')[0]] = round(chars / max(1, tokens), 3)
        report = {'vocab_size': len(tok.tokens), 'sha256': file_sha256(self.tokenizer_dir / 'bbpe.json'),
                  'chars_per_token_val': stats}
        if seconds is not None:
            report['seconds'] = round(seconds, 1)
        return report

    # ---------------- 阶段 4：编码 ----------------
    def encode(self, active_domains):
        manifest_path = self.token_dir / 'manifest.json'
        if manifest_path.exists():
            return read_json(manifest_path)
        tok_path = self.tokenizer_dir / 'bbpe.json'
        workers = self.cfg['encode_workers'] or max(2, (os.cpu_count() or 4) - 2)
        manifest = {'tokenizer': {'path': str(tok_path.relative_to(RUN_DIR)), 'sha256': file_sha256(tok_path),
                                  'vocab_size': len(load_bbpe(tok_path).tokens)},
                    'bos_id': BOS_ID, 'eos_id': EOS_ID, 'dtype': 'uint16', 'domains': {}}
        ctx = multiprocessing.get_context('spawn')
        started = now()
        with ctx.Pool(workers, initializer=_init_encoder, initargs=(str(tok_path),)) as pool:
            for domain in active_domains:
                entry = {}
                for split in ('train', 'val', 'test'):
                    src = self.clean_dir / f'{domain}.{split}.jsonl'
                    dst = self.token_dir / f'{domain}.{split}.bin'
                    info = encode_jsonl(pool, src, dst)
                    info['path'] = str(dst.relative_to(RUN_DIR))
                    entry[split] = [info] if split == 'train' else info
                manifest['domains'][domain] = entry
                self.say('decision', 'report', {'action': 'domain_encoded', 'domain': domain,
                                                'train_tokens': entry['train'][0]['tokens'],
                                                'val_tokens': entry['val']['tokens']})
        # 抽查：解码若干文档必须与原文一致
        tok = load_bbpe(tok_path)
        for domain in active_domains[:3]:
            with (self.clean_dir / f'{domain}.val.jsonl').open(encoding='utf-8') as f:
                text = json.loads(f.readline())['text']
            assert tok.decode(tok.encode(text)) == text
        manifest['encode_seconds'] = round(now() - started, 1)
        write_json(manifest_path, manifest)
        return manifest

    # ---------------- 训练中补数据 ----------------
    def topup(self, domain, extra_chars, deadline):
        """同一来源续读后续行 → 清洗 → 对已有哈希去重 → 编码为新的 train 分片。"""
        receipt = read_json(self.raw_dir / f'{domain}.receipt.json')
        if not receipt or receipt.get('finished_stream'):
            return {'domain': domain, 'error': 'source exhausted or missing receipt', 'tokens': 0}
        k = len(list(self.raw_dir.glob(f'{domain}.topup*.jsonl'))) + 1
        job = {'domain': domain, 'url': receipt.get('url'), 'local_file': receipt.get('local_file'),
               'target_chars': int(extra_chars),
               'output': str(self.raw_dir / f'{domain}.topup{k}.jsonl'), 'deadline': deadline,
               'skip_rows': receipt['last_row'] + 1}
        new_receipt = acquire_domain(job)
        doc_seen = set(array('Q', (RUN_DIR / 'data' / 'doc_hashes.u64').read_bytes()))
        para_arr = array('Q')
        para_arr.frombytes((RUN_DIR / 'data' / 'para_hashes.u64').read_bytes())
        para_seen = set(para_arr)
        kept = self.clean_dir / f'{domain}.train.topup{k}.jsonl'
        n_docs = 0
        with open(job['output'], encoding='utf-8') as f, kept.open('w', encoding='utf-8') as out:
            for line in f:
                rec = json.loads(line)
                if int(rec['h'][:16], 16) in doc_seen:
                    continue
                paras = cleaning.paragraph_hashes(rec['text'])
                if len(paras) >= 3 and sum(p in para_seen for p in paras) >= 0.5 * len(paras):
                    continue
                doc_seen.add(int(rec['h'][:16], 16))
                para_seen.update(paras)
                out.write(json.dumps({'text': rec['text'], 'h': rec['h']}, ensure_ascii=False) + '\n')
                n_docs += 1
        tok_path = self.tokenizer_dir / 'bbpe.json'
        dst = self.token_dir / f'{domain}.train.topup{k}.bin'
        ctx = multiprocessing.get_context('spawn')
        with ctx.Pool(2, initializer=_init_encoder, initargs=(str(tok_path),)) as pool:
            info = encode_jsonl(pool, kept, dst)
        info['path'] = str(dst.relative_to(RUN_DIR))
        # 更新回执：下一次补数据从更后面的行开始
        receipt['last_row'] = new_receipt['last_row']
        receipt['finished_stream'] = new_receipt['finished_stream']
        write_json(self.raw_dir / f'{domain}.receipt.json', receipt)
        array('Q', doc_seen).tofile(open(RUN_DIR / 'data' / 'doc_hashes.u64', 'wb'))
        array('Q', para_seen).tofile(open(RUN_DIR / 'data' / 'para_hashes.u64', 'wb'))
        return {'domain': domain, 'shard': info, 'error': new_receipt.get('error'), 'tokens': info['tokens']}
