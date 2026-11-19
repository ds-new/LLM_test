"""数据清洗智能体：采样 → 训练 tokenizer → 快速编码器 → 流式"下载-清洗-去重-划分-编码"→ token 分片。

关键设计（数据盘只有约 50GB，存不下原始文本）：
  * 流式处理：jsonl / jsonl.gz 边下载边处理，不落盘；parquet 必须完整下载才能读，先下到系统盘临时目录，读完即删；
  * 每个文件是一个独立任务（多进程并行），产物是该文件的 train/val/test 三个 .bin（uint16 token），
    完成后写回执；进程被杀后重启只重做没有回执的文件；
  * 去重：文件内精确去重 + 段落级近重复去重。各来源上游已做过全局去重（CCI3-HQ、FineWeb-Edu-dedup、StarCoderData），
    跨文件重复这里不再处理——这是为了可扩展而做的取舍；
  * 划分：按文档哈希决定 train/val/test，同一文档永远落在同一划分，验证集在训练前就固定。
"""

from array import array
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
import gzip
import json
import random
import multiprocessing
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from urllib.request import Request, urlopen

from . import cleaning
from .common import (DECODER_DIR, RUN_DIR, SCRATCH_DIR, TOKENIZER_PROJECT, cpu_quota, file_sha256, free_gb, heartbeat, now,
                     python_executable, read_json, write_json)
from .sources import SOURCES, enumerate_source

NAME = 'data'
BOS_ID, EOS_ID = 2, 3


# ----------------------------------------------------------------------------------------------
# tokenizer 加载（子进程中使用）
# ----------------------------------------------------------------------------------------------

class HFTokenizerAdapter:
    """把 HF tokenizers 的 tokenizer.json 包装成 ScratchBBPETokenizer 的接口（encode(text, add_bos) / decode(ids)）。
    路线 B（v2 tokenizer：数字逐位、前导空格并入单词）用它复用整条流水线；特殊 token 编号与 v1 相同（BOS=2，EOS=3）。"""

    def __init__(self, path):
        from tokenizers import Tokenizer
        self.tk = Tokenizer.from_file(str(path))
        self.vocab_size = self.tk.get_vocab_size()
        self.bos_id, self.eos_id = BOS_ID, EOS_ID

    def encode(self, text, add_bos=False, add_eos=False):
        ids = self.tk.encode(text, add_special_tokens=False).ids
        return ([BOS_ID] if add_bos else []) + ids + ([EOS_ID] if add_eos else [])

    def decode(self, ids, skip_special_tokens=True):
        return self.tk.decode([int(i) for i in ids], skip_special_tokens=skip_special_tokens)


def load_bbpe(path):
    try:
        head = json.loads(Path(path).read_text(encoding='utf-8'))
        if isinstance(head, dict) and 'model' in head and 'pre_tokenizer' in head:
            return HFTokenizerAdapter(path)
    except (OSError, ValueError):
        pass
    """工程内的 ScratchBBPETokenizer（推理时用的也是它），保证训练与推理分词一致。"""
    if str(DECODER_DIR.parent) not in sys.path:
        sys.path.insert(0, str(DECODER_DIR.parent))
    try:
        from decoder_from_scratch.tokenizer import load_tokenizer_file
    except ImportError:  # 没有 torch 的环境（只做分词）：用桩替代 data 模块，CharTokenizer 在这里用不到
        import types
        stub = types.ModuleType('decoder_from_scratch.data')
        stub.CharTokenizer = None
        sys.modules['decoder_from_scratch.data'] = stub
        from decoder_from_scratch.tokenizer import load_tokenizer_file
    return load_tokenizer_file(path)


class _Encoder:
    """优先用验证过的 HF 快速编码器；没有则用纯 Python BBPE。"""

    def __init__(self, bbpe_path, hf_path=None):
        self.fast = None
        if hf_path and Path(hf_path).exists():
            os.environ.setdefault('RAYON_NUM_THREADS', '1')
            os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
            from tokenizers import Tokenizer
            self.fast = Tokenizer.from_file(str(hf_path))
        self.slow = None if self.fast else load_bbpe(bbpe_path)

    def encode_many(self, texts):
        if self.fast:
            return [e.ids for e in self.fast.encode_batch(texts, add_special_tokens=False)]
        return [self.slow.encode(t) for t in texts]


# ----------------------------------------------------------------------------------------------
# 单文件任务（子进程中运行）
# ----------------------------------------------------------------------------------------------

def _open_stream(job):
    request = Request(job['url'], headers={'User-Agent': 'decoder-a800/1.0', 'Accept-Encoding': 'identity'})
    response = urlopen(request, timeout=120)
    if job['format'] == 'jsonl.gz':
        return response, gzip.GzipFile(fileobj=response)
    return response, response


def _download(job, dest):
    """parquet 整文件下载到系统盘临时目录（支持断点续传），校验大小。"""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size == job['size']:
        return dest
    partial = dest.with_name(dest.name + '.partial')
    have = partial.stat().st_size if partial.exists() else 0
    headers = {'User-Agent': 'decoder-a800/1.0'}
    if have:
        headers['Range'] = f'bytes={have}-'
    with urlopen(Request(job['url'], headers=headers), timeout=120) as r, partial.open('ab' if have else 'wb') as f:
        if have and r.status != 206:  # 服务器不支持续传：从头下载
            f.seek(0)
            f.truncate()
        shutil.copyfileobj(r, f, 4 << 20)
    if partial.stat().st_size != job['size']:
        raise IOError(f'size mismatch {partial.stat().st_size} != {job["size"]}')
    os.replace(partial, dest)
    return dest


def _iter_texts(job):
    fields = job['fields']
    if job['format'] == 'parquet':
        import pyarrow.parquet as pq
        path = _download(job, Path(job['download_path']))
        pf = pq.ParquetFile(path)
        names = pf.schema_arrow.names
        field = next((f for f in fields if f in names), None)
        if field is None:
            raise ValueError(f'no text field in {names}')
        for batch in pf.iter_batches(batch_size=512, columns=[field]):
            for value in batch.column(0).to_pylist():
                yield value
        return
    response, stream = _open_stream(job)
    try:
        for line in stream:
            try:
                row = json.loads(line)
            except ValueError:
                yield None
                continue
            yield next((row[f] for f in fields if isinstance(row, dict) and f in row), None)
    finally:
        response.close()


def process_file(job):
    """job['mode'] == 'sample'：只清洗并收集文本（给 tokenizer 训练用）；'build'：清洗后编码成 token 分片。"""
    started = time.time()
    stats = Counter()
    seen_docs, seen_paras = set(), set()
    out_dir = Path(job['out_dir'])
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f'{job["index"]:05d}'
    encoder = _Encoder(job['bbpe_path'], job.get('hf_path')) if job['mode'] == 'build' else None
    buffers = {s: array('H') for s in ('train', 'val', 'test')}
    handles = {}
    if job['mode'] == 'build':
        handles = {s: (out_dir / f'{stem}.{s}.bin.partial').open('wb') for s in buffers}
    sample_handle = (out_dir / f'{stem}.sample.jsonl.partial').open('w', encoding='utf-8') if job['mode'] == 'sample' else None
    pending = []

    def flush_pending():
        for (split, text), ids in zip(pending, encoder.encode_many([t for _, t in pending])):
            buf = buffers[split]
            buf.append(BOS_ID)
            buf.extend(ids)
            buf.append(EOS_ID)
            stats[f'tokens_{split}'] += len(ids) + 2
        pending.clear()
        for split, buf in buffers.items():
            if len(buf) > (1 << 20):
                buf.tofile(handles[split])
                del buf[:]

    try:
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
            u = (h & 0xFFFFFFFF) / 2 ** 32
            split = 'val' if u < job['val_fraction'] else ('test' if u < job['val_fraction'] + job['test_fraction'] else 'train')
            stats['docs_' + split] += 1
            stats['chars_' + split] += len(text)
            if job['mode'] == 'sample':
                if split == 'train':
                    sample_handle.write(json.dumps({'text': text, 'source': job['source']}, ensure_ascii=False) + '\n')
                if stats['chars_train'] >= job['char_budget']:
                    break
                continue
            pending.append((split, text))
            if len(pending) >= 256:
                flush_pending()
                if stats['tokens_train'] >= job['token_cap']:
                    stats['stopped_at_cap'] = 1
                    break
        if job['mode'] == 'build':
            flush_pending()
            for split, buf in buffers.items():
                buf.tofile(handles[split])
    finally:
        for h in handles.values():
            h.close()
        if sample_handle:
            sample_handle.close()
    # 全部成功才把 .partial 改名；失败时由协调者删除并重试
    if job['mode'] == 'build':
        for split in buffers:
            os.replace(out_dir / f'{stem}.{split}.bin.partial', out_dir / f'{stem}.{split}.bin')
    else:
        os.replace(out_dir / f'{stem}.sample.jsonl.partial', out_dir / f'{stem}.sample.jsonl')
    if job['format'] == 'parquet' and job['mode'] == 'build':
        Path(job['download_path']).unlink(missing_ok=True)
    stats = dict(stats)
    stats['seconds'] = round(time.time() - started, 1)
    stats['accept_rate'] = round(sum(stats.get('docs_' + s, 0) for s in ('train', 'val', 'test')) / max(1, stats.get('rows', 0)), 3)
    return {'source': job['source'], 'index': job['index'], 'path': job['path'], 'stats': stats}


# ----------------------------------------------------------------------------------------------
# 智能体
# ----------------------------------------------------------------------------------------------

class DataAgent:
    def __init__(self, config, bus):
        self.config, self.bus = config, bus
        self.cfg = config['data']
        self.data_dir = RUN_DIR / 'data'
        self.token_dir = self.data_dir / 'tokens'
        self.sample_dir = self.data_dir / 'tokenizer_sample'
        self.tokenizer_dir = RUN_DIR / 'tokenizer'
        self.download_dir = SCRATCH_DIR / 'downloads'
        for folder in (self.data_dir, self.token_dir, self.sample_dir, RUN_DIR / 'logs', self.download_dir):
            folder.mkdir(parents=True, exist_ok=True)
        self.workers = self.cfg['workers'] or max(2, cpu_quota() - 2)

    def say(self, recipient, kind, content):
        if self.bus is not None:
            self.bus.post(NAME, recipient, kind, content)

    # ---------------- 文件清单 ----------------
    def file_lists(self):
        path = self.data_dir / 'file_lists.json'
        cached = read_json(path)
        if cached:
            return cached
        lists = {}
        for spec in SOURCES:
            files = enumerate_source(spec)
            lists[spec['name']] = files
            self.say('decision', 'report', {'action': 'source_listed', 'source': spec['name'], 'dataset': spec['dataset'],
                                            'files': len(files), 'total_gb': round(sum(f['size'] for f in files) / 1e9, 1)})
            heartbeat('listing')
        write_json(path, lists)
        return lists

    def _job(self, spec, f, mode, **extra):
        job = {'source': spec['name'], 'kind': spec['kind'], 'format': spec['format'], 'fields': spec['fields'],
               'url': f['url'], 'path': f['path'], 'size': f['size'], 'index': f['index'], 'mode': mode,
               'val_fraction': self.cfg['val_fraction'], 'test_fraction': self.cfg['test_fraction'],
               'download_path': str(self.download_dir / spec['name'] / f'{f["index"]:05d}.parquet'),
               'bbpe_path': str(self.tokenizer_dir / 'bbpe.json'), 'hf_path': None}
        job.update(extra)
        return job

    # ---------------- 阶段：tokenizer 采样 ----------------
    def sample(self, weights):
        """每个来源取每个子目录的第一个文件，清洗后按权重收集字符，合成 tokenizer 训练语料。"""
        out = self.data_dir / 'tokenizer_corpus.jsonl'
        if out.exists():
            return read_json(self.data_dir / 'tokenizer_corpus.json')
        lists = self.file_lists()
        total = self.cfg['tokenizer_sample_chars']
        jobs = []
        for spec in SOURCES:
            files = lists[spec['name']]
            firsts, roots = [], []
            for f in files:
                if f['root'] not in roots:
                    roots.append(f['root'])
                    firsts.append(f)
            budget = total * weights[spec['name']] / len(firsts)
            for f in firsts:
                jobs.append(self._job(spec, f, 'sample', out_dir=str(self.sample_dir / spec['name']),
                                      char_budget=int(budget)))
        results = self._run_jobs(jobs, phase='sample')
        lines = []
        for spec in SOURCES:
            for path in sorted((self.sample_dir / spec['name']).glob('*.sample.jsonl')):
                lines.extend(line for line in path.read_text(encoding='utf-8').splitlines() if line.strip())
        # 打乱顺序：超时重试时按 max_texts 截取前一半，必须保证每个来源都还在
        random.Random(20261102).shuffle(lines)
        with out.open('w', encoding='utf-8') as dest:
            dest.write('\n'.join(lines) + '\n')
        chars = {r['source']: 0 for r in results}
        for r in results:
            chars[r['source']] += r['stats'].get('chars_train', 0)
        report = {'chars_by_source': chars, 'total_chars': sum(chars.values()), 'files': len(results)}
        write_json(self.data_dir / 'tokenizer_corpus.json', report)
        # 每个来源存 3 条清洗后的样例，供人工抽查
        samples = {}
        for spec in SOURCES:
            rows = []
            for path in sorted((self.sample_dir / spec['name']).glob('*.sample.jsonl'))[:1]:
                for line in path.read_text(encoding='utf-8').splitlines()[:3]:
                    rows.append(json.loads(line)['text'][:800])
            samples[spec['name']] = rows
        write_json(self.data_dir / 'samples_after_cleaning.json', samples)
        return report

    def _run_jobs(self, jobs, phase):
        """简单并行执行一组文件任务（失败重试，超过次数则跳过）。"""
        ctx = multiprocessing.get_context('spawn')
        results, attempts = [], Counter()
        queue = list(jobs)
        with ProcessPoolExecutor(max_workers=min(self.workers, max(1, len(queue))), mp_context=ctx) as pool:
            inflight = {}
            while queue or inflight:
                while queue and len(inflight) < self.workers:
                    job = queue.pop(0)
                    inflight[pool.submit(process_file, job)] = job
                finished, _ = wait(list(inflight), timeout=60, return_when=FIRST_COMPLETED)
                heartbeat(phase)
                for future in finished:
                    job = inflight.pop(future)
                    try:
                        results.append(future.result())
                    except Exception as exc:
                        attempts[(job['source'], job['index'])] += 1
                        if attempts[(job['source'], job['index'])] < self.cfg['file_attempts']:
                            queue.append(job)
                        else:
                            self.say('decision', 'report', {'action': 'file_skipped', 'phase': phase, 'source': job['source'],
                                                            'path': job['path'], 'error': f'{type(exc).__name__}: {str(exc)[:200]}'})
        return results

    # ---------------- 阶段：训练 tokenizer（调用用户自己的 tokenizer_training 工程） ----------------
    def train_tokenizer(self, deadline):
        target = self.tokenizer_dir / 'bbpe.json'
        if target.exists():
            return self.tokenizer_report()
        trainer = TOKENIZER_PROJECT / 'train_tokenizer.py'
        if not trainer.exists():
            raise FileNotFoundError(f'tokenizer project not found: {trainer}')
        corpus = self.data_dir / 'tokenizer_corpus.jsonl'
        max_texts = 0
        for attempt in range(3):
            out_dir = RUN_DIR / f'tokenizer_attempt{attempt}'
            if out_dir.exists():
                shutil.rmtree(out_dir)
            tok_config = RUN_DIR / 'tokenizer_config.json'
            write_json(tok_config, {'algorithm': 'bbpe', 'backend': 'memory', 'corpus_files': [str(corpus)],
                                    'fields': ['text'], 'output_dir': str(out_dir),
                                    'vocab_size': self.cfg['tokenizer_vocab'],
                                    'min_frequency': self.cfg['tokenizer_min_frequency'], 'lowercase': False,
                                    'model_max_length': 2048, 'max_texts': max_texts})
            cmd = [python_executable(), '-u', str(trainer), '--config', str(tok_config), '--corpus', str(corpus), '--fields', 'text',
                   '--backend', 'memory', '--vocab-size', str(self.cfg['tokenizer_vocab']),
                   '--min-frequency', str(self.cfg['tokenizer_min_frequency']), '--output-dir', str(out_dir),
                   '--max-texts', str(max_texts)]
            self.say('decision', 'progress', {'action': 'tokenizer_start', 'attempt': attempt, 'max_texts': max_texts,
                                              'vocab': self.cfg['tokenizer_vocab'], 'trainer': str(trainer)})
            limit = min(self.cfg['tokenizer_timeout_hours'] * 3600, max(900, deadline - now()))
            start = now()
            with (RUN_DIR / 'logs' / f'tokenizer_attempt{attempt}.log').open('w') as log:
                proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=str(TOKENIZER_PROJECT.parent))
                while proc.poll() is None and now() - start < limit:
                    heartbeat('tokenizer')
                    time.sleep(30)
                if proc.poll() is None:
                    proc.kill()
                    proc.wait()
                    reason = f'timeout after {limit:.0f}s'
                else:
                    reason = f'exit {proc.returncode}'
            if (out_dir / 'bbpe.json').exists():
                self.tokenizer_dir.mkdir(parents=True, exist_ok=True)
                for name in ('bbpe.json', 'vocab.json', 'training_report.json'):
                    if (out_dir / name).exists():
                        shutil.copyfile(out_dir / name, self.tokenizer_dir / name)
                report = self.tokenizer_report(seconds=now() - start)
                self.say('decision', 'report', {'action': 'tokenizer_done', **report})
                return report
            self.say('decision', 'report', {'action': 'tokenizer_failed', 'attempt': attempt, 'reason': reason})
            n = sum(1 for _ in corpus.open(encoding='utf-8'))
            max_texts = (max_texts or n) // 2  # 超时 / 内存不足：语料减半重试
        raise RuntimeError('tokenizer training failed 3 times')

    def tokenizer_report(self, seconds=None):
        tok = load_bbpe(self.tokenizer_dir / 'bbpe.json')
        rows = [json.loads(line) for line in (self.data_dir / 'tokenizer_corpus.jsonl').open(encoding='utf-8')]
        by_source = {}
        for row in rows[::max(1, len(rows) // 3000)]:
            s = by_source.setdefault(row['source'], [0, 0])
            s[0] += len(row['text'])
            s[1] += len(tok.encode(row['text']))
        report = {'vocab_size': len(tok.tokens), 'sha256': file_sha256(self.tokenizer_dir / 'bbpe.json'),
                  'chars_per_token': {k: round(c / max(1, t), 3) for k, (c, t) in by_source.items()}}
        if seconds is not None:
            report['seconds'] = round(seconds, 1)
        return report

    # ---------------- 阶段：快速编码器 ----------------
    def fast_tokenizer(self):
        from .tokenizer_hf import convert, sample_texts, verify
        hf_path = self.tokenizer_dir / 'tokenizer_hf.json'
        result_path = self.tokenizer_dir / 'fast_check.json'
        if result_path.exists():
            return read_json(result_path)
        try:
            convert(self.tokenizer_dir / 'bbpe.json', hf_path)
            texts = sample_texts(self.data_dir / 'tokenizer_corpus.jsonl', n=20000)
            check = verify(self.tokenizer_dir / 'bbpe.json', hf_path, texts)
            check['use_fast'] = check['mismatches'] <= max(1, check['checked'] // 10000)
        except Exception as exc:  # tokenizers 未安装 / 转换失败：退回慢速编码
            check = {'use_fast': False, 'error': f'{type(exc).__name__}: {exc}'}
        if not check['use_fast'] and hf_path.exists():
            hf_path.rename(hf_path.with_suffix('.rejected.json'))
        write_json(result_path, check)
        return check

    # ---------------- 阶段：流式构建 token 分片 ----------------
    def build(self, targets):
        """targets: {source: 唯一训练 token 目标}。并行处理文件直到每个来源达到目标。"""
        lists = self.file_lists()
        hf_path = self.tokenizer_dir / 'tokenizer_hf.json'
        receipts_dir = self.data_dir / 'receipts'
        receipts_dir.mkdir(exist_ok=True)
        done = {s['name']: [] for s in SOURCES}
        for path in receipts_dir.glob('*/*.json'):
            r = read_json(path)
            done[r['source']].append(r)
        committed = {k: sum(r['stats'].get('tokens_train', 0) for r in v) for k, v in done.items()}
        done_idx = {k: {r['index'] for r in v} for k, v in done.items()}
        failed = Counter()
        skipped = {k: set() for k in done}
        cursor = {k: 0 for k in done}
        ratio = {}  # 每字节文件产生多少训练 token（随完成的文件更新），用于估计在途任务
        for k, rs in done.items():
            b = sum(r['stats'].get('file_bytes', 0) for r in rs)
            if b:
                ratio[k] = committed[k] / b
        guess = {'jsonl': 0.18, 'jsonl.gz': 0.5, 'parquet': 0.6}  # 文件字节 → 训练 token 的初始估计，完成后用实测值
        specs = {s['name']: s for s in SOURCES}
        inflight = {}
        disk_full = [False]
        ctx = multiprocessing.get_context('spawn')
        last_report = now()

        def estimate(name, f):
            return int(f['size'] * ratio.get(name, guess[specs[name]['format']]))

        def next_job(name):
            files = lists[name]
            while cursor[name] < len(files) and (files[cursor[name]]['index'] in done_idx[name]
                                                 or files[cursor[name]]['index'] in skipped[name]
                                                 or any(j['source'] == name and j['index'] == files[cursor[name]]['index']
                                                        for j in inflight.values())):
                cursor[name] += 1
            if cursor[name] >= len(files):
                return None
            f = files[cursor[name]]
            pending_est = sum(j['estimate'] for j in inflight.values() if j['source'] == name)
            remaining = targets[name] - committed[name] - pending_est
            if remaining <= 0:
                return None
            spec = specs[name]
            if spec['format'] == 'parquet':
                running_dl = sum(1 for j in inflight.values() if j['format'] == 'parquet')
                if running_dl >= self.cfg['max_parallel_downloads'] or free_gb(SCRATCH_DIR) < f['size'] / 1e9 + 3:
                    return None
            if free_gb(self.token_dir) < self.cfg['disk_margin_gb']:
                disk_full[0] = True  # 数据盘只剩安全余量：不再提交新文件，已在跑的任务照常完成
                return None
            cursor[name] += 1
            job = self._job(spec, f, 'build', out_dir=str(self.token_dir / name), hf_path=str(hf_path) if hf_path.exists() else None,
                            token_cap=max(remaining, 1))
            job['estimate'] = min(estimate(name, f), remaining)
            return job

        self.say('decision', 'progress', {'action': 'build_start', 'workers': self.workers, 'targets': targets,
                                          'committed': committed, 'fast_tokenizer': hf_path.exists()})
        with ProcessPoolExecutor(max_workers=self.workers, mp_context=ctx) as pool:
            while True:
                # 依据缺口最大的来源优先提交，保证各来源进度均衡
                order = sorted(targets, key=lambda n: (committed[n] / max(1, targets[n])))
                submitted = True
                while len(inflight) < self.workers and submitted:
                    submitted = False
                    for name in order:
                        if len(inflight) >= self.workers:
                            break
                        job = next_job(name)
                        if job:
                            future = pool.submit(process_file, job)
                            inflight[future] = job
                            submitted = True
                if not inflight:
                    break
                finished, _ = wait(list(inflight), timeout=60, return_when=FIRST_COMPLETED)
                heartbeat('build')
                for future in finished:
                    job = inflight.pop(future)
                    name = job['source']
                    try:
                        result = future.result()
                    except Exception as exc:
                        failed[(name, job['index'])] += 1
                        for split in ('train', 'val', 'test'):
                            (Path(job['out_dir']) / f'{job["index"]:05d}.{split}.bin.partial').unlink(missing_ok=True)
                        if failed[(name, job['index'])] >= self.cfg['file_attempts']:
                            skipped[name].add(job['index'])
                            self.say('decision', 'report', {'action': 'file_skipped', 'source': name, 'path': job['path'],
                                                            'error': f'{type(exc).__name__}: {str(exc)[:200]}'})
                        else:
                            cursor[name] = min(cursor[name], job['index'])
                            time.sleep(10)
                        continue
                    result['stats']['file_bytes'] = job['size']
                    write_json(receipts_dir / name / f'{job["index"]:05d}.json', result)
                    done[name].append(result)
                    done_idx[name].add(job['index'])
                    committed[name] += result['stats'].get('tokens_train', 0)
                    ratio[name] = committed[name] / max(1, sum(r['stats'].get('file_bytes', 0) for r in done[name]))
                if now() - last_report > 900:
                    last_report = now()
                    self.say('decision', 'progress', {
                        'action': 'build_progress',
                        'committed_M': {k: round(v / 1e6) for k, v in committed.items()},
                        'targets_M': {k: round(v / 1e6) for k, v in targets.items()},
                        'data_disk_free_gb': round(free_gb(self.token_dir), 1)})
        manifest = self.write_manifest(done)
        short = {k: (committed[k], targets[k]) for k in targets if committed[k] < 0.9 * targets[k]}
        return manifest, short, disk_full[0]

    def write_manifest(self, done):
        tok = self.tokenizer_dir / 'bbpe.json'
        manifest = {'tokenizer': {'path': str(tok.relative_to(RUN_DIR)), 'sha256': file_sha256(tok),
                                  'vocab_size': self.cfg['tokenizer_vocab']},
                    'bos_id': BOS_ID, 'eos_id': EOS_ID, 'dtype': 'uint16', 'sources': {}}
        for name, results in done.items():
            entry = {'train': [], 'val': [], 'test': [], 'stats': Counter()}
            for r in sorted(results, key=lambda r: r['index']):
                for split in ('train', 'val', 'test'):
                    path = self.token_dir / name / f'{r["index"]:05d}.{split}.bin'
                    tokens = path.stat().st_size // 2 if path.exists() else 0
                    if tokens:
                        entry[split].append({'path': str(path.relative_to(RUN_DIR)), 'tokens': tokens})
                for k, v in r['stats'].items():
                    if isinstance(v, (int, float)) and (k.startswith(('docs_', 'chars_', 'tokens_', 'reject_')) or k == 'rows'):
                        entry['stats'][k] += v
            entry['stats'] = dict(entry['stats'])
            entry['train_tokens'] = sum(s['tokens'] for s in entry['train'])
            entry['val_tokens'] = sum(s['tokens'] for s in entry['val'])
            entry['test_tokens'] = sum(s['tokens'] for s in entry['test'])
            if entry['train_tokens']:
                manifest['sources'][name] = entry
        write_json(self.token_dir / 'manifest.json', manifest)
        return manifest
