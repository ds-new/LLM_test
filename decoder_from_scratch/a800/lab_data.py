"""实验室的数据集：构建、清单与清理。每个数据集放在 <LAB>/data/<名字>/，manifest.json 与 MixtureSampler 的格式一致，
另带默认配比 weights（= 各来源实际 token 占比）；构建完成后写 READY。构建在后台子进程里进行，每个文件一份回执，断点续跑。
所有构建都剔除与 v1 验证 / 测试文档近似重复的文档（MinHash 参数与数据工程统计相同），保证评测口径干净。

  * proxy：按对照实验 / 第一批消融的配比（7 个来源，与 v1 相同的权重），从 v1 和对照实验都没用过的文件里取约 50 亿 token，
    外加 3% 合成运算题（与对照实验同一个生成器，避开评测题）。μP 基准扫描、scaling law、调度、优化器、配比对比共用。
  * cptmix：路线 A 结束后，从其第 2 段训练分片里按来源各取一部分（合计约 15 亿 token）给 μP 宽度扫描用，其余分片删除。
  * zhfw45 / zhfw34：中文 FineWeb-Edu 只取 4-5 分档 / 3-4 分档的文件（质量对照，各约 8 亿 token），其余来源复用 proxy 的分片。
  * dd_<来源>_raw / dd_<来源>_dedup：数据工程统计出跨文件重复率高（按字符 ≥ 10%）的来源，用路线 A 第 2 段的同一批文件分别按原样 /
    全局去重构建（去重对照）。没有这样的来源就不构建。
用法：python -m decoder_from_scratch.a800.lab_data build [名字...] | status
"""

from array import array
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
import json
import multiprocessing
import os
from pathlib import Path
import random
import shutil
import sys
import time

import numpy as np

from .common import heartbeat, read_json, stamp, write_json
from .lab_core import ARITH_EVAL, BOS, EOS, LAB, V1_TOK_HF, free_gb
from .sources import SOURCES

V1 = Path('/root/autodl-tmp/a800_pretrain_v1')
V1_TOK_BBPE = V1 / 'tokenizer' / 'bbpe.json'
CPT = Path('/root/autodl-tmp/a800_cpt_v1')
DATA_ENG = Path('/root/autodl-tmp/data_eng')
USED_RECEIPTS = [V1 / 'data' / 'receipts', Path('/root/autodl-tmp/a800_tok_ab/data/receipts')]
DL = Path(os.environ.get('LAB_DL', '/dev/shm/a800_lab_dl'))
V1_MIX = {'zh_cci3': 0.28, 'zh_fwedu': 0.14, 'zh_industry': 0.13, 'en_fwedu': 0.18, 'en_cosmo': 0.07, 'code': 0.12, 'math': 0.08}
CPT_STAGE2_MIX = {'zh_cci3': 0.17, 'zh_fwedu': 0.18, 'zh_industry': 0.06, 'en_fwedu': 0.15, 'en_cosmo': 0.12, 'code': 0.15, 'math': 0.17}
PROXY_TOKENS = 5_000_000_000
ARITH_FRACTION = 0.03
BUCKET_TOKENS = 800_000_000
DEDUP_MIN_RATE = 0.10
DEDUP_TOKENS = 1_000_000_000
WORKERS = int(os.environ.get('LAB_DATA_WORKERS', '12'))
EVAL_INDEX = LAB / 'data' / 'eval_index.npz'


def log(msg):
    (LAB / 'logs').mkdir(parents=True, exist_ok=True)
    with open(LAB / 'logs' / 'data.log', 'a', encoding='utf-8') as f:
        f.write(f'{stamp()} {msg}\n')
    print(stamp(), msg, flush=True)


def ds_dir(name):
    return LAB / 'data' / name


def is_ready(name):
    return (ds_dir(name) / 'READY').exists()


def used_files(extra=()):
    used = set()
    for d in list(USED_RECEIPTS) + list(extra):
        for p in Path(d).glob('*/*.json'):
            r = read_json(p)
            used.add((r['source'], r['index']))
    return used


def tokens_per_byte():
    fl = read_json(V1 / 'data' / 'file_lists.json')
    tpb = {}
    for name, files in fl.items():
        size = {f['index']: f['size'] for f in files}
        toks = nbytes = 0
        for p in (V1 / 'data' / 'receipts' / name).glob('*.json'):
            r = read_json(p)
            if r['index'] in size:
                toks += r['stats'].get('tokens_train', 0) + r['stats'].get('tokens_val', 0) + r['stats'].get('tokens_test', 0)
                nbytes += size[r['index']]
        tpb[name] = toks / max(1, nbytes)
    return fl, tpb


def plan_source(ds, source, target, used, root=None, files=None, decon=True):
    """按文件列表顺序选没用过的文件，直到估计 token 数够 target；最后一个文件用 token_cap 截断。"""
    fl, tpb = tokens_per_byte()
    spec = {s['name']: s for s in SOURCES}[source]
    candidates = files if files is not None else fl[source]
    jobs, est = [], 0.0
    for f in candidates:
        if est >= target:
            break
        if files is None and (source, f['index']) in used:
            continue
        if root and f.get('root') != root and not f['path'].startswith(root + '/') and ('/' + root + '/') not in f['path']:
            continue
        cap = int((target - est) * 1.02)
        est += f['size'] * max(tpb.get(source, 0.25), 1e-3)
        jobs.append({'source': source, 'kind': spec['kind'], 'format': spec['format'], 'fields': spec['fields'],
                     'url': f['url'], 'path': f['path'], 'size': f['size'], 'index': f['index'], 'mode': 'build',
                     'val_fraction': 0.0, 'test_fraction': 0.0,
                     'download_path': str(DL / ds / source / f'{f["index"]:05d}.parquet'),
                     'bbpe_path': str(V1_TOK_BBPE), 'hf_path': str(V1_TOK_HF),
                     'out_dir': str(ds_dir(ds) / 'tokens' / source), 'token_cap': max(cap, 1)})
        if decon:
            jobs[-1]['eval_index'] = str(EVAL_INDEX)
    return jobs


def run_jobs(ds, jobs, fn=None):
    from .data_agent import process_file
    rec_dir = ds_dir(ds) / 'receipts'
    pending = [j for j in jobs if not (rec_dir / j['source'] / f'{j["index"]:05d}.json').exists()]
    attempts, done_n, failed = Counter(), 0, []
    log(f'{ds}: {len(pending)} 个文件待处理（共 {len(jobs)} 个）')
    with ProcessPoolExecutor(max_workers=WORKERS, mp_context=multiprocessing.get_context('spawn')) as ex:
        futs = {}
        while pending or futs:
            while pending and len(futs) < WORKERS:
                j = pending.pop(0)
                func = fn(j) if fn else (process_file_decon if j.get('eval_index') else process_file)
                futs[ex.submit(func, j)] = j
            finished, _ = wait(list(futs), timeout=60, return_when=FIRST_COMPLETED)
            for fut in finished:
                j = futs.pop(fut)
                try:
                    res = fut.result()
                    write_json(rec_dir / j['source'] / f'{j["index"]:05d}.json', res)
                    done_n += 1
                except Exception as exc:  # noqa: BLE001
                    key = (j['source'], j['index'])
                    attempts[key] += 1
                    for suf in ('train', 'val', 'test'):
                        Path(j['out_dir'], f'{j["index"]:05d}.{suf}.bin.partial').unlink(missing_ok=True)
                    if attempts[key] < 3:
                        pending.append(j)
                    else:
                        failed.append({'source': j['source'], 'index': j['index'], 'error': f'{type(exc).__name__}: {str(exc)[:300]}'})
                        log(f'{ds}: 文件 {key} 三次失败，跳过：{type(exc).__name__}: {str(exc)[:200]}')
            heartbeat(f'data {ds} {done_n}/{len(jobs)}')
    return failed


def shards_of(ds, source):
    out = []
    for p in sorted((ds_dir(ds) / 'tokens' / source).glob('*.train.bin')):
        n = p.stat().st_size // 2
        if n > 1025:
            out.append({'path': str(p), 'tokens': n})
    return out


def write_manifest(ds, sources, note, extra=None):
    totals = {n: sum(s['tokens'] for s in e['train']) for n, e in sources.items()}
    total = max(1, sum(totals.values()))
    man = {'sources': sources, 'weights': {n: t / total for n, t in totals.items() if t > 0}, 'tokens': totals,
           'total_B': round(total / 1e9, 3), 'note': note, 'built': stamp(), **(extra or {})}
    write_json(ds_dir(ds) / 'manifest.json', man)
    (ds_dir(ds) / 'READY').write_text(stamp() + '\n')
    log(f'{ds}: 完成，共 {man["total_B"]}B token：' + '，'.join(f'{n} {t / 1e9:.3f}B' for n, t in totals.items()))
    return man


def wait_disk(need_gb, what):
    waited = 0
    while free_gb(LAB) < need_gb:
        if waited == 0:
            log(f'{what}：数据盘可用 {free_gb(LAB):.1f}GB，不够（约需 {need_gb:.1f}GB），等待实验室清理里程碑后再构建')
        if waited > 24 * 3600:
            raise RuntimeError(f'{what}：等待磁盘空间超过 24 小时')
        heartbeat(f'data wait disk {what}')
        time.sleep(600)
        waited += 600


# ------------------------------------------------------------------ 评测污染：v1 验证 / 测试文档的 MinHash 索引 ----------
def band_keys(sig):
    from .data_eng import BAND_MIX, BANDS, ROWS
    sig = np.asarray(sig)
    return np.stack([(sig[:, b * ROWS:(b + 1) * ROWS].astype(np.uint64) * BAND_MIX).sum(axis=1) for b in range(BANDS)], 1)


def build_eval_index():
    if EVAL_INDEX.exists():
        return EVAL_INDEX
    from .data_eng import eval_docs, signature
    sigs = []
    for name in V1_MIX:
        for _, text in eval_docs(name):
            s = signature(text)
            if s is not None:
                sigs.append(s)
        heartbeat(f'data eval index {name}')
    sig = np.stack(sigs).astype(np.uint32)
    EVAL_INDEX.parent.mkdir(parents=True, exist_ok=True)
    tmp = EVAL_INDEX.with_name('eval_index.tmp.npz')
    np.savez(tmp, sig=sig, keys=band_keys(sig))
    os.replace(tmp, EVAL_INDEX)
    log(f'评测污染索引：{len(sig)} 篇 v1 验证 / 测试文档')
    return EVAL_INDEX


_EIDX = {}


def is_contaminated(text, path):
    from .data_eng import THRESH, signature
    if path not in _EIDX:
        z = np.load(path)
        sig, keys = z['sig'], z['keys']
        look = []
        for b in range(keys.shape[1]):
            order = np.argsort(keys[:, b], kind='stable')
            look.append((keys[order, b], order))
        _EIDX[path] = (sig, look)
    sig, look = _EIDX[path]
    s = signature(text)
    if s is None:
        return False
    k = band_keys(s[None, :])[0]
    for b, (sk, order) in enumerate(look):
        i = int(np.searchsorted(sk, k[b]))
        while i < len(sk) and sk[i] == k[b]:
            if (sig[order[i]] == s).mean() >= THRESH:
                return True
            i += 1
    return False


def process_file_decon(job):
    """与 process_file 相同，另外剔除与 v1 验证 / 测试文档近似重复的文档（回执里记为 reject_eval_contamination）。"""
    from . import cleaning
    from .data_agent import process_file
    orig = cleaning.clean_text

    def clean_text(raw, kind, *args, **kwargs):
        text, reason = orig(raw, kind, *args, **kwargs)
        if reason is None and is_contaminated(text, job['eval_index']):
            return None, 'eval_contamination'
        return text, reason

    cleaning.clean_text = clean_text
    try:
        return process_file(job)
    finally:
        cleaning.clean_text = orig


# ------------------------------------------------------------------ 合成运算题（与对照实验同一个生成器） ----------
OPS = {'+': lambda a, b: a + b, '-': lambda a, b: a - b, '×': lambda a, b: a * b}
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


def build_arith(ds, target, seed=4243):
    out = ds_dir(ds) / 'tokens' / 'arith'
    out.mkdir(parents=True, exist_ok=True)
    if (out / '00000.train.bin').exists():
        return
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(str(V1_TOK_HF))
    ev = read_json(ARITH_EVAL)
    banned = {(a, o, b) for group in (ev['problems'], ev['shots']) for items in group.values() for a, o, b, c in items}
    rng = random.Random(seed)
    buf, n = array('H'), 0
    while n < target:
        docs = [arith_doc(rng, banned) for _ in range(512)]
        for x in tok.encode_batch(docs, add_special_tokens=False):
            buf.append(BOS)
            buf.extend(x.ids)
            buf.append(EOS)
            n += len(x.ids) + 2
        heartbeat(f'data {ds} arith {n / 1e6:.0f}M')
    tmp = out / '00000.train.bin.partial'
    with open(tmp, 'wb') as fh:
        buf.tofile(fh)
    os.replace(tmp, out / '00000.train.bin')


# ------------------------------------------------------------------ 各数据集 ----------
def build_proxy(ds='proxy', total=PROXY_TOKENS):
    if is_ready(ds):
        return
    wait_disk(total * 2 * (1 + ARITH_FRACTION) / 1e9 * 1.05 + 3, ds)
    build_eval_index()
    used = used_files()
    jobs = []
    for source, w in V1_MIX.items():
        jobs += plan_source(ds, source, int(total * w), used)
    failed = run_jobs(ds, jobs)
    build_arith(ds, int(total * ARITH_FRACTION / (1 - ARITH_FRACTION)))
    sources = {s: {'train': shards_of(ds, s)} for s in list(V1_MIX) + ['arith']}
    write_manifest(ds, sources, '按 v1 配比、v1 / 对照实验没用过的文件，外加 3% 合成运算题', {'failed_files': failed})


def build_bucket(ds, root):
    """中文 FineWeb-Edu 只取某个分档目录的文件；其余来源和运算题复用 proxy 的分片。"""
    if is_ready(ds):
        return
    if not is_ready('proxy'):
        raise RuntimeError('需要先构建 proxy')
    wait_disk(BUCKET_TOKENS * 2 / 1e9 * 1.05 + 3, ds)
    build_eval_index()
    used = used_files([ds_dir('proxy') / 'receipts'])
    jobs = plan_source(ds, 'zh_fwedu', BUCKET_TOKENS, used, root=root)
    if not jobs:
        raise RuntimeError(f'{ds}：分档 {root} 没有可用的文件')
    failed = run_jobs(ds, jobs)
    proxy = read_json(ds_dir('proxy') / 'manifest.json')
    sources = {n: e for n, e in proxy['sources'].items() if n != 'zh_fwedu'}
    sources['zh_fwedu'] = {'train': shards_of(ds, 'zh_fwedu')}
    write_manifest(ds, sources, f'zh_fwedu 只取 {root} 分档；其余来源复用 proxy', {'failed_files': failed, 'bucket': root})


def dedup_sources():
    dec = read_json(DATA_ENG / 'stage2_dedup.json') or {}
    if dec.get('status') != 'done':
        return [], '数据工程统计没有完成，去重对照不做'
    out = []
    for name, d in (dec.get('sources') or {}).items():
        if d.get('apply_dup') and float(d.get('dup_rate_chars', 0)) >= DEDUP_MIN_RATE:
            out.append(name)
    rates = '，'.join(f'{n} {float(d.get("dup_rate_chars", 0)):.1%}' for n, d in (dec.get('sources') or {}).items())
    return out, f'第 2 段各来源跨文件重复率（按字符）：{rates}；阈值 {DEDUP_MIN_RATE:.0%}'


def build_dedup(source):
    """同一批文件（路线 A 第 2 段用过的该来源文件）分别按原样、按全局去重构建，各取约 10 亿 token。"""
    dec = read_json(DATA_ENG / 'stage2_dedup.json')
    recs = [read_json(p) for p in (CPT / 'data' / 'stage2' / 'receipts' / source).glob('*.json')]
    idx = {r['index'] for r in recs}
    fl = read_json(V1 / 'data' / 'file_lists.json')
    files = [f for f in fl[source] if f['index'] in idx]
    for variant in ('raw', 'dedup'):
        ds = f'dd_{source}_{variant}'
        if is_ready(ds):
            continue
        wait_disk(DEDUP_TOKENS * 2 / 1e9 * 1.05 + 3, ds)
        build_eval_index()
        jobs = plan_source(ds, source, DEDUP_TOKENS, set(), files=files, decon=(variant == 'raw'))  # dedup 版的排除清单里已含评测污染
        fn = None
        if variant == 'dedup':
            from .global_dedup import process_file_excl
            for j in jobs:
                j.update(exclude_path=dec['exclude_path'], exclude_dup=True)
            fn = lambda j: process_file_excl  # noqa: E731
        failed = run_jobs(ds, jobs, fn)
        write_manifest(ds, {source: {'train': shards_of(ds, source)}}, f'{source}：{"按原样" if variant == "raw" else "全局去重后"}',
                       {'failed_files': failed})


def make_cptmix(total=1_500_000_000, delete_rest=True):
    """路线 A 第 2 段分片的子集（按来源比例），给 μP 宽度扫描用；其余分片删除，给 proxy 腾数据盘。只在路线 A 已完成时执行。"""
    ds = 'cptmix'
    if is_ready(ds):
        return read_json(ds_dir(ds) / 'manifest.json')
    phase = (read_json(CPT / 'state.json', {}) or {}).get('phase')
    if phase != 'done':
        raise RuntimeError(f'路线 A 还没有完成（phase={phase}），不能动它的分片')
    man = read_json(CPT / 'data' / 'stage2' / 'manifest.json')
    totals = {n: sum(s['tokens'] for s in e['train']) for n, e in man['sources'].items()}
    all_t = max(1, sum(totals.values()))
    keep, sources = set(), {}
    for n, e in man['sources'].items():
        want = total * totals[n] / all_t
        got, chosen = 0, []
        for s in e['train']:
            if got >= want:
                break
            chosen.append(s)
            got += s['tokens']
            keep.add(str(Path(s['path']).resolve()))
        sources[n] = {'train': chosen}
    ds_dir(ds).mkdir(parents=True, exist_ok=True)
    out = write_manifest(ds, sources, '路线 A 第 2 段分片的子集（按来源比例）')
    if delete_rest:
        freed = 0
        for p in (CPT / 'data' / 'stage2' / 'tokens').glob('*/*.train.bin'):
            if str(p.resolve()) not in keep:
                freed += p.stat().st_size
                p.unlink()
        log(f'cptmix：删除路线 A 第 2 段其余训练分片，释放 {freed / 1e9:.1f}GB')
    return out


def build_all(names=None):
    queue = ['proxy', 'zhfw45', 'zhfw34', 'dedup']
    if names:
        queue = [q for q in queue if q in names]
    for q in queue:
        heartbeat(f'data {q}')
        if q == 'proxy':
            build_proxy()
        elif q == 'zhfw45':
            build_bucket('zhfw45', '4_5')
        elif q == 'zhfw34':
            build_bucket('zhfw34', '3_4')
        elif q == 'dedup':
            srcs, why = dedup_sources()
            write_json(ds_dir('dedup_plan.json').parent / 'dedup_plan.json', {'sources': srcs, 'why': why, 'time': stamp()})
            log('去重对照：' + why + ('；构建：' + '，'.join(srcs) if srcs else '；没有达到阈值的来源，不构建'))
            for s in srcs:
                build_dedup(s)
    shutil.rmtree(DL, ignore_errors=True)
    write_json(LAB / 'data' / 'builder_done.json', {'time': stamp(), 'queue': queue})
    log('数据构建队列完成')


if __name__ == '__main__':
    cmd = sys.argv[1] if len(sys.argv) > 1 else 'status'
    if cmd == 'build':
        build_all(sys.argv[2:] or None)
    elif cmd == 'cptmix':
        make_cptmix()
    else:
        for p in sorted((LAB / 'data').glob('*/manifest.json')):
            m = read_json(p)
            print(p.parent.name, 'READY' if (p.parent / 'READY').exists() else '-', m.get('total_B'), m.get('note'))
