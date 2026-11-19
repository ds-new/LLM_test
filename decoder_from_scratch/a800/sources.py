"""数据来源（全部从 ModelScope 下载）与文件枚举。

配比按"唯一 token 占比"设定：中文约 55%、英文约 25%、代码约 12%、数学约 8%。
这是第二档方案的起始配方（不是证明过的最优配比），训练中决策智能体只在出现遗忘/过度重复时微调采样权重。
"""

import json
import time
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen

API = 'https://www.modelscope.cn/api/v1/datasets'

INDUSTRY_ZH = ['edu', 'tech', 'med', 'law', 'fin', 'lit', 'new', 'travel', 'agri', 'sport', 'film',
               'politic', 'auto', 'comp', 'emo']

SOURCES = [
    {'name': 'zh_cci3', 'dataset': 'BAAI/CCI3-HQ', 'revision': 'master', 'roots': ['data'],
     'format': 'jsonl', 'fields': ['text', 'content'], 'kind': 'zh', 'weight': 0.28,
     'about': '智源 CCI3.0-HQ：两阶段过滤的高质量中文网页'},
    {'name': 'zh_fwedu', 'dataset': 'opencsg/Fineweb-Edu-Chinese-V2.1', 'revision': 'master',
     'roots': ['4_5', '3_4'], 'format': 'parquet', 'fields': ['text', 'content'], 'kind': 'zh', 'weight': 0.14,
     'about': 'OpenCSG 中文 FineWeb-Edu：按教育价值打分，先用 4-5 分再用 3-4 分'},
    {'name': 'zh_industry', 'dataset': 'BAAI/IndustryCorpus', 'revision': '8a03761a1a792ee7aa68e44496ed47c60aee1c30',
     'roots': [f'IndustryCorpus/zh/{d}' for d in INDUSTRY_ZH], 'format': 'jsonl.gz', 'fields': ['text'],
     'kind': 'zh', 'weight': 0.13, 'interleave': True,
     'about': '智源行业语料（中文 15 个行业轮流取文件，保证领域多样性）'},
    {'name': 'en_fwedu', 'dataset': 'AI-ModelScope/smollm-corpus', 'revision': 'master',
     'roots': ['fineweb-edu-dedup'], 'format': 'parquet', 'fields': ['text'], 'kind': 'en', 'weight': 0.18,
     'about': 'FineWeb-Edu（去重版）：英文教育类网页'},
    {'name': 'en_cosmo', 'dataset': 'AI-ModelScope/smollm-corpus', 'revision': 'master',
     'roots': ['cosmopedia-v2'], 'format': 'parquet', 'fields': ['text'], 'kind': 'en', 'weight': 0.07,
     'about': 'Cosmopedia v2：大模型生成的英文教材、故事，对小模型很有效'},
    {'name': 'code', 'dataset': 'AI-ModelScope/starcoderdata', 'revision': 'master',
     'roots': ['python', 'python', 'javascript', 'java', 'cpp', 'c', 'go', 'typescript', 'shell'],
     'format': 'parquet', 'fields': ['content', 'text'], 'kind': 'code', 'weight': 0.12, 'interleave': True,
     'about': 'StarCoderData：GitHub 代码，Python 占双倍份额'},
    {'name': 'math', 'dataset': 'AI-ModelScope/finemath', 'revision': 'master', 'roots': ['finemath-4plus'],
     'format': 'parquet', 'fields': ['text'], 'kind': 'math', 'weight': 0.08,
     'about': 'FineMath 4+：高质量数学网页（含推导步骤）'},
]

SUFFIX = {'jsonl': '.jsonl', 'jsonl.gz': '.jsonl.gz', 'parquet': '.parquet'}


def http_json(url, attempts=4):
    for i in range(attempts):
        try:
            with urlopen(Request(url, headers={'User-Agent': 'decoder-a800/1.0'}), timeout=60) as r:
                return json.loads(r.read())
        except Exception:
            if i == attempts - 1:
                raise
            time.sleep(5 * (i + 1))


def list_files(dataset, revision, root, suffix):
    files, page = [], 1
    while True:
        query = urlencode(dict(Revision=revision, Root=root, Recursive='false', PageNumber=page, PageSize=500))
        data = http_json(f'{API}/{dataset}/repo/tree?{query}')
        if data.get('Code') != 200:
            raise RuntimeError(f'ModelScope list failed for {dataset}/{root}: {data.get("Message")}')
        batch = data['Data']['Files']
        files += [{'path': f['Path'], 'size': f['Size'], 'sha256': f.get('Sha256', '')}
                  for f in batch if f['Type'] == 'blob' and f['Path'].endswith(suffix)]
        if len(batch) < 500:
            break
        page += 1
    return sorted(files, key=lambda f: f['path'])


def download_url(dataset, revision, path):
    return f'{API}/{dataset}/repo?' + urlencode(dict(Source='SDK', Revision=revision, FilePath=path, View='False'))


def enumerate_source(spec):
    """返回该来源按处理顺序排好的文件列表；interleave=True 时在多个子目录之间轮流取文件。"""
    suffix = SUFFIX[spec['format']]
    per_root, seen_roots = [], {}
    for root in spec['roots']:
        if root not in seen_roots:
            seen_roots[root] = list_files(spec['dataset'], spec['revision'], root, suffix)
        per_root.append(seen_roots[root])
    ordered, used = [], set()
    if spec.get('interleave'):
        cursors = [0] * len(per_root)
        progress = True
        while progress:
            progress = False
            for i, files in enumerate(per_root):
                while cursors[i] < len(files) and files[cursors[i]]['path'] in used:
                    cursors[i] += 1
                if cursors[i] < len(files):
                    ordered.append(files[cursors[i]])
                    used.add(files[cursors[i]]['path'])
                    cursors[i] += 1
                    progress = True
    else:
        for files in per_root:
            for f in files:
                if f['path'] not in used:
                    ordered.append(f)
                    used.add(f['path'])
    for i, f in enumerate(ordered):
        f['index'] = i
        f['url'] = download_url(spec['dataset'], spec['revision'], f['path'])
        f['root'] = f['path'].rsplit('/', 1)[0]
    return ordered
