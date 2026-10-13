import json
import os
from pathlib import Path
import random
import subprocess
import sys
from collections import Counter

import pytest

from tokenizer_training.bbpe import BBPETokenizer, split_text
from tokenizer_training.corpus import iter_texts
from tokenizer_training.train_tokenizer import train_tokenizer


def test_hand_computed_weighted_merges_and_non_overlapping_replacement():
    tokenizer = BBPETokenizer.train(['abab', 'abab', 'aaaa'], vocab_size=263, min_frequency=1)
    a, b = 5 + ord('a'), 5 + ord('b')
    assert tokenizer.merges == [(a, b), (a, a)]
    assert tokenizer.encode('abab', add_special_tokens=False) == [261, 261]
    assert tokenizer.encode('aaaa', add_special_tokens=False) == [262, 262]
    assert tokenizer.decode([262, 262]) == 'aaaa'


def reference_merges(texts, size, min_frequency):
    # 慢速全量重计数参考实现，用于对照增量索引，不复用生产统计或替换函数。
    corpus = Counter(tuple(5 + b for b in piece.encode('utf-8')) for t in texts for piece in split_text(t))
    merges = []
    while 261 + len(merges) < size:
        counts = Counter()
        for seq, weight in corpus.items():
            for i in range(len(seq) - 1):
                counts[seq[i], seq[i + 1]] += weight
        if not counts:
            break
        pair = min(counts, key=lambda p: (-counts[p], p))
        if counts[pair] < min_frequency:
            break
        new_id = 261 + len(merges)
        changed = Counter()
        for seq, weight in corpus.items():
            out, i = [], 0
            while i < len(seq):
                if i + 1 < len(seq) and (seq[i], seq[i + 1]) == pair:
                    out.append(new_id)
                    i += 2
                else:
                    out.append(seq[i])
                    i += 1
            changed[tuple(out)] += weight
        corpus = changed
        merges.append(pair)
    return merges


def test_incremental_trainer_matches_slow_reference_including_ties():
    rng = random.Random(2026)
    for _ in range(40):
        texts = [''.join(rng.choice('aaabcb 中') for _ in range(rng.randint(1, 20))) for _ in range(12)]
        expected = reference_merges(texts, 285, 2)
        assert BBPETokenizer.train(texts, vocab_size=285, min_frequency=2).merges == expected
        assert BBPETokenizer.train(list(reversed(texts)), vocab_size=285, min_frequency=2).merges == expected


def test_bytes_unicode_literal_specials_roundtrip_and_local_reload(tmp_path):
    tokenizer = BBPETokenizer.train(['ascii only training'], vocab_size=300, min_frequency=1)
    tokenizer.save_pretrained(tmp_path / 'saved')
    loaded = BBPETokenizer.from_pretrained(tmp_path / 'saved')
    for text in ['中文𠮷🙂 café e\u0301', '  a\tB\n\r\n C  ', '\x00\x01\x7f', '[CLS][UNK][SEP]', '']:
        ids = loaded.encode(text)
        assert ids == tokenizer.encode(text)
        assert loaded.unk_token_id not in ids
        assert loaded.decode(ids, skip_special_tokens=True) == text
    raw = bytes(range(256))
    assert loaded.decode_bytes([5 + b for b in raw]) == raw
    assert not loaded.is_fast
    assert loaded.vocab_size == len(loaded)


def test_pair_padding_truncation_and_lowercase():
    tokenizer = BBPETokenizer.train(['ABC'], vocab_size=261, lowercase=True, model_max_length=7)
    encoded = tokenizer('ABC', 'DEF', truncation=True)
    assert len(encoded['input_ids']) == 7
    assert encoded['input_ids'][0] == 2
    assert encoded['input_ids'].count(3) == 2
    assert encoded['token_type_ids'][-1] == 1
    assert tokenizer.decode(tokenizer.encode('ABC'), skip_special_tokens=True) == 'abc'
    batch = tokenizer(['', 'Ab'], padding=True)
    assert batch['input_ids'][0][-1] == 0
    assert batch['attention_mask'][0][-1] == 0
    with pytest.raises(ValueError):
        tokenizer('a', 'b', truncation=True, max_length=2)


@pytest.mark.parametrize('size', [5, 256, 260])
def test_requires_full_byte_alphabet(size):
    with pytest.raises(ValueError, match='261'):
        BBPETokenizer.train(['a'], vocab_size=size)


@pytest.mark.parametrize('merges', [[[999, 5]], [[5, 6], [5, 6]], [[True, 5]], [[5]]])
def test_invalid_merge_file_is_rejected(tmp_path, merges):
    tokenizer = BBPETokenizer.train(['a'], vocab_size=261)
    tokenizer.save_pretrained(tmp_path)
    path = tmp_path / 'bbpe.json'
    payload = json.loads(path.read_text())
    payload['merges'] = merges
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError):
        BBPETokenizer.from_pretrained(tmp_path)


def test_stream_txt_jsonl_preserves_whitespace_and_counts(tmp_path):
    text = tmp_path / 'data.txt'
    text.write_text('中文\n\n  abc  \n')
    rows = tmp_path / 'data.jsonl'
    rows.write_text(json.dumps({'id': 'ignored', 'query': '问题', 'pos': [' 正例 '], 'neg': ['负例', '负例']}))
    assert list(iter_texts([text, rows], ['query', 'pos', 'neg'])) == ['中文', '  abc  ', '问题', ' 正例 ', '负例', '负例']
    assert list(iter_texts([text], max_texts=1)) == ['中文']


@pytest.mark.parametrize('row', ['not json', '[]', '{"text":4}', '{"other":"missing"}'])
def test_bad_jsonl_reports_line(tmp_path, row):
    path = tmp_path / 'bad.jsonl'
    path.write_text('\n' + row)
    with pytest.raises(ValueError, match='bad.jsonl:2'):
        list(iter_texts([path]))


def test_empty_and_failed_training_leave_no_output(tmp_path):
    path = tmp_path / 'empty.txt'
    path.write_text('\n\n')
    with pytest.raises(ValueError, match='为空'):
        train_tokenizer([path], tmp_path / 'out', vocab_size=300)
    assert not (tmp_path / 'out').exists()
    assert not list(tmp_path.glob('.tokenizer-*'))


def test_standard_library_only_cli_and_inspection(tmp_path):
    scripts = Path(__file__).resolve().parents[1]
    out = tmp_path / 'trained'
    # -S 不加载 site-packages，证明训练和检查无需第三方库。
    command = [sys.executable, '-S', str(scripts / 'train_tokenizer.py'), '--output-dir', str(out), '--max-texts', '3']
    result = subprocess.run(command, cwd=tmp_path, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads((out / 'training_report.json').read_text())
    assert report['texts'] == 3
    assert report['implementation'] == 'python-standard-library'
    assert (out / 'bbpe.json').is_file()
    inspect = [sys.executable, '-S', str(scripts / 'inspect_tokenizer.py'), '--tokenizer', str(out),
               '--text', '中文🙂', '--corpus', str(scripts / 'data/sample.txt'), '--max-texts', '2']
    result = subprocess.run(inspect, cwd=tmp_path, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    data = json.loads(result.stdout)
    assert data['samples'][0]['decoded'] == '中文🙂'
    assert data['corpus_metrics']['unknown_tokens'] == 0
    assert data['is_fast'] is False
    assert data['corpus_metrics']['texts'] == 2
    result = subprocess.run(command, cwd=tmp_path, capture_output=True, text=True, timeout=30)
    assert result.returncode != 0 and '已存在' in result.stderr


def test_encoding_uses_merge_rank_and_does_not_cross_piece_boundaries():
    a, b, c = (5 + ord(char) for char in 'abc')
    tokenizer = BBPETokenizer([(b, c), (a, b), (a, 261)])
    assert tokenizer.encode('abc', add_special_tokens=False) == [263]
    assert tokenizer.encode('a bc', add_special_tokens=False) == [a, 37, 261]
    assert BBPETokenizer.train(['a a a'], vocab_size=300, min_frequency=1).merges == []


def test_minimum_frequency_and_vocab_budget_stop_training():
    assert BBPETokenizer.train(['ab'], vocab_size=300, min_frequency=2).merges == []
    assert len(BBPETokenizer.train(['abababab'], vocab_size=262, min_frequency=1).merges) == 1


@pytest.mark.parametrize('ids', [[-1], [99999], [True]])
def test_decode_rejects_invalid_ids(ids):
    with pytest.raises(ValueError, match='无效 token ID'):
        BBPETokenizer().decode(ids)


def test_disk_backend_matches_exact_reference_and_cleans_workspace(tmp_path):
    rng = random.Random(78)
    for _ in range(12):
        texts = [''.join(rng.choice('aaabcb 中文🙂') for _ in range(rng.randint(1, 18))) for _ in range(15)]
        expected = reference_merges(texts, 295, 2)
        model = BBPETokenizer.train(texts, vocab_size=295, min_frequency=2, backend='disk', work_dir=tmp_path)
        assert model.merges == expected
        assert model.training_stats['texts'] == len(texts)
        for text in texts:
            assert model.decode(model.encode(text), skip_special_tokens=True) == text
        assert not list(tmp_path.glob('.bbpe-work-*'))


def test_disk_backend_failure_cleans_temporary_database(tmp_path):
    def broken():
        yield 'abc'
        raise ValueError('模拟读取失败')
    with pytest.raises(ValueError, match='模拟读取失败'):
        BBPETokenizer.train(broken(), backend='disk', work_dir=tmp_path)
    assert not list(tmp_path.glob('.bbpe-work-*'))


def test_disk_backend_pages_large_posting_list_without_missing_words(tmp_path):
    texts = ['aaaa' + chr(0x4e00 + i) for i in range(900)] + ['aaaa中'] * 8
    model = BBPETokenizer.train(texts, vocab_size=275, min_frequency=2, backend='disk', work_dir=tmp_path)
    assert model.merges == reference_merges(texts, 275, 2)


@pytest.mark.parametrize('code', [0, 137])
def test_training_wrapper_keeps_python_exit_code_and_logs(tmp_path, code):
    scripts = tmp_path / 'tokenizer_training'
    scripts.mkdir()
    wrapper = scripts / 'run_train.sh'
    wrapper.write_bytes((Path(__file__).resolve().parents[1] / 'run_train.sh').read_bytes())
    fake = tmp_path / 'fake-python'
    fake.write_text(f'#!/bin/sh\necho "child output"\nexit {code}\n')
    fake.chmod(0o755)
    result = subprocess.run(['bash', str(wrapper)], cwd=tmp_path, env={**os.environ, 'PYTHON': str(fake)},
                            capture_output=True, text=True, timeout=20)
    assert result.returncode == code
    assert f'Python 退出码：{code}' in result.stdout
    assert 'child output' in next((tmp_path / 'output/logs').glob('bbpe-train-*.log')).read_text()
