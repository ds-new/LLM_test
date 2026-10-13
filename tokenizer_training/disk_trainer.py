"""BBPE 的磁盘存储后端。SQLite 仅存放状态，词频更新和合并规则仍由 Python 实现。"""

from array import array
from collections import Counter
from pathlib import Path
import sqlite3
import tempfile

from tokenizer_training.bbpe import BASE_VOCAB_SIZE, BYTE_OFFSET, BBPETokenizer, merge_pair, split_text


def pack(sequence):
    return array('I', sequence).tobytes()


def unpack(blob):
    sequence = array('I')
    sequence.frombytes(blob)
    return sequence


def train_disk(texts, vocab_size, min_frequency, lowercase, model_max_length,
               progress=None, status=None, work_dir=None):
    if vocab_size > 2**32:
        raise ValueError('disk 后端的词元 ID 必须能用 32 位无符号整数表示')
    if work_dir is not None:
        Path(work_dir).mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.bbpe-work-', dir=work_dir) as temporary:
        db_path = Path(temporary) / 'training.sqlite3'
        if status:
            status(f'磁盘工作文件：{db_path}；SQLite 页面缓存 8 MiB')
        connection = sqlite3.connect(db_path)
        try:
            connection.executescript('''
                PRAGMA cache_size=-8192;
                PRAGMA temp_store=FILE;
                PRAGMA mmap_size=0;
                CREATE TABLE words (id INTEGER PRIMARY KEY, raw BLOB UNIQUE, sequence BLOB, frequency INTEGER NOT NULL);
                CREATE TABLE pairs (a INTEGER, b INTEGER, frequency INTEGER NOT NULL CHECK(frequency >= 0),
                                    PRIMARY KEY(a,b)) WITHOUT ROWID;
                CREATE TABLE positions (a INTEGER, b INTEGER, word_id INTEGER,
                                        PRIMARY KEY(a,b,word_id)) WITHOUT ROWID;
            ''')
            texts_count, pieces_count = 0, 0
            for text in texts:
                if not isinstance(text, str):
                    raise ValueError('训练文本必须为字符串')
                texts_count += 1
                for piece in split_text(text.lower() if lowercase else text):
                    connection.execute('INSERT INTO words(raw,frequency) VALUES (?,1) '
                                       'ON CONFLICT(raw) DO UPDATE SET frequency=frequency+1', (piece.encode('utf-8'),))
                    pieces_count += 1
                    if pieces_count % 10000 == 0:
                        connection.commit()
            connection.commit()
            unique = connection.execute('SELECT count(*) FROM words').fetchone()[0]
            if not unique:
                raise ValueError('训练语料为空')
            if status:
                status(f'语料聚合完成：文本 {texts_count}，唯一片段 {unique}；开始建立磁盘相邻对索引')
            # 主键分页，避免 fetchall() 把整个语料或倒排表读回内存。
            last_id, indexed = 0, 0
            while True:
                batch = connection.execute('SELECT id,raw,frequency FROM words WHERE id>? ORDER BY id LIMIT 256',
                                           (last_id,)).fetchall()
                if not batch:
                    break
                for word_id, raw, weight in batch:
                    sequence = array('I', (BYTE_OFFSET + byte for byte in raw))
                    for pair, n in Counter(zip(sequence, sequence[1:])).items():
                        connection.execute('INSERT INTO pairs VALUES (?,?,?) ON CONFLICT(a,b) '
                                           'DO UPDATE SET frequency=frequency+excluded.frequency', (*pair, n * weight))
                        connection.execute('INSERT INTO positions VALUES (?,?,?)', (*pair, word_id))
                    # 原始字节只用于去重，索引建立后释放其逻辑存储。
                    connection.execute('UPDATE words SET sequence=?,raw=NULL WHERE id=?', (pack(sequence), word_id))
                connection.commit()
                last_id = batch[-1][0]
                indexed += len(batch)
                if status and (indexed % 25600 == 0 or indexed == unique):
                    status(f'建立磁盘索引：{indexed}/{unique}')
            connection.execute('CREATE INDEX best_pair ON pairs(frequency DESC,a,b)')
            connection.commit()
            if status:
                status(f'磁盘索引完成：数据库 {db_path.stat().st_size / 1024**2:.1f} MiB')
            merges = []
            if progress:
                progress(0, None, 0)
            while BASE_VOCAB_SIZE + len(merges) < vocab_size:
                candidate = connection.execute('SELECT a,b,frequency FROM pairs ORDER BY frequency DESC,a,b LIMIT 1').fetchone()
                if candidate is None or candidate[2] < min_frequency:
                    break
                a, b, frequency = candidate
                pair, new_id, after_id = (a, b), BASE_VOCAB_SIZE + len(merges), 0
                while True:
                    word_ids = connection.execute('SELECT word_id FROM positions WHERE a=? AND b=? AND word_id>? '
                                                  'ORDER BY word_id LIMIT 256', (a, b, after_id)).fetchall()
                    if not word_ids:
                        break
                    for (word_id,) in word_ids:
                        blob, weight = connection.execute('SELECT sequence,frequency FROM words WHERE id=?', (word_id,)).fetchone()
                        old = unpack(blob)
                        new = merge_pair(old, pair, new_id)
                        before, after = Counter(zip(old, old[1:])), Counter(zip(new, new[1:]))
                        # 差分更新。相同权重的未变化相邻对无需写盘。
                        for key in before.keys() | after.keys():
                            delta = (after[key] - before[key]) * weight
                            if delta > 0:
                                connection.execute('INSERT INTO pairs VALUES (?,?,?) ON CONFLICT(a,b) '
                                                   'DO UPDATE SET frequency=frequency+excluded.frequency', (*key, delta))
                            elif delta < 0:
                                connection.execute('UPDATE pairs SET frequency=frequency+? WHERE a=? AND b=?',
                                                   (delta, *key))
                        for key in before.keys() - after.keys():
                            connection.execute('DELETE FROM positions WHERE a=? AND b=? AND word_id=?', (*key, word_id))
                        for key in after.keys() - before.keys():
                            connection.execute('INSERT INTO positions VALUES (?,?,?)', (*key, word_id))
                        connection.execute('UPDATE words SET sequence=? WHERE id=?', (pack(new), word_id))
                    # 已处理的倒排项会被删除，始终通过主键继续分页。
                    after_id = word_ids[-1][0]
                    connection.commit()
                merges.append(pair)
                if progress:
                    progress(len(merges), pair, frequency)
            model = BBPETokenizer(merges, lowercase=lowercase, model_max_length=model_max_length)
            model.training_stats = {'texts': texts_count, 'unique_pieces': unique, 'merges': len(merges), 'backend': 'disk',
                                    'work_database_bytes': db_path.stat().st_size}
            return model
        finally:
            connection.close()
