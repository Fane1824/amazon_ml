"""Streaming TSV import into an indexed, disk-backed SQLite record store."""
from __future__ import annotations

import csv
import json
import os
from pathlib import Path
import sqlite3
import time

from common import cached, chunks, complete, digest, file_hash, fingerprint, log
from text import views

RECORD_HEADER = ['entity_id', 'business_name', 'business_address', 'country']
TRUTH_HEADER = ['source1_entity_id', 'matched_entity_ids']


def read_tsv(path, header):
    with Path(path).open(encoding='utf-8', newline='') as f:
        reader = csv.reader(f, delimiter='\t')
        if next(reader, None) != header:
            raise ValueError(f'{path}: expected header {header}')
        for line, row in enumerate(reader, 2):
            if len(row) != len(header):
                raise ValueError(f'{path}:{line}: expected {len(header)} columns, got {len(row)}')
            yield row


def id_list(value):
    ids = value.split(',') if value else []
    if len(set(ids)) != len(ids) or any(not x.startswith(('S2-', 'S3-')) or x.strip() != x or not x for x in ids):
        raise ValueError(f'Malformed or duplicate target IDs: {value[:120]}')
    return ids


def connect(path, readonly=True):
    con = sqlite3.connect(f'file:{Path(path).resolve()}?mode=ro', uri=True) if readonly else sqlite3.connect(path)
    con.row_factory = sqlite3.Row
    con.execute('PRAGMA cache_size=-65536')
    con.execute('PRAGMA temp_store=FILE')
    return con


def partition(signature, seed):
    value = int(digest([seed, signature])[:12], 16) / 16**12
    return 'train' if value < .8 else 'dev' if value < .85 else 'threshold' if value < .9 else 'eval'


def prepare(data_dir, run_dir, split, cfg):
    paths = [Path(data_dir)/split/f'{split}_source{s}.tsv' for s in (1, 2, 3)]
    if split == 'train':
        paths.append(Path(data_dir)/split/'train_ground_truth.tsv')
    inputs = {p.name: {'sha256': file_hash(p), 'bytes': p.stat().st_size} for p in paths}
    key = fingerprint(inputs, cfg['seed'], split)
    out = Path(run_dir)/'prepared'/split
    old = cached(out, key)
    if old:
        return old
    out.mkdir(parents=True, exist_ok=True)
    tmp = out/'records.sqlite.tmp'
    if tmp.exists():
        tmp.unlink()
    con = connect(tmp, readonly=False)
    # This database is disposable until the final rename and manifest. A failed
    # build restarts from the TSVs, so per-batch disk journals/fsyncs buy no recovery
    # and are very expensive on rotational scratch disks. Keep read-worker caches
    # small, but give this single writer enough cache for the growing ID indexes.
    con.execute('PRAGMA journal_mode=MEMORY')
    con.execute('PRAGMA synchronous=OFF')
    con.execute('PRAGMA cache_size=-524288')  # 512 MiB; writer only.
    con.executescript('''
      CREATE TABLE records(rid INTEGER PRIMARY KEY, id TEXT NOT NULL UNIQUE,
        source INTEGER NOT NULL, country TEXT NOT NULL, name TEXT, address TEXT,
        v TEXT NOT NULL, part TEXT NOT NULL, sample_key TEXT NOT NULL);
      CREATE TABLE truth(qid TEXT PRIMARY KEY, ids TEXT NOT NULL, n INTEGER NOT NULL);
    ''')
    started = time.monotonic()
    counts = {}
    try:
        for source, path in zip((1, 2, 3), paths):
            n = 0
            for batch in chunks(read_tsv(path, RECORD_HEADER), 5000):
                values = []
                for eid, name, address, country in batch:
                    if not eid.startswith(f'S{source}-') or any(c in eid for c in ', \t\n\r'):
                        raise ValueError(f'Invalid entity ID {eid!r}')
                    country = country.strip()
                    if not country:
                        raise ValueError(f'Missing country for {eid}')
                    v = views(name, address)
                    sig = [country, v['n'], v['a']]
                    part = partition(sig, cfg['seed']) if split == 'train' and source == 1 else split
                    values.append((eid, source, country, name, address, json.dumps(v, ensure_ascii=False),
                                   part, digest([cfg['seed'], eid])))
                con.executemany('INSERT INTO records(id,source,country,name,address,v,part,sample_key) VALUES(?,?,?,?,?,?,?,?)', values)
                con.commit()
                n += len(batch)
                if n % 100000 == 0:
                    log(f'prepare {split} S{source}: {n:,}')
            counts[f'S{source}'] = n
            if not n:
                raise ValueError(f'Empty source file: {path}')
        con.executescript('''
          CREATE INDEX by_source_country ON records(source,country,rid);
          CREATE INDEX by_part ON records(source,part,sample_key);
        ''')
        if split == 'train':
            for batch in chunks(read_tsv(paths[-1], TRUTH_HEADER), 5000):
                rows = [(qid, json.dumps(id_list(ids)), len(id_list(ids))) for qid, ids in batch]
                con.executemany('INSERT INTO truth VALUES(?,?,?)', rows)
                con.commit()
            missing = con.execute('SELECT COUNT(*) FROM records r LEFT JOIN truth t ON t.qid=r.id WHERE r.source=1 AND t.qid IS NULL').fetchone()[0]
            invalid_refs = con.execute('SELECT COUNT(*) FROM truth t LEFT JOIN records r ON r.id=t.qid WHERE r.id IS NULL OR r.source!=1').fetchone()[0]
            invalid_targets = con.execute('''SELECT COUNT(*) FROM truth t JOIN records q ON q.id=t.qid,
                json_each(t.ids) j LEFT JOIN records r ON r.id=j.value
                WHERE r.id IS NULL OR r.source=1 OR r.country!=q.country''').fetchone()[0]
            if missing or invalid_refs or invalid_targets:
                raise ValueError(f'Invalid ground truth: missing={missing}, references={invalid_refs}, targets/countries={invalid_targets}')
            counts['positive_pairs'] = con.execute('SELECT SUM(n) FROM truth').fetchone()[0]
            counts['singletons'] = con.execute('SELECT COUNT(*) FROM truth WHERE n=0').fetchone()[0]
        counts['by_country'] = [dict(r) for r in con.execute('SELECT source,country,COUNT(*) AS count FROM records GROUP BY source,country')]
        con.commit()
    finally:
        con.close()
    # Flush the completed database before publishing it. Interrupted builds have
    # no completion manifest and are discarded on the next invocation.
    with tmp.open('rb') as database_file:
        os.fsync(database_file.fileno())
    os.replace(tmp, out/'records.sqlite')
    result = complete(out, key, [out/'records.sqlite'], inputs=inputs, counts=counts, seconds=time.monotonic()-started)
    log(f'prepared {split}: {counts}')
    return result


def record(row):
    item = dict(row)
    item['v'] = json.loads(item['v'])
    return item


def get_records(con, rids):
    result = {}
    for batch in chunks(sorted(set(rids)), 500):
        rows = con.execute(f'SELECT * FROM records WHERE rid IN ({",".join("?" for _ in batch)})', batch)
        result.update((r['rid'], record(r)) for r in rows)
    return result


def queries(con, part, limit=0):
    # Hash sampling is deterministic and independent of file order. The corpus is never sampled here.
    sql = 'SELECT * FROM records WHERE source=1 AND part=? ORDER BY sample_key'
    args = [part]
    if limit:
        sql += ' LIMIT ?'
        args.append(limit)
    sql = f'SELECT * FROM ({sql}) ORDER BY country,sample_key'
    for r in con.execute(sql, args):
        yield record(r)
