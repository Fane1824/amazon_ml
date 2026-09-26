"""Resumable query shards. Parallel work queues are explicitly bounded."""
from collections import deque
from concurrent.futures import ProcessPoolExecutor
import json
import os
from pathlib import Path
import time

import numpy as np

from blocking import Retriever
from common import cached, chunks, complete, fingerprint, log, read_json, verify, write_json
from data import connect, get_records, queries
from features import matrix, feature_names
from metrics import f05

_worker = None


def init_worker(run_dir, split, cfg):
    global _worker
    _worker = (connect(Path(run_dir)/'prepared'/split/'records.sqlite'), Retriever(run_dir, split, cfg))


def read_rows(path):
    with Path(path).open(encoding='utf-8') as f:
        for line in f:
            yield json.loads(line)


def write_rows(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name+'.tmp')
    with tmp.open('w', encoding='utf-8') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, separators=(',', ':'), allow_nan=False)+'\n')
    os.replace(tmp, path)


def retrieve_job(task):
    out, key, batch = task
    out = Path(out)
    old = cached(out, key)
    if old:
        return old
    con, retriever = _worker
    rows = []
    for q in batch:
        truth = con.execute('SELECT ids FROM truth WHERE qid=?', (q['id'],)).fetchone()
        rows.append(dict(rid=q['rid'], id=q['id'], country=q['country'],
                         truth=json.loads(truth[0]) if truth else [], candidates=retriever.retrieve(q)))
    path = out/'queries.jsonl'
    write_rows(path, rows)
    return complete(out, key, [path], references=len(rows), pairs=sum(len(r['candidates']) for r in rows))


def feature_job(task):
    out, key, source = task
    out = Path(out)
    old = cached(out, key)
    if old:
        return old
    con, retriever = _worker
    rows = list(read_rows(Path(source)/'queries.jsonl'))
    ids = [r['rid'] for r in rows] + [c['rid'] for r in rows for c in r['candidates']]
    records = get_records(con, ids)
    x, y, weights, offsets, names = matrix(rows, records, retriever)
    if not np.isfinite(x).all():
        raise ValueError('Non-finite features')
    out.mkdir(parents=True, exist_ok=True)
    path = out/'features.npz'
    with (out/'features.npz.tmp').open('wb') as f:
        np.savez_compressed(f, x=x, y=y, weights=weights, offsets=offsets)
    os.replace(out/'features.npz.tmp', path)
    return complete(out, key, [path], references=len(rows), pairs=len(y), feature_names=names)


def bounded_map(fn, tasks, workers, initializer, initargs):
    if workers == 1:
        initializer(*initargs)
        yield from map(fn, tasks)
        return
    with ProcessPoolExecutor(max_workers=workers, initializer=initializer, initargs=initargs) as executor:
        pending = deque()
        for task in tasks:
            pending.append(executor.submit(fn, task))
            if len(pending) >= 2*workers:
                yield pending.popleft().result()
        while pending:
            yield pending.popleft().result()


def retrieve(run_dir, part, cfg):
    root = Path(run_dir)
    split = 'test' if part == 'test' else 'train'
    prepared = verify(root/'prepared'/split)
    indexes = verify(root/'indexes'/split)
    for country in indexes['countries']:
        from blocking import country_key
        verify(root/'indexes'/split/country_key(country))
    key = fingerprint(prepared['fingerprint'], indexes['fingerprint'], cfg['retrieval'], cfg['samples'].get(part, 0), cfg['batch_size'], part)
    out = root/'candidates'/part
    old = cached(out, key)
    if old:
        return old
    con = connect(root/'prepared'/split/'records.sqlite')
    paths = []
    def tasks():
        limit = 0 if part == 'test' else cfg['samples'][part]
        for i, batch in enumerate(chunks(queries(con, part, limit), cfg['batch_size'])):
            path = out/f'{i:06d}'
            paths.append(path/'manifest.json')
            yield (str(path), fingerprint(key, [q['id'] for q in batch]), batch)
    refs = pairs = 0
    start = time.monotonic()
    for i, result in enumerate(bounded_map(retrieve_job, tasks(), cfg['workers'], init_worker, (str(root), split, cfg))):
        refs += result['references']
        pairs += result['pairs']
        log(f'retrieve {part} shard {i}: {refs:,} references, {pairs:,} pairs')
    con.close()
    if not refs:
        raise ValueError(f'No references in {part}; increase the data sample')
    return complete(out, key, paths, references=refs, pairs=pairs, seconds=time.monotonic()-start,
                    shards=[str(p.parent.relative_to(out)) for p in paths])


def features(run_dir, part, cfg):
    root = Path(run_dir)
    source = root/'candidates'/part
    candidates = verify(source)
    out = root/'features'/part
    key = fingerprint(candidates['fingerprint'], feature_names())
    old = cached(out, key)
    if old:
        return old
    tasks, paths = [], []
    for shard in candidates['shards']:
        meta = verify(source/shard)
        paths.append(out/shard/'manifest.json')
        tasks.append((str(out/shard), fingerprint(key, meta['fingerprint']), str(source/shard)))
    start = time.monotonic()
    for i, result in enumerate(bounded_map(feature_job, tasks, cfg['workers'], init_worker,
                                          (str(root), 'test' if part == 'test' else 'train', cfg))):
        log(f'features {part} shard {i}: {result["pairs"]:,} pairs')
    return complete(out, key, paths, references=candidates['references'], pairs=candidates['pairs'],
                    feature_names=feature_names(), shards=candidates['shards'], seconds=time.monotonic()-start)


def blocking_report(run_dir, part):
    root = Path(run_dir)/'candidates'/part
    meta = verify(root)
    counts, scores, recalls, countries = [], [], [], {}
    recovered = total = all_found = none_found = nonsingletons = 0
    for shard in meta['shards']:
        verify(root/shard)
        for row in read_rows(root/shard/'queries.jsonl'):
            truth = set(row['truth'])
            found = {c['id'] for c in row['candidates']} & truth
            counts.append(len(row['candidates']))
            score = f05(truth, found)
            scores.append(score)
            stats = countries.setdefault(row['country'], dict(references=0, truth=0, recovered=0, oracle_sum=0.))
            stats['references'] += 1
            stats['truth'] += len(truth)
            stats['recovered'] += len(found)
            stats['oracle_sum'] += score
            if truth:
                recalls.append(len(found)/len(truth))
                recovered += len(found)
                total += len(truth)
                nonsingletons += 1
                all_found += found == truth
                none_found += not found
    result = dict(references=len(counts), mean_candidates=float(np.mean(counts)),
                  candidate_percentiles=dict(zip(('median','p95','p99','max'), map(float, np.percentile(counts,[50,95,99,100])))))
    if part != 'test':
        result.update(positive_pair_recall=recovered/total if total else None,
                      mean_reference_recall=float(np.mean(recalls)) if recalls else None,
                      oracle_macro_f05=float(np.mean(scores)),
                      all_matches_fraction=all_found/nonsingletons if nonsingletons else None,
                      no_matches_fraction=none_found/nonsingletons if nonsingletons else None,
                      candidate_precision=recovered/sum(counts) if sum(counts) else 0.,
                      per_country={c: dict(references=s['references'],
                                           recall=s['recovered']/s['truth'] if s['truth'] else None,
                                           oracle_macro_f05=s['oracle_sum']/s['references']) for c,s in countries.items()})
    write_json(Path(run_dir)/'reports'/f'blocking_{part}.json', result)
    return result
