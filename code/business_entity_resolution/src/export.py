"""Submission exports and disk-backed, strict ID/provenance checks."""
import csv
from itertools import zip_longest
import json
import os
from pathlib import Path

from common import cached, complete, fingerprint, read_json, verify, write_json
from data import connect, id_list, read_tsv
from learn import scored_rows

MATCH_HEADER = ['source1_entity_id', 'matched_entity_ids']
CANDIDATE_HEADER = ['source1_entity_id', 'candidate_entity_ids']


def export(run_dir):
    root = Path(run_dir)
    score_meta = verify(root/'scores'/'test')
    decision_meta = verify(root/'decision')
    decision = read_json(root/'decision'/'threshold.json')
    if decision['model_hash'] != score_meta['model_hash']:
        raise ValueError('Model and threshold do not match')
    out = root/'output'
    key = fingerprint(score_meta['fingerprint'], decision_meta['fingerprint'])
    if cached(out, key):
        return
    out.mkdir(parents=True, exist_ok=True)
    matching, candidates = out/'matching_results.tsv', out/'candidate_pairs.tsv'
    with matching.with_suffix('.tmp').open('w', encoding='utf-8', newline='') as mf, candidates.with_suffix('.tmp').open('w', encoding='utf-8', newline='') as cf:
        mw, cw = csv.writer(mf, delimiter='\t', lineterminator='\n'), csv.writer(cf, delimiter='\t', lineterminator='\n')
        mw.writerow(MATCH_HEADER)
        cw.writerow(CANDIDATE_HEADER)
        for row in scored_rows(root, 'test'):
            ids = [c['id'] for c in row['candidates']]
            accepted = [c['id'] for c,s in zip(row['candidates'],row['scores']) if s >= decision['threshold']]
            mw.writerow([row['id'], ','.join(sorted(accepted))])
            cw.writerow([row['id'], ','.join(sorted(ids))])
    os.replace(matching.with_suffix('.tmp'), matching)
    os.replace(candidates.with_suffix('.tmp'), candidates)
    complete(out, key, [matching,candidates], references=score_meta['references'])


def validate(run_dir):
    root = Path(run_dir)
    verify(root/'prepared'/'test')
    verify(root/'output')
    verify(root/'decision')
    decision = read_json(root/'decision'/'threshold.json')
    score_meta = verify(root/'scores'/'test')
    if score_meta['model_hash'] != decision['model_hash']:
        raise ValueError('Scored model differs from threshold model')
    con = connect(root/'prepared'/'test'/'records.sqlite')
    # TEMP objects are writable on a read-only main database; they spill to disk.
    con.execute('CREATE TEMP TABLE seen(id TEXT PRIMARY KEY)')
    expected = con.execute('SELECT COUNT(*) FROM records WHERE source=1').fetchone()[0]
    count = pairs = matches = 0
    countries = {}
    streams = (read_tsv(root/'output'/'matching_results.tsv', MATCH_HEADER),
               read_tsv(root/'output'/'candidate_pairs.tsv', CANDIDATE_HEADER),
               scored_rows(root,'test'))
    try:
        for m,c,s in zip_longest(*streams):
            if m is None or c is None or s is None:
                raise ValueError('Output/scored-reference row counts differ')
            if m[0] != c[0] or m[0] != s['id']:
                raise ValueError('Output rows differ from scored-reference order')
            q = con.execute('SELECT source,country FROM records WHERE id=?', (m[0],)).fetchone()
            if not q or q['source'] != 1:
                raise ValueError(f'Unknown reference {m[0]}')
            con.execute('INSERT INTO seen VALUES(?)', (m[0],))
            mids, cids = set(id_list(m[1])), set(id_list(c[1]))
            if not mids <= cids:
                raise ValueError('Match outside candidate set')
            if cids != {t['id'] for t in s['candidates']}:
                raise ValueError('Exported candidates differ from scored candidates')
            accepted = {t['id'] for t,p in zip(s['candidates'],s['scores']) if p >= decision['threshold']}
            if mids != accepted:
                raise ValueError('Exported matches differ from threshold decisions')
            if cids:
                valid = con.execute(f'SELECT COUNT(*) FROM records WHERE source IN (2,3) AND id IN ({",".join("?" for _ in cids)})', sorted(cids)).fetchone()[0]
                if valid != len(cids):
                    raise ValueError('Unknown target ID')
            count += 1
            pairs += len(cids)
            matches += len(mids)
            countries[q['country']] = countries.get(q['country'],0)+1
        if count != expected:
            raise ValueError(f'Missing references: expected {expected}, got {count}')
    finally:
        con.close()
    result = dict(status='PASS', references=count, candidates=pairs, matches=matches, countries=countries,
                  model_hash=score_meta['model_hash'], threshold=decision['threshold'])
    write_json(root/'reports'/'submission_validation.json', result)
    return result
