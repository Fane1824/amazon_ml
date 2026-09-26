"""Create a deliberately small execution fixture from real records, never a benchmark."""
import csv
from pathlib import Path

from common import file_hash, log, read_json, write_json
from data import RECORD_HEADER, TRUTH_HEADER, id_list, read_tsv


def make_smoke(data_dir, out, references=2000, distractors=3000):
    data_dir, out = Path(data_dir).resolve(), Path(out).resolve()
    if data_dir == out or data_dir in out.parents:
        raise ValueError('Put smoke data outside the original dataset directory')
    if out.exists() and any(out.iterdir()):
        marker = out/'SMOKE_ONLY.json'
        if marker.exists():
            meta = read_json(marker)
            if meta['source'] != str(data_dir) or meta['references'] != references or meta['distractors_per_source'] != distractors:
                raise ValueError('Existing smoke dataset uses different settings')
            for name, expected in meta['files'].items():
                if file_hash(out/name) != expected:
                    raise ValueError(f'Modified smoke data: {name}')
            return
        raise ValueError('Smoke destination is not empty; use a new directory')
    out.mkdir(parents=True, exist_ok=True)
    def write(path, header, rows):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('w', encoding='utf-8', newline='') as f:
            w = csv.writer(f, delimiter='\t', lineterminator='\n')
            w.writerow(header)
            w.writerows(rows)
    train_refs = []
    counts = {}
    for row in read_tsv(data_dir/'train'/'train_source1.tsv', RECORD_HEADER):
        country = row[3]
        if counts.get(country,0) < references//2:
            train_refs.append(row)
            counts[country] = counts.get(country,0)+1
        if len(train_refs) >= references:
            break
    wanted = {r[0] for r in train_refs}
    labels, targets = [], set()
    for row in read_tsv(data_dir/'train'/'train_ground_truth.tsv', TRUTH_HEADER):
        if row[0] in wanted:
            labels.append(row)
            targets.update(id_list(row[1]))
    if len(labels) != len(wanted):
        raise ValueError('Missing smoke reference labels')
    write(out/'train'/'train_source1.tsv', RECORD_HEADER, train_refs)
    write(out/'train'/'train_ground_truth.tsv', TRUTH_HEADER, labels)
    for source in (2,3):
        selected, extras = [], 0
        for row in read_tsv(data_dir/'train'/f'train_source{source}.tsv', RECORD_HEADER):
            if row[0] in targets:
                selected.append(row)
            elif extras < distractors:
                selected.append(row)
                extras += 1
        write(out/'train'/f'train_source{source}.tsv', RECORD_HEADER, selected)
        log(f'smoke train S{source}: {len(selected):,} targets (all selected labels + distractors)')
    test_refs, counts = [], {}
    for row in read_tsv(data_dir/'test'/'test_source1.tsv', RECORD_HEADER):
        if counts.get(row[3],0) < 20:
            test_refs.append(row)
            counts[row[3]] = counts.get(row[3],0)+1
        if all(counts.get(c,0) >= 20 for c in ('US','India','France')):
            break
    write(out/'test'/'test_source1.tsv', RECORD_HEADER, test_refs)
    for source in (2,3):
        selected = []
        for row in read_tsv(data_dir/'test'/f'test_source{source}.tsv', RECORD_HEADER):
            selected.append(row)
            if len(selected) >= distractors:
                break
        write(out/'test'/f'test_source{source}.tsv', RECORD_HEADER, selected)
    write_json(out/'SMOKE_ONLY.json', dict(source=str(data_dir), references=references,
               distractors_per_source=distractors,
               warning='Execution test only. Reduced target corpus. Not a valid competition submission or accuracy estimate.',
               files={str(p.relative_to(out)): file_hash(p) for p in sorted(out.rglob('*.tsv'))}))
