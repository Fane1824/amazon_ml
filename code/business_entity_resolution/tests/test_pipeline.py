import csv
import json
from pathlib import Path
import sqlite3

import pytest

from common import file_hash, read_json, verify, write_json
from data import prepare, RECORD_HEADER, TRUTH_HEADER
from export import export, validate
from main import config, run_all
from pipeline import retrieve, features, read_rows
from learn import score, evaluate


def write_tsv(path, header, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w',encoding='utf-8',newline='') as f:
        w = csv.writer(f,delimiter='\t',lineterminator='\n')
        w.writerow(header)
        w.writerows(rows)


@pytest.fixture(scope='module')
def completed_run(tmp_path_factory):
    root = tmp_path_factory.mktemp('pipeline')
    data, run = root/'data', root/'run'
    s1, s2, s3, truth = [], [], [], []
    for i in range(240):
        country = 'US' if i%2 else 'India'
        name = f'Business number {i} Private Limited'
        address = f'{i} Orchard Road, City {i%10}'
        s1.append([f'S1-{i}',name,address,country])
        s2.append([f'S2-{i}',name.replace('Private Limited','Pvt Ltd'),address,country])
        s3.append([f'S3-{i}',f'Business number {i+1}',f'{i+80} Station Road',country])
        truth.append([f'S1-{i}', f'S2-{i}' if i%9 else ''])
    for source, rows in enumerate((s1,s2,s3),1):
        write_tsv(data/'train'/f'train_source{source}.tsv',RECORD_HEADER,rows)
    write_tsv(data/'train'/'train_ground_truth.tsv',TRUTH_HEADER,truth)
    write_tsv(data/'test'/'test_source1.tsv',RECORD_HEADER,[
        ['S1-fr1','École Lumière','12 Rue des Fleurs','France'],
        ['S1-fr2','École Lumière','12 Rue des Fleurs','France'],
        ['S1-zero','zzzz','','Unseen'],
        ['S1-ind','भारत ट्रेडिंग','12 बाजार','India']])
    write_tsv(data/'test'/'test_source2.tsv',RECORD_HEADER,[
        ['S2-fr','Ecole Lumiere','12 Rue des Fleurs','France'],
        ['S2-none','aaaa','','Unseen'],
        ['S2-ind','भारत ट्रेडिंग','12 बाजार','India']])
    write_tsv(data/'test'/'test_source3.tsv',RECORD_HEADER,[
        ['S3-fr','Ecole Lumiere','12 Rue des Fleurs','France'],
        ['S3-none','bbbb','','Unseen']])
    cfg = config(Path(__file__).parents[1]/'configs'/'mac_smoke.json')
    cfg['model']['iterations'] = 15
    cfg['batch_size'] = 50
    result = run_all(data,run,cfg)
    return data,run,cfg,result


def test_end_to_end_and_empty_candidates(completed_run):
    _,run,_,result = completed_run
    assert result['status'] == 'PASS'
    assert result['references'] == 4
    assert result['countries'] == {'France':2,'India':1,'Unseen':1}
    candidates = dict(list(csv.reader((run/'output'/'candidate_pairs.tsv').open(),delimiter='\t'))[1:])
    assert candidates['S1-zero'] == ''
    assert candidates['S1-fr1'] == candidates['S1-fr2']
    assert set(candidates['S1-fr1'].split(',')) == {'S2-fr','S3-fr'}


def test_resume_and_export_are_identical(completed_run):
    data,run,cfg,_ = completed_run
    before = {p.name:file_hash(p) for p in (run/'output').glob('*.tsv')}
    run_all(data,run,cfg)
    assert before == {p.name:file_hash(p) for p in (run/'output').glob('*.tsv')}
    # Simulate interruption before the parent score manifest was committed.
    (run/'scores'/'test'/'manifest.json').unlink()
    score(run,'test',cfg)
    (run/'output'/'manifest.json').unlink()
    export(run)
    assert before == {p.name:file_hash(p) for p in (run/'output').glob('*.tsv')}


def test_locked_evaluation_is_separate(completed_run):
    _,run,cfg,_ = completed_run
    threshold_hash = file_hash(run/'decision'/'threshold.json')
    retrieve(run,'eval',cfg)
    features(run,'eval',cfg)
    score(run,'eval',cfg)
    result = evaluate(run,'eval')
    assert 0 <= result['macro_f05'] <= 1
    assert threshold_hash == file_hash(run/'decision'/'threshold.json')


def test_incompatible_resume_is_rejected(completed_run):
    _,run,cfg,_ = completed_run
    cfg = json.loads(json.dumps(cfg))
    cfg['retrieval']['per_source_limit'] += 1
    with pytest.raises(ValueError,match='incompatible'):
        retrieve(run,'test',cfg)


@pytest.mark.parametrize('selection',['similarity_v1','adaptive_v1'])
def test_similarity_reducer_runs_on_real_index(completed_run,selection):
    from blocking import Retriever
    from data import connect,queries
    _,run,cfg,_ = completed_run
    cfg = json.loads(json.dumps(cfg))
    cfg['retrieval']['selection'] = selection
    retriever = Retriever(run,'test',cfg)
    con = connect(run/'prepared/test/records.sqlite')
    result = {q['id']:{c['id'] for c in retriever.retrieve(q)} for q in queries(con,'test')}
    con.close()
    assert result['S1-zero']==set()
    assert result['S1-fr1']==result['S1-fr2']=={'S2-fr','S3-fr'}


def test_unknown_truth_target_is_rejected(tmp_path):
    for source in (1,2,3):
        write_tsv(tmp_path/'data'/'train'/f'train_source{source}.tsv',RECORD_HEADER,
                  [[f'S{source}-x','Name','Address','Country']])
    write_tsv(tmp_path/'data'/'train'/'train_ground_truth.tsv',TRUTH_HEADER,[['S1-x','S2-missing']])
    with pytest.raises(ValueError,match='ground truth'):
        prepare(tmp_path/'data',tmp_path/'run','train',{'seed':2026})


def test_prepare_recovers_unfinished_database(completed_run, tmp_path):
    data, _, cfg, _ = completed_run
    destination = tmp_path/'prepared'/'test'
    destination.mkdir(parents=True)
    (destination/'records.sqlite.tmp').write_bytes(b'interrupted incomplete database')
    result = prepare(data, tmp_path, 'test', cfg)
    assert result['counts']['S1'] == 4
    assert not (destination/'records.sqlite.tmp').exists()
    verify(destination)
    with sqlite3.connect(destination/'records.sqlite') as con:
        assert con.execute('PRAGMA integrity_check').fetchone()[0] == 'ok'
        assert con.execute('SELECT COUNT(*) FROM records WHERE source=1').fetchone()[0] == 4


def test_strict_validation_catches_candidate_provenance(completed_run):
    _,run,_,_ = completed_run
    output = run/'output'
    path, manifest = output/'candidate_pairs.tsv', output/'manifest.json'
    original, original_meta = path.read_bytes(), manifest.read_bytes()
    rows = list(csv.reader(path.open(),delimiter='\t'))
    for row in rows[1:]:
        if row[0] == 'S1-zero':
            row[1] = 'S2-none'
    try:
        write_tsv(path,rows[0],rows[1:])
        # Update the file checksum so this test exercises semantic validation as well.
        meta = read_json(manifest)
        meta['files'][path.name] = file_hash(path)
        write_json(manifest,meta)
        with pytest.raises(ValueError,match='scored candidates'):
            validate(run)
    finally:
        path.write_bytes(original)
        manifest.write_bytes(original_meta)
