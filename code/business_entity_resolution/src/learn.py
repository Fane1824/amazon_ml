"""CatBoost baseline, CPU scoring, and independent decision calibration."""
import os
from pathlib import Path
import time

from catboost import CatBoostClassifier, Pool
import numpy as np

from common import cached, complete, file_hash, fingerprint, log, read_json, verify, write_json
from features import feature_names
from metrics import summarize
from pipeline import read_rows, write_rows


def training_arrays(root, part, out):
    meta = verify(root/'features'/part)
    n, width = meta['pairs'], len(meta['feature_names'])
    if not n:
        raise ValueError(f'{part} has no candidates')
    arrays = [np.lib.format.open_memmap(out/f'{part}_{key}.npy', mode='w+', dtype=dtype, shape=shape)
              for key, dtype, shape in [('x', 'float32', (n,width)), ('y', 'uint8', (n,)), ('weights', 'float32', (n,))]]
    pos = 0
    for shard in meta['shards']:
        path = root/'features'/part/shard
        verify(path)
        with np.load(path/'features.npz', allow_pickle=False) as data:
            size = len(data['y'])
            for arr, key in zip(arrays, ('x','y','weights')):
                arr[pos:pos+size] = data[key]
            pos += size
    if pos != n or set(np.unique(arrays[1])) != {0,1}:
        raise ValueError(f'{part} must contain both positive and negative candidate pairs')
    for arr in arrays:
        arr.flush()
    return arrays


def train(run_dir, cfg):
    root = Path(run_dir)
    train_meta, dev_meta = verify(root/'features'/'train'), verify(root/'features'/'dev')
    key = fingerprint(train_meta['fingerprint'], dev_meta['fingerprint'], cfg['model'])
    out = root/'model'
    old = cached(out, key)
    if old:
        return old
    out.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    arrays = training_arrays(root, 'train', out)
    dev = training_arrays(root, 'dev', out)
    names = feature_names()
    log(f'train: {len(arrays[1]):,} pairs, {len(names)} features, device={cfg["model"]["task_type"]}')
    training = Pool(arrays[0], label=arrays[1], weight=arrays[2], feature_names=names)
    validation = Pool(dev[0], label=dev[1], weight=dev[2], feature_names=names)
    params = dict(cfg['model'])
    params.update(random_seed=cfg['seed'], loss_function='Logloss', eval_metric='Logloss',
                  allow_writing_files=False)
    if params['task_type'] == 'CPU':
        params.pop('devices', None)
        params.pop('gpu_ram_part', None)
    model = CatBoostClassifier(**params)
    model.fit(training, eval_set=validation, use_best_model=True, verbose=100)
    model.save_model(str(out/'model.cbm.tmp'))
    os.replace(out/'model.cbm.tmp', out/'model.cbm')
    write_json(out/'schema.json', dict(feature_names=names))
    write_json(out/'config.json', cfg)
    result = complete(out, key, [out/'model.cbm', out/'schema.json', out/'config.json'],
                      best_iteration=model.get_best_iteration(), parameters=params,
                      seconds=time.monotonic()-start)
    del training, validation, arrays, dev
    # These are rebuildable assembly buffers, not model artifacts.
    for part in ('train','dev'):
        for name in ('x','y','weights'):
            (out/f'{part}_{name}.npy').unlink()
    return result


def score(run_dir, part, cfg):
    root = Path(run_dir)
    model_meta, feature_meta = verify(root/'model'), verify(root/'features'/part)
    candidates = verify(root/'candidates'/part)
    key = fingerprint(model_meta['fingerprint'], feature_meta['fingerprint'], candidates['fingerprint'])
    out = root/'scores'/part
    old = cached(out, key)
    if old:
        return old
    if read_json(root/'model'/'schema.json')['feature_names'] != feature_meta['feature_names']:
        raise ValueError('Model/feature schema mismatch')
    model = CatBoostClassifier()
    model.load_model(str(root/'model'/'model.cbm'))
    paths = []
    for shard in feature_meta['shards']:
        dest = out/shard
        fm, cm = verify(root/'features'/part/shard), verify(root/'candidates'/part/shard)
        skey = fingerprint(key, fm['fingerprint'], cm['fingerprint'])
        paths.append(dest/'manifest.json')
        if cached(dest, skey):
            continue
        rows = list(read_rows(root/'candidates'/part/shard/'queries.jsonl'))
        with np.load(root/'features'/part/shard/'features.npz', allow_pickle=False) as data:
            x, offsets = data['x'], data['offsets']
            predictions = model.predict_proba(x, thread_count=cfg['model']['thread_count'])[:,1] if len(x) else np.empty(0)
            for i, row in enumerate(rows):
                values = predictions[offsets[i]:offsets[i+1]].tolist()
                if len(values) != len(row['candidates']):
                    raise ValueError('Candidate/score alignment failure')
                row['scores'] = values
        path = dest/'scores.jsonl'
        write_rows(path, rows)
        complete(dest, skey, [path], references=len(rows), pairs=len(predictions))
        log(f'score {part} shard {shard}: {len(predictions):,} pairs')
    return complete(out, key, paths, shards=feature_meta['shards'], references=candidates['references'],
                    model_hash=file_hash(root/'model'/'model.cbm'))


def scored_rows(root, part):
    meta = verify(root/'scores'/part)
    for shard in meta['shards']:
        verify(root/'scores'/part/shard)
        yield from read_rows(root/'scores'/part/shard/'scores.jsonl')


def calibrate(run_dir, cfg):
    root = Path(run_dir)
    meta = verify(root/'scores'/'threshold')
    key = fingerprint(meta['fingerprint'], cfg['threshold'])
    out = root/'decision'
    old = cached(out, key)
    if old:
        return read_json(out/'threshold.json')
    thresholds = np.append(np.arange(1001)/1000., 1.000001)
    sums, counts = {}, {}
    for row in scored_rows(root, 'threshold'):
        country = row['country']
        truth = set(row['truth'])
        scores = np.asarray(row['scores'])
        order = np.argsort(-scores, kind='stable')
        labels = np.asarray([c['id'] in truth for c in row['candidates']], dtype=np.int64)
        cumulative = np.concatenate(([0], np.cumsum(labels[order])))
        predicted = np.searchsorted(-scores[order], -thresholds, side='right')
        values = (1.25*cumulative[predicted]/(predicted+.25*len(truth))) if truth else (predicted == 0).astype(float)
        sums[country] = sums.get(country, np.zeros(len(thresholds))) + values
        counts[country] = counts.get(country, 0) + 1
    if not counts:
        raise ValueError('Empty threshold partition')
    official = sum(sums.values())/sum(counts.values())
    weights = cfg['threshold']['country_weights']
    if weights and not set(weights).issubset(counts):
        raise ValueError(f'Threshold partition lacks configured countries: {set(weights)-set(counts)}')
    objective = sum(weights[c]*sums[c]/counts[c] for c in weights)/sum(weights.values()) if weights else official
    eligible = np.flatnonzero(objective >= objective.max()-cfg['threshold']['tie_tolerance'])
    idx = int(eligible[-1])
    result = dict(threshold=float(thresholds[idx]), official_macro_f05=float(official[idx]),
                  selection_score=float(objective[idx]), country_weights=weights,
                  model_hash=meta['model_hash'], references=sum(counts.values()))
    write_json(out/'threshold.json', result)
    write_json(out/'curve.json', [dict(threshold=float(t), macro_f05=float(v), selection_score=float(s))
                                for t,v,s in zip(thresholds, official, objective)])
    complete(out, key, [out/'threshold.json', out/'curve.json'])
    return result


def evaluate(run_dir, part):
    root = Path(run_dir)
    verify(root/'decision')
    decision = read_json(root/'decision'/'threshold.json')
    meta = verify(root/'scores'/part)
    if decision['model_hash'] != meta['model_hash']:
        raise ValueError('Threshold belongs to a different model')
    def rows():
        for row in scored_rows(root, part):
            prediction = [c['id'] for c,s in zip(row['candidates'],row['scores']) if s >= decision['threshold']]
            yield row['truth'], prediction, row['country']
    report = summarize(rows())
    report.update(threshold=decision['threshold'], model_hash=meta['model_hash'])
    write_json(root/'reports'/f'matching_{part}.json', report)
    return report
