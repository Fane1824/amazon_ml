#!/usr/bin/env python3
"""Run from any working directory: python /path/to/src/main.py --help."""
import argparse
import json
from pathlib import Path
import platform
import resource
import time

from common import ROOT, environment, log, read_json, write_json


def config(path, device=None, workers=None):
    cfg = read_json(path)
    if device:
        cfg['model']['task_type'] = device
    if workers is not None:
        cfg['workers'] = workers
    if cfg['workers'] < 1 or cfg['batch_size'] < 1:
        raise ValueError('workers and batch_size must be positive')
    r = cfg['retrieval']
    if r.get('search_mode','filtered') not in ('filtered','shared'):
        raise ValueError('Unknown retrieval search mode')
    if r.get('selection','fusion') not in ('fusion','similarity_v1','adaptive_v1'):
        raise ValueError('Unknown retrieval selection mode')
    if not 1 <= r['per_source_limit'] <= 3*r['raw_hits'] or r['tokens_per_field'] < 1:
        raise ValueError('Invalid retrieval limits')
    if any(n < 0 for n in cfg['samples'].values()):
        raise ValueError('Sample sizes must be non-negative; zero means all')
    weights = cfg['threshold']['country_weights']
    if weights and (min(weights.values()) < 0 or sum(weights.values()) <= 0):
        raise ValueError('Invalid country weights')
    return cfg


def run_all(data_dir, run_dir, cfg):
    from data import prepare
    from blocking import build_index
    from pipeline import retrieve, features, blocking_report
    from learn import train, score, calibrate, evaluate
    from export import export, validate
    if (Path(data_dir)/'SMOKE_ONLY.json').exists():
        write_json(Path(run_dir)/'SMOKE_ONLY.json', {'warning':'Reduced smoke dataset: do not upload these outputs.'})
    prepare(data_dir, run_dir, 'train', cfg)
    build_index(run_dir, 'train', cfg)
    for part in ('train','dev','threshold'):
        retrieve(run_dir, part, cfg)
        blocking_report(run_dir, part)
        features(run_dir, part, cfg)
    train(run_dir, cfg)
    score(run_dir, 'threshold', cfg)
    log(f'calibration: {calibrate(run_dir,cfg)}')
    evaluate(run_dir, 'threshold')
    # Locked evaluation is an explicit separate command, not repeatedly consumed by run.
    prepare(data_dir, run_dir, 'test', cfg)
    build_index(run_dir, 'test', cfg)
    retrieve(run_dir, 'test', cfg)
    blocking_report(run_dir, 'test')
    features(run_dir, 'test', cfg)
    score(run_dir, 'test', cfg)
    export(run_dir)
    return validate(run_dir)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['smoke','smoke-data','run','prepare','index','retrieve','features','train','score','calibrate','evaluate','predict','export','validate','blocking-report'])
    parser.add_argument('--data-dir', type=Path, required=True, help='Directory containing train/ and test/')
    parser.add_argument('--run-dir', type=Path, required=True, help='Immutable configuration/run artifacts')
    parser.add_argument('--config', type=Path)
    parser.add_argument('--split', choices=['train','test'], default='train')
    parser.add_argument('--part', choices=['train','dev','threshold','eval','test'], default='dev')
    parser.add_argument('--device', choices=['CPU','GPU'])
    parser.add_argument('--workers', type=int)
    args = parser.parse_args()
    cfg = config(args.config or ROOT/'configs'/('mac_smoke.json' if args.stage in ('smoke','smoke-data') else 'default.json'), args.device, args.workers)
    root = args.run_dir.resolve()
    root.mkdir(parents=True, exist_ok=True)
    write_json(root/'invocations'/f'{time.time_ns()}_{args.stage}.json', dict(stage=args.stage, config=cfg, environment=environment(), python=platform.python_version(), data_dir=str(args.data_dir.resolve())))
    start = time.monotonic()
    from data import prepare
    from blocking import build_index
    from pipeline import retrieve, features, blocking_report
    from learn import train, score, calibrate, evaluate
    from export import export, validate
    result = None
    if args.stage in ('smoke','smoke-data'):
        from smoke import make_smoke
        subset = root/'smoke_dataset'
        make_smoke(args.data_dir, subset)
        if args.stage == 'smoke':
            log('SMOKE ONLY: reduced target corpus; metrics are not competition estimates')
            result = run_all(subset, root, cfg)
            for action in (retrieve, features, score):
                action(root, 'eval', cfg)
            evaluate(root, 'eval')
            write_json(root/'SMOKE_ONLY.json', {'warning':'Do not upload these outputs. They cover only the smoke dataset.'})
    elif args.stage == 'run':
        result = run_all(args.data_dir, root, cfg)
    elif args.stage == 'prepare':
        result = prepare(args.data_dir, root, args.split, cfg)
    elif args.stage == 'index':
        result = build_index(root, args.split, cfg)
    elif args.stage == 'retrieve':
        retrieve(root,args.part,cfg)
        result = blocking_report(root,args.part)
    elif args.stage == 'features':
        result = features(root,args.part,cfg)
    elif args.stage == 'train':
        result = train(root,cfg)
    elif args.stage == 'score':
        result = score(root,args.part,cfg)
    elif args.stage == 'calibrate':
        result = calibrate(root,cfg)
    elif args.stage == 'evaluate':
        if args.part in ('test','train','dev'):
            parser.error('evaluate expects --part threshold or eval')
        result = evaluate(root,args.part)
    elif args.stage == 'predict':
        retrieve(root,'test',cfg)
        features(root,'test',cfg)
        result = score(root,'test',cfg)
    elif args.stage == 'export':
        export(root)
    elif args.stage == 'validate':
        result = validate(root)
    elif args.stage == 'blocking-report':
        result = blocking_report(root,args.part)
    elapsed = time.monotonic()-start
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    rss_mb = rss/(1024**2 if platform.system() == 'Darwin' else 1024)
    write_json(root/'reports'/f'runtime_{args.stage}.json', dict(seconds=elapsed, parent_peak_rss_mb=rss_mb,
              note='Parent process only; cluster worker RSS must be measured separately.'))
    log(f'{args.stage} complete in {elapsed:.1f}s; parent peak RSS {rss_mb:.1f} MiB')
    if result:
        print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
