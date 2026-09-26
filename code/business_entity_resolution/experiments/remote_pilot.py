"""Bounded full-training-corpus timing pilot; never creates a submission.

Run with the project Python 3.12 environment. Uses production stage commands
without editing src/. The train/dev query samples are small, but S2/S3 are full.
Reports stage wall time, commands, exit codes, and the exact pilot configuration.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=36)
    parser.add_argument('--references', type=int, default=10000)
    parser.add_argument('--devices', default='0:1:2')
    args = parser.parse_args()
    if args.references < 1 or args.workers < 1:
        parser.error('--references and --workers must be positive; zero would request all queries')
    package = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(package/'src'))
    from common import code_hash
    root = args.run_dir.resolve()
    root.mkdir(parents=True, exist_ok=True)
    cfg = json.loads((package/'configs/default.json').read_text())
    cfg['workers'] = args.workers
    cfg['samples']['train'] = args.references
    cfg['samples']['dev'] = args.references
    cfg['model'].update(task_type='GPU', devices=args.devices,
                        thread_count=args.workers)
    config = root/'pilot_config.json'
    if config.exists() and json.loads(config.read_text()) != cfg:
        raise ValueError('Existing pilot configuration differs; use a new run directory')
    config.write_text(json.dumps(cfg, indent=2)+'\n')
    report_path = root/'pilot_runtime.json'
    report = json.loads(report_path.read_text()) if report_path.exists() else dict(
        started=time.strftime('%Y-%m-%d %H:%M:%S %Z'),
        purpose='Timing only: full training targets, sampled queries, up to 1500 GPU iterations',
        data_dir=str(args.data_dir.resolve()), config=cfg, code_hash=code_hash(), stages=[])

    def save():
        temporary = report_path.with_suffix('.tmp')
        temporary.write_text(json.dumps(report, indent=2)+'\n')
        os.replace(temporary, report_path)

    env = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
               MKL_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1')
    stages = [('prepare', ['--split', 'train']), ('index', ['--split', 'train']),
              ('retrieve_dev', ['--part', 'dev']), ('features_dev', ['--part', 'dev']),
              ('retrieve_train', ['--part', 'train']), ('features_train', ['--part', 'train']),
              ('train', []), ('score_dev', ['--part', 'dev'])]
    for name, extra in stages:
        if report['code_hash'] != code_hash():
            report.update(status='FAILED', error='Source changed during the pilot')
            save()
            raise ValueError(report['error'])
        if any(s['name'] == name and s['returncode'] == 0 for s in report['stages']):
            continue
        if __import__('shutil').disk_usage(root).free < 20*1024**3:
            report.update(status='FAILED', error='Less than 20 GiB scratch free')
            save()
            raise RuntimeError('Less than 20 GiB scratch free; stopping pilot')
        stage = name.split('_')[0]
        command = [sys.executable, '-u', str(package/'src/main.py'), stage,
                   '--data-dir', str(args.data_dir.resolve()), '--run-dir', str(root),
                   '--config', str(config), *extra]
        report.update(status='RUNNING', active_stage=name)
        save()
        print(f'{time.strftime("%H:%M:%S")} Starting {name}', flush=True)
        started = time.monotonic()
        with (root/f'{name}.log').open('a') as log:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, env=env)
        report['stages'].append(dict(name=name, seconds=time.monotonic()-started,
                                     returncode=result.returncode, command=command))
        report['status'] = 'RUNNING' if result.returncode == 0 else 'FAILED'
        save()
        if result.returncode:
            raise SystemExit(f'{name} failed; see {root/name}.log')
    # This projects measured full-corpus query costs, not reduced smoke timings.
    # Training-size and unseen-country scaling still need an explicit allowance.
    durations = {s['name']: s['seconds'] for s in report['stages'] if s['returncode'] == 0}
    def meta(relative):
        return json.loads((root/relative/'manifest.json').read_text())
    retrieval = [meta(f'candidates/{p}') for p in ('train', 'dev')]
    feature = [meta(f'features/{p}') for p in ('train', 'dev')]
    retrieve_per_ref = sum(m['seconds'] for m in retrieval)/sum(m['references'] for m in retrieval)
    feature_per_ref = sum(m['seconds'] for m in feature)/sum(m['references'] for m in feature)
    trees = meta('model')['best_iteration']+1
    train_refs, test_refs = 150000, 1732544
    components = dict(
        preparation=durations['prepare']*(1+11702133/12527040),
        indexing=durations['index']*(1+9969589/10320219),
        retrieval=retrieve_per_ref*(train_refs+test_refs),
        features=feature_per_ref*(train_refs+test_refs),
        training=durations['train']*(100000/args.references)*(1500/trees),
        scoring=durations['score_dev']/args.references*(test_refs+40000)*(1500/trees),
        calibration_export_validation_allowance=3600)
    estimate = dict(component_seconds=components,
                    fresh_rebuild_hours=sum(components.values())/3600,
                    with_50_percent_contingency_hours=1.5*sum(components.values())/3600,
                    limitations=[
                        'Query costs measured on full US/India training corpus; French test cost may differ.',
                        'Training extrapolates from sampled real pairs; larger GPU pools may scale nonlinearly.',
                        'Scoring scales all measured time by the tree-count ratio, conservatively including I/O.',
                        'Calibration/export/validation use an unmeasured one-hour allowance.',
                        'Estimate is for a fresh run and does not subtract already completed parallel work.'])
    (root/'runtime_estimate.json').write_text(json.dumps(estimate, indent=2)+'\n')
    report.update(status='PASS', active_stage=None,
                  finished=time.strftime('%Y-%m-%d %H:%M:%S %Z'))
    save()
    print('Full-corpus pilot complete. This is not a submission model.', flush=True)


if __name__ == '__main__':
    main()
