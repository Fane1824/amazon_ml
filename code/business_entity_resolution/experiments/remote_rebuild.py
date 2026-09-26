"""Rebuild the full model after remote_pilot, then validate complete test outputs.

Reuses only verified preparation, indexes and the identical development sample.
The pilot's small training sample, model and scores are never promoted.
"""
import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from common import code_hash, environment, read_json, verify, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--pilot-run', type=Path, required=True)
    parser.add_argument('--test-corpus', type=Path, required=True)
    parser.add_argument('--validator', type=Path, required=True)
    args = parser.parse_args()
    root, pilot = args.run_dir.resolve(), args.pilot_run.resolve()
    package = Path(__file__).resolve().parents[1]
    if read_json(pilot/'pilot_runtime.json')['status'] != 'PASS':
        raise ValueError('The full-corpus timing pilot must pass first')
    cfg = read_json(package/'configs/default.json')
    pilot_cfg = read_json(pilot/'pilot_config.json')
    cfg['workers'] = pilot_cfg['workers']
    cfg['model'].update(task_type='GPU', devices=pilot_cfg['model']['devices'],
                        thread_count=cfg['workers'])
    for setting in ('seed', 'index', 'retrieval', 'batch_size'):
        if cfg[setting] != pilot_cfg[setting]:
            raise ValueError(f'Incompatible pilot setting: {setting}')
    if cfg['samples']['dev'] != pilot_cfg['samples']['dev']:
        raise ValueError('Development samples differ')
    root.mkdir(parents=True, exist_ok=True)
    config_path = root/'training_config.json'
    if config_path.exists() and read_json(config_path) != cfg:
        raise ValueError('Run configuration changed; use a new run directory')
    write_json(config_path, cfg)
    report_path = root/'rebuild_runtime.json'
    report = read_json(report_path) if report_path.exists() else dict(
        started=time.strftime('%Y-%m-%d %H:%M:%S %Z'), code_hash=code_hash(), stages=[])
    if report['code_hash'] != code_hash():
        raise ValueError('Source changed; use a new run directory')

    def link(source, target):
        meta = verify(source)
        if meta['code_hash'] != report['code_hash']:
            raise ValueError(f'Corpus source version differs: {source}')
        if meta['environment'] != environment():
            raise ValueError(f'Corpus dependency versions differ: {source}')
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() or target.is_symlink():
            if target.resolve() != source.resolve():
                raise ValueError(f'Refusing to replace existing artifact: {target}')
        else:
            target.symlink_to(os.path.relpath(source, target.parent), target_is_directory=True)

    for relative in ('prepared/train', 'indexes/train', 'candidates/dev', 'features/dev'):
        link(pilot/relative, root/relative)
    env = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
               MKL_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1')

    def run(name, command):
        if code_hash() != report['code_hash']:
            raise ValueError('Source changed during the run')
        if read_json(config_path) != cfg:
            raise ValueError('Configuration changed during the run')
        if shutil.disk_usage(root).free < 20*1024**3:
            raise RuntimeError('Less than 20 GiB free; stopped before the next stage')
        report.update(status='RUNNING', active_stage=name)
        write_json(report_path, report)
        start = time.monotonic()
        print(f'{time.strftime("%H:%M:%S")} Starting {name}', flush=True)
        with (root/f'{name}.log').open('a') as log:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, env=env)
        report['stages'].append(dict(name=name, seconds=time.monotonic()-start,
                                     returncode=result.returncode, command=command))
        write_json(report_path, report)
        if result.returncode:
            raise RuntimeError(f'{name} failed with exit {result.returncode}')

    def stage(name, part=None):
        command = [sys.executable, '-u', str(package/'src/main.py'), name,
                   '--data-dir', str(args.data_dir.resolve()), '--run-dir', str(root),
                   '--config', str(config_path)]
        if part:
            command.extend(['--part', part])
        run(name+('_'+part if part else ''), command)

    try:
        stage('retrieve', 'train')
        stage('features', 'train')
        stage('train')
        stage('retrieve', 'threshold')
        stage('features', 'threshold')
        stage('score', 'threshold')
        stage('calibrate')
        stage('evaluate', 'threshold')
        for name in ('retrieve', 'features', 'score', 'evaluate'):
            stage(name, 'eval')
        # Preparation was launched independently. Refuse incomplete input instead
        # of racing another writer or presenting a partial test set as complete.
        link(args.test_corpus.resolve()/'prepared/test', root/'prepared/test')
        run('test_pipeline', [sys.executable, '-u', str(package/'experiments/run_submission.py'),
                             '--run-dir', str(root), '--data-dir', str(args.data_dir.resolve()),
                             '--validator', str(args.validator.resolve())])
        report.update(status='PASS', active_stage=None,
                      finished=time.strftime('%Y-%m-%d %H:%M:%S %Z'))
    except BaseException as error:
        report.update(status='FAILED', error=repr(error))
        raise
    finally:
        write_json(report_path, report)


if __name__ == '__main__':
    main()
