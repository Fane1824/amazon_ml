"""Resume full-test stages using an already frozen model and threshold."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time


def sha256(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--validator', type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    run, data = args.run_dir.resolve(), args.data_dir.resolve()
    if (run/'SMOKE_ONLY.json').exists() or (data/'SMOKE_ONLY.json').exists():
        raise ValueError('Full submission cannot use a smoke fixture')
    frozen = {str(p): sha256(p) for p in [run/'model/model.cbm', run/'decision/threshold.json',
                                        run/'training_config.json', *sorted((root/'src').glob('*.py'))]}
    base = [sys.executable, '-u', str(root/'src/main.py')]
    common = ['--data-dir', str(data), '--run-dir', str(run), '--config', str(run/'training_config.json')]
    stages = [(stage, base+[stage]+common+extra) for stage, extra in [
        ('prepare', ['--split', 'test']), ('index', ['--split', 'test']),
        ('retrieve', ['--part', 'test']), ('features', ['--part', 'test']),
        ('score', ['--part', 'test']), ('export', []), ('validate', [])]]
    stages.append(('official_validator', [sys.executable, '-u', str(args.validator.resolve()),
        '--matching', str(run/'output/matching_results.tsv'),
        '--candidate', str(run/'output/candidate_pairs.tsv'), '--test-dir', str(data/'test'), '--check-ids']))
    started = time.monotonic()
    report = dict(started_at=datetime.now(timezone.utc).isoformat(), frozen_sha256=frozen,
                  status='RUNNING', stages=[])
    path = run/'reports/test_pipeline_runtime.json'
    def save():
        report['seconds'] = time.monotonic()-started
        tmp = path.with_suffix('.tmp')
        tmp.write_text(json.dumps(report, indent=2)+'\n')
        tmp.replace(path)
    save()
    try:
        for stage, cmd in stages:
            if any(sha256(Path(p)) != h for p,h in frozen.items()):
                raise ValueError('Frozen model, threshold, config, or source changed')
            free = shutil.disk_usage(run).free
            if free < 5*1024**3:
                raise RuntimeError('Fewer than 5 GiB free; preserve completed artifacts and free space before resuming')
            report['active_stage'] = stage
            save()
            print(f'PIPELINE START {stage}; free disk {free/1024**3:.1f} GiB', flush=True)
            t = time.monotonic()
            log = run/f'test_{stage}.log'
            with log.open('w') as f:
                subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, check=True)
            report['stages'].append(dict(stage=stage, seconds=time.monotonic()-t, log=log.name,
                                         free_disk_gib=shutil.disk_usage(run).free/1024**3))
            save()
            print(f'PIPELINE DONE {stage}: {time.monotonic()-t:.1f}s', flush=True)
        report.update(status='PASS', active_stage=None)
        save()
        print('PIPELINE COMPLETE: internal and official validators passed', flush=True)
    except BaseException as exc:
        report.update(status='FAILED', error=repr(exc))
        save()
        raise


if __name__ == '__main__':
    main()
