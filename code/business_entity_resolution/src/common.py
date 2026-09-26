"""Shared configuration, atomic artifacts, and provenance utilities."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[1]


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')
    os.replace(tmp, path)


def log(message):
    print(time.strftime('%H:%M:%S'), message, flush=True)


def environment():
    return {p: importlib.metadata.version(p) for p in ('numpy', 'catboost', 'tantivy', 'rapidfuzz', 'anyascii')}


def code_hash():
    return digest({p.name: file_hash(p) for p in sorted((ROOT / 'src').glob('*.py'))})


def fingerprint(*inputs):
    return digest([code_hash(), environment(), *inputs])


def complete(directory, key, files, **details):
    directory = Path(directory)
    result = dict(fingerprint=key, files={str(Path(p).relative_to(directory)): file_hash(p) for p in files},
                  environment=environment(), code_hash=code_hash(), **details)
    write_json(directory / 'manifest.json', result)
    return result


def cached(directory, key):
    path = Path(directory) / 'manifest.json'
    if not path.exists():
        return None
    result = read_json(path)
    if result['fingerprint'] != key:
        raise ValueError(f'{directory}: incompatible existing artifact; use a new run directory/configuration')
    verify(directory)
    return result


def verify(directory):
    directory = Path(directory)
    result = read_json(directory / 'manifest.json')
    for name, expected in result['files'].items():
        path = directory / name
        if not path.exists() or file_hash(path) != expected:
            raise ValueError(f'Incomplete or modified artifact: {path}')
    return result


def chunks(iterable, size):
    batch = []
    for item in iterable:
        batch.append(item)
        if len(batch) == size:
            yield batch
            batch = []
    if batch:
        yield batch
