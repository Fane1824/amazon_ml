"""Explicit reuse of a verified corpus when only downstream retrieval code changed.

Requires the original source snapshot; do not infer compatibility from file names.
This creates read-through links to immutable train artifacts, not new fingerprints.
Use the staged CLI on this run; `run` would attempt to rebuild its train corpus.
"""
import argparse
import ast
import os
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from common import ROOT,digest,environment,file_hash,verify,write_json


def functions(path,names):
    module=ast.parse(path.read_text())
    return {n.name:ast.dump(n,include_attributes=False) for n in module.body
            if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in names}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-run',type=Path,required=True)
    parser.add_argument('--source-snapshot',type=Path,required=True,help='Directory containing original src/')
    parser.add_argument('--run-dir',type=Path,required=True)
    args=parser.parse_args()
    source=args.source_run.resolve();out=args.run_dir.resolve()
    snapshot_hash=digest({p.name:file_hash(p) for p in sorted((args.source_snapshot/'src').glob('*.py'))})
    for name in ('data.py','text.py','common.py'):
        if file_hash(ROOT/'src'/name)!=file_hash(args.source_snapshot/'src'/name):
            raise ValueError(f'Corpus compatibility cannot be established: {name} changed')
    names={'build_index','country_key','register'}
    if functions(ROOT/'src/blocking.py',names)!=functions(args.source_snapshot/'src/blocking.py',names):
        raise ValueError('Index construction changed; rebuild indexes')
    provenance={}
    for stage in ('prepared','indexes'):
        meta=verify(source/stage/'train')
        if meta['code_hash']!=snapshot_hash:
            raise ValueError('Source snapshot does not match the recorded corpus version')
        if meta['environment']!=environment():
            raise ValueError('Dependency versions changed')
        if stage=='indexes':
            from blocking import country_key
            for country in meta['countries']:verify(source/stage/'train'/country_key(country))
        target=out/stage/'train'
        target.parent.mkdir(parents=True,exist_ok=True)
        expected=source/stage/'train'
        if target.exists():
            if target.resolve()!=expected:raise ValueError(f'Existing different artifact: {target}')
        else:
            target.symlink_to(os.path.relpath(expected,target.parent),target_is_directory=True)
        provenance[stage]=dict(path=str(expected),fingerprint=meta['fingerprint'],original_code_hash=meta['code_hash'])
    write_json(out/'reused_corpus.json',dict(inputs=provenance,compatibility='Identical preparation/normalization files and index-construction AST',
               commands='Use retrieve/features/train/score/calibrate stages. Prepare/index test separately. Do not use run on this linked train corpus.'))


if __name__=='__main__':main()
