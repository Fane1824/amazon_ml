"""Evaluate one fixed adaptive reduction rule on saved, development-only raw hits."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from blocking import reduce_candidates
from common import read_json,verify,write_json,file_hash
from metrics import f05
from pipeline import read_rows
from selection import select_adaptive


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--comparison-dir',type=Path,required=True)
    args=parser.parse_args()
    meta=verify(args.comparison_dir)
    stats=defaultdict(lambda:dict(refs=0,truth=0,recovered=0,oracle=0.,pairs=0))
    delta=[];counts=[];regressed=0
    for shard in meta['shards']:
        verify(args.comparison_dir/shard)
        for row in read_rows(args.comparison_dir/shard/'comparison.jsonl'):
            scores={int(k):v for k,v in row['similarities'].items()}
            ids=set()
            for hits in row['raw'].values():
                pool=reduce_candidates(hits,sum(map(len,hits.values())))
                base=reduce_candidates(hits,12)
                ids.update(c['id'] for c in select_adaptive(pool,scores,32,base))
            truth=set(row['truth']);found=truth&ids;oracle=f05(truth,found)
            base_found=truth&set(row['choices']['saved_baseline'])
            delta.append(oracle-f05(truth,base_found));counts.append(len(ids))
            regressed+=not base_found<=found
            for country in ('ALL',row['country']):
                s=stats[country];s['refs']+=1;s['truth']+=len(truth);s['recovered']+=len(found)
                s['oracle']+=oracle;s['pairs']+=len(ids)
    delta=np.asarray(delta);rng=np.random.default_rng(2026)
    boot=[float(delta[rng.integers(0,len(delta),len(delta))].mean()) for _ in range(500)]
    result=dict(raw_hits=meta['raw_hits'],policy='adaptive_v1',per_source_limit=32,
                upstream_fingerprint=meta['fingerprint'],selection_source_sha256=file_hash(Path(__file__).resolve().parents[1]/'src/selection.py'),
                countries={c:dict(references=s['refs'],recall=s['recovered']/s['truth'],oracle_macro_f05=s['oracle']/s['refs'],mean_candidates=s['pairs']/s['refs']) for c,s in stats.items()},
                paired_oracle_delta=dict(mean=float(delta.mean()),ci95=np.percentile(boot,[2.5,97.5]).tolist()),
                reference_recall_regressions=regressed,candidate_percentiles=np.percentile(counts,[50,95,99,100]).tolist())
    write_json(args.comparison_dir/'adaptive_report.json',result)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
