"""Bounded development-only comparison; reuses immutable prepared data/indexes."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import resource
import sys
import time

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from blocking import Retriever,reduce_candidates
from common import cached,complete,file_hash,fingerprint,log,read_json,verify,write_json
from data import connect,get_records
from metrics import f05
from pipeline import bounded_map,read_rows,write_rows
from selection import evidence,select_similarity

STATE = None


def initialize(root,cfg):
    global STATE
    STATE = (connect(Path(root)/'prepared/train/records.sqlite'),Retriever(root,'train',cfg))


def job(task):
    source,out,key = task
    out = Path(out)
    previous = cached(out,key)
    if previous:
        return previous
    con,retriever = STATE
    start = time.monotonic()
    rows = list(read_rows(Path(source)/'queries.jsonl'))
    queries = get_records(con,[r['rid'] for r in rows])
    def results():
        for row in rows:
            q = queries[row['rid']]
            raw = retriever.raw_hits(q)
            pools = {s:reduce_candidates(h,sum(map(len,h.values()))) for s,h in raw.items()}
            records = get_records(con,[c['rid'] for pool in pools.values() for c in pool])
            similarities = {c['rid']:evidence(q,records[c['rid']],retriever) for pool in pools.values() for c in pool}
            choices = {'saved_baseline':[c['id'] for c in row['candidates']],
                       'raw_union':[c['id'] for pool in pools.values() for c in pool]}
            for limit in (12,20,32):
                choices[f'fusion_{limit}'] = [c['id'] for hits in raw.values() for c in reduce_candidates(hits,limit)]
                choices[f'similarity_{limit}'] = [c['id'] for pool in pools.values() for c in select_similarity(pool,similarities,limit)]
            yield dict(id=row['id'],country=row['country'],truth=row['truth'],choices=choices,
                       raw=raw,similarities=similarities)
    write_rows(out/'comparison.jsonl',results())
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/(1024**2 if sys.platform=='darwin' else 1024)
    return complete(out,key,[out/'comparison.jsonl'],seconds=time.monotonic()-start,references=len(rows),worker_peak_rss_mib=rss)


def aggregate(out,shards,seconds):
    stats = defaultdict(lambda:defaultdict(lambda:dict(refs=0,truth=0,recovered=0,pairs=0,oracle_sum=0.)))
    paired = defaultdict(list)
    baseline_changed = 0
    for shard in shards:
        verify(out/shard)
        for row in read_rows(out/shard/'comparison.jsonl'):
            truth = set(row['truth'])
            baseline_changed += set(row['choices']['saved_baseline']) != set(row['choices']['fusion_12'])
            for policy,ids in row['choices'].items():
                found = truth&set(ids)
                score = f05(truth,found)
                paired[policy].append(score)
                for country in ('ALL',row['country']):
                    s = stats[policy][country]
                    s['refs'] += 1;s['truth'] += len(truth);s['recovered'] += len(found)
                    s['pairs'] += len(ids);s['oracle_sum'] += score
    report = dict(seconds=seconds,baseline_candidate_sets_changed=baseline_changed,policies={})
    base = np.asarray(paired['saved_baseline'])
    for policy,countries in stats.items():
        result = {}
        for country,s in countries.items():
            result[country] = dict(references=s['refs'],positive_pair_recall=s['recovered']/s['truth'],
                                  oracle_macro_f05=s['oracle_sum']/s['refs'],mean_candidates=s['pairs']/s['refs'])
        delta = np.asarray(paired[policy])-base
        rng = np.random.default_rng(2026)
        boot = [float(delta[rng.integers(0,len(delta),len(delta))].mean()) for _ in range(500)]
        result['paired_oracle_delta'] = dict(mean=float(delta.mean()),ci95=np.percentile(boot,[2.5,97.5]).tolist())
        report['policies'][policy] = result
    write_json(out/'report.json',report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',type=Path,required=True)
    parser.add_argument('--out-dir',type=Path,required=True)
    parser.add_argument('--raw-hits',type=int,default=32)
    parser.add_argument('--workers',type=int,default=4)
    args = parser.parse_args()
    cfg = read_json(Path(__file__).resolve().parents[1]/'configs/default.json')
    cfg['retrieval']['raw_hits'] = args.raw_hits
    cfg['retrieval']['selection'] = 'fusion'
    candidate_meta = verify(args.run_dir/'candidates/dev')
    prepared = verify(args.run_dir/'prepared/train')
    indexes = verify(args.run_dir/'indexes/train')
    for country in indexes['countries']:
        from blocking import country_key
        verify(args.run_dir/'indexes/train'/country_key(country))
    key = fingerprint(file_hash(__file__),candidate_meta['fingerprint'],prepared['fingerprint'],indexes['fingerprint'],cfg['retrieval'])
    start = time.monotonic()
    tasks = []
    for shard in candidate_meta['shards']:
        meta = verify(args.run_dir/'candidates/dev'/shard)
        tasks.append((str(args.run_dir/'candidates/dev'/shard),str(args.out_dir/shard),fingerprint(key,meta['fingerprint'])))
    for i,result in enumerate(bounded_map(job,tasks,args.workers,initialize,(str(args.run_dir),cfg))):
        log(f'comparison shard {i}: {result["seconds"]:.1f}s, worker peak RSS {result["worker_peak_rss_mib"]:.0f} MiB')
    elapsed = time.monotonic()-start
    report = aggregate(args.out_dir,candidate_meta['shards'],elapsed)
    complete(args.out_dir,key,[args.out_dir/s/'manifest.json' for s in candidate_meta['shards']]+[args.out_dir/'report.json'],
             seconds=elapsed,raw_hits=args.raw_hits,shards=candidate_meta['shards'])
    for policy,result in report['policies'].items():
        log(f'{policy}: {result["ALL"]}')


if __name__=='__main__':
    main()
