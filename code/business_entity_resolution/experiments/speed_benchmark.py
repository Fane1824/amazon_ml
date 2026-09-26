"""Compare worker scaling and shared top-k search on identical development queries."""
import argparse
import json
from pathlib import Path
import resource
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import tantivy
from blocking import CHANNELS,Retriever
from common import chunks,read_json,write_json,log
from data import connect,get_records
from metrics import f05
from pipeline import bounded_map,read_rows
from text import fields


class SharedRetriever(Retriever):
    """Global BM25 top-k, partitioned afterwards; exact filtered fallback.

    Omitting the constant source term preserves mathematical within-source
    rankings. Float rounding and tied top-k boundaries still require validation.
    """
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.fallbacks=0

    def raw_hits(self,query):
        self.country(query['country'])
        terms=fields(query['v'])
        selected={f:sorted((t for t in ts if self.df(f,t)),key=lambda t:(self.df(f,t),t))[:self.cfg['tokens_per_field']] for f,ts in terms.items()}
        result={2:{},3:{}}
        cap=self.cfg['raw_hits']
        for channel,boosts in CHANNELS.items():
            clauses=[]
            for field,boost in boosts.items():
                for term in selected[field]:
                    q=tantivy.Query.term_query(self.index.schema,field,term,index_option='freq')
                    clauses.append((tantivy.Occur.Should,tantivy.Query.boost_query(q,boost)))
            if not clauses:
                for s in (2,3):result[s][channel]=[]
                continue
            words=tantivy.Query.boolean_query(clauses,minimum_number_should_match=1)
            found={2:[],3:[]}
            for score,address in self.searcher.search(words,limit=4*cap,count=False).hits:
                doc=self.searcher.doc(address);eid=doc['id'][0];source=int(eid[1])
                if len(found[source])<cap:found[source].append((score,doc['rid'][0],eid))
            for source in (2,3):
                if len(found[source])<cap:
                    self.fallbacks+=1
                    src=tantivy.Query.term_query(self.index.schema,'source',source)
                    q=tantivy.Query.boolean_query([(tantivy.Occur.Must,words),(tantivy.Occur.Must,src)])
                    found[source]=[]
                    for score,address in self.searcher.search(q,limit=cap,count=False).hits:
                        doc=self.searcher.doc(address)
                        found[source].append((score,doc['rid'][0],doc['id'][0]))
                found[source].sort(key=lambda x:(-x[0],x[2]))
                result[source][channel]=[(rid,eid) for _,rid,eid in found[source]]
        return result


WORKER=None


def initialize(root,cfg,mode):
    global WORKER
    cfg['retrieval']['search_mode']='filtered'
    WORKER=(SharedRetriever if mode=='shared' else Retriever)(root,'train',cfg)


def job(batch):
    before=getattr(WORKER,'fallbacks',0)
    result=[(q['id'],[c['id'] for c in WORKER.retrieve(q)]) for q in batch]
    rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/(1024**2 if sys.platform=='darwin' else 1024)
    return result,getattr(WORKER,'fallbacks',0)-before,rss


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir',type=Path,required=True)
    p.add_argument('--out-dir',type=Path,required=True)
    p.add_argument('--references',type=int,default=2000)
    p.add_argument('--workers',type=int,nargs='+',default=[4,8,12])
    p.add_argument('--modes',nargs='+',default=['filtered','shared'])
    args=p.parse_args()
    rows=[r for path in sorted((args.run_dir/'candidates/dev').glob('*/queries.jsonl')) for r in read_rows(path)]
    if len(rows)!=10000:raise ValueError('Expected the complete 10,000-reference development pilot')
    step=max(1,len(rows)//args.references);rows=rows[::step][:args.references]
    expected={r['id']:r for r in rows}
    con=connect(args.run_dir/'prepared/train/records.sqlite')
    records=get_records(con,[r['rid'] for r in rows]);con.close()
    batches=list(chunks((records[r['rid']] for r in rows),64))
    cfg=read_json(Path(__file__).resolve().parents[1]/'configs/default.json')
    reports=[]
    for mode in args.modes:
        for workers in args.workers:
            start=time.monotonic();found=[];fallbacks=0;peak=0
            for result,n,rss in bounded_map(job,batches,workers,initialize,(str(args.run_dir),cfg,mode)):
                found.extend(result);fallbacks+=n;peak=max(peak,rss)
            seconds=time.monotonic()-start
            countries={};changed=lost=gained=0
            for qid,ids in found:
                row=expected[qid];truth=set(row['truth']);base={c['id'] for c in row['candidates']};ids=set(ids)
                changed+=ids!=base;lost+=len((truth&base)-ids);gained+=len((truth&ids)-base)
                for country in ('ALL',row['country']):
                    s=countries.setdefault(country,dict(refs=0,truth=0,recovered=0,oracle=0.,pairs=0))
                    s['refs']+=1;s['truth']+=len(truth);s['recovered']+=len(truth&ids)
                    s['oracle']+=f05(truth,truth&ids);s['pairs']+=len(ids)
            report=dict(mode=mode,workers=workers,seconds=seconds,queries_per_second=len(rows)/seconds,
                        projected_test_hours=1732544*seconds/len(rows)/3600,worker_peak_rss_mib=peak,
                        changed_candidate_sets=changed,lost_positive_pairs=lost,gained_positive_pairs=gained,
                        fallback_searches=fallbacks,countries={c:dict(recall=s['recovered']/s['truth'],
                        oracle_macro_f05=s['oracle']/s['refs'],mean_candidates=s['pairs']/s['refs']) for c,s in countries.items()})
            reports.append(report);write_json(args.out_dir/'report.json',reports)
            log(json.dumps(report))


if __name__=='__main__':main()
