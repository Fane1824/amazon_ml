"""Disk-backed, country-partitioned lexical search and deterministic rank fusion."""
from __future__ import annotations

from functools import lru_cache
import json
from pathlib import Path
import shutil
import time

import tantivy

from common import cached, complete, digest, fingerprint, log, read_json, verify
from data import connect, record, get_records
from text import fields

CHANNELS = {'name': {'name': 3., 'gram': 1.},
            'address': {'address': 3., 'number': 1.},
            'combined': {'name': 2., 'gram': 1., 'address': 2.}}


def country_key(country):
    return digest(country)[:20]


def register(index):
    index.register_tokenizer('split_space', tantivy.TextAnalyzerBuilder(tantivy.Tokenizer.whitespace()).build())


def build_index(run_dir, split, cfg):
    base = Path(run_dir)
    prepared = verify(base/'prepared'/split)
    con = connect(base/'prepared'/split/'records.sqlite')
    schema_builder = tantivy.SchemaBuilder()
    schema_builder.add_integer_field('rid', stored=True)
    schema_builder.add_integer_field('source', indexed=True)
    schema_builder.add_text_field('id', stored=True, tokenizer_name='raw')
    for name in ('name', 'gram', 'address', 'number'):
        schema_builder.add_text_field(name, tokenizer_name='split_space', index_option='freq')
    schema = schema_builder.build()
    countries = [r[0] for r in con.execute('SELECT DISTINCT country FROM records ORDER BY country')]
    for country in countries:
        out = base/'indexes'/split/country_key(country)
        key = fingerprint(prepared['fingerprint'], country, cfg['index'])
        if cached(out, key):
            continue
        # Only incomplete indexes are rebuilt; completed artifacts are immutable.
        if out.exists():
            shutil.rmtree(out)
        out.mkdir(parents=True)
        index = tantivy.Index(schema, path=str(out))
        register(index)
        writer = index.writer(heap_size=cfg['index']['heap_mb']*1_000_000, num_threads=cfg['index']['threads'])
        count, start = 0, time.monotonic()
        for row in con.execute('SELECT * FROM records WHERE source IN (2,3) AND country=? ORDER BY source,id', (country,)):
            r = record(row)
            doc = tantivy.Document(rid=r['rid'], source=r['source'], id=r['id'])
            for field, terms in fields(r['v']).items():
                doc.add_text(field, ' '.join(terms))
            writer.add_document(doc)
            count += 1
            if count % 100000 == 0:
                log(f'index {split}/{country}: {count:,}')
        writer.commit()
        writer.wait_merging_threads()
        # Lock files are runtime state and not reproducible index content.
        files = [p for p in out.iterdir() if p.is_file() and not p.name.startswith('.')]
        complete(out, key, files, country=country, records=count, seconds=time.monotonic()-start)
        log(f'indexed {split}/{country}: {count:,}')
    con.close()
    # Parent manifest binds every country index to this prepared corpus.
    parent = base/'indexes'/split
    files = [parent/country_key(c)/'manifest.json' for c in countries]
    complete(parent, fingerprint(prepared['fingerprint'], cfg['index']), files, countries=countries)


def reduce_candidates(hits, limit):
    ranks, ids = {}, {}
    for channel, channel_hits in hits.items():
        for rank, (rid, eid) in enumerate(channel_hits, 1):
            ranks.setdefault(rid, {})[channel] = rank
            ids[rid] = eid
    fusion = {rid: sum(1/(60+r) for r in rr.values()) for rid, rr in ranks.items()}
    order = sorted(ranks, key=lambda rid: (-fusion[rid], ids[rid]))
    selected = {rid for values in hits.values() for rid, _ in values[:limit//4]}
    for rid in order:
        if len(selected) >= limit:
            break
        selected.add(rid)
    return [dict(rid=rid, id=ids[rid], ranks=ranks[rid], fusion=fusion[rid], rank=i)
            for i, rid in enumerate(order, 1) if rid in selected]


class Retriever:
    def __init__(self, run_dir, split, cfg):
        self.root = Path(run_dir)/'indexes'/split
        self.cfg = cfg['retrieval']
        self.active = None
        self.records = None
        if self.cfg.get('selection','fusion') in ('similarity_v1','adaptive_v1'):
            self.records = connect(Path(run_dir)/'prepared'/split/'records.sqlite')

    def country(self, country):
        if self.active != country:
            path = self.root/country_key(country)
            if not (path/'manifest.json').exists():
                raise ValueError(f'Missing index for country {country!r}; run index first')
            self.index = tantivy.Index.open(str(path))
            register(self.index)
            self.searcher = self.index.searcher()
            self.active = country
            self.df.cache_clear()
        return self

    @lru_cache(maxsize=100000)
    def df(self, field, term):
        return self.searcher.doc_freq(field, term)

    def raw_hits(self, query):
        self.country(query['country'])
        terms = fields(query['v'])
        selected = {f: sorted((t for t in ts if self.df(f, t)), key=lambda t: (self.df(f, t), t))[:self.cfg['tokens_per_field']]
                    for f, ts in terms.items()}
        mode = self.cfg.get('search_mode', 'filtered')
        if mode not in ('filtered', 'shared'):
            raise ValueError(f'Unknown search mode: {mode}')
        result = {2: {}, 3: {}}
        cap = self.cfg['raw_hits']
        for channel, boosts in CHANNELS.items():
            clauses = []
            for field, boost in boosts.items():
                for term in selected[field]:
                    q = tantivy.Query.term_query(self.index.schema, field, term, index_option='freq')
                    clauses.append((tantivy.Occur.Should, tantivy.Query.boost_query(q, boost)))
            if not clauses:
                for source in (2, 3):
                    result[source][channel] = []
                continue
            words = tantivy.Query.boolean_query(clauses, minimum_number_should_match=1)
            found = {2: [], 3: []}
            if mode == 'shared':
                # A pure term union permits fast top-k pruning. The omitted source
                # score is constant within each source; floating-point ties may
                # still differ from filtered search. Keep native order until cap.
                for score, address in self.searcher.search(words, limit=4*cap, count=False).hits:
                    doc = self.searcher.doc(address)
                    eid = doc['id'][0]
                    source = int(eid[1])  # IDs are validated S2-/S3- at preparation.
                    if len(found[source]) < cap:
                        found[source].append((score, doc['rid'][0], eid))
            for source in (2, 3):
                # A dominant source must never starve the other source's quota.
                if mode == 'filtered' or len(found[source]) < cap:
                    src = tantivy.Query.term_query(self.index.schema, 'source', source)
                    q = tantivy.Query.boolean_query([(tantivy.Occur.Must, words), (tantivy.Occur.Must, src)])
                    found[source] = []
                    for score, address in self.searcher.search(q, limit=cap, count=False).hits:
                        doc = self.searcher.doc(address)
                        found[source].append((score, doc['rid'][0], doc['id'][0]))
                found[source].sort(key=lambda x: (-x[0], x[2]))
                result[source][channel] = [(rid, eid) for _, rid, eid in found[source]]
        return result

    def retrieve(self, query):
        result = []
        raw = self.raw_hits(query)
        for source,hits in raw.items():
            limit = self.cfg['per_source_limit']
            mode = self.cfg.get('selection','fusion')
            if mode == 'fusion':
                candidates = reduce_candidates(hits,limit)
            elif mode in ('similarity_v1','adaptive_v1'):
                from selection import evidence,select_similarity,select_adaptive
                pool = reduce_candidates(hits,sum(map(len,hits.values())))
                records = get_records(self.records,[c['rid'] for c in pool])
                scores = {c['rid']:evidence(query,records[c['rid']],self) for c in pool}
                if mode == 'adaptive_v1':
                    baseline = reduce_candidates(hits,min(12,limit))
                    candidates = select_adaptive(pool,scores,limit,baseline)
                else:
                    candidates = select_similarity(pool,scores,limit)
            else:
                raise ValueError(f'Unknown candidate selection: {mode}')
            for c in candidates:
                c['source'] = source
            result.extend(candidates)
        return result
