"""Exercise shared top-k source partitioning and its filtered fallback."""
import json

import pytest
import tantivy

from blocking import Retriever, country_key, register
from text import fields, views


@pytest.mark.parametrize('mode', ['filtered', 'shared'])
def test_skewed_sources_and_empty_query(tmp_path, mode):
    schema = tantivy.SchemaBuilder()
    schema.add_integer_field('rid', stored=True)
    schema.add_integer_field('source', indexed=True)
    schema.add_text_field('id', stored=True, tokenizer_name='raw')
    for field in ('name', 'gram', 'address', 'number'):
        schema.add_text_field(field, tokenizer_name='split_space', index_option='freq')
    path = tmp_path/'indexes/train'/country_key('US')
    path.mkdir(parents=True)
    index = tantivy.Index(schema.build(), path=str(path))
    register(index)
    writer = index.writer(heap_size=32_000_000, num_threads=1)
    # The global top eight hits are all S2; S3 still needs its own candidates.
    for i in range(41):
        source = 2 if i < 40 else 3
        doc = tantivy.Document(rid=i+1, source=source, id=f'S{source}-{i}')
        value = views('Acme Trading' if source == 2 else 'Acme Other', '12 Main Road')
        for field, terms in fields(value).items():
            doc.add_text(field, ' '.join(terms))
        writer.add_document(doc)
    writer.commit()
    writer.wait_merging_threads()
    (path/'manifest.json').write_text(json.dumps({}))
    cfg = {'retrieval': {'raw_hits': 2, 'per_source_limit': 2,
                        'tokens_per_field': 12, 'search_mode': mode}}
    retriever = Retriever(tmp_path, 'train', cfg)
    hits = retriever.raw_hits({'country': 'US', 'v': views('Acme Trading', '12 Main Road')})
    for channel in ('name', 'address', 'combined'):
        assert len(hits[2][channel]) == 2
        assert hits[3][channel] == [(41, 'S3-40')]
        assert all(eid.startswith('S2-') for _, eid in hits[2][channel])
    empty = retriever.raw_hits({'country': 'US', 'v': views('', '')})
    assert all(not values for channels in empty.values() for values in channels.values())
