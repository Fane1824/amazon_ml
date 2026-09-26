import pytest

from blocking import reduce_candidates
from data import id_list, partition
from features import feature_dict
from metrics import f05, summarize
from text import views, normalize


@pytest.mark.parametrize('truth,pred,expected', [([],[],1),([],['x'],0),(['x','y'],[],0),
    (['x','y'],['x','y'],1),(['x','y'],['x'],5/6),(['x','y'],['x','y','z'],5/7)])
def test_metric(truth,pred,expected):
    assert f05(truth,pred) == pytest.approx(expected)


def test_macro_includes_singletons():
    assert summarize([([],[],'US'),(['a','b'],['a'],'India')])['macro_f05'] == pytest.approx(11/12)


def test_unicode_and_numbers():
    v = views('École प्राइवेट', '12/4-6 Rue de Paris, 3 bis')
    assert 'प्राइवेट' in v['n']
    assert 'ecole' in v['nf']
    assert '12/4-6' in v['nums']
    assert normalize(normalize('École प्राइवेट')) == normalize('École प्राइवेट')


def test_empty_fields_are_not_positive_evidence():
    class Context:
        def df(self,*args): return 0
    r = dict(v=views('',''),source=2)
    f = feature_dict(r,r,dict(ranks={},fusion=0,rank=1),1,Context())
    assert f['a_equal'] == f['at_edit'] == f['a_tokens'] == 0
    assert f['a_missing_both'] == 1


def test_reduction_preserves_address_channel():
    hits = {'name': [(1,'S2-a'),(2,'S2-b'),(3,'S2-c'),(4,'S2-d')],
            'address': [(5,'S2-e')], 'combined': [(1,'S2-a'),(2,'S2-b')]}
    result = reduce_candidates(hits,4)
    assert len(result) == 4
    assert 5 in {c['rid'] for c in result}
    assert len({c['id'] for c in result}) == 4


def test_signature_split_groups_duplicates():
    assert partition(['France','ecole','rue'],2026) == partition(['France','ecole','rue'],2026)


@pytest.mark.parametrize('value', ['S2-a,S2-a','S1-a','S2-a, S3-b','S2-a,'])
def test_invalid_id_lists(value):
    with pytest.raises(ValueError): id_list(value)
