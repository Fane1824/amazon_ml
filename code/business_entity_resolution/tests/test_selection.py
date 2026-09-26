from selection import evidence,select_similarity,select_adaptive
from text import views


class Context:
    def df(self,*args): return 1


def test_missing_address_keeps_strong_name_candidate():
    candidates = [dict(rid=i,id=f'S2-{i:02d}',ranks={'combined':i+1},fusion=1/(60+i),rank=i+1) for i in range(20)]
    scores = {i:dict(name=.6,address=.8,joint=.7,missing_target_address=False) for i in range(20)}
    scores[19] = dict(name=1.,address=0.,joint=.85,missing_target_address=True)
    chosen = select_similarity(candidates,scores,12)
    assert len(chosen)==12 and 19 in {c['rid'] for c in chosen}
    assert select_similarity(list(reversed(candidates)),scores,12)==chosen


def test_empty_evidence_is_zero():
    r={'v':views('','')}
    s=evidence(r,r,Context())
    assert s['name']==s['address']==s['joint']==0


def test_limit_and_no_candidate_mutation():
    c=dict(rid=1,id='S2-1',ranks={'name':1},fusion=.2,rank=7)
    score={1:dict(name=1,address=1,joint=1,missing_target_address=False)}
    assert select_similarity([c],score,12)==[c]
    assert c['rank']==7
    assert select_similarity([],{},12)==[]


def test_adaptive_preserves_baseline_without_padding():
    candidates=[dict(rid=i,id=f'S2-{i}',ranks={},fusion=.1,rank=i+1) for i in range(5)]
    scores={i:dict(name=.1,address=.1,joint=.1,missing_target_address=False) for i in range(5)}
    scores[4].update(name=1.,joint=.9)
    chosen=select_adaptive(candidates,scores,5,candidates[:2])
    assert {c['rid'] for c in chosen}=={0,1,4}
    assert len(chosen)<5
