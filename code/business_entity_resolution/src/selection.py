"""Label-free, deterministic reduction of raw lexical hits before ML scoring."""
import math

from rapidfuzz.fuzz import ratio, token_sort_ratio

from text import fields


def similarity(a, b):
    return max(ratio(a,b), token_sort_ratio(a,b))/100 if a and b else 0.


def evidence(query, target, context):
    a,b = query['v'],target['v']
    name = max(similarity(a[k],b[k]) for k in ('n','nt','nf','core'))
    qa,ta = set(fields(a)['address']),set(fields(b)['address'])
    weights = {t: 1/(1+math.log1p(context.df('address',t))) for t in qa|ta}
    shared = sum(weights[t] for t in qa&ta)
    union = sum(weights.values())
    smaller = min(sum(weights[t] for t in qa),sum(weights[t] for t in ta))
    overlap = .5*shared/union + .5*shared/smaller if union and smaller else 0.
    address = max(similarity(a['at'],b['at']),overlap)
    missing_address = not a['a'] or not b['a']
    joint = .85*name if missing_address else .55*name+.35*address+.1*name*address
    return dict(name=name,address=address,joint=joint,
                missing_target_address=not b['a'])


def select_similarity(candidates, scores, limit):
    """Reserve independent evidence, then fill by joint name/address agreement.

    Candidate dictionaries contain only retrieval metadata. Scores come from text
    and corpus frequencies; neither this function nor evidence accepts labels.
    """
    if limit < 1:
        raise ValueError('Candidate limit must be positive')
    selected = set()
    by_id = {c['rid']:c for c in candidates}
    def ordered(field):
        return sorted(candidates,key=lambda c:(-scores[c['rid']][field],-c['fusion'],c['id']))
    def reserve(items,n):
        for c in items[:n]:
            if len(selected) < limit:
                selected.add(c['rid'])
    # A strong one-field candidate must survive even when its other field is absent.
    missing = [c for c in ordered('name') if scores[c['rid']]['missing_target_address'] and scores[c['rid']]['name'] >= .85]
    reserve(missing,max(1,limit//12))
    reserve(ordered('name'),limit//5)
    reserve(ordered('address'),limit//5)
    for channel in ('name','address','combined'):
        channel_hits = sorted((c for c in candidates if channel in c['ranks']),
                              key=lambda c:(c['ranks'][channel],c['id']))
        reserve(channel_hits,limit//8)
    for c in ordered('joint'):
        if len(selected) >= limit:
            break
        selected.add(c['rid'])
    # Preserve original fusion ranks as classifier features; selection does not
    # pretend that a candidate was retrieved at a better position.
    return sorted((by_id[rid] for rid in selected),key=lambda c:(c['rank'],c['id']))


def select_adaptive(candidates, scores, limit, baseline):
    """Preserve the rank-fusion shortlist; add qualified text-similar alternatives.

    The thresholds are heuristic prefilter rules, not match acceptance rules.
    Every returned candidate still goes through the final matching classifier.
    """
    by_id = {c['rid']:c for c in candidates}
    selected = {c['rid'] for c in baseline}
    if len(selected)>limit or not selected<=by_id.keys():
        raise ValueError('Baseline is inconsistent with the candidate pool/limit')
    extras = [c for c in select_similarity(candidates,scores,limit) if c['rid'] not in selected]
    extras.sort(key=lambda c:(-scores[c['rid']]['joint'],c['id']))
    for c in extras:
        if len(selected)>=limit:
            break
        s = scores[c['rid']]
        if s['joint']>=.60 or s['name']>=.90 or s['address']>=.90:
            selected.add(c['rid'])
    return sorted((by_id[rid] for rid in selected),key=lambda c:(c['rank'],c['id']))
