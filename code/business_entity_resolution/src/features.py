"""Versioned numerical pair features; no IDs, row order, or country categories."""
import math

import numpy as np
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler, Levenshtein

from text import fields, grams


def overlap(a, b):
    a, b = set(a), set(b)
    return len(a & b)/len(a | b) if a or b else 0.


def equal(a, b):
    return float(bool(a) and a == b)


def ratio(a, b):
    return min(len(a), len(b))/max(len(a), len(b)) if a and b else 0.


def edit(a, b):
    return Levenshtein.normalized_similarity(a, b) if a and b else 0.


def token_sort(a, b):
    return fuzz.token_sort_ratio(a, b)/100 if a and b else 0.


def weighted(a, b, field, context):
    a, b = set(a), set(b)
    weights = {t: 1/(1+math.log1p(context.df(field, t))) for t in a | b}
    return sum(weights[t] for t in a & b)/sum(weights.values()) if weights else 0.


def feature_dict(q, t, candidate, count, context):
    a, b = q['v'], t['v']
    af, bf = fields(a), fields(b)
    f = {}
    for key in ('n', 'core', 'nt', 'nf', 'a', 'ac', 'at', 'af'):
        f[key+'_equal'] = equal(a[key], b[key])
        f[key+'_edit'] = edit(a[key], b[key])
    for key in ('n', 'nt', 'a', 'at'):
        f[key+'_sort'] = token_sort(a[key], b[key])
        f[key+'_tokens'] = overlap(a[key].split(), b[key].split())
        f[key+'_length'] = ratio(a[key], b[key])
    f['name_jaro'] = JaroWinkler.normalized_similarity(a['n'], b['n']) if a['n'] and b['n'] else 0.
    f['name_grams'] = overlap(grams(a['n']), grams(b['n']))
    for field in ('name', 'address'):
        f[field+'_weighted'] = weighted(af[field], bf[field], field, context)
        common = set(af[field]) & set(bf[field])
        f[field+'_rarest_shared'] = max((1/(1+math.log1p(context.df(field, x))) for x in common), default=0.)
    f['address_numbers'] = overlap(a['nums'], b['nums'])
    f['number_conflict'] = float(bool(a['nums']) and bool(b['nums']) and not set(a['nums']) & set(b['nums']))
    f['name_numbers'] = overlap(a['name_nums'], b['name_nums'])
    ai, bi = ''.join(t[0] for t in a['core'].split()), ''.join(t[0] for t in b['core'].split())
    f['acronym'] = float((len(ai)>1 and ai == b['core'].replace(' ', '')) or (len(bi)>1 and bi == a['core'].replace(' ', '')))
    f['scripts'] = overlap(a['scripts'], b['scripts'])
    f['transliteration_gain'] = f['nt_edit']-f['n_edit']
    for key in ('n', 'a'):
        f[key+'_missing_any'] = float(not a[key] or not b[key])
        f[key+'_missing_both'] = float(not a[key] and not b[key])
    f['joint_min'] = min(f['nt_edit'], f['at_tokens'])
    f['joint_product'] = f['nt_edit']*f['at_tokens']
    for channel in ('name', 'address', 'combined'):
        r = candidate['ranks'].get(channel)
        f[channel+'_retrieval_rr'] = 1/r if r else 0.
    f['fusion'] = candidate['fusion']
    f['fusion_rr'] = 1/candidate['rank']
    f['candidate_count'] = count
    f['source3'] = float(t['source'] == 3)
    return f


def matrix(rows, records, context):
    values, labels, weights, offsets, names = [], [], [], [0], None
    for row in rows:
        q = records[row['rid']]
        context.country(q['country'])
        truth = set(row['truth'])
        candidates = row['candidates']
        for c in candidates:
            f = feature_dict(q, records[c['rid']], c, len(candidates), context)
            names = list(f)
            values.append(list(f.values()))
            labels.append(int(c['id'] in truth))
            weights.append(1/len(candidates))
        offsets.append(len(values))
    if names is None:
        names = feature_names()
    return (np.asarray(values, dtype=np.float32).reshape(-1, len(names)), np.asarray(labels, dtype=np.uint8),
            np.asarray(weights, dtype=np.float32), np.asarray(offsets, dtype=np.int64), names)


def feature_names():
    from text import views
    class Context:
        def df(self, *args):
            return 0
    r = {'v': views('', ''), 'source': 2}
    return list(feature_dict(r, r, {'ranks': {}, 'fusion': 0., 'rank': 1}, 1, Context()))
