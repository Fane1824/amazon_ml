"""Read-only diagnostics for a frozen model, threshold, and scored partition."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from catboost import CatBoostClassifier
from common import read_json, verify, write_json
from learn import scored_rows
from metrics import summarize


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--part', choices=['threshold', 'eval'], default='eval')
    args = parser.parse_args()
    root = args.run_dir
    verify(root/'decision')
    decision = read_json(root/'decision/threshold.json')
    meta = verify(root/'scores'/args.part)
    if meta['model_hash'] != decision['model_hash']:
        raise ValueError('Model and threshold differ')
    groups, errors, examples = {}, {}, []
    for row in scored_rows(root, args.part):
        truth = set(row['truth'])
        candidates = {c['id'] for c in row['candidates']}
        pred = {c['id'] for c, s in zip(row['candidates'], row['scores']) if s >= decision['threshold']}
        bucket = '0' if not truth else '1' if len(truth) == 1 else '2-4' if len(truth) <= 4 else '5+'
        for group in ('ALL', row['country'], 'truth_count:'+bucket):
            groups.setdefault(group, []).append((truth, pred, row['country']))
            e = errors.setdefault(group, dict(singleton_references=0, false_positive_pairs=0,
                                 blocking_missed_pairs=0, classifier_missed_pairs=0,
                                 true_positive_pairs=0))
            e['singleton_references'] += not truth
            e['false_positive_pairs'] += len(pred-truth)
            e['blocking_missed_pairs'] += len(truth-candidates)
            e['classifier_missed_pairs'] += len((truth & candidates)-pred)
            e['true_positive_pairs'] += len(truth & pred)
        for c, score in zip(row['candidates'], row['scores']):
            if c['id'] in pred-truth:
                examples.append(dict(reference=row['id'], target=c['id'], country=row['country'],
                                     score=score, singleton=not truth))
    summaries = {g: dict(**summarize(rows), **errors[g],
                        all_empty_baseline=errors[g]['singleton_references']/len(rows))
                 for g, rows in groups.items()}
    cfg = read_json(root/'model/config.json')
    weights = cfg['threshold']['country_weights']
    proxy = sum(w*summaries[c]['macro_f05'] for c,w in weights.items())/sum(weights.values()) if weights else None
    model = CatBoostClassifier()
    model.load_model(str(root/'model/model.cbm'))
    importance = sorted(zip(model.feature_names_, map(float, model.get_feature_importance())), key=lambda x: -x[1])
    write_json(root/'reports'/f'diagnostics_{args.part}.json', dict(
        threshold=decision['threshold'], model_hash=decision['model_hash'],
        country_weighted_proxy=proxy, groups=summaries, trees=model.tree_count_,
        feature_importance=[dict(feature=n, importance=v) for n,v in importance],
        highest_confidence_false_positives=sorted(examples,key=lambda x: -x['score'])[:25],
        note='Diagnostic only. No model or threshold is modified; France has no labeled score.'))
    print(summaries)


if __name__ == '__main__':
    main()
