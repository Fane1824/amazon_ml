"""Official singleton-aware, per-reference macro F0.5."""
import numpy as np


def f05(truth, prediction):
    truth, prediction = set(truth), set(prediction)
    if not truth:
        return float(not prediction)
    tp = len(truth & prediction)
    return 1.25 * tp / (len(prediction) + .25 * len(truth))


def summarize(rows):
    scores, countries, singleton, non, exact = [], {}, [], [], []
    tp = fp = fn = 0
    for truth, pred, country in rows:
        truth, pred = set(truth), set(pred)
        score = f05(truth, pred)
        scores.append(score)
        countries.setdefault(country, []).append(score)
        (non if truth else singleton).append(score if truth else float(bool(pred)))
        exact.append(float(truth == pred))
        tp += len(truth & pred)
        fp += len(pred-truth)
        fn += len(truth-pred)
    if not scores:
        raise ValueError('No evaluation references')
    mean = lambda x: float(np.mean(x)) if x else None
    return dict(references=len(scores), macro_f05=mean(scores),
                per_country={c: mean(s) for c, s in countries.items()},
                singleton_false_positive_rate=mean(singleton), non_singleton_macro_f05=mean(non),
                exact_set_accuracy=mean(exact), micro_precision=tp/(tp+fp) if tp+fp else 0.,
                micro_recall=tp/(tp+fn) if tp+fn else 0.)
