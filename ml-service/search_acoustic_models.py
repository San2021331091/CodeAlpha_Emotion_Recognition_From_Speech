"""Compare acoustic classifiers using validation only; test the winner once."""
import json
from pathlib import Path
import joblib
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score
from app.ml.datasets import load_multicorpus_records, split_records
from app.ml.trainer import _cache_features, _source_balanced_weights


def main():
    records = load_multicorpus_records(Path('data'))
    splits = split_records(records)
    flat, sequence = _cache_features(Path('data'), records)
    labels = np.array([r.label for r in records])
    sources = np.array([r.source for r in records])
    train, val, test = [np.array(splits[k]) for k in ('train', 'validation', 'test')]
    weights = _source_balanced_weights(records, train)
    val_weights = _source_balanced_weights(records, val)
    blocks = np.concatenate([sequence[:, :, i:i+32].mean(2) for i in range(0,128,32)], axis=1)
    feature_sets = {'mfcc': flat[:, :80], 'acoustic': flat, 'temporal': np.c_[flat, blocks]}
    trials = []
    best = None
    for feature_name, features in feature_sets.items():
        for scope in ('all', 'ravdess'):
            indices = train if scope == 'all' else train[sources[train] == 'RAVDESS']
            for c in (1, 10, 100):
                for gamma in ('scale', .001, .01):
                    pipeline = make_pipeline(StandardScaler(), SVC(C=c, gamma=gamma, class_weight='balanced'))
                    fit_weights = weights if scope == 'all' else np.ones(len(indices))
                    pipeline.fit(features[indices], labels[indices], svc__sample_weight=fit_weights)
                    pred = pipeline.predict(features[val])
                    mask = sources[val] == 'RAVDESS'
                    rav_f1 = f1_score(labels[val][mask], pred[mask], average='macro')
                    global_f1 = f1_score(labels[val], pred, average='macro', sample_weight=val_weights)
                    score = .6 * rav_f1 + .4 * global_f1
                    row = dict(features=feature_name, scope=scope, C=c, gamma=gamma,
                               validation_ravdess_accuracy=float(accuracy_score(labels[val][mask], pred[mask])),
                               validation_ravdess_f1=float(rav_f1), validation_balanced_f1=float(global_f1), score=float(score))
                    trials.append(row)
                    if best is None or score > best[0]:
                        best = (score, pipeline, feature_name, row)
                        print('Best validation:', json.dumps(row), flush=True)
    _, pipeline, feature_name, row = best
    pred = pipeline.predict(feature_sets[feature_name][test])
    results = {}
    for source in sorted(set(sources)):
        mask = sources[test] == source
        results[source] = {'samples': int(mask.sum()), 'accuracy': float(accuracy_score(labels[test][mask], pred[mask])),
                           'macro_f1':float(f1_score(labels[test][mask], pred[mask], average='macro'))}
    report = {'selection_policy':'.6 RAVDESS validation macro F1 + .4 corpus-balanced validation macro F1',
              'winner': row, 'test': results, 'test_used_for_selection':False, 'trials':trials}
    Path('acoustic_search_evaluation.json').write_text(json.dumps(report, indent=2))
    joblib.dump({'pipeline':pipeline,'features':feature_name},'acoustic_search_candidate.joblib')
    print(json.dumps({'winner':row,'test':results},indent=2),flush=True)

if __name__ == '__main__':
    main()
