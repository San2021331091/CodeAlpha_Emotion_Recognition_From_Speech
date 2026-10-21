"""Fit emotion heads on frozen speech representations, selecting on validation only."""
import json
from pathlib import Path
import time
import joblib
import soundfile as sf
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, precision_recall_fscore_support
from app.ml.datasets import COMMON_EMOTIONS, load_multicorpus_records, split_records
from app.ml.speech_encoder import MODEL_ID, REVISION, LAYERS, encode_many, WavLMAudioModel
from app.ml.trainer import _source_balanced_weights


def main():
    records = load_multicorpus_records(Path('data'))
    splits = split_records(records)
    names = [r.cache_key for r in records]
    cache = Path('data/wavlm_features_v2.npy')
    descriptor = Path('data/wavlm_features_v2.json')
    configuration = {'model':MODEL_ID,'revision':REVISION,'layers':list(LAYERS),'records':names,
                     'padding':'whole-second buckets, attention mask, valid-frame pooling'}
    if descriptor.exists() and json.loads(descriptor.read_text()) != configuration:
        raise ValueError('Speech feature cache configuration mismatch.')
    if not descriptor.exists():
        if cache.exists():
            raise ValueError('Unversioned feature cache exists.')
        descriptor.write_text(json.dumps(configuration))
        x = np.lib.format.open_memmap(cache, mode='w+', dtype='float32', shape=(len(records),3,1536))
        x[:] = np.nan
        x.flush()
    x = np.lib.format.open_memmap(cache, mode='r+')
    started = time.perf_counter()
    pending = [i for i in range(len(records)) if not np.isfinite(x[i]).all()]
    buckets = {}
    for i in pending:
        bucket = int(np.ceil(sf.info(records[i].path).duration))
        buckets.setdefault(bucket, []).append(i)
    completed = len(records)-len(pending)
    last_report = completed
    for bucket in sorted(buckets):
        indices = buckets[bucket]
        for start in range(0,len(indices),8):
            batch = indices[start:start+8]
            x[batch] = encode_many([records[i].path for i in batch])
            completed += len(batch)
            if completed-last_report >= 100 or completed == len(records):
                x.flush()
                print(f'Encoded {completed}/{len(records)} originals ({time.perf_counter()-started:.0f}s)',flush=True)
                last_report = completed
    y = np.array([r.label for r in records])
    sources = np.array([r.source for r in records])
    train, val, test = [np.array(splits[k]) for k in ('train','validation','test')]
    weights = _source_balanced_weights(records, train)
    val_weights = _source_balanced_weights(records, val)
    rav = sources[val] == 'RAVDESS'
    trials, best = [], None
    for layers in ((0,), (1,), (2,), (0,1,2)):
        features = x[:, list(layers)].reshape(len(x), -1)
        for scope in ('all','ravdess'):
            indices = train if scope == 'all' else train[sources[train] == 'RAVDESS']
            for kernel in ('linear','rbf'):
                for c in (.1,1,10):
                    model = make_pipeline(StandardScaler(),SVC(C=c,kernel=kernel,probability=False,class_weight='balanced',random_state=42))
                    model.fit(features[indices],y[indices],svc__sample_weight=weights if scope == 'all' else np.ones(len(indices)))
                    pred = model.predict(features[val])
                    rav_f1 = float(f1_score(y[val][rav],pred[rav],average='macro'))
                    balanced_f1 = float(f1_score(y[val],pred,average='macro',sample_weight=val_weights))
                    score = .6*rav_f1+.4*balanced_f1
                    row = dict(layers=[LAYERS[i] for i in layers],scope=scope,kernel=kernel,C=c,
                               validation_ravdess_accuracy=float(accuracy_score(y[val][rav],pred[rav])),
                               validation_ravdess_f1=rav_f1,validation_balanced_f1=balanced_f1,selection_score=score)
                    trials.append(row)
                    if best is None or score > best[0]:
                        best=(score,layers,scope,kernel,c,row)
                        print('Best validation:',json.dumps(row),flush=True)
    _, layers, scope, kernel, c, row = best
    features = x[:,list(layers)].reshape(len(x),-1)
    indices = train if scope == 'all' else train[sources[train] == 'RAVDESS']
    model = make_pipeline(StandardScaler(),SVC(C=c,kernel=kernel,probability=True,class_weight='balanced',random_state=42))
    model.fit(features[indices],y[indices],svc__sample_weight=weights if scope == 'all' else np.ones(len(indices)))
    # Probability calibration uses the training set only; select labels using the
    # probability output actually served by the application.
    pred = model.predict_proba(features[test]).argmax(1)
    test_results = {}
    for source in sorted(set(sources)):
        mask = sources[test] == source
        test_results[source]={'samples':int(mask.sum()),'accuracy':float(accuracy_score(y[test][mask],pred[mask])),
                              'macro_f1':float(f1_score(y[test][mask],pred[mask],average='macro'))}
    val_pred = model.predict_proba(features[val]).argmax(1)
    report = {'model':'WavLM + SVM','encoder':MODEL_ID,'revision':REVISION,
              'training_originals':len(indices),'augmentation_used':False,
              'winner':row,'test_by_dataset':test_results,'test_accuracy':float(accuracy_score(y[test],pred)),
              'served_validation_macro_f1':float(f1_score(y[val],val_pred,average='macro',sample_weight=val_weights)),
              'confusion_matrix':confusion_matrix(y[test],pred,labels=np.arange(6)).tolist(),
              'emotions':COMMON_EMOTIONS,'test_used_for_selection':False,'trials':trials}
    out=Path('saved_models_speech_encoder')
    out.mkdir(exist_ok=True)
    joblib.dump(WavLMAudioModel(model,layers),out/'speech_encoder_model.joblib')
    (out/'evaluation.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({k:v for k,v in report.items() if k!='trials'},indent=2),flush=True)

if __name__=='__main__':
    main()
