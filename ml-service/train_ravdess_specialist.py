"""Select a RAVDESS-focused head using validation, then report every test corpus."""
import json
from pathlib import Path
import joblib
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score
from app.ml.datasets import load_multicorpus_records, split_records
from app.ml.speech_encoder import LAYERS, WavLMAudioModel
from app.ml.trainer import _source_balanced_weights


def main():
    general = json.loads(Path('saved_models_speech_encoder/evaluation.json').read_text())
    # This objective is declared independently of any test result.
    winner = max(general['trials'],key=lambda row:(row['validation_ravdess_f1'],row['validation_ravdess_accuracy'],row['validation_balanced_f1']))
    records = load_multicorpus_records(Path('data'))
    configuration = json.loads(Path('data/wavlm_features_v2.json').read_text())
    if configuration['records'] != [record.cache_key for record in records]:
        raise ValueError('Feature-cache record order differs from the current dataset.')
    splits = split_records(records)
    cache = np.load('data/wavlm_features_v2.npy',mmap_mode='r')
    layers = tuple(LAYERS.index(layer) for layer in winner['layers'])
    features = cache[:,list(layers)].reshape(len(records),-1)
    labels = np.array([r.label for r in records])
    sources = np.array([r.source for r in records])
    train, val, test = [np.array(splits[k]) for k in ('train','validation','test')]
    weights = _source_balanced_weights(records,train)
    indices = train if winner['scope']=='all' else train[sources[train]=='RAVDESS']
    model = make_pipeline(StandardScaler(),SVC(C=winner['C'],kernel=winner['kernel'],probability=True,class_weight='balanced',random_state=42))
    model.fit(features[indices],labels[indices],svc__sample_weight=weights if winner['scope']=='all' else np.ones(len(indices)))
    pred = model.predict_proba(features[test]).argmax(1)
    val_pred = model.predict_proba(features[val]).argmax(1)
    results={}
    for source in sorted(set(sources)):
        mask=sources[test]==source
        results[source]={'samples':int(mask.sum()),'accuracy':float(accuracy_score(labels[test][mask],pred[mask])),
                         'macro_f1':float(f1_score(labels[test][mask],pred[mask],average='macro'))}
    report={'model':'WavLM RAVDESS specialist','selection':'RAVDESS validation macro F1, then validation accuracy',
            'winner':winner,'test_by_dataset':results,'test_used_for_selection':False,
            'validation_balanced_f1':float(f1_score(labels[val],val_pred,average='macro',sample_weight=_source_balanced_weights(records,val)))}
    joblib.dump(WavLMAudioModel(model,layers),'saved_models_speech_encoder/ravdess_specialist_model.joblib')
    Path('saved_models_speech_encoder/specialist_evaluation.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2),flush=True)

if __name__=='__main__':
    main()
