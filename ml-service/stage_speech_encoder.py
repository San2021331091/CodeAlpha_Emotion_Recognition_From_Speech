"""Create deployable metrics for the validation-selected speech classifier."""
import json
from pathlib import Path
import shutil
import time
import joblib
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support, confusion_matrix, roc_auc_score
from app.ml.datasets import COMMON_EMOTIONS, load_multicorpus_records, split_records
from app.ml.trainer import _cache_speech_features, _source_balanced_weights, _split_calibration


def main():
    source=Path('saved_models_speech_encoder')
    model=joblib.load(source/'speech_encoder_model.joblib')
    records=load_multicorpus_records(Path('data'))
    split=split_records(records)
    features=_cache_speech_features(Path('data'),records)
    labels=np.array([r.label for r in records])
    test=np.array(split['test'])
    _, selection=_split_calibration(records,split['validation'])
    weights=_source_balanced_weights(records,selection)
    started=time.perf_counter()
    probabilities=model.predict_feature_proba(features[test])
    pred=probabilities.argmax(1)
    val_pred=model.predict_feature_proba(features[selection]).argmax(1)
    p,r,f,_=precision_recall_fscore_support(labels[test],pred,average='macro',zero_division=0)
    metrics=json.loads(Path('saved_models/metrics.json').read_text())
    name='WavLM Speech Classifier'
    report=json.loads((source/'evaluation.json').read_text())
    metrics['models'][name]={'key':'speech_encoder','accuracy':round(float(accuracy_score(labels[test],pred)),4),
        'precision':round(float(p),4),'recall':round(float(r),4),'f1_score':round(float(f),4),
        'roc_auc':round(float(roc_auc_score(labels[test],probabilities,multi_class='ovr')),4),
        'inference_latency_ms':0,'validation_accuracy':round(float(accuracy_score(labels[selection],val_pred,sample_weight=weights)),4),
        'validation_macro_f1':round(float(f1_score(labels[selection],val_pred,average='macro',sample_weight=weights)),4),
        'test_by_dataset':report['test_by_dataset']}
    metrics['best_model']=name
    clip=Path(records[test[0]].path).read_bytes()
    model.predict_audio_proba(clip)  # Exclude the first encoder load from warm latency.
    started=time.perf_counter()
    for _ in range(3):
        model.predict_audio_proba(clip)
    metrics['models'][name]['inference_latency_ms']=round((time.perf_counter()-started)*1000/3,2)
    matrix=confusion_matrix(labels[test],pred,labels=np.arange(6))
    metrics['confusion_matrix']={emotion:{other:int(matrix[i,j]) for j,other in enumerate(COMMON_EMOTIONS)} for i,emotion in enumerate(COMMON_EMOTIONS)}
    ep,er,ef,support=precision_recall_fscore_support(labels[test],pred,labels=np.arange(6),zero_division=0)
    metrics['per_emotion_metrics']={emotion:{'precision':round(float(ep[i]),4),'recall':round(float(er[i]),4),'f1_score':round(float(ef[i]),4),'support':int(support[i])} for i,emotion in enumerate(COMMON_EMOTIONS)}
    metrics['top_features']=[]
    metrics['dataset_summary']['speech_encoder']={'name':report['encoder'],'revision':report['revision'],
        'training_originals':report['training_originals'],'augmented_training_samples':0,
        'sample_rate':16000,'features':'Layer 8 hidden-state mean and standard deviation',
        'selection':'Validation only; frozen self-supervised encoder and SVM head trained on original training recordings.'}
    shutil.copy2('saved_models/scaler.joblib',source/'scaler.joblib')
    shutil.copy2(source/'speech_encoder_model.joblib',source/'best_model.joblib')
    (source/'metrics.json').write_text(json.dumps(metrics,indent=2))
    print(json.dumps(metrics['models'][name],indent=2))

if __name__=='__main__':
    main()
