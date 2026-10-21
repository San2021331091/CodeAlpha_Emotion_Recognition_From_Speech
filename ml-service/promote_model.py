"""Promote a staged multicorpus training run if it wins balanced validation F1."""
import argparse
import json
import os
from pathlib import Path
import shutil
import tempfile
import time

import numpy as np
from sklearn.metrics import f1_score

from app.ml.datasets import load_multicorpus_records, split_records
from app.ml.predictor import EmotionPredictor
from app.ml.trainer import _cache_features, _cache_speech_features, _predict_model, _source_balanced_weights, _split_calibration


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="saved_models_balanced")
    parser.add_argument("--report-file", default="balanced_training_evaluation.json")
    args = parser.parse_args()
    source, destination = Path(args.source), Path("saved_models")
    candidate = EmotionPredictor(str(source))
    previous = EmotionPredictor(str(destination))
    records = load_multicorpus_records(Path("data"))
    flat, sequence = _cache_features(Path("data"), records)
    _, indices = _split_calibration(records, split_records(records)["validation"])
    truth = np.asarray([records[i].label for i in indices])
    weights = _source_balanced_weights(records, indices)
    scores = {}
    speech_features = _cache_speech_features(Path('data'), records) if any(
        getattr(p.models['best'], 'audio_model', False) for p in (candidate, previous)
    ) else None
    for name, predictor in (("previous", previous), ("candidate", candidate)):
        key = predictor.metrics["models"][predictor.best_model_name].get("key", "best")
        probabilities = _predict_model(key, predictor.models["best"], flat[indices], sequence[indices], predictor.scaler.transform(flat[indices]),
                                       audio_features=speech_features[indices] if speech_features is not None else None)
        scores[name] = float(f1_score(truth, probabilities.argmax(1), average="macro", sample_weight=weights))
    report = {"selection_macro_f1": scores,
              "previous_model": previous.best_model_name, "candidate_model": candidate.best_model_name,
              "previous_happy_metrics": previous.metrics["per_emotion_metrics"]["happy"],
              "candidate_happy_metrics": candidate.metrics["per_emotion_metrics"]["happy"],
              "candidate_test_metrics": candidate.metrics["models"][candidate.best_model_name],
              "promoted": False}
    if scores["candidate"] > scores["previous"]:
        backup = destination / ("backup_before_balanced_" + time.strftime("%Y%m%d_%H%M%S"))
        backup.mkdir()
        artifacts = [p for p in source.iterdir() if p.suffix == ".joblib" or p.name == "metrics.json"]
        for artifact in artifacts:
            if (destination / artifact.name).exists():
                shutil.copy2(destination / artifact.name, backup / artifact.name)
        with tempfile.TemporaryDirectory(dir=destination) as temporary:
            staging = Path(temporary)
            for artifact in artifacts:
                shutil.copy2(artifact, staging / artifact.name)
            for artifact in staging.iterdir():
                os.replace(artifact, destination / artifact.name)
        report["promoted"] = True
    Path(args.report_file).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
