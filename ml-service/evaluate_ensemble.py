"""Compare deployed models using the existing validation/test split (no training)."""
import argparse
import json
import os
import shutil
import tempfile
import time
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support, roc_auc_score

from app.ml.datasets import COMMON_EMOTIONS, load_multicorpus_records, split_records
from app.ml.ensemble_model import EmotionEnsemble, select_ensemble
from app.ml.trainer import _cache_features, _predict_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--activate", action="store_true", help="Save the validation winner and updated metrics, preserving previous artifacts.")
    args = parser.parse_args()
    records = load_multicorpus_records(Path("data"))
    flat, sequence = _cache_features(Path("data"), records)
    splits = split_records(records)
    labels = np.asarray([r.label for r in records])
    scaler = joblib.load("saved_models/scaler.joblib")
    keys = ["hist_gradient_boosting", "cnn", "lstm", "crnn", "svm", "mlp", "random_forest"]
    probabilities = {}
    models = {}
    for key in keys:
        model = joblib.load(f"saved_models/{key}_model.joblib")
        models[key] = model
        probabilities[key] = {}
        for split in ("validation", "test"):
            indices = splits[split]
            probabilities[key][split] = _predict_model(
                key, model, flat[indices], sequence[indices], scaler.transform(flat[indices]))
    y = labels[splits["validation"]]
    (left, right), (weight, _), score = select_ensemble(
        {key: values["validation"] for key, values in probabilities.items()}, y)
    report = {"validation_accuracy": score[0], "validation_macro_f1": score[1], "left": left, "right": right, "weight": weight}
    p = weight * probabilities[left]["test"] + (1-weight) * probabilities[right]["test"]
    indices = np.asarray(splits["test"])
    report["test_accuracy"] = accuracy_score(labels[indices], p.argmax(1))
    report["test_macro_f1"] = f1_score(labels[indices], p.argmax(1), average="macro")
    for source in sorted({r.source for r in records}):
        mask = np.asarray([records[i].source == source for i in indices])
        report[source] = accuracy_score(labels[indices][mask], p.argmax(1)[mask])
    print(json.dumps(report, indent=2))
    if args.activate:
        metrics = json.loads(Path("saved_models/metrics.json").read_text())
        if metrics["dataset_summary"].get("calibration_samples"):
            raise SystemExit("This model uses separate calibration/selection splits and balanced macro F1. Run train_pipeline.py to update it; legacy accuracy voting will not replace it.")
        previous = metrics["models"][metrics["best_model"]]
        if score <= (previous["validation_accuracy"], previous["validation_macro_f1"]):
            print("Existing model wins validation; artifacts unchanged.")
            return
        ensemble = EmotionEnsemble([(left, models[left]), (right, models[right])], (weight, 1-weight), scaler)
        # Check the persisted inference interface against the evaluation path.
        actual = ensemble.predict_proba((flat[indices], sequence[indices]))
        np.testing.assert_allclose(actual, p, atol=1e-7)
        predicted, truth = p.argmax(1), labels[indices]
        precision, recall, f1, _ = precision_recall_fscore_support(truth, predicted, average="macro", zero_division=0)
        started = time.perf_counter()
        ensemble.predict_proba((flat[indices[:1]], sequence[indices[:1]]))
        result = {"key": "ensemble", "accuracy": round(report["test_accuracy"], 4),
                  "precision": round(float(precision), 4), "recall": round(float(recall), 4),
                  "f1_score": round(float(f1), 4), "roc_auc": round(float(roc_auc_score(truth, p, multi_class="ovr")), 4),
                  "inference_latency_ms": round((time.perf_counter() - started) * 1000, 2),
                  "validation_accuracy": round(score[0], 4), "validation_macro_f1": round(score[1], 4),
                  "test_by_dataset": {}}
        for source in sorted({r.source for r in records}):
            mask = np.asarray([records[i].source == source for i in indices])
            result["test_by_dataset"][source] = {"samples": int(mask.sum()), "accuracy": round(float(accuracy_score(truth[mask], predicted[mask])), 4),
                "macro_f1": round(float(f1_score(truth[mask], predicted[mask], average="macro", zero_division=0)), 4)}
        metrics["models"]["Validation-selected Ensemble"] = result
        metrics["best_model"] = "Validation-selected Ensemble"
        metrics["ensemble"] = {"members": [left, right], "weights": [weight, 1-weight]}
        matrix = confusion_matrix(truth, predicted, labels=np.arange(len(COMMON_EMOTIONS)))
        metrics["confusion_matrix"] = {emotion: {name: int(matrix[i, j]) for j, name in enumerate(COMMON_EMOTIONS)} for i, emotion in enumerate(COMMON_EMOTIONS)}
        ep, er, ef, support = precision_recall_fscore_support(truth, predicted, labels=np.arange(len(COMMON_EMOTIONS)), zero_division=0)
        metrics["per_emotion_metrics"] = {emotion: {"precision": round(float(ep[i]), 4), "recall": round(float(er[i]), 4),
            "f1_score": round(float(ef[i]), 4), "support": int(support[i])} for i, emotion in enumerate(COMMON_EMOTIONS)}
        destination = Path("saved_models")
        backup = destination / ("backup_before_ensemble_" + time.strftime("%Y%m%d_%H%M%S"))
        backup.mkdir()
        for name in ("best_model.joblib", "metrics.json", "ensemble_model.joblib"):
            if (destination / name).exists():
                shutil.copy2(destination / name, backup / name)
        with tempfile.TemporaryDirectory(dir=destination) as temporary:
            staging = Path(temporary)
            joblib.dump(ensemble, staging / "ensemble_model.joblib")
            joblib.dump(ensemble, staging / "best_model.joblib")
            (staging / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
            for artifact in staging.iterdir():
                os.replace(artifact, destination / artifact.name)
        Path("ensemble_evaluation.json").write_text(json.dumps({"previous_test_accuracy": previous["accuracy"], **report}, indent=2))
        print("Activated validation-selected ensemble; previous best model and metrics backed up.")


if __name__ == "__main__":
    main()
