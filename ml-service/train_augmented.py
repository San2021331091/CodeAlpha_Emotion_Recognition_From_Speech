"""Train a CNN on the expanded audio corpus, then evaluate on original held-out clips."""
import argparse
import json
from pathlib import Path
import shutil
import time

import joblib
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support, roc_auc_score

from app.ml.calibrated_model import CalibratedEmotionModel
from app.ml.datasets import COMMON_EMOTIONS, load_multicorpus_records, split_records
from app.ml.ensemble_model import EmotionEnsemble, select_ensemble
from app.ml.predictor import EmotionPredictor
from app.ml.torch_models import TorchSequenceModel
from app.ml.trainer import _cache_features, _cache_speech_features, _predict_model, _source_balanced_weights, _split_calibration


def load_training_corpus(directory, records, train_indices):
    directory = Path(directory).resolve()
    metadata = json.loads((directory / "metadata.json").read_text())
    with (directory / "manifest.jsonl").open(encoding="utf-8") as manifest:
        entries = [json.loads(line) for line in manifest]
    originals = {records[i].cache_key: records[i] for i in train_indices}
    if set(metadata["original_training_keys"]) != set(originals):
        raise ValueError("Expanded corpus does not match this training split.")
    if len(entries) != metadata["target_count"]:
        raise ValueError("Training manifest count is incorrect.")
    original_entries = []
    for index, entry in enumerate(entries):
        parent = originals.get(entry["parent_key"])
        if parent is None or entry["split"] != "train":
            raise ValueError("Held-out or unrecognized parent found in training manifest.")
        if entry["index"] != index or entry["label"] != parent.label or entry["emotion"] != COMMON_EMOTIONS[parent.label] or entry["source"] != parent.source or entry["speaker"] != parent.speaker:
            raise ValueError("Augmented sample metadata does not match its original recording.")
        path = (directory / entry["path"]).resolve()
        if entry["kind"] == "original":
            if path != Path(parent.path).resolve() or entry["seed"] is not None:
                raise ValueError("Original training audio has an unexpected path or augmentation seed.")
            original_entries.append(entry["parent_key"])
        elif entry["kind"] == "augmented":
            if not path.is_relative_to(directory / "wav") or entry["seed"] is None:
                raise ValueError("Augmented waveform must be inside the corpus WAV directory.")
        else:
            raise ValueError("Unrecognized training sample kind.")
        if not path.is_file():
            raise ValueError(f"Training waveform is missing: {path}")
        if metadata.get("organized_by_emotion"):
            class_path = (directory / entry["class_audio_path"]).resolve()
            expected_folder = directory / "wav" / COMMON_EMOTIONS[parent.label]
            if class_path.parent != expected_folder or not class_path.is_file():
                raise ValueError("Audio is missing or organized under the wrong emotion folder.")
    if len(original_entries) != len(originals) or set(original_entries) != set(originals):
        raise ValueError("Each original training recording must occur exactly once.")
    sequence = np.load(directory / "sequences.npy", mmap_mode="r")
    labels = np.load(directory / "labels.npy")
    if sequence.shape != (len(entries), 40, 128) or labels.shape != (len(entries),):
        raise ValueError("Expanded feature cache has invalid dimensions.")
    if not np.array_equal(labels, [entry["label"] for entry in entries]):
        raise ValueError("Feature-cache labels do not match the waveform manifest.")
    if metadata.get("exact_class_balance"):
        counts = np.bincount(labels, minlength=len(COMMON_EMOTIONS))
        if len(set(counts.tolist())) != 1:
            raise ValueError("Training corpus claims exact class balance but its emotion counts differ.")
    if "maximum_class_count_difference" in metadata:
        counts = np.bincount(labels, minlength=len(COMMON_EMOTIONS))
        difference = int(counts.max() - counts.min())
        if difference > 1 or difference != metadata["maximum_class_count_difference"]:
            raise ValueError("Training emotion counts do not match the balanced corpus metadata.")
    for start in range(0, len(sequence), 1024):
        if not np.isfinite(sequence[start:start + 1024]).all():
            raise ValueError("Expanded feature cache contains invalid numbers.")
    return metadata, sequence, labels


def summarize(probabilities, truth, sources, selection_score, key, latency):
    predicted = probabilities.argmax(1)
    precision, recall, f1, _ = precision_recall_fscore_support(truth, predicted, average="macro", zero_division=0)
    result = {"key": key, "accuracy": round(float(accuracy_score(truth, predicted)), 4),
              "precision": round(float(precision), 4), "recall": round(float(recall), 4), "f1_score": round(float(f1), 4),
              "roc_auc": round(float(roc_auc_score(truth, probabilities, multi_class="ovr")), 4),
              "inference_latency_ms": round(latency, 2), "validation_accuracy": round(selection_score[1], 4),
              "validation_macro_f1": round(selection_score[0], 4), "test_by_dataset": {}}
    for source in sorted(set(sources)):
        mask = sources == source
        result["test_by_dataset"][source] = {"samples": int(mask.sum()),
            "accuracy": round(float(accuracy_score(truth[mask], predicted[mask])), 4),
            "macro_f1": round(float(f1_score(truth[mask], predicted[mask], average="macro", zero_division=0)), 4)}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-dir", default="data/training_50000")
    parser.add_argument("--models-dir", default="saved_models_50000")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--ravdess-weight", type=float, default=1.0,
                        help="Relative training-loss weight for RAVDESS examples; does not change held-out splits.")
    parser.add_argument("--evaluate-only", action="store_true", help="Re-evaluate the CNN already saved in the staging directory without training again.")
    args = parser.parse_args()
    if not np.isfinite(args.ravdess_weight) or args.ravdess_weight <= 0:
        parser.error("--ravdess-weight must be finite and positive")
    records = load_multicorpus_records(Path("data").resolve())
    splits = split_records(records)
    metadata, training, training_labels = load_training_corpus(args.corpus_dir, records, splits["train"])
    flat, sequence = _cache_features(Path("data"), records)
    labels = np.asarray([r.label for r in records])
    calibration, selection = _split_calibration(records, splits["validation"])
    selection_weights = _source_balanced_weights(records, selection)
    calibration_weights = _source_balanced_weights(records, calibration)
    previous = EmotionPredictor("saved_models")
    baseline = previous.models["best"]
    baseline_key = previous.metrics["models"][previous.best_model_name].get("key", "best")
    speech_features = _cache_speech_features(Path('data'), records) if getattr(baseline,'audio_model',False) else None
    training_weights = None
    if args.ravdess_weight != 1.0:
        with (Path(args.corpus_dir) / "manifest.jsonl").open(encoding="utf-8") as handle:
            training_sources = [json.loads(line)["source"] for line in handle]
        training_weights = np.asarray([
            args.ravdess_weight if source == "RAVDESS" else 1.0 for source in training_sources
        ], dtype=np.float32)
        print(f"RAVDESS training-loss weight: {args.ravdess_weight:g}x; held-out splits unchanged.", flush=True)
    if args.evaluate_only:
        cnn = joblib.load(Path(args.models_dir) / "augmented_cnn_model.joblib")
    else:
        print(f"Training CNN on {len(training):,} waveforms, including {metadata['augmented_training_samples']:,} augmented examples.", flush=True)
        cnn = TorchSequenceModel("cnn", num_classes=len(COMMON_EMOTIONS)).fit(
            training, training_labels, validation_data=(sequence[selection], labels[selection]),
            epochs=args.epochs, batch_size=args.batch_size, validation_sample_weights=selection_weights,
            sample_weights=training_weights)
    selection_probs = {"cnn": cnn.predict_proba(sequence[selection]),
                       "deployed": _predict_model(baseline_key, baseline, flat[selection], sequence[selection], previous.scaler.transform(flat[selection]),
                                                  audio_features=speech_features[selection] if speech_features is not None else None)}
    if speech_features is None:
        members, weights, _ = select_ensemble(selection_probs, labels[selection], selection_weights, prefer_macro_f1=True)
        ensemble = EmotionEnsemble([(key, {"cnn": cnn, "deployed": baseline}[key]) for key in members], weights, previous.scaler)
    else:
        # MFCC ensembles cannot call a raw-audio encoder. Compare it as an
        # independent baseline using its exact cached training representation.
        members, weights, ensemble = ['cnn'], [1.0], None
    calibrated = CalibratedEmotionModel(cnn, "cnn", previous.scaler).fit_probabilities(
        cnn.predict_proba(sequence[calibration]), labels[calibration], calibration_weights)
    candidates = {"augmented_cnn": cnn, "augmented_calibrated_cnn": calibrated}
    # A zero-weight augmented branch is simply the previous model. Never
    # describe that result as a newly trained 50,000-example ensemble.
    if ensemble is not None and sum(weight for key, weight in zip(members, weights) if key == "cnn") > 0:
        candidates["augmented_ensemble"] = ensemble
    names = {"augmented_cnn": f"Augmented 1D-CNN ({len(training):,} training examples)",
             "augmented_ensemble": f"Augmented CNN Ensemble ({len(training):,} training examples)",
             "augmented_calibrated_cnn": f"Calibrated Augmented CNN ({len(training):,} training examples)"}
    scores = {}
    for key, model in candidates.items():
        p = _predict_model("cnn" if key == "augmented_cnn" else key, model, flat[selection], sequence[selection], previous.scaler.transform(flat[selection]))
        pred = p.argmax(1)
        scores[key] = (float(f1_score(labels[selection], pred, average="macro", sample_weight=selection_weights)),
                       float(accuracy_score(labels[selection], pred, sample_weight=selection_weights)))
    best = max(scores, key=scores.get)
    destination = Path(args.models_dir)
    if destination.resolve() == Path("saved_models").resolve():
        raise ValueError("Stage the expanded model in a separate directory before promotion.")
    destination.mkdir(exist_ok=True)
    for artifact in Path("saved_models").iterdir():
        if artifact.is_file() and (artifact.suffix == ".joblib" or artifact.name == "metrics.json"):
            shutil.copy2(artifact, destination / artifact.name)
    metrics = json.loads(json.dumps(previous.metrics))
    test = np.asarray(splits["test"])
    test_sources = np.asarray([records[i].source for i in test])
    test_probabilities = {}
    for key, model in candidates.items():
        routed_key = "cnn" if key == "augmented_cnn" else key
        p = _predict_model(routed_key, model, flat[test], sequence[test], previous.scaler.transform(flat[test]))
        started = time.perf_counter()
        _predict_model(routed_key, model, flat[test[:1]], sequence[test[:1]], previous.scaler.transform(flat[test[:1]]))
        metrics["models"][names[key]] = summarize(p, labels[test], test_sources, scores[key], key, (time.perf_counter() - started) * 1000)
        test_probabilities[key] = p
        joblib.dump(model, destination / f"{key}_model.joblib")
    best_pred = test_probabilities[best].argmax(1)
    matrix = confusion_matrix(labels[test], best_pred, labels=np.arange(6))
    metrics["confusion_matrix"] = {emotion: {name: int(matrix[i, j]) for j, name in enumerate(COMMON_EMOTIONS)} for i, emotion in enumerate(COMMON_EMOTIONS)}
    ep, er, ef, support = precision_recall_fscore_support(labels[test], best_pred, labels=np.arange(6), zero_division=0)
    metrics["per_emotion_metrics"] = {emotion: {"precision": round(float(ep[i]), 4), "recall": round(float(er[i]), 4),
        "f1_score": round(float(ef[i]), 4), "support": int(support[i])} for i, emotion in enumerate(COMMON_EMOTIONS)}
    metrics["best_model"] = names[best]
    metrics["dataset_summary"].update({"effective_training_samples": len(training),
        "augmented_training_samples": metadata["augmented_training_samples"],
        "training_augmentation": {k: v for k, v in metadata.items() if k != "original_training_keys"}})
    baseline_score = float(f1_score(labels[selection], selection_probs["deployed"].argmax(1), average="macro", sample_weight=selection_weights))
    metrics["expanded_training_run"] = {"epochs_limit": args.epochs, "batch_size": args.batch_size,
        "ravdess_training_weight": args.ravdess_weight,
        "physical_training_examples": len(training),
        "baseline_selection_macro_f1": baseline_score, "candidate_selection_macro_f1": scores[best][0],
        "ensemble_members": list(members), "ensemble_weights": list(weights), "test_used_for_selection": False}
    joblib.dump(candidates[best], destination / "best_model.joblib")
    (destination / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps({"best_model": names[best], "selection": metrics["expanded_training_run"],
        "test": metrics["models"][names[best]], "happy": metrics["per_emotion_metrics"]["happy"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
