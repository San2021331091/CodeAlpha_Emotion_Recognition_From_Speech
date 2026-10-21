import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from app.ml.datasets import COMMON_EMOTIONS, AudioRecord, load_multicorpus_records, split_records
from app.ml.feature_extractor import extract_features_from_audio, extract_sequence_features_from_audio
from app.ml.hybrid_model import HybridEmotionModel
from app.ml.benchmarks import create_benchmarks
from app.ml.torch_models import TorchSequenceModel
from app.ml.ensemble_model import EmotionEnsemble, select_ensemble
from app.ml.calibrated_model import CalibratedEmotionModel


def _split_calibration(records, indices):
    """Separate class-bias fitting clips from model-selection clips."""
    rng = np.random.default_rng(42)
    calibration, selection = [], []
    groups = sorted({(records[i].source, records[i].label) for i in indices})
    for source, label in groups:
        members = np.asarray([i for i in indices if records[i].source == source and records[i].label == label])
        if len(members) < 2:
            raise ValueError(f"Need two validation recordings for {source}/{COMMON_EMOTIONS[label]}.")
        members = rng.permutation(members)
        boundary = max(1, len(members) // 2)
        calibration.extend(members[:boundary])
        selection.extend(members[boundary:])
    return np.sort(calibration), np.sort(selection)


def _cache_features(data_dir: Path, records: list[AudioRecord]) -> tuple[np.ndarray, np.ndarray]:
    # v2 uses full-utterance temporal interpolation instead of truncation/padding.
    cache_path = data_dir / "multicorpus_features_v2.npz"
    expected_names = [record.cache_key for record in records]

    if cache_path.is_file():
        with np.load(cache_path, allow_pickle=False) as cache:
            names = [str(name) for name in cache["filenames"]]
            if (
                len(names) == len(expected_names)
                and len(set(names)) == len(names)
                and set(names) == set(expected_names)
            ):
                order = {name: index for index, name in enumerate(names)}
                positions = [order[name] for name in expected_names]
                flat = np.asarray(cache["X_flat"][positions], dtype=np.float32)
                sequence = np.asarray(cache["X_seq"][positions], dtype=np.float32)
                if (
                    flat.shape == (len(records), 231)
                    and sequence.shape == (len(records), 40, 128)
                    and np.isfinite(flat).all()
                    and np.isfinite(sequence).all()
                ):
                    return flat, sequence

    flat_features = np.empty((len(records), 231), dtype=np.float32)
    for index, record in enumerate(records):
        flat_features[index], _, _ = extract_features_from_audio(record.path)
        if (index + 1) % 100 == 0 or index + 1 == len(records):
            print(f"  Extracted flat features for {index + 1}/{len(records)} recordings", flush=True)

    sequence_features = np.empty((len(records), 40, 128), dtype=np.float32)
    for index, record in enumerate(records):
        sequence_features[index] = extract_sequence_features_from_audio(record.path)
        if (index + 1) % 100 == 0 or index + 1 == len(records):
            print(f"  Extracted temporal features for {index + 1}/{len(records)} recordings", flush=True)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=cache_path.parent, suffix=".npz", delete=False) as temporary:
        temporary_path = Path(temporary.name)
        np.savez_compressed(
            temporary,
            X_flat=flat_features,
            X_seq=sequence_features,
            filenames=np.asarray(expected_names),
        )
    os.replace(temporary_path, cache_path)
    return flat_features, sequence_features


def _feature_names() -> list[str]:
    return (
        [f"mfcc_mean_{index}" for index in range(40)]
        + [f"mfcc_std_{index}" for index in range(40)]
        + [f"mfcc_delta_mean_{index}" for index in range(40)]
        + [f"mfcc_delta_std_{index}" for index in range(40)]
        + [f"chroma_mean_{index}" for index in range(12)]
        + [f"chroma_std_{index}" for index in range(12)]
        + [f"mel_subband_{index}" for index in range(16)]
        + [f"spectral_contrast_mean_{index}" for index in range(7)]
        + [f"spectral_contrast_std_{index}" for index in range(7)]
        + [f"tonnetz_mean_{index}" for index in range(6)]
        + [
            "zero_crossing_rate_mean",
            "zero_crossing_rate_std",
            "rms_energy_mean",
            "rms_energy_std",
            "spectral_centroid_mean",
            "spectral_centroid_std",
            "spectral_rolloff_mean",
            "spectral_rolloff_std",
            "pitch_f0_mean",
            "pitch_f0_std",
            "tempo",
        ]
    )


def _source_balanced_weights(records: list[AudioRecord], indices: np.ndarray) -> np.ndarray:
    counts: dict[tuple[str, int], int] = {}
    sources = {records[index].source for index in indices}
    for index in indices:
        record = records[index]
        key = (record.source, record.label)
        counts[key] = counts.get(key, 0) + 1
    missing = [
        f"{source}/{emotion}"
        for source in sorted(sources)
        for label, emotion in enumerate(COMMON_EMOTIONS)
        if (source, label) not in counts
    ]
    if missing:
        raise ValueError(
            "Each included training corpus must contain examples for every shared emotion; "
            f"missing: {', '.join(missing)}."
        )
    weights = np.asarray(
        [
            1.0 / (len(sources) * len(COMMON_EMOTIONS) * counts[(records[index].source, records[index].label)])
            for index in indices
        ],
        dtype=np.float32,
    )
    return weights / weights.mean()


def _describe_split(records: list[AudioRecord], indices: np.ndarray) -> dict[str, int]:
    return {
        source: sum(records[index].source == source for index in indices)
        for source in sorted({record.source for record in records})
    }


def _predict_model(
    key: str,
    model: Any,
    flat_features: np.ndarray,
    sequence_features: np.ndarray,
    scaled_flat_features: np.ndarray,
    audio_features=None,
) -> np.ndarray:
    if getattr(model, "audio_model", False):
        if audio_features is None:
            raise ValueError("Cached speech-encoder features are required to evaluate an audio model.")
        return model.predict_feature_proba(audio_features)
    if getattr(model, "multimodal_model", False):
        return model.predict_proba((flat_features, sequence_features))
    if getattr(model, "sequence_model", False) or key in {"cnn", "lstm", "crnn"}:
        return model.predict_proba(sequence_features)
    if key in {"mlp", "random_forest", "svm"}:
        return model.predict_proba(scaled_flat_features)
    return model.predict_proba(flat_features)


def _cache_speech_features(data_dir, records):
    from app.ml.speech_encoder import MODEL_ID, REVISION
    data_dir = Path(data_dir)
    configuration = json.loads((data_dir / "wavlm_features_v2.json").read_text())
    if configuration['model'] != MODEL_ID or configuration['revision'] != REVISION or configuration['records'] != [r.cache_key for r in records]:
        raise ValueError("Speech feature cache does not match the encoder or dataset record order.")
    features = np.load(data_dir / "wavlm_features_v2.npy", mmap_mode='r')
    if features.shape != (len(records),3,1536) or not np.isfinite(features).all():
        raise ValueError("Speech feature cache is incomplete or invalid.")
    return features


def train_speech_emotion_pipeline(data_dir: str, models_dir: str) -> dict[str, Any]:
    data_path = Path(data_dir)
    models_path = Path(models_dir)
    models_path.mkdir(parents=True, exist_ok=True)

    records = load_multicorpus_records(data_path)
    create_benchmarks(data_path, records)
    flat, sequence = _cache_features(data_path, records)
    labels = np.asarray([record.label for record in records], dtype=np.int64)
    sources = np.asarray([record.source for record in records])
    split_indices = split_records(records)
    train_indices = np.asarray(split_indices["train"], dtype=np.int64)
    validation_indices = np.asarray(split_indices["validation"], dtype=np.int64)
    test_indices = np.asarray(split_indices["test"], dtype=np.int64)
    if not len(validation_indices) or not len(test_indices):
        raise ValueError("The combined corpus needs non-empty speaker-disjoint validation and test splits.")
    expected_labels = set(range(len(COMMON_EMOTIONS)))
    for split_name, indices in split_indices.items():
        found_labels = set(labels[indices])
        if found_labels != expected_labels:
            missing = [COMMON_EMOTIONS[label] for label in sorted(expected_labels - found_labels)]
            raise ValueError(f"The {split_name} split is missing shared emotion classes: {', '.join(missing)}.")

    calibration_indices, validation_indices = _split_calibration(records, validation_indices)
    selection_weights = _source_balanced_weights(records, validation_indices)
    calibration_weights = _source_balanced_weights(records, calibration_indices)

    train_weights = _source_balanced_weights(records, train_indices)
    scaler = StandardScaler()
    scaler.fit(flat[train_indices], sample_weight=train_weights)
    flat_train = scaler.transform(flat[train_indices])
    flat_validation = scaler.transform(flat[validation_indices])
    flat_test = scaler.transform(flat[test_indices])
    sequence_train = sequence[train_indices]
    sequence_validation = sequence[validation_indices]
    sequence_test = sequence[test_indices]
    y_train = labels[train_indices]
    y_validation = labels[validation_indices]
    y_test = labels[test_indices]

    models: dict[str, Any] = {
        "mlp": MLPClassifier(
            hidden_layer_sizes=(256, 128, 64),
            activation="relu",
            solver="adam",
            alpha=0.001,
            learning_rate_init=0.001,
            max_iter=300,
            early_stopping=True,
            n_iter_no_change=20,
            random_state=42,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=150,
            random_state=42,
            n_jobs=-1,
        ),
        "svm": SVC(
            kernel="rbf",
            C=3.0,
            probability=True,
            random_state=42,
        ),
        "hist_gradient_boosting": HistGradientBoostingClassifier(
            learning_rate=0.05,
            max_iter=200,
            max_leaf_nodes=15,
            max_depth=4,
            l2_regularization=1.0,
            early_stopping=True,
            random_state=42,
        ),
        "cnn": TorchSequenceModel("cnn", num_classes=len(COMMON_EMOTIONS)),
        "lstm": TorchSequenceModel("lstm", num_classes=len(COMMON_EMOTIONS)),
        "crnn": TorchSequenceModel("crnn", num_classes=len(COMMON_EMOTIONS)),
    }

    scaled_flat_models = {"mlp", "random_forest", "svm"}
    flat_models = scaled_flat_models | {"hist_gradient_boosting"}
    validation_scores: dict[str, tuple[float, float]] = {}
    validation_probabilities: dict[str, np.ndarray] = {}
    for key, model in models.items():
        print(f"Training {key} on {len(train_indices)} training recordings...", flush=True)
        if key in flat_models:
            train_features = flat_train if key in scaled_flat_models else flat[train_indices]
            model.fit(train_features, y_train, sample_weight=train_weights)
            validation_features = flat_validation if key in scaled_flat_models else flat[validation_indices]
            probabilities = model.predict_proba(validation_features)
        else:
            model.fit(
                sequence_train,
                y_train,
                validation_data=(sequence_validation, y_validation),
                sample_weights=train_weights,
            )
            probabilities = model.predict_proba(sequence_validation)
        validation_probabilities[key] = probabilities
        predictions = probabilities.argmax(axis=1)
        validation_scores[key] = (
            float(accuracy_score(y_validation, predictions, sample_weight=selection_weights)),
            float(f1_score(y_validation, predictions, average="macro", sample_weight=selection_weights)),
        )

    hybrid_model = HybridEmotionModel(
        models["hist_gradient_boosting"],
        models["lstm"],
    )
    validation_probabilities["hybrid"] = hybrid_model.predict_proba(
        (flat[validation_indices], sequence_validation)
    )
    hybrid_predictions = validation_probabilities["hybrid"].argmax(axis=1)
    validation_scores["hybrid"] = (
        float(accuracy_score(y_validation, hybrid_predictions, sample_weight=selection_weights)),
        float(f1_score(y_validation, hybrid_predictions, average="macro", sample_weight=selection_weights)),
    )

    member_keys, member_weights, ensemble_score = select_ensemble(
        {key: validation_probabilities[key] for key in models}, y_validation,
        sample_weight=selection_weights, prefer_macro_f1=True,
    )
    ensemble_model = EmotionEnsemble(
        [(key, models[key]) for key in member_keys], member_weights, scaler
    )
    validation_scores["ensemble"] = (ensemble_score[1], ensemble_score[0])
    all_models = {**models, "hybrid": hybrid_model, "ensemble": ensemble_model}
    calibrated_candidates = {}
    for key, model in all_models.items():
        calibration_probs = _predict_model(key, model, flat[calibration_indices], sequence[calibration_indices], scaler.transform(flat[calibration_indices]))
        calibrated = CalibratedEmotionModel(model, key, scaler).fit_probabilities(
            calibration_probs, labels[calibration_indices], calibration_weights)
        predicted = calibrated.predict_proba((flat[validation_indices], sequence_validation)).argmax(axis=1)
        candidate_key = "calibrated_" + key
        calibrated_candidates[candidate_key] = calibrated
        validation_scores[candidate_key] = (
            float(accuracy_score(y_validation, predicted, sample_weight=selection_weights)),
            float(f1_score(y_validation, predicted, average="macro", sample_weight=selection_weights)),
        )
    all_models.update(calibrated_candidates)
    best_key = max(validation_scores, key=lambda key: validation_scores[key][::-1])
    display_names = {
        "crnn": "CRNN (CNN + Bi-LSTM Hybrid)",
        "cnn": "1D-CNN (Temporal Convolutional Network)",
        "lstm": "Bi-LSTM (Bidirectional Recurrent Network)",
        "mlp": "Deep MLP (Multi-Layer Perceptron)",
        "random_forest": "Random Forest",
        "svm": "Support Vector Classifier (SVM)",
        "hist_gradient_boosting": "Histogram Gradient Boosting",
        "hybrid": "Hybrid (Gradient Boosting + Bi-LSTM)",
        "ensemble": "Validation-selected Ensemble",
    }
    display_names.update({"calibrated_" + key: "Calibrated " + name for key, name in list(display_names.items())})
    metrics: dict[str, Any] = {
        "best_model": display_names[best_key],
        "ensemble": {"members": list(member_keys), "weights": list(member_weights)},
        "models": {},
        "dataset_summary": {
            "name": "RAVDESS + TESS + EMO-DB",
            "datasets": {
                "RAVDESS": {"source": "https://zenodo.org/records/1188976"},
                "TESS": {"source": "https://doi.org/10.5683/SP2/E8H2MF"},
                "EMO-DB": {"source": "http://emodb.bilderbar.info/"},
            },
            "total_samples": len(records),
            "train_samples": len(train_indices),
            "validation_samples": len(validation_indices) + len(calibration_indices),
            "calibration_samples": len(calibration_indices),
            "model_selection_samples": len(validation_indices),
            "test_samples": len(test_indices),
            "samples_by_dataset": {
                split_name: _describe_split(records, indices)
                for split_name, indices in split_indices.items()
            },
            "emotions_count": len(COMMON_EMOTIONS),
            "emotions_list": list(COMMON_EMOTIONS),
            "sample_rate": 22050,
            "feature_dimension": flat.shape[1],
            "temporal_sequence_shape": list(sequence.shape[1:]),
            "temporal_sequence_method": "MFCC frames interpolated across the full utterance",
            "split_method": (
                "RAVDESS actors 01-16/17-20/21-24; TESS OAF train and disjoint YAF clips "
                "for validation/test (same held-out speaker because TESS has only two); "
                "EMO-DB speakers split 60/20/20"
            ),
            "model_selection_metric": "macro F1 balanced by dataset and class; balanced accuracy breaks ties. Separate calibration and selection clips.",
            "training_weighting": "equal weight per dataset and shared emotion class",
        },
    }

    test_probabilities: dict[str, np.ndarray] = {}
    for key, model in all_models.items():
        probabilities = _predict_model(
            key,
            model,
            flat[test_indices],
            sequence_test,
            flat_test,
        )
        test_probabilities[key] = probabilities
        predictions = probabilities.argmax(axis=1)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test, predictions, labels=np.arange(len(COMMON_EMOTIONS)), average="macro", zero_division=0
        )

        _predict_model(
            key,
            model,
            flat[test_indices[:1]],
            sequence_test[:1],
            flat_test[:1],
        )
        started = time.perf_counter()
        for _ in range(10):
            _predict_model(
                key,
                model,
                flat[test_indices[:1]],
                sequence_test[:1],
                flat_test[:1],
            )
        latency_ms = (time.perf_counter() - started) * 100
        metrics["models"][display_names[key]] = {
            "key": key,
            "accuracy": round(float(accuracy_score(y_test, predictions)), 4),
            "precision": round(float(precision), 4),
            "recall": round(float(recall), 4),
            "f1_score": round(float(f1), 4),
            "roc_auc": round(
                float(
                    roc_auc_score(
                        y_test,
                        probabilities,
                        labels=np.arange(len(COMMON_EMOTIONS)),
                        multi_class="ovr",
                        average="macro",
                    )
                ),
                4,
            ),
            "inference_latency_ms": round(latency_ms, 2),
            "validation_accuracy": round(validation_scores[key][0], 4),
            "validation_macro_f1": round(validation_scores[key][1], 4),
        }

        per_dataset = {}
        for source in sorted(set(sources[test_indices])):
            source_mask = sources[test_indices] == source
            source_y = y_test[source_mask]
            source_predictions = predictions[source_mask]
            per_dataset[source] = {
                "samples": int(source_mask.sum()),
                "accuracy": round(float(accuracy_score(source_y, source_predictions)), 4),
                "macro_f1": round(
                    float(
                        f1_score(
                            source_y,
                            source_predictions,
                            labels=np.arange(len(COMMON_EMOTIONS)),
                            average="macro",
                            zero_division=0,
                        )
                    ),
                    4,
                ),
            }
        metrics["models"][display_names[key]]["test_by_dataset"] = per_dataset

    best_predictions = test_probabilities[best_key].argmax(axis=1)
    matrix = confusion_matrix(y_test, best_predictions, labels=np.arange(len(COMMON_EMOTIONS)))
    metrics["confusion_matrix"] = {
        emotion: {predicted: int(matrix[i, j]) for j, predicted in enumerate(COMMON_EMOTIONS)}
        for i, emotion in enumerate(COMMON_EMOTIONS)
    }
    per_emotion_precision, per_emotion_recall, per_emotion_f1, support = precision_recall_fscore_support(
        y_test, best_predictions,
        labels=np.arange(len(COMMON_EMOTIONS)), average=None, zero_division=0
    )
    metrics["per_emotion_metrics"] = {
        emotion: {
            "precision": round(float(per_emotion_precision[index]), 4),
            "recall": round(float(per_emotion_recall[index]), 4),
            "f1_score": round(float(per_emotion_f1[index]), 4),
            "support": int(support[index]),
        }
        for index, emotion in enumerate(COMMON_EMOTIONS)
    }
    importances = models["random_forest"].feature_importances_
    feature_names = _feature_names()
    top_indices = sorted(range(len(importances)), key=lambda index: (-importances[index], index))[:20]
    metrics["top_features"] = [
        {"feature": feature_names[index], "importance": round(float(importances[index]), 4)}
        for index in top_indices
    ]

    with tempfile.TemporaryDirectory(dir=models_path, prefix=".training-") as temporary_dir:
        staging = Path(temporary_dir)
        for key, model in models.items():
            artifact_name = "hist_gradient_boosting_model.joblib" if key == "hist_gradient_boosting" else f"{key}_model.joblib"
            joblib.dump(model, staging / artifact_name)
        joblib.dump(hybrid_model, staging / "hybrid_model.joblib")
        joblib.dump(ensemble_model, staging / "ensemble_model.joblib")
        for key, model in calibrated_candidates.items():
            joblib.dump(model, staging / f"{key}_model.joblib")
        joblib.dump(all_models[best_key], staging / "best_model.joblib")
        joblib.dump(scaler, staging / "scaler.joblib")
        metrics_path = staging / "metrics.json"
        metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        for artifact in staging.iterdir():
            os.replace(artifact, models_path / artifact.name)

    print(
        f"Best model by balanced validation macro F1 ({validation_scores[best_key][1]:.3f}): {metrics['best_model']}",
        flush=True,
    )
    print(f"Speaker-disjoint test results use {len(test_indices)} recordings.", flush=True)
    return metrics
