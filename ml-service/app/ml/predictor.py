import os
import time
import json
import hashlib
import threading
import joblib
import numpy as np
from typing import Dict, Any, Optional, Union, Tuple

from .feature_extractor import extract_features_from_audio
from .data_generator import EMOTIONS, EMOTION_COORDINATES


class EmotionPredictor:
    """
    Inference and Diagnostic Engine for Speech Emotion Recognition.
    Manages loaded models, feature transformations, valence-arousal mapping,
    and acoustic explainability.
    """
    def __init__(self, models_dir: str):
        self.models_dir = models_dir
        self.models: Dict[str, Any] = {}
        self.scaler = None
        self.metrics = None
        self.best_model_name = "CRNN (CNN + Bi-LSTM Hybrid)"
        self.class_emotions = list(EMOTIONS)
        self._reload_lock = threading.RLock()
        self.load_artifacts()

    def _artifact_signature(self):
        files = []
        if os.path.isdir(self.models_dir):
            for name in sorted(os.listdir(self.models_dir)):
                if name.endswith(".joblib") or name == "metrics.json":
                    stat = os.stat(os.path.join(self.models_dir, name))
                    files.append((name, stat.st_mtime_ns, stat.st_size))
        return hashlib.sha256(repr(files).encode()).hexdigest()[:16]

    def ensure_current_artifacts(self):
        """Model replacement does not trigger Uvicorn's Python-file watcher."""
        if not hasattr(self, "models_dir"):
            return
        with self._reload_lock:
            if self._artifact_signature() != getattr(self, "artifact_revision", None):
                self.load_artifacts()

    def load_artifacts(self):
        """Loads serialized models, scaler, and metrics from disk."""
        revision = self._artifact_signature()
        self.models = {}
        self.scaler = None
        self.metrics = None
        self.class_emotions = list(EMOTIONS)
        scaler_path = os.path.join(self.models_dir, "scaler.joblib")
        if os.path.exists(scaler_path):
            self.scaler = joblib.load(scaler_path)

        metrics_path = os.path.join(self.models_dir, "metrics.json")
        if os.path.exists(metrics_path):
            with open(metrics_path, "r") as f:
                self.metrics = json.load(f)
                if "best_model" in self.metrics:
                    self.best_model_name = self.metrics["best_model"]
                self.class_emotions = self.metrics.get("dataset_summary", {}).get(
                    "emotions_list", list(EMOTIONS)
                )

        # Load available models
        model_files = {
            "crnn": "crnn_model.joblib",
            "cnn": "cnn_model.joblib",
            "lstm": "lstm_model.joblib",
            "mlp": "mlp_model.joblib",
            "random_forest": "random_forest_model.joblib",
            "svm": "svm_model.joblib",
            "hist_gradient_boosting": "hist_gradient_boosting_model.joblib",
            "hybrid": "hybrid_model.joblib",
            "ensemble": "ensemble_model.joblib",
            "balanced_audio": "balanced_audio_model.joblib",
            "speech_encoder": "speech_encoder_model.joblib",
            "ravdess_specialist": "ravdess_specialist_model.joblib",
            "best": "best_model.joblib"
        }

        for key, fname in model_files.items():
            path = os.path.join(self.models_dir, fname)
            if os.path.exists(path):
                try:
                    self.models[key] = joblib.load(path)
                except Exception as e:
                    print(f"Warning: Failed to load model {key}: {e}")
        self.artifact_revision = revision

    def _resolve_model(self, model_choice: Optional[str] = None):
        """Resolves model object and display name from user selection."""
        if not self.models:
            raise RuntimeError("No trained models found in saved_models directory. Please run training pipeline first.")

        if not model_choice or model_choice.lower() in ["best", "auto", "default"]:
            return self.models.get("best") or list(self.models.values())[0], self.best_model_name

        choice_lower = model_choice.lower()
        if choice_lower in {"speech_encoder", "wavlm speech classifier"}:
            if "speech_encoder" not in self.models:
                raise RuntimeError("The WavLM speech classifier is not installed.")
            return self.models["speech_encoder"], "WavLM Speech Classifier"
        elif choice_lower in {"ravdess_specialist", "ravdess wavlm specialist"}:
            if "ravdess_specialist" not in self.models:
                raise RuntimeError("The RAVDESS WavLM specialist is not installed.")
            return self.models["ravdess_specialist"], "RAVDESS WavLM Specialist"
        elif choice_lower == "balanced_audio" or choice_lower.startswith("balanced cnn"):
            if "balanced_audio" not in self.models:
                raise RuntimeError("The balanced audio model is not installed. Train the equal-class corpus first.")
            return self.models["balanced_audio"], "Balanced CNN (8,333 samples per emotion)"
        elif "ensemble" in choice_lower:
            return self.models.get("ensemble") or self.models.get("best"), "Validation-selected Ensemble"
        elif "histogram gradient" in choice_lower or "hist_gradient_boosting" in choice_lower:
            return self.models.get("hist_gradient_boosting") or self.models.get("best"), "Histogram Gradient Boosting"
        elif "gradient boosting" in choice_lower or choice_lower == "hybrid":
            return self.models.get("hybrid") or self.models.get("best"), "Hybrid (Gradient Boosting + Bi-LSTM)"
        elif "crnn" in choice_lower:
            return self.models.get("crnn") or self.models.get("best"), "CRNN (CNN + Bi-LSTM Hybrid)"
        elif "cnn" in choice_lower:
            return self.models.get("cnn") or self.models.get("best"), "1D-CNN (Temporal Convolutional Network)"
        elif "lstm" in choice_lower:
            return self.models.get("lstm") or self.models.get("best"), "Bi-LSTM (Bidirectional Recurrent Network)"
        elif "mlp" in choice_lower or "neural" in choice_lower:
            return self.models.get("mlp") or self.models.get("best"), "Deep MLP (Multi-Layer Perceptron)"
        elif "forest" in choice_lower or "rf" in choice_lower:
            return self.models.get("random_forest") or self.models.get("best"), "Random Forest Classifier"
        elif "svm" in choice_lower:
            return self.models.get("svm") or self.models.get("best"), "Support Vector Classifier (SVM)"

        return self.models.get("best") or list(self.models.values())[0], self.best_model_name

    def _generate_explanation(self, emotion: str, confidence: float, summary: Dict[str, Any]) -> str:
        """Generates natural language acoustic diagnostic rationale."""
        return (
            f"Predicted {emotion.title()} with model probability {confidence * 100:.1f}%. "
            "The model compares MFCC and acoustic patterns learned from acted RAVDESS, TESS, "
            "and EMO-DB recordings. This probability is not a calibrated measure of certainty; "
            "natural speech, accents, languages, and recording conditions can differ from the "
            "training data."
        )

    def predict_audio(
        self,
        audio_source: Union[str, bytes],
        model_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Processes audio input, extracts features, performs model inference,
        and returns complete prediction payload.
        """
        start_time = time.time()
        self.ensure_current_artifacts()
        flat_feat, seq_feat, summary = extract_features_from_audio(audio_source)

        model, resolved_name = self._resolve_model(model_name)

        # Check model type to provide correct input tensor
        if getattr(model, "audio_model", False):
            probs = model.predict_audio_proba(audio_source)[0]
        elif getattr(model, "multimodal_model", False):
            probs = model.predict_proba((flat_feat.reshape(1, -1), seq_feat))[0]
        elif "histogram gradient boosting" in resolved_name.lower():
            probs = model.predict_proba(flat_feat.reshape(1, -1))[0]
        elif getattr(model, "sequence_model", False) or hasattr(model, "cnn_block1") or hasattr(model, "block1") or hasattr(model, "W_fwd"):
            # Deep neural network sequence model
            probs = model.predict_proba(seq_feat)[0]
        else:
            # Tabular / MLP / Ensemble model
            if self.scaler is not None:
                scaled_feat = self.scaler.transform(flat_feat.reshape(1, -1))
            else:
                scaled_feat = flat_feat.reshape(1, -1)
            probs = model.predict_proba(scaled_feat)[0]

        if probs.ndim != 1 or len(probs) != len(self.class_emotions):
            raise RuntimeError(
                f"Model returned {len(probs)} class probabilities, but metrics define "
                f"{len(self.class_emotions)} emotion classes."
            )
        pred_idx = int(np.argmax(probs))
        top_emotion = self.class_emotions[pred_idx]
        top_prob = float(probs[pred_idx])

        # Emotion probability distribution map
        prob_dict = {
            emotion: round(float(probs[index]), 4)
            for index, emotion in enumerate(self.class_emotions)
        }

        # Valence-Arousal coordinates
        coord = EMOTION_COORDINATES.get(top_emotion, {"valence": 0.0, "arousal": 0.0, "quadrant": "Neutral"})

        inference_time_ms = round((time.time() - start_time) * 1000.0, 2)
        explanation = self._generate_explanation(top_emotion, top_prob, summary)
        if getattr(model, "audio_model", False):
            explanation = (
                f"Predicted {top_emotion.title()} with model probability {top_prob * 100:.1f}%. "
                "This classifier uses frozen WavLM speech representations and an emotion head trained on local corpus recordings. "
                "The probability is not a guarantee of correctness; music, singing, and unfamiliar speech can be misclassified."
            )

        return {
            "top_emotion": top_emotion,
            "confidence": round(top_prob, 4),
            "probabilities": prob_dict,
            "valence_arousal": coord,
            "acoustic_summary": summary,
            "model_used": resolved_name,
            "inference_time_ms": inference_time_ms,
            "explanation": explanation
        }
