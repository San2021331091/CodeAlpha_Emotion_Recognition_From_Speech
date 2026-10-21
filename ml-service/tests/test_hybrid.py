import unittest
from unittest.mock import patch

import numpy as np

from app.ml.hybrid_model import HybridEmotionModel
from app.ml.predictor import EmotionPredictor


class FixedProbabilityModel:
    def __init__(self, probabilities):
        self.probabilities = np.asarray(probabilities, dtype=np.float32)

    def predict_proba(self, features):
        batch = features[0] if isinstance(features, tuple) else features
        return np.repeat(self.probabilities[None, :], len(batch), axis=0)


class RecordingProbabilityModel(FixedProbabilityModel):
    def predict_proba(self, features):
        self.features = features
        return super().predict_proba(features)


class HybridModelTests(unittest.TestCase):
    def test_audio_model_receives_original_audio_instead_of_mfcc_features(self):
        class RawAudioModel:
            audio_model = True
            def predict_audio_proba(self, audio):
                self.received = audio
                return np.array([[0, 1, 0, 0, 0, 0]])
        model = RawAudioModel()
        predictor = EmotionPredictor.__new__(EmotionPredictor)
        predictor.models = {"best": model}
        predictor.best_model_name = "WavLM Speech Classifier"
        predictor.class_emotions = ["neutral", "happy", "sad", "angry", "fearful", "disgust"]
        predictor.scaler = None
        with patch("app.ml.predictor.extract_features_from_audio", return_value=(np.zeros(231), np.zeros((40,128)), {})):
            result = predictor.predict_audio(b"original-audio")
        self.assertEqual(model.received, b"original-audio")
        self.assertEqual(result["top_emotion"], "happy")
        self.assertIn("WavLM", result["explanation"])

    def test_predictor_reloads_replaced_weights_without_a_service_restart(self):
        import json
        import os
        from pathlib import Path
        import tempfile
        import joblib
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            path = directory / "best_model.joblib"
            joblib.dump(FixedProbabilityModel([1, 0, 0, 0, 0, 0]), path)
            (directory / "metrics.json").write_text(json.dumps({"best_model": "model", "dataset_summary": {
                "emotions_list": ["neutral", "happy", "sad", "angry", "fearful", "disgust"]}}))
            predictor = EmotionPredictor(temporary)
            with patch("app.ml.predictor.extract_features_from_audio", return_value=(np.zeros(231), np.zeros((40, 128)), {})):
                self.assertEqual(predictor.predict_audio(b"audio")["top_emotion"], "neutral")
                previous_revision = predictor.artifact_revision
                old_time = path.stat().st_mtime_ns
                joblib.dump(FixedProbabilityModel([0, 1, 0, 0, 0, 0]), path)
                os.utime(path, ns=(old_time + 2000000000, old_time + 2000000000))
                self.assertEqual(predictor.predict_audio(b"audio")["top_emotion"], "happy")
                self.assertNotEqual(predictor.artifact_revision, previous_revision)

    def test_balanced_selection_uses_the_equal_class_candidate(self):
        predictor = EmotionPredictor.__new__(EmotionPredictor)
        balanced = object()
        predictor.models = {"balanced_audio": balanced, "cnn": object(), "best": object()}
        predictor.best_model_name = "Validation-selected Ensemble"
        model, name = predictor._resolve_model("Balanced CNN (8,333 samples per emotion)")
        self.assertIs(model, balanced)
        self.assertEqual(name, "Balanced CNN (8,333 samples per emotion)")

    def test_blends_acoustic_and_sequence_probabilities(self):
        acoustic = FixedProbabilityModel([0.8, 0.2])
        sequence = FixedProbabilityModel([0.4, 0.6])
        model = HybridEmotionModel(acoustic, sequence)

        actual = model.predict_proba((np.zeros((1, 3)), np.zeros((1, 2, 4))))

        np.testing.assert_allclose(actual, [[0.6, 0.4]])

    def test_predictor_routes_best_model_inputs_to_both_branches(self):
        acoustic = FixedProbabilityModel([0.1, 0.9, 0, 0, 0, 0, 0, 0])
        sequence = FixedProbabilityModel([0.9, 0.1, 0, 0, 0, 0, 0, 0])
        hybrid = HybridEmotionModel(acoustic, sequence)
        predictor = EmotionPredictor.__new__(EmotionPredictor)
        predictor.models = {"hybrid": hybrid}
        predictor.scaler = None
        predictor.best_model_name = "Hybrid (Gradient Boosting + Bi-LSTM)"
        predictor.class_emotions = list(
            ("neutral", "calm", "happy", "sad", "angry", "fearful", "disgust", "surprised")
        )

        with patch(
            "app.ml.predictor.extract_features_from_audio",
            return_value=(np.zeros(231), np.zeros((40, 128)), {}),
        ):
            result = predictor.predict_audio(b"audio")

        self.assertEqual(result["top_emotion"], "neutral")
        self.assertEqual(result["model_used"], predictor.best_model_name)

    def test_predictor_maps_shared_six_class_output_to_api_emotions(self):
        model = FixedProbabilityModel([0, 0.9, 0, 0.1, 0, 0])
        predictor = EmotionPredictor.__new__(EmotionPredictor)
        predictor.models = {"best": model}
        predictor.scaler = None
        predictor.best_model_name = "six-class model"
        predictor.class_emotions = ["neutral", "happy", "sad", "angry", "fearful", "disgust"]

        with patch(
            "app.ml.predictor.extract_features_from_audio",
            return_value=(np.zeros(231), np.zeros((40, 128)), {}),
        ):
            result = predictor.predict_audio(b"audio")

        self.assertEqual(result["top_emotion"], "happy")
        self.assertEqual(result["probabilities"]["happy"], 0.9)
        self.assertEqual(set(result["probabilities"]), set(predictor.class_emotions))
        self.assertNotIn("calm", result["probabilities"])
        self.assertNotIn("surprised", result["probabilities"])

    def test_explicit_crnn_selection_is_not_confused_with_hybrid_name(self):
        predictor = EmotionPredictor.__new__(EmotionPredictor)
        crnn = object()
        predictor.models = {"crnn": crnn, "hybrid": object(), "best": object()}
        predictor.best_model_name = "Hybrid (Gradient Boosting + Bi-LSTM)"

        model, name = predictor._resolve_model("CRNN (CNN + Bi-LSTM Hybrid)")

        self.assertIs(model, crnn)
        self.assertEqual(name, "CRNN (CNN + Bi-LSTM Hybrid)")

    def test_histogram_boosting_receives_unscaled_acoustic_features(self):
        model = RecordingProbabilityModel([1, 0, 0, 0, 0, 0, 0, 0])
        predictor = EmotionPredictor.__new__(EmotionPredictor)
        predictor.models = {"hist_gradient_boosting": model, "best": model}
        predictor.scaler = type("Scaler", (), {"transform": lambda _, values: values * 2})()
        predictor.best_model_name = "Histogram Gradient Boosting"
        predictor.class_emotions = list(
            ("neutral", "calm", "happy", "sad", "angry", "fearful", "disgust", "surprised")
        )
        flat = np.ones(231)

        with patch(
            "app.ml.predictor.extract_features_from_audio",
            return_value=(flat, np.zeros((40, 128)), {}),
        ):
            predictor.predict_audio(b"audio", model_name="hist_gradient_boosting")

        np.testing.assert_array_equal(model.features, flat.reshape(1, -1))


if __name__ == "__main__":
    unittest.main()
