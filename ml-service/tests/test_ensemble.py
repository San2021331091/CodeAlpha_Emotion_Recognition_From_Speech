import unittest

import numpy as np

from app.ml.ensemble_model import EmotionEnsemble, select_ensemble
try:
    from tests.test_hybrid import RecordingProbabilityModel
except ImportError:
    from test_hybrid import RecordingProbabilityModel


class EnsembleTests(unittest.TestCase):
    def test_members_receive_their_training_preprocessing(self):
        acoustic = RecordingProbabilityModel([0.8, 0.2])
        sequence = RecordingProbabilityModel([0.2, 0.8])
        scaler = type("Scaler", (), {"transform": lambda _, x: x * 2})()
        model = EmotionEnsemble([("svm", acoustic), ("lstm", sequence)], [0.25, 0.75], scaler)
        flat, frames = np.ones((1, 231)), np.ones((1, 40, 128))
        np.testing.assert_allclose(model.predict_proba((flat, frames)), [[0.35, 0.65]])
        np.testing.assert_array_equal(acoustic.features, flat * 2)
        np.testing.assert_array_equal(sequence.features, frames)

    def test_selection_can_improve_over_both_individual_models(self):
        probabilities = {"a": np.array([[0.9, 0.1], [0.6, 0.4]]),
                         "b": np.array([[0.4, 0.6], [0.1, 0.9]])}
        members, weights, score = select_ensemble(probabilities, np.array([0, 1]))
        self.assertEqual(members, ("a", "b"))
        self.assertAlmostEqual(sum(weights), 1)
        self.assertEqual(score, (1.0, 1.0))
