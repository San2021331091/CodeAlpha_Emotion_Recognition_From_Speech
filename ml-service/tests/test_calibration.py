import unittest

import numpy as np

from app.ml.calibrated_model import CalibratedEmotionModel
from app.ml.datasets import AudioRecord, COMMON_EMOTIONS
from app.ml.trainer import _split_calibration
try:
    from tests.test_hybrid import FixedProbabilityModel
except ImportError:
    from test_hybrid import FixedProbabilityModel


class CalibrationTests(unittest.TestCase):
    def test_calibration_and_selection_are_disjoint_and_stratified(self):
        records = [AudioRecord(f"{source}/{label}/{clip}.wav", label, source, "heldout")
                   for source in ("RAVDESS", "TESS", "EMO-DB")
                   for label in range(len(COMMON_EMOTIONS)) for clip in range(5)]
        calibration, selection = _split_calibration(records, np.arange(len(records)))
        self.assertFalse(set(calibration) & set(selection))
        self.assertEqual(set(calibration) | set(selection), set(range(len(records))))
        for indices in (calibration, selection):
            self.assertEqual({(records[i].source, records[i].label) for i in indices},
                             {(source, label) for source in ("RAVDESS", "TESS", "EMO-DB") for label in range(6)})

    def test_bias_corrects_systematic_underprediction_without_relabeling_classes(self):
        # Both classes initially favor class 0; their scores still contain a
        # discriminating signal which held-out class-bias fitting can recover.
        probabilities = np.array([[0.95, 0.05], [0.55, 0.45]] * 20)
        labels = np.tile([0, 1], 20)
        model = CalibratedEmotionModel(FixedProbabilityModel([0.55, 0.45]), "hist_gradient_boosting", None)
        model.fit_probabilities(probabilities, labels)
        result = model.predict_proba((np.zeros((1, 231)), np.zeros((1, 40, 128))))
        self.assertEqual(result.argmax(1)[0], 1)
        self.assertAlmostEqual(float(result.sum()), 1)
        self.assertTrue(np.isfinite(result).all())
