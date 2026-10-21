"""Class-bias correction learned from held-out calibration recordings."""
import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp, softmax


class CalibratedEmotionModel:
    multimodal_model = True

    def __init__(self, model, key, scaler):
        self.model = model
        self.key = key
        self.scaler = scaler
        self.bias = None

    def _base_probabilities(self, features):
        flat, sequence = features
        if getattr(self.model, "multimodal_model", False):
            return self.model.predict_proba(features)
        if getattr(self.model, "sequence_model", False):
            return self.model.predict_proba(sequence)
        if self.key in {"svm", "mlp", "random_forest"}:
            flat = self.scaler.transform(flat)
        return self.model.predict_proba(flat)

    def fit_probabilities(self, probabilities, labels, sample_weight=None):
        logits = np.log(np.clip(probabilities, 1e-7, 1))
        labels = np.asarray(labels, dtype=np.int64)
        weights = np.ones(len(labels)) if sample_weight is None else np.asarray(sample_weight)
        weights = weights / weights.sum()
        target = np.eye(logits.shape[1])[labels]

        def objective(bias):
            adjusted = logits + bias
            loss = np.sum(weights * (logsumexp(adjusted, axis=1) - adjusted[np.arange(len(labels)), labels]))
            # Regularization keeps corrections conservative on a small corpus.
            loss += 0.01 * np.sum(bias ** 2)
            gradient = np.sum(weights[:, None] * (softmax(adjusted, axis=1) - target), axis=0) + 0.02 * bias
            return loss, gradient

        result = minimize(objective, np.zeros(logits.shape[1]), jac=True,
                          method="L-BFGS-B", bounds=[(-3, 3)] * logits.shape[1])
        if not result.success:
            raise RuntimeError(f"Class-bias fitting failed: {result.message}")
        self.bias = result.x
        return self

    def predict_proba(self, features):
        probabilities = self._base_probabilities(features)
        return softmax(np.log(np.clip(probabilities, 1e-7, 1)) + self.bias, axis=1)
