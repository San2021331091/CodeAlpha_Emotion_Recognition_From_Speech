"""Validation-selected voting with each member's training-time preprocessing."""
import numpy as np
from sklearn.metrics import accuracy_score, f1_score


class EmotionEnsemble:
    multimodal_model = True

    def __init__(self, members, weights, scaler):
        self.members = members
        self.weights = np.asarray(weights, dtype=np.float64)
        self.scaler = scaler

    def predict_proba(self, features):
        flat, sequence = features
        result = None
        for (key, model), weight in zip(self.members, self.weights):
            if getattr(model, "multimodal_model", False):
                probabilities = model.predict_proba(features)
                result = weight * probabilities if result is None else result + weight * probabilities
                continue
            if getattr(model, "sequence_model", False) or key in {"cnn", "lstm", "crnn"}:
                values = sequence
            elif key in {"mlp", "random_forest", "svm"}:
                values = self.scaler.transform(flat)
            else:
                values = flat
            probabilities = model.predict_proba(values)
            result = weight * probabilities if result is None else result + weight * probabilities
        return result


def select_ensemble(probabilities, labels, sample_weight=None, prefer_macro_f1=False):
    """Select two members and their weight using validation data exclusively."""
    keys = list(probabilities)
    best_score, best_members, best_weights = (-1, -1), None, None
    for index, left in enumerate(keys):
        for right in keys[index:]:
            for weight in np.linspace(0, 1, 21):
                blended = weight * probabilities[left] + (1 - weight) * probabilities[right]
                predicted = blended.argmax(axis=1)
                accuracy = float(accuracy_score(labels, predicted, sample_weight=sample_weight))
                f1 = float(f1_score(labels, predicted, average="macro", zero_division=0, sample_weight=sample_weight))
                score = (f1, accuracy) if prefer_macro_f1 else (accuracy, f1)
                if score > best_score:
                    best_score = score
                    best_members = (left, right)
                    best_weights = (float(weight), float(1 - weight))
    return best_members, best_weights, best_score
