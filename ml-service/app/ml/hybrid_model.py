import numpy as np


class HybridEmotionModel:
    multimodal_model = True

    def __init__(self, acoustic_model, sequence_model, acoustic_weight=0.5):
        self.acoustic_model = acoustic_model
        self.sequence_model = sequence_model
        self.acoustic_weight = acoustic_weight

    def predict_proba(self, features):
        acoustic_features, sequence_features = features
        acoustic_probabilities = self.acoustic_model.predict_proba(acoustic_features)
        sequence_probabilities = self.sequence_model.predict_proba(sequence_features)
        return (
            self.acoustic_weight * acoustic_probabilities
            + (1.0 - self.acoustic_weight) * sequence_probabilities
        )
