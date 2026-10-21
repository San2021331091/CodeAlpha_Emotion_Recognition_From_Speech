from enum import Enum
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field


class EmotionType(str, Enum):
    NEUTRAL = "neutral"
    CALM = "calm"
    HAPPY = "happy"
    SAD = "sad"
    ANGRY = "angry"
    FEARFUL = "fearful"
    DISGUST = "disgust"
    SURPRISED = "surprised"


class ValenceArousal(BaseModel):
    valence: float = Field(..., description="Valence value between -1.0 (negative) and +1.0 (positive)")
    arousal: float = Field(..., description="Arousal value between -1.0 (low energy) and +1.0 (high energy)")
    quadrant: str = Field(..., description="Russell Circumplex Quadrant name")


class AcousticSummary(BaseModel):
    duration_seconds: float
    sample_rate: int
    rms_energy_mean: float
    zero_crossing_rate_mean: float
    spectral_centroid_mean: float
    spectral_rolloff_mean: float
    pitch_f0_mean: float
    pitch_f0_std: float
    tempo: float
    mfcc_means: List[float] = Field(default_factory=list)
    chroma_means: List[float] = Field(default_factory=list)


class EmotionPredictionResponse(BaseModel):
    request_id: Optional[str] = None
    input_sha256: Optional[str] = None
    artifact_revision: Optional[str] = None
    top_emotion: str
    confidence: float
    probabilities: Dict[str, float]
    valence_arousal: ValenceArousal
    acoustic_summary: AcousticSummary
    model_used: str
    inference_time_ms: float
    explanation: str


class FeatureVectorPredictRequest(BaseModel):
    features: List[float]
    model_name: Optional[str] = "CRNN (CNN + Bi-LSTM)"


class ModelBenchmarkMetrics(BaseModel):
    key: Optional[str] = None
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    roc_auc: float
    inference_latency_ms: float
    validation_accuracy: Optional[float] = None
    validation_macro_f1: Optional[float] = None
    test_by_dataset: Dict[str, Dict[str, Any]] = Field(default_factory=dict)


class MetricsResponse(BaseModel):
    best_model: str
    models: Dict[str, ModelBenchmarkMetrics]
    dataset_summary: Dict[str, Any]
    confusion_matrix: Dict[str, Dict[str, int]]
    per_emotion_metrics: Dict[str, Dict[str, float]]
    top_features: List[Dict[str, Any]]


class AudioSampleItem(BaseModel):
    id: str
    title: str
    emotion: str
    speaker: str
    duration_seconds: float
    description: str
    filename: str
    dataset: Optional[str] = "RAVDESS"
    language: Optional[str] = "English"
