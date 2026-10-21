import os
import json
import hashlib
import logging
import uuid
from typing import Optional, List
from fastapi import APIRouter, File, UploadFile, Query, HTTPException, Form
from fastapi.responses import FileResponse

from app.models.schemas import (
    EmotionPredictionResponse,
    MetricsResponse,
    AudioSampleItem,
    
)
from app.ml.predictor import EmotionPredictor
from app.ml.feature_extractor import InvalidAudioError
from app.ml.training_jobs import start_job, get_status


router = APIRouter(prefix="/api", tags=["Speech Emotion Recognition"])

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
MODELS_DIR = os.path.join(PROJECT_ROOT, "saved_models")

# Global predictor instance
predictor = EmotionPredictor(models_dir=MODELS_DIR)

@router.get("/health")
def health_check():
    predictor.ensure_current_artifacts()
    return {
        "status": "healthy",
        "service": "Speech Emotion Recognition ML Engine",
        "models_loaded": list(predictor.models.keys()),
        "best_model": predictor.best_model_name,
        "artifact_revision": predictor.artifact_revision,
        "prediction_protocol": 2,
    }


@router.post("/predict/audio", response_model=EmotionPredictionResponse)
async def predict_audio_file(
    file: UploadFile = File(..., description="Audio file in WAV, MP3, OGG, or WebM format"),
    model_name: Optional[str] = Form(None, description="Model to use for inference (CRNN, CNN, LSTM, MLP, RF, SVM)"),
    request_id: Optional[str] = Form(None, max_length=64),
):
    try:
        audio_bytes = await file.read(25 * 1024 * 1024 + 1)
        if len(audio_bytes) > 25 * 1024 * 1024:
            raise HTTPException(status_code=413, detail="Audio files must be 25 MB or smaller.")
        if not audio_bytes:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")

        prediction = predictor.predict_audio(audio_bytes, model_name=model_name)
        # Unit/direct callers may omit FastAPI's optional Form dependency.
        prediction["request_id"] = request_id if isinstance(request_id, str) else None
        prediction["input_sha256"] = hashlib.sha256(audio_bytes).hexdigest()
        prediction["artifact_revision"] = predictor.artifact_revision
        diagnostic = {"request_id": prediction["request_id"], "input_sha256": prediction["input_sha256"],
            "filename": file.filename, "input_bytes": len(audio_bytes), "model_requested": model_name,
            "model_used": prediction["model_used"], "artifact_revision": predictor.artifact_revision,
            "top_emotion": prediction["top_emotion"],
            "probabilities": prediction["probabilities"], "acoustic_summary": prediction["acoustic_summary"]}
        # Keep metadata for the latest request only; do not store uploaded audio.
        temporary = os.path.join(PROJECT_ROOT, f".inference-{uuid.uuid4().hex}.tmp")
        try:
            with open(temporary, "w", encoding="utf-8") as handle:
                json.dump(diagnostic, handle, indent=2)
            os.replace(temporary, os.path.join(PROJECT_ROOT, "last_inference_diagnostic.json"))
        except OSError:
            logging.getLogger(__name__).warning("Could not write the latest inference diagnostic.")
            if os.path.exists(temporary):
                try:
                    os.remove(temporary)
                except OSError:
                    pass
        return prediction
    except HTTPException:
        raise
    except InvalidAudioError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")


@router.get("/models/metrics", response_model=MetricsResponse)
def get_model_metrics():
    metrics_path = os.path.join(MODELS_DIR, "metrics.json")
    if not os.path.exists(metrics_path):
        raise HTTPException(status_code=404, detail="Model metrics not found. Train models first.")
    
    with open(metrics_path, "r") as f:
        metrics = json.load(f)
    # Keep completed expanded-run counts separate from deployed-model metrics.
    for directory in ("saved_models_ravdess_weighted", "saved_models_50000"):
        candidate_path = os.path.join(PROJECT_ROOT, directory, "metrics.json")
        if not os.path.isfile(candidate_path):
            continue
        try:
            with open(candidate_path, encoding="utf-8") as handle:
                candidate = json.load(handle)
            summary = candidate["dataset_summary"]
            run = candidate["expanded_training_run"]
            metrics["dataset_summary"]["expanded_training_candidate"] = {
                "model": candidate["best_model"],
                "total": summary["effective_training_samples"],
                "original": summary["train_samples"],
                "augmented": summary["augmented_training_samples"],
                "samples_by_dataset": summary["training_augmentation"]["samples_by_source"],
                "ravdess_training_weight": run.get("ravdess_training_weight", 1.0),
                "validation_macro_f1": run["candidate_selection_macro_f1"],
                "deployed": candidate["best_model"] == metrics["best_model"],
            }
            break
        except (OSError, ValueError, KeyError):
            logging.getLogger(__name__).warning("Could not read expanded-run metrics from %s", directory)
    return metrics


@router.get("/audio/samples", response_model=List[AudioSampleItem])
def list_benchmark_samples():
    manifest_path = os.path.join(DATA_DIR, "benchmark_manifest.json")
    if not os.path.exists(manifest_path):
        return []
    with open(manifest_path, "r") as f:
        samples = json.load(f)
    return samples


@router.get("/audio/sample/{sample_id}")
def stream_sample_audio(sample_id: str):
    benchmark_dir = os.path.join(DATA_DIR, "audio_samples", "benchmark")
    if not os.path.exists(benchmark_dir):
        raise HTTPException(status_code=404, detail="Benchmark directory not found.")

    manifest_path = os.path.join(DATA_DIR, "benchmark_manifest.json")
    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)
    sample = next((item for item in manifest if item["id"] == sample_id), None)
    if sample is None:
        raise HTTPException(status_code=404, detail="Audio sample not found.")
    matched_file = os.path.join(benchmark_dir, os.path.basename(sample["filename"]))
    if not os.path.isfile(matched_file):
        raise HTTPException(status_code=404, detail="Audio sample file not found.")

    return FileResponse(matched_file, media_type="audio/wav", filename=os.path.basename(matched_file))


@router.get("/train/status")
def training_status():
    return get_status()


@router.post("/train", status_code=202)
def retrain_models():
    try:
        return start_job()
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")
