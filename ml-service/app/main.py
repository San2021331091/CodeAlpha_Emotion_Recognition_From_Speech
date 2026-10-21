import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from app.api.routes import router, predictor, DATA_DIR, MODELS_DIR
from app.ml.trainer import train_speech_emotion_pipeline


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Ensure dataset and models exist on startup
    metrics_path = os.path.join(MODELS_DIR, "metrics.json")
    if not os.path.exists(metrics_path) or len(predictor.models) == 0:
        print("Initial launch: No trained models found. Running initial training pipeline...")
        train_speech_emotion_pipeline(data_dir=DATA_DIR, models_dir=MODELS_DIR)
        predictor.load_artifacts()
        print("Initial training complete and models loaded!")
    yield


app = FastAPI(
    title="CodeAlpha Speech Emotion Recognition ML Microservice",
    description="Deep Learning API for classifying human emotional states from acoustic speech signals (RAVDESS)",
    version="1.0.0",
    lifespan=lifespan
)

# Enable CORS for Next.js frontend and NestJS gateway
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)


@app.get("/")
def root():
    return {
        "service": "CodeAlpha Speech Emotion Recognition ML Service",
        "version": "1.0.0",
        "docs_url": "/docs",
        "health_url": "/api/health"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=True)
