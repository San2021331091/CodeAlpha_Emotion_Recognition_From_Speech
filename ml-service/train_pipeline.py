import os
import sys
import argparse

# Ensure app package is in sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from app.ml.trainer import train_speech_emotion_pipeline

if __name__ == "__main__":
    print("=" * 65)
    print("CodeAlpha Task 2: Speech Emotion Recognition Model Training")
    print("=" * 65)

    parser = argparse.ArgumentParser(description="Train on local RAVDESS, TESS, and EMO-DB recordings.")
    parser.add_argument("--data-dir", default=os.path.join(current_dir, "data"))
    parser.add_argument("--models-dir", default=os.path.join(current_dir, "saved_models"))
    args = parser.parse_args()
    data_directory = args.data_dir
    saved_models_directory = args.models_dir

    # Official RAVDESS: 1,440 human speech recordings from 24 actors.
    metrics = train_speech_emotion_pipeline(
        data_dir=data_directory,
        models_dir=saved_models_directory
    )

    print("\nTraining Metrics Summary:")
    print(f"  Best Model: {metrics['best_model']}")
    for model_name, score in metrics["models"].items():
        print(f"  - {model_name:42s} | Acc: {score['accuracy']*100:.1f}% | F1: {score['f1_score']*100:.1f}% | AUC: {score['roc_auc']:.3f} | Latency: {score['inference_latency_ms']}ms")
    print("=" * 65)
