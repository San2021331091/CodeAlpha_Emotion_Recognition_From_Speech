"""Compare the same speech across upload conversion and recording conditions."""
import io
import json
from collections import Counter
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

from app.ml.predictor import EmotionPredictor


def encode_wav(waveform, sample_rate):
    output = io.BytesIO()
    sf.write(output, waveform, sample_rate, format="WAV", subtype="PCM_16")
    return output.getvalue()


def main():
    predictor = EmotionPredictor("saved_models")
    samples = json.loads(Path("data/benchmark_manifest.json").read_text())
    report = []
    for sample in samples:
        path = Path("data/audio_samples/benchmark") / sample["filename"]
        waveform, sr = sf.read(path, dtype="float32")
        if waveform.ndim > 1:
            waveform = waveform.mean(axis=1)
        mp3 = io.BytesIO()
        mp3_signal = librosa.resample(waveform, orig_sr=sr, target_sr=44100)
        sf.write(mp3, mp3_signal, 44100, format="MP3")
        decoded, decoded_sr = sf.read(io.BytesIO(mp3.getvalue()), dtype="float32")
        # Browser AudioContext decodes to the output device's rate before the
        # frontend writes PCM WAV and the server resamples to 22,050 Hz.
        browser_pcm = librosa.resample(decoded, orig_sr=decoded_sr, target_sr=48000)
        padded = np.pad(waveform, (sr * 2, sr * 2))
        variants = {"wav": path.read_bytes(), "mp3_direct": mp3.getvalue(),
                    "mp3_browser_pcm": encode_wav(browser_pcm, 48000),
                    "boundary_silence": encode_wav(padded, sr)}
        row = {"id": sample["id"], "expected": sample["emotion"], "models": {}}
        for model in ("best", "balanced_audio"):
            row["models"][model] = {}
            for name, data in variants.items():
                result = predictor.predict_audio(data, model)
                row["models"][model][name] = {"emotion": result["top_emotion"],
                    "probabilities": result["probabilities"]}
        report.append(row)
    summary = {}
    for model in ("best", "balanced_audio"):
        summary[model] = {}
        for name in ("wav", "mp3_direct", "mp3_browser_pcm", "boundary_silence"):
            outputs = [r["models"][model][name]["emotion"] for r in report]
            summary[model][name] = {"predicted_counts": dict(Counter(outputs)),
                "correct": sum(p == r["expected"] for p, r in zip(outputs, report)),
                "changed_from_wav": sum(p != r["models"][model]["wav"]["emotion"] for p, r in zip(outputs, report))}
    Path("audio_format_diagnostics.json").write_text(json.dumps({"summary": summary, "samples": report}, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
