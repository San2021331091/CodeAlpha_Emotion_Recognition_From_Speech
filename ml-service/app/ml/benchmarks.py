"""Build benchmark cards from held-out recordings in every training corpus."""
import json
import shutil
from pathlib import Path

import soundfile as sf

from app.ml.datasets import COMMON_EMOTIONS, split_records


def create_benchmarks(data_dir, records):
    root = Path(data_dir)
    destination = root / "audio_samples" / "benchmark"
    destination.mkdir(parents=True, exist_ok=True)
    test_records = [records[index] for index in split_records(records)["test"]]
    manifest = []
    for corpus, prefix, language in (
        ("RAVDESS", "ravdess", "English"),
        ("TESS", "tess", "English (Canada)"),
        ("EMO-DB", "emodb", "German"),
    ):
        for label, emotion in enumerate(COMMON_EMOTIONS):
            record = next(
                (item for item in test_records if item.source == corpus and item.label == label),
                None,
            )
            if record is None:
                raise ValueError(f"No held-out benchmark available for {corpus}/{emotion}")
            source = Path(record.path)
            filename = f"{prefix}_{emotion}_{source.name}"
            shutil.copy2(source, destination / filename)
            manifest.append({
                "id": f"{prefix}_{emotion}",
                "title": f"{corpus}: {emotion.title()}",
                "emotion": emotion,
                "dataset": corpus,
                "speaker": record.speaker,
                "language": language,
                "duration_seconds": round(sf.info(source).duration, 2),
                "description": f"Human speech from a held-out {corpus} test speaker with {emotion} emotion.",
                "filename": filename,
            })
    temporary = root / "benchmark_manifest.json.tmp"
    temporary.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    temporary.replace(root / "benchmark_manifest.json")
    return manifest
