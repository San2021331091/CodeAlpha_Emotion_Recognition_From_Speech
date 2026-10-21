"""Audit generated emotion folders against corpus labels and training splits."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path

import soundfile as sf

from app.ml.datasets import COMMON_EMOTIONS, load_multicorpus_records, split_records
from app.ml.feature_extractor import TARGET_SR
from train_augmented import load_training_corpus


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(chunk)
    return result.digest()


def audit(directory):
    directory = Path(directory).resolve()
    records = load_multicorpus_records(Path("data").resolve())
    splits = split_records(records)
    metadata, _, _ = load_training_corpus(directory, records, splits["train"])
    if not metadata.get("organized_by_emotion"):
        raise ValueError("This audit requires an emotion-organized corpus.")
    with (directory / "manifest.jsonl").open(encoding="utf-8") as manifest:
        entries = [json.loads(line) for line in manifest]

    def inspect(entry):
        path = directory / entry["class_audio_path"]
        information = sf.info(path)
        if information.frames < information.samplerate * 0.25 or information.format != "WAV":
            raise ValueError(f"Invalid audio file: {path}")
        if entry["kind"] == "augmented":
            if information.samplerate != TARGET_SR or information.channels != 1 or information.subtype != "PCM_16":
                raise ValueError(f"Unexpected generated audio encoding: {path}")
        else:
            parent = (directory / entry["path"]).resolve()
            if not os.path.samefile(path, parent) and digest(path) != digest(parent):
                raise ValueError(f"The copied original differs from its source: {path}")
        return entry["emotion"], entry["kind"], information.duration

    totals, kinds = Counter(), Counter()
    duration = 0.0
    with ThreadPoolExecutor(max_workers=4) as executor:
        for start in range(0, len(entries), 1024):
            for emotion, kind, seconds in executor.map(inspect, entries[start:start + 1024]):
                totals[emotion] += 1
                kinds[kind] += 1
                duration += seconds
    folders = {emotion: sum(1 for _ in (directory / "wav" / emotion).glob("*.wav")) for emotion in COMMON_EMOTIONS}
    if folders != dict(totals) or max(totals.values()) - min(totals.values()) > 1:
        raise ValueError("Emotion-folder file counts are inconsistent or unbalanced.")
    report = {"status": "passed", "total_audio_examples": len(entries), "samples_by_emotion": dict(totals),
        "counts_by_kind": dict(kinds), "total_audio_hours": round(duration / 3600, 3),
        "native_wav_headers_checked": len(entries), "original_copies_match_sources": True,
        "training_parents_only": True, "finite_feature_cache": True,
        "label_check": "Matches original corpus annotations; no independent listening review or new human annotation.",
        "generation_seed": metadata["generation_configuration"]["seed"],
        "augmentation_profile": metadata["generation_configuration"]["profile"]}
    (directory / "quality_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-dir", default="data/training_50000")
    args = parser.parse_args()
    audit(args.corpus_dir)
