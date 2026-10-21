"""Regenerate emotion-organized audio from labeled training recordings only."""
import os
# Prevent each audio worker from spawning its own large BLAS thread pool.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import shutil
import time

import numpy as np
import soundfile as sf

from app.ml.audio_augmentation import plan_training_samples, render_training_sample
from app.ml.datasets import COMMON_EMOTIONS, load_multicorpus_records, split_records
from app.ml.feature_extractor import extract_sequence_features_from_audio
from app.ml.trainer import _cache_features


def prepare(data_dir, output_dir, target_count=50000, workers=4, seed=20261021, profile="conservative"):
    data_dir, output_dir = Path(data_dir).resolve(), Path(output_dir).resolve()
    records = load_multicorpus_records(data_dir)
    splits = split_records(records)
    training = splits["train"]
    plan = plan_training_samples(records, training, target_count, seed=seed)
    requested_count = target_count
    target_count = len(plan)
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = output_dir / "metadata.json"
    original_keys = [records[i].cache_key for i in training]
    configuration = {"generator_version": 3, "seed": seed, "profile": profile,
                     "target_count": target_count, "original_training_keys": original_keys}
    if metadata_path.exists():
        existing = json.loads(metadata_path.read_text())
        if existing.get("generation_configuration") == configuration:
            paths = [output_dir / name for name in ("sequences.npy", "labels.npy", "manifest.jsonl")]
            if all(path.is_file() for path in paths):
                print(f"Reusing completed {target_count:,}-example training corpus.", flush=True)
                return existing
        raise ValueError("Output contains a different completed corpus; choose another output directory.")
    plan_path = output_dir / "generation_configuration.json"
    if plan_path.exists():
        if json.loads(plan_path.read_text()) != configuration:
            raise ValueError("Incomplete output was generated with different settings. Choose a new directory.")
    elif (output_dir / "wav").exists() and any((output_dir / "wav").rglob("*.wav")):
        raise ValueError("Unversioned audio already exists here. Choose a new output directory.")
    else:
        plan_path.write_text(json.dumps(configuration, indent=2), encoding="utf-8")
    source_durations = {i: sf.info(records[i].path).duration for i in training}
    estimated = sum((source_durations[p["parent_index"]] + 0.24) * 44100 for p in plan if p["seed"] is not None)
    estimated += target_count * 40 * 128 * 4
    free = shutil.disk_usage(output_dir).free
    if estimated * 1.2 + 2 * 1024**3 > free:
        raise RuntimeError(f"Not enough free disk space: estimated {estimated / 1e9:.1f} GB plus working space.")
    print(f"Preparing {len(training):,} originals + {target_count - len(training):,} augmented clips; estimated {estimated / 1e9:.1f} GB.", flush=True)
    _, cached_sequences = _cache_features(data_dir, records)
    waves_dir = output_dir / "wav"
    waves_dir.mkdir(exist_ok=True)
    for emotion in COMMON_EMOTIONS:
        (waves_dir / emotion).mkdir(exist_ok=True)
    sequences = np.lib.format.open_memmap(output_dir / "sequences.npy", mode="w+", dtype="float32", shape=(target_count, 40, 128))
    labels = np.empty(target_count, dtype=np.int64)

    def process(item):
        index, entry = item
        parent_index, seed = entry["parent_index"], entry["seed"]
        parent = records[parent_index]
        emotion_dir = waves_dir / COMMON_EMOTIONS[parent.label]
        if seed is None:
            path = Path(parent.path)
            source_tag = parent.source.lower().replace("-", "_")
            class_path = emotion_dir / f"original_{source_tag}_{parent.speaker}_{path.name}"
            if not class_path.exists():
                try:
                    os.link(path, class_path)
                except OSError:
                    shutil.copy2(path, class_path)
            sequence = cached_sequences[parent_index]
        else:
            path = emotion_dir / f"aug_{index:05d}.wav"
            class_path = path
            if not path.exists():
                render_training_sample(parent, seed, path, profile=profile)
            sequence = extract_sequence_features_from_audio(str(path))
        sequences[index] = sequence
        labels[index] = parent.label
        return {"index": index, "path": os.path.relpath(path, output_dir),
                "class_audio_path": os.path.relpath(class_path, output_dir),
                "label": parent.label, "emotion": COMMON_EMOTIONS[parent.label],
                "source": parent.source, "speaker": parent.speaker, "split": "train",
                "parent_key": parent.cache_key, "seed": seed,
                "kind": "original" if seed is None else "augmented"}

    started = time.perf_counter()
    # executor.map limits queued work using buffersize (Python 3.14 only), so
    # submit bounded chunks instead to support this project's Python 3.11+.
    with ThreadPoolExecutor(max_workers=workers) as executor, (output_dir / "manifest.jsonl.tmp").open("w", encoding="utf-8") as manifest:
        for start in range(0, target_count, 256):
            chunk = list(enumerate(plan[start:start + 256], start=start))
            for item in executor.map(process, chunk):
                manifest.write(json.dumps(item) + "\n")
            completed = min(start + 256, target_count)
            if start == 0 or completed % 2048 == 0 or completed == target_count:
                elapsed = time.perf_counter() - started
                print(f"  Prepared {completed:,}/{target_count:,} examples ({elapsed:.0f}s)", flush=True)
    sequences.flush()
    np.save(output_dir / "labels.npy", labels)
    (output_dir / "manifest.jsonl.tmp").replace(output_dir / "manifest.jsonl")
    by_source = Counter(records[p["parent_index"]].source for p in plan)
    by_emotion = Counter(COMMON_EMOTIONS[records[p["parent_index"]].label] for p in plan)
    folder_counts = {emotion: sum(1 for _ in (waves_dir / emotion).glob("*.wav")) for emotion in COMMON_EMOTIONS}
    if folder_counts != dict(by_emotion):
        raise ValueError("Emotion-folder counts do not match manifest labels.")
    metadata = {"version": 3, "generation_configuration": configuration,
                "target_count": target_count, "original_training_samples": len(training),
                "requested_maximum": requested_count, "exact_class_balance": len(set(by_emotion.values())) == 1,
                "maximum_class_count_difference": max(by_emotion.values()) - min(by_emotion.values()),
                "augmented_training_samples": target_count - len(training), "unique_corpus_recordings": len(records),
                "validation_samples": len(splits["validation"]), "test_samples": len(splits["test"]),
                "samples_by_source": dict(by_source), "samples_by_emotion": dict(by_emotion),
                "original_training_keys": original_keys, "emotions_list": list(COMMON_EMOTIONS),
                "organized_by_emotion": True, "emotion_folder_counts": folder_counts,
                "augmentation": ("45-60 dB SNR noise, subtle echo/filtering, up to 30ms boundary silence; pitch and speed preserved"
                                 if profile == "conservative" else "25-40 dB SNR noise, room echo/filtering, up to 120ms boundary silence; pitch and speed preserved"),
                "label_policy": "Inherited from the original corpus annotation. Augmentation does not create new emotional performances or new speakers.",
                "evaluation_policy": "Original validation/test recordings and their derivatives are excluded from augmentation.",
                "duration_seconds": round(time.perf_counter() - started, 2)}
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in metadata.items() if key not in {"original_training_keys", "generation_configuration"}}, indent=2), flush=True)
    return metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output-dir", default="data/training_50000")
    parser.add_argument("--target-count", type=int, default=50000)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20261021)
    parser.add_argument("--profile", choices=["conservative", "recording_conditions"], default="conservative")
    args = parser.parse_args()
    prepare(args.data_dir, args.output_dir, args.target_count, args.workers, args.seed, args.profile)
