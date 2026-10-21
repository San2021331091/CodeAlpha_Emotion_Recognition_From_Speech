"""Balance an existing expanded corpus without regenerating its audio."""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import shutil

import numpy as np

from app.ml.datasets import COMMON_EMOTIONS, load_multicorpus_records, split_records
from train_augmented import load_training_corpus


def balance(source, destination, maximum=50000):
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if source == destination:
        raise ValueError("Use a separate output directory to preserve the original corpus.")
    records = load_multicorpus_records(Path("data").resolve())
    metadata, sequences, labels = load_training_corpus(source, records, split_records(records)["train"])
    with (source / "manifest.jsonl").open(encoding="utf-8") as manifest:
        entries = [json.loads(line) for line in manifest]
    quota = min(maximum // len(COMMON_EMOTIONS), min(np.bincount(labels, minlength=len(COMMON_EMOTIONS))))
    keep = np.ones(len(entries), dtype=bool)
    counts = Counter(labels.tolist())
    # Discard only surplus augmented entries, keeping every original clip.
    for index in range(len(entries) - 1, -1, -1):
        label = int(labels[index])
        if counts[label] > quota and entries[index]["kind"] == "augmented":
            keep[index] = False
            counts[label] -= 1
    if set(counts.values()) != {quota}:
        raise ValueError("Cannot balance at this maximum without discarding original recordings.")
    positions = np.flatnonzero(keep)
    destination.mkdir(parents=True, exist_ok=True)
    if (destination / "metadata.json").exists():
        raise ValueError("The destination is already complete; choose a new directory.")
    (destination / "wav").mkdir(exist_ok=True)
    output = np.lib.format.open_memmap(destination / "sequences.npy", mode="w+", dtype="float32", shape=(len(positions), 40, 128))
    output_entries = []
    for new_index, old_index in enumerate(positions):
        entry = dict(entries[old_index])
        source_audio = (source / entry["path"]).resolve()
        if entry["kind"] == "augmented":
            target_audio = destination / "wav" / source_audio.name
            if not target_audio.exists():
                try:
                    os.link(source_audio, target_audio)
                except OSError:
                    shutil.copy2(source_audio, target_audio)
        else:
            target_audio = source_audio
        entry["index"] = new_index
        entry["path"] = os.path.relpath(target_audio, destination)
        output_entries.append(entry)
        if (new_index + 1) % 10000 == 0:
            print(f"Linked {new_index + 1:,}/{len(positions):,} balanced examples", flush=True)
    for start in range(0, len(positions), 512):
        output[start:start + 512] = sequences[positions[start:start + 512]]
    output.flush()
    np.save(destination / "labels.npy", labels[positions])
    with (destination / "manifest.jsonl").open("w", encoding="utf-8") as manifest:
        for entry in output_entries:
            manifest.write(json.dumps(entry) + "\n")
    metadata.update({"target_count": len(positions), "requested_maximum": maximum, "exact_class_balance": True,
        "augmented_training_samples": sum(e["kind"] == "augmented" for e in output_entries),
        "samples_by_source": dict(Counter(e["source"] for e in output_entries)),
        "samples_by_emotion": dict(Counter(e["emotion"] for e in output_entries)),
        "balanced_from": str(source), "discarded_augmented_examples": int((~keep).sum())})
    (destination / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    # Reuse the training loader's provenance, dimensions, labels and balance checks.
    load_training_corpus(destination, records, split_records(records)["train"])
    print(json.dumps({"total": len(positions), "samples_by_emotion": metadata["samples_by_emotion"],
                      "removed_augmented": int((~keep).sum())}, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="data/training_50000")
    parser.add_argument("--destination", default="data/training_balanced")
    parser.add_argument("--maximum", type=int, default=50000)
    args = parser.parse_args()
    balance(args.source, args.destination, args.maximum)
