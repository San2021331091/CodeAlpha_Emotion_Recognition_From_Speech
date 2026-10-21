import os
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from app.ml.data_generator import EMOTIONS
from app.ml.ravdess import (
    TEST_ACTORS,
    TRAIN_ACTORS,
    VALIDATION_ACTORS,
    prepare_dataset,
)


COMMON_EMOTIONS = ("neutral", "happy", "sad", "angry", "fearful", "disgust")
_EMODB_LABELS = {
    "N": "neutral",
    "F": "happy",
    "T": "sad",
    "W": "angry",
    "A": "fearful",
    "E": "disgust",
}
_IGNORED_EMOTIONS = {"calm", "surprise", "surprised", "pleasant_surprise", "boredom"}


@dataclass(frozen=True)
class AudioRecord:
    path: str
    label: int
    source: str
    speaker: str

    @property
    def cache_key(self) -> str:
        return f"{self.source}/{self.speaker}/{Path(self.path).name}"


def _shared_label(name: str) -> str | None:
    normalized = name.lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "anger": "angry",
        "angry": "angry",
        "disgust": "disgust",
        "fear": "fearful",
        "fearful": "fearful",
        "happy": "happy",
        "happiness": "happy",
        "neutral": "neutral",
        "sad": "sad",
        "sadness": "sad",
    }
    if normalized in _IGNORED_EMOTIONS:
        return None
    return aliases.get(normalized)


def _label_from_tess_path(path: Path) -> str | None:
    tokens = re.split(r"[_\W]+", "_".join((path.parent.name, path.stem)).lower())
    for token in reversed(tokens):
        label = _shared_label(token)
        if label:
            return label
    if any(token in {"surprise", "surprised", "ps"} for token in tokens):
        return None
    if any(token in {"calm", "boredom"} for token in tokens):
        return None
    return None


def read_tess_records(root: Path) -> list[AudioRecord]:
    if not root.is_dir():
        raise FileNotFoundError(
            f"TESS dataset folder not found: {root}. "
            "Place the extracted TESS files under ml-service/data/tess or set TESS_DATA_DIR."
        )

    records = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() != ".wav":
            continue
        label = _label_from_tess_path(path)
        if label is None:
            continue
        speaker_match = re.match(r"^(OAF|YAF)(?:_|$)", path.stem, re.IGNORECASE)
        if speaker_match is None:
            speaker_match = re.match(r"^(OAF|YAF)(?:_|$)", path.parent.name, re.IGNORECASE)
        if speaker_match is None:
            raise ValueError(f"Could not identify a TESS speaker in '{path}'. Expected an OAF or YAF filename.")
        records.append(AudioRecord(str(path), COMMON_EMOTIONS.index(label), "TESS", speaker_match.group(1).upper()))

    if not records:
        raise ValueError(f"No supported TESS WAV recordings found under '{root}'.")
    return records


def read_emodb_records(root: Path) -> list[AudioRecord]:
    if not root.is_dir():
        raise FileNotFoundError(
            f"EMO-DB dataset folder not found: {root}. "
            "Place the extracted EMO-DB files under ml-service/data/emo-db or set EMODB_DATA_DIR."
        )

    records = []
    for path in sorted(root.rglob("*.wav")):
        match = re.fullmatch(r"(?P<speaker>\d{2})[a-z]\d{2}(?P<emotion>[A-Z])[a-z]", path.stem)
        if match is None:
            raise ValueError(f"Unrecognized EMO-DB filename '{path.name}'; expected e.g. 03a01Fa.wav.")
        label = _EMODB_LABELS.get(match.group("emotion"))
        if label is None:
            continue
        records.append(
            AudioRecord(
                str(path),
                COMMON_EMOTIONS.index(label),
                "EMO-DB",
                match.group("speaker"),
            )
        )

    if not records:
        raise ValueError(f"No supported EMO-DB WAV recordings found under '{root}'.")
    return records


def _partition_speakers(speakers: list[str]) -> dict[str, set[str]]:
    ordered = sorted(speakers)
    if len(ordered) < 2:
        raise ValueError("Each dataset needs at least two speakers for a speaker-disjoint train/test split.")
    if len(ordered) == 2:
        return {"train": {ordered[0]}, "validation": set(), "test": {ordered[1]}}

    train_end = max(1, int(len(ordered) * 0.6))
    validation_count = max(1, int(len(ordered) * 0.2))
    validation_end = train_end + validation_count
    if validation_end >= len(ordered):
        validation_end = len(ordered) - 1
    return {
        "train": set(ordered[:train_end]),
        "validation": set(ordered[train_end:validation_end]),
        "test": set(ordered[validation_end:]),
    }


def split_records(records: list[AudioRecord]) -> dict[str, list[int]]:
    by_source: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
    for index, record in enumerate(records):
        by_source[record.source][record.speaker].append(index)

    splits: dict[str, list[int]] = {"train": [], "validation": [], "test": []}
    for source, speakers in sorted(by_source.items()):
        ordered_speakers = sorted(speakers)
        if source == "RAVDESS":
            groups = {
                "train": {f"{actor:02d}" for actor in TRAIN_ACTORS},
                "validation": {f"{actor:02d}" for actor in VALIDATION_ACTORS},
                "test": {f"{actor:02d}" for actor in TEST_ACTORS},
            }
        elif source == "TESS" and len(ordered_speakers) == 2:
            # TESS provides only two speakers. Train on OAF and reserve clips
            # from YAF for validation and test so model selection sees this
            # corpus. The two held-out partitions share a speaker, but never
            # share recordings; this limitation is recorded in the metrics.
            train_speaker, heldout_speaker = ordered_speakers
            splits["train"].extend(speakers[train_speaker])
            for label in range(len(COMMON_EMOTIONS)):
                clips = sorted(
                    index for index in speakers[heldout_speaker]
                    if records[index].label == label
                )
                if len(clips) < 2:
                    raise ValueError(f"TESS/{COMMON_EMOTIONS[label]} needs at least two held-out clips.")
                validation_count = max(1, int(len(clips) * 0.2))
                splits["validation"].extend(clips[:validation_count])
                splits["test"].extend(clips[validation_count:])
            continue
        else:
            groups = _partition_speakers(ordered_speakers)

        assigned = set().union(*groups.values())
        if assigned != set(ordered_speakers):
            raise ValueError(f"Speaker split for {source} omitted or added speakers.")
        if any(groups[left] & groups[right] for left, right in (("train", "validation"), ("train", "test"), ("validation", "test"))):
            raise ValueError(f"Speaker leakage detected in the {source} split.")
        for split_name, split_speakers in groups.items():
            for speaker in sorted(split_speakers):
                splits[split_name].extend(speakers[speaker])

    for split_name in splits:
        splits[split_name].sort()
    return splits


def load_multicorpus_records(data_dir: Path) -> list[AudioRecord]:
    ravdess_records = prepare_dataset(str(data_dir))
    shared_labels = set(COMMON_EMOTIONS)
    records = [
        AudioRecord(
            path,
            COMMON_EMOTIONS.index(EMOTIONS[label]),
            "RAVDESS",
            f"{actor:02d}",
        )
        for path, label, actor in ravdess_records
        if EMOTIONS[label] in shared_labels
    ]

    tess_root = Path(os.environ.get("TESS_DATA_DIR", data_dir / "tess"))
    emodb_root = Path(os.environ.get("EMODB_DATA_DIR", data_dir / "emo-db"))
    records.extend(read_tess_records(tess_root))
    records.extend(read_emodb_records(emodb_root))

    if not records:
        raise ValueError("No supported speech recordings were found in the selected datasets.")
    return records
