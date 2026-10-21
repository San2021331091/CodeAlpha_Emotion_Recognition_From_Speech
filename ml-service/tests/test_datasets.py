import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf

from app.ml.datasets import (
    COMMON_EMOTIONS,
    AudioRecord,
    read_emodb_records,
    read_tess_records,
    split_records,
)
from app.ml.trainer import _source_balanced_weights


def _write_silent_wav(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, np.zeros(2205, dtype=np.float32), 22050)


class MultiCorpusDatasetTests(unittest.TestCase):
    def test_tess_standard_folder_names_and_unsupported_classes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for speaker in ("OAF", "YAF"):
                for emotion in (
                    "neutral",
                    "happy",
                    "sad",
                    "angry",
                    "fear",
                    "disgust",
                    "pleasant_surprise",
                ):
                    for word in ("back", "book"):
                        _write_silent_wav(
                            root / f"{speaker}_{emotion}" / f"{speaker}_{word}_{emotion}.wav"
                        )

            records = read_tess_records(root)

            self.assertEqual(len(records), 24)
            self.assertEqual({record.speaker for record in records}, {"OAF", "YAF"})
            self.assertEqual(
                {COMMON_EMOTIONS[record.label] for record in records},
                set(COMMON_EMOTIONS),
            )
            splits = split_records(records)
            self.assertEqual(len(splits["train"]), 12)
            self.assertEqual(len(splits["test"]), 6)
            self.assertEqual(len(splits["validation"]), 6)
            self.assertFalse(set(splits["validation"]) & set(splits["test"]))
            self.assertEqual({records[i].speaker for i in splits["train"]}, {"OAF"})
            self.assertEqual({records[i].speaker for i in splits["validation"] + splits["test"]}, {"YAF"})

    def test_emodb_codes_map_to_shared_labels_and_ignore_boredom(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            codes = {
                "N": "neutral",
                "F": "happy",
                "T": "sad",
                "W": "angry",
                "A": "fearful",
                "E": "disgust",
                "L": "boredom",
            }
            for speaker in range(10):
                for emotion_code, _ in codes.items():
                    _write_silent_wav(
                        root / f"{speaker:02d}a01{emotion_code}a.wav"
                    )

            records = read_emodb_records(root)

            self.assertEqual(len(records), 60)
            self.assertEqual(
                {COMMON_EMOTIONS[record.label] for record in records},
                set(COMMON_EMOTIONS),
            )
            splits = split_records(records)
            self.assertEqual([len(splits[name]) for name in ("train", "validation", "test")], [36, 12, 12])
            for split in splits.values():
                self.assertEqual(len({records[index].speaker for index in split}), len(split) // 6)

    def test_source_balanced_weights_equalize_source_emotion_cells(self):
        records = [
            AudioRecord(f"{source}-{emotion}-{index}.wav", emotion, source, str(index))
            for source, count in (("RAVDESS", 8), ("TESS", 3), ("EMO-DB", 2))
            for emotion in range(len(COMMON_EMOTIONS))
            for index in range(count)
        ]

        weights = _source_balanced_weights(records, np.arange(len(records)))

        self.assertAlmostEqual(float(weights.mean()), 1.0, places=6)
        cell_totals = {}
        for record, weight in zip(records, weights):
            cell_totals[(record.source, record.label)] = (
                cell_totals.get((record.source, record.label), 0.0) + float(weight)
            )
        self.assertLess(max(cell_totals.values()) - min(cell_totals.values()), 1e-5)

    def test_missing_shared_emotion_in_source_has_explicit_error(self):
        records = [
            AudioRecord(f"{source}-{emotion}.wav", emotion, source, str(emotion))
            for source in ("RAVDESS", "TESS", "EMO-DB")
            for emotion in range(len(COMMON_EMOTIONS))
        ]
        records.pop()

        with self.assertRaisesRegex(ValueError, "every shared emotion"):
            _source_balanced_weights(records, np.arange(len(records)))


if __name__ == "__main__":
    unittest.main()
