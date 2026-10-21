import unittest

import numpy as np

from app.ml.audio_augmentation import augment_waveform, plan_training_samples
from app.ml.datasets import AudioRecord


class AugmentationTests(unittest.TestCase):
    def test_plan_contains_only_training_parents_and_balances_sources_and_labels(self):
        records = [AudioRecord(f"{source}-{label}-{i}.wav", label, source, str(i))
                   for source in ("RAVDESS", "TESS", "EMO-DB") for label in range(6) for i in range(3)]
        training = [i for i, record in enumerate(records) if record.speaker != "2"]
        plan = plan_training_samples(records, training, 50000)
        self.assertEqual(len(plan), 50000)
        self.assertEqual(sum(p["seed"] is None for p in plan), len(training))
        self.assertTrue({p["parent_index"] for p in plan} <= set(training))
        from collections import Counter
        counts = Counter((records[p["parent_index"]].source, records[p["parent_index"]].label) for p in plan)
        self.assertLessEqual(max(counts.values()) - min(counts.values()), 1)
        emotions = Counter(records[p["parent_index"]].label for p in plan)
        self.assertEqual(sorted(emotions.values()), [8333, 8333, 8333, 8333, 8334, 8334])

    def test_waveform_changes_are_reproducible_finite_and_do_not_modify_parent(self):
        signal = (0.5 * np.sin(2 * np.pi * 220 * np.arange(22050) / 22050)).astype(np.float32)
        parent = signal.copy()
        first = augment_waveform(signal, 42)
        np.testing.assert_array_equal(first, augment_waveform(signal, 42))
        np.testing.assert_array_equal(signal, parent)
        self.assertTrue(np.isfinite(first).all())
        self.assertLessEqual(float(np.abs(first).max()), 1)
        self.assertGreaterEqual(len(first), len(signal))
        self.assertLessEqual(len(first), len(signal) + 2 * int(22050 * 0.12))

    def test_conservative_profile_preserves_fundamental_frequency(self):
        sr = 22050
        signal = (0.5 * np.sin(2 * np.pi * 220 * np.arange(sr) / sr)).astype(np.float32)
        augmented = augment_waveform(signal, 20261021, profile="conservative")
        frequencies = np.fft.rfftfreq(len(augmented), 1 / sr)
        fundamental = frequencies[np.argmax(np.abs(np.fft.rfft(augmented)))]
        self.assertAlmostEqual(float(fundamental), 220, delta=2)

    def test_valid_low_gain_corpus_audio_is_not_rejected_as_a_quiet_microphone(self):
        from pathlib import Path
        import tempfile
        import soundfile as sf
        from app.ml.audio_augmentation import _original_waveform
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "quiet_labeled_speech.wav"
            signal = (0.0001 * np.sin(2 * np.pi * 220 * np.arange(22050) / 22050)).astype(np.float32)
            sf.write(path, signal, 22050, subtype="FLOAT")
            normalized = _original_waveform(str(path))
            self.assertAlmostEqual(float(np.abs(normalized).max()), 1.0, places=5)
            silence = Path(temporary) / "silence.wav"
            sf.write(silence, np.zeros(22050), 22050)
            with self.assertRaisesRegex(ValueError, "Silent"):
                _original_waveform(str(silence))

    def test_regeneration_organizes_and_validates_every_emotion(self):
        import json
        from pathlib import Path
        import tempfile
        from unittest.mock import patch
        import soundfile as sf
        from prepare_training_audio import prepare
        from train_augmented import load_training_corpus
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            records = []
            signal = (0.3 * np.sin(2 * np.pi * 220 * np.arange(6615) / 22050)).astype(np.float32)
            for source in ("RAVDESS", "TESS", "EMO-DB"):
                for label in range(6):
                    path = root / f"{source}-{label}.wav"
                    sf.write(path, signal, 22050)
                    records.append(AudioRecord(str(path), label, source, "01"))
            training = list(range(len(records)))
            heldout_path = root / "heldout.wav"
            sf.write(heldout_path, signal, 22050)
            records.append(AudioRecord(str(heldout_path), 0, "RAVDESS", "21"))
            split = {"train": training, "validation": [], "test": [len(records) - 1]}
            output = root / "regenerated"
            cached = (np.zeros((len(records), 231), np.float32), np.zeros((len(records), 40, 128), np.float32))
            with patch("prepare_training_audio.load_multicorpus_records", return_value=records), patch("prepare_training_audio.split_records", return_value=split), patch("prepare_training_audio._cache_features", return_value=cached):
                metadata = prepare(root, output, 74, workers=2)
                self.assertEqual(sorted(metadata["emotion_folder_counts"].values()), [12, 12, 12, 12, 13, 13])
                self.assertFalse(metadata["exact_class_balance"])
                self.assertEqual(metadata["maximum_class_count_difference"], 1)
                manifest = [json.loads(line) for line in (output / "manifest.jsonl").read_text().splitlines()]
                self.assertNotIn(records[-1].cache_key, {entry["parent_key"] for entry in manifest})
                load_training_corpus(output, records, training)
                self.assertEqual(prepare(root, output, 74, workers=2), metadata)
                with self.assertRaisesRegex(ValueError, "different completed corpus"):
                    prepare(root, output, 74, workers=2, seed=99)

    def test_target_cannot_drop_original_training_clips(self):
        records = [AudioRecord("a.wav", 0, "RAVDESS", "01")]
        with self.assertRaises(ValueError):
            plan_training_samples(records, [0], 0)

    def test_training_loader_rejects_a_held_out_parent(self):
        import json
        from pathlib import Path
        import tempfile
        from train_augmented import load_training_corpus
        records = [AudioRecord("train.wav", 0, "RAVDESS", "01"),
                   AudioRecord("heldout.wav", 1, "RAVDESS", "21")]
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            (directory / "metadata.json").write_text(json.dumps({
                "original_training_keys": [records[0].cache_key], "target_count": 1}))
            (directory / "manifest.jsonl").write_text(json.dumps({
                "parent_key": records[1].cache_key, "split": "train"}) + "\n")
            with self.assertRaisesRegex(ValueError, "Held-out"):
                load_training_corpus(directory, records, [0])
