"""Auditable waveform augmentation of training recordings only."""
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy.signal import lfilter

from app.ml.datasets import COMMON_EMOTIONS
from app.ml.feature_extractor import TARGET_SR


def plan_training_samples(records, train_indices, target_count, seed=42):
    if target_count < len(train_indices):
        raise ValueError("Target cannot be smaller than the original training set.")
    groups = defaultdict(list)
    plan = []
    for index in train_indices:
        record = records[index]
        groups[(record.source, record.label)].append(int(index))
        plan.append({"parent_index": int(index), "seed": None})
    rng = np.random.default_rng(seed)
    keys = sorted(groups)
    if not keys:
        raise ValueError("No training recordings were provided.")
    labels = sorted({label for _, label in keys})
    quota = target_count // len(labels)
    emotion_counts = Counter(records[index].label for index in train_indices)
    if any(count > quota for count in emotion_counts.values()):
        raise ValueError("The target is too small to balance emotions while keeping all original training recordings.")
    # Fill the least represented dataset/emotion cell first. Every original
    # remains present, and the final corpus has equal weight per source/class.
    counts = Counter({key: len(value) for key, value in groups.items()})
    cycles = {key: [] for key in keys}
    for _ in range(target_count - len(plan)):
        key = min(keys, key=lambda item: (emotion_counts[item[1]], counts[item], item))
        # Visit every source recording before reusing one from the same cell.
        # This avoids generating many variants of a few randomly chosen clips.
        if not cycles[key]:
            cycles[key] = rng.permutation(groups[key]).tolist()
        parent = int(cycles[key].pop())
        plan.append({"parent_index": parent, "seed": int(rng.integers(0, 2**31))})
        counts[key] += 1
        emotion_counts[key[1]] += 1
    return plan


@lru_cache(maxsize=32)
def _original_waveform(path):
    import librosa
    import soundfile as sf
    waveform, sample_rate = sf.read(path, dtype="float32")
    if waveform.ndim > 1:
        waveform = waveform.mean(axis=1)
    if len(waveform) < sample_rate * 0.25 or not np.isfinite(waveform).all():
        raise ValueError(f"Invalid training source: {path}")
    peak = float(np.max(np.abs(waveform)))
    if peak <= 1e-6:
        raise ValueError(f"Silent training source: {path}")
    # Corpus clips have validated labels and may be recorded at low gain.
    # The live-microphone rejection threshold must not discard these originals.
    if sample_rate != TARGET_SR:
        waveform = librosa.resample(waveform, orig_sr=sample_rate, target_sr=TARGET_SR)
    return waveform / max(float(np.max(np.abs(waveform))), 1e-8)


def augment_waveform(waveform, seed, sr=TARGET_SR, profile="conservative"):
    """Small recording-condition changes without shifting vocal pitch/speed."""
    rng = np.random.default_rng(seed)
    if profile not in {"conservative", "recording_conditions"}:
        raise ValueError(f"Unknown augmentation profile: {profile}")
    conservative = profile == "conservative"
    signal = np.asarray(waveform, dtype=np.float32).copy()
    # A quiet room reflection and mild microphone frequency-response changes.
    delay = int(sr * rng.uniform(0.008, 0.035))
    if delay < len(signal):
        echo_gain = rng.uniform(0.0, 0.03) if conservative else rng.uniform(0.03, 0.12)
        signal[delay:] += echo_gain * signal[:-delay].copy()
    tilt = float(rng.uniform(-0.03, 0.03) if conservative else rng.uniform(-0.12, 0.12))
    signal = lfilter([1.0, tilt], [1.0], signal).astype(np.float32)
    noise = rng.normal(0, 1, len(signal)).astype(np.float32)
    # Mix a small amount of colored background noise with broadband noise.
    noise = 0.8 * noise + 0.2 * lfilter([0.2], [1.0, -0.8], noise)
    noise_rms = np.sqrt(np.mean(noise ** 2))
    signal_rms = np.sqrt(np.mean(signal ** 2))
    snr_db = float(rng.uniform(45, 60) if conservative else rng.uniform(25, 40))
    signal += noise * (signal_rms / max(noise_rms, 1e-8)) * 10 ** (-snr_db / 20)
    signal *= float(rng.uniform(0.7, 1.0)) / max(float(np.max(np.abs(signal))), 1e-8)
    before, after = rng.integers(0, int(sr * (0.03 if conservative else 0.12)) + 1, size=2)
    return np.pad(signal, (int(before), int(after))).astype(np.float32)


def render_training_sample(record, seed, destination, profile="conservative"):
    import soundfile as sf
    signal = augment_waveform(_original_waveform(str(Path(record.path).resolve())), seed, profile=profile)
    sf.write(destination, signal, TARGET_SR, subtype="PCM_16")
