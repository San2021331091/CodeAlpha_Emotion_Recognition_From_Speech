import os
import math
import numpy as np
import scipy.signal as signal
import soundfile as sf
from typing import List, Tuple, Dict, Any


EMOTIONS = [
    "neutral",
    "calm",
    "happy",
    "sad",
    "angry",
    "fearful",
    "disgust",
    "surprised"
]

# Russell's Circumplex Model (Valence, Arousal) coordinates
EMOTION_COORDINATES = {
    "neutral": {"valence": 0.05, "arousal": 0.02, "quadrant": "Neutral Equilibrium"},
    "calm": {"valence": 0.65, "arousal": -0.45, "quadrant": "Low Arousal / Positive Valence (Serene)"},
    "happy": {"valence": 0.82, "arousal": 0.72, "quadrant": "High Arousal / Positive Valence (Elated)"},
    "sad": {"valence": -0.78, "arousal": -0.62, "quadrant": "Low Arousal / Negative Valence (Melancholy)"},
    "angry": {"valence": -0.75, "arousal": 0.88, "quadrant": "High Arousal / Negative Valence (Hostile)"},
    "fearful": {"valence": -0.68, "arousal": 0.74, "quadrant": "High Arousal / Negative Valence (Alarmed)"},
    "disgust": {"valence": -0.62, "arousal": 0.15, "quadrant": "Moderate Arousal / Negative Valence (Averse)"},
    "surprised": {"valence": 0.35, "arousal": 0.80, "quadrant": "High Arousal / Positive-Neutral (Astonished)"},
}

# Formant frequencies (Hz) for typical vowels
VOWEL_FORMANTS = {
    "a": [800, 1200, 2500, 3500],
    "i": [300, 2300, 3000, 3700],
    "u": [350, 800, 2200, 3300],
    "e": [500, 1800, 2600, 3600],
    "o": [500, 900, 2400, 3400]
}


def synthesize_emotional_utterance(
    emotion: str,
    duration: float = 2.0,
    sr: int = 22050,
    speaker_gender: str = "female",
    seed: int = None
) -> np.ndarray:
    """
    Synthesizes a realistic acoustic utterance representing the specified emotion
    using source-filter formant synthesis, dynamic pitch trajectory, jitter/shimmer,
    and emotion-specific prosody.
    """
    if seed is not None:
        np.random.seed(seed)

    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    n_samples = len(t)

    # Base pitch range (F0) based on speaker gender
    base_f0 = 210.0 if speaker_gender == "female" else 135.0

    # Emotion-specific pitch contour, energy, and prosodic parameters
    if emotion == "angry":
        # Sharp high pitch, aggressive upward pitch excursions, high vocal tension
        f0_contour = base_f0 * 1.45 + 50.0 * np.sin(2 * np.pi * 3.5 * t) + 30.0 * np.exp(-t * 2.0)
        energy_env = np.clip(1.0 - 0.2 * t, 0.4, 1.0)
        noise_level = 0.08
        jitter_factor = 0.04
        vowels = ["a", "e", "a"]
    elif emotion == "happy":
        # Elevated pitch, buoyant rhythmic rise and fall, bright timbre
        f0_contour = base_f0 * 1.30 + 45.0 * np.sin(2 * np.pi * 2.5 * t) + 20.0 * np.cos(2 * np.pi * 5.0 * t)
        energy_env = 0.7 + 0.3 * np.sin(2 * np.pi * 2.0 * t)**2
        noise_level = 0.04
        jitter_factor = 0.02
        vowels = ["a", "i", "e"]
    elif emotion == "sad":
        # Low pitch, downward drooping contour, very low energy, slow decay
        f0_contour = (base_f0 * 0.85) - 25.0 * (t / duration) + 6.0 * np.sin(2 * np.pi * 1.2 * t)
        energy_env = np.clip(0.6 * np.exp(-t * 1.2), 0.1, 0.6)
        noise_level = 0.06
        jitter_factor = 0.02
        vowels = ["u", "o", "u"]
    elif emotion == "fearful":
        # High pitch with rapid 7Hz micro-tremor, breathless aspiration
        f0_contour = base_f0 * 1.35 + 28.0 * np.sin(2 * np.pi * 7.5 * t) + 15.0 * np.random.randn(n_samples) * 0.15
        energy_env = 0.5 + 0.3 * np.abs(np.sin(2 * np.pi * 4.0 * t))
        noise_level = 0.18
        jitter_factor = 0.06
        vowels = ["i", "e", "i"]
    elif emotion == "disgust":
        # Slower cadence, low guttural pitch sag, vocal constriction
        f0_contour = base_f0 * 0.92 - 18.0 * np.sin(2 * np.pi * 1.8 * t)
        energy_env = 0.6 + 0.2 * np.cos(2 * np.pi * 1.5 * t)
        noise_level = 0.09
        jitter_factor = 0.035
        vowels = ["u", "a", "o"]
    elif emotion == "surprised":
        # Abrupt steep pitch climb in the first 0.4s followed by high sustain
        pitch_jump = 90.0 * np.tanh(t * 8.0)
        f0_contour = base_f0 * 1.20 + pitch_jump + 15.0 * np.sin(2 * np.pi * 3.0 * t)
        energy_env = np.where(t < 0.2, t / 0.2, 0.85)
        noise_level = 0.07
        jitter_factor = 0.025
        vowels = ["o", "a", "i"]
    elif emotion == "calm":
        # Smooth, low, gentle pitch contour, low arousal
        f0_contour = base_f0 * 0.90 + 8.0 * np.sin(2 * np.pi * 1.0 * t)
        energy_env = 0.45 + 0.1 * np.sin(2 * np.pi * 0.8 * t)
        noise_level = 0.03
        jitter_factor = 0.015
        vowels = ["e", "o", "a"]
    else:  # neutral
        # Flat, balanced, standard pitch contour
        f0_contour = base_f0 + 10.0 * np.sin(2 * np.pi * 1.5 * t)
        energy_env = 0.55 * np.ones_like(t)
        noise_level = 0.03
        jitter_factor = 0.015
        vowels = ["a", "e", "o"]

    # Synthesize Glottal Pulse Source via phase integration with jitter
    f0_contour = np.clip(f0_contour, 70.0, 500.0)
    jitter = 1.0 + jitter_factor * np.random.randn(n_samples)
    phase = np.cumsum(2 * np.pi * (f0_contour * jitter) / sr)

    # Rosenberg glottal pulse model simulation
    glottal_source = np.zeros(n_samples, dtype=np.float32)
    # Generate harmonic pulse train
    for harmonic_k in range(1, 16):
        harmonic_decay = 1.0 / (harmonic_k ** 1.35)
        glottal_source += harmonic_decay * np.sin(harmonic_k * phase)

    # Add aspiration noise
    aspiration_noise = np.random.randn(n_samples).astype(np.float32)
    b_noise, a_noise = signal.butter(2, [1000 / (sr / 2), 6000 / (sr / 2)], btype="band")
    filtered_noise = signal.lfilter(b_noise, a_noise, aspiration_noise)
    excitation = glottal_source + noise_level * filtered_noise

    # Vocal tract resonance filtering through 4 emotional vowel formants
    segment_len = n_samples // len(vowels)
    filtered_audio = np.zeros(n_samples, dtype=np.float32)

    for seg_idx, vowel_char in enumerate(vowels):
        start_idx = seg_idx * segment_len
        end_idx = min((seg_idx + 1) * segment_len, n_samples)
        formants = VOWEL_FORMANTS[vowel_char]
        segment_signal = excitation[start_idx:end_idx]

        # Multi-formant resonator filter bank
        filtered_segment = np.zeros_like(segment_signal)
        for f_res in formants:
            bw = f_res * 0.1  # Bandwidth
            q = f_res / bw
            # Biquad peak / bandpass filter
            b, a = signal.iirpeak(f_res / (sr / 2), q)
            filtered_segment += signal.lfilter(b, a, segment_signal) * 0.35

        filtered_audio[start_idx:end_idx] = filtered_segment

    # Apply emotional energy envelope and windowing
    window = signal.windows.tukey(n_samples, alpha=0.15)
    output_audio = filtered_audio * energy_env * window

    # Amplitude normalization
    max_val = np.max(np.abs(output_audio))
    if max_val > 1e-4:
        output_audio = (output_audio / max_val) * 0.92

    return output_audio.astype(np.float32)


def generate_benchmark_samples(output_dir: str, sr: int = 22050) -> List[Dict[str, Any]]:
    """
    Generates 8 high-fidelity benchmark WAV audio samples (one for each emotion)
    for interactive testing in the frontend.
    """
    os.makedirs(output_dir, exist_ok=True)
    manifest = []

    benchmark_configs = [
        {"id": "bench_happy", "title": "Joyful Exclamation", "emotion": "happy", "speaker": "Female-01", "dur": 2.2, "seed": 42,
         "desc": "High pitch variability and bright melodic cadence indicating joy and celebration."},
        {"id": "bench_angry", "title": "Hostile Reproach", "emotion": "angry", "speaker": "Male-01", "dur": 1.9, "seed": 101,
         "desc": "Aggressive vocal tract tension, sharp initial attack, and elevated RMS intensity."},
        {"id": "bench_sad", "title": "Melancholy Statement", "emotion": "sad", "speaker": "Female-02", "dur": 2.5, "seed": 202,
         "desc": "Low fundamental frequency, descending prosodic slope, and subdued acoustic energy."},
        {"id": "bench_neutral", "title": "Matter-of-fact Briefing", "emotion": "neutral", "speaker": "Male-02", "dur": 2.0, "seed": 303,
         "desc": "Balanced spectral distribution, standard conversational pitch, and neutral valence."},
        {"id": "bench_calm", "title": "Soothing Guidance", "emotion": "calm", "speaker": "Female-01", "dur": 2.4, "seed": 404,
         "desc": "Gentle, relaxed pitch contour, low arousal, and smooth harmonic transitions."},
        {"id": "bench_fearful", "title": "Startled Warning", "emotion": "fearful", "speaker": "Female-03", "dur": 1.8, "seed": 505,
         "desc": "High frequency vocal micro-tremor with breathy aspiration noise and rapid pitch jitter."},
        {"id": "bench_disgust", "title": "Visceral Rejection", "emotion": "disgust", "speaker": "Male-01", "dur": 2.1, "seed": 606,
         "desc": "Vocal tract constriction, guttural pitch sagging, and negative emotional valence."},
        {"id": "bench_surprised", "title": "Astonished Reaction", "emotion": "surprised", "speaker": "Female-02", "dur": 2.0, "seed": 707,
         "desc": "Abrupt pitch jump exceeding 100 Hz in onset with sharp acoustic arousal."}
    ]

    for cfg in benchmark_configs:
        gender = "female" if "Female" in cfg["speaker"] else "male"
        audio = synthesize_emotional_utterance(
            emotion=cfg["emotion"],
            duration=cfg["dur"],
            sr=sr,
            speaker_gender=gender,
            seed=cfg["seed"]
        )
        filename = f"{cfg['id']}_{cfg['emotion']}.wav"
        filepath = os.path.join(output_dir, filename)
        sf.write(filepath, audio, sr)

        manifest.append({
            "id": cfg["id"],
            "title": cfg["title"],
            "emotion": cfg["emotion"],
            "speaker": cfg["speaker"],
            "duration_seconds": cfg["dur"],
            "description": cfg["desc"],
            "filename": filename
        })

    return manifest


def generate_training_dataset(
    output_dir: str,
    samples_per_emotion: int = 150,
    sr: int = 22050
) -> List[Tuple[str, str]]:
    """
    Generates a balanced dataset of synthetic audio samples across all 8 emotions.
    Returns list of (filepath, emotion) tuples.
    """
    os.makedirs(output_dir, exist_ok=True)
    dataset = []

    total = len(EMOTIONS) * samples_per_emotion
    print(f"Generating synthetic speech emotion dataset ({total} total samples)...")

    sample_counter = 0
    for emotion in EMOTIONS:
        for i in range(samples_per_emotion):
            gender = "female" if i % 2 == 0 else "male"
            dur = np.random.uniform(1.6, 2.8)
            seed = 1000 + sample_counter
            audio = synthesize_emotional_utterance(
                emotion=emotion,
                duration=dur,
                sr=sr,
                speaker_gender=gender,
                seed=seed
            )
            filename = f"sample_{sample_counter:04d}_{emotion}_{gender}.wav"
            filepath = os.path.join(output_dir, filename)
            sf.write(filepath, audio, sr)
            dataset.append((filepath, emotion))
            sample_counter += 1

    return dataset
