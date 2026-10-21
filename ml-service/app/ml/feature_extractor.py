import io
import os
os.environ.setdefault("NUMBA_CACHE_DIR", os.path.join(os.path.dirname(__file__), "..", "..", "..", ".numba_cache"))
import numpy as np
import librosa
import soundfile as sf
from typing import Tuple, Dict, Any, Union


TARGET_SR = 22050
FIXED_FRAMES = 128  # Fixed time frames for CNN/LSTM temporal sequence


class InvalidAudioError(ValueError):
    """Invalid audio supplied by a caller, rather than an inference failure."""


def load_audio_signal(audio_source: Union[str, bytes, io.BytesIO], sr: int = TARGET_SR) -> Tuple[np.ndarray, int]:
    """
    Loads audio from either a file path or in-memory byte buffer into a normalized float32 waveform.
    """
    try:
        if isinstance(audio_source, (bytes, bytearray)):
            y, file_sr = sf.read(io.BytesIO(audio_source), dtype="float32")
        elif isinstance(audio_source, io.BytesIO):
            audio_source.seek(0)
            y, file_sr = sf.read(audio_source, dtype="float32")
        elif isinstance(audio_source, str) and os.path.exists(audio_source):
            y, file_sr = sf.read(audio_source, dtype="float32")
        else:
            raise InvalidAudioError("Invalid audio source.")
    except (sf.LibsndfileError, RuntimeError) as exc:
        raise InvalidAudioError("Audio format could not be decoded. Convert the recording to PCM WAV and try again.") from exc

    if not y.size or not np.all(np.isfinite(y)):
        raise InvalidAudioError("Audio must contain valid, non-empty samples.")

    # If stereo or multi-channel, average to mono
    if y.ndim > 1:
        y = np.mean(y, axis=1)

    # Resample if sample rate doesn't match TARGET_SR
    if file_sr != sr:
        y = librosa.resample(y, orig_sr=file_sr, target_sr=sr)

    # Validate before peak normalization: otherwise even near-silent input is
    # amplified to full scale and the classifier invents an emotion for it.
    if len(y) < int(sr * 0.25):
        raise InvalidAudioError("Record at least 0.25 seconds of audible speech and try again.")
    max_amp = float(np.max(np.abs(y)))
    rms_val = float(np.sqrt(np.mean(np.square(y, dtype=np.float64))))
    if max_amp < 0.015 or rms_val < 0.001:
        raise InvalidAudioError("The recording is silent or too quiet. Move closer to the microphone and try again.")

    # Normalize amplitude to [-1.0, 1.0] safely. The currently deployed model
    # was trained with this preprocessing, so inference must preserve it.
    max_amp = np.max(np.abs(y))
    if max_amp > 1e-5:
        y = y / max_amp

    return y, sr


def extract_features_from_audio(
    audio_source: Union[str, bytes, io.BytesIO],
    sr: int = TARGET_SR
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Extracts comprehensive acoustic features from audio:
    1. flat_features: 1D vector of acoustic statistics (MFCCs, Chroma, Contrast, Tonnetz, ZCR, RMS, etc.)
    2. sequence_features: 2D matrix (40, FIXED_FRAMES) representing MFCC time-frequency evolution for CNN/LSTM.
    3. summary: Dictionary of acoustic statistics for frontend rendering.
    """
    y, sr = load_audio_signal(audio_source, sr=sr)
    duration = float(len(y) / sr)

    # Voice Activity Detection (VAD) / speech boundary trimming:
    # Strips leading and trailing non-speech dead silence so the feature extractor
    # and sequence model analyze the actual vocal utterance, matching training conditions
    # and preventing silent/noisy edges from triggering false Sad/Fearful classifications.
    try:
        y_speech, _ = librosa.effects.trim(y, top_db=20)
        if len(y_speech) >= int(sr * 0.25):
            y = y_speech
    except Exception:
        pass

    # Ensure minimum audio length (at least 0.25 seconds)
    if len(y) < sr * 0.25:
        padding = int(sr * 0.25) - len(y)
        y = np.pad(y, (0, padding), mode="constant")

    # 1. MFCC Extraction (40 coefficients)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40, n_fft=1024, hop_length=512)
    mfcc_delta = librosa.feature.delta(mfcc)
    

    mfcc_mean = np.mean(mfcc, axis=1)
    mfcc_std = np.std(mfcc, axis=1)
    mfcc_delta_mean = np.mean(mfcc_delta, axis=1)
    mfcc_delta_std = np.std(mfcc_delta, axis=1)

    # 2. Chroma STFT (12 pitch classes)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr, n_fft=1024, hop_length=512)
    chroma_mean = np.mean(chroma, axis=1)
    chroma_std = np.std(chroma, axis=1)

    # 3. Mel-Spectrogram (128 bands aggregated to 16 bands)
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, n_fft=1024, hop_length=512)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    # Pool 128 mels into 16 bands
    mel_subbands = np.array([np.mean(mel_spec_db[i*8:(i+1)*8, :], axis=(0, 1)) for i in range(16)])

    # 4. Spectral Contrast (6 sub-bands + base = 7)
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr, n_fft=1024, hop_length=512)
    contrast_mean = np.mean(contrast, axis=1)
    contrast_std = np.std(contrast, axis=1)

    # 5. Tonnetz (Tonal Centroid Features using precomputed chroma for instant calculation)
    try:
        tonnetz = librosa.feature.tonnetz(chroma=chroma)
        tonnetz_mean = np.mean(tonnetz, axis=1)
    except Exception:
        tonnetz_mean = np.zeros(6, dtype=np.float32)

    # 6. Zero Crossing Rate (ZCR)
    zcr = librosa.feature.zero_crossing_rate(y=y, hop_length=512)
    zcr_mean = float(np.mean(zcr))
    zcr_std = float(np.std(zcr))

    # 7. Root-Mean-Square Energy (RMS)
    rms = librosa.feature.rms(y=y, hop_length=512)
    rms_mean = float(np.mean(rms))
    rms_std = float(np.std(rms))

    # 8. Spectral Centroid & Rolloff
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=512)
    centroid_mean = float(np.mean(centroid))
    centroid_std = float(np.std(centroid))

    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, hop_length=512)
    rolloff_mean = float(np.mean(rolloff))
    rolloff_std = float(np.std(rolloff))

    # 9. Fast Pitch (F0) Estimation via harmonic autocorrelation on voiced speech
    try:
        rms_vec = rms[0]
        if len(rms_vec) > 0 and np.max(rms_vec) > 1e-4:
            peak_frame = int(np.argmax(rms_vec))
            center = peak_frame * 512
            window_len = min(len(y), int(sr * 1.0))
            half_win = window_len // 2
            start = max(0, min(len(y) - window_len, center - half_win))
            y_pitch = y[start : start + window_len]
        else:
            y_pitch = y[:min(len(y), sr)]

        corr = np.correlate(y_pitch, y_pitch, mode="full")
        corr = corr[len(corr)//2:]
        min_lag = int(sr / 500)  # max 500 Hz
        max_lag = int(sr / 65)   # min 65 Hz
        peak = np.argmax(corr[min_lag:max_lag]) + min_lag
        pitch_mean = float(sr / peak) if peak > 0 else 150.0
        pitch_std = 25.0
    except Exception:
        pitch_mean = 150.0
        pitch_std = 25.0

    # 10. Tempo / Rhythm
    try:
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        tempo = float(librosa.feature.rhythm.tempo(onset_envelope=onset_env, sr=sr)[0])
    except Exception:
        tempo = 120.0

    # Build flat 1D feature vector (231 dimensions)
    flat_features = np.concatenate([
        mfcc_mean,          # 40
        mfcc_std,           # 40
        mfcc_delta_mean,    # 40
        mfcc_delta_std,     # 40
        chroma_mean,        # 12
        chroma_std,         # 12
        mel_subbands,       # 16
        contrast_mean,      # 7
        contrast_std,       # 7
        tonnetz_mean,       # 6
        [zcr_mean, zcr_std, rms_mean, rms_std, centroid_mean, centroid_std, rolloff_mean, rolloff_std, pitch_mean, pitch_std, tempo]  # 11
    ]).astype(np.float32)

    # Build 2D sequential frame matrix for CNN/LSTM (40 MFCC features x FIXED_FRAMES)
    # Represent the whole utterance with a fixed number of frames. Truncating
    # here discarded the end of longer recordings, while zero-padding shorter
    # ones gave the sequence model a duration-dependent block of artificial
    # silence. Interpolation preserves the temporal contour across the clip.
    time_steps = mfcc.shape[1]
    frame_positions = np.linspace(0, time_steps - 1, FIXED_FRAMES)
    source_positions = np.arange(time_steps)
    sequence_features = np.stack(
        [np.interp(frame_positions, source_positions, coefficient) for coefficient in mfcc]
    )
    sequence_features = sequence_features.astype(np.float32)

    # Build UI diagnostic summary
    summary = {
        "duration_seconds": round(duration, 3),
        "sample_rate": int(sr),
        "rms_energy_mean": round(rms_mean, 4),
        "zero_crossing_rate_mean": round(zcr_mean, 4),
        "spectral_centroid_mean": round(centroid_mean, 2),
        "spectral_rolloff_mean": round(rolloff_mean, 2),
        "pitch_f0_mean": round(pitch_mean, 2),
        "pitch_f0_std": round(pitch_std, 2),
        "tempo": round(tempo, 1),
        "mfcc_means": [round(float(v), 3) for v in mfcc_mean[:13]],
        "chroma_means": [round(float(v), 3) for v in chroma_mean],
    }

    return flat_features, sequence_features, summary


def extract_sequence_features_from_audio(
    audio_source: Union[str, bytes, io.BytesIO],
    sr: int = TARGET_SR,
) -> np.ndarray:
    """Return fixed-size MFCC sequences spanning the complete recording."""
    y, sr = load_audio_signal(audio_source, sr=sr)
    try:
        y_speech, _ = librosa.effects.trim(y, top_db=20)
        if len(y_speech) >= int(sr * 0.25):
            y = y_speech
    except Exception:
        pass
    if len(y) < sr * 0.25:
        y = np.pad(y, (0, int(sr * 0.25) - len(y)), mode="constant")
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40, n_fft=1024, hop_length=512)
    frame_positions = np.linspace(0, mfcc.shape[1] - 1, FIXED_FRAMES)
    source_positions = np.arange(mfcc.shape[1])
    return np.stack(
        [np.interp(frame_positions, source_positions, coefficient) for coefficient in mfcc]
    ).astype(np.float32)
