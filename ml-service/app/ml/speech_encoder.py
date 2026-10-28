"""Frozen, self-supervised speech features and a separately trained emotion head."""
from functools import lru_cache
import io
from pathlib import Path
import numpy as np
import soundfile as sf

MODEL_ID = 'microsoft/wavlm-base-plus'
REVISION = '4c66d4806a428f2e922ccfa1a962776e232d487b'
LAYERS = (4, 8, 12)
CACHE_DIRECTORY = Path(__file__).resolve().parents[2] / '.model_cache'


@lru_cache(maxsize=1)
def load_encoder():
    import torch
    from transformers import AutoFeatureExtractor, AutoModel
    from huggingface_hub import hf_hub_download
    torch.set_num_threads(4)
    try:
        # This pinned revision uses PyTorch weights. Check their presence before
        # Transformers 4.57 can pass a missing checkpoint (None) to endswith().
        hf_hub_download(MODEL_ID, 'pytorch_model.bin', revision=REVISION,
                        cache_dir=str(CACHE_DIRECTORY), local_files_only=True)
        processor = AutoFeatureExtractor.from_pretrained(MODEL_ID, revision=REVISION, cache_dir=str(CACHE_DIRECTORY), local_files_only=True)
        model = AutoModel.from_pretrained(MODEL_ID, revision=REVISION, cache_dir=str(CACHE_DIRECTORY), local_files_only=True)
    except OSError as exc:
        raise RuntimeError(
            f"The local WavLM encoder cache is missing, incomplete, or unreadable at {CACHE_DIRECTORY}. "
            "From ml-service, run `python download_speech_encoder.py` with internet access, "
            "then retry inference. Keep the .model_cache directory for offline inference."
        ) from exc
    model.eval()
    return processor, model


def read_signal(source):
    import librosa
    if isinstance(source, tuple):
        signal, sr = source
    else:
        signal, sr = sf.read(io.BytesIO(source) if isinstance(source, bytes) else source, dtype='float32')
    if signal.ndim > 1:
        signal = signal.mean(1)
    if len(signal) < sr * .25 or not np.isfinite(signal).all() or np.abs(signal).max() <= 1e-6:
        raise ValueError('Speech encoder requires audible, finite audio of at least 0.25 seconds.')
    if sr != 16000:
        signal = librosa.resample(signal, orig_sr=sr, target_sr=16000)
    return signal


def encode_many(sources):
    import torch
    signals = [read_signal(source) for source in sources]
    buckets = [int(np.ceil(len(signal) / 16000)) * 16000 for signal in signals]
    if len(set(buckets)) != 1:
        raise ValueError('Batch audio must share a whole-second padding bucket.')
    processor, model = load_encoder()
    inputs = processor(signals, sampling_rate=16000, return_tensors='pt', padding='max_length',
                       max_length=buckets[0], return_attention_mask=True)
    with torch.inference_mode():
        hidden = model(**inputs, output_hidden_states=True).hidden_states
        lengths = model._get_feat_extract_output_lengths(torch.tensor([len(signal) for signal in signals]))
        output = []
        for i, length in enumerate(lengths):
            features = [torch.cat((hidden[layer][i,:length].mean(0), hidden[layer][i,:length].std(0, unbiased=False))).numpy()
                        for layer in LAYERS]
            output.append(np.stack(features))
    return np.stack(output).astype(np.float32)


def encode_audio(source):
    return encode_many([source])[0]


class WavLMAudioModel:
    audio_model = True

    def __init__(self, classifier, layers):
        self.classifier = classifier
        self.layers = tuple(layers)
        self.encoder_revision = REVISION

    def predict_audio_proba(self, audio_bytes):
        signal = read_signal(audio_bytes)
        # Bound transformer attention memory for long uploads. Original corpus
        # utterances stay intact; longer speech uses duration-weighted chunks.
        chunks = np.array_split(signal, int(np.ceil(len(signal)/(8*16000)))) if len(signal)>12*16000 else [signal]
        probabilities = []
        for chunk in chunks:
            features = encode_audio((chunk,16000))
            probabilities.append(self.classifier.predict_proba(features[list(self.layers)].reshape(1,-1))[0])
        return np.average(probabilities,axis=0,weights=[len(chunk) for chunk in chunks])[None,:]

    def predict_feature_proba(self, features):
        features = np.asarray(features)
        return self.classifier.predict_proba(features[:,list(self.layers)].reshape(len(features),-1))
