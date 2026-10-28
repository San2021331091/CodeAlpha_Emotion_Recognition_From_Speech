"""Download the pinned, official self-supervised encoder used by the app."""
from transformers import AutoFeatureExtractor, AutoModel
from app.ml.speech_encoder import MODEL_ID, REVISION, CACHE_DIRECTORY, load_encoder


if __name__=='__main__':
    try:
        load_encoder()
    except RuntimeError as exc:
        if not isinstance(exc.__cause__, OSError):
            raise
        print('Downloading missing speech encoder files to:', CACHE_DIRECTORY, flush=True)
        AutoFeatureExtractor.from_pretrained(MODEL_ID,revision=REVISION,cache_dir=str(CACHE_DIRECTORY))
        AutoModel.from_pretrained(MODEL_ID,revision=REVISION,cache_dir=str(CACHE_DIRECTORY))
        load_encoder()
    print('Speech encoder ready:',MODEL_ID,REVISION)
