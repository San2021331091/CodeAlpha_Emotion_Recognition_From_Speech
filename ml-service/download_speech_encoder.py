"""Download the pinned, official self-supervised encoder used by the app."""
from transformers import AutoFeatureExtractor, AutoModel
from app.ml.speech_encoder import MODEL_ID, REVISION, CACHE_DIRECTORY


if __name__=='__main__':
    AutoFeatureExtractor.from_pretrained(MODEL_ID,revision=REVISION,cache_dir=str(CACHE_DIRECTORY))
    AutoModel.from_pretrained(MODEL_ID,revision=REVISION,cache_dir=str(CACHE_DIRECTORY))
    print('Speech encoder ready:',MODEL_ID,REVISION)
