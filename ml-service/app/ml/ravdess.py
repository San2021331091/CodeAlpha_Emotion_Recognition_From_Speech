"""Official RAVDESS speech-only dataset; no synthetic fallback."""
import hashlib
import json
from pathlib import Path
import shutil
import urllib.request
import zipfile
import soundfile as sf
from .data_generator import EMOTIONS

SOURCE = "https://zenodo.org/records/1188976"
ARCHIVE_MD5 = "bc696df654c87fed845eb13823edef8a"
TRAIN_ACTORS = tuple(range(1, 17))
VALIDATION_ACTORS = (17, 18, 19, 20)
TEST_ACTORS = (21, 22, 23, 24)


def prepare_dataset(data_dir):
    root = Path(data_dir) / "ravdess"
    root.mkdir(parents=True, exist_ok=True)
    files = sorted(root.glob("Actor_*/*.wav"))
    if len(files) != 1440:
        archive = root / "speech.zip"
        if not archive.exists():
            print("Downloading official RAVDESS speech archive...", flush=True)
            urllib.request.urlretrieve(SOURCE + "/files/Audio_Speech_Actors_01-24.zip?download=1", archive)
        if hashlib.md5(archive.read_bytes()).hexdigest() != ARCHIVE_MD5:
            raise ValueError("RAVDESS archive checksum mismatch; replace data/ravdess/speech.zip.")
        with zipfile.ZipFile(archive) as z:
            for member in z.infolist():
                target = (root / member.filename).resolve()
                if not target.is_relative_to(root.resolve()):
                    raise ValueError("Unsafe archive path")
            z.extractall(root)
        files = sorted(root.glob("Actor_*/*.wav"))
    if len(files) != 1440:
        raise ValueError(f"Expected 1440 RAVDESS speech files, found {len(files)}")
    records = []
    for path in files:
        modality, channel, emotion, intensity, statement, repetition, actor = map(int, path.stem.split("-"))
        if modality != 3 or channel != 1 or not 1 <= emotion <= 8 or not 1 <= actor <= 24:
            raise ValueError(f"Unexpected RAVDESS filename: {path.name}")
        records.append((str(path), emotion - 1, actor))
    return records


def create_benchmarks(data_dir, records, emotions=None):
    root = Path(data_dir)
    dest = root / "audio_samples" / "benchmark"
    dest.mkdir(parents=True, exist_ok=True)
    manifest = []
    if emotions is None:
        emotions = EMOTIONS
    labels = {emotion: label for label, emotion in enumerate(EMOTIONS)}
    for emotion in emotions:
        label = labels[emotion]
        path, _, actor = next(r for r in records if r[1] == label and r[2] in TEST_ACTORS)
        source = Path(path)
        filename = f"ravdess_{emotion}_{source.name}"
        shutil.copy2(source, dest / filename)
        manifest.append(dict(id=f"ravdess_{emotion}", title=f"RAVDESS: {emotion.title()}",
                             emotion=emotion, speaker=f"Actor {actor:02d}",
                             duration_seconds=round(sf.info(path).duration, 2),
                             description="Human speech from a held-out test actor. " + SOURCE,
                             filename=filename))
    (root / "benchmark_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
