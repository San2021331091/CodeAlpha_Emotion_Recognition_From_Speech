"""Verify live gateway input identity and recording/upload equivalence."""
from collections import Counter
import hashlib
import uuid

import httpx


def main():
    with httpx.Client(timeout=30) as client:
        response = client.get("http://localhost:3001/speech-emotion/samples")
        response.raise_for_status()
        samples = response.json()
        if not samples:
            raise RuntimeError("No benchmark samples returned by the running app.")
        counts = Counter()
        matched = 0
        for sample in samples:
            audio_response = client.get("http://localhost:3001/speech-emotion/sample/" + sample["id"])
            audio_response.raise_for_status()
            audio = audio_response.content
            previous = None
            # The content remains the same WAV. Vary the upload filename to
            # prove neither its extension nor the recording prefix sets a label.
            for filename in ("mic_recording.wav", "uploaded.mp3"):
                request_id = str(uuid.uuid4())
                response = client.post("http://localhost:3001/speech-emotion/predict",
                    files={"file": (filename, audio, "audio/wav")},
                    data={"modelName": "best", "requestId": request_id})
                response.raise_for_status()
                prediction = response.json()
                assert prediction["request_id"] == request_id
                assert prediction["input_sha256"] == hashlib.sha256(audio).hexdigest()
                if previous is None:
                    counts[prediction["top_emotion"]] += 1
                    previous = prediction["probabilities"]
                else:
                    assert previous == prediction["probabilities"]
                    matched += 1
        assert len(counts) >= 3, f"Model output may have collapsed: {dict(counts)}"
        print("Matched recording/upload results:", matched)
        print("Predicted emotions:", dict(counts))
        print("Frontend HTTP:", client.get("http://localhost:3000").status_code)


if __name__ == "__main__":
    main()
