import asyncio
import hashlib
import io
import unittest
import numpy as np
import soundfile as sf
from fastapi import HTTPException, UploadFile
from app.api.routes import predict_audio_file


class AudioRegressionTests(unittest.TestCase):
    def predict(self, data):
        return asyncio.run(predict_audio_file(
            UploadFile(filename="test.wav", file=io.BytesIO(data)), None))

    def test_invalid_uploads(self):
        for data, status in [(b"", 400), (b"not audio", 400),
                             (b"x" * (25 * 1024 * 1024 + 1), 413)]:
            with self.subTest(status=status, length=len(data)):
                with self.assertRaises(HTTPException) as caught:
                    self.predict(data)
                self.assertEqual(caught.exception.status_code, status)

    def test_pcm_wav_inference(self):
        buffer = io.BytesIO()
        sr = 22050
        signal = 0.3 * np.sin(2 * np.pi * 220 * np.arange(sr) / sr)
        sf.write(buffer, signal, sr, format="WAV", subtype="PCM_16")
        result = self.predict(buffer.getvalue())
        self.assertTrue(result["top_emotion"])
        self.assertEqual(result["acoustic_summary"]["duration_seconds"], 1.0)
        self.assertEqual(result["input_sha256"], hashlib.sha256(buffer.getvalue()).hexdigest())

    def test_result_identifies_the_exact_audio_and_request(self):
        buffer = io.BytesIO()
        signal = 0.3 * np.sin(2 * np.pi * 220 * np.arange(22050) / 22050)
        sf.write(buffer, signal, 22050, format="WAV")
        data = buffer.getvalue()
        predictions = []
        for filename, request_id in (("mic_recording.wav", "recording-request"), ("uploaded.mp3", "upload-request")):
            result = asyncio.run(predict_audio_file(
                UploadFile(filename=filename, file=io.BytesIO(data)), None, request_id))
            self.assertEqual(result["request_id"], request_id)
            self.assertEqual(result["input_sha256"], hashlib.sha256(data).hexdigest())
            predictions.append(result)
        self.assertEqual(predictions[0]["probabilities"], predictions[1]["probabilities"])

    def test_silence_quiet_noise_and_too_short_clips_return_400(self):
        for signal in (np.zeros(22050), np.full(22050, 1e-5), np.ones(2205) * 0.1):
            with self.subTest(samples=len(signal), amplitude=float(signal.max())):
                buffer = io.BytesIO()
                sf.write(buffer, signal, 22050, format="WAV", subtype="FLOAT")
                with self.assertRaises(HTTPException) as caught:
                    self.predict(buffer.getvalue())
                self.assertEqual(caught.exception.status_code, 400)


if __name__ == "__main__":
    unittest.main()
