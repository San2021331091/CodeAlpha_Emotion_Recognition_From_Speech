import unittest
from unittest.mock import patch

from app.ml.speech_encoder import CACHE_DIRECTORY, MODEL_ID, REVISION, load_encoder


class EncoderCacheTests(unittest.TestCase):
    def tearDown(self):
        load_encoder.cache_clear()

    def test_missing_cache_can_be_repaired_and_retried_offline(self):
        for failed_component in ('processor', 'weights', 'checkpoint'):
            with self.subTest(component=failed_component):
                load_encoder.cache_clear()
                with patch('transformers.AutoFeatureExtractor.from_pretrained') as processor, \
                     patch('transformers.AutoModel.from_pretrained') as model, \
                     patch('huggingface_hub.hf_hub_download') as checkpoint:
                    failed_loader = {'processor': processor, 'weights': model, 'checkpoint': checkpoint}[failed_component]
                    cause = OSError('file missing')
                    failed_loader.side_effect = cause
                    with self.assertRaisesRegex(RuntimeError, 'python download_speech_encoder.py') as error:
                        load_encoder()
                    self.assertIs(error.exception.__cause__, cause)
                    self.assertIn(str(CACHE_DIRECTORY), str(error.exception))

                    failed_loader.side_effect = None
                    loaded = load_encoder()
                    self.assertEqual(loaded, (processor.return_value, model.return_value))
                    model.return_value.eval.assert_called_once_with()
                    expected = dict(revision=REVISION, cache_dir=str(CACHE_DIRECTORY), local_files_only=True)
                    processor.assert_called_with(MODEL_ID, **expected)
                    model.assert_called_with(MODEL_ID, **expected)
                    checkpoint.assert_called_with(MODEL_ID, 'pytorch_model.bin', **expected)
                    self.assertIs(load_encoder(), loaded)


if __name__ == '__main__':
    unittest.main()
