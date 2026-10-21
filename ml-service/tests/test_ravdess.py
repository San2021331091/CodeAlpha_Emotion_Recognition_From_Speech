import unittest
import numpy as np
import torch
from app.ml.ravdess import prepare_dataset, TRAIN_ACTORS, VALIDATION_ACTORS, TEST_ACTORS
from app.ml.torch_models import SequenceNetwork, TorchSequenceModel


class RAVDESSRegressionTests(unittest.TestCase):
    def test_speakers_are_disjoint_and_all_emotions_present(self):
        records = prepare_dataset("data")
        splits = [set(TRAIN_ACTORS), set(VALIDATION_ACTORS), set(TEST_ACTORS)]
        self.assertEqual(len(records), 1440)
        self.assertEqual(set.union(*splits), set(range(1, 25)))
        for i, split in enumerate(splits):
            for other in splits[i + 1:]:
                self.assertFalse(split & other)
            self.assertEqual({label for _, label, actor in records if actor in split}, set(range(8)))

    def test_feature_layers_receive_gradients(self):
        torch.set_num_threads(2)
        for kind in ("cnn", "lstm", "crnn"):
            with self.subTest(kind=kind):
                net = SequenceNetwork(kind)
                loss = torch.nn.functional.cross_entropy(net(torch.randn(4, 40, 128)), torch.tensor([0, 1, 2, 3]))
                loss.backward()
                layers = [p for n, p in net.named_parameters() if "weight" in n and not n.startswith("head")]
                self.assertTrue(layers)
                self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0 for p in layers))

    def test_batch_training_normalizes_from_training_data_and_handles_partial_batch(self):
        rng = np.random.default_rng(7)
        x = rng.normal(size=(15, 40, 128)).astype(np.float32)
        y = np.arange(15) % 2
        model = TorchSequenceModel("cnn", num_classes=2).fit(
            x, y, validation_data=(x[:4], y[:4]), epochs=1, batch_size=7)
        np.testing.assert_allclose(model.mean, x.mean(axis=(0, 2), keepdims=True), atol=1e-6)
        np.testing.assert_allclose(model.std, x.std(axis=(0, 2), keepdims=True), atol=1e-6)
        probabilities = model.predict_proba(x[:4])
        self.assertEqual(probabilities.shape, (4, 2))
        np.testing.assert_allclose(probabilities.sum(1), 1, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
