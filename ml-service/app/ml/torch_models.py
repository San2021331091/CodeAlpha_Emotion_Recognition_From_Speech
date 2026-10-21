"""End-to-end trainable MFCC sequence classifiers."""
import copy
import numpy as np
import torch
from torch import nn
from sklearn.metrics import f1_score


class SequenceNetwork(nn.Module):
    def __init__(self, kind, classes=8):
        super().__init__()
        self.kind = kind
        self.conv = nn.Sequential(
            nn.Conv1d(40, 64, 5, padding=2), nn.BatchNorm1d(64), nn.ReLU(),
            nn.MaxPool1d(2), nn.Dropout(0.2),
            nn.Conv1d(64, 64, 3, padding=1), nn.BatchNorm1d(64), nn.ReLU(),
            nn.MaxPool1d(2)) if kind != "lstm" else nn.Identity()
        self.rnn = nn.LSTM(40 if kind == "lstm" else 64, 48,
                           batch_first=True, bidirectional=True) if kind != "cnn" else None
        self.head = nn.Sequential(nn.Dropout(0.4), nn.Linear(96 if self.rnn else 64, classes))

    def forward(self, x):
        x = self.conv(x)
        if self.rnn is not None:
            x, _ = self.rnn(x.transpose(1, 2))
            x = x.mean(dim=1)
        else:
            x = x.mean(dim=2)
        return self.head(x)


class TorchSequenceModel:
    sequence_model = True

    def __init__(self, kind, num_classes=8):
        self.kind = kind
        self.num_classes = num_classes
        self.name = {"cnn": "1D-CNN (Temporal Convolutional Network)",
                     "lstm": "Bi-LSTM (Bidirectional Recurrent Network)",
                     "crnn": "CRNN (CNN + Bi-LSTM Hybrid)"}[kind]

    def fit(self, x, y, validation_data, epochs=60, sample_weights=None, batch_size=32,
            validation_sample_weights=None):
        torch.set_num_threads(4)
        torch.manual_seed(42)
        rng = np.random.default_rng(42)
        sums = np.zeros(x.shape[1], dtype=np.float64)
        squares = np.zeros_like(sums)
        for start in range(0, len(x), 512):
            chunk = np.asarray(x[start:start + 512], dtype=np.float64)
            sums += chunk.sum(axis=(0, 2))
            squares += np.square(chunk).sum(axis=(0, 2))
        count = len(x) * x.shape[2]
        mean = sums / count
        self.mean = mean.astype(np.float32)[None, :, None]
        self.std = np.sqrt(np.maximum(squares / count - mean ** 2, 1e-10)).astype(np.float32)[None, :, None]
        self.network = SequenceNetwork(self.kind, self.num_classes)
        # Normalize one batch at a time so a 50,000-clip memory-mapped dataset
        # does not require another full tensor copy in memory.
        yt = torch.from_numpy(y.astype(np.int64))
        if sample_weights is None:
            class_weights = len(y) / (self.num_classes * np.bincount(y, minlength=self.num_classes))
            loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float32))
            sample_weight_tensor = None
        else:
            sample_weight_tensor = torch.from_numpy(np.asarray(sample_weights, dtype=np.float32))
            if sample_weight_tensor.shape != (len(y),) or not torch.isfinite(sample_weight_tensor).all():
                raise ValueError("sample_weights must contain one finite weight per training example.")
            if torch.any(sample_weight_tensor <= 0):
                raise ValueError("sample_weights must be positive.")
            sample_weight_tensor = sample_weight_tensor / sample_weight_tensor.mean()
            loss_fn = nn.CrossEntropyLoss(reduction="none")
        optimizer = torch.optim.AdamW(self.network.parameters(), lr=0.001, weight_decay=0.01)
        best, stale, state = -1, 0, None
        for epoch in range(epochs):
            self.network.train()
            for batch in np.array_split(rng.permutation(len(x)), max(1, int(np.ceil(len(x) / batch_size)))):
                xb = torch.from_numpy(((x[batch] - self.mean) / self.std).astype(np.float32))
                # Small feature noise and time masking only on training examples.
                xb += torch.randn_like(xb) * 0.03
                start = int(rng.integers(0, xb.shape[2] - 8))
                xb[:, :, start:start + 8] = 0
                optimizer.zero_grad()
                batch_loss = loss_fn(self.network(xb), yt[batch])
                loss = (
                    (batch_loss * sample_weight_tensor[batch]).mean()
                    if sample_weight_tensor is not None
                    else batch_loss
                )
                loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), 5)
                optimizer.step()
            score = f1_score(validation_data[1], self.predict(validation_data[0]), average="macro",
                             sample_weight=validation_sample_weights)
            if score > best:
                best, stale, state = score, 0, copy.deepcopy(self.network.state_dict())
            else:
                stale += 1
            if (epoch + 1) % 5 == 0 or epochs <= 20:
                print(f"  {self.kind} epoch {epoch + 1}: validation F1={score:.3f}", flush=True)
            if stale >= 15:
                break
        self.network.load_state_dict(state)
        self.network.eval()
        return self

    def predict_proba(self, x):
        torch.set_num_threads(4)
        x = np.asarray(x, dtype=np.float32)
        if x.ndim == 2:
            x = x[None]
        xt = torch.from_numpy(((x - self.mean) / self.std).astype(np.float32))
        self.network.eval()
        with torch.no_grad():
            return np.concatenate([self.network(b).softmax(1).numpy() for b in xt.split(64)])

    def predict(self, x):
        return self.predict_proba(x).argmax(axis=1)
