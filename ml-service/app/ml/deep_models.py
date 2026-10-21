
import numpy as np
from typing import Dict, Any
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC


def softmax(x: np.ndarray) -> np.ndarray:
    """Stable softmax computation."""
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / np.sum(e_x, axis=-1, keepdims=True)


def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0, x)


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -15.0, 15.0)))


class Conv1DBlock:
    """1D Convolution + BatchNorm + ReLU + MaxPool1D."""
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 5):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        limit = np.sqrt(6.0 / (in_channels * kernel_size + out_channels))
        self.W = np.random.uniform(-limit, limit, (out_channels, in_channels, kernel_size)).astype(np.float32)
        self.b = np.zeros(out_channels, dtype=np.float32)
        self.gamma = np.ones(out_channels, dtype=np.float32)
        self.beta = np.zeros(out_channels, dtype=np.float32)

    def forward(self, x: np.ndarray) -> np.ndarray:
        # x: (B, C_in, L)
        B, L = x.shape
        pad = self.kernel_size // 2
        x_pad = np.pad(x, ((0, 0), (0, 0), (pad, pad)), mode="constant")
        
        # Convolve
        out = np.zeros((B, self.out_channels, L), dtype=np.float32)
        for oc in range(self.out_channels):
            conv_filter = self.W[oc]  # (C_in, K)
            for k in range(self.kernel_size):
                out[:, oc, :] += np.sum(x_pad[:, :, k:k+L] * conv_filter[:, k:k+1], axis=1)
            out[:, oc, :] += self.b[oc]

        # BatchNorm (eval mode)
        mean = np.mean(out, axis=(0, 2), keepdims=True)
        var = np.var(out, axis=(0, 2), keepdims=True) + 1e-5
        out = (out - mean) / np.sqrt(var) * self.gamma.reshape(1, -1, 1) + self.beta.reshape(1, -1, 1)

        # ReLU
        out = relu(out)

        # MaxPool 1D with stride 2
        L_pool = L // 2
        out_pool = np.zeros((B, self.out_channels, L_pool), dtype=np.float32)
        for i in range(L_pool):
            out_pool[:, :, i] = np.maximum(out[:, :, 2*i], out[:, :, 2*i + 1])

        return out_pool


class DeepCNN1DModel:
    """
    1D-CNN architecture for Speech Emotion Recognition.
    Processes acoustic temporal sequences (B, C_in=40, L=128).
    Applies hierarchical temporal convolutions to detect prosodic and spectral cues.
    """
    def __init__(self, in_channels: int = 40, num_classes: int = 8, hidden_dim: int = 64):
        self.name = "1D-CNN (Temporal Convolutional Network)"
        self.num_classes = num_classes
        self.block1 = Conv1DBlock(in_channels, 64, kernel_size=5)
        self.block2 = Conv1DBlock(64, 128, kernel_size=5)
        self.block3 = Conv1DBlock(128, hidden_dim, kernel_size=3)
        
        # Dense classification head
        limit = np.sqrt(6.0 / (hidden_dim + num_classes))
        self.fc_W = np.random.uniform(-limit, limit, (hidden_dim, num_classes)).astype(np.float32)
        self.fc_b = np.zeros(num_classes, dtype=np.float32)

    def extract_latent_features(self, x: np.ndarray) -> np.ndarray:
        if x.ndim == 2:
            x = np.expand_dims(x, 0)
        h = self.block1.forward(x)
        h = self.block2.forward(h)
        h = self.block3.forward(h)
        # Global Average Pooling across time frames
        pooled = np.mean(h, axis=2)  # (B, hidden_dim)
        return pooled

    def fit(self, X_seq: np.ndarray, y: np.ndarray, lr: float = 0.005, epochs: int = 40, batch_size: int = 32):
        N = X_seq.shape[0]
        num_classes = self.num_classes
        y_one_hot = np.zeros((N, num_classes), dtype=np.float32)
        y_one_hot[np.arange(N), y] = 1.0

        # Pre-extract pooled representations
        features = self.extract_latent_features(X_seq)  # (N, hidden_dim)
        
        # Adam optimizer parameters for classification head
        m_W, v_W = np.zeros_like(self.fc_W), np.zeros_like(self.fc_W)
        m_b, v_b = np.zeros_like(self.fc_b), np.zeros_like(self.fc_b)
        beta1, beta2, eps = 0.9, 0.999, 1e-8

        for epoch in range(1, epochs + 1):
            indices = np.random.permutation(N)
            for start in range(0, N, batch_size):
                batch_idx = indices[start:start + batch_size]
                xb = features[batch_idx]
                yb = y_one_hot[batch_idx]

                logits = np.dot(xb, self.fc_W) + self.fc_b
                probs = softmax(logits)

                # Gradient
                grad_logits = (probs - yb) / len(batch_idx)
                grad_W = np.dot(xb.T, grad_logits) + 0.001 * self.fc_W
                grad_b = np.sum(grad_logits, axis=0)

                # Adam update
                m_W = beta1 * m_W + (1 - beta1) * grad_W
                v_W = beta2 * v_W + (1 - beta2) * (grad_W ** 2)
                m_b = beta1 * m_b + (1 - beta1) * grad_b
                v_b = beta2 * v_b + (1 - beta2) * (grad_b ** 2)

                m_W_hat = m_W / (1 - beta1 ** epoch)
                v_W_hat = v_W / (1 - beta2 ** epoch)
                m_b_hat = m_b / (1 - beta1 ** epoch)
                v_b_hat = v_b / (1 - beta2 ** epoch)

                self.fc_W -= lr * m_W_hat / (np.sqrt(v_W_hat) + eps)
                self.fc_b -= lr * m_b_hat / (np.sqrt(v_b_hat) + eps)

    def predict_proba(self, X_seq: np.ndarray) -> np.ndarray:
        if X_seq.ndim == 2:
            X_seq = np.expand_dims(X_seq, 0)
        features = self.extract_latent_features(X_seq)
        logits = np.dot(features, self.fc_W) + self.fc_b
        return softmax(logits)

    def predict(self, X_seq: np.ndarray) -> np.ndarray:
        probs = self.predict_proba(X_seq)
        return np.argmax(probs, axis=1)


class BiLSTMModel:
    """
    Bidirectional LSTM with Temporal Attention for Speech Emotion Recognition.
    Processes sequential frame vectors (B, T, D=input_dim) over time.
    """
    def __init__(self, input_dim: int = 40, hidden_dim: int = 64, num_classes: int = 8):
        self.name = "Bi-LSTM (Bidirectional Recurrent Network)"
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes

        # Weights for forward and backward passes
        limit = np.sqrt(6.0 / (input_dim + hidden_dim))
        self.W_fwd = np.random.uniform(-limit, limit, (4 * hidden_dim, input_dim)).astype(np.float32)
        self.U_fwd = np.random.uniform(-limit, limit, (4 * hidden_dim, hidden_dim)).astype(np.float32)
        self.b_fwd = np.zeros(4 * hidden_dim, dtype=np.float32)
        self.b_fwd[hidden_dim:2*hidden_dim] = 1.0

        self.W_bwd = np.random.uniform(-limit, limit, (4 * hidden_dim, input_dim)).astype(np.float32)
        self.U_bwd = np.random.uniform(-limit, limit, (4 * hidden_dim, hidden_dim)).astype(np.float32)
        self.b_bwd = np.zeros(4 * hidden_dim, dtype=np.float32)
        self.b_bwd[hidden_dim:2*hidden_dim] = 1.0

        # Temporal Attention projection
        self.att_w = np.random.uniform(-0.1, 0.1, (2 * hidden_dim, 1)).astype(np.float32)

        # Classification Head
        limit_fc = np.sqrt(6.0 / (2 * hidden_dim + num_classes))
        self.fc_W = np.random.uniform(-limit_fc, limit_fc, (2 * hidden_dim, num_classes)).astype(np.float32)
        self.fc_b = np.zeros(num_classes, dtype=np.float32)

    def _step_lstm(self, x_t: np.ndarray, h_prev: np.ndarray, c_prev: np.ndarray, W: np.ndarray, U: np.ndarray, b: np.ndarray):
        gates = np.dot(x_t, W.T) + np.dot(h_prev, U.T) + b
        H = self.hidden_dim
        i_t = sigmoid(gates[:, :H])
        f_t = sigmoid(gates[:, H:2*H])
        c_tilde = np.tanh(gates[:, 2*H:3*H])
        o_t = sigmoid(gates[:, 3*H:4*H])
        c_t = f_t * c_prev + i_t * c_tilde
        h_t = o_t * np.tanh(c_t)
        return h_t, c_t

    def forward_sequence(self, x: np.ndarray) -> np.ndarray:
        if x.ndim == 2:
            x = np.expand_dims(x, 0)
        # If input shape is (B, C, T) where C == input_dim, transpose to (B, T, C)
        if x.shape[1] == self.input_dim and x.shape[2] != self.input_dim:
            x = np.transpose(x, (0, 2, 1))

        B, T = x.shape
        H = self.hidden_dim

        # Forward pass
        h_f = np.zeros((B, H), dtype=np.float32)
        c_f = np.zeros((B, H), dtype=np.float32)
        h_fwd_seq = []
        for t in range(T):
            h_f, c_f = self._step_lstm(x[:, t, :], h_f, c_f, self.W_fwd, self.U_fwd, self.b_fwd)
            h_fwd_seq.append(h_f)

        # Backward pass
        h_b = np.zeros((B, H), dtype=np.float32)
        c_b = np.zeros((B, H), dtype=np.float32)
        h_bwd_seq = []
        for t in reversed(range(T)):
            h_b, c_b = self._step_lstm(x[:, t, :], h_b, c_b, self.W_bwd, self.U_bwd, self.b_bwd)
            h_bwd_seq.insert(0, h_b)

        # Concatenate forward and backward hidden states
        h_seq = [np.concatenate([hf, hb], axis=1) for hf, hb in zip(h_fwd_seq, h_bwd_seq)]
        h_stack = np.stack(h_seq, axis=1)  # (B, T, 2*H)

        # Temporal Attention Pooling
        att_scores = np.dot(h_stack, self.att_w).squeeze(-1)  # (B, T)
        att_weights = softmax(att_scores)  # (B, T)
        context = np.sum(h_stack * np.expand_dims(att_weights, -1), axis=1)  # (B, 2*H)

        return context

    def fit(self, X_seq: np.ndarray, y: np.ndarray, lr: float = 0.005, epochs: int = 40, batch_size: int = 32):
        N = X_seq.shape[0]
        y_one_hot = np.zeros((N, self.num_classes), dtype=np.float32)
        y_one_hot[np.arange(N), y] = 1.0

        features = self.forward_sequence(X_seq)

        m_W, v_W = np.zeros_like(self.fc_W), np.zeros_like(self.fc_W)
        m_b, v_b = np.zeros_like(self.fc_b), np.zeros_like(self.fc_b)
        beta1, beta2, eps = 0.9, 0.999, 1e-8

        for epoch in range(1, epochs + 1):
            indices = np.random.permutation(N)
            for start in range(0, N, batch_size):
                batch_idx = indices[start:start + batch_size]
                xb = features[batch_idx]
                yb = y_one_hot[batch_idx]

                logits = np.dot(xb, self.fc_W) + self.fc_b
                probs = softmax(logits)

                grad_logits = (probs - yb) / len(batch_idx)
                grad_W = np.dot(xb.T, grad_logits) + 0.001 * self.fc_W
                grad_b = np.sum(grad_logits, axis=0)

                m_W = beta1 * m_W + (1 - beta1) * grad_W
                v_W = beta2 * v_W + (1 - beta2) * (grad_W ** 2)
                m_b = beta1 * m_b + (1 - beta1) * grad_b
                v_b = beta2 * v_b + (1 - beta2) * (grad_b ** 2)

                m_W_hat = m_W / (1 - beta1 ** epoch)
                v_W_hat = v_W / (1 - beta2 ** epoch)
                m_b_hat = m_b / (1 - beta1 ** epoch)
                v_b_hat = v_b / (1 - beta2 ** epoch)

                self.fc_W -= lr * m_W_hat / (np.sqrt(v_W_hat) + eps)
                self.fc_b -= lr * m_b_hat / (np.sqrt(v_b_hat) + eps)

    def predict_proba(self, X_seq: np.ndarray) -> np.ndarray:
        features = self.forward_sequence(X_seq)
        logits = np.dot(features, self.fc_W) + self.fc_b
        return softmax(logits)

    def predict(self, X_seq: np.ndarray) -> np.ndarray:
        probs = self.predict_proba(X_seq)
        return np.argmax(probs, axis=1)


class CRNNHybridModel:
    """
    CRNN (CNN + Bi-LSTM Hybrid):
    Uses 1D-CNN temporal convolution layers to extract low-level acoustic/timbral features,
    and feeds the resulting representations into Bi-LSTM for long-term prosodic modeling.
    """
    def __init__(self, in_channels: int = 40, num_classes: int = 8):
        self.name = "CRNN (CNN + Bi-LSTM Hybrid)"
        self.num_classes = num_classes
        self.cnn_block1 = Conv1DBlock(in_channels, 64, kernel_size=5)
        self.cnn_block2 = Conv1DBlock(64, 64, kernel_size=3)
        self.lstm = BiLSTMModel(input_dim=64, hidden_dim=64, num_classes=num_classes)

    def fit(self, X_seq: np.ndarray, y: np.ndarray, epochs: int = 45):
        if X_seq.ndim == 2:
            X_seq = np.expand_dims(X_seq, 0)
        # Apply CNN blocks
        conv1 = self.cnn_block1.forward(X_seq)
        conv2 = self.cnn_block2.forward(conv1)  # (B, 64, L_reduced)
        self.lstm.fit(conv2, y, epochs=epochs)

    def predict_proba(self, X_seq: np.ndarray) -> np.ndarray:
        if X_seq.ndim == 2:
            X_seq = np.expand_dims(X_seq, 0)
        conv1 = self.cnn_block1.forward(X_seq)
        conv2 = self.cnn_block2.forward(conv1)
        return self.lstm.predict_proba(conv2)

    def predict(self, X_seq: np.ndarray) -> np.ndarray:
        probs = self.predict_proba(X_seq)
        return np.argmax(probs, axis=1)


def build_models(input_dim_flat: int = 180, num_classes: int = 8) -> Dict[str, Any]:
    """
    Builds the model portfolio:
    1. CRNN (CNN + Bi-LSTM Hybrid)
    2. 1D-CNN (Convolutional Neural Network)
    3. Bi-LSTM (Recurrent Neural Network)
    4. Deep MLP (Multi-Layer Perceptron)
    5. Random Forest Classifier
    6. Support Vector Classifier (SVM)
    """
    from .torch_models import TorchSequenceModel
    return {
        "crnn": TorchSequenceModel("crnn", num_classes),
        "cnn": TorchSequenceModel("cnn", num_classes),
        "lstm": TorchSequenceModel("lstm", num_classes),
        "mlp": MLPClassifier(
            hidden_layer_sizes=(256, 128, 64),
            activation="relu",
            solver="adam",
            max_iter=350,
            early_stopping=True,
            random_state=42
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=150,
            max_depth=16,
            random_state=42,
            n_jobs=-1
        ),
        "svm": SVC(
            kernel="rbf",
            probability=True,
            C=3.0,
            gamma="scale",
            random_state=42
        )
    }
