"""Synthetic RGB classification, exact output permutation and numerical controls."""
from dataclasses import dataclass
import numpy as np

CLASS_NAMES = (
    "red horizontal stripe",
    "green vertical stripe",
    "blue descending diagonal",
    "yellow central square",
)
IMAGE_SIDE = 8


def make_prototype(label: int) -> np.ndarray:
    """HWC image with an interpretable, deliberately simple visual concept."""
    x = np.full((IMAGE_SIDE, IMAGE_SIDE, 3), 0.06, dtype=np.float64)
    if label == 0:
        x[3:5, :, 0] = 0.92
    elif label == 1:
        x[:, 3:5, 1] = 0.92
    elif label == 2:
        for i in range(IMAGE_SIDE):
            x[i, i, 2] = 0.98
    elif label == 3:
        x[2:6, 2:6, 0] = 0.91
        x[2:6, 2:6, 1] = 0.91
    else:
        raise ValueError("label must be between 0 and 3")
    return x


def dataset(n_per_class: int, seed: int):
    rng = np.random.default_rng(seed)
    xs, ys = [], []
    for label in range(len(CLASS_NAMES)):
        for _ in range(n_per_class):
            patch = make_prototype(label)
            noisy = np.clip(patch + rng.normal(0, 0.07, patch.shape), 0, 1)
            xs.append(noisy.reshape(-1))
            ys.append(label)
    return np.asarray(xs), np.asarray(ys, dtype=np.int64)


@dataclass(frozen=True)
class LinearClassifier:
    weight: np.ndarray  # (192, 4)
    bias: np.ndarray  # (4,)
    class_names: tuple = CLASS_NAMES

    def logits(self, x: np.ndarray) -> np.ndarray:
        return x @ self.weight + self.bias

    def predict(self, x: np.ndarray) -> np.ndarray:
        return np.argmax(self.logits(x), axis=-1)

    def accuracy(self, x: np.ndarray, y: np.ndarray) -> float:
        return float(np.mean(self.predict(x) == y))

    def permute_outputs(self, new_to_old):
        """Return a functionally equivalent classifier with permuted output coordinates.

        The *semantic* output label map must be permuted along with columns.  The
        function's vector-valued coordinates change but class predictions retain
        their meaning after the inverse label remapping.
        """
        order = np.asarray(new_to_old, dtype=np.int64)
        if sorted(order.tolist()) != list(range(len(self.class_names))):
            raise ValueError("new_to_old must be a permutation of the classes")
        return LinearClassifier(
            self.weight[:, order].copy(),
            self.bias[order].copy(),
            tuple(self.class_names[i] for i in order),
        )

    def shuffle_weights(self, seed: int):
        """Destroy the spatial associations while preserving each output's weight histogram."""
        rng = np.random.default_rng(seed)
        w = self.weight.copy()
        for j in range(w.shape[1]):
            rng.shuffle(w[:, j])
        return LinearClassifier(w, self.bias.copy(), self.class_names)


def train_classifier(seed: int = 7, epochs: int = 350, lr: float = 1.2):
    x_train, y_train = dataset(240, seed)
    rng = np.random.default_rng(seed + 1)
    w = rng.normal(0, 0.008, size=(x_train.shape[1], len(CLASS_NAMES)))
    b = np.zeros(len(CLASS_NAMES), dtype=np.float64)
    target = np.eye(len(CLASS_NAMES))[y_train]
    for _ in range(epochs):
        logits = x_train @ w + b
        e = np.exp(logits - logits.max(axis=1, keepdims=True))
        p = e / e.sum(axis=1, keepdims=True)
        delta = (p - target) / len(y_train)
        w -= lr * (x_train.T @ delta)
        b -= lr * delta.sum(axis=0)
    return LinearClassifier(w, b)


def map_labels_to_new_indices(y_old, new_to_old):
    inv = np.argsort(np.asarray(new_to_old, dtype=int))
    return inv[np.asarray(y_old, dtype=int)]


def tiny_forward_exercise():
    """Deterministic, exact two-layer ReLU tracing exercise; not a trained model."""
    x = np.array([1.0, -2.0])
    w1 = np.array([[1., -1., 2.], [0.5, 2., -1.]])
    b1 = np.array([0.5, -0.25, 0.0])
    w2 = np.array([[1., -2.], [-1., 0.5], [2., 1.]])
    b2 = np.array([0.1, -0.2])
    z1 = x @ w1 + b1
    h1 = np.maximum(0.0, z1)
    logits = h1 @ w2 + b2
    return {"x": x.tolist(), "w1": w1.tolist(), "b1": b1.tolist(),
            "w2": w2.tolist(), "b2": b2.tolist(),
            "hidden_preactivation": z1.tolist(), "hidden_activation": h1.tolist(),
            "output_logits": logits.tolist()}


def hidden_unit_symmetry():
    """Permute internal ReLU units and compensate in the next layer: same function."""
    e = tiny_forward_exercise()
    x = np.array(e['x'])
    w1 = np.array(e['w1']); b1 = np.array(e['b1'])
    w2 = np.array(e['w2']); b2 = np.array(e['b2'])
    order = np.array([2, 0, 1])
    w1_new = w1[:, order]
    b1_new = b1[order]
    w2_new = w2[order, :]
    h_new = np.maximum(0, x @ w1_new + b1_new)
    logits_new = h_new @ w2_new + b2
    assert np.allclose(logits_new, e['output_logits'])
    return {"order_new_to_old":order.tolist(), "w1_new":w1_new.tolist(),
            "b1_new":b1_new.tolist(), "w2_new":w2_new.tolist(),
            "b2":b2.tolist(), "input":x.tolist(),
            "unchanged_output_logits":logits_new.tolist()}
