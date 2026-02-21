from __future__ import annotations

import json
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np


class AdamOptimizer:
    def __init__(self, lr: float = 1e-3, beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8) -> None:
        self.lr = float(lr)
        self.beta1 = float(beta1)
        self.beta2 = float(beta2)
        self.eps = float(eps)

        self.m_w: List[np.ndarray] = []
        self.v_w: List[np.ndarray] = []
        self.m_b: List[np.ndarray] = []
        self.v_b: List[np.ndarray] = []
        self.t = 0

    def _ensure_state(self, weights: Sequence[np.ndarray], biases: Sequence[np.ndarray]) -> None:
        if len(self.m_w) == len(weights) and len(self.m_b) == len(biases):
            return
        self.m_w = [np.zeros_like(w) for w in weights]
        self.v_w = [np.zeros_like(w) for w in weights]
        self.m_b = [np.zeros_like(b) for b in biases]
        self.v_b = [np.zeros_like(b) for b in biases]

    def step(
        self,
        weights: Sequence[np.ndarray],
        biases: Sequence[np.ndarray],
        grad_w: Sequence[np.ndarray],
        grad_b: Sequence[np.ndarray],
    ) -> None:
        self._ensure_state(weights, biases)
        self.t += 1

        for i in range(len(weights)):
            self.m_w[i] = self.beta1 * self.m_w[i] + (1.0 - self.beta1) * grad_w[i]
            self.v_w[i] = self.beta2 * self.v_w[i] + (1.0 - self.beta2) * (grad_w[i] ** 2)
            self.m_b[i] = self.beta1 * self.m_b[i] + (1.0 - self.beta1) * grad_b[i]
            self.v_b[i] = self.beta2 * self.v_b[i] + (1.0 - self.beta2) * (grad_b[i] ** 2)

            m_w_hat = self.m_w[i] / (1.0 - self.beta1**self.t)
            v_w_hat = self.v_w[i] / (1.0 - self.beta2**self.t)
            m_b_hat = self.m_b[i] / (1.0 - self.beta1**self.t)
            v_b_hat = self.v_b[i] / (1.0 - self.beta2**self.t)

            weights[i] -= self.lr * m_w_hat / (np.sqrt(v_w_hat) + self.eps)
            biases[i] -= self.lr * m_b_hat / (np.sqrt(v_b_hat) + self.eps)


class MLPQNetwork:
    """From-scratch MLP Q-network with manual backprop, NumPy-accelerated."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_sizes: Tuple[int, ...] = (128, 128),
        seed: int | None = None,
    ) -> None:
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.hidden_sizes = tuple(int(v) for v in hidden_sizes)
        self.rng = np.random.default_rng(seed)

        layer_sizes = [self.input_dim, *self.hidden_sizes, self.output_dim]
        self.weights: List[np.ndarray] = []
        self.biases: List[np.ndarray] = []

        for in_dim, out_dim in zip(layer_sizes[:-1], layer_sizes[1:]):
            scale = np.sqrt(2.0 / in_dim)
            w = (self.rng.standard_normal((in_dim, out_dim), dtype=np.float32) * scale).astype(np.float32)
            b = np.zeros((1, out_dim), dtype=np.float32)
            self.weights.append(w)
            self.biases.append(b)

    @staticmethod
    def _as_batch(states: Sequence[Sequence[float]] | np.ndarray) -> np.ndarray:
        arr = np.asarray(states, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        return arr

    @staticmethod
    def _relu(x: np.ndarray) -> np.ndarray:
        return np.maximum(0.0, x)

    @staticmethod
    def _relu_grad(x: np.ndarray) -> np.ndarray:
        return (x > 0).astype(np.float32)

    def forward(self, states: Sequence[Sequence[float]] | np.ndarray) -> Tuple[np.ndarray, List[np.ndarray], List[np.ndarray]]:
        x = self._as_batch(states)
        activations = [x]
        pre_activations: List[np.ndarray] = []

        current = x
        for li, (w, b) in enumerate(zip(self.weights, self.biases)):
            z = current @ w + b
            pre_activations.append(z)
            if li == len(self.weights) - 1:
                current = z
            else:
                current = self._relu(z)
            activations.append(current)

        return current, activations, pre_activations

    def predict_one(self, state: Sequence[float] | np.ndarray) -> List[float]:
        q_values, _, _ = self.forward(np.asarray(state, dtype=np.float32).reshape(1, -1))
        return [float(v) for v in q_values[0].tolist()]

    def predict_batch(self, states: Sequence[Sequence[float]] | np.ndarray) -> List[List[float]]:
        q_values, _, _ = self.forward(states)
        return [[float(v) for v in row] for row in q_values.tolist()]

    def train_batch(
        self,
        states: Sequence[Sequence[float]] | np.ndarray,
        actions: Sequence[int] | np.ndarray,
        targets: Sequence[float] | np.ndarray,
        optimizer: AdamOptimizer,
    ) -> Tuple[float, float]:
        states_arr = self._as_batch(states)
        actions_arr = np.asarray(actions, dtype=np.int64)
        targets_arr = np.asarray(targets, dtype=np.float32)

        q_values, activations, pre_activations = self.forward(states_arr)
        batch_size = states_arr.shape[0]

        pred = q_values[np.arange(batch_size), actions_arr]
        td_errors = pred - targets_arr
        loss = float(0.5 * np.mean(td_errors**2))
        td_abs_mean = float(np.mean(np.abs(td_errors)))

        grad_output = np.zeros_like(q_values, dtype=np.float32)
        grad_output[np.arange(batch_size), actions_arr] = td_errors / batch_size

        grad_w = [np.zeros_like(w, dtype=np.float32) for w in self.weights]
        grad_b = [np.zeros_like(b, dtype=np.float32) for b in self.biases]

        grad = grad_output
        for li in reversed(range(len(self.weights))):
            a_prev = activations[li]
            grad_w[li] = a_prev.T @ grad
            grad_b[li] = np.sum(grad, axis=0, keepdims=True)

            if li > 0:
                grad = grad @ self.weights[li].T
                grad *= self._relu_grad(pre_activations[li - 1])

        optimizer.step(self.weights, self.biases, grad_w, grad_b)
        return loss, td_abs_mean

    def copy_from(self, other: "MLPQNetwork") -> None:
        if len(self.weights) != len(other.weights):
            raise ValueError("Network architectures do not match")
        for i in range(len(self.weights)):
            self.weights[i] = other.weights[i].copy()
            self.biases[i] = other.biases[i].copy()

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
            "hidden_sizes": list(self.hidden_sizes),
            "weights": [w.tolist() for w in self.weights],
            "biases": [b.tolist() for b in self.biases],
        }
        path.write_text(json.dumps(payload), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "MLPQNetwork":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        network = cls(
            input_dim=int(payload["input_dim"]),
            output_dim=int(payload["output_dim"]),
            hidden_sizes=tuple(int(v) for v in payload["hidden_sizes"]),
            seed=0,
        )

        for i, w in enumerate(payload["weights"]):
            network.weights[i] = np.asarray(w, dtype=np.float32)
        for i, b in enumerate(payload["biases"]):
            network.biases[i] = np.asarray(b, dtype=np.float32)

        return network
