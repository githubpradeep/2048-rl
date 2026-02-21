from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np


@dataclass
class AdamState:
    m_w: List[np.ndarray]
    v_w: List[np.ndarray]
    m_b: List[np.ndarray]
    v_b: List[np.ndarray]
    t: int = 0


class AdamOptimizer:
    def __init__(self, lr: float = 1e-3, beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8) -> None:
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.state: AdamState | None = None

    def _init_state(self, weights: Sequence[np.ndarray], biases: Sequence[np.ndarray]) -> None:
        self.state = AdamState(
            m_w=[np.zeros_like(w) for w in weights],
            v_w=[np.zeros_like(w) for w in weights],
            m_b=[np.zeros_like(b) for b in biases],
            v_b=[np.zeros_like(b) for b in biases],
        )

    def step(
        self,
        weights: List[np.ndarray],
        biases: List[np.ndarray],
        grad_w: List[np.ndarray],
        grad_b: List[np.ndarray],
    ) -> None:
        if self.state is None:
            self._init_state(weights, biases)

        assert self.state is not None
        self.state.t += 1

        for i in range(len(weights)):
            self.state.m_w[i] = self.beta1 * self.state.m_w[i] + (1.0 - self.beta1) * grad_w[i]
            self.state.v_w[i] = self.beta2 * self.state.v_w[i] + (1.0 - self.beta2) * (grad_w[i] ** 2)
            self.state.m_b[i] = self.beta1 * self.state.m_b[i] + (1.0 - self.beta1) * grad_b[i]
            self.state.v_b[i] = self.beta2 * self.state.v_b[i] + (1.0 - self.beta2) * (grad_b[i] ** 2)

            m_w_hat = self.state.m_w[i] / (1.0 - self.beta1 ** self.state.t)
            v_w_hat = self.state.v_w[i] / (1.0 - self.beta2 ** self.state.t)
            m_b_hat = self.state.m_b[i] / (1.0 - self.beta1 ** self.state.t)
            v_b_hat = self.state.v_b[i] / (1.0 - self.beta2 ** self.state.t)

            weights[i] -= self.lr * m_w_hat / (np.sqrt(v_w_hat) + self.eps)
            biases[i] -= self.lr * m_b_hat / (np.sqrt(v_b_hat) + self.eps)


class MLPQNetwork:
    """Simple MLP Q-network with manual backprop implemented in NumPy."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_sizes: Tuple[int, ...] = (128, 128),
        seed: int | None = None,
    ) -> None:
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_sizes = hidden_sizes
        self.rng = np.random.default_rng(seed)

        layer_sizes = [input_dim, *hidden_sizes, output_dim]
        self.weights: List[np.ndarray] = []
        self.biases: List[np.ndarray] = []

        for in_dim, out_dim in zip(layer_sizes[:-1], layer_sizes[1:]):
            scale = np.sqrt(2.0 / in_dim)
            w = (self.rng.standard_normal((in_dim, out_dim)) * scale).astype(np.float32)
            b = np.zeros((1, out_dim), dtype=np.float32)
            self.weights.append(w)
            self.biases.append(b)

    @staticmethod
    def _relu(x: np.ndarray) -> np.ndarray:
        return np.maximum(0.0, x)

    @staticmethod
    def _relu_grad(x: np.ndarray) -> np.ndarray:
        return (x > 0).astype(np.float32)

    def forward(self, x: np.ndarray) -> Tuple[np.ndarray, List[np.ndarray], List[np.ndarray]]:
        activations = [x]
        pre_activations: List[np.ndarray] = []

        current = x
        for i in range(len(self.weights)):
            z = current @ self.weights[i] + self.biases[i]
            pre_activations.append(z)
            if i == len(self.weights) - 1:
                current = z
            else:
                current = self._relu(z)
            activations.append(current)

        return current, activations, pre_activations

    def predict(self, x: np.ndarray) -> np.ndarray:
        q_values, _, _ = self.forward(x.astype(np.float32))
        return q_values

    def train_batch(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        targets: np.ndarray,
        optimizer: AdamOptimizer,
    ) -> Tuple[float, float]:
        states = states.astype(np.float32)
        actions = actions.astype(np.int64)
        targets = targets.astype(np.float32)

        q_values, activations, pre_activations = self.forward(states)
        batch_size = states.shape[0]

        pred = q_values[np.arange(batch_size), actions]
        td_errors = pred - targets
        loss = float(0.5 * np.mean(td_errors**2))
        td_abs_mean = float(np.mean(np.abs(td_errors)))

        grad_output = np.zeros_like(q_values, dtype=np.float32)
        grad_output[np.arange(batch_size), actions] = td_errors / batch_size

        grad_w = [np.zeros_like(w) for w in self.weights]
        grad_b = [np.zeros_like(b) for b in self.biases]

        grad = grad_output
        for layer_idx in reversed(range(len(self.weights))):
            a_prev = activations[layer_idx]
            grad_w[layer_idx] = a_prev.T @ grad
            grad_b[layer_idx] = np.sum(grad, axis=0, keepdims=True)

            if layer_idx > 0:
                grad = grad @ self.weights[layer_idx].T
                grad *= self._relu_grad(pre_activations[layer_idx - 1])

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

        arrays = {
            "input_dim": np.array([self.input_dim], dtype=np.int32),
            "output_dim": np.array([self.output_dim], dtype=np.int32),
            "hidden_sizes": np.array(self.hidden_sizes, dtype=np.int32),
        }
        for i, w in enumerate(self.weights):
            arrays[f"w_{i}"] = w
            arrays[f"b_{i}"] = self.biases[i]
        np.savez_compressed(path, **arrays)

    @classmethod
    def load(cls, path: str | Path) -> "MLPQNetwork":
        data = np.load(path)
        input_dim = int(data["input_dim"][0])
        output_dim = int(data["output_dim"][0])
        hidden_sizes = tuple(int(v) for v in data["hidden_sizes"])  # type: ignore[arg-type]

        network = cls(input_dim=input_dim, output_dim=output_dim, hidden_sizes=hidden_sizes)
        for i in range(len(network.weights)):
            network.weights[i] = data[f"w_{i}"].astype(np.float32)
            network.biases[i] = data[f"b_{i}"].astype(np.float32)
        return network
