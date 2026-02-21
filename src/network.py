from __future__ import annotations

import json
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np


class AdamOptimizer:
    def __init__(
        self,
        lr: float = 1e-3,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
        max_grad_norm: float | None = None,
    ) -> None:
        self.lr = float(lr)
        self.beta1 = float(beta1)
        self.beta2 = float(beta2)
        self.eps = float(eps)
        self.max_grad_norm = None if max_grad_norm is None else float(max_grad_norm)

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

        if self.max_grad_norm is not None and self.max_grad_norm > 0.0:
            global_sq = 0.0
            for g in grad_w:
                global_sq += float(np.sum(g * g))
            for g in grad_b:
                global_sq += float(np.sum(g * g))
            global_norm = float(np.sqrt(global_sq))
            if global_norm > self.max_grad_norm:
                scale = self.max_grad_norm / (global_norm + 1e-12)
                for i in range(len(grad_w)):
                    grad_w[i] = grad_w[i] * scale
                for i in range(len(grad_b)):
                    grad_b[i] = grad_b[i] * scale

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
        dueling: bool = False,
    ) -> None:
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.hidden_sizes = tuple(int(v) for v in hidden_sizes)
        self.dueling = bool(dueling)
        self.rng = np.random.default_rng(seed)

        # For non-dueling: includes output layer.
        # For dueling: this is the shared trunk only.
        self.weights: List[np.ndarray] = []
        self.biases: List[np.ndarray] = []

        if not self.dueling:
            layer_sizes = [self.input_dim, *self.hidden_sizes, self.output_dim]
            for in_dim, out_dim in zip(layer_sizes[:-1], layer_sizes[1:]):
                scale = np.sqrt(2.0 / in_dim)
                w = (self.rng.standard_normal((in_dim, out_dim), dtype=np.float32) * scale).astype(np.float32)
                b = np.zeros((1, out_dim), dtype=np.float32)
                self.weights.append(w)
                self.biases.append(b)

            self.value_w = None
            self.value_b = None
            self.advantage_w = None
            self.advantage_b = None
        else:
            trunk_sizes = [self.input_dim, *self.hidden_sizes]
            for in_dim, out_dim in zip(trunk_sizes[:-1], trunk_sizes[1:]):
                scale = np.sqrt(2.0 / in_dim)
                w = (self.rng.standard_normal((in_dim, out_dim), dtype=np.float32) * scale).astype(np.float32)
                b = np.zeros((1, out_dim), dtype=np.float32)
                self.weights.append(w)
                self.biases.append(b)

            feature_dim = self.hidden_sizes[-1] if self.hidden_sizes else self.input_dim
            value_scale = np.sqrt(2.0 / feature_dim)
            self.value_w = (self.rng.standard_normal((feature_dim, 1), dtype=np.float32) * value_scale).astype(np.float32)
            self.value_b = np.zeros((1, 1), dtype=np.float32)
            self.advantage_w = (
                self.rng.standard_normal((feature_dim, self.output_dim), dtype=np.float32) * value_scale
            ).astype(np.float32)
            self.advantage_b = np.zeros((1, self.output_dim), dtype=np.float32)

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

    def _forward_non_dueling(
        self,
        states_arr: np.ndarray,
    ) -> tuple[np.ndarray, dict]:
        activations = [states_arr]
        pre_activations: List[np.ndarray] = []

        current = states_arr
        for li, (w, b) in enumerate(zip(self.weights, self.biases)):
            z = current @ w + b
            pre_activations.append(z)
            if li == len(self.weights) - 1:
                current = z
            else:
                current = self._relu(z)
            activations.append(current)

        cache = {
            "kind": "non_dueling",
            "activations": activations,
            "pre_activations": pre_activations,
        }
        return current, cache

    def _forward_dueling(
        self,
        states_arr: np.ndarray,
    ) -> tuple[np.ndarray, dict]:
        trunk_activations = [states_arr]
        trunk_pre_activations: List[np.ndarray] = []

        current = states_arr
        for w, b in zip(self.weights, self.biases):
            z = current @ w + b
            trunk_pre_activations.append(z)
            current = self._relu(z)
            trunk_activations.append(current)

        features = current
        if len(self.weights) == 0:
            features = states_arr

        assert self.value_w is not None and self.value_b is not None
        assert self.advantage_w is not None and self.advantage_b is not None

        value = features @ self.value_w + self.value_b
        advantage = features @ self.advantage_w + self.advantage_b
        advantage_mean = np.mean(advantage, axis=1, keepdims=True)
        q_values = value + (advantage - advantage_mean)

        cache = {
            "kind": "dueling",
            "trunk_activations": trunk_activations,
            "trunk_pre_activations": trunk_pre_activations,
            "features": features,
            "value": value,
            "advantage": advantage,
        }
        return q_values, cache

    def forward(self, states: Sequence[Sequence[float]] | np.ndarray) -> tuple[np.ndarray, dict]:
        states_arr = self._as_batch(states)
        if self.dueling:
            return self._forward_dueling(states_arr)
        return self._forward_non_dueling(states_arr)

    def predict_one(self, state: Sequence[float] | np.ndarray) -> List[float]:
        q_values, _ = self.forward(np.asarray(state, dtype=np.float32).reshape(1, -1))
        return [float(v) for v in q_values[0].tolist()]

    def predict_batch(self, states: Sequence[Sequence[float]] | np.ndarray) -> List[List[float]]:
        q_values, _ = self.forward(states)
        return [[float(v) for v in row] for row in q_values.tolist()]

    def _parameter_lists(self) -> tuple[List[np.ndarray], List[np.ndarray]]:
        weight_list = [w for w in self.weights]
        bias_list = [b for b in self.biases]

        if self.dueling:
            assert self.value_w is not None and self.value_b is not None
            assert self.advantage_w is not None and self.advantage_b is not None
            weight_list.extend([self.value_w, self.advantage_w])
            bias_list.extend([self.value_b, self.advantage_b])

        return weight_list, bias_list

    def _train_batch_non_dueling(
        self,
        q_values: np.ndarray,
        cache: dict,
        actions_arr: np.ndarray,
        targets_arr: np.ndarray,
        optimizer: AdamOptimizer,
        loss: str,
        huber_delta: float,
    ) -> tuple[float, float]:
        activations: List[np.ndarray] = cache["activations"]
        pre_activations: List[np.ndarray] = cache["pre_activations"]
        batch_size = q_values.shape[0]

        pred = q_values[np.arange(batch_size), actions_arr]
        td_errors = pred - targets_arr
        td_abs = np.abs(td_errors)
        td_abs_mean = float(np.mean(td_abs))
        if loss == "huber":
            delta = float(huber_delta)
            quadratic = td_abs <= delta
            loss_val = np.where(
                quadratic,
                0.5 * (td_errors**2),
                delta * (td_abs - 0.5 * delta),
            )
            grad_errors = np.where(quadratic, td_errors, delta * np.sign(td_errors))
            loss_value = float(np.mean(loss_val))
        else:
            grad_errors = td_errors
            loss_value = float(0.5 * np.mean(td_errors**2))

        grad_output = np.zeros_like(q_values, dtype=np.float32)
        grad_output[np.arange(batch_size), actions_arr] = grad_errors / batch_size

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
        return loss_value, td_abs_mean

    def _train_batch_dueling(
        self,
        q_values: np.ndarray,
        cache: dict,
        actions_arr: np.ndarray,
        targets_arr: np.ndarray,
        optimizer: AdamOptimizer,
        loss: str,
        huber_delta: float,
    ) -> tuple[float, float]:
        batch_size = q_values.shape[0]

        pred = q_values[np.arange(batch_size), actions_arr]
        td_errors = pred - targets_arr
        td_abs = np.abs(td_errors)
        td_abs_mean = float(np.mean(td_abs))
        if loss == "huber":
            delta = float(huber_delta)
            quadratic = td_abs <= delta
            loss_val = np.where(
                quadratic,
                0.5 * (td_errors**2),
                delta * (td_abs - 0.5 * delta),
            )
            grad_errors = np.where(quadratic, td_errors, delta * np.sign(td_errors))
            loss_value = float(np.mean(loss_val))
        else:
            grad_errors = td_errors
            loss_value = float(0.5 * np.mean(td_errors**2))

        grad_q = np.zeros_like(q_values, dtype=np.float32)
        grad_q[np.arange(batch_size), actions_arr] = grad_errors / batch_size

        features: np.ndarray = cache["features"]
        trunk_activations: List[np.ndarray] = cache["trunk_activations"]
        trunk_pre_activations: List[np.ndarray] = cache["trunk_pre_activations"]

        assert self.value_w is not None and self.value_b is not None
        assert self.advantage_w is not None and self.advantage_b is not None

        grad_value = np.sum(grad_q, axis=1, keepdims=True)
        grad_advantage = grad_q - np.mean(grad_q, axis=1, keepdims=True)

        grad_value_w = features.T @ grad_value
        grad_value_b = np.sum(grad_value, axis=0, keepdims=True)
        grad_advantage_w = features.T @ grad_advantage
        grad_advantage_b = np.sum(grad_advantage, axis=0, keepdims=True)

        grad_features = grad_value @ self.value_w.T + grad_advantage @ self.advantage_w.T

        grad_trunk_w = [np.zeros_like(w, dtype=np.float32) for w in self.weights]
        grad_trunk_b = [np.zeros_like(b, dtype=np.float32) for b in self.biases]

        grad = grad_features
        for li in reversed(range(len(self.weights))):
            a_prev = trunk_activations[li]
            grad_trunk_w[li] = a_prev.T @ grad
            grad_trunk_b[li] = np.sum(grad, axis=0, keepdims=True)
            if li > 0:
                grad = grad @ self.weights[li].T
                grad *= self._relu_grad(trunk_pre_activations[li - 1])

        param_w = [*self.weights, self.value_w, self.advantage_w]
        param_b = [*self.biases, self.value_b, self.advantage_b]
        grad_w = [*grad_trunk_w, grad_value_w, grad_advantage_w]
        grad_b = [*grad_trunk_b, grad_value_b, grad_advantage_b]

        optimizer.step(param_w, param_b, grad_w, grad_b)
        return loss_value, td_abs_mean

    def train_batch(
        self,
        states: Sequence[Sequence[float]] | np.ndarray,
        actions: Sequence[int] | np.ndarray,
        targets: Sequence[float] | np.ndarray,
        optimizer: AdamOptimizer,
        loss: str = "mse",
        huber_delta: float = 1.0,
    ) -> Tuple[float, float]:
        actions_arr = np.asarray(actions, dtype=np.int64)
        targets_arr = np.asarray(targets, dtype=np.float32)

        q_values, cache = self.forward(states)
        if self.dueling:
            return self._train_batch_dueling(
                q_values,
                cache,
                actions_arr,
                targets_arr,
                optimizer,
                loss=loss,
                huber_delta=huber_delta,
            )
        return self._train_batch_non_dueling(
            q_values,
            cache,
            actions_arr,
            targets_arr,
            optimizer,
            loss=loss,
            huber_delta=huber_delta,
        )

    def copy_from(self, other: "MLPQNetwork") -> None:
        if self.dueling != other.dueling:
            raise ValueError("Network architectures do not match (dueling flag differs)")
        if len(self.weights) != len(other.weights):
            raise ValueError("Network architectures do not match")

        for i in range(len(self.weights)):
            self.weights[i] = other.weights[i].copy()
            self.biases[i] = other.biases[i].copy()

        if self.dueling:
            assert self.value_w is not None and self.value_b is not None
            assert self.advantage_w is not None and self.advantage_b is not None
            assert other.value_w is not None and other.value_b is not None
            assert other.advantage_w is not None and other.advantage_b is not None

            self.value_w = other.value_w.copy()
            self.value_b = other.value_b.copy()
            self.advantage_w = other.advantage_w.copy()
            self.advantage_b = other.advantage_b.copy()

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
            "hidden_sizes": list(self.hidden_sizes),
            "dueling": self.dueling,
        }

        if self.dueling:
            assert self.value_w is not None and self.value_b is not None
            assert self.advantage_w is not None and self.advantage_b is not None

            payload["trunk_weights"] = [w.tolist() for w in self.weights]
            payload["trunk_biases"] = [b.tolist() for b in self.biases]
            payload["value_w"] = self.value_w.tolist()
            payload["value_b"] = self.value_b.tolist()
            payload["advantage_w"] = self.advantage_w.tolist()
            payload["advantage_b"] = self.advantage_b.tolist()
        else:
            payload["weights"] = [w.tolist() for w in self.weights]
            payload["biases"] = [b.tolist() for b in self.biases]

        path.write_text(json.dumps(payload), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "MLPQNetwork":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        dueling = bool(payload.get("dueling", False))

        network = cls(
            input_dim=int(payload["input_dim"]),
            output_dim=int(payload["output_dim"]),
            hidden_sizes=tuple(int(v) for v in payload["hidden_sizes"]),
            seed=0,
            dueling=dueling,
        )

        if dueling:
            trunk_weights = payload["trunk_weights"]
            trunk_biases = payload["trunk_biases"]
            for i, w in enumerate(trunk_weights):
                network.weights[i] = np.asarray(w, dtype=np.float32)
            for i, b in enumerate(trunk_biases):
                network.biases[i] = np.asarray(b, dtype=np.float32)

            network.value_w = np.asarray(payload["value_w"], dtype=np.float32)
            network.value_b = np.asarray(payload["value_b"], dtype=np.float32)
            network.advantage_w = np.asarray(payload["advantage_w"], dtype=np.float32)
            network.advantage_b = np.asarray(payload["advantage_b"], dtype=np.float32)
        else:
            for i, w in enumerate(payload["weights"]):
                network.weights[i] = np.asarray(w, dtype=np.float32)
            for i, b in enumerate(payload["biases"]):
                network.biases[i] = np.asarray(b, dtype=np.float32)

        return network
