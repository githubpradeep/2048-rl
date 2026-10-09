import unittest

import numpy as np

from src.dust import DustConfig, DustOptimizer, estimate_gradients
from src.network import MLPQNetwork


def _batch(seed: int = 0, batch: int = 24, dim: int = 4, actions: int = 2):
    rng = np.random.default_rng(seed)
    states = rng.standard_normal((batch, dim)).astype(np.float32)
    taken = rng.integers(0, actions, size=batch)
    targets = rng.standard_normal(batch).astype(np.float32)
    return states, taken, targets


def _backprop_linear(net: MLPQNetwork, states: np.ndarray, actions: np.ndarray, targets: np.ndarray):
    q_values, _ = net.forward(states)
    batch = states.shape[0]
    grad_q = np.zeros_like(q_values)
    grad_q[np.arange(batch), actions] = (q_values[np.arange(batch), actions] - targets) / batch
    return states.T @ grad_q, np.sum(grad_q, axis=0, keepdims=True)


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0:
        return 0.0
    return float(np.sum(a * b) / denom)


class TestDustGradient(unittest.TestCase):
    def test_linear_layer_aligns_with_backprop(self) -> None:
        net = MLPQNetwork(input_dim=4, output_dim=2, hidden_sizes=(), seed=1)
        states, actions, targets = _batch()
        true_w, true_b = _backprop_linear(net, states, actions, targets)
        grad_w, grad_b = estimate_gradients(
            net,
            states,
            actions,
            targets,
            np.random.default_rng(2),
            population=128,
            sigma=0.05,
            loss="mse",
            huber_delta=1.0,
            sample_weights=None,
        )
        self.assertGreater(_cosine(grad_w[0], true_w), 0.9)
        self.assertGreater(_cosine(grad_b[0], true_b), 0.9)

    def test_rejects_non_positive_sigma(self) -> None:
        with self.assertRaises(ValueError):
            DustConfig(sigma=0.0).validate()

    def test_reduces_td_loss(self) -> None:
        net = MLPQNetwork(input_dim=4, output_dim=2, hidden_sizes=(16, 16), seed=3, dueling=True)
        states, actions, targets = _batch(seed=4, batch=32)
        optimizer = DustOptimizer(DustConfig(population=32, sigma=0.1, lr=1e-2, seed=5))

        def mse() -> float:
            q_values, _ = net.forward(states)
            pred = q_values[np.arange(len(actions)), actions]
            return float(np.mean(0.5 * (pred - targets) ** 2))

        before = mse()
        for _ in range(25):
            optimizer.step(net, states, actions, targets, loss="mse")
        after = mse()
        self.assertLess(after, 0.5 * before, f"{before:.5f} -> {after:.5f}")


if __name__ == "__main__":
    unittest.main()
