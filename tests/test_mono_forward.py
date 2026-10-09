import unittest

import numpy as np

from src.mono_forward import MonoForwardQNetwork, MonoForwardTrainer
from src.network import MLPQNetwork


class TestMonoForward(unittest.TestCase):
    def test_forward_matches_plain_mlp(self) -> None:
        net = MonoForwardQNetwork(input_dim=4, output_dim=3, hidden_sizes=(5,), seed=1)
        states = np.random.default_rng(2).standard_normal((6, 4)).astype(np.float32)
        q_values, _ = net.forward(states)
        self.assertEqual(q_values.shape, (6, 3))
        loaded = MLPQNetwork.load  # eval loads the base class
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.json"
            net.save(path)
            restored = loaded(path)
        again, _ = restored.forward(states)
        self.assertTrue(np.allclose(q_values, again))

    def test_hidden_layer_learns_without_the_output_layer(self) -> None:
        net = MonoForwardQNetwork(input_dim=4, output_dim=2, hidden_sizes=(8, 8), seed=3)
        net.weights[-1][:] = 0.0
        states = np.random.default_rng(4).standard_normal((16, 4)).astype(np.float32)
        actions = np.zeros(16, dtype=np.int64)
        targets = np.ones(16, dtype=np.float32)
        trainer = MonoForwardTrainer(net, lr=1e-2)
        before = net.weights[0].copy()
        trainer.step(net, states, actions, targets)
        self.assertFalse(np.allclose(net.weights[0], before))

    def test_readout_warmup_does_not_move_hidden_weights(self) -> None:
        net = MonoForwardQNetwork(input_dim=4, output_dim=2, hidden_sizes=(8,), seed=7)
        hidden_before = net.weights[0].copy()
        readout_before = net.projections[0].copy()
        states = np.random.default_rng(8).standard_normal((16, 4)).astype(np.float32)
        actions = np.zeros(16, dtype=np.int64)
        targets = np.ones(16, dtype=np.float32)
        trainer = MonoForwardTrainer(net, lr=1e-2)
        trainer.hidden_weight_freeze_steps = 2
        trainer.step(net, states, actions, targets)
        self.assertTrue(np.allclose(net.weights[0], hidden_before))
        self.assertFalse(np.allclose(net.projections[0], readout_before))
        self.assertEqual(trainer.hidden_weight_freeze_steps, 1)

    def test_reduces_played_q_loss(self) -> None:
        net = MonoForwardQNetwork(input_dim=4, output_dim=2, hidden_sizes=(16, 16), seed=5)
        rng = np.random.default_rng(6)
        states = rng.standard_normal((32, 4)).astype(np.float32)
        actions = rng.integers(0, 2, size=32)
        targets = rng.standard_normal(32).astype(np.float32)
        trainer = MonoForwardTrainer(net, lr=1e-2)

        def mse() -> float:
            q_values, _ = net.forward(states)
            pred = q_values[np.arange(32), actions]
            return float(np.mean(0.5 * (pred - targets) ** 2))

        before = mse()
        for _ in range(40):
            trainer.step(net, states, actions, targets, loss="huber", huber_delta=1.0)
        after = mse()
        self.assertLess(after, 0.5 * before, f"{before:.5f} -> {after:.5f}")


if __name__ == "__main__":
    unittest.main()
