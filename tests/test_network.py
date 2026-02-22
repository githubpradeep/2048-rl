import tempfile
import unittest
from pathlib import Path

from src.network import AdamOptimizer, MLPQNetwork


class TestPurePythonNetwork(unittest.TestCase):
    def test_train_batch_reduces_loss(self) -> None:
        network = MLPQNetwork(input_dim=4, output_dim=2, hidden_sizes=(12,), seed=7)
        optimizer = AdamOptimizer(lr=0.01)

        states = [
            [0.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [1.0, 1.0, 0.0, 0.0],
            [0.0, 1.0, 1.0, 0.0],
            [0.0, 0.0, 1.0, 1.0],
        ]
        actions = [0, 0, 1, 1, 0, 0, 1, 1]
        targets = [0.0, 1.0, 1.0, 0.5, 0.7, 1.2, 1.3, 0.8]

        def mse() -> float:
            err = 0.0
            for s, a, t in zip(states, actions, targets):
                pred = network.predict_one(s)[a]
                err += 0.5 * ((pred - t) ** 2)
            return err / len(states)

        loss_before = mse()
        for _ in range(300):
            network.train_batch(states, actions, targets, optimizer)
        loss_after = mse()

        self.assertLess(loss_after, loss_before)
        self.assertLess(loss_after, 0.2 * loss_before)

    def test_save_load_roundtrip(self) -> None:
        network = MLPQNetwork(input_dim=3, output_dim=4, hidden_sizes=(6, 6), seed=12)
        state = [0.1, 0.2, 0.3]
        pred_before = network.predict_one(state)

        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "model.json"
            network.save(path)
            loaded = MLPQNetwork.load(path)
            pred_after = loaded.predict_one(state)

        self.assertEqual(len(pred_before), len(pred_after))
        for a, b in zip(pred_before, pred_after):
            self.assertAlmostEqual(a, b, places=8)

    def test_dueling_train_batch_reduces_loss(self) -> None:
        network = MLPQNetwork(input_dim=5, output_dim=3, hidden_sizes=(16,), seed=9, dueling=True)
        optimizer = AdamOptimizer(lr=0.01)

        states = [
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0],
            [1.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 1.0, 0.0, 0.0],
        ]
        actions = [0, 1, 2, 0, 1, 2, 1, 2]
        targets = [0.2, 1.2, 0.7, 0.4, 1.3, 0.9, 1.5, 1.0]

        def mse() -> float:
            err = 0.0
            for s, a, t in zip(states, actions, targets):
                pred = network.predict_one(s)[a]
                err += 0.5 * ((pred - t) ** 2)
            return err / len(states)

        loss_before = mse()
        for _ in range(300):
            network.train_batch(states, actions, targets, optimizer)
        loss_after = mse()

        self.assertLess(loss_after, loss_before)
        self.assertLess(loss_after, 0.2 * loss_before)

    def test_dueling_save_load_roundtrip(self) -> None:
        network = MLPQNetwork(input_dim=4, output_dim=4, hidden_sizes=(8, 8), seed=13, dueling=True)
        state = [0.4, 0.3, 0.2, 0.1]
        pred_before = network.predict_one(state)

        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "dueling_model.json"
            network.save(path)
            loaded = MLPQNetwork.load(path)
            pred_after = loaded.predict_one(state)

        self.assertEqual(len(pred_before), len(pred_after))
        for a, b in zip(pred_before, pred_after):
            self.assertAlmostEqual(a, b, places=8)


if __name__ == "__main__":
    unittest.main()
