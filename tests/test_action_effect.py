import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.generic_dqn_train import NStepAccumulator, _StepTransition
from src.mono_forward import MonoForwardQNetwork, MonoForwardTrainer
from src.network import AdamOptimizer, MLPQNetwork


class TestActionEffect(unittest.TestCase):
    def _batch(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        # State e_i means action i changes the observation and the others do not.
        states = []
        actions = []
        labels = []
        for state_i in range(3):
            for action in range(3):
                state = np.zeros(3, dtype=np.float32)
                state[state_i] = 1.0
                states.append(state)
                actions.append(action)
                labels.append(1.0 if action == state_i else 0.0)
        return (
            np.stack(states).astype(np.float32),
            np.asarray(actions, dtype=np.int64),
            np.asarray(labels, dtype=np.float32),
        )

    def test_head_learns_which_action_changes_the_observation(self) -> None:
        net = MLPQNetwork(input_dim=3, output_dim=3, hidden_sizes=(16,), seed=1, action_effect=True)
        optimizer = AdamOptimizer(lr=1e-2)
        states, actions, labels = self._batch()
        targets = np.zeros(len(actions), dtype=np.float32)
        for _ in range(80):
            net.train_batch(states, actions, targets, optimizer, effect_labels=labels)
        probs = net.predict_action_effects(states)
        pred = (probs[np.arange(len(actions)), actions] >= 0.5).astype(np.float32)
        accuracy = float(np.mean(pred == labels))
        self.assertGreater(accuracy, 0.9)
        self.assertGreater(float(net.last_effect_acc or 0.0), 0.9)

    def test_save_load_keeps_the_head(self) -> None:
        net = MLPQNetwork(input_dim=3, output_dim=2, hidden_sizes=(4,), seed=2, action_effect=True)
        states = np.ones((2, 3), dtype=np.float32)
        before = net.predict_action_effects(states)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.json"
            net.save(path)
            loaded = MLPQNetwork.load(path)
        after = loaded.predict_action_effects(states)
        self.assertTrue(np.allclose(before, after))

    def test_old_checkpoint_has_no_head(self) -> None:
        net = MLPQNetwork(input_dim=3, output_dim=2, hidden_sizes=(4,), seed=3)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.json"
            net.save(path)
            payload = json.loads(path.read_text())
            payload.pop("action_effect", None)
            path.write_text(json.dumps(payload))
            loaded = MLPQNetwork.load(path)
        self.assertFalse(loaded.action_effect)

    def test_mono_forward_trains_the_head(self) -> None:
        net = MonoForwardQNetwork(input_dim=3, output_dim=3, hidden_sizes=(16,), seed=4, action_effect=True)
        trainer = MonoForwardTrainer(net, lr=1e-2)
        states, actions, labels = self._batch()
        targets = np.zeros(len(actions), dtype=np.float32)
        for _ in range(80):
            trainer.step(net, states, actions, targets, effect_labels=labels)
        probs = net.predict_action_effects(states)
        pred = (probs[np.arange(len(actions)), actions] >= 0.5).astype(np.float32)
        self.assertGreater(float(np.mean(pred == labels)), 0.9)

    def test_n_step_keeps_the_immediate_change(self) -> None:
        acc = NStepAccumulator(n_step=3)
        ready = []
        for i in range(3):
            ready.extend(
                acc.append_and_pop_ready(
                    _StepTransition(
                        state=np.zeros(2, dtype=np.float32),
                        action=i,
                        reward=1.0,
                        next_state=np.ones(2, dtype=np.float32),
                        done=False,
                        next_legal_actions=None,
                        changed=1.0 if i == 0 else 0.0,
                    ),
                    gamma=0.99,
                )
            )
        self.assertEqual(len(ready), 1)
        self.assertEqual(ready[0].action, 0)
        self.assertEqual(ready[0].changed, 1.0)


if __name__ == "__main__":
    unittest.main()
