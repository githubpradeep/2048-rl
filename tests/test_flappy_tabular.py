import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.flappy_tabular import FlappyDiscretizerConfig, FlappyStateDiscretizer, FlappyTabularQAgent


class TestFlappyDiscretizer(unittest.TestCase):
    def test_encode_clips_and_returns_three_bins(self) -> None:
        discretizer = FlappyStateDiscretizer(
            FlappyDiscretizerConfig(dx_bins=8, dy_bins=9, vel_bins=7)
        )
        state = np.asarray([0.5, 2.0, 99.0, 0.5, -5.0, 0.0, 0.0, 0.0], dtype=np.float32)
        key = discretizer.encode(state)
        self.assertEqual(len(key), 3)
        self.assertTrue(0 <= key[0] < 8)
        self.assertTrue(0 <= key[1] < 9)
        self.assertTrue(0 <= key[2] < 7)


class TestFlappyTabularQAgent(unittest.TestCase):
    def test_q_learning_update_increases_rewarded_action(self) -> None:
        agent = FlappyTabularQAgent(alpha=0.5, gamma=0.9)
        s = (1, 2, 3)
        ns = (1, 2, 4)
        before = float(agent.q_values(s)[1])
        td_abs = agent.update_q_learning(s, 1, reward=2.0, next_state_key=ns, done=True)
        after = float(agent.q_values(s)[1])
        self.assertGreater(td_abs, 0.0)
        self.assertGreater(after, before)

    def test_save_load_roundtrip(self) -> None:
        agent = FlappyTabularQAgent(alpha=0.2, gamma=0.95)
        agent.update_q_learning((0, 0, 0), 0, reward=1.0, next_state_key=(0, 0, 1), done=False, next_legal_actions=[0, 1])
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "model.json"
            agent.save(path)
            loaded = FlappyTabularQAgent.load(path)
        self.assertEqual(loaded.action_size, agent.action_size)
        self.assertEqual(len(loaded.q_table), len(agent.q_table))
        self.assertEqual(loaded.predict_one(np.zeros(8, dtype=np.float32)).__class__, list)


if __name__ == "__main__":
    unittest.main()

