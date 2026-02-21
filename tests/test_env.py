import unittest

import numpy as np

from src.env import EnvConfig, Game2048Env


class TestGame2048Env(unittest.TestCase):
    def test_reset_state_shape(self) -> None:
        env = Game2048Env(seed=7)
        state = env.reset(seed=7)
        self.assertEqual(state.shape, (16,))
        self.assertEqual(state.dtype, np.float32)

    def test_invalid_action_penalty(self) -> None:
        env = Game2048Env(seed=7, config=EnvConfig(invalid_move_penalty=-2.5, lose_penalty=0.0))
        env.game.board = np.array(
            [
                [2, 4, 8, 16],
                [32, 64, 128, 256],
                [512, 1024, 2, 4],
                [8, 16, 32, 64],
            ],
            dtype=np.int32,
        )
        next_state, reward, done, info = env.step(2)
        self.assertEqual(next_state.shape, (16,))
        self.assertEqual(reward, -2.5)
        self.assertTrue(done)
        self.assertFalse(info["moved"])

    def test_invalid_streak_forced_termination(self) -> None:
        env = Game2048Env(
            seed=7,
            config=EnvConfig(invalid_move_penalty=-1.0, lose_penalty=0.0, max_invalid_streak=2),
        )
        env.game.board = np.array(
            [
                [2, 4, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
            ],
            dtype=np.int32,
        )
        _, _, done1, info1 = env.step(0)  # up is invalid here
        self.assertFalse(done1)
        self.assertEqual(info1["invalid_streak"], 1)

        _, reward2, done2, info2 = env.step(0)
        self.assertTrue(done2)
        self.assertEqual(info2["invalid_streak"], 2)
        self.assertEqual(reward2, -1.0)


if __name__ == "__main__":
    unittest.main()
