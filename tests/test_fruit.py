import unittest

import numpy as np

from src.games.fruit_cutter import FruitCutterConfig, FruitCutterEnv, FruitCutterGame, STAY


class TestFruitCutterGame(unittest.TestCase):
    def test_slice_fruit_increases_score(self) -> None:
        cfg = FruitCutterConfig(grid_size=8, spawn_prob=0.0, step_reward=0.1, slice_reward=2.0)
        game = FruitCutterGame(config=cfg, seed=1)
        game.reset(seed=1)

        game.player_col = 3
        game.items = [(6, 3, False)]
        result = game.step(STAY)

        self.assertFalse(result.done)
        self.assertEqual(result.sliced, 1)
        self.assertEqual(game.score, 1)
        self.assertAlmostEqual(result.reward, 2.1, places=6)

    def test_bomb_hit_ends_game(self) -> None:
        cfg = FruitCutterConfig(grid_size=8, spawn_prob=0.0, step_reward=0.1, bomb_hit_penalty=-7.0)
        game = FruitCutterGame(config=cfg, seed=2)
        game.reset(seed=2)

        game.player_col = 2
        game.items = [(6, 2, True)]
        result = game.step(STAY)

        self.assertTrue(result.done)
        self.assertTrue(result.bomb_hit)
        self.assertAlmostEqual(result.reward, -6.9, places=6)

    def test_missed_fruit_penalty(self) -> None:
        cfg = FruitCutterConfig(grid_size=8, spawn_prob=0.0, step_reward=0.2, miss_penalty=-0.5)
        game = FruitCutterGame(config=cfg, seed=3)
        game.reset(seed=3)

        game.player_col = 5
        game.items = [(6, 1, False)]
        result = game.step(STAY)

        self.assertFalse(result.done)
        self.assertEqual(result.missed, 1)
        self.assertAlmostEqual(result.reward, -0.3, places=6)


class TestFruitCutterEnv(unittest.TestCase):
    def test_state_shape(self) -> None:
        env = FruitCutterEnv(config=FruitCutterConfig(grid_size=9), seed=4)
        state = env.reset(seed=4)
        self.assertEqual(state.shape, (3 * 9 * 9,))
        self.assertEqual(state.dtype, np.float32)

    def test_state_shape_with_padding(self) -> None:
        env = FruitCutterEnv(config=FruitCutterConfig(grid_size=8), seed=5, state_grid_size=10)
        state = env.reset(seed=5)
        self.assertEqual(state.shape, (3 * 10 * 10,))

    def test_legal_actions(self) -> None:
        env = FruitCutterEnv(config=FruitCutterConfig(grid_size=8), seed=6)
        env.reset(seed=6)
        self.assertEqual(env.legal_actions(), [0, 1, 2])


if __name__ == "__main__":
    unittest.main()
