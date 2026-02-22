import unittest

import numpy as np

from src.games.shooter import SHOOT, STAY, ShooterConfig, ShooterEnv, ShooterGame


class TestShooterGame(unittest.TestCase):
    def test_enemy_kill_with_bullet(self) -> None:
        cfg = ShooterConfig(grid_size=8, spawn_prob=0.0, step_reward=0.0, kill_reward=3.0)
        game = ShooterGame(config=cfg, seed=1)
        game.reset(seed=1)

        # Bullet at row 3 -> row 2, enemy at row 1 -> row 2 => collision.
        game.bullets = [(3, 4)]
        game.enemies = [(1, 4)]

        result = game.step(STAY)
        self.assertFalse(result.done)
        self.assertEqual(result.kills, 1)
        self.assertEqual(game.score, 1)
        self.assertAlmostEqual(result.reward, 3.0, places=6)

    def test_enemy_escape_penalty(self) -> None:
        cfg = ShooterConfig(grid_size=8, spawn_prob=0.0, step_reward=0.0, escape_penalty=-1.5)
        game = ShooterGame(config=cfg, seed=2)
        game.reset(seed=2)

        game.player_col = 6
        game.enemies = [(6, 1)]  # moves to bottom row and escapes
        result = game.step(STAY)

        self.assertFalse(result.done)
        self.assertEqual(result.escaped, 1)
        self.assertAlmostEqual(result.reward, -1.5, places=6)

    def test_player_hit_reduces_life_and_can_end(self) -> None:
        cfg = ShooterConfig(grid_size=8, spawn_prob=0.0, step_reward=0.0, player_hit_penalty=-4.0, start_lives=1)
        game = ShooterGame(config=cfg, seed=3)
        game.reset(seed=3)

        game.player_col = 3
        game.enemies = [(6, 3)]
        result = game.step(STAY)

        self.assertTrue(result.done)
        self.assertEqual(game.lives, 0)
        self.assertEqual(result.player_hits, 1)
        self.assertAlmostEqual(result.reward, -4.0, places=6)

    def test_shoot_action_creates_bullet(self) -> None:
        cfg = ShooterConfig(grid_size=8, spawn_prob=0.0)
        game = ShooterGame(config=cfg, seed=4)
        game.reset(seed=4)

        game.player_col = 2
        game.step(SHOOT)
        self.assertTrue(any(c == 2 for _, c in game.bullets))


class TestShooterEnv(unittest.TestCase):
    def test_state_shape(self) -> None:
        env = ShooterEnv(config=ShooterConfig(grid_size=9), seed=5)
        state = env.reset(seed=5)
        self.assertEqual(state.shape, (3 * 9 * 9,))
        self.assertEqual(state.dtype, np.float32)

    def test_state_shape_with_padding(self) -> None:
        env = ShooterEnv(config=ShooterConfig(grid_size=8), seed=6, state_grid_size=10)
        state = env.reset(seed=6)
        self.assertEqual(state.shape, (3 * 10 * 10,))

    def test_legal_actions(self) -> None:
        env = ShooterEnv(config=ShooterConfig(grid_size=8), seed=7)
        env.reset(seed=7)
        self.assertEqual(env.legal_actions(), [0, 1, 2, 3])


if __name__ == "__main__":
    unittest.main()
