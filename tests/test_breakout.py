from __future__ import annotations

import unittest

from src.games.breakout import BreakoutConfig, BreakoutEnv, BreakoutGame, LEFT, RIGHT, STAY


class TestBreakoutGame(unittest.TestCase):
    def test_reset_initializes_bricks_and_lives(self) -> None:
        game = BreakoutGame(BreakoutConfig(width=10, height=12, brick_rows=3, start_lives=2), seed=1)
        board = game.reset(seed=1)
        self.assertEqual(board.shape, (12, 10))
        self.assertEqual(game.lives, 2)
        self.assertGreater(game.bricks_left(), 0)

    def test_paddle_bounce_flips_vertical_velocity(self) -> None:
        game = BreakoutGame(BreakoutConfig(width=10, height=12, paddle_width=3), seed=1)
        game.reset(seed=1)
        game.paddle_x = 4
        game.ball_x = 5
        game.ball_y = game.paddle_row - 1
        game.ball_vx = 0
        game.ball_vy = 1

        result = game.step(STAY)

        self.assertTrue(result.paddle_hit)
        self.assertLess(game.ball_vy, 0)

    def test_brick_hit_increases_score(self) -> None:
        game = BreakoutGame(BreakoutConfig(width=10, height=12, brick_rows=1, brick_top=2), seed=1)
        game.reset(seed=1)
        game.bricks.fill(0)
        game.bricks[3, 5] = 1
        game.ball_x = 5
        game.ball_y = 4
        game.ball_vx = 0
        game.ball_vy = -1

        result = game.step(STAY)

        self.assertEqual(result.bricks_hit, 1)
        self.assertEqual(game.score, 1)
        self.assertFalse(game.bricks[3, 5])

    def test_miss_loses_life(self) -> None:
        game = BreakoutGame(BreakoutConfig(width=10, height=12, start_lives=2), seed=1)
        game.reset(seed=1)
        game.paddle_x = 0
        game.ball_x = 9
        game.ball_y = game.paddle_row
        game.ball_vx = 0
        game.ball_vy = 1

        result = game.step(STAY)

        self.assertTrue(result.life_lost)
        self.assertEqual(game.lives, 1)
        self.assertFalse(result.done)

    def test_clear_all_bricks_ends_episode(self) -> None:
        game = BreakoutGame(BreakoutConfig(width=8, height=10, brick_rows=1, brick_top=1), seed=1)
        game.reset(seed=1)
        game.bricks.fill(0)
        game.bricks[2, 3] = 1
        game.ball_x = 3
        game.ball_y = 3
        game.ball_vx = 0
        game.ball_vy = -1

        result = game.step(STAY)

        self.assertTrue(result.cleared)
        self.assertTrue(result.done)


class TestBreakoutEnv(unittest.TestCase):
    def test_state_shape(self) -> None:
        env = BreakoutEnv(BreakoutConfig(width=12, height=14), seed=1)
        state = env.reset(seed=1)
        self.assertEqual(state.ndim, 1)
        self.assertEqual(state.shape[0], 3 * 12 * 14 + 3)

    def test_legal_actions(self) -> None:
        env = BreakoutEnv(BreakoutConfig(), seed=1)
        env.reset(seed=1)
        self.assertEqual(env.legal_actions(), [LEFT, STAY, RIGHT])


if __name__ == "__main__":
    unittest.main()

