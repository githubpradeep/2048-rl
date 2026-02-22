from __future__ import annotations

import unittest

from src.games.pong import DOWN, STAY, UP, PongConfig, PongEnv, PongGame


class TestPongGame(unittest.TestCase):
    def test_reset_initializes_state(self) -> None:
        game = PongGame(PongConfig(width=10, height=12, paddle_height=3, start_lives=2), seed=1)
        board = game.reset(seed=1)
        self.assertEqual(board.shape, (12, 10))
        self.assertEqual(game.lives, 2)
        self.assertIn(game.ball_vx, (-1, 1))
        self.assertIn(game.ball_vy, (-1, 1))

    def test_player_paddle_hit_bounces_ball(self) -> None:
        game = PongGame(PongConfig(width=10, height=12, paddle_height=3), seed=1)
        game.reset(seed=1)
        game.player_y = 4
        game.ball_x = 1
        game.ball_y = 5
        game.ball_vx = -1
        game.ball_vy = 0

        result = game.step(STAY)

        self.assertTrue(result.player_hit)
        self.assertGreater(game.ball_vx, 0)

    def test_player_scores_when_opponent_misses(self) -> None:
        game = PongGame(PongConfig(width=10, height=12, paddle_height=3), seed=1)
        game.reset(seed=1)
        game.opponent_y = 0
        game.ball_x = game.right_paddle_col - 1
        game.ball_y = 11
        game.ball_vx = 1
        game.ball_vy = 0

        result = game.step(STAY)

        self.assertTrue(result.player_scored)
        self.assertEqual(game.player_score, 1)

    def test_concede_loses_life(self) -> None:
        game = PongGame(PongConfig(width=10, height=12, paddle_height=3, start_lives=2), seed=1)
        game.reset(seed=1)
        game.player_y = 0
        game.ball_x = 1
        game.ball_y = 11
        game.ball_vx = -1
        game.ball_vy = 0

        result = game.step(STAY)

        self.assertTrue(result.opponent_scored)
        self.assertEqual(game.lives, 1)

    def test_game_ends_when_lives_depleted(self) -> None:
        game = PongGame(PongConfig(width=10, height=12, paddle_height=3, start_lives=1), seed=1)
        game.reset(seed=1)
        game.player_y = 0
        game.ball_x = 1
        game.ball_y = 11
        game.ball_vx = -1
        game.ball_vy = 0

        result = game.step(STAY)

        self.assertTrue(result.done)
        self.assertEqual(game.lives, 0)


class TestPongEnv(unittest.TestCase):
    def test_state_shape(self) -> None:
        env = PongEnv(PongConfig(width=12, height=16), seed=1)
        state = env.reset(seed=1)
        self.assertEqual(state.ndim, 1)
        self.assertEqual(state.shape[0], 3 * 12 * 16 + 7)

    def test_legal_actions(self) -> None:
        env = PongEnv(PongConfig(), seed=1)
        env.reset(seed=1)
        self.assertEqual(env.legal_actions(), [UP, STAY, DOWN])


if __name__ == "__main__":
    unittest.main()

