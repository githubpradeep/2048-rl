import unittest

import numpy as np

from src.games.flappy import FLAP, NO_FLAP, FlappyConfig, FlappyEnv, FlappyGame, Pipe


class TestFlappyGame(unittest.TestCase):
    def test_reset_spawns_pipes(self) -> None:
        game = FlappyGame(config=FlappyConfig(width=84, height=84), seed=1)
        board = game.reset(seed=1)
        self.assertEqual(board.shape, (84, 84))
        self.assertGreaterEqual(len(game.pipes), 2)
        self.assertFalse(game.game_over)

    def test_flap_gives_negative_velocity(self) -> None:
        cfg = FlappyConfig(width=84, height=84, gravity=1.0, flap_velocity=-6.0)
        game = FlappyGame(config=cfg, seed=2)
        game.reset(seed=2)
        game.step(FLAP)
        self.assertLess(game.bird_velocity, 0.0)

    def test_pipe_pass_increases_score(self) -> None:
        cfg = FlappyConfig(width=84, height=84, pipe_speed=2.0, step_reward=0.0, pass_reward=1.0)
        game = FlappyGame(config=cfg, seed=3)
        game.reset(seed=3)

        # Place a pipe already behind the bird so it should be counted as passed on step.
        game.pipes = [Pipe(x=cfg.bird_x - cfg.pipe_width - 1.0, gap_y=game.bird_y, passed=False)]
        result = game.step(NO_FLAP)

        self.assertEqual(result.passed_pipes, 1)
        self.assertEqual(game.score, 1)
        self.assertGreaterEqual(result.reward, 1.0)

    def test_out_of_bounds_is_terminal(self) -> None:
        cfg = FlappyConfig(width=84, height=84, step_reward=0.0, crash_penalty=-3.0)
        game = FlappyGame(config=cfg, seed=4)
        game.reset(seed=4)
        game.bird_y = -1.0
        result = game.step(NO_FLAP)
        self.assertTrue(result.done)
        self.assertTrue(result.collision)
        self.assertAlmostEqual(result.reward, -3.0, places=6)

    def test_reset_uses_initial_pipe_offset(self) -> None:
        cfg = FlappyConfig(width=84, height=84, initial_pipe_offset=56.0)
        game = FlappyGame(config=cfg, seed=7)
        game.reset(seed=7)
        xs = sorted(pipe.x for pipe in game.pipes)
        self.assertAlmostEqual(xs[0], cfg.width + cfg.initial_pipe_offset, places=6)

    def test_gap_delta_is_constrained(self) -> None:
        cfg = FlappyConfig(width=84, height=84, max_gap_delta=5.0)
        game = FlappyGame(config=cfg, seed=8)
        game.reset(seed=8)
        for _ in range(5):
            game._spawn_pipe(None)  # test helper behavior directly
        gaps = [pipe.gap_y for pipe in game.pipes]
        for a, b in zip(gaps, gaps[1:]):
            self.assertLessEqual(abs(a - b), 5.0 + 1e-6)


class TestFlappyEnv(unittest.TestCase):
    def test_state_shape(self) -> None:
        env = FlappyEnv(config=FlappyConfig(width=84, height=84), seed=5)
        state = env.reset(seed=5)
        self.assertEqual(state.shape, (8,))
        self.assertEqual(state.dtype, np.float32)

    def test_legal_actions(self) -> None:
        env = FlappyEnv(config=FlappyConfig(width=84, height=84), seed=6)
        env.reset(seed=6)
        self.assertEqual(env.legal_actions(), [0, 1])


if __name__ == "__main__":
    unittest.main()
