from __future__ import annotations

import unittest

import numpy as np

from src.games.match3 import Match3Config, Match3Env, Match3Game


class TestMatch3Game(unittest.TestCase):
    def test_reset_has_no_initial_matches_and_has_legal_move(self) -> None:
        game = Match3Game(Match3Config(width=6, height=6, num_colors=5), seed=1)
        game.reset(seed=1)
        self.assertFalse(np.any(game._find_matches_mask(game.board_arr)))
        self.assertGreater(len(game.legal_actions()), 0)

    def test_action_size_for_6x6(self) -> None:
        game = Match3Game(Match3Config(width=6, height=6), seed=1)
        self.assertEqual(game.action_size, 60)  # horizontal 30 + vertical 30

    def test_find_matches_detects_row_and_col(self) -> None:
        board = np.array(
            [
                [1, 1, 1, 2],
                [2, 3, 4, 2],
                [2, 4, 5, 2],
                [2, 1, 3, 4],
            ],
            dtype=np.int32,
        )
        mask = Match3Game._find_matches_mask(board)
        self.assertTrue(np.all(mask[0, :3]))
        self.assertTrue(np.all(mask[:4, 0]))
        self.assertTrue(np.all(mask[:3, 3]))
        self.assertFalse(mask[3, 3])

    def test_step_clears_tiles_after_valid_swap(self) -> None:
        game = Match3Game(Match3Config(width=4, height=4, num_colors=4, max_steps=10), seed=1)
        game.reset(seed=1)
        game.board_arr = np.array(
            [
                [1, 2, 3, 4],
                [1, 3, 2, 4],
                [4, 2, 2, 4],
                [3, 1, 4, 1],
            ],
            dtype=np.int32,
        )
        game._invalidate_legal_cache()
        # Swap (1,1) with (1,2) to make column of 2s at col=1.
        action = None
        for a in game.legal_actions():
            s1, s2 = game.action_to_swap(a)
            if {s1, s2} == { (1, 1), (1, 2) }:
                action = a
                break
        self.assertIsNotNone(action)
        result = game.step(int(action))
        self.assertGreaterEqual(result.tiles_cleared, 3)
        self.assertGreater(game.score, 0)
        self.assertFalse(result.invalid)

    def test_invalid_swap_gets_penalty(self) -> None:
        game = Match3Game(Match3Config(width=5, height=5, num_colors=4, invalid_penalty=-0.7), seed=1)
        game.reset(seed=1)
        illegal = next(a for a in range(game.action_size) if a not in set(game.legal_actions()))
        before = game.board_arr.copy()
        result = game.step(illegal)
        self.assertTrue(result.invalid)
        self.assertAlmostEqual(result.reward, game.config.step_reward + game.config.invalid_penalty, places=5)
        np.testing.assert_array_equal(game.board_arr, before)


class TestMatch3Env(unittest.TestCase):
    def test_state_shape(self) -> None:
        cfg = Match3Config(width=6, height=6, num_colors=5)
        env = Match3Env(cfg, seed=1)
        state = env.reset(seed=1)
        self.assertEqual(state.ndim, 1)
        self.assertEqual(state.shape[0], 5 * 6 * 6 + 2)

    def test_legal_actions_non_empty_after_reset(self) -> None:
        env = Match3Env(Match3Config(), seed=1)
        env.reset(seed=1)
        self.assertGreater(len(env.legal_actions()), 0)


if __name__ == "__main__":
    unittest.main()
