import unittest

import numpy as np

from src.games.tetris import HARD_DROP, ROTATE, SOFT_DROP, TetrisConfig, TetrisEnv, TetrisGame, TetrisPlacementEnv
from src.replay_buffer import ReplayBuffer
from src.train_tetris_dqn import masked_double_q, masked_max_q


class TestTetrisGame(unittest.TestCase):
    def test_reset_spawns_piece(self) -> None:
        game = TetrisGame(config=TetrisConfig(height=12, width=8), seed=11)
        board = game.reset(seed=11)
        self.assertEqual(board.shape, (12, 8))
        self.assertFalse(game.game_over)
        self.assertGreater(np.sum(board == 2), 0)

    def test_hard_drop_moves_piece_and_advances(self) -> None:
        game = TetrisGame(config=TetrisConfig(height=12, width=8), seed=12)
        game.reset(seed=12)
        result = game.step(HARD_DROP)
        self.assertGreaterEqual(result.hard_drop_distance, 0)
        self.assertGreaterEqual(game.steps, 1)

    def test_line_clear_updates_counter(self) -> None:
        cfg = TetrisConfig(height=8, width=6, step_reward=0.0, line_clear_reward=1.0)
        game = TetrisGame(config=cfg, seed=13)
        game.reset(seed=13)

        # Prepare a nearly full bottom row with one gap at col=2.
        game.board[-1] = np.array([1, 1, 0, 1, 1, 1], dtype=np.int32)
        game.current_name = "I"
        game.current_piece = np.array([[1]], dtype=np.int32)
        game.piece_row = game.height - 1
        game.piece_col = 2

        result = game.step(SOFT_DROP)
        self.assertEqual(result.lines_cleared, 1)
        self.assertEqual(game.lines, 1)

    def test_spawn_collision_sets_game_over(self) -> None:
        game = TetrisGame(config=TetrisConfig(height=8, width=6), seed=14)
        game.reset(seed=14)
        game.board[:2, :] = 1
        game._spawn_piece()
        self.assertTrue(game.game_over)

    def test_rotate_changes_piece_shape(self) -> None:
        game = TetrisGame(config=TetrisConfig(height=12, width=8), seed=15)
        game.reset(seed=15)

        game.current_name = "L"
        game.current_piece = np.array([[0, 0, 1], [1, 1, 1]], dtype=np.int32)
        game.piece_row = 2
        game.piece_col = 2
        old_shape = game.current_piece.shape

        game.step(ROTATE)
        self.assertNotEqual(game.current_piece.shape, old_shape)


class TestTetrisEnv(unittest.TestCase):
    def test_state_shape(self) -> None:
        env = TetrisEnv(config=TetrisConfig(height=10, width=8), seed=20)
        state = env.reset(seed=20)
        self.assertEqual(state.shape, (2 * 10 * 8 + 7,))
        self.assertEqual(state.dtype, np.float32)

    def test_legal_actions(self) -> None:
        env = TetrisEnv(config=TetrisConfig(height=10, width=8), seed=21)
        env.reset(seed=21)
        actions = env.legal_actions()
        self.assertIn(SOFT_DROP, actions)
        self.assertIn(HARD_DROP, actions)
        self.assertTrue(set(actions).issubset({0, 1, 2, 3, 4}))


class TestTetrisPlacementEnv(unittest.TestCase):
    def test_state_shape(self) -> None:
        env = TetrisPlacementEnv(config=TetrisConfig(height=10, width=8), seed=30)
        state = env.reset(seed=30)
        self.assertEqual(state.shape, (10 * 8 + 7 + 8 + 8 + 3,))
        self.assertEqual(state.dtype, np.float32)

    def test_legal_actions(self) -> None:
        env = TetrisPlacementEnv(config=TetrisConfig(height=10, width=8), seed=31)
        env.reset(seed=31)
        actions = env.legal_actions()
        self.assertGreater(len(actions), 0)
        self.assertTrue(all(0 <= a < 8 * 4 for a in actions))

    def test_step_runs(self) -> None:
        env = TetrisPlacementEnv(config=TetrisConfig(height=10, width=8), seed=32)
        env.reset(seed=32)
        action = env.legal_actions()[0]
        _, _, _, info = env.step(action)
        self.assertIn("score", info)
        self.assertIn("lines", info)

    def test_legal_afterstates_shape(self) -> None:
        env = TetrisPlacementEnv(config=TetrisConfig(height=10, width=8), seed=33)
        env.reset(seed=33)
        pairs = env.legal_afterstates()
        self.assertGreater(len(pairs), 0)
        action, features = pairs[0]
        self.assertTrue(0 <= action < env.action_size)
        self.assertEqual(features.shape, (5,))


class TestTetrisTargets(unittest.TestCase):
    def test_masked_max_q_ignores_invalid_actions(self) -> None:
        target_q = np.array([[1.0, 4.0, 9.0], [3.0, 2.0, 1.0]], dtype=np.float32)
        masks = np.array([[1.0, 1.0, 0.0], [0.0, 0.0, 0.0]], dtype=np.float32)
        max_q = masked_max_q(target_q, masks)
        np.testing.assert_allclose(max_q, np.array([4.0, 0.0], dtype=np.float32))

    def test_masked_double_q_ignores_invalid_actions(self) -> None:
        online_q = np.array([[1.0, 8.0, 9.0], [3.0, 2.0, 1.0]], dtype=np.float32)
        target_q = np.array([[2.0, 5.0, 7.0], [4.0, 6.0, 1.0]], dtype=np.float32)
        masks = np.array([[1.0, 1.0, 0.0], [0.0, 0.0, 0.0]], dtype=np.float32)
        max_q = masked_double_q(online_q, target_q, masks)
        np.testing.assert_allclose(max_q, np.array([5.0, 0.0], dtype=np.float32))

    def test_replay_buffer_stores_next_legal_masks(self) -> None:
        buffer = ReplayBuffer(capacity=4, state_dim=3, action_dim=5)
        state = np.array([0.0, 1.0, 2.0], dtype=np.float32)
        next_state = np.array([3.0, 4.0, 5.0], dtype=np.float32)
        buffer.add(state, 1, 0.5, next_state, False, next_legal_actions=[0, 3])
        batch = buffer.sample(1, np.random.default_rng(7))
        self.assertIsNotNone(batch.next_legal_masks)
        assert batch.next_legal_masks is not None
        np.testing.assert_allclose(batch.next_legal_masks[0], np.array([1.0, 0.0, 0.0, 1.0, 0.0], dtype=np.float32))


if __name__ == "__main__":
    unittest.main()
