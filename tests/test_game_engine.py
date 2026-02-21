import unittest

import numpy as np

from src.game_engine import Game2048


class TestGame2048(unittest.TestCase):
    def test_merge_chain_rule(self) -> None:
        row = np.array([2, 2, 2, 0], dtype=np.int32)
        merged, gain = Game2048._compress_and_merge(row)
        np.testing.assert_array_equal(merged, np.array([4, 2, 0, 0], dtype=np.int32))
        self.assertEqual(gain, 4)

    def test_double_pair_merge(self) -> None:
        row = np.array([2, 2, 2, 2], dtype=np.int32)
        merged, gain = Game2048._compress_and_merge(row)
        np.testing.assert_array_equal(merged, np.array([4, 4, 0, 0], dtype=np.int32))
        self.assertEqual(gain, 8)

    def test_apply_move_up(self) -> None:
        game = Game2048(seed=1)
        game.board = np.array(
            [
                [2, 0, 0, 2],
                [2, 2, 0, 0],
                [0, 2, 0, 0],
                [0, 0, 0, 0],
            ],
            dtype=np.int32,
        )
        moved_board, gain, moved = game._apply_move(0)
        expected = np.array(
            [
                [4, 4, 0, 2],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
            ],
            dtype=np.int32,
        )
        np.testing.assert_array_equal(moved_board, expected)
        self.assertEqual(gain, 8)
        self.assertTrue(moved)

    def test_invalid_move_has_no_gain(self) -> None:
        game = Game2048(seed=1)
        game.board = np.array(
            [
                [2, 4, 8, 16],
                [32, 64, 128, 256],
                [512, 1024, 2, 4],
                [8, 16, 32, 64],
            ],
            dtype=np.int32,
        )
        result = game.step(2)
        self.assertFalse(result.moved)
        self.assertEqual(result.gain, 0)
        self.assertTrue(result.done)

    def test_can_move_false_when_blocked(self) -> None:
        game = Game2048(seed=1)
        game.board = np.array(
            [
                [2, 4, 2, 4],
                [4, 2, 4, 2],
                [2, 4, 2, 4],
                [4, 2, 4, 2],
            ],
            dtype=np.int32,
        )
        self.assertFalse(game.can_move())


if __name__ == "__main__":
    unittest.main()
