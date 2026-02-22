from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np


ACTION_TO_NAME = {
    0: "up",
    1: "down",
    2: "left",
    3: "right",
}


@dataclass
class MoveResult:
    moved: bool
    gain: int
    done: bool


class Game2048:
    """Pure 2048 game logic with deterministic RNG support."""

    def __init__(self, size: int = 4, seed: int | None = None) -> None:
        if size < 2:
            raise ValueError("size must be at least 2")
        self.size = size
        self.rng = np.random.default_rng(seed)
        self.board = np.zeros((size, size), dtype=np.int32)
        self.score = 0

    def reset(self, seed: int | None = None) -> np.ndarray:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.board.fill(0)
        self.score = 0
        self._spawn_tile()
        self._spawn_tile()
        return self.board.copy()

    def copy_board(self) -> np.ndarray:
        return self.board.copy()

    def _spawn_tile(self) -> bool:
        empties = np.argwhere(self.board == 0)
        if len(empties) == 0:
            return False
        idx = int(self.rng.integers(0, len(empties)))
        r, c = empties[idx]
        self.board[r, c] = 2 if self.rng.random() < 0.9 else 4
        return True

    @staticmethod
    def _compress_and_merge(row: np.ndarray) -> Tuple[np.ndarray, int]:
        non_zero = row[row != 0]
        merged: List[int] = []
        gain = 0
        i = 0
        while i < len(non_zero):
            if i + 1 < len(non_zero) and non_zero[i] == non_zero[i + 1]:
                value = int(non_zero[i] * 2)
                merged.append(value)
                gain += value
                i += 2
            else:
                merged.append(int(non_zero[i]))
                i += 1

        result = np.zeros_like(row)
        if merged:
            result[: len(merged)] = np.array(merged, dtype=row.dtype)
        return result, gain

    def _move_left(self, board: np.ndarray) -> Tuple[np.ndarray, int, bool]:
        new_board = np.zeros_like(board)
        total_gain = 0
        moved = False
        for r in range(self.size):
            merged_row, gain = self._compress_and_merge(board[r])
            new_board[r] = merged_row
            total_gain += gain
            if not np.array_equal(merged_row, board[r]):
                moved = True
        return new_board, total_gain, moved

    def _apply_move(self, action: int) -> Tuple[np.ndarray, int, bool]:
        if action not in ACTION_TO_NAME:
            raise ValueError(f"Invalid action {action}")

        if action == 2:  # left
            return self._move_left(self.board)

        if action == 3:  # right
            flipped = np.fliplr(self.board)
            moved_board, gain, moved = self._move_left(flipped)
            return np.fliplr(moved_board), gain, moved

        if action == 0:  # up
            transposed = self.board.T
            moved_board, gain, moved = self._move_left(transposed)
            return moved_board.T, gain, moved

        # down
        transposed_flipped = np.fliplr(self.board.T)
        moved_board, gain, moved = self._move_left(transposed_flipped)
        return np.fliplr(moved_board).T, gain, moved

    def step(self, action: int) -> MoveResult:
        candidate_board, gain, moved = self._apply_move(action)
        if moved:
            self.board = candidate_board
            self.score += gain
            self._spawn_tile()

        done = not self.can_move()
        return MoveResult(moved=moved, gain=gain, done=done)

    def can_move(self) -> bool:
        if np.any(self.board == 0):
            return True

        for r in range(self.size):
            for c in range(self.size - 1):
                if self.board[r, c] == self.board[r, c + 1]:
                    return True

        for r in range(self.size - 1):
            for c in range(self.size):
                if self.board[r, c] == self.board[r + 1, c]:
                    return True

        return False

    def legal_actions(self) -> List[int]:
        legal: List[int] = []
        original = self.board.copy()
        for action in range(4):
            next_board, _, moved = self._apply_move(action)
            if moved:
                legal.append(action)
            self.board = original
        return legal

    def max_tile(self) -> int:
        return int(np.max(self.board))

    def render(self) -> str:
        width = max(4, len(str(max(2048, int(np.max(self.board))))))
        sep = "+" + "+".join(["-" * (width + 2)] * self.size) + "+"
        lines = [sep]
        for row in self.board:
            formatted = []
            for value in row:
                cell = "." if value == 0 else str(int(value))
                formatted.append(cell.rjust(width))
            lines.append("| " + " | ".join(formatted) + " |")
            lines.append(sep)
        lines.append(f"Score: {self.score}")
        return "\n".join(lines)
