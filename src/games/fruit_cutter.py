from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np


LEFT = 0
STAY = 1
RIGHT = 2


@dataclass
class FruitCutterConfig:
    grid_size: int = 10
    spawn_prob: float = 0.45
    bomb_prob: float = 0.18
    slice_reward: float = 2.0
    miss_penalty: float = -0.3
    bomb_hit_penalty: float = -10.0
    step_reward: float = 0.05
    max_steps: int = 500


@dataclass
class FruitCutterStepResult:
    done: bool
    reward: float
    sliced: int
    missed: int
    bomb_hit: bool


class FruitCutterGame:
    """Simple fruit cutter game: fruits/bombs fall, player slices at bottom row."""

    def __init__(self, config: FruitCutterConfig | None = None, seed: int | None = None) -> None:
        self.config = config or FruitCutterConfig()
        if self.config.grid_size < 5:
            raise ValueError("grid_size must be at least 5")

        self.rng = np.random.default_rng(seed)
        self.grid_size = self.config.grid_size

        self.player_col = self.grid_size // 2
        self.items: List[Tuple[int, int, bool]] = []  # (row, col, is_bomb)
        self.score = 0
        self.steps = 0

    def reset(self, seed: int | None = None) -> np.ndarray:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.player_col = self.grid_size // 2
        self.items = []
        self.score = 0
        self.steps = 0
        return self.board()

    def board(self) -> np.ndarray:
        # 0 empty, 1 fruit, 2 bomb, 3 player
        board = np.zeros((self.grid_size, self.grid_size), dtype=np.int32)
        for row, col, is_bomb in self.items:
            board[row, col] = 2 if is_bomb else 1
        board[self.grid_size - 1, self.player_col] = 3
        return board

    def legal_actions(self) -> list[int]:
        return [LEFT, STAY, RIGHT]

    def _spawn_item(self) -> None:
        if self.rng.random() >= self.config.spawn_prob:
            return

        col = int(self.rng.integers(0, self.grid_size))
        is_bomb = bool(self.rng.random() < self.config.bomb_prob)

        # Avoid stacking same-cell spawns at row 0.
        occupied = {(r, c) for r, c, _ in self.items}
        if (0, col) in occupied:
            return
        self.items.append((0, col, is_bomb))

    def step(self, action: int) -> FruitCutterStepResult:
        if action not in (LEFT, STAY, RIGHT):
            raise ValueError(f"Invalid action {action}")

        self.steps += 1

        if action == LEFT:
            self.player_col = max(0, self.player_col - 1)
        elif action == RIGHT:
            self.player_col = min(self.grid_size - 1, self.player_col + 1)

        self._spawn_item()

        moved_items: List[Tuple[int, int, bool]] = []
        for row, col, is_bomb in self.items:
            moved_items.append((row + 1, col, is_bomb))
        self.items = moved_items

        reward = self.config.step_reward
        sliced = 0
        missed = 0
        bomb_hit = False

        remaining_items: List[Tuple[int, int, bool]] = []
        bottom_row = self.grid_size - 1

        for row, col, is_bomb in self.items:
            if row < bottom_row:
                remaining_items.append((row, col, is_bomb))
                continue

            # Item reached bottom row this step.
            if col == self.player_col:
                if is_bomb:
                    reward += self.config.bomb_hit_penalty
                    bomb_hit = True
                else:
                    self.score += 1
                    sliced += 1
                    reward += self.config.slice_reward
            else:
                if not is_bomb:
                    missed += 1
                    reward += self.config.miss_penalty

        self.items = remaining_items

        done = bomb_hit or (self.steps >= self.config.max_steps)
        return FruitCutterStepResult(
            done=done,
            reward=float(reward),
            sliced=sliced,
            missed=missed,
            bomb_hit=bomb_hit,
        )

    def render(self) -> str:
        board = self.board()
        lines: List[str] = []
        border = "+" + "-" * self.grid_size + "+"
        lines.append(border)
        for r in range(self.grid_size):
            row_chars = []
            for c in range(self.grid_size):
                value = int(board[r, c])
                if value == 0:
                    row_chars.append(" ")
                elif value == 1:
                    row_chars.append("F")
                elif value == 2:
                    row_chars.append("B")
                else:
                    row_chars.append("P")
            lines.append("|" + "".join(row_chars) + "|")
        lines.append(border)
        lines.append(f"Score: {self.score}  Steps: {self.steps}  PlayerCol: {self.player_col}")
        return "\n".join(lines)


class FruitCutterEnv:
    """RL wrapper for Fruit Cutter. State channels: fruit, bomb, player."""

    action_size = 3

    def __init__(
        self,
        config: FruitCutterConfig | None = None,
        seed: int | None = None,
        state_grid_size: int | None = None,
    ) -> None:
        self.game = FruitCutterGame(config=config, seed=seed)
        self.state_grid_size = int(state_grid_size) if state_grid_size is not None else self.game.grid_size
        if self.state_grid_size < self.game.grid_size:
            raise ValueError("state_grid_size must be >= game grid_size")

    def reset(self, seed: int | None = None) -> np.ndarray:
        self.game.reset(seed=seed)
        return self.get_state()

    def legal_actions(self) -> list[int]:
        return self.game.legal_actions()

    def get_state(self) -> np.ndarray:
        return self.encode_board(self.game.board(), state_grid_size=self.state_grid_size)

    @staticmethod
    def encode_board(board: np.ndarray, state_grid_size: int | None = None) -> np.ndarray:
        board_size = board.shape[0]
        size = int(state_grid_size) if state_grid_size is not None else board_size
        if size < board_size:
            raise ValueError("state_grid_size must be >= board size")

        state = np.zeros((3, size, size), dtype=np.float32)
        offset = (size - board_size) // 2

        fruit_rows, fruit_cols = np.where(board == 1)
        bomb_rows, bomb_cols = np.where(board == 2)
        player_rows, player_cols = np.where(board == 3)

        state[0, fruit_rows + offset, fruit_cols + offset] = 1.0
        state[1, bomb_rows + offset, bomb_cols + offset] = 1.0
        state[2, player_rows + offset, player_cols + offset] = 1.0
        return state.reshape(-1)

    def step(self, action: int):
        result = self.game.step(action)
        info = {
            "score": int(self.game.score),
            "steps": int(self.game.steps),
            "sliced": int(result.sliced),
            "missed": int(result.missed),
            "bomb_hit": bool(result.bomb_hit),
        }
        return self.get_state(), float(result.reward), bool(result.done), info

    def render(self) -> str:
        return self.game.render()
