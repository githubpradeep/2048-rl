from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np

from .game_engine import Game2048


@dataclass
class EnvConfig:
    invalid_move_penalty: float = -1.0
    lose_penalty: float = -5.0
    win_bonus: float = 50.0
    target_tile: int = 2048
    max_invalid_streak: int = 20


class Game2048Env:
    """Custom RL environment API for 2048 (no external RL frameworks)."""

    action_size = 4

    def __init__(self, seed: int | None = None, config: EnvConfig | None = None) -> None:
        self.game = Game2048(seed=seed)
        self.config = config or EnvConfig()
        self.invalid_streak = 0

    def reset(self, seed: int | None = None) -> np.ndarray:
        self.game.reset(seed=seed)
        self.invalid_streak = 0
        return self.get_state()

    def get_state(self) -> np.ndarray:
        return self.encode_board(self.game.board)

    @staticmethod
    def encode_board(board: np.ndarray) -> np.ndarray:
        encoded = np.zeros_like(board, dtype=np.float32)
        non_zero = board > 0
        encoded[non_zero] = np.log2(board[non_zero]) / 16.0
        return encoded.reshape(-1)

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, float | int | bool]]:
        result = self.game.step(action)
        reward = float(result.gain)
        done = result.done

        if not result.moved:
            self.invalid_streak += 1
            reward += self.config.invalid_move_penalty
            if self.invalid_streak >= self.config.max_invalid_streak:
                done = True
        else:
            self.invalid_streak = 0

        if done:
            if self.game.max_tile() >= self.config.target_tile:
                reward += self.config.win_bonus
            else:
                reward += self.config.lose_penalty

        info: Dict[str, float | int | bool] = {
            "score": int(self.game.score),
            "max_tile": int(self.game.max_tile()),
            "moved": bool(result.moved),
            "gain": int(result.gain),
            "invalid_streak": int(self.invalid_streak),
        }
        return self.get_state(), reward, done, info

    def render(self) -> str:
        return self.game.render()
