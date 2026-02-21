from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np


LEFT = 0
STAY = 1
RIGHT = 2
SHOOT = 3


@dataclass
class ShooterConfig:
    grid_size: int = 10
    spawn_prob: float = 0.35
    kill_reward: float = 2.5
    escape_penalty: float = -1.0
    player_hit_penalty: float = -6.0
    step_reward: float = 0.05
    max_steps: int = 600
    start_lives: int = 3


@dataclass
class ShooterStepResult:
    done: bool
    reward: float
    kills: int
    escaped: int
    player_hits: int


class ShooterGame:
    """Arcade shooter: dodge descending enemies and shoot them."""

    def __init__(self, config: ShooterConfig | None = None, seed: int | None = None) -> None:
        self.config = config or ShooterConfig()
        if self.config.grid_size < 6:
            raise ValueError("grid_size must be at least 6")

        self.rng = np.random.default_rng(seed)
        self.grid_size = self.config.grid_size

        self.player_col = self.grid_size // 2
        self.enemies: List[Tuple[int, int]] = []
        self.bullets: List[Tuple[int, int]] = []

        self.score = 0
        self.steps = 0
        self.lives = self.config.start_lives
        self.total_kills = 0
        self.total_escaped = 0

    def reset(self, seed: int | None = None) -> np.ndarray:
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self.player_col = self.grid_size // 2
        self.enemies = []
        self.bullets = []

        self.score = 0
        self.steps = 0
        self.lives = self.config.start_lives
        self.total_kills = 0
        self.total_escaped = 0
        return self.board()

    def legal_actions(self) -> list[int]:
        return [LEFT, STAY, RIGHT, SHOOT]

    def _spawn_enemy(self) -> None:
        if self.rng.random() >= self.config.spawn_prob:
            return
        col = int(self.rng.integers(0, self.grid_size))
        occupied_top = {(r, c) for r, c in self.enemies if r == 0}
        if (0, col) in occupied_top:
            return
        self.enemies.append((0, col))

    def board(self) -> np.ndarray:
        # 0 empty, 1 enemy, 2 bullet, 3 player
        board = np.zeros((self.grid_size, self.grid_size), dtype=np.int32)
        for r, c in self.enemies:
            board[r, c] = 1
        for r, c in self.bullets:
            if 0 <= r < self.grid_size:
                board[r, c] = 2
        board[self.grid_size - 1, self.player_col] = 3
        return board

    def step(self, action: int) -> ShooterStepResult:
        if action not in (LEFT, STAY, RIGHT, SHOOT):
            raise ValueError(f"Invalid action {action}")

        self.steps += 1

        if action == LEFT:
            self.player_col = max(0, self.player_col - 1)
        elif action == RIGHT:
            self.player_col = min(self.grid_size - 1, self.player_col + 1)
        elif action == SHOOT:
            self.bullets.append((self.grid_size - 2, self.player_col))

        self._spawn_enemy()

        # Move bullets up.
        moved_bullets: List[Tuple[int, int]] = []
        for r, c in self.bullets:
            nr = r - 1
            if nr >= 0:
                moved_bullets.append((nr, c))
        self.bullets = moved_bullets

        # Move enemies down.
        moved_enemies: List[Tuple[int, int]] = []
        for r, c in self.enemies:
            moved_enemies.append((r + 1, c))
        self.enemies = moved_enemies

        bullet_set = set(self.bullets)
        enemy_set = set(self.enemies)
        collided = bullet_set & enemy_set

        kills = len(collided)
        if kills > 0:
            self.total_kills += kills
            self.score += kills

        self.bullets = [pos for pos in self.bullets if pos not in collided]
        self.enemies = [pos for pos in self.enemies if pos not in collided]

        escaped = 0
        player_hits = 0
        remaining_enemies: List[Tuple[int, int]] = []
        bottom = self.grid_size - 1

        for r, c in self.enemies:
            if r < bottom:
                remaining_enemies.append((r, c))
                continue

            if c == self.player_col:
                player_hits += 1
                self.lives -= 1
            else:
                escaped += 1
                self.total_escaped += 1

        self.enemies = remaining_enemies

        reward = self.config.step_reward
        reward += kills * self.config.kill_reward
        reward += escaped * self.config.escape_penalty
        reward += player_hits * self.config.player_hit_penalty

        done = self.lives <= 0 or self.steps >= self.config.max_steps

        return ShooterStepResult(
            done=done,
            reward=float(reward),
            kills=kills,
            escaped=escaped,
            player_hits=player_hits,
        )

    def render(self) -> str:
        board = self.board()
        lines: List[str] = []
        border = "+" + "-" * self.grid_size + "+"
        lines.append(border)
        for r in range(self.grid_size):
            row = []
            for c in range(self.grid_size):
                v = int(board[r, c])
                if v == 0:
                    row.append(" ")
                elif v == 1:
                    row.append("E")
                elif v == 2:
                    row.append("|")
                else:
                    row.append("A")
            lines.append("|" + "".join(row) + "|")
        lines.append(border)
        lines.append(f"Score: {self.score}  Lives: {self.lives}  Steps: {self.steps}")
        return "\n".join(lines)


class ShooterEnv:
    """RL wrapper for Shooter. State channels: enemy, bullet, player."""

    action_size = 4

    def __init__(
        self,
        config: ShooterConfig | None = None,
        seed: int | None = None,
        state_grid_size: int | None = None,
    ) -> None:
        self.game = ShooterGame(config=config, seed=seed)
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

        enemy_rows, enemy_cols = np.where(board == 1)
        bullet_rows, bullet_cols = np.where(board == 2)
        player_rows, player_cols = np.where(board == 3)

        state[0, enemy_rows + offset, enemy_cols + offset] = 1.0
        state[1, bullet_rows + offset, bullet_cols + offset] = 1.0
        state[2, player_rows + offset, player_cols + offset] = 1.0
        return state.reshape(-1)

    def step(self, action: int):
        result = self.game.step(action)
        info = {
            "score": int(self.game.score),
            "steps": int(self.game.steps),
            "lives": int(self.game.lives),
            "kills": int(result.kills),
            "escaped": int(result.escaped),
            "player_hits": int(result.player_hits),
            "total_kills": int(self.game.total_kills),
            "total_escaped": int(self.game.total_escaped),
        }
        return self.get_state(), float(result.reward), bool(result.done), info

    def render(self) -> str:
        return self.game.render()
