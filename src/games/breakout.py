from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np


LEFT = 0
STAY = 1
RIGHT = 2


@dataclass
class BreakoutConfig:
    width: int = 12
    height: int = 14
    brick_rows: int = 4
    brick_top: int = 1
    paddle_width: int = 3
    start_lives: int = 3
    step_reward: float = 0.02
    brick_reward: float = 1.0
    paddle_hit_reward: float = 0.03
    life_loss_penalty: float = -3.0
    clear_bonus: float = 5.0
    max_steps: int = 1200


@dataclass
class BreakoutStepResult:
    done: bool
    reward: float
    bricks_hit: int
    paddle_hit: bool
    life_lost: bool
    cleared: bool


class BreakoutGame:
    """Simplified Breakout game with a single ball and brick wall."""

    def __init__(self, config: BreakoutConfig | None = None, seed: int | None = None) -> None:
        self.config = config or BreakoutConfig()
        if self.config.width < 6:
            raise ValueError("width must be at least 6")
        if self.config.height < 8:
            raise ValueError("height must be at least 8")
        if self.config.paddle_width < 1 or self.config.paddle_width >= self.config.width:
            raise ValueError("paddle_width must be in [1, width)")
        if self.config.brick_rows < 1:
            raise ValueError("brick_rows must be >= 1")
        if self.config.brick_top < 0:
            raise ValueError("brick_top must be >= 0")
        if self.config.brick_top + self.config.brick_rows >= self.config.height - 2:
            raise ValueError("brick rows leave no space for ball and paddle")

        self.rng = np.random.default_rng(seed)
        self.bricks = np.zeros((self.config.height, self.config.width), dtype=np.int8)

        self.paddle_x = 0
        self.ball_x = 0
        self.ball_y = 0
        self.ball_vx = 1
        self.ball_vy = -1

        self.score = 0
        self.steps = 0
        self.lives = self.config.start_lives
        self.bricks_broken = 0
        self.total_paddle_hits = 0
        self.game_over = False

    @property
    def paddle_row(self) -> int:
        return self.config.height - 1

    def _reset_bricks(self) -> None:
        self.bricks.fill(0)
        r0 = self.config.brick_top
        r1 = r0 + self.config.brick_rows
        self.bricks[r0:r1, :] = 1

    def _random_initial_vx(self) -> int:
        return int(self.rng.choice(np.asarray([-1, 1], dtype=np.int64)))

    def _serve_ball(self) -> None:
        self.ball_x = int(np.clip(self.paddle_x + self.config.paddle_width // 2, 0, self.config.width - 1))
        self.ball_y = self.paddle_row - 1
        self.ball_vx = self._random_initial_vx()
        self.ball_vy = -1

    def reset(self, seed: int | None = None) -> np.ndarray:
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self._reset_bricks()
        self.paddle_x = (self.config.width - self.config.paddle_width) // 2
        self._serve_ball()

        self.score = 0
        self.steps = 0
        self.lives = self.config.start_lives
        self.bricks_broken = 0
        self.total_paddle_hits = 0
        self.game_over = False
        return self.board()

    def legal_actions(self) -> list[int]:
        if self.game_over:
            return []
        return [LEFT, STAY, RIGHT]

    def _paddle_range(self) -> range:
        return range(self.paddle_x, self.paddle_x + self.config.paddle_width)

    def _move_paddle(self, action: int) -> None:
        if action == LEFT:
            self.paddle_x = max(0, self.paddle_x - 1)
        elif action == RIGHT:
            self.paddle_x = min(self.config.width - self.config.paddle_width, self.paddle_x + 1)

    def _bounce_off_paddle(self, impact_x: int) -> None:
        self.ball_vy = -1
        center = self.paddle_x + (self.config.paddle_width - 1) / 2.0
        offset = impact_x - center
        if offset < -0.33:
            self.ball_vx = -1
        elif offset > 0.33:
            self.ball_vx = 1
        elif self.ball_vx == 0:
            self.ball_vx = self._random_initial_vx()
        # else keep existing horizontal velocity for center hits

    def _brick_present(self, row: int, col: int) -> bool:
        if row < 0 or row >= self.config.height or col < 0 or col >= self.config.width:
            return False
        return bool(self.bricks[row, col])

    def _remove_brick(self, row: int, col: int) -> None:
        if self._brick_present(row, col):
            self.bricks[row, col] = 0
            self.score += 1
            self.bricks_broken += 1

    def bricks_left(self) -> int:
        return int(np.sum(self.bricks))

    def board(self) -> np.ndarray:
        # 0 empty, 1 brick, 2 ball, 3 paddle
        board = np.zeros((self.config.height, self.config.width), dtype=np.int32)
        board[self.bricks > 0] = 1
        bx = int(np.clip(self.ball_x, 0, self.config.width - 1))
        by = int(np.clip(self.ball_y, 0, self.config.height - 1))
        board[by, bx] = 2
        board[self.paddle_row, self.paddle_x : self.paddle_x + self.config.paddle_width] = 3
        return board

    def step(self, action: int) -> BreakoutStepResult:
        if action not in (LEFT, STAY, RIGHT):
            raise ValueError(f"Invalid action {action}")
        if self.game_over:
            return BreakoutStepResult(done=True, reward=0.0, bricks_hit=0, paddle_hit=False, life_lost=False, cleared=False)

        self.steps += 1
        self._move_paddle(action)

        reward = float(self.config.step_reward)
        paddle_hit = False
        bricks_hit = 0
        life_lost = False
        cleared = False

        next_x = self.ball_x + self.ball_vx
        next_y = self.ball_y + self.ball_vy

        if next_x < 0 or next_x >= self.config.width:
            self.ball_vx *= -1
            next_x = self.ball_x + self.ball_vx

        if next_y < 0:
            self.ball_vy *= -1
            next_y = self.ball_y + self.ball_vy

        # Paddle collision on descent into paddle row.
        if self.ball_vy > 0 and next_y == self.paddle_row and next_x in self._paddle_range():
            paddle_hit = True
            self.total_paddle_hits += 1
            self._bounce_off_paddle(next_x)
            reward += float(self.config.paddle_hit_reward)
            next_x = self.ball_x + self.ball_vx
            next_y = self.ball_y + self.ball_vy
            if next_x < 0 or next_x >= self.config.width:
                self.ball_vx *= -1
                next_x = self.ball_x + self.ball_vx

        # Brick collision against the next occupied cell.
        if 0 <= next_y < self.paddle_row and self._brick_present(next_y, next_x):
            self._remove_brick(next_y, next_x)
            bricks_hit = 1
            reward += float(self.config.brick_reward)
            self.ball_vy *= -1
            next_y = self.ball_y + self.ball_vy
            if self.bricks_left() == 0:
                cleared = True
                reward += float(self.config.clear_bonus)

        # Update ball position after collisions.
        self.ball_x = int(np.clip(next_x, 0, self.config.width - 1))
        self.ball_y = int(next_y)

        # Missed the paddle / fell below the board.
        if self.ball_y >= self.config.height:
            life_lost = True
            self.lives -= 1
            reward += float(self.config.life_loss_penalty)
            if self.lives > 0:
                self._serve_ball()

        done = bool(cleared or self.lives <= 0 or self.steps >= self.config.max_steps)
        self.game_over = done

        return BreakoutStepResult(
            done=done,
            reward=float(reward),
            bricks_hit=bricks_hit,
            paddle_hit=paddle_hit,
            life_lost=life_lost,
            cleared=cleared,
        )

    def render(self) -> str:
        board = self.board()
        lines: List[str] = []
        border = "+" + "-" * self.config.width + "+"
        lines.append(border)
        for r in range(self.config.height):
            row_chars: list[str] = []
            for c in range(self.config.width):
                v = int(board[r, c])
                if v == 0:
                    row_chars.append(" ")
                elif v == 1:
                    row_chars.append("#")
                elif v == 2:
                    row_chars.append("o")
                else:
                    row_chars.append("=")
            lines.append("|" + "".join(row_chars) + "|")
        lines.append(border)
        lines.append(
            f"Score: {self.score}  Lives: {self.lives}  Steps: {self.steps}  "
            f"Ball: ({self.ball_x},{self.ball_y})  Vel: ({self.ball_vx},{self.ball_vy})  Bricks: {self.bricks_left()}"
        )
        return "\n".join(lines)


class BreakoutEnv:
    """RL wrapper for Breakout.

    State = one-hot board channels (brick, ball, paddle) + [ball_vx, ball_vy, lives_norm].
    """

    action_size = 3

    def __init__(self, config: BreakoutConfig | None = None, seed: int | None = None) -> None:
        self.game = BreakoutGame(config=config, seed=seed)

    def reset(self, seed: int | None = None) -> np.ndarray:
        self.game.reset(seed=seed)
        return self.get_state()

    def legal_actions(self) -> list[int]:
        return self.game.legal_actions()

    @staticmethod
    def encode_board(board: np.ndarray) -> np.ndarray:
        h, w = board.shape
        state = np.zeros((3, h, w), dtype=np.float32)
        brick_rows, brick_cols = np.where(board == 1)
        ball_rows, ball_cols = np.where(board == 2)
        paddle_rows, paddle_cols = np.where(board == 3)
        state[0, brick_rows, brick_cols] = 1.0
        state[1, ball_rows, ball_cols] = 1.0
        state[2, paddle_rows, paddle_cols] = 1.0
        return state.reshape(-1)

    def get_state(self) -> np.ndarray:
        board_features = self.encode_board(self.game.board())
        cfg = self.game.config
        extra = np.asarray(
            [
                float(self.game.ball_vx),
                float(self.game.ball_vy),
                float(self.game.lives) / float(max(cfg.start_lives, 1)),
            ],
            dtype=np.float32,
        )
        return np.concatenate([board_features, extra], axis=0)

    def step(self, action: int):
        result = self.game.step(action)
        info = {
            "score": int(self.game.score),
            "steps": int(self.game.steps),
            "lives": int(self.game.lives),
            "bricks_left": int(self.game.bricks_left()),
            "bricks_broken": int(self.game.bricks_broken),
            "bricks_hit": int(result.bricks_hit),
            "paddle_hit": bool(result.paddle_hit),
            "life_lost": bool(result.life_lost),
            "cleared": bool(result.cleared),
            "total_paddle_hits": int(self.game.total_paddle_hits),
        }
        return self.get_state(), float(result.reward), bool(result.done), info

    def render(self) -> str:
        return self.game.render()
