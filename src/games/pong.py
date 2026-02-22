from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np


UP = 0
STAY = 1
DOWN = 2


@dataclass
class PongConfig:
    width: int = 12
    height: int = 16
    paddle_height: int = 4
    paddle_speed: int = 1
    opponent_track_prob: float = 0.85
    start_lives: int = 3
    step_reward: float = 0.01
    paddle_hit_reward: float = 0.05
    score_reward: float = 2.0
    concede_penalty: float = -2.0
    max_steps: int = 1000


@dataclass
class PongStepResult:
    done: bool
    reward: float
    player_hit: bool
    opponent_hit: bool
    player_scored: bool
    opponent_scored: bool


class PongGame:
    """Single-agent Pong vs scripted opponent."""

    def __init__(self, config: PongConfig | None = None, seed: int | None = None) -> None:
        self.config = config or PongConfig()
        if self.config.width < 8:
            raise ValueError("width must be at least 8")
        if self.config.height < 8:
            raise ValueError("height must be at least 8")
        if self.config.paddle_height < 1 or self.config.paddle_height >= self.config.height:
            raise ValueError("paddle_height must be in [1, height)")
        if self.config.paddle_speed < 1:
            raise ValueError("paddle_speed must be >= 1")
        if not (0.0 <= self.config.opponent_track_prob <= 1.0):
            raise ValueError("opponent_track_prob must be in [0,1]")

        self.rng = np.random.default_rng(seed)

        self.player_y = 0
        self.opponent_y = 0
        self.ball_x = 0
        self.ball_y = 0
        self.ball_vx = 0
        self.ball_vy = 0

        self.player_score = 0
        self.opponent_score = 0
        self.player_hits = 0
        self.opponent_hits = 0
        self.lives = self.config.start_lives
        self.steps = 0
        self.game_over = False

    @property
    def left_paddle_col(self) -> int:
        return 0

    @property
    def right_paddle_col(self) -> int:
        return self.config.width - 1

    def _clamp_paddle(self, y: int) -> int:
        return int(np.clip(y, 0, self.config.height - self.config.paddle_height))

    def _paddle_center(self, y: int) -> float:
        return float(y + (self.config.paddle_height - 1) / 2.0)

    def _serve(self, toward_player: bool | None = None) -> None:
        self.ball_x = self.config.width // 2
        self.ball_y = self.config.height // 2
        if toward_player is None:
            toward_player = bool(self.rng.integers(0, 2))
        self.ball_vx = -1 if toward_player else 1
        self.ball_vy = int(self.rng.choice(np.asarray([-1, 1], dtype=np.int64)))
        if self.ball_vy == 0:
            self.ball_vy = 1

    def reset(self, seed: int | None = None) -> np.ndarray:
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        mid = (self.config.height - self.config.paddle_height) // 2
        self.player_y = mid
        self.opponent_y = mid
        self._serve(toward_player=bool(self.rng.integers(0, 2)))

        self.player_score = 0
        self.opponent_score = 0
        self.player_hits = 0
        self.opponent_hits = 0
        self.lives = self.config.start_lives
        self.steps = 0
        self.game_over = False
        return self.board()

    def legal_actions(self) -> list[int]:
        if self.game_over:
            return []
        return [UP, STAY, DOWN]

    def _move_player(self, action: int) -> None:
        if action == UP:
            self.player_y = self._clamp_paddle(self.player_y - self.config.paddle_speed)
        elif action == DOWN:
            self.player_y = self._clamp_paddle(self.player_y + self.config.paddle_speed)

    def _move_opponent(self) -> None:
        if self.rng.random() > self.config.opponent_track_prob:
            return
        target = self.ball_y
        center = self._paddle_center(self.opponent_y)
        if target < center - 0.25:
            self.opponent_y = self._clamp_paddle(self.opponent_y - self.config.paddle_speed)
        elif target > center + 0.25:
            self.opponent_y = self._clamp_paddle(self.opponent_y + self.config.paddle_speed)

    def _in_paddle(self, paddle_y: int, y: int) -> bool:
        return paddle_y <= y < (paddle_y + self.config.paddle_height)

    def _bounce_from_paddle(self, paddle_y: int, is_player: bool) -> None:
        center = self._paddle_center(paddle_y)
        offset = self.ball_y - center
        if offset < -0.5:
            self.ball_vy = -1
        elif offset > 0.5:
            self.ball_vy = 1
        else:
            # keep or randomize slight vertical motion to avoid horizontal loops
            if self.ball_vy == 0:
                self.ball_vy = int(self.rng.choice(np.asarray([-1, 1], dtype=np.int64)))
        self.ball_vx = 1 if is_player else -1

    def board(self) -> np.ndarray:
        # 0 empty, 1 player paddle, 2 opponent paddle, 3 ball
        board = np.zeros((self.config.height, self.config.width), dtype=np.int32)
        board[self.player_y : self.player_y + self.config.paddle_height, self.left_paddle_col] = 1
        board[self.opponent_y : self.opponent_y + self.config.paddle_height, self.right_paddle_col] = 2
        bx = int(np.clip(self.ball_x, 0, self.config.width - 1))
        by = int(np.clip(self.ball_y, 0, self.config.height - 1))
        board[by, bx] = 3
        return board

    def step(self, action: int) -> PongStepResult:
        if action not in (UP, STAY, DOWN):
            raise ValueError(f"Invalid action {action}")
        if self.game_over:
            return PongStepResult(True, 0.0, False, False, False, False)

        self.steps += 1
        self._move_player(action)
        self._move_opponent()

        reward = float(self.config.step_reward)
        player_hit = False
        opponent_hit = False
        player_scored = False
        opponent_scored = False

        next_x = self.ball_x + self.ball_vx
        next_y = self.ball_y + self.ball_vy

        if next_y < 0 or next_y >= self.config.height:
            self.ball_vy *= -1
            next_y = self.ball_y + self.ball_vy

        # Left side (player paddle / concede)
        if next_x <= self.left_paddle_col:
            if self._in_paddle(self.player_y, next_y):
                self.ball_x = self.left_paddle_col
                self.ball_y = int(np.clip(next_y, 0, self.config.height - 1))
                self._bounce_from_paddle(self.player_y, is_player=True)
                player_hit = True
                self.player_hits += 1
                reward += float(self.config.paddle_hit_reward)
            else:
                opponent_scored = True
                self.opponent_score += 1
                self.lives -= 1
                reward += float(self.config.concede_penalty)
                if self.lives > 0:
                    self._serve(toward_player=True)
        # Right side (opponent paddle / player point)
        elif next_x >= self.right_paddle_col:
            if self._in_paddle(self.opponent_y, next_y):
                self.ball_x = self.right_paddle_col
                self.ball_y = int(np.clip(next_y, 0, self.config.height - 1))
                self._bounce_from_paddle(self.opponent_y, is_player=False)
                opponent_hit = True
                self.opponent_hits += 1
            else:
                player_scored = True
                self.player_score += 1
                reward += float(self.config.score_reward)
                self._serve(toward_player=False)
        else:
            self.ball_x = int(next_x)
            self.ball_y = int(np.clip(next_y, 0, self.config.height - 1))

        done = bool(self.lives <= 0 or self.steps >= self.config.max_steps)
        self.game_over = done

        return PongStepResult(
            done=done,
            reward=float(reward),
            player_hit=player_hit,
            opponent_hit=opponent_hit,
            player_scored=player_scored,
            opponent_scored=opponent_scored,
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
                    row_chars.append("|")
                elif v == 2:
                    row_chars.append("!")
                else:
                    row_chars.append("o")
            lines.append("|" + "".join(row_chars) + "|")
        lines.append(border)
        lines.append(
            f"Player: {self.player_score}  Opp: {self.opponent_score}  Lives: {self.lives}  Steps: {self.steps}  "
            f"Ball: ({self.ball_x},{self.ball_y})  Vel: ({self.ball_vx},{self.ball_vy})"
        )
        return "\n".join(lines)


class PongEnv:
    """RL wrapper for Pong.

    State = board channels (player paddle, opp paddle, ball) + normalized physics scalars.
    """

    action_size = 3

    def __init__(self, config: PongConfig | None = None, seed: int | None = None) -> None:
        self.game = PongGame(config=config, seed=seed)

    def reset(self, seed: int | None = None) -> np.ndarray:
        self.game.reset(seed=seed)
        return self.get_state()

    def legal_actions(self) -> list[int]:
        return self.game.legal_actions()

    @staticmethod
    def encode_board(board: np.ndarray) -> np.ndarray:
        h, w = board.shape
        state = np.zeros((3, h, w), dtype=np.float32)
        p_rows, p_cols = np.where(board == 1)
        o_rows, o_cols = np.where(board == 2)
        b_rows, b_cols = np.where(board == 3)
        state[0, p_rows, p_cols] = 1.0
        state[1, o_rows, o_cols] = 1.0
        state[2, b_rows, b_cols] = 1.0
        return state.reshape(-1)

    def get_state(self) -> np.ndarray:
        board_features = self.encode_board(self.game.board())
        cfg = self.game.config
        extra = np.asarray(
            [
                float(self.game.ball_x) / float(max(cfg.width - 1, 1)),
                float(self.game.ball_y) / float(max(cfg.height - 1, 1)),
                float(self.game.ball_vx),
                float(self.game.ball_vy),
                float(self.game.player_y) / float(max(cfg.height - cfg.paddle_height, 1)),
                float(self.game.opponent_y) / float(max(cfg.height - cfg.paddle_height, 1)),
                float(self.game.lives) / float(max(cfg.start_lives, 1)),
            ],
            dtype=np.float32,
        )
        return np.concatenate([board_features, extra], axis=0)

    def step(self, action: int):
        result = self.game.step(action)
        info = {
            "score": int(self.game.player_score),
            "opponent_score": int(self.game.opponent_score),
            "steps": int(self.game.steps),
            "lives": int(self.game.lives),
            "player_hits": int(self.game.player_hits),
            "opponent_hits": int(self.game.opponent_hits),
            "player_hit": bool(result.player_hit),
            "opponent_hit": bool(result.opponent_hit),
            "player_scored": bool(result.player_scored),
            "opponent_scored": bool(result.opponent_scored),
        }
        return self.get_state(), float(result.reward), bool(result.done), info

    def render(self) -> str:
        return self.game.render()

