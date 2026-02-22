from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np


NO_FLAP = 0
FLAP = 1


@dataclass
class FlappyConfig:
    width: int = 84
    height: int = 84
    bird_x: float = 20.0
    gravity: float = 0.18
    flap_velocity: float = -2.2
    max_velocity: float = 3.0
    pipe_width: float = 14.0
    gap_size: float = 30.0
    pipe_speed: float = 1.3
    pipe_spacing: float = 44.0
    initial_pipe_offset: float = 24.0
    gap_margin: float = 8.0
    floor_height: float = 10.0
    max_gap_delta: float | None = 8.0
    step_reward: float = 0.1
    pass_reward: float = 1.0
    crash_penalty: float = -5.0
    max_steps: int = 1000


@dataclass
class Pipe:
    x: float
    gap_y: float
    passed: bool = False


@dataclass
class FlappyStepResult:
    done: bool
    reward: float
    passed_pipes: int
    collision: bool


class FlappyGame:
    """Minimal Flappy Bird logic with deterministic physics and pipe spawning."""

    def __init__(self, config: FlappyConfig | None = None, seed: int | None = None) -> None:
        self.config = config or FlappyConfig()
        if self.config.width < 40 or self.config.height < 40:
            raise ValueError("Flappy board dimensions are too small")
        if self.config.gap_size <= 8:
            raise ValueError("gap_size must be > 8")
        if self.config.floor_height < 0 or self.config.floor_height >= (self.config.height - 10):
            raise ValueError("floor_height must leave enough playable air space")
        if self.config.initial_pipe_offset < 0:
            raise ValueError("initial_pipe_offset must be non-negative")

        self.rng = np.random.default_rng(seed)

        self.bird_y = self.config.height / 2.0
        self.bird_velocity = 0.0
        self.pipes: List[Pipe] = []
        self.score = 0
        self.steps = 0
        self.game_over = False

    def _playable_bottom(self) -> float:
        return float(self.config.height - self.config.floor_height)

    def _sample_gap_y(self) -> float:
        half_gap = self.config.gap_size / 2.0
        low = self.config.gap_margin + half_gap
        high = self._playable_bottom() - self.config.gap_margin - half_gap
        if high <= low:
            raise ValueError("Invalid gap configuration: reduce gap_size or margins")
        gap_y = float(self.rng.uniform(low, high))
        max_gap_delta = self.config.max_gap_delta
        if max_gap_delta is not None and self.pipes:
            prev_gap_y = float(self.pipes[-1].gap_y)
            gap_y = float(np.clip(gap_y, prev_gap_y - max_gap_delta, prev_gap_y + max_gap_delta))
            gap_y = float(np.clip(gap_y, low, high))
        return gap_y

    def _spawn_pipe(self, x: float | None = None) -> None:
        if x is None:
            if not self.pipes:
                x = self.config.width + self.config.initial_pipe_offset
            else:
                x = max(pipe.x for pipe in self.pipes) + self.config.pipe_spacing
        self.pipes.append(Pipe(x=float(x), gap_y=self._sample_gap_y()))

    def reset(self, seed: int | None = None) -> np.ndarray:
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self.bird_y = self.config.height / 2.0
        self.bird_velocity = 0.0
        self.pipes = []
        self.score = 0
        self.steps = 0
        self.game_over = False

        first_x = self.config.width + self.config.initial_pipe_offset
        self._spawn_pipe(first_x)
        self._spawn_pipe(first_x + self.config.pipe_spacing)
        return self.board()

    def legal_actions(self) -> list[int]:
        if self.game_over:
            return []
        return [NO_FLAP, FLAP]

    def _is_collision(self) -> bool:
        if self.bird_y < 0.0 or self.bird_y >= self._playable_bottom():
            return True

        half_gap = self.config.gap_size / 2.0
        for pipe in self.pipes:
            in_pipe_x = pipe.x <= self.config.bird_x <= (pipe.x + self.config.pipe_width)
            if not in_pipe_x:
                continue
            gap_top = pipe.gap_y - half_gap
            gap_bottom = pipe.gap_y + half_gap
            if self.bird_y < gap_top or self.bird_y > gap_bottom:
                return True
        return False

    def _ensure_future_pipes(self) -> None:
        while len(self.pipes) < 3:
            self._spawn_pipe(None)

    def board(self) -> np.ndarray:
        # 0 empty, 1 pipe/ground, 2 bird
        board = np.zeros((self.config.height, self.config.width), dtype=np.int32)
        half_gap = self.config.gap_size / 2.0
        playable_bottom = int(max(0, min(self.config.height, round(self._playable_bottom()))))

        for pipe in self.pipes:
            x0 = int(max(0, round(pipe.x)))
            x1 = int(min(self.config.width, round(pipe.x + self.config.pipe_width)))
            if x1 <= 0 or x0 >= self.config.width:
                continue
            gap_top = int(max(0, round(pipe.gap_y - half_gap)))
            gap_bottom = int(min(playable_bottom, round(pipe.gap_y + half_gap)))
            if gap_top > 0:
                board[:gap_top, x0:x1] = 1
            if gap_bottom < playable_bottom:
                board[gap_bottom:playable_bottom, x0:x1] = 1

        if playable_bottom < self.config.height:
            board[playable_bottom:, :] = 1

        bx = int(np.clip(round(self.config.bird_x), 0, self.config.width - 1))
        by = int(np.clip(round(self.bird_y), 0, self.config.height - 1))
        board[by, bx] = 2
        return board

    def step(self, action: int) -> FlappyStepResult:
        if action not in (NO_FLAP, FLAP):
            raise ValueError(f"Invalid action {action}")
        if self.game_over:
            return FlappyStepResult(done=True, reward=0.0, passed_pipes=0, collision=True)

        self.steps += 1
        if action == FLAP:
            self.bird_velocity = self.config.flap_velocity

        self.bird_velocity += self.config.gravity
        self.bird_velocity = float(np.clip(self.bird_velocity, -self.config.max_velocity, self.config.max_velocity))
        self.bird_y += self.bird_velocity

        for pipe in self.pipes:
            pipe.x -= self.config.pipe_speed

        passed_pipes = 0
        for pipe in self.pipes:
            if not pipe.passed and (pipe.x + self.config.pipe_width) < self.config.bird_x:
                pipe.passed = True
                passed_pipes += 1
                self.score += 1

        self.pipes = [pipe for pipe in self.pipes if (pipe.x + self.config.pipe_width) >= -2.0]
        self._ensure_future_pipes()

        collision = self._is_collision()
        self.game_over = collision
        done = collision or self.steps >= self.config.max_steps

        reward = self.config.step_reward + self.config.pass_reward * float(passed_pipes)
        if collision:
            reward += self.config.crash_penalty

        return FlappyStepResult(
            done=done,
            reward=float(reward),
            passed_pipes=passed_pipes,
            collision=collision,
        )

    def render(self) -> str:
        term_h = 20
        term_w = 32
        grid = [[" " for _ in range(term_w)] for _ in range(term_h)]

        x_scale = self.config.width / float(term_w)
        y_scale = self.config.height / float(term_h)
        half_gap = self.config.gap_size / 2.0
        floor_y = int(self._playable_bottom() / y_scale)

        for cy in range(max(0, floor_y), term_h):
            for cx in range(term_w):
                grid[cy][cx] = "="

        for pipe in self.pipes:
            x0 = int((pipe.x) / x_scale)
            x1 = int((pipe.x + self.config.pipe_width) / x_scale)
            gap_top = int((pipe.gap_y - half_gap) / y_scale)
            gap_bot = int((pipe.gap_y + half_gap) / y_scale)
            for cx in range(max(0, x0), min(term_w, x1 + 1)):
                for cy in range(0, max(0, gap_top)):
                    grid[cy][cx] = "#"
                for cy in range(min(term_h, gap_bot), min(term_h, max(0, floor_y))):
                    grid[cy][cx] = "#"

        bx = int(self.config.bird_x / x_scale)
        by = int(self.bird_y / y_scale)
        if 0 <= by < term_h and 0 <= bx < term_w:
            grid[by][bx] = "@"

        lines: list[str] = []
        border = "+" + "-" * term_w + "+"
        lines.append(border)
        for row in grid:
            lines.append("|" + "".join(row) + "|")
        lines.append(border)
        lines.append(
            f"Score: {self.score}  Steps: {self.steps}  Y: {self.bird_y:.1f}  Vy: {self.bird_velocity:.2f}"
        )
        return "\n".join(lines)


class FlappyEnv:
    """RL wrapper for Flappy Bird. State is low-dimensional physics/pipe features."""

    action_size = 2

    def __init__(self, config: FlappyConfig | None = None, seed: int | None = None) -> None:
        self.game = FlappyGame(config=config, seed=seed)

    def reset(self, seed: int | None = None) -> np.ndarray:
        self.game.reset(seed=seed)
        return self.get_state()

    def legal_actions(self) -> list[int]:
        return self.game.legal_actions()

    def _next_pipes(self) -> list[Pipe]:
        pipes = sorted(self.game.pipes, key=lambda p: p.x)
        ahead = [p for p in pipes if (p.x + self.game.config.pipe_width) >= self.game.config.bird_x]
        if not ahead:
            return pipes[:2]
        if len(ahead) == 1:
            return [ahead[0], ahead[0]]
        return ahead[:2]

    def get_state(self) -> np.ndarray:
        cfg = self.game.config
        next_two = self._next_pipes()
        p1, p2 = next_two[0], next_two[1]
        vertical_scale = float(max(self.game._playable_bottom(), 1e-6))

        state = np.asarray(
            [
                self.game.bird_y / vertical_scale,
                self.game.bird_velocity / float(max(cfg.max_velocity, 1e-6)),
                (p1.x - cfg.bird_x) / float(cfg.width),
                p1.gap_y / vertical_scale,
                (self.game.bird_y - p1.gap_y) / vertical_scale,
                (p2.x - cfg.bird_x) / float(cfg.width),
                p2.gap_y / vertical_scale,
                (self.game.bird_y - p2.gap_y) / vertical_scale,
            ],
            dtype=np.float32,
        )
        return state

    def step(self, action: int):
        result = self.game.step(action)
        info = {
            "score": int(self.game.score),
            "steps": int(self.game.steps),
            "bird_y": float(self.game.bird_y),
            "bird_velocity": float(self.game.bird_velocity),
            "passed_pipes": int(result.passed_pipes),
            "collision": bool(result.collision),
        }
        return self.get_state(), float(result.reward), bool(result.done), info

    def render(self) -> str:
        return self.game.render()
