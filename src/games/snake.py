from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np


UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3

DIRECTION_VECTORS: Dict[int, Tuple[int, int]] = {
    UP: (-1, 0),
    RIGHT: (0, 1),
    DOWN: (1, 0),
    LEFT: (0, -1),
}

OPPOSITE_DIRECTION: Dict[int, int] = {
    UP: DOWN,
    DOWN: UP,
    LEFT: RIGHT,
    RIGHT: LEFT,
}


def turn_left(direction: int) -> int:
    return (direction - 1) % 4


def turn_right(direction: int) -> int:
    return (direction + 1) % 4


@dataclass
class SnakeStepResult:
    done: bool
    ate_food: bool
    collision: bool
    reward: float
    reversed_input: bool


@dataclass
class SnakeConfig:
    grid_size: int = 10
    food_reward: float = 10.0
    step_reward: float = 0.1
    death_penalty: float = -10.0
    reverse_penalty: float = -0.2
    distance_reward_scale: float = 0.2
    max_steps_without_food: int = 150


class SnakeGame:
    """Pure Snake game logic with deterministic RNG support."""

    def __init__(self, config: SnakeConfig | None = None, seed: int | None = None) -> None:
        self.config = config or SnakeConfig()
        if self.config.grid_size < 4:
            raise ValueError("grid_size must be at least 4")

        self.rng = np.random.default_rng(seed)
        self.grid_size = self.config.grid_size

        self.snake: List[Tuple[int, int]] = []
        self.direction = RIGHT
        self.food: Tuple[int, int] = (0, 0)
        self.score = 0
        self.steps = 0
        self.steps_since_food = 0

    def reset(self, seed: int | None = None) -> np.ndarray:
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        center = self.grid_size // 2
        # Horizontal start: tail -> head
        self.snake = [
            (center, center - 2),
            (center, center - 1),
            (center, center),
        ]
        self.direction = RIGHT
        self.score = 0
        self.steps = 0
        self.steps_since_food = 0
        self.food = self._spawn_food()
        return self.board()

    def _spawn_food(self) -> Tuple[int, int]:
        occupied = set(self.snake)
        empties = [(r, c) for r in range(self.grid_size) for c in range(self.grid_size) if (r, c) not in occupied]
        if not empties:
            return self.snake[-1]
        idx = int(self.rng.integers(0, len(empties)))
        return empties[idx]

    def board(self) -> np.ndarray:
        board = np.zeros((self.grid_size, self.grid_size), dtype=np.int32)
        for r, c in self.snake[:-1]:
            board[r, c] = 1
        hr, hc = self.snake[-1]
        board[hr, hc] = 2
        fr, fc = self.food
        board[fr, fc] = 3
        return board

    def legal_actions(self) -> list[int]:
        return [action for action in range(4) if action != OPPOSITE_DIRECTION[self.direction]]

    @staticmethod
    def _manhattan(a: Tuple[int, int], b: Tuple[int, int]) -> int:
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def would_collide(self, next_head: Tuple[int, int]) -> bool:
        ate_food = next_head == self.food
        out_of_bounds = not (0 <= next_head[0] < self.grid_size and 0 <= next_head[1] < self.grid_size)
        # Moving into the current tail cell is legal when not eating because the tail moves away.
        occupied = self.snake if ate_food else self.snake[1:]
        hit_body = next_head in occupied
        return out_of_bounds or hit_body

    def step(self, action: int) -> SnakeStepResult:
        if action not in DIRECTION_VECTORS:
            raise ValueError(f"Invalid action {action}")

        reversed_input = False
        if action == OPPOSITE_DIRECTION[self.direction]:
            # Ignore reverse and keep current direction.
            action = self.direction
            reversed_input = True

        self.direction = action
        self.steps += 1
        self.steps_since_food += 1

        dr, dc = DIRECTION_VECTORS[self.direction]
        head_r, head_c = self.snake[-1]
        old_distance = self._manhattan((head_r, head_c), self.food)
        next_head = (head_r + dr, head_c + dc)
        ate_food = next_head == self.food

        if self.would_collide(next_head):
            reward = self.config.death_penalty
            if reversed_input:
                reward += self.config.reverse_penalty
            return SnakeStepResult(done=True, ate_food=False, collision=True, reward=reward, reversed_input=reversed_input)

        self.snake.append(next_head)

        if ate_food:
            self.score += 1
            self.steps_since_food = 0
            if len(self.snake) < self.grid_size * self.grid_size:
                self.food = self._spawn_food()
            reward = self.config.food_reward
        else:
            self.snake.pop(0)
            reward = self.config.step_reward
            new_distance = self._manhattan(next_head, self.food)
            reward += self.config.distance_reward_scale * float(old_distance - new_distance)

        done = False
        if self.steps_since_food >= self.config.max_steps_without_food:
            done = True
            reward += self.config.death_penalty

        if reversed_input:
            reward += self.config.reverse_penalty

        # Win condition: fill the board.
        if len(self.snake) == self.grid_size * self.grid_size:
            done = True

        return SnakeStepResult(
            done=done,
            ate_food=ate_food,
            collision=False,
            reward=float(reward),
            reversed_input=reversed_input,
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
                    row_chars.append("o")
                elif value == 2:
                    row_chars.append("O")
                else:
                    row_chars.append("*")
            lines.append("|" + "".join(row_chars) + "|")
        lines.append(border)
        lines.append(f"Score: {self.score}  Length: {len(self.snake)}  Steps: {self.steps}")
        return "\n".join(lines)


class SnakeEnv:
    """RL environment wrapper for Snake using a 3-channel board encoding."""

    action_size = 4

    def __init__(
        self,
        config: SnakeConfig | None = None,
        seed: int | None = None,
        state_grid_size: int | None = None,
    ) -> None:
        self.game = SnakeGame(config=config, seed=seed)
        self.state_grid_size = int(state_grid_size) if state_grid_size is not None else self.game.grid_size
        if self.state_grid_size < self.game.grid_size:
            raise ValueError("state_grid_size must be >= game grid_size")

    def reset(self, seed: int | None = None) -> np.ndarray:
        self.game.reset(seed=seed)
        return self.get_state()

    def get_state(self) -> np.ndarray:
        return self.encode_board(self.game.board(), state_grid_size=self.state_grid_size)

    @staticmethod
    def encode_board(board: np.ndarray, state_grid_size: int | None = None) -> np.ndarray:
        # Channels: snake body(1), head(1), food(1)
        board_size = board.shape[0]
        size = int(state_grid_size) if state_grid_size is not None else board_size
        if size < board_size:
            raise ValueError("state_grid_size must be >= board size")

        state = np.zeros((3, size, size), dtype=np.float32)
        offset = (size - board_size) // 2

        body_rows, body_cols = np.where(board == 1)
        head_rows, head_cols = np.where(board == 2)
        food_rows, food_cols = np.where(board == 3)

        state[0, body_rows + offset, body_cols + offset] = 1.0
        state[1, head_rows + offset, head_cols + offset] = 1.0
        state[2, food_rows + offset, food_cols + offset] = 1.0
        return state.reshape(-1)

    def legal_actions(self) -> list[int]:
        return self.game.legal_actions()

    def step(self, action: int):
        result = self.game.step(action)
        info = {
            "score": int(self.game.score),
            "length": int(len(self.game.snake)),
            "steps": int(self.game.steps),
            "ate_food": bool(result.ate_food),
            "collision": bool(result.collision),
            "reversed_input": bool(result.reversed_input),
        }
        return self.get_state(), float(result.reward), bool(result.done), info

    def render(self) -> str:
        return self.game.render()


class SnakeFeatureEnv:
    """RL environment wrapper for Snake using engineered features instead of a board tensor."""

    action_size = 4

    def __init__(self, config: SnakeConfig | None = None, seed: int | None = None) -> None:
        self.game = SnakeGame(config=config, seed=seed)

    def reset(self, seed: int | None = None) -> np.ndarray:
        self.game.reset(seed=seed)
        return self.get_state()

    def _next_pos(self, origin: Tuple[int, int], direction: int) -> Tuple[int, int]:
        dr, dc = DIRECTION_VECTORS[direction]
        return (origin[0] + dr, origin[1] + dc)

    def _clear_distance(self, direction: int) -> float:
        """Free cells until collision in a ray direction, normalized to [0, 1]."""
        head = self.game.snake[-1]
        pos = head
        free = 0
        max_free = max(self.game.grid_size - 1, 1)
        for _ in range(max_free):
            pos = self._next_pos(pos, direction)
            if self.game.would_collide(pos):
                break
            free += 1
        return float(free) / float(max_free)

    def _food_relative_egocentric(self) -> Tuple[float, float, float, float]:
        head_r, head_c = self.game.snake[-1]
        food_r, food_c = self.game.food
        dr = food_r - head_r
        dc = food_c - head_c

        forward_dir = self.game.direction
        right_dir = turn_right(forward_dir)
        fvr, fvc = DIRECTION_VECTORS[forward_dir]
        rvr, rvc = DIRECTION_VECTORS[right_dir]

        forward_comp = dr * fvr + dc * fvc
        right_comp = dr * rvr + dc * rvc
        food_ahead = 1.0 if forward_comp > 0 else 0.0
        food_right = 1.0 if right_comp > 0 else 0.0
        food_behind = 1.0 if forward_comp < 0 else 0.0
        food_left = 1.0 if right_comp < 0 else 0.0
        return food_ahead, food_right, food_behind, food_left

    def get_state(self) -> np.ndarray:
        head_r, head_c = self.game.snake[-1]
        food_r, food_c = self.game.food
        grid = self.game.grid_size
        max_dist = max(2 * (grid - 1), 1)
        food_manhattan = abs(food_r - head_r) + abs(food_c - head_c)

        direction = self.game.direction
        dir_left = turn_left(direction)
        dir_right = turn_right(direction)

        next_straight = self._next_pos((head_r, head_c), direction)
        next_left = self._next_pos((head_r, head_c), dir_left)
        next_right = self._next_pos((head_r, head_c), dir_right)

        danger_straight = 1.0 if self.game.would_collide(next_straight) else 0.0
        danger_left = 1.0 if self.game.would_collide(next_left) else 0.0
        danger_right = 1.0 if self.game.would_collide(next_right) else 0.0

        food_ahead, food_right_rel, food_behind, food_left_rel = self._food_relative_egocentric()

        features = np.array(
            [
                # Immediate collision risk (relative to current heading)
                danger_straight,
                danger_left,
                danger_right,
                # Free space depth in relative directions
                self._clear_distance(direction),
                self._clear_distance(dir_left),
                self._clear_distance(dir_right),
                # Current heading (absolute one-hot)
                1.0 if direction == UP else 0.0,
                1.0 if direction == RIGHT else 0.0,
                1.0 if direction == DOWN else 0.0,
                1.0 if direction == LEFT else 0.0,
                # Food location (absolute relative to head)
                1.0 if food_r < head_r else 0.0,
                1.0 if food_c > head_c else 0.0,
                1.0 if food_r > head_r else 0.0,
                1.0 if food_c < head_c else 0.0,
                # Food location (egocentric relative to heading)
                food_ahead,
                food_left_rel,
                food_right_rel,
                food_behind,
                # Normalized geometry/progress signals
                float(food_r - head_r) / float(max(grid - 1, 1)),
                float(food_c - head_c) / float(max(grid - 1, 1)),
                float(food_manhattan) / float(max_dist),
                float(len(self.game.snake)) / float(grid * grid),
                float(self.game.steps_since_food) / float(max(self.game.config.max_steps_without_food, 1)),
            ],
            dtype=np.float32,
        )
        return features

    def legal_actions(self) -> list[int]:
        return self.game.legal_actions()

    def step(self, action: int):
        result = self.game.step(action)
        info = {
            "score": int(self.game.score),
            "length": int(len(self.game.snake)),
            "steps": int(self.game.steps),
            "ate_food": bool(result.ate_food),
            "collision": bool(result.collision),
            "reversed_input": bool(result.reversed_input),
        }
        return self.get_state(), float(result.reward), bool(result.done), info

    def render(self) -> str:
        return self.game.render()
