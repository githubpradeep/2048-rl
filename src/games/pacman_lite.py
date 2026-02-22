from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

UP = 0
DOWN = 1
LEFT = 2
RIGHT = 3
STAY = 4

DIRS = {
    UP: (-1, 0),
    DOWN: (1, 0),
    LEFT: (0, -1),
    RIGHT: (0, 1),
    STAY: (0, 0),
}


@dataclass
class PacmanLiteConfig:
    grid_size: int = 11
    num_ghosts: int = 2
    ghost_chase_prob: float = 0.75
    pellet_reward: float = 1.0
    step_reward: float = -0.02
    ghost_collision_penalty: float = -5.0
    clear_bonus: float = 8.0
    max_steps: int = 500
    start_lives: int = 3


@dataclass
class PacmanLiteStepResult:
    done: bool
    reward: float
    pellets_eaten: int
    ghost_collision: bool
    cleared: bool


class PacmanLiteGame:
    """Small Pacman-like grid with pellets and chasing ghosts."""

    def __init__(self, config: PacmanLiteConfig | None = None, seed: int | None = None) -> None:
        self.config = config or PacmanLiteConfig()
        if self.config.grid_size < 7 or self.config.grid_size % 2 == 0:
            raise ValueError("grid_size must be odd and >= 7")
        if self.config.num_ghosts < 1 or self.config.num_ghosts > 4:
            raise ValueError("num_ghosts must be in [1,4]")
        if not (0.0 <= self.config.ghost_chase_prob <= 1.0):
            raise ValueError("ghost_chase_prob must be in [0,1]")

        self.rng = np.random.default_rng(seed)
        self.grid_size = self.config.grid_size
        self.walls = self._build_maze(self.grid_size)

        self.player_pos = (1, 1)
        self.ghost_positions: list[tuple[int, int]] = []
        self.pellets = np.zeros((self.grid_size, self.grid_size), dtype=np.int8)

        self.score = 0
        self.steps = 0
        self.lives = self.config.start_lives
        self.total_pellets_eaten = 0
        self.total_collisions = 0
        self.game_over = False

    @staticmethod
    def _build_maze(n: int) -> np.ndarray:
        walls = np.zeros((n, n), dtype=np.int8)
        walls[0, :] = 1
        walls[-1, :] = 1
        walls[:, 0] = 1
        walls[:, -1] = 1

        # Bomberman-like pillars create corridors while preserving connectivity.
        for r in range(2, n - 2, 2):
            for c in range(2, n - 2, 2):
                walls[r, c] = 1

        # Add a couple of deterministic bars with gaps for a Pacman-ish feel.
        if n >= 9:
            mid = n // 2
            walls[mid, 2 : n - 2] = 1
            walls[mid, mid] = 0
            walls[mid, 2] = 0
            walls[mid, n - 3] = 0
            walls[2 : n - 2, mid] = 1
            walls[mid, mid] = 0
            walls[2, mid] = 0
            walls[n - 3, mid] = 0

        return walls

    def _ghost_spawn_points(self) -> list[tuple[int, int]]:
        n = self.grid_size
        points = [
            (n - 2, n - 2),
            (1, n - 2),
            (n - 2, 1),
            (n // 2, n // 2 + 1 if n // 2 + 1 < n - 1 else n // 2 - 1),
        ]
        valid = [(r, c) for (r, c) in points if self.walls[r, c] == 0 and (r, c) != (1, 1)]
        return valid[: self.config.num_ghosts]

    def _reset_entities(self) -> None:
        self.player_pos = (1, 1)
        self.ghost_positions = self._ghost_spawn_points()

    def _reset_pellets(self) -> None:
        self.pellets = (1 - self.walls).astype(np.int8)
        self.pellets[self.walls == 1] = 0
        self.pellets[self.player_pos] = 0
        for g in self.ghost_positions:
            self.pellets[g] = 0

    def reset(self, seed: int | None = None) -> np.ndarray:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self._reset_entities()
        self._reset_pellets()
        self.score = 0
        self.steps = 0
        self.lives = self.config.start_lives
        self.total_pellets_eaten = 0
        self.total_collisions = 0
        self.game_over = False
        return self.board()

    def _in_bounds(self, pos: tuple[int, int]) -> bool:
        r, c = pos
        return 0 <= r < self.grid_size and 0 <= c < self.grid_size

    def _is_open(self, pos: tuple[int, int]) -> bool:
        r, c = pos
        return self._in_bounds(pos) and self.walls[r, c] == 0

    def _move(self, pos: tuple[int, int], action: int) -> tuple[int, int]:
        dr, dc = DIRS[action]
        cand = (pos[0] + dr, pos[1] + dc)
        return cand if self._is_open(cand) else pos

    def legal_actions(self) -> list[int]:
        if self.game_over:
            return []
        legal: list[int] = []
        for action in (UP, DOWN, LEFT, RIGHT, STAY):
            if self._move(self.player_pos, action) == self.player_pos and action != STAY:
                continue
            legal.append(action)
        return legal

    def _ghost_legal_actions(self, pos: tuple[int, int]) -> list[int]:
        actions = []
        for action in (UP, DOWN, LEFT, RIGHT, STAY):
            nxt = self._move(pos, action)
            if action != STAY and nxt == pos:
                continue
            actions.append(action)
        return actions or [STAY]

    def _move_ghosts(self) -> None:
        new_positions: list[tuple[int, int]] = []
        for pos in self.ghost_positions:
            legal = self._ghost_legal_actions(pos)
            if self.rng.random() < self.config.ghost_chase_prob:
                best_dist = None
                best_actions: list[int] = []
                for a in legal:
                    nxt = self._move(pos, a)
                    dist = abs(nxt[0] - self.player_pos[0]) + abs(nxt[1] - self.player_pos[1])
                    if best_dist is None or dist < best_dist:
                        best_dist = dist
                        best_actions = [a]
                    elif dist == best_dist:
                        best_actions.append(a)
                action = int(self.rng.choice(np.asarray(best_actions, dtype=np.int64)))
            else:
                action = int(self.rng.choice(np.asarray(legal, dtype=np.int64)))
            new_positions.append(self._move(pos, action))
        self.ghost_positions = new_positions

    def pellets_left(self) -> int:
        return int(np.sum(self.pellets))

    def board(self) -> np.ndarray:
        # 0 empty, 1 wall, 2 pellet, 3 player, 4 ghost
        board = np.zeros((self.grid_size, self.grid_size), dtype=np.int32)
        board[self.walls == 1] = 1
        board[self.pellets == 1] = 2
        for gr, gc in self.ghost_positions:
            board[gr, gc] = 4
        pr, pc = self.player_pos
        board[pr, pc] = 3
        return board

    def step(self, action: int) -> PacmanLiteStepResult:
        if action not in DIRS:
            raise ValueError(f"Invalid action {action}")
        if self.game_over:
            return PacmanLiteStepResult(done=True, reward=0.0, pellets_eaten=0, ghost_collision=False, cleared=False)

        self.steps += 1
        reward = float(self.config.step_reward)
        pellets_eaten = 0
        ghost_collision = False
        cleared = False

        # Player move
        self.player_pos = self._move(self.player_pos, action)

        # Eat pellet
        if self.pellets[self.player_pos] == 1:
            self.pellets[self.player_pos] = 0
            pellets_eaten = 1
            self.score += 1
            self.total_pellets_eaten += 1
            reward += float(self.config.pellet_reward)

        # Collision before ghost movement (running into ghost)
        if self.player_pos in self.ghost_positions:
            ghost_collision = True
        else:
            self._move_ghosts()
            if self.player_pos in self.ghost_positions:
                ghost_collision = True

        if ghost_collision:
            self.total_collisions += 1
            self.lives -= 1
            reward += float(self.config.ghost_collision_penalty)
            if self.lives > 0:
                # Respawn entities, keep pellets/progress.
                self._reset_entities()
                self.pellets[self.player_pos] = 0
                for g in self.ghost_positions:
                    self.pellets[g] = 0

        if self.pellets_left() == 0:
            cleared = True
            reward += float(self.config.clear_bonus)

        done = bool(self.lives <= 0 or cleared or self.steps >= self.config.max_steps)
        self.game_over = done
        return PacmanLiteStepResult(done=done, reward=float(reward), pellets_eaten=pellets_eaten, ghost_collision=ghost_collision, cleared=cleared)

    def render(self) -> str:
        board = self.board()
        chars = {0: ' ', 1: '#', 2: '.', 3: 'P', 4: 'G'}
        lines: List[str] = []
        border = '+' + '-' * self.grid_size + '+'
        lines.append(border)
        for r in range(self.grid_size):
            row = ''.join(chars[int(v)] for v in board[r])
            lines.append('|' + row + '|')
        lines.append(border)
        lines.append(f"Score: {self.score}  Lives: {self.lives}  Pellets left: {self.pellets_left()}  Steps: {self.steps}")
        return '\n'.join(lines)


class PacmanLiteEnv:
    """RL wrapper. State channels: wall, pellet, player, ghost."""

    action_size = 5

    def __init__(self, config: PacmanLiteConfig | None = None, seed: int | None = None, state_grid_size: int | None = None) -> None:
        self.game = PacmanLiteGame(config=config, seed=seed)
        self.state_grid_size = int(state_grid_size) if state_grid_size is not None else self.game.grid_size
        if self.state_grid_size < self.game.grid_size:
            raise ValueError("state_grid_size must be >= game grid_size")

    def reset(self, seed: int | None = None) -> np.ndarray:
        self.game.reset(seed=seed)
        return self.get_state()

    def legal_actions(self) -> list[int]:
        return self.game.legal_actions()

    @staticmethod
    def encode_board(board: np.ndarray, state_grid_size: int | None = None) -> np.ndarray:
        h, w = board.shape
        size = int(state_grid_size) if state_grid_size is not None else h
        if size < h or size < w:
            raise ValueError("state_grid_size must be >= board size")
        state = np.zeros((4, size, size), dtype=np.float32)
        off_r = (size - h) // 2
        off_c = (size - w) // 2
        rows, cols = np.where(board == 1)
        state[0, rows + off_r, cols + off_c] = 1.0
        rows, cols = np.where(board == 2)
        state[1, rows + off_r, cols + off_c] = 1.0
        rows, cols = np.where(board == 3)
        state[2, rows + off_r, cols + off_c] = 1.0
        rows, cols = np.where(board == 4)
        state[3, rows + off_r, cols + off_c] = 1.0
        return state.reshape(-1)

    def get_state(self) -> np.ndarray:
        return self.encode_board(self.game.board(), state_grid_size=self.state_grid_size)

    def step(self, action: int):
        result = self.game.step(action)
        info = {
            "score": int(self.game.score),
            "steps": int(self.game.steps),
            "lives": int(self.game.lives),
            "pellets_left": int(self.game.pellets_left()),
            "pellets_eaten": int(result.pellets_eaten),
            "ghost_collision": bool(result.ghost_collision),
            "cleared": bool(result.cleared),
            "total_pellets_eaten": int(self.game.total_pellets_eaten),
            "total_collisions": int(self.game.total_collisions),
        }
        return self.get_state(), float(result.reward), bool(result.done), info

    def render(self) -> str:
        return self.game.render()
