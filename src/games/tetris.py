from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np


MOVE_LEFT = 0
MOVE_RIGHT = 1
ROTATE = 2
SOFT_DROP = 3
HARD_DROP = 4
MAX_ROTATIONS = 4

PIECE_ORDER = ["I", "O", "T", "S", "Z", "J", "L"]

PIECES: Dict[str, np.ndarray] = {
    "I": np.array([[1, 1, 1, 1]], dtype=np.int32),
    "O": np.array([[1, 1], [1, 1]], dtype=np.int32),
    "T": np.array([[0, 1, 0], [1, 1, 1]], dtype=np.int32),
    "S": np.array([[0, 1, 1], [1, 1, 0]], dtype=np.int32),
    "Z": np.array([[1, 1, 0], [0, 1, 1]], dtype=np.int32),
    "J": np.array([[1, 0, 0], [1, 1, 1]], dtype=np.int32),
    "L": np.array([[0, 0, 1], [1, 1, 1]], dtype=np.int32),
}


@dataclass
class TetrisConfig:
    height: int = 20
    width: int = 10
    step_reward: float = -0.01
    soft_drop_bonus: float = 0.0
    hard_drop_bonus_scale: float = 0.0
    line_clear_reward: float = 8.0
    board_delta_scale: float = 1.0
    height_penalty: float = 0.04
    holes_penalty: float = 0.35
    bumpiness_penalty: float = 0.10
    game_over_penalty: float = -8.0
    max_steps: int = 1000


@dataclass
class TetrisStepResult:
    done: bool
    reward: float
    lines_cleared: int
    hard_drop_distance: int
    game_over: bool


class TetrisGame:
    """Pure Tetris logic (single active piece, gravity, line clear, game over)."""

    def __init__(self, config: TetrisConfig | None = None, seed: int | None = None) -> None:
        self.config = config or TetrisConfig()
        if self.config.height < 8 or self.config.width < 6:
            raise ValueError("Tetris board too small")

        self.rng = np.random.default_rng(seed)
        self.height = self.config.height
        self.width = self.config.width

        self.board = np.zeros((self.height, self.width), dtype=np.int32)
        self.current_name = "I"
        self.current_piece = PIECES[self.current_name].copy()
        self.piece_row = 0
        self.piece_col = 0

        self.score = 0
        self.lines = 0
        self.steps = 0
        self.game_over = False

    def reset(self, seed: int | None = None) -> np.ndarray:
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self.board.fill(0)
        self.score = 0
        self.lines = 0
        self.steps = 0
        self.game_over = False

        self._spawn_piece()
        return self.board_with_piece()

    def legal_actions(self) -> list[int]:
        if self.game_over:
            return []
        actions: list[int] = []
        if self._can_place(self.current_piece, self.piece_row, self.piece_col - 1):
            actions.append(MOVE_LEFT)
        if self._can_place(self.current_piece, self.piece_row, self.piece_col + 1):
            actions.append(MOVE_RIGHT)
        if self._can_rotate():
            actions.append(ROTATE)
        actions.extend([SOFT_DROP, HARD_DROP])
        return actions

    def legal_placement_actions(self) -> list[int]:
        if self.game_over:
            return []

        actions: list[int] = []
        seen: set[bytes] = set()
        shape = self.current_piece.copy()
        for rot_idx in range(MAX_ROTATIONS):
            key = shape.tobytes()
            if key not in seen:
                seen.add(key)
                max_col = self.width - shape.shape[1]
                for col in range(max_col + 1):
                    if self._drop_row(shape, col) is not None:
                        actions.append(rot_idx * self.width + col)
            shape = np.rot90(shape, -1)
        return actions

    def _spawn_piece(self) -> None:
        self.current_name = PIECE_ORDER[int(self.rng.integers(0, len(PIECE_ORDER)))]
        self.current_piece = PIECES[self.current_name].copy()
        self.piece_row = 0
        self.piece_col = (self.width - self.current_piece.shape[1]) // 2

        if not self._can_place(self.current_piece, self.piece_row, self.piece_col):
            self.game_over = True

    def _can_place(self, shape: np.ndarray, row: int, col: int) -> bool:
        h, w = shape.shape
        if row < 0 or col < 0 or row + h > self.height or col + w > self.width:
            return False

        occupied = self.board[row : row + h, col : col + w]
        return not np.any((shape == 1) & (occupied == 1))

    def _drop_row(self, shape: np.ndarray, col: int) -> int | None:
        row = 0
        if not self._can_place(shape, row, col):
            return None
        while self._can_place(shape, row + 1, col):
            row += 1
        return row

    def _lock_piece(self) -> int:
        h, w = self.current_piece.shape
        area = self.board[self.piece_row : self.piece_row + h, self.piece_col : self.piece_col + w]
        area[self.current_piece == 1] = 1

        full_rows = np.all(self.board == 1, axis=1)
        lines_cleared = int(np.sum(full_rows))
        if lines_cleared > 0:
            remaining = self.board[~full_rows]
            cleared = np.zeros((lines_cleared, self.width), dtype=np.int32)
            self.board = np.vstack([cleared, remaining])

        self.lines += lines_cleared
        self.score += lines_cleared
        return lines_cleared

    def _try_move(self, drow: int, dcol: int) -> bool:
        nr = self.piece_row + drow
        nc = self.piece_col + dcol
        if self._can_place(self.current_piece, nr, nc):
            self.piece_row = nr
            self.piece_col = nc
            return True
        return False

    def _try_rotate(self) -> bool:
        rotated = np.rot90(self.current_piece, -1)
        for kick in (0, -1, 1, -2, 2):
            nc = self.piece_col + kick
            if self._can_place(rotated, self.piece_row, nc):
                self.current_piece = rotated
                self.piece_col = nc
                return True
        return False

    def _can_rotate(self) -> bool:
        rotated = np.rot90(self.current_piece, -1)
        for kick in (0, -1, 1, -2, 2):
            nc = self.piece_col + kick
            if self._can_place(rotated, self.piece_row, nc):
                return True
        return False

    def _board_value(self, board: np.ndarray) -> float:
        heights: list[int] = []
        holes = 0
        for c in range(self.width):
            col = board[:, c]
            filled = np.where(col == 1)[0]
            if filled.size == 0:
                heights.append(0)
                continue
            top = int(filled[0])
            heights.append(self.height - top)
            holes += int(np.sum(col[top:] == 0))

        aggregate_height = float(sum(heights))
        bumpiness = float(sum(abs(heights[i] - heights[i + 1]) for i in range(self.width - 1)))
        return -(
            self.config.height_penalty * aggregate_height
            + self.config.holes_penalty * float(holes)
            + self.config.bumpiness_penalty * bumpiness
        )

    def _decode_placement_action(self, action: int) -> tuple[np.ndarray, int] | None:
        rot_idx = int(action // self.width)
        col = int(action % self.width)
        if rot_idx < 0 or rot_idx >= MAX_ROTATIONS:
            return None

        shape = self.current_piece.copy()
        for _ in range(rot_idx):
            shape = np.rot90(shape, -1)
        if col + shape.shape[1] > self.width:
            return None
        return shape, col

    def board_with_piece(self) -> np.ndarray:
        board = self.board.copy()
        if self.game_over:
            return board

        h, w = self.current_piece.shape
        r0, c0 = self.piece_row, self.piece_col
        if 0 <= r0 < self.height and 0 <= c0 < self.width:
            region = board[r0 : r0 + h, c0 : c0 + w]
            region[self.current_piece == 1] = 2
        return board

    def step(self, action: int) -> TetrisStepResult:
        if action not in self.legal_actions():
            raise ValueError(f"Invalid action {action}")

        if self.game_over:
            return TetrisStepResult(done=True, reward=0.0, lines_cleared=0, hard_drop_distance=0, game_over=True)

        self.steps += 1
        reward = self.config.step_reward
        lines_cleared = 0
        hard_drop_distance = 0
        locked_piece = False
        board_value_before = self._board_value(self.board)

        if action == MOVE_LEFT:
            self._try_move(0, -1)
        elif action == MOVE_RIGHT:
            self._try_move(0, 1)
        elif action == ROTATE:
            self._try_rotate()

        if action == SOFT_DROP:
            if self._try_move(1, 0):
                reward += self.config.soft_drop_bonus
            else:
                lines_cleared = self._lock_piece()
                reward += self.config.line_clear_reward * float(lines_cleared * lines_cleared)
                locked_piece = True
                self._spawn_piece()
        elif action == HARD_DROP:
            while self._try_move(1, 0):
                hard_drop_distance += 1
            reward += self.config.hard_drop_bonus_scale * hard_drop_distance
            lines_cleared = self._lock_piece()
            reward += self.config.line_clear_reward * float(lines_cleared * lines_cleared)
            locked_piece = True
            self._spawn_piece()
        else:
            # Gravity step for left/right/rotate/no-op style actions.
            if not self._try_move(1, 0):
                lines_cleared = self._lock_piece()
                reward += self.config.line_clear_reward * float(lines_cleared * lines_cleared)
                locked_piece = True
                self._spawn_piece()

        if locked_piece:
            board_value_after = self._board_value(self.board)
            reward += self.config.board_delta_scale * (board_value_after - board_value_before)

        done = False
        if self.game_over:
            done = True
            reward += self.config.game_over_penalty

        if self.steps >= self.config.max_steps:
            done = True

        return TetrisStepResult(
            done=done,
            reward=float(reward),
            lines_cleared=lines_cleared,
            hard_drop_distance=hard_drop_distance,
            game_over=self.game_over,
        )

    def step_placement(self, action: int) -> TetrisStepResult:
        if action not in self.legal_placement_actions():
            raise ValueError(f"Invalid placement action {action}")

        if self.game_over:
            return TetrisStepResult(done=True, reward=0.0, lines_cleared=0, hard_drop_distance=0, game_over=True)

        decoded = self._decode_placement_action(action)
        if decoded is None:
            raise ValueError(f"Invalid placement action {action}")
        shape, col = decoded
        row = self._drop_row(shape, col)
        if row is None:
            raise ValueError(f"Placement action cannot be applied: {action}")

        self.steps += 1
        reward = self.config.step_reward
        board_value_before = self._board_value(self.board)

        self.current_piece = shape
        self.piece_col = col
        self.piece_row = row

        hard_drop_distance = row
        reward += self.config.hard_drop_bonus_scale * hard_drop_distance
        lines_cleared = self._lock_piece()
        reward += self.config.line_clear_reward * float(lines_cleared * lines_cleared)

        board_value_after = self._board_value(self.board)
        reward += self.config.board_delta_scale * (board_value_after - board_value_before)
        self._spawn_piece()

        done = False
        if self.game_over:
            done = True
            reward += self.config.game_over_penalty

        if self.steps >= self.config.max_steps:
            done = True

        return TetrisStepResult(
            done=done,
            reward=float(reward),
            lines_cleared=lines_cleared,
            hard_drop_distance=hard_drop_distance,
            game_over=self.game_over,
        )

    def render(self) -> str:
        board = self.board_with_piece()
        lines: List[str] = []
        border = "+" + "-" * self.width + "+"
        lines.append(border)
        for r in range(self.height):
            row_chars = []
            for c in range(self.width):
                v = int(board[r, c])
                if v == 0:
                    row_chars.append(" ")
                elif v == 1:
                    row_chars.append("#")
                else:
                    row_chars.append("@");
            lines.append("|" + "".join(row_chars) + "|")
        lines.append(border)
        lines.append(f"Score: {self.score}  Lines: {self.lines}  Steps: {self.steps}  Piece: {self.current_name}")
        return "\n".join(lines)


class TetrisEnv:
    """RL environment wrapper for Tetris."""

    action_size = 5

    def __init__(self, config: TetrisConfig | None = None, seed: int | None = None) -> None:
        self.game = TetrisGame(config=config, seed=seed)

    def reset(self, seed: int | None = None) -> np.ndarray:
        self.game.reset(seed=seed)
        return self.get_state()

    def legal_actions(self) -> list[int]:
        return self.game.legal_actions()

    def get_state(self) -> np.ndarray:
        return self.encode_state(
            board_locked=self.game.board,
            board_with_piece=self.game.board_with_piece(),
            piece_name=self.game.current_name,
        )

    @staticmethod
    def encode_state(board_locked: np.ndarray, board_with_piece: np.ndarray, piece_name: str) -> np.ndarray:
        # Channels: locked blocks + active piece + piece one-hot (7)
        locked = (board_locked == 1).astype(np.float32)
        active = (board_with_piece == 2).astype(np.float32)
        flat = np.concatenate([locked.reshape(-1), active.reshape(-1)])

        piece_oh = np.zeros(len(PIECE_ORDER), dtype=np.float32)
        if piece_name in PIECE_ORDER:
            piece_oh[PIECE_ORDER.index(piece_name)] = 1.0

        return np.concatenate([flat, piece_oh], axis=0)

    def step(self, action: int):
        result = self.game.step(action)
        info = {
            "score": int(self.game.score),
            "lines": int(self.game.lines),
            "steps": int(self.game.steps),
            "piece": self.game.current_name,
            "lines_cleared": int(result.lines_cleared),
            "hard_drop_distance": int(result.hard_drop_distance),
            "game_over": bool(result.game_over),
        }
        return self.get_state(), float(result.reward), bool(result.done), info

    def render(self) -> str:
        return self.game.render()


class TetrisPlacementEnv:
    """RL env where each action is a full placement (rotation + column)."""

    def __init__(self, config: TetrisConfig | None = None, seed: int | None = None) -> None:
        self.game = TetrisGame(config=config, seed=seed)
        self.action_size = self.game.width * MAX_ROTATIONS

    def reset(self, seed: int | None = None) -> np.ndarray:
        self.game.reset(seed=seed)
        return self.get_state()

    def legal_actions(self) -> list[int]:
        return self.game.legal_placement_actions()

    def simulate_placement(self, action: int) -> tuple[np.ndarray, int]:
        decoded = self.game._decode_placement_action(action)
        if decoded is None:
            raise ValueError(f"Invalid placement action {action}")
        shape, col = decoded
        row = self.game._drop_row(shape, col)
        if row is None:
            raise ValueError(f"Placement action cannot be applied: {action}")

        board = self.game.board.copy()
        h, w = shape.shape
        area = board[row : row + h, col : col + w]
        area[shape == 1] = 1

        full_rows = np.all(board == 1, axis=1)
        lines_cleared = int(np.sum(full_rows))
        if lines_cleared > 0:
            remaining = board[~full_rows]
            cleared = np.zeros((lines_cleared, self.game.width), dtype=np.int32)
            board = np.vstack([cleared, remaining])

        return board, lines_cleared

    def get_state(self) -> np.ndarray:
        return self.encode_state(board_locked=self.game.board, piece_name=self.game.current_name)

    @staticmethod
    def _board_features(board_locked: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        h, w = board_locked.shape
        heights = np.zeros(w, dtype=np.float32)
        holes = np.zeros(w, dtype=np.float32)

        for c in range(w):
            col = board_locked[:, c]
            filled = np.where(col == 1)[0]
            if filled.size == 0:
                heights[c] = 0.0
                holes[c] = 0.0
                continue
            top = int(filled[0])
            heights[c] = float(h - top) / float(max(h, 1))
            holes[c] = float(np.sum(col[top:] == 0)) / float(max(h, 1))

        agg_height = float(np.sum(heights)) / float(max(w, 1))
        total_holes = float(np.sum(holes)) / float(max(w, 1))
        bumpiness = 0.0
        for i in range(w - 1):
            bumpiness += abs(float(heights[i]) - float(heights[i + 1]))
        bumpiness /= float(max(w - 1, 1))
        globals_feat = np.asarray([agg_height, total_holes, bumpiness], dtype=np.float32)
        return heights, holes, globals_feat

    @staticmethod
    def afterstate_features_from_board(
        board_locked: np.ndarray,
        lines_cleared: int,
        include_max_height: bool = True,
    ) -> np.ndarray:
        heights, holes, globals_feat = TetrisPlacementEnv._board_features(board_locked)
        lines_feat = np.asarray([float(lines_cleared) / 4.0], dtype=np.float32)
        if include_max_height:
            max_height = np.asarray([float(np.max(heights)) if heights.size > 0 else 0.0], dtype=np.float32)
            return np.concatenate([lines_feat, globals_feat, max_height], axis=0)
        return np.concatenate([lines_feat, globals_feat], axis=0)

    def legal_afterstates(self, include_max_height: bool = True) -> list[tuple[int, np.ndarray]]:
        pairs: list[tuple[int, np.ndarray]] = []
        for action in self.legal_actions():
            board_after, lines_cleared = self.simulate_placement(action)
            features = self.afterstate_features_from_board(
                board_after,
                lines_cleared,
                include_max_height=include_max_height,
            )
            pairs.append((action, features))
        return pairs

    @staticmethod
    def encode_state(board_locked: np.ndarray, piece_name: str) -> np.ndarray:
        locked = (board_locked == 1).astype(np.float32).reshape(-1)
        piece_oh = np.zeros(len(PIECE_ORDER), dtype=np.float32)
        if piece_name in PIECE_ORDER:
            piece_oh[PIECE_ORDER.index(piece_name)] = 1.0
        heights, holes, globals_feat = TetrisPlacementEnv._board_features(board_locked)
        return np.concatenate([locked, piece_oh, heights, holes, globals_feat], axis=0)

    def step(self, action: int):
        result = self.game.step_placement(action)
        info = {
            "score": int(self.game.score),
            "lines": int(self.game.lines),
            "steps": int(self.game.steps),
            "piece": self.game.current_name,
            "lines_cleared": int(result.lines_cleared),
            "hard_drop_distance": int(result.hard_drop_distance),
            "game_over": bool(result.game_over),
        }
        return self.get_state(), float(result.reward), bool(result.done), info

    def render(self) -> str:
        return self.game.render()
