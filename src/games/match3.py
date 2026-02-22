from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np


@dataclass
class Match3Config:
    width: int = 6
    height: int = 6
    num_colors: int = 5
    max_steps: int = 150
    step_reward: float = -0.01
    tile_reward: float = 0.20
    combo_bonus: float = 0.15
    invalid_penalty: float = -0.50
    reshuffle_on_stuck: bool = True


@dataclass
class Match3StepResult:
    done: bool
    reward: float
    tiles_cleared: int
    cascades: int
    invalid: bool
    reshuffled: bool


Swap = Tuple[Tuple[int, int], Tuple[int, int]]
CascadeFrame = Tuple[np.ndarray, np.ndarray]  # (board_before_clear, clear_mask)


class Match3Game:
    """Small match-3 board with swap actions, cascades, gravity, and refill."""

    def __init__(self, config: Match3Config | None = None, seed: int | None = None) -> None:
        self.config = config or Match3Config()
        if self.config.width < 4 or self.config.height < 4:
            raise ValueError("board must be at least 4x4")
        if self.config.num_colors < 3:
            raise ValueError("num_colors must be >= 3")

        self.rng = np.random.default_rng(seed)
        self.board_arr = np.zeros((self.config.height, self.config.width), dtype=np.int32)
        self._actions = self._build_action_map()
        self._legal_cache: list[int] | None = None
        self.capture_step_traces = False
        self.last_swap: Swap | None = None
        self.last_swapped_board: np.ndarray | None = None
        self.last_cascade_frames: list[CascadeFrame] = []
        self.last_invalid_swap = False

        self.score = 0
        self.steps = 0
        self.total_tiles_cleared = 0
        self.total_cascades = 0
        self.game_over = False

    @property
    def action_size(self) -> int:
        return len(self._actions)

    def _build_action_map(self) -> list[Swap]:
        actions: list[Swap] = []
        h, w = self.config.height, self.config.width
        for r in range(h):
            for c in range(w - 1):
                actions.append(((r, c), (r, c + 1)))  # horizontal
        for r in range(h - 1):
            for c in range(w):
                actions.append(((r, c), (r + 1, c)))  # vertical
        return actions

    def action_to_swap(self, action: int) -> Swap:
        if action < 0 or action >= len(self._actions):
            raise ValueError(f"Invalid action index {action}")
        return self._actions[action]

    def _invalidate_legal_cache(self) -> None:
        self._legal_cache = None

    def _random_board(self) -> np.ndarray:
        return self.rng.integers(1, self.config.num_colors + 1, size=(self.config.height, self.config.width), dtype=np.int32)

    def _random_board_no_matches(self) -> np.ndarray:
        h, w = self.config.height, self.config.width
        board = np.zeros((h, w), dtype=np.int32)
        all_colors = np.arange(1, self.config.num_colors + 1, dtype=np.int32)
        for r in range(h):
            for c in range(w):
                forbidden: set[int] = set()
                if c >= 2 and board[r, c - 1] == board[r, c - 2] != 0:
                    forbidden.add(int(board[r, c - 1]))
                if r >= 2 and board[r - 1, c] == board[r - 2, c] != 0:
                    forbidden.add(int(board[r - 1, c]))
                choices = [int(v) for v in all_colors if int(v) not in forbidden]
                if not choices:
                    choices = [int(v) for v in all_colors]
                board[r, c] = int(self.rng.choice(np.asarray(choices, dtype=np.int32)))
        return board

    @staticmethod
    def _find_matches_mask(board: np.ndarray) -> np.ndarray:
        h, w = board.shape
        mask = np.zeros((h, w), dtype=bool)

        # Horizontal runs
        for r in range(h):
            c = 0
            while c < w:
                v = int(board[r, c])
                start = c
                c += 1
                while c < w and int(board[r, c]) == v:
                    c += 1
                if v != 0 and (c - start) >= 3:
                    mask[r, start:c] = True

        # Vertical runs
        for c in range(w):
            r = 0
            while r < h:
                v = int(board[r, c])
                start = r
                r += 1
                while r < h and int(board[r, c]) == v:
                    r += 1
                if v != 0 and (r - start) >= 3:
                    mask[start:r, c] = True

        return mask

    @classmethod
    def _has_matches(cls, board: np.ndarray) -> bool:
        return bool(np.any(cls._find_matches_mask(board)))

    @staticmethod
    def _forms_match_at(board: np.ndarray, r: int, c: int) -> bool:
        h, w = board.shape
        if r < 0 or r >= h or c < 0 or c >= w:
            return False
        v = int(board[r, c])
        if v == 0:
            return False

        # Horizontal run through (r, c)
        left = c
        while left - 1 >= 0 and int(board[r, left - 1]) == v:
            left -= 1
        right = c
        while right + 1 < w and int(board[r, right + 1]) == v:
            right += 1
        if (right - left + 1) >= 3:
            return True

        # Vertical run through (r, c)
        up = r
        while up - 1 >= 0 and int(board[up - 1, c]) == v:
            up -= 1
        down = r
        while down + 1 < h and int(board[down + 1, c]) == v:
            down += 1
        return (down - up + 1) >= 3

    def _swap_inplace(self, board: np.ndarray, a: Tuple[int, int], b: Tuple[int, int]) -> None:
        (r1, c1), (r2, c2) = a, b
        board[r1, c1], board[r2, c2] = board[r2, c2], board[r1, c1]

    def _swap_creates_match(self, action: int) -> bool:
        a, b = self._actions[action]
        (r1, c1), (r2, c2) = a, b
        if self.board_arr[r1, c1] == self.board_arr[r2, c2]:
            return False
        # Only the swapped cells can create *new* matches, so check locally after an in-place swap.
        self._swap_inplace(self.board_arr, a, b)
        creates = self._forms_match_at(self.board_arr, r1, c1) or self._forms_match_at(self.board_arr, r2, c2)
        self._swap_inplace(self.board_arr, a, b)
        return bool(creates)

    def _generate_valid_start_board(self) -> np.ndarray:
        for _ in range(2000):
            board = self._random_board_no_matches()
            self.board_arr = board
            self._invalidate_legal_cache()
            if self.legal_actions():
                return board.copy()
        raise RuntimeError("Failed to generate valid Match-3 start board")

    def _apply_gravity_and_refill(self) -> None:
        h, w = self.board_arr.shape
        for c in range(w):
            col = self.board_arr[:, c]
            kept = col[col != 0]
            n_missing = h - kept.size
            if n_missing > 0:
                refill = self.rng.integers(1, self.config.num_colors + 1, size=n_missing, dtype=np.int32)
                self.board_arr[:, c] = np.concatenate([refill, kept]).astype(np.int32)

    def _reshuffle_until_playable(self) -> bool:
        if not self.config.reshuffle_on_stuck:
            return False
        for _ in range(2000):
            flat = self.board_arr.reshape(-1).copy()
            self.rng.shuffle(flat)
            candidate = flat.reshape(self.board_arr.shape)
            if self._has_matches(candidate):
                continue
            old = self.board_arr.copy()
            self.board_arr = candidate.astype(np.int32)
            self._invalidate_legal_cache()
            if self.legal_actions():
                return True
            self.board_arr = old
        # Fallback: regenerate a fresh valid board
        self._generate_valid_start_board()
        return True

    def reset(self, seed: int | None = None) -> np.ndarray:
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self.last_swap = None
        self.last_swapped_board = None
        self.last_cascade_frames = []
        self.last_invalid_swap = False
        self.score = 0
        self.steps = 0
        self.total_tiles_cleared = 0
        self.total_cascades = 0
        self.game_over = False
        self.board_arr = self._generate_valid_start_board()
        self._invalidate_legal_cache()
        return self.board()

    def board(self) -> np.ndarray:
        return self.board_arr.copy()

    def legal_actions(self) -> list[int]:
        if self.game_over:
            return []
        if self._legal_cache is not None:
            return list(self._legal_cache)
        legal = [a for a in range(len(self._actions)) if self._swap_creates_match(a)]
        self._legal_cache = legal
        return list(legal)

    def step(self, action: int) -> Match3StepResult:
        if self.game_over:
            return Match3StepResult(done=True, reward=0.0, tiles_cleared=0, cascades=0, invalid=False, reshuffled=False)

        self.last_swap = None
        self.last_swapped_board = None
        self.last_cascade_frames = []
        self.last_invalid_swap = False
        self.steps += 1
        reward = float(self.config.step_reward)
        tiles_cleared = 0
        cascades = 0
        invalid = False
        reshuffled = False

        if action < 0 or action >= self.action_size:
            invalid = True
            reward += float(self.config.invalid_penalty)
        else:
            a, b = self._actions[action]
            self.last_swap = (a, b)
            self._swap_inplace(self.board_arr, a, b)
            if self.capture_step_traces:
                self.last_swapped_board = self.board_arr.copy()
            mask = self._find_matches_mask(self.board_arr)
            if not np.any(mask):
                # Revert invalid swap.
                self._swap_inplace(self.board_arr, a, b)
                invalid = True
                self.last_invalid_swap = True
                reward += float(self.config.invalid_penalty)
            else:
                while np.any(mask):
                    cascades += 1
                    if self.capture_step_traces:
                        self.last_cascade_frames.append((self.board_arr.copy(), mask.copy()))
                    cleared = int(np.sum(mask))
                    tiles_cleared += cleared
                    self.total_tiles_cleared += cleared
                    self.total_cascades += 1
                    self.score += cleared

                    reward += float(cleared) * float(self.config.tile_reward)
                    if cascades > 1:
                        reward += float(cascades - 1) * float(self.config.combo_bonus)

                    self.board_arr[mask] = 0
                    self._apply_gravity_and_refill()
                    mask = self._find_matches_mask(self.board_arr)

        self._invalidate_legal_cache()
        legal = self.legal_actions()
        if not legal:
            reshuffled = self._reshuffle_until_playable()
            self._invalidate_legal_cache()
            legal = self.legal_actions()

        done = bool(self.steps >= self.config.max_steps)
        self.game_over = done
        return Match3StepResult(
            done=done,
            reward=float(reward),
            tiles_cleared=tiles_cleared,
            cascades=cascades,
            invalid=invalid,
            reshuffled=reshuffled,
        )

    def render(self) -> str:
        symbols = " .123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        lines: List[str] = []
        border = "+" + "-" * (2 * self.config.width - 1) + "+"
        lines.append(border)
        for r in range(self.config.height):
            row = []
            for c in range(self.config.width):
                v = int(self.board_arr[r, c])
                ch = symbols[v] if 0 <= v < len(symbols) else "?"
                row.append(ch)
            lines.append("|" + " ".join(row) + "|")
        lines.append(border)
        lines.append(
            f"Score: {self.score}  Steps: {self.steps}/{self.config.max_steps}  "
            f"Tiles: {self.total_tiles_cleared}  Cascades: {self.total_cascades}  Legal: {len(self.legal_actions())}"
        )
        return "\n".join(lines)


class Match3Env:
    """RL wrapper for Match-3.

    Action space = all adjacent swaps in a fixed mapping (horizontal then vertical).
    State = one-hot gem channels + [progress, legal_move_fraction].
    """

    def __init__(self, config: Match3Config | None = None, seed: int | None = None) -> None:
        self.game = Match3Game(config=config, seed=seed)
        self.action_size = self.game.action_size

    def reset(self, seed: int | None = None) -> np.ndarray:
        self.game.reset(seed=seed)
        return self.get_state()

    def legal_actions(self) -> list[int]:
        return self.game.legal_actions()

    def action_to_swap(self, action: int) -> Swap:
        return self.game.action_to_swap(action)

    @staticmethod
    def encode_board(board: np.ndarray, num_colors: int) -> np.ndarray:
        h, w = board.shape
        state = np.zeros((num_colors, h, w), dtype=np.float32)
        for color in range(1, num_colors + 1):
            rr, cc = np.where(board == color)
            state[color - 1, rr, cc] = 1.0
        return state.reshape(-1)

    def get_state(self) -> np.ndarray:
        cfg = self.game.config
        board_features = self.encode_board(self.game.board_arr, cfg.num_colors)
        progress = float(self.game.steps) / float(max(cfg.max_steps, 1))
        legal_frac = float(len(self.game.legal_actions())) / float(max(self.action_size, 1))
        extra = np.asarray([progress, legal_frac], dtype=np.float32)
        return np.concatenate([board_features, extra], axis=0)

    def step(self, action: int):
        result = self.game.step(action)
        info = {
            "score": int(self.game.score),
            "steps": int(self.game.steps),
            "tiles_cleared": int(result.tiles_cleared),
            "cascades": int(result.cascades),
            "invalid": bool(result.invalid),
            "reshuffled": bool(result.reshuffled),
            "legal_count": int(len(self.game.legal_actions()) if not self.game.game_over else 0),
            "total_tiles_cleared": int(self.game.total_tiles_cleared),
            "total_cascades": int(self.game.total_cascades),
        }
        return self.get_state(), float(result.reward), bool(result.done), info

    def render(self) -> str:
        return self.game.render()
