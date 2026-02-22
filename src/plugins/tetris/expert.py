from __future__ import annotations

import numpy as np

from ...games.tetris import (
    HARD_DROP,
    MAX_ROTATIONS,
    MOVE_LEFT,
    MOVE_RIGHT,
    ROTATE,
    TetrisEnv,
    TetrisPlacementEnv,
)


def _can_place(board: np.ndarray, shape: np.ndarray, row: int, col: int) -> bool:
    h, w = shape.shape
    bh, bw = board.shape
    if row < 0 or col < 0 or row + h > bh or col + w > bw:
        return False
    occupied = board[row : row + h, col : col + w]
    return not np.any((shape == 1) & (occupied == 1))


def _drop_row(board: np.ndarray, shape: np.ndarray, col: int) -> int | None:
    row = 0
    if not _can_place(board, shape, row, col):
        return None
    while _can_place(board, shape, row + 1, col):
        row += 1
    return row


def _board_features(board: np.ndarray) -> tuple[int, int, int]:
    h, w = board.shape
    heights = []
    holes = 0
    for c in range(w):
        col = board[:, c]
        filled = np.where(col == 1)[0]
        if filled.size == 0:
            heights.append(0)
            continue
        top = int(filled[0])
        heights.append(h - top)
        holes += int(np.sum(col[top:] == 0))
    bumpiness = sum(abs(heights[i] - heights[i + 1]) for i in range(w - 1))
    return int(sum(heights)), int(holes), int(bumpiness)


def _placement_value(board: np.ndarray, shape: np.ndarray, col: int) -> float | None:
    row = _drop_row(board, shape, col)
    if row is None:
        return None
    placed = board.copy()
    h, w = shape.shape
    area = placed[row : row + h, col : col + w]
    area[shape == 1] = 1

    full = np.all(placed == 1, axis=1)
    lines_cleared = int(np.sum(full))
    if lines_cleared > 0:
        remaining = placed[~full]
        cleared = np.zeros((lines_cleared, placed.shape[1]), dtype=np.int32)
        placed = np.vstack([cleared, remaining])

    agg_h, holes, bump = _board_features(placed)
    return 20.0 * lines_cleared - 0.35 * agg_h - 0.9 * holes - 0.18 * bump


def _best_placement(board: np.ndarray, current_piece: np.ndarray, width: int) -> tuple[np.ndarray, int] | None:
    best_value: float | None = None
    best: tuple[np.ndarray, int] | None = None
    seen: set[bytes] = set()
    shape = current_piece.copy()
    for _ in range(4):
        key = shape.tobytes()
        if key in seen:
            shape = np.rot90(shape, -1)
            continue
        seen.add(key)
        max_col = width - shape.shape[1]
        for col in range(max_col + 1):
            value = _placement_value(board, shape, col)
            if value is None:
                continue
            if best_value is None or value > best_value:
                best_value = value
                best = (shape.copy(), col)
        shape = np.rot90(shape, -1)
    return best


def choose_expert_action(env: TetrisEnv | TetrisPlacementEnv, placement_actions: bool) -> int:
    game = env.game
    legal = env.legal_actions()
    if not legal:
        return 0

    if placement_actions:
        target = _best_placement(game.board, game.current_piece, game.width)
        if target is None:
            return int(legal[0])
        target_shape, target_col = target
        shape = game.current_piece.copy()
        for rot_idx in range(MAX_ROTATIONS):
            if shape.tobytes() == target_shape.tobytes():
                action = rot_idx * game.width + target_col
                if action in legal:
                    return action
            shape = np.rot90(shape, -1)
        return int(legal[0])

    target = _best_placement(game.board, game.current_piece, game.width)
    if target is None:
        return HARD_DROP if HARD_DROP in legal else int(legal[0])

    target_shape, target_col = target
    if game.current_piece.tobytes() != target_shape.tobytes():
        return ROTATE if ROTATE in legal else (HARD_DROP if HARD_DROP in legal else int(legal[0]))
    if game.piece_col < target_col and MOVE_RIGHT in legal:
        return MOVE_RIGHT
    if game.piece_col > target_col and MOVE_LEFT in legal:
        return MOVE_LEFT
    return HARD_DROP if HARD_DROP in legal else int(legal[0])
