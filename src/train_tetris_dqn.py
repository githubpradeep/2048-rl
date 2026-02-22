from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import numpy as np

from .games.tetris import (
    HARD_DROP,
    MAX_ROTATIONS,
    MOVE_LEFT,
    MOVE_RIGHT,
    ROTATE,
    TetrisConfig,
    TetrisEnv,
    TetrisPlacementEnv,
)
from .network import AdamOptimizer, MLPQNetwork
from .replay_buffer import ReplayBuffer
from .tetris_eval_utils import evaluate_tetris_policy


def parse_hidden_sizes(raw: str) -> Tuple[int, ...]:
    values = [v.strip() for v in raw.split(",") if v.strip()]
    if not values:
        raise ValueError("hidden sizes cannot be empty")
    return tuple(int(v) for v in values)


def epsilon_by_step(step: int, start: float, end: float, decay_steps: int) -> float:
    if step >= decay_steps:
        return end
    frac = step / max(decay_steps, 1)
    return start + frac * (end - start)


def masked_argmax(values: list[float], legal_actions: list[int]) -> int:
    if not legal_actions:
        return max(range(len(values)), key=values.__getitem__)
    return max(legal_actions, key=lambda action: values[action])


def argmax(values: list[float]) -> int:
    return max(range(len(values)), key=values.__getitem__)


def masked_max_q(target_q: np.ndarray, legal_masks: np.ndarray) -> np.ndarray:
    legal = legal_masks > 0.5
    masked_target = np.where(legal, target_q, -1e9)
    has_legal = np.any(legal, axis=1)
    max_next_q = np.max(masked_target, axis=1)
    return np.where(has_legal, max_next_q, 0.0).astype(np.float32)


def masked_double_q(online_q: np.ndarray, target_q: np.ndarray, legal_masks: np.ndarray) -> np.ndarray:
    legal = legal_masks > 0.5
    masked_online = np.where(legal, online_q, -1e9)
    next_actions = np.argmax(masked_online, axis=1)
    has_legal = np.any(legal, axis=1)
    selected = target_q[np.arange(target_q.shape[0]), next_actions]
    return np.where(has_legal, selected, 0.0).astype(np.float32)


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
        return HARD_DROP if HARD_DROP in legal else legal[0]

    target_shape, target_col = target

    if game.current_piece.tobytes() != target_shape.tobytes():
        return ROTATE if ROTATE in legal else (HARD_DROP if HARD_DROP in legal else legal[0])

    if game.piece_col < target_col and MOVE_RIGHT in legal:
        return MOVE_RIGHT
    if game.piece_col > target_col and MOVE_LEFT in legal:
        return MOVE_LEFT
    return HARD_DROP if HARD_DROP in legal else legal[0]


def main() -> None:
    parser = argparse.ArgumentParser(description="Train from-scratch DQN agent for Tetris")
    parser.add_argument("--episodes", type=int, default=3000)
    parser.add_argument("--max-steps", type=int, default=1200)
    parser.add_argument("--height", type=int, default=20)
    parser.add_argument("--width", type=int, default=10)
    parser.add_argument("--placement-actions", action="store_true")
    parser.add_argument("--buffer-size", type=int, default=120_000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--target-update", type=int, default=500)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--target-clip", type=float, default=30.0)
    parser.add_argument("--double-dqn", action="store_true")
    parser.add_argument("--dueling", action="store_true")
    parser.add_argument("--train-every", type=int, default=4)
    parser.add_argument("--warmup-steps", type=int, default=1000)
    parser.add_argument("--eps-start", type=float, default=1.0)
    parser.add_argument("--eps-end", type=float, default=0.03)
    parser.add_argument("--eps-decay-steps", type=int, default=40_000)
    parser.add_argument("--expert-start", type=float, default=0.30)
    parser.add_argument("--expert-end", type=float, default=0.05)
    parser.add_argument("--expert-decay-steps", type=int, default=40_000)
    parser.add_argument(
        "--expert-warmup-steps",
        type=int,
        default=12_000,
        help="Use only expert actions for initial replay warm start.",
    )
    parser.add_argument("--hidden-sizes", type=str, default="256,256")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-every", type=int, default=50)
    parser.add_argument("--eval-episodes", type=int, default=30)
    parser.add_argument("--save-dir", type=str, default="models/tetris")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    hidden_sizes = parse_hidden_sizes(args.hidden_sizes)

    config = TetrisConfig(height=args.height, width=args.width, max_steps=args.max_steps)
    if args.placement_actions:
        env = TetrisPlacementEnv(config=config, seed=args.seed)
    else:
        env = TetrisEnv(config=config, seed=args.seed)
    state_dim = env.reset(seed=args.seed).shape[0]

    online = MLPQNetwork(
        input_dim=state_dim,
        output_dim=env.action_size,
        hidden_sizes=hidden_sizes,
        seed=args.seed,
        dueling=args.dueling,
    )
    target = MLPQNetwork(
        input_dim=state_dim,
        output_dim=env.action_size,
        hidden_sizes=hidden_sizes,
        seed=args.seed + 1,
        dueling=args.dueling,
    )
    target.copy_from(online)

    optimizer = AdamOptimizer(lr=args.lr, max_grad_norm=args.grad_clip)
    buffer = ReplayBuffer(capacity=args.buffer_size, state_dim=state_dim, action_dim=env.action_size)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    global_step = 0
    best_eval = -float("inf")

    print(
        f"Tetris training config | board={args.height}x{args.width} max_steps={args.max_steps} "
        f"double_dqn={args.double_dqn} dueling={args.dueling} placement_actions={args.placement_actions}"
    )

    for episode in range(1, args.episodes + 1):
        state = env.reset(seed=args.seed + episode)
        done = False
        episode_reward = 0.0
        losses = []
        td_abs_vals = []
        steps = 0
        total_lines_cleared = 0
        info = {"score": 0, "lines": 0, "lines_cleared": 0}

        while not done and steps < args.max_steps:
            legal_actions = env.legal_actions()
            epsilon = epsilon_by_step(global_step, args.eps_start, args.eps_end, args.eps_decay_steps)
            expert_prob = epsilon_by_step(global_step, args.expert_start, args.expert_end, args.expert_decay_steps)
            if global_step < args.expert_warmup_steps and legal_actions:
                action = choose_expert_action(env, placement_actions=args.placement_actions)
            elif rng.random() < epsilon:
                if legal_actions and rng.random() < expert_prob:
                    action = choose_expert_action(env, placement_actions=args.placement_actions)
                else:
                    action = int(rng.choice(legal_actions)) if legal_actions else int(rng.integers(0, env.action_size))
            else:
                q_values = online.predict_one(state)
                action = masked_argmax(q_values, legal_actions)

            next_state, reward, done, info = env.step(action)
            total_lines_cleared += int(info["lines_cleared"])

            next_legal = env.legal_actions()
            buffer.add(state, action, reward, next_state, done, next_legal_actions=next_legal)
            state = next_state

            episode_reward += reward
            steps += 1
            global_step += 1

            can_train = len(buffer) >= args.batch_size and global_step >= args.warmup_steps
            if can_train and global_step % args.train_every == 0:
                batch = buffer.sample(args.batch_size, rng)
                target_q = np.asarray(target.predict_batch(batch.next_states), dtype=np.float32)
                legal_masks = batch.next_legal_masks
                if legal_masks is not None:
                    if args.double_dqn:
                        online_q = np.asarray(online.predict_batch(batch.next_states), dtype=np.float32)
                        max_next_q = masked_double_q(online_q, target_q, legal_masks)
                    else:
                        max_next_q = masked_max_q(target_q, legal_masks)
                elif args.double_dqn:
                    online_q = online.predict_batch(batch.next_states)
                    next_actions = [argmax(values) for values in online_q]
                    max_next_q = np.asarray([float(target_q[i][next_actions[i]]) for i in range(len(next_actions))], dtype=np.float32)
                else:
                    max_next_q = np.asarray([max(values) for values in target_q], dtype=np.float32)
                if args.target_clip > 0:
                    max_next_q = np.clip(max_next_q, -args.target_clip, args.target_clip)
                td_target = batch.rewards + args.gamma * (1.0 - batch.dones) * max_next_q
                if args.target_clip > 0:
                    td_target = np.clip(td_target, -args.target_clip, args.target_clip)
                loss, td_abs = online.train_batch(batch.states, batch.actions, td_target, optimizer)
                losses.append(loss)
                td_abs_vals.append(td_abs)

            if global_step % args.target_update == 0:
                target.copy_from(online)

        mean_loss = float(np.mean(losses)) if losses else 0.0
        mean_td_abs = float(np.mean(td_abs_vals)) if td_abs_vals else 0.0

        print(
            f"ep={episode:4d} steps={steps:4d} score={int(info['score']):4d} lines={int(info['lines']):4d} "
            f"episode_lines={total_lines_cleared:3d} reward={episode_reward:8.2f} "
            f"eps={epsilon_by_step(global_step, args.eps_start, args.eps_end, args.eps_decay_steps):.3f} "
            f"expert={epsilon_by_step(global_step, args.expert_start, args.expert_end, args.expert_decay_steps):.3f} "
            f"loss={mean_loss:.5f} td_abs={mean_td_abs:.5f}"
        )

        if episode % args.eval_every == 0:
            if args.placement_actions:
                eval_env = TetrisPlacementEnv(config=config, seed=args.seed + 999)
            else:
                eval_env = TetrisEnv(config=config, seed=args.seed + 999)
            stats = evaluate_tetris_policy(eval_env, online, episodes=args.eval_episodes, seed_start=args.seed + 7000)
            print(
                f"  eval: avg_score={stats.avg_score:.2f} median={stats.median_score:.2f} "
                f"avg_lines={stats.avg_lines:.2f} avg_steps={stats.avg_steps:.2f}"
            )
            if stats.avg_score > best_eval:
                best_eval = stats.avg_score
                best_path = save_dir / "tetris_dqn_best.json"
                online.save(best_path)
                print(f"  saved new best checkpoint: {best_path}")

    final_path = save_dir / "tetris_dqn_final.json"
    online.save(final_path)
    print(f"Training complete. Final model saved to: {final_path}")


if __name__ == "__main__":
    main()
