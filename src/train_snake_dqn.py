from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np

from .games.snake import SnakeConfig, SnakeEnv, SnakeFeatureEnv
from .network import AdamOptimizer, MLPQNetwork
from .replay_buffer import ReplayBuffer
from .snake_eval_utils import evaluate_snake_policy


def parse_hidden_sizes(raw: str) -> Tuple[int, ...]:
    values = [v.strip() for v in raw.split(",") if v.strip()]
    if not values:
        raise ValueError("hidden sizes cannot be empty")
    return tuple(int(v) for v in values)


def parse_int_list(raw: str) -> List[int]:
    values = [v.strip() for v in raw.split(",") if v.strip()]
    if not values:
        return []
    return [int(v) for v in values]


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


def curriculum_grid_size(episode: int, total_episodes: int, stages: List[int]) -> int:
    if len(stages) == 1:
        return stages[0]
    idx = min(len(stages) - 1, ((episode - 1) * len(stages)) // max(total_episodes, 1))
    return stages[idx]


def make_snake_env(
    *,
    grid_size: int,
    distance_reward_scale: float,
    seed: int,
    state_mode: str,
    state_grid_size: int,
):
    config = SnakeConfig(grid_size=grid_size, distance_reward_scale=distance_reward_scale)
    if state_mode == "features":
        return SnakeFeatureEnv(config=config, seed=seed)
    return SnakeEnv(config=config, seed=seed, state_grid_size=state_grid_size)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train from-scratch DQN agent for Snake")
    parser.add_argument("--episodes", type=int, default=1500)
    parser.add_argument("--max-steps", type=int, default=1200)
    parser.add_argument("--grid-size", type=int, default=10)
    parser.add_argument("--state-mode", type=str, choices=["board", "features"], default="board")
    parser.add_argument(
        "--curriculum-grid-sizes",
        type=str,
        default="",
        help="Comma-separated grid sizes by curriculum stage, e.g. '8,10'. Empty disables curriculum.",
    )
    parser.add_argument(
        "--state-grid-size",
        type=int,
        default=0,
        help="Encoded state grid size. 0 means auto (max of curriculum/grid size).",
    )
    parser.add_argument("--buffer-size", type=int, default=80_000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--max-grad-norm", type=float, default=5.0)
    parser.add_argument("--distance-reward-scale", type=float, default=0.2)
    parser.add_argument("--target-update", type=int, default=1000)
    parser.add_argument("--double-dqn", action="store_true", help="Use Double DQN target action selection.")
    parser.add_argument("--dueling", action="store_true", help="Use dueling architecture for Q-network.")
    parser.add_argument("--train-every", type=int, default=4)
    parser.add_argument("--warmup-steps", type=int, default=2000)
    parser.add_argument("--eps-start", type=float, default=1.0)
    parser.add_argument("--eps-end", type=float, default=0.02)
    parser.add_argument("--eps-decay-steps", type=int, default=60_000)
    parser.add_argument("--hidden-sizes", type=str, default="256,256")
    parser.add_argument("--loss", type=str, default="huber", choices=["mse", "huber"])
    parser.add_argument("--huber-delta", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-every", type=int, default=50)
    parser.add_argument("--eval-episodes", type=int, default=25)
    parser.add_argument("--save-dir", type=str, default="models/snake")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    hidden_sizes = parse_hidden_sizes(args.hidden_sizes)
    curriculum_stages = parse_int_list(args.curriculum_grid_sizes)
    if curriculum_stages:
        if any(size < 4 for size in curriculum_stages):
            raise ValueError("All curriculum grid sizes must be >= 4")
        grid_stages = curriculum_stages
    else:
        grid_stages = [args.grid_size]

    max_grid_size = max(grid_stages)
    state_grid_size = args.state_grid_size if args.state_grid_size > 0 else max_grid_size
    if args.state_mode == "board" and state_grid_size < max_grid_size:
        raise ValueError("state-grid-size must be >= max curriculum grid size")

    first_grid = grid_stages[0]
    env = make_snake_env(
        grid_size=first_grid,
        distance_reward_scale=args.distance_reward_scale,
        seed=args.seed,
        state_mode=args.state_mode,
        state_grid_size=state_grid_size,
    )
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

    optimizer = AdamOptimizer(lr=args.lr, max_grad_norm=args.max_grad_norm)
    buffer = ReplayBuffer(capacity=args.buffer_size, state_dim=state_dim, action_dim=env.action_size)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    global_step = 0
    best_eval = -float("inf")
    print(
        f"Snake training config | curriculum={grid_stages} state_mode={args.state_mode} "
        f"state_grid={state_grid_size if args.state_mode == 'board' else 'n/a'} "
        f"distance_reward_scale={args.distance_reward_scale} "
        f"double_dqn={args.double_dqn} dueling={args.dueling}"
    )

    for episode in range(1, args.episodes + 1):
        episode_grid = curriculum_grid_size(episode, args.episodes, grid_stages)
        env = make_snake_env(
            grid_size=episode_grid,
            distance_reward_scale=args.distance_reward_scale,
            seed=args.seed + episode,
            state_mode=args.state_mode,
            state_grid_size=state_grid_size,
        )
        state = env.reset(seed=args.seed + episode)
        done = False
        episode_reward = 0.0
        losses = []
        td_abs_vals = []
        steps = 0
        foods = 0
        info = {"score": 0, "length": 3, "ate_food": False}

        while not done and steps < args.max_steps:
            legal_actions = env.legal_actions()
            epsilon = epsilon_by_step(global_step, args.eps_start, args.eps_end, args.eps_decay_steps)
            if rng.random() < epsilon:
                action = int(rng.choice(legal_actions)) if legal_actions else int(rng.integers(0, env.action_size))
            else:
                q_values = online.predict_one(state)
                action = masked_argmax(q_values, legal_actions)

            next_state, reward, done, info = env.step(action)
            if bool(info["ate_food"]):
                foods += 1

            buffer.add(state, action, reward, next_state, done, next_legal_actions=env.legal_actions())
            state = next_state

            episode_reward += reward
            steps += 1
            global_step += 1

            can_train = len(buffer) >= args.batch_size and global_step >= args.warmup_steps
            if can_train and global_step % args.train_every == 0:
                batch = buffer.sample(args.batch_size, rng)
                target_q = np.asarray(target.predict_batch(batch.next_states), dtype=np.float32)
                legal_masks = batch.next_legal_masks
                if legal_masks is None:
                    raise RuntimeError("Snake trainer requires next_legal_masks in replay buffer")
                if args.double_dqn:
                    online_q = np.asarray(online.predict_batch(batch.next_states), dtype=np.float32)
                    max_next_q = masked_double_q(online_q, target_q, legal_masks)
                else:
                    max_next_q = masked_max_q(target_q, legal_masks)
                td_target = batch.rewards + args.gamma * (1.0 - batch.dones) * max_next_q
                loss, td_abs = online.train_batch(
                    batch.states,
                    batch.actions,
                    td_target,
                    optimizer,
                    loss=args.loss,
                    huber_delta=args.huber_delta,
                )
                losses.append(loss)
                td_abs_vals.append(td_abs)

            if global_step % args.target_update == 0:
                target.copy_from(online)

        mean_loss = float(np.mean(losses)) if losses else 0.0
        mean_td_abs = float(np.mean(td_abs_vals)) if td_abs_vals else 0.0

        print(
            f"ep={episode:4d} grid={episode_grid:2d} steps={steps:4d} score={int(info['score']):4d} len={int(info['length']):3d} "
            f"foods={foods:3d} reward={episode_reward:8.2f} "
            f"eps={epsilon_by_step(global_step, args.eps_start, args.eps_end, args.eps_decay_steps):.3f} "
            f"loss={mean_loss:.5f} td_abs={mean_td_abs:.5f}"
        )

        if episode % args.eval_every == 0:
            eval_grid = grid_stages[-1]
            eval_env = make_snake_env(
                grid_size=eval_grid,
                distance_reward_scale=args.distance_reward_scale,
                seed=args.seed + 999,
                state_mode=args.state_mode,
                state_grid_size=state_grid_size,
            )
            stats = evaluate_snake_policy(eval_env, online, episodes=args.eval_episodes, seed_start=args.seed + 7000)
            print(
                f"  eval(grid={eval_grid}): avg_score={stats.avg_score:.2f} median={stats.median_score:.2f} "
                f"avg_len={stats.avg_length:.2f} avg_steps={stats.avg_steps:.2f} food_rate={stats.food_rate:.4f}"
            )
            if stats.avg_score > best_eval:
                best_eval = stats.avg_score
                best_path = save_dir / "snake_dqn_best.json"
                online.save(best_path)
                print(f"  saved new best checkpoint: {best_path}")

    final_path = save_dir / "snake_dqn_final.json"
    online.save(final_path)
    print(f"Training complete. Final model saved to: {final_path}")


if __name__ == "__main__":
    main()
