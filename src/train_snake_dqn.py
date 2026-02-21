from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import numpy as np

from .games.snake import SnakeConfig, SnakeEnv
from .network import AdamOptimizer, MLPQNetwork
from .replay_buffer import ReplayBuffer
from .snake_eval_utils import evaluate_snake_policy


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


def main() -> None:
    parser = argparse.ArgumentParser(description="Train from-scratch DQN agent for Snake")
    parser.add_argument("--episodes", type=int, default=1500)
    parser.add_argument("--max-steps", type=int, default=1200)
    parser.add_argument("--grid-size", type=int, default=10)
    parser.add_argument("--buffer-size", type=int, default=80_000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--target-update", type=int, default=1000)
    parser.add_argument("--train-every", type=int, default=4)
    parser.add_argument("--warmup-steps", type=int, default=4000)
    parser.add_argument("--eps-start", type=float, default=1.0)
    parser.add_argument("--eps-end", type=float, default=0.05)
    parser.add_argument("--eps-decay-steps", type=int, default=250_000)
    parser.add_argument("--hidden-sizes", type=str, default="256,256")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-every", type=int, default=50)
    parser.add_argument("--eval-episodes", type=int, default=25)
    parser.add_argument("--save-dir", type=str, default="models/snake")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    hidden_sizes = parse_hidden_sizes(args.hidden_sizes)

    env = SnakeEnv(config=SnakeConfig(grid_size=args.grid_size), seed=args.seed)
    state_dim = env.reset(seed=args.seed).shape[0]

    online = MLPQNetwork(
        input_dim=state_dim,
        output_dim=env.action_size,
        hidden_sizes=hidden_sizes,
        seed=args.seed,
    )
    target = MLPQNetwork(
        input_dim=state_dim,
        output_dim=env.action_size,
        hidden_sizes=hidden_sizes,
        seed=args.seed + 1,
    )
    target.copy_from(online)

    optimizer = AdamOptimizer(lr=args.lr)
    buffer = ReplayBuffer(capacity=args.buffer_size, state_dim=state_dim)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    global_step = 0
    best_eval = -float("inf")

    for episode in range(1, args.episodes + 1):
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

            buffer.add(state, action, reward, next_state, done)
            state = next_state

            episode_reward += reward
            steps += 1
            global_step += 1

            can_train = len(buffer) >= args.batch_size and global_step >= args.warmup_steps
            if can_train and global_step % args.train_every == 0:
                batch = buffer.sample(args.batch_size, rng)
                target_q = target.predict_batch(batch.next_states)
                max_next_q = [max(values) for values in target_q]
                td_target = [
                    float(reward_i) + args.gamma * (1.0 - float(done_i)) * float(max_q)
                    for reward_i, done_i, max_q in zip(batch.rewards, batch.dones, max_next_q)
                ]
                loss, td_abs = online.train_batch(batch.states, batch.actions, td_target, optimizer)
                losses.append(loss)
                td_abs_vals.append(td_abs)

            if global_step % args.target_update == 0:
                target.copy_from(online)

        mean_loss = float(np.mean(losses)) if losses else 0.0
        mean_td_abs = float(np.mean(td_abs_vals)) if td_abs_vals else 0.0

        print(
            f"ep={episode:4d} steps={steps:4d} score={int(info['score']):4d} len={int(info['length']):3d} "
            f"foods={foods:3d} reward={episode_reward:8.2f} "
            f"eps={epsilon_by_step(global_step, args.eps_start, args.eps_end, args.eps_decay_steps):.3f} "
            f"loss={mean_loss:.5f} td_abs={mean_td_abs:.5f}"
        )

        if episode % args.eval_every == 0:
            eval_env = SnakeEnv(config=SnakeConfig(grid_size=args.grid_size), seed=args.seed + 999)
            stats = evaluate_snake_policy(eval_env, online, episodes=args.eval_episodes, seed_start=args.seed + 7000)
            print(
                f"  eval: avg_score={stats.avg_score:.2f} median={stats.median_score:.2f} "
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
