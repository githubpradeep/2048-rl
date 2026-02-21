from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import numpy as np

from .env import EnvConfig, Game2048Env
from .eval_utils import evaluate_policy
from .network import AdamOptimizer, MLPQNetwork
from .replay_buffer import ReplayBuffer


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


def argmax(values: list[float]) -> int:
    return max(range(len(values)), key=values.__getitem__)


def masked_argmax(values: list[float], legal_actions: list[int]) -> int:
    if not legal_actions:
        return argmax(values)
    return max(legal_actions, key=lambda action: values[action])


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a from-scratch DQN agent for 2048")
    parser.add_argument("--episodes", type=int, default=2000)
    parser.add_argument("--max-steps", type=int, default=2000)
    parser.add_argument("--buffer-size", type=int, default=100_000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--target-update", type=int, default=1000)
    parser.add_argument("--train-every", type=int, default=4)
    parser.add_argument("--warmup-steps", type=int, default=5000)
    parser.add_argument("--eps-start", type=float, default=1.0)
    parser.add_argument("--eps-end", type=float, default=0.05)
    parser.add_argument("--eps-decay-steps", type=int, default=300_000)
    parser.add_argument("--hidden-sizes", type=str, default="128,128")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-every", type=int, default=50)
    parser.add_argument("--eval-episodes", type=int, default=30)
    parser.add_argument("--save-dir", type=str, default="models")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    hidden_sizes = parse_hidden_sizes(args.hidden_sizes)

    env = Game2048Env(seed=args.seed, config=EnvConfig())
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
        invalid_moves = 0
        steps = 0
        info = {"score": 0, "max_tile": 0, "moved": True}

        while not done and steps < args.max_steps:
            legal_actions = env.game.legal_actions()
            epsilon = epsilon_by_step(global_step, args.eps_start, args.eps_end, args.eps_decay_steps)
            if rng.random() < epsilon:
                if legal_actions:
                    action = int(rng.choice(legal_actions))
                else:
                    action = int(rng.integers(0, env.action_size))
            else:
                q_values = online.predict_one(state)
                action = masked_argmax(q_values, legal_actions)

            next_state, reward, done, info = env.step(action)
            if not bool(info["moved"]):
                invalid_moves += 1

            buffer.add(state, action, reward, next_state, done)
            state = next_state

            episode_reward += reward
            steps += 1
            global_step += 1

            can_train = len(buffer) >= args.batch_size and global_step >= args.warmup_steps
            if can_train and global_step % args.train_every == 0:
                batch = buffer.sample(args.batch_size, rng)
                target_q = target.predict_batch(batch.next_states)
                max_next_q = [max(q_values) for q_values in target_q]
                td_target = [
                    float(reward) + args.gamma * (1.0 - float(done)) * float(max_q)
                    for reward, done, max_q in zip(batch.rewards, batch.dones, max_next_q)
                ]
                loss, td_abs = online.train_batch(batch.states, batch.actions, td_target, optimizer)
                losses.append(loss)
                td_abs_vals.append(td_abs)

            if global_step % args.target_update == 0:
                target.copy_from(online)

        mean_loss = float(np.mean(losses)) if losses else 0.0
        mean_td_abs = float(np.mean(td_abs_vals)) if td_abs_vals else 0.0
        invalid_ratio = invalid_moves / max(steps, 1)

        print(
            f"ep={episode:4d} steps={steps:4d} score={int(info['score']):6d} "
            f"max_tile={int(info['max_tile']):4d} reward={episode_reward:8.1f} "
            f"eps={epsilon_by_step(global_step, args.eps_start, args.eps_end, args.eps_decay_steps):.3f} "
            f"loss={mean_loss:.5f} td_abs={mean_td_abs:.5f} invalid={invalid_ratio:.3f}"
        )

        if episode % args.eval_every == 0:
            eval_env = Game2048Env(seed=args.seed + 777)
            stats = evaluate_policy(eval_env, online, episodes=args.eval_episodes, seed_start=args.seed + 5000)
            print(
                f"  eval: avg_score={stats.avg_score:.1f} median={stats.median_score:.1f} "
                f"reach_512={stats.reach_512:.2%} reach_1024={stats.reach_1024:.2%} reach_2048={stats.reach_2048:.2%}"
            )
            if stats.avg_score > best_eval:
                best_eval = stats.avg_score
                best_path = save_dir / "dqn_2048_best.json"
                online.save(best_path)
                print(f"  saved new best checkpoint: {best_path}")

    final_path = save_dir / "dqn_2048_final.json"
    online.save(final_path)
    print(f"Training complete. Final model saved to: {final_path}")


if __name__ == "__main__":
    main()
