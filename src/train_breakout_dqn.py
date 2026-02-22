from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import numpy as np

from .breakout_eval_utils import evaluate_breakout_policy
from .games.breakout import BreakoutConfig, BreakoutEnv
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


def masked_argmax(values: list[float], legal_actions: list[int]) -> int:
    if not legal_actions:
        return max(range(len(values)), key=values.__getitem__)
    return max(legal_actions, key=lambda action: values[action])


def argmax(values: list[float]) -> int:
    return max(range(len(values)), key=values.__getitem__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train from-scratch DQN agent for Breakout")
    parser.add_argument("--episodes", type=int, default=3000)
    parser.add_argument("--max-steps", type=int, default=1200)
    parser.add_argument("--width", type=int, default=12)
    parser.add_argument("--height", type=int, default=14)
    parser.add_argument("--brick-rows", type=int, default=4)
    parser.add_argument("--brick-top", type=int, default=1)
    parser.add_argument("--paddle-width", type=int, default=3)
    parser.add_argument("--start-lives", type=int, default=3)
    parser.add_argument("--step-reward", type=float, default=0.02)
    parser.add_argument("--brick-reward", type=float, default=1.0)
    parser.add_argument("--paddle-hit-reward", type=float, default=0.03)
    parser.add_argument("--life-loss-penalty", type=float, default=-3.0)
    parser.add_argument("--clear-bonus", type=float, default=5.0)
    parser.add_argument("--buffer-size", type=int, default=100_000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--target-update", type=int, default=1000)
    parser.add_argument("--double-dqn", action="store_true")
    parser.add_argument("--dueling", action="store_true")
    parser.add_argument("--train-every", type=int, default=4)
    parser.add_argument("--warmup-steps", type=int, default=2000)
    parser.add_argument("--eps-start", type=float, default=1.0)
    parser.add_argument("--eps-end", type=float, default=0.05)
    parser.add_argument("--eps-decay-steps", type=int, default=150_000)
    parser.add_argument("--hidden-sizes", type=str, default="256,256")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-every", type=int, default=50)
    parser.add_argument("--eval-episodes", type=int, default=30)
    parser.add_argument("--save-dir", type=str, default="models/breakout")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    hidden_sizes = parse_hidden_sizes(args.hidden_sizes)

    config = BreakoutConfig(
        width=args.width,
        height=args.height,
        brick_rows=args.brick_rows,
        brick_top=args.brick_top,
        paddle_width=args.paddle_width,
        start_lives=args.start_lives,
        step_reward=args.step_reward,
        brick_reward=args.brick_reward,
        paddle_hit_reward=args.paddle_hit_reward,
        life_loss_penalty=args.life_loss_penalty,
        clear_bonus=args.clear_bonus,
        max_steps=args.max_steps,
    )
    env = BreakoutEnv(config=config, seed=args.seed)
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

    optimizer = AdamOptimizer(lr=args.lr)
    buffer = ReplayBuffer(capacity=args.buffer_size, state_dim=state_dim)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    global_step = 0
    best_eval = -float("inf")

    print(
        f"Breakout training config | board={args.width}x{args.height} bricks={args.brick_rows}x{args.width} "
        f"paddle={args.paddle_width} lives={args.start_lives} double_dqn={args.double_dqn} dueling={args.dueling}"
    )

    for episode in range(1, args.episodes + 1):
        state = env.reset(seed=args.seed + episode)
        done = False
        episode_reward = 0.0
        losses = []
        td_abs_vals = []
        steps = 0
        brick_hits = 0
        life_losses = 0
        paddle_hits = 0
        info = {"score": 0, "lives": config.start_lives, "bricks_left": env.game.bricks_left(), "cleared": False}

        while not done and steps < args.max_steps:
            legal_actions = env.legal_actions()
            epsilon = epsilon_by_step(global_step, args.eps_start, args.eps_end, args.eps_decay_steps)
            if rng.random() < epsilon:
                action = int(rng.choice(legal_actions)) if legal_actions else int(rng.integers(0, env.action_size))
            else:
                q_values = online.predict_one(state)
                action = masked_argmax(q_values, legal_actions)

            next_state, reward, done, info = env.step(action)
            brick_hits += int(info["bricks_hit"])
            life_losses += 1 if bool(info["life_lost"]) else 0
            paddle_hits += 1 if bool(info["paddle_hit"]) else 0

            buffer.add(state, action, reward, next_state, done)
            state = next_state

            episode_reward += reward
            steps += 1
            global_step += 1

            can_train = len(buffer) >= args.batch_size and global_step >= args.warmup_steps
            if can_train and global_step % args.train_every == 0:
                batch = buffer.sample(args.batch_size, rng)
                target_q = target.predict_batch(batch.next_states)
                if args.double_dqn:
                    online_q = online.predict_batch(batch.next_states)
                    next_actions = [argmax(values) for values in online_q]
                    max_next_q = [float(target_q[i][next_actions[i]]) for i in range(len(next_actions))]
                else:
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
            f"ep={episode:4d} steps={steps:4d} score={int(info['score']):4d} lives={int(info['lives']):2d} "
            f"brick_hits={brick_hits:3d} paddle_hits={paddle_hits:3d} life_loss={life_losses:2d} "
            f"reward={episode_reward:8.2f} eps={epsilon_by_step(global_step, args.eps_start, args.eps_end, args.eps_decay_steps):.3f} "
            f"loss={mean_loss:.5f} td_abs={mean_td_abs:.5f}"
        )

        if episode % args.eval_every == 0:
            eval_env = BreakoutEnv(config=config, seed=args.seed + 999)
            stats = evaluate_breakout_policy(eval_env, online, episodes=args.eval_episodes, seed_start=args.seed + 7000)
            print(
                f"  eval: avg_score={stats.avg_score:.2f} median={stats.median_score:.2f} "
                f"avg_steps={stats.avg_steps:.2f} avg_lives={stats.avg_lives_left:.2f} "
                f"avg_bricks={stats.avg_bricks_broken:.2f} clear_rate={100.0*stats.clear_rate:.1f}%"
            )
            if stats.avg_score > best_eval:
                best_eval = stats.avg_score
                best_path = save_dir / "breakout_dqn_best.json"
                online.save(best_path)
                print(f"  saved new best checkpoint: {best_path}")

    final_path = save_dir / "breakout_dqn_final.json"
    online.save(final_path)
    print(f"Training complete. Final model saved to: {final_path}")


if __name__ == "__main__":
    main()

