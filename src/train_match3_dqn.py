from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import numpy as np

from .games.match3 import Match3Config, Match3Env
from .match3_eval_utils import evaluate_match3_policy
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Train from-scratch DQN agent for Match-3")
    parser.add_argument("--episodes", type=int, default=4000)
    parser.add_argument("--max-steps", type=int, default=150)
    parser.add_argument("--width", type=int, default=6)
    parser.add_argument("--height", type=int, default=6)
    parser.add_argument("--num-colors", type=int, default=5)
    parser.add_argument("--step-reward", type=float, default=-0.01)
    parser.add_argument("--tile-reward", type=float, default=0.20)
    parser.add_argument("--combo-bonus", type=float, default=0.15)
    parser.add_argument("--invalid-penalty", type=float, default=-0.50)
    parser.add_argument("--no-reshuffle-on-stuck", dest="reshuffle_on_stuck", action="store_false")
    parser.set_defaults(reshuffle_on_stuck=True)
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
    parser.add_argument("--eps-decay-steps", type=int, default=120_000)
    parser.add_argument("--hidden-sizes", type=str, default="256,256")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-every", type=int, default=50)
    parser.add_argument("--eval-episodes", type=int, default=30)
    parser.add_argument("--save-dir", type=str, default="models/match3")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    hidden_sizes = parse_hidden_sizes(args.hidden_sizes)

    config = Match3Config(
        width=args.width,
        height=args.height,
        num_colors=args.num_colors,
        max_steps=args.max_steps,
        step_reward=args.step_reward,
        tile_reward=args.tile_reward,
        combo_bonus=args.combo_bonus,
        invalid_penalty=args.invalid_penalty,
        reshuffle_on_stuck=args.reshuffle_on_stuck,
    )
    env = Match3Env(config=config, seed=args.seed)
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
    buffer = ReplayBuffer(capacity=args.buffer_size, state_dim=state_dim, action_dim=env.action_size)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    global_step = 0
    best_eval = -float("inf")

    print(
        f"Match3 training config | board={args.width}x{args.height} colors={args.num_colors} "
        f"actions={env.action_size} double_dqn={args.double_dqn} dueling={args.dueling}"
    )

    for episode in range(1, args.episodes + 1):
        state = env.reset(seed=args.seed + episode)
        done = False
        episode_reward = 0.0
        losses = []
        td_abs_vals = []
        steps = 0
        total_tiles = 0
        total_cascades = 0
        invalid_steps = 0
        info = {"score": 0, "invalid": False, "tiles_cleared": 0, "cascades": 0, "legal_count": len(env.legal_actions())}

        while not done and steps < args.max_steps:
            legal_actions = env.legal_actions()
            epsilon = epsilon_by_step(global_step, args.eps_start, args.eps_end, args.eps_decay_steps)
            if rng.random() < epsilon:
                action = int(rng.choice(np.asarray(legal_actions, dtype=np.int64))) if legal_actions else int(rng.integers(0, env.action_size))
            else:
                q_values = online.predict_one(state)
                action = masked_argmax(q_values, legal_actions)

            next_state, reward, done, info = env.step(action)
            total_tiles += int(info["tiles_cleared"])
            total_cascades += int(info["cascades"])
            invalid_steps += 1 if bool(info["invalid"]) else 0

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
                if legal_masks is None:
                    raise RuntimeError("Match3 trainer requires next_legal_masks in replay buffer")
                if args.double_dqn:
                    online_q = np.asarray(online.predict_batch(batch.next_states), dtype=np.float32)
                    max_next_q = masked_double_q(online_q, target_q, legal_masks)
                else:
                    max_next_q = masked_max_q(target_q, legal_masks)

                td_target = batch.rewards + args.gamma * (1.0 - batch.dones) * max_next_q
                loss, td_abs = online.train_batch(batch.states, batch.actions, td_target, optimizer)
                losses.append(loss)
                td_abs_vals.append(td_abs)

            if global_step % args.target_update == 0:
                target.copy_from(online)

        mean_loss = float(np.mean(losses)) if losses else 0.0
        mean_td_abs = float(np.mean(td_abs_vals)) if td_abs_vals else 0.0

        print(
            f"ep={episode:4d} steps={steps:4d} score={int(info['score']):4d} tiles={total_tiles:4d} "
            f"cascades={total_cascades:3d} reward={episode_reward:8.2f} "
            f"eps={epsilon_by_step(global_step, args.eps_start, args.eps_end, args.eps_decay_steps):.3f} "
            f"loss={mean_loss:.5f} td_abs={mean_td_abs:.5f} invalid={invalid_steps/max(steps,1):.3f}"
        )

        if episode % args.eval_every == 0:
            eval_env = Match3Env(config=config, seed=args.seed + 999)
            stats = evaluate_match3_policy(eval_env, online, episodes=args.eval_episodes, seed_start=args.seed + 7000, max_steps=args.max_steps)
            print(
                f"  eval: avg_score={stats.avg_score:.2f} median={stats.median_score:.2f} "
                f"avg_steps={stats.avg_steps:.2f} avg_tiles={stats.avg_tiles_cleared:.2f} "
                f"avg_cascades={stats.avg_cascades:.2f} invalid_rate={stats.invalid_rate:.3f}"
            )
            if stats.avg_score > best_eval:
                best_eval = stats.avg_score
                best_path = save_dir / "match3_dqn_best.json"
                online.save(best_path)
                print(f"  saved new best checkpoint: {best_path}")

    final_path = save_dir / "match3_dqn_final.json"
    online.save(final_path)
    print(f"Training complete. Final model saved to: {final_path}")


if __name__ == "__main__":
    main()

