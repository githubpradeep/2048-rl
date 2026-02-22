from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import numpy as np

from .flappy_eval_utils import evaluate_flappy_policy
from .games.flappy import FlappyConfig, FlappyEnv
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


def flappy_gap_alignment_shaping(
    state: np.ndarray,
    next_state: np.ndarray,
    passed_pipes: int,
    scale: float,
) -> float:
    """Dense shaping: reward moving closer to the next pipe gap center."""
    if scale == 0.0 or passed_pipes > 0:
        return 0.0

    # State layout:
    # [bird_y, bird_vel, p1_dx, p1_gap_y, bird_minus_p1_gap, p2_dx, p2_gap_y, bird_minus_p2_gap]
    curr_dx = float(state[2])
    next_dx = float(next_state[2])
    curr_abs_dy = abs(float(state[4]))
    next_abs_dy = abs(float(next_state[4]))

    # Only apply shaping while approaching the next pipe.
    if curr_dx < -0.1 or next_dx < -0.2:
        return 0.0

    return float(scale * (curr_abs_dy - next_abs_dy))


def main() -> None:
    parser = argparse.ArgumentParser(description="Train from-scratch DQN agent for Flappy Bird")
    parser.add_argument("--episodes", type=int, default=3000)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--width", type=int, default=84)
    parser.add_argument("--height", type=int, default=84)
    parser.add_argument("--pipe-gap", type=float, default=30.0)
    parser.add_argument("--pipe-speed", type=float, default=1.3)
    parser.add_argument("--pipe-spacing", type=float, default=44.0)
    parser.add_argument("--initial-pipe-offset", type=float, default=20.0)
    parser.add_argument("--floor-height", type=float, default=10.0)
    parser.add_argument("--max-gap-delta", type=float, default=8.0)
    parser.add_argument("--gravity", type=float, default=0.18)
    parser.add_argument("--flap-velocity", type=float, default=-2.2)
    parser.add_argument("--buffer-size", type=int, default=100_000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--max-grad-norm", type=float, default=5.0)
    parser.add_argument("--target-update", type=int, default=1000)
    parser.add_argument("--double-dqn", action="store_true")
    parser.add_argument("--dueling", action="store_true")
    parser.add_argument("--train-every", type=int, default=4)
    parser.add_argument("--warmup-steps", type=int, default=1000)
    parser.add_argument("--eps-start", type=float, default=1.0)
    parser.add_argument("--eps-end", type=float, default=0.05)
    parser.add_argument("--eps-decay-steps", type=int, default=40_000)
    parser.add_argument("--hidden-sizes", type=str, default="128,128")
    parser.add_argument("--step-reward", type=float, default=0.02)
    parser.add_argument("--pass-reward", type=float, default=5.0)
    parser.add_argument("--crash-penalty", type=float, default=-3.0)
    parser.add_argument("--align-shaping-scale", type=float, default=2.0)
    parser.add_argument("--loss", type=str, default="huber", choices=["mse", "huber"])
    parser.add_argument("--huber-delta", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-every", type=int, default=50)
    parser.add_argument("--eval-episodes", type=int, default=30)
    parser.add_argument("--save-dir", type=str, default="models/flappy")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    hidden_sizes = parse_hidden_sizes(args.hidden_sizes)

    config = FlappyConfig(
        width=args.width,
        height=args.height,
        gap_size=args.pipe_gap,
        pipe_speed=args.pipe_speed,
        pipe_spacing=args.pipe_spacing,
        initial_pipe_offset=args.initial_pipe_offset,
        floor_height=args.floor_height,
        max_gap_delta=args.max_gap_delta,
        gravity=args.gravity,
        flap_velocity=args.flap_velocity,
        step_reward=args.step_reward,
        pass_reward=args.pass_reward,
        crash_penalty=args.crash_penalty,
        max_steps=args.max_steps,
    )
    env = FlappyEnv(config=config, seed=args.seed)
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
    buffer = ReplayBuffer(capacity=args.buffer_size, state_dim=state_dim)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    global_step = 0
    best_eval = -float("inf")

    print(
        f"Flappy training config | board={args.width}x{args.height} "
        f"double_dqn={args.double_dqn} dueling={args.dueling} "
        f"rewards(step={args.step_reward},pass={args.pass_reward},crash={args.crash_penalty}) "
        f"align_shaping={args.align_shaping_scale}"
    )

    for episode in range(1, args.episodes + 1):
        state = env.reset(seed=args.seed + episode)
        done = False
        episode_reward = 0.0
        losses = []
        td_abs_vals = []
        steps = 0
        total_passed = 0
        info = {"score": 0, "passed_pipes": 0}

        while not done and steps < args.max_steps:
            legal_actions = env.legal_actions()
            epsilon = epsilon_by_step(global_step, args.eps_start, args.eps_end, args.eps_decay_steps)
            if rng.random() < epsilon:
                action = int(rng.choice(legal_actions)) if legal_actions else int(rng.integers(0, env.action_size))
            else:
                q_values = online.predict_one(state)
                action = masked_argmax(q_values, legal_actions)

            next_state, reward, done, info = env.step(action)
            reward += flappy_gap_alignment_shaping(
                state=state,
                next_state=next_state,
                passed_pipes=int(info["passed_pipes"]),
                scale=args.align_shaping_scale,
            )
            total_passed += int(info["passed_pipes"])

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
            f"ep={episode:4d} steps={steps:4d} score={int(info['score']):4d} "
            f"passed={total_passed:3d} reward={episode_reward:8.2f} "
            f"eps={epsilon_by_step(global_step, args.eps_start, args.eps_end, args.eps_decay_steps):.3f} "
            f"loss={mean_loss:.5f} td_abs={mean_td_abs:.5f}"
        )

        if episode % args.eval_every == 0:
            eval_env = FlappyEnv(config=config, seed=args.seed + 999)
            stats = evaluate_flappy_policy(eval_env, online, episodes=args.eval_episodes, seed_start=args.seed + 7000)
            print(
                f"  eval: avg_score={stats.avg_score:.2f} median={stats.median_score:.2f} "
                f"avg_steps={stats.avg_steps:.2f} avg_reward={stats.avg_reward:.2f}"
            )
            if stats.avg_score > best_eval:
                best_eval = stats.avg_score
                best_path = save_dir / "flappy_dqn_best.json"
                online.save(best_path)
                print(f"  saved new best checkpoint: {best_path}")

    final_path = save_dir / "flappy_dqn_final.json"
    online.save(final_path)
    print(f"Training complete. Final model saved to: {final_path}")


if __name__ == "__main__":
    main()
