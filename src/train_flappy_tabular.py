from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .flappy_env_config import (
    add_flappy_preset_arg,
    apply_flappy_preset,
    build_flappy_config_from_args,
    flappy_model_metadata,
    write_model_metadata,
)
from .flappy_eval_utils import evaluate_flappy_policy
from .flappy_tabular import FlappyDiscretizerConfig, FlappyStateDiscretizer, FlappyTabularQAgent
from .games.flappy import FlappyConfig, FlappyEnv


def epsilon_by_episode(ep_idx: int, start: float, end: float, decay_episodes: int) -> float:
    if ep_idx >= decay_episodes:
        return end
    frac = ep_idx / max(decay_episodes, 1)
    return float(start + frac * (end - start))


def flappy_gap_alignment_shaping(
    state: np.ndarray,
    next_state: np.ndarray,
    passed_pipes: int,
    scale: float,
) -> float:
    if scale == 0.0 or passed_pipes > 0:
        return 0.0
    curr_dx = float(state[2])
    next_dx = float(next_state[2])
    curr_abs_dy = abs(float(state[4]))
    next_abs_dy = abs(float(next_state[4]))
    if curr_dx < -0.1 or next_dx < -0.2:
        return 0.0
    return float(scale * (curr_abs_dy - next_abs_dy))


def main() -> None:
    parser = argparse.ArgumentParser(description="Train tabular/discretized Q-learning agent for Flappy Bird")
    parser.add_argument("--episodes", type=int, default=10000)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)

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

    parser.add_argument("--step-reward", type=float, default=1.0)
    parser.add_argument("--pass-reward", type=float, default=100.0)
    parser.add_argument("--crash-penalty", type=float, default=-100.0)
    parser.add_argument("--align-shaping-scale", type=float, default=0.0)

    parser.add_argument("--alpha", type=float, default=0.10)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--optimistic-init", type=float, default=0.0)
    parser.add_argument("--eps-start", type=float, default=1.0)
    parser.add_argument("--eps-end", type=float, default=0.02)
    parser.add_argument("--eps-decay-episodes", type=int, default=1000)

    parser.add_argument("--dx-bins", type=int, default=24)
    parser.add_argument("--dy-bins", type=int, default=31)
    parser.add_argument("--vel-bins", type=int, default=15)

    parser.set_defaults(reverse_sweep=True)
    parser.add_argument("--reverse-sweep", dest="reverse_sweep", action="store_true", help="Run a reverse TD sweep over each episode.")
    parser.add_argument("--no-reverse-sweep", dest="reverse_sweep", action="store_false")
    parser.add_argument("--reverse-passes", type=int, default=1)

    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--eval-episodes", type=int, default=50)
    parser.add_argument("--save-dir", type=str, default="models/flappy_tabular")
    add_flappy_preset_arg(parser)
    args = parser.parse_args()
    preset_name = apply_flappy_preset(args)

    rng = np.random.default_rng(args.seed)
    config = build_flappy_config_from_args(args, include_rewards=True)
    env = FlappyEnv(config=config, seed=args.seed)

    discretizer = FlappyStateDiscretizer(
        FlappyDiscretizerConfig(
            dx_bins=args.dx_bins,
            dy_bins=args.dy_bins,
            vel_bins=args.vel_bins,
        )
    )
    agent = FlappyTabularQAgent(
        action_size=env.action_size,
        discretizer=discretizer,
        alpha=args.alpha,
        gamma=args.gamma,
        optimistic_init=args.optimistic_init,
    )

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    best_eval = -float("inf")

    print(
        "Flappy tabular training config | "
        f"preset={preset_name or 'custom'} "
        f"bins(dx={args.dx_bins},dy={args.dy_bins},vel={args.vel_bins}) "
        f"alpha={args.alpha} gamma={args.gamma} reverse_sweep={args.reverse_sweep}"
    )

    for episode in range(1, args.episodes + 1):
        state = env.reset(seed=args.seed + episode)
        done = False
        steps = 0
        total_passed = 0
        episode_reward = 0.0
        td_abs_vals: list[float] = []
        info = {"score": 0, "passed_pipes": 0}
        transitions: list[tuple[tuple[int, int, int], int, float, tuple[int, int, int], bool]] = []

        epsilon = epsilon_by_episode(episode - 1, args.eps_start, args.eps_end, args.eps_decay_episodes)

        while not done and steps < args.max_steps:
            legal = env.legal_actions()
            action, state_key = agent.select_action(state, legal, epsilon, rng)
            next_state, reward, done, info = env.step(action)
            reward += flappy_gap_alignment_shaping(
                state=state,
                next_state=next_state,
                passed_pipes=int(info["passed_pipes"]),
                scale=args.align_shaping_scale,
            )
            next_key = discretizer.encode(next_state)
            total_passed += int(info["passed_pipes"])

            if args.reverse_sweep:
                transitions.append((state_key, int(action), float(reward), next_key, bool(done)))
            else:
                td_abs = agent.update_q_learning(state_key, int(action), float(reward), next_key, bool(done), env.legal_actions())
                td_abs_vals.append(td_abs)

            state = next_state
            episode_reward += float(reward)
            steps += 1

        if args.reverse_sweep and transitions:
            reverse_passes = max(int(args.reverse_passes), 1)
            for _ in range(reverse_passes):
                for state_key, action, reward, next_key, done_t in reversed(transitions):
                    td_abs = agent.update_q_learning(state_key, action, reward, next_key, done_t, [0, 1] if not done_t else [])
                    td_abs_vals.append(td_abs)

        mean_td = float(np.mean(td_abs_vals)) if td_abs_vals else 0.0
        print(
            f"ep={episode:5d} steps={steps:4d} score={int(info['score']):4d} "
            f"passed={total_passed:3d} reward={episode_reward:8.2f} eps={epsilon:.3f} "
            f"td_abs={mean_td:.5f} q_states={len(agent.q_table):6d}"
        )

        if episode % args.eval_every == 0:
            eval_env = FlappyEnv(config=config, seed=args.seed + 999)
            stats = evaluate_flappy_policy(eval_env, agent, episodes=args.eval_episodes, seed_start=args.seed + 10000)
            print(
                f"  eval: avg_score={stats.avg_score:.2f} median={stats.median_score:.2f} "
                f"avg_steps={stats.avg_steps:.2f} avg_reward={stats.avg_reward:.2f}"
            )
            if stats.avg_score > best_eval:
                best_eval = stats.avg_score
                best_path = save_dir / "flappy_tabular_best.json"
                agent.save(best_path)
                write_model_metadata(best_path, flappy_model_metadata(config, preset_name, algo="tabular_q"))
                print(f"  saved new best checkpoint: {best_path}")

    final_path = save_dir / "flappy_tabular_final.json"
    agent.save(final_path)
    write_model_metadata(final_path, flappy_model_metadata(config, preset_name, algo="tabular_q"))
    print(f"Training complete. Final model saved to: {final_path}")


if __name__ == "__main__":
    main()
