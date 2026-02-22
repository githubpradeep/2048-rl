from __future__ import annotations

import argparse

from .flappy_env_config import add_flappy_preset_arg, apply_flappy_preset, build_flappy_config_from_args
from .flappy_eval_utils import evaluate_flappy_policy
from .flappy_heuristic import FlappyHeuristicPolicy
from .games.flappy import FlappyEnv


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate heuristic Flappy policy (env solvability baseline)")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=123)
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
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--bird-y-threshold", type=float, default=0.62)
    parser.add_argument("--dy-flap-threshold", type=float, default=0.08)
    parser.add_argument("--vel-flap-threshold", type=float, default=0.70)
    parser.add_argument("--bottom-emergency-y", type=float, default=0.90)
    add_flappy_preset_arg(parser)
    args = parser.parse_args()
    preset_name = apply_flappy_preset(args)

    config = build_flappy_config_from_args(args, include_rewards=False)
    env = FlappyEnv(config=config, seed=args.seed)
    policy = FlappyHeuristicPolicy(
        bird_y_threshold=args.bird_y_threshold,
        dy_flap_threshold=args.dy_flap_threshold,
        vel_flap_threshold=args.vel_flap_threshold,
        bottom_emergency_y=args.bottom_emergency_y,
    )
    stats = evaluate_flappy_policy(env, policy, episodes=args.episodes, seed_start=args.seed, max_steps=args.max_steps)

    print("Flappy Heuristic Evaluation Results")
    print("----------------------------------")
    print(f"Episodes: {args.episodes}")
    print(f"Average score: {stats.avg_score:.3f}")
    print(f"Median score: {stats.median_score:.3f}")
    print(f"Average steps: {stats.avg_steps:.2f}")
    print(f"Average reward: {stats.avg_reward:.2f}")
    print(f"Env preset: {preset_name or 'custom'}")


if __name__ == "__main__":
    main()

