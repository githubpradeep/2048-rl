from __future__ import annotations

import argparse

from .flappy_env_config import (
    add_env_mismatch_args,
    add_flappy_preset_arg,
    apply_flappy_preset,
    build_flappy_config_from_args,
    validate_model_env_or_raise,
)
from .flappy_eval_utils import evaluate_flappy_policy
from .games.flappy import FlappyConfig, FlappyEnv
from .network import MLPQNetwork


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate trained DQN for Flappy Bird")
    parser.add_argument("--model", type=str, required=True)
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
    add_flappy_preset_arg(parser)
    add_env_mismatch_args(parser)
    args = parser.parse_args()
    preset_name = apply_flappy_preset(args)

    config = build_flappy_config_from_args(args, include_rewards=False)
    validate_model_env_or_raise(
        args.model,
        config,
        allow_mismatch=bool(args.allow_env_mismatch),
        print_model_env=bool(args.print_model_env),
    )
    env = FlappyEnv(config=config, seed=args.seed)
    network = MLPQNetwork.load(args.model)
    stats = evaluate_flappy_policy(env, network, episodes=args.episodes, seed_start=args.seed, max_steps=args.max_steps)

    print("Flappy Evaluation Results")
    print("-------------------------")
    print(f"Episodes: {args.episodes}")
    print(f"Average score: {stats.avg_score:.3f}")
    print(f"Median score: {stats.median_score:.3f}")
    print(f"Average steps: {stats.avg_steps:.2f}")
    print(f"Average reward: {stats.avg_reward:.2f}")
    print(f"Env preset: {preset_name or 'custom'}")


if __name__ == "__main__":
    main()
