from __future__ import annotations

import argparse

from .flappy_env_config import (
    add_env_mismatch_args,
    add_flappy_preset_arg,
    apply_flappy_preset,
    build_flappy_config_from_args,
    validate_model_env_or_raise,
)
from .flappy_tabular import FlappyTabularQAgent
from .games.flappy import FlappyConfig, FlappyEnv
from .play_flappy_agent import run_pygame, run_terminal


def main() -> None:
    parser = argparse.ArgumentParser(description="Autoplay Flappy Bird with a tabular agent")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--seed", type=int, default=2026)
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
    parser.add_argument("--delay", type=float, default=0.08)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--mode", choices=["terminal", "pygame"], default="terminal")
    parser.add_argument("--close-on-end", action="store_true")
    add_flappy_preset_arg(parser)
    add_env_mismatch_args(parser)
    args = parser.parse_args()
    apply_flappy_preset(args)

    config = build_flappy_config_from_args(args, include_rewards=True)
    validate_model_env_or_raise(
        args.model,
        config,
        allow_mismatch=bool(args.allow_env_mismatch),
        print_model_env=bool(args.print_model_env),
    )
    env = FlappyEnv(config=config, seed=args.seed)
    agent = FlappyTabularQAgent.load(args.model)

    if args.mode == "pygame":
        result = run_pygame(env, agent, args.seed, args.delay, args.max_steps, args.close_on_end)
    else:
        result = run_terminal(env, agent, args.seed, args.delay, args.max_steps)

    print("Flappy game finished")
    print(f"Final score: {result.score}")
    print(f"Steps: {result.steps}")


if __name__ == "__main__":
    main()
