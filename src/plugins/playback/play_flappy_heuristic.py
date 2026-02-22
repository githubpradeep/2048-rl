from __future__ import annotations

import argparse

from ..flappy.env_config import add_flappy_preset_arg, apply_flappy_preset, build_flappy_config_from_args
from ..flappy.heuristic import FlappyHeuristicPolicy
from ...games.flappy import FlappyEnv
from .play_flappy_agent import run_pygame, run_terminal


def main() -> None:
    parser = argparse.ArgumentParser(description="Autoplay Flappy Bird with a heuristic policy")
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
    parser.add_argument("--delay", type=float, default=0.08)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--mode", choices=["terminal", "pygame"], default="terminal")
    parser.add_argument("--close-on-end", action="store_true")
    parser.add_argument("--bird-y-threshold", type=float, default=0.62)
    parser.add_argument("--dy-flap-threshold", type=float, default=0.08)
    parser.add_argument("--vel-flap-threshold", type=float, default=0.70)
    parser.add_argument("--bottom-emergency-y", type=float, default=0.90)
    add_flappy_preset_arg(parser)
    args = parser.parse_args()
    apply_flappy_preset(args)

    config = build_flappy_config_from_args(args, include_rewards=False)
    env = FlappyEnv(config=config, seed=args.seed)
    policy = FlappyHeuristicPolicy(
        bird_y_threshold=args.bird_y_threshold,
        dy_flap_threshold=args.dy_flap_threshold,
        vel_flap_threshold=args.vel_flap_threshold,
        bottom_emergency_y=args.bottom_emergency_y,
    )

    if args.mode == "pygame":
        result = run_pygame(env, policy, args.seed, args.delay, args.max_steps, args.close_on_end)
    else:
        result = run_terminal(env, policy, args.seed, args.delay, args.max_steps)

    print("Flappy heuristic run finished")
    print(f"Final score: {result.score}")
    print(f"Steps: {result.steps}")


if __name__ == "__main__":
    main()

