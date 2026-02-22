from __future__ import annotations

import argparse

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
    args = parser.parse_args()

    env = FlappyEnv(
        config=FlappyConfig(
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
        ),
        seed=args.seed,
    )
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
