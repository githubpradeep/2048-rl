from __future__ import annotations

import argparse

from .games.match3 import Match3Config, Match3Env
from .match3_eval_utils import evaluate_match3_policy
from .network import MLPQNetwork


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate trained DQN for Match-3")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--width", type=int, default=6)
    parser.add_argument("--height", type=int, default=6)
    parser.add_argument("--num-colors", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=150)
    parser.add_argument("--step-reward", type=float, default=-0.01)
    parser.add_argument("--tile-reward", type=float, default=0.20)
    parser.add_argument("--combo-bonus", type=float, default=0.15)
    parser.add_argument("--invalid-penalty", type=float, default=-0.50)
    parser.add_argument("--no-reshuffle-on-stuck", dest="reshuffle_on_stuck", action="store_false")
    parser.set_defaults(reshuffle_on_stuck=True)
    args = parser.parse_args()

    env = Match3Env(
        config=Match3Config(
            width=args.width,
            height=args.height,
            num_colors=args.num_colors,
            max_steps=args.max_steps,
            step_reward=args.step_reward,
            tile_reward=args.tile_reward,
            combo_bonus=args.combo_bonus,
            invalid_penalty=args.invalid_penalty,
            reshuffle_on_stuck=args.reshuffle_on_stuck,
        ),
        seed=args.seed,
    )
    network = MLPQNetwork.load(args.model)
    stats = evaluate_match3_policy(env, network, episodes=args.episodes, seed_start=args.seed, max_steps=args.max_steps)

    print("Match-3 Evaluation Results")
    print("--------------------------")
    print(f"Episodes: {args.episodes}")
    print(f"Average score: {stats.avg_score:.3f}")
    print(f"Median score: {stats.median_score:.3f}")
    print(f"Average steps: {stats.avg_steps:.2f}")
    print(f"Average tiles cleared: {stats.avg_tiles_cleared:.2f}")
    print(f"Average cascades: {stats.avg_cascades:.2f}")
    print(f"Invalid rate: {stats.invalid_rate:.4f}")


if __name__ == "__main__":
    main()

