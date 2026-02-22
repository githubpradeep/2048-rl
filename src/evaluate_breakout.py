from __future__ import annotations

import argparse

from .breakout_eval_utils import evaluate_breakout_policy
from .games.breakout import BreakoutConfig, BreakoutEnv
from .network import MLPQNetwork


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate trained DQN for Breakout")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=123)
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
    parser.add_argument("--max-steps", type=int, default=1200)
    args = parser.parse_args()

    env = BreakoutEnv(
        config=BreakoutConfig(
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
        ),
        seed=args.seed,
    )
    network = MLPQNetwork.load(args.model)
    stats = evaluate_breakout_policy(env, network, episodes=args.episodes, seed_start=args.seed, max_steps=args.max_steps)

    print("Breakout Evaluation Results")
    print("--------------------------")
    print(f"Episodes: {args.episodes}")
    print(f"Average score: {stats.avg_score:.3f}")
    print(f"Median score: {stats.median_score:.3f}")
    print(f"Average steps: {stats.avg_steps:.2f}")
    print(f"Average lives left: {stats.avg_lives_left:.2f}")
    print(f"Average bricks broken: {stats.avg_bricks_broken:.2f}")
    print(f"Clear rate: {100.0 * stats.clear_rate:.2f}%")


if __name__ == "__main__":
    main()

