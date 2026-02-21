from __future__ import annotations

import argparse

from .env import Game2048Env
from .eval_utils import evaluate_policy
from .network import MLPQNetwork


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained from-scratch DQN for 2048")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    env = Game2048Env(seed=args.seed)
    network = MLPQNetwork.load(args.model)
    stats = evaluate_policy(env, network, episodes=args.episodes, seed_start=args.seed)

    print("Evaluation Results")
    print("------------------")
    print(f"Episodes: {args.episodes}")
    print(f"Average score: {stats.avg_score:.2f}")
    print(f"Median score: {stats.median_score:.2f}")
    print(f"Average steps: {stats.avg_steps:.2f}")
    print(f"Reach >= 512:  {stats.reach_512:.2%}")
    print(f"Reach >= 1024: {stats.reach_1024:.2%}")
    print(f"Reach >= 2048: {stats.reach_2048:.2%}")
    print("Max tile distribution:")
    for tile in sorted(stats.tile_distribution):
        print(f"  {tile:4d}: {stats.tile_distribution[tile]}")


if __name__ == "__main__":
    main()
