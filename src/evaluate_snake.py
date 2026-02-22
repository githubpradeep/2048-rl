from __future__ import annotations

import argparse

from .games.snake import SnakeConfig, SnakeEnv, SnakeFeatureEnv
from .network import MLPQNetwork
from .snake_eval_utils import evaluate_snake_policy


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate trained DQN for Snake")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--grid-size", type=int, default=10)
    parser.add_argument("--state-grid-size", type=int, default=0)
    parser.add_argument("--state-mode", type=str, choices=["board", "features"], default="board")
    args = parser.parse_args()

    state_grid_size = args.state_grid_size if args.state_grid_size > 0 else args.grid_size
    if args.state_mode == "features":
        env = SnakeFeatureEnv(config=SnakeConfig(grid_size=args.grid_size), seed=args.seed)
    else:
        env = SnakeEnv(config=SnakeConfig(grid_size=args.grid_size), seed=args.seed, state_grid_size=state_grid_size)
    network = MLPQNetwork.load(args.model)
    stats = evaluate_snake_policy(env, network, episodes=args.episodes, seed_start=args.seed)

    print("Snake Evaluation Results")
    print("------------------------")
    print(f"Episodes: {args.episodes}")
    print(f"Average score: {stats.avg_score:.3f}")
    print(f"Median score: {stats.median_score:.3f}")
    print(f"Average steps: {stats.avg_steps:.2f}")
    print(f"Average length: {stats.avg_length:.2f}")
    print(f"Food rate (foods/step): {stats.food_rate:.5f}")


if __name__ == "__main__":
    main()
