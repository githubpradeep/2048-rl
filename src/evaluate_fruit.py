from __future__ import annotations

import argparse

from .fruit_eval_utils import evaluate_fruit_policy
from .games.fruit_cutter import FruitCutterConfig, FruitCutterEnv
from .network import MLPQNetwork


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate trained DQN for Fruit Cutter")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--grid-size", type=int, default=10)
    parser.add_argument("--state-grid-size", type=int, default=0)
    parser.add_argument("--spawn-prob", type=float, default=0.45)
    parser.add_argument("--bomb-prob", type=float, default=0.18)
    args = parser.parse_args()

    state_grid_size = args.state_grid_size if args.state_grid_size > 0 else args.grid_size
    env = FruitCutterEnv(
        config=FruitCutterConfig(grid_size=args.grid_size, spawn_prob=args.spawn_prob, bomb_prob=args.bomb_prob),
        seed=args.seed,
        state_grid_size=state_grid_size,
    )
    network = MLPQNetwork.load(args.model)
    stats = evaluate_fruit_policy(env, network, episodes=args.episodes, seed_start=args.seed)

    print("Fruit Cutter Evaluation Results")
    print("-------------------------------")
    print(f"Episodes: {args.episodes}")
    print(f"Average score: {stats.avg_score:.3f}")
    print(f"Median score: {stats.median_score:.3f}")
    print(f"Average steps: {stats.avg_steps:.2f}")
    print(f"Average sliced/episode: {stats.avg_sliced_per_episode:.2f}")
    print(f"Average missed/episode: {stats.avg_missed_per_episode:.2f}")


if __name__ == "__main__":
    main()
