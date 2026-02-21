from __future__ import annotations

import argparse

from .games.shooter import ShooterConfig, ShooterEnv
from .network import MLPQNetwork
from .shooter_eval_utils import evaluate_shooter_policy


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate trained DQN for Shooter")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--grid-size", type=int, default=10)
    parser.add_argument("--state-grid-size", type=int, default=0)
    parser.add_argument("--spawn-prob", type=float, default=0.35)
    parser.add_argument("--max-steps", type=int, default=1000)
    args = parser.parse_args()

    state_grid_size = args.state_grid_size if args.state_grid_size > 0 else args.grid_size
    env = ShooterEnv(
        config=ShooterConfig(grid_size=args.grid_size, spawn_prob=args.spawn_prob, max_steps=args.max_steps),
        seed=args.seed,
        state_grid_size=state_grid_size,
    )
    network = MLPQNetwork.load(args.model)
    stats = evaluate_shooter_policy(env, network, episodes=args.episodes, seed_start=args.seed)

    print("Shooter Evaluation Results")
    print("--------------------------")
    print(f"Episodes: {args.episodes}")
    print(f"Average score: {stats.avg_score:.3f}")
    print(f"Median score: {stats.median_score:.3f}")
    print(f"Average steps: {stats.avg_steps:.2f}")
    print(f"Average lives left: {stats.avg_lives_left:.2f}")
    print(f"Average kills/episode: {stats.avg_kills_per_episode:.2f}")
    print(f"Average escaped/episode: {stats.avg_escaped_per_episode:.2f}")


if __name__ == "__main__":
    main()
