from __future__ import annotations

import argparse

from .games.tetris import TetrisConfig, TetrisEnv, TetrisPlacementEnv
from .network import MLPQNetwork
from .tetris_eval_utils import evaluate_tetris_policy


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate trained DQN for Tetris")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--height", type=int, default=20)
    parser.add_argument("--width", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=1200)
    parser.add_argument("--placement-actions", action="store_true")
    args = parser.parse_args()

    if args.placement_actions:
        env = TetrisPlacementEnv(config=TetrisConfig(height=args.height, width=args.width, max_steps=args.max_steps), seed=args.seed)
    else:
        env = TetrisEnv(config=TetrisConfig(height=args.height, width=args.width, max_steps=args.max_steps), seed=args.seed)
    network = MLPQNetwork.load(args.model)
    stats = evaluate_tetris_policy(env, network, episodes=args.episodes, seed_start=args.seed, max_steps=args.max_steps)

    print("Tetris Evaluation Results")
    print("-------------------------")
    print(f"Episodes: {args.episodes}")
    print(f"Average score: {stats.avg_score:.3f}")
    print(f"Median score: {stats.median_score:.3f}")
    print(f"Average lines: {stats.avg_lines:.2f}")
    print(f"Average steps: {stats.avg_steps:.2f}")


if __name__ == "__main__":
    main()
