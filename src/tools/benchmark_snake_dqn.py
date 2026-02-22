from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Dict

from ..games.snake import SnakeConfig, SnakeEnv
from ..network import MLPQNetwork
from ..evals.snake_eval_utils import evaluate_snake_policy


def run_command(cmd: list[str]) -> None:
    print("$", " ".join(cmd))
    subprocess.run(cmd, check=True)


def evaluate_model(model_path: Path, grid_size: int, state_grid_size: int, episodes: int, seed: int) -> Dict[str, float]:
    env = SnakeEnv(
        config=SnakeConfig(grid_size=grid_size),
        seed=seed,
        state_grid_size=state_grid_size,
    )
    net = MLPQNetwork.load(model_path)
    stats = evaluate_snake_policy(env, net, episodes=episodes, seed_start=seed + 10000)
    return {
        "avg_score": stats.avg_score,
        "median_score": stats.median_score,
        "avg_steps": stats.avg_steps,
        "avg_length": stats.avg_length,
        "food_rate": stats.food_rate,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark Snake DQN baseline vs Double-DQN+Dueling on same seeds")
    parser.add_argument("--episodes", type=int, default=2000)
    parser.add_argument("--eval-episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--grid-size", type=int, default=10)
    parser.add_argument("--state-grid-size", type=int, default=10)
    parser.add_argument("--curriculum-grid-sizes", type=str, default="8,10")
    parser.add_argument("--distance-reward-scale", type=float, default=0.2)
    parser.add_argument("--save-root", type=str, default="models/snake_benchmark")
    parser.add_argument("--max-steps", type=int, default=1200)
    parser.add_argument("--warmup-steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--buffer-size", type=int, default=80_000)
    parser.add_argument("--hidden-sizes", type=str, default="256,256")
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--target-update", type=int, default=500)
    parser.add_argument("--train-every", type=int, default=2)
    parser.add_argument("--eps-start", type=float, default=1.0)
    parser.add_argument("--eps-end", type=float, default=0.02)
    parser.add_argument("--eps-decay-steps", type=int, default=30_000)
    args = parser.parse_args()

    save_root = Path(args.save_root)
    baseline_dir = save_root / "baseline_dqn"
    advanced_dir = save_root / "double_dueling"
    baseline_dir.mkdir(parents=True, exist_ok=True)
    advanced_dir.mkdir(parents=True, exist_ok=True)

    base_train_cmd = [
        sys.executable,
        "-m",
        "src.train_snake_dqn",
        "--episodes",
        str(args.episodes),
        "--eval-every",
        str(max(1, min(50, args.episodes))),
        "--eval-episodes",
        str(min(50, args.eval_episodes)),
        "--seed",
        str(args.seed),
        "--grid-size",
        str(args.grid_size),
        "--state-grid-size",
        str(args.state_grid_size),
        "--curriculum-grid-sizes",
        args.curriculum_grid_sizes,
        "--distance-reward-scale",
        str(args.distance_reward_scale),
        "--max-steps",
        str(args.max_steps),
        "--warmup-steps",
        str(args.warmup_steps),
        "--batch-size",
        str(args.batch_size),
        "--buffer-size",
        str(args.buffer_size),
        "--hidden-sizes",
        args.hidden_sizes,
        "--lr",
        str(args.lr),
        "--target-update",
        str(args.target_update),
        "--train-every",
        str(args.train_every),
        "--eps-start",
        str(args.eps_start),
        "--eps-end",
        str(args.eps_end),
        "--eps-decay-steps",
        str(args.eps_decay_steps),
    ]

    print("\n=== Training baseline DQN ===")
    run_command(base_train_cmd + ["--save-dir", str(baseline_dir)])

    print("\n=== Training Double DQN + Dueling ===")
    run_command(base_train_cmd + ["--double-dqn", "--dueling", "--save-dir", str(advanced_dir)])

    baseline_best = baseline_dir / "snake_dqn_best.json"
    advanced_best = advanced_dir / "snake_dqn_best.json"
    baseline_final = baseline_dir / "snake_dqn_final.json"
    advanced_final = advanced_dir / "snake_dqn_final.json"
    baseline_model = baseline_best if baseline_best.exists() else baseline_final
    advanced_model = advanced_best if advanced_best.exists() else advanced_final

    baseline_stats = evaluate_model(
        baseline_model,
        grid_size=args.grid_size,
        state_grid_size=args.state_grid_size,
        episodes=args.eval_episodes,
        seed=args.seed,
    )
    advanced_stats = evaluate_model(
        advanced_model,
        grid_size=args.grid_size,
        state_grid_size=args.state_grid_size,
        episodes=args.eval_episodes,
        seed=args.seed,
    )

    print("\n=== Benchmark Summary ===")
    print(f"Baseline model: {baseline_model}")
    print(f"Advanced model: {advanced_model}")
    print("Metric               Baseline DQN    Double+Dueling")
    print("---------------------------------------------------")
    for key in ["avg_score", "median_score", "avg_steps", "avg_length", "food_rate"]:
        print(f"{key:18s} {baseline_stats[key]:12.4f} {advanced_stats[key]:16.4f}")


if __name__ == "__main__":
    main()
