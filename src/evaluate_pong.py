from __future__ import annotations

import argparse

from .games.pong import PongConfig, PongEnv
from .network import MLPQNetwork
from .pong_eval_utils import evaluate_pong_policy


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate trained DQN for Pong")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--width", type=int, default=12)
    parser.add_argument("--height", type=int, default=16)
    parser.add_argument("--paddle-height", type=int, default=4)
    parser.add_argument("--paddle-speed", type=int, default=1)
    parser.add_argument("--opponent-track-prob", type=float, default=0.85)
    parser.add_argument("--start-lives", type=int, default=3)
    parser.add_argument("--step-reward", type=float, default=0.01)
    parser.add_argument("--paddle-hit-reward", type=float, default=0.05)
    parser.add_argument("--score-reward", type=float, default=2.0)
    parser.add_argument("--concede-penalty", type=float, default=-2.0)
    parser.add_argument("--max-steps", type=int, default=1000)
    args = parser.parse_args()

    env = PongEnv(
        config=PongConfig(
            width=args.width,
            height=args.height,
            paddle_height=args.paddle_height,
            paddle_speed=args.paddle_speed,
            opponent_track_prob=args.opponent_track_prob,
            start_lives=args.start_lives,
            step_reward=args.step_reward,
            paddle_hit_reward=args.paddle_hit_reward,
            score_reward=args.score_reward,
            concede_penalty=args.concede_penalty,
            max_steps=args.max_steps,
        ),
        seed=args.seed,
    )
    network = MLPQNetwork.load(args.model)
    stats = evaluate_pong_policy(env, network, episodes=args.episodes, seed_start=args.seed, max_steps=args.max_steps)

    print("Pong Evaluation Results")
    print("-----------------------")
    print(f"Episodes: {args.episodes}")
    print(f"Average score: {stats.avg_score:.3f}")
    print(f"Median score: {stats.median_score:.3f}")
    print(f"Average steps: {stats.avg_steps:.2f}")
    print(f"Average lives left: {stats.avg_lives_left:.2f}")
    print(f"Average opponent score: {stats.avg_opponent_score:.2f}")
    print(f"Average player hits: {stats.avg_player_hits:.2f}")


if __name__ == "__main__":
    main()

