from __future__ import annotations

from dataclasses import dataclass
from statistics import median

from ..games.shooter import ShooterEnv
from ..network import MLPQNetwork


@dataclass
class ShooterEvalStats:
    avg_score: float
    median_score: float
    avg_steps: float
    avg_lives_left: float
    avg_kills_per_episode: float
    avg_escaped_per_episode: float


def masked_argmax(values: list[float], legal_actions: list[int]) -> int:
    if not legal_actions:
        return max(range(len(values)), key=values.__getitem__)
    return max(legal_actions, key=lambda action: values[action])


def evaluate_shooter_policy(
    env: ShooterEnv,
    network: MLPQNetwork,
    episodes: int,
    seed_start: int = 40000,
    max_steps: int = 2000,
) -> ShooterEvalStats:
    scores = []
    steps_arr = []
    lives_arr = []
    kills_arr = []
    escaped_arr = []

    for ep in range(episodes):
        state = env.reset(seed=seed_start + ep)
        done = False
        steps = 0
        info = {"score": 0, "lives": 0, "total_kills": 0, "total_escaped": 0}

        while not done and steps < max_steps:
            q_values = network.predict_one(state)
            action = masked_argmax(q_values, env.legal_actions())
            state, _, done, info = env.step(action)
            steps += 1

        scores.append(float(info["score"]))
        steps_arr.append(float(steps))
        lives_arr.append(float(info["lives"]))
        kills_arr.append(float(info["total_kills"]))
        escaped_arr.append(float(info["total_escaped"]))

    n = max(len(scores), 1)
    return ShooterEvalStats(
        avg_score=float(sum(scores) / n),
        median_score=float(median(scores) if scores else 0.0),
        avg_steps=float(sum(steps_arr) / n),
        avg_lives_left=float(sum(lives_arr) / n),
        avg_kills_per_episode=float(sum(kills_arr) / n),
        avg_escaped_per_episode=float(sum(escaped_arr) / n),
    )
