from __future__ import annotations

from dataclasses import dataclass
from statistics import median

from .games.snake import SnakeEnv, SnakeFeatureEnv
from .network import MLPQNetwork


@dataclass
class SnakeEvalStats:
    avg_score: float
    median_score: float
    avg_steps: float
    avg_length: float
    food_rate: float


def masked_argmax(values: list[float], legal_actions: list[int]) -> int:
    if not legal_actions:
        return max(range(len(values)), key=values.__getitem__)
    return max(legal_actions, key=lambda action: values[action])


def evaluate_snake_policy(
    env: SnakeEnv | SnakeFeatureEnv,
    network: MLPQNetwork,
    episodes: int,
    seed_start: int = 20000,
    max_steps: int = 2000,
) -> SnakeEvalStats:
    scores = []
    steps_arr = []
    lengths = []

    for ep in range(episodes):
        state = env.reset(seed=seed_start + ep)
        done = False
        steps = 0
        info = {"score": 0, "length": 3}

        while not done and steps < max_steps:
            q_values = network.predict_one(state)
            action = masked_argmax(q_values, env.legal_actions())
            state, _, done, info = env.step(action)
            steps += 1

        scores.append(float(info["score"]))
        lengths.append(float(info["length"]))
        steps_arr.append(float(steps))

    n = max(len(scores), 1)
    avg_score = float(sum(scores) / n)
    avg_steps = float(sum(steps_arr) / n)
    avg_length = float(sum(lengths) / n)

    return SnakeEvalStats(
        avg_score=avg_score,
        median_score=float(median(scores) if scores else 0.0),
        avg_steps=avg_steps,
        avg_length=avg_length,
        food_rate=float(avg_score / max(avg_steps, 1e-6)),
    )
