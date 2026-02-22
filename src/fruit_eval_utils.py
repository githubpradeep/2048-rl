from __future__ import annotations

from dataclasses import dataclass
from statistics import median

from .games.fruit_cutter import FruitCutterEnv
from .network import MLPQNetwork


@dataclass
class FruitEvalStats:
    avg_score: float
    median_score: float
    avg_steps: float
    avg_sliced_per_episode: float
    avg_missed_per_episode: float


def masked_argmax(values: list[float], legal_actions: list[int]) -> int:
    if not legal_actions:
        return max(range(len(values)), key=values.__getitem__)
    return max(legal_actions, key=lambda action: values[action])


def evaluate_fruit_policy(
    env: FruitCutterEnv,
    network: MLPQNetwork,
    episodes: int,
    seed_start: int = 30000,
    max_steps: int = 2000,
) -> FruitEvalStats:
    scores = []
    steps_arr = []
    sliced_arr = []
    missed_arr = []

    for ep in range(episodes):
        state = env.reset(seed=seed_start + ep)
        done = False
        steps = 0
        info = {"score": 0, "sliced": 0, "missed": 0}
        total_sliced = 0
        total_missed = 0

        while not done and steps < max_steps:
            q_values = network.predict_one(state)
            action = masked_argmax(q_values, env.legal_actions())
            state, _, done, info = env.step(action)
            steps += 1
            total_sliced += int(info["sliced"])
            total_missed += int(info["missed"])

        scores.append(float(info["score"]))
        steps_arr.append(float(steps))
        sliced_arr.append(float(total_sliced))
        missed_arr.append(float(total_missed))

    n = max(len(scores), 1)
    return FruitEvalStats(
        avg_score=float(sum(scores) / n),
        median_score=float(median(scores) if scores else 0.0),
        avg_steps=float(sum(steps_arr) / n),
        avg_sliced_per_episode=float(sum(sliced_arr) / n),
        avg_missed_per_episode=float(sum(missed_arr) / n),
    )
