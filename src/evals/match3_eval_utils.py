from __future__ import annotations

from dataclasses import dataclass
from statistics import median

from ..games.match3 import Match3Env
from ..network import MLPQNetwork


@dataclass
class Match3EvalStats:
    avg_score: float
    median_score: float
    avg_steps: float
    avg_tiles_cleared: float
    avg_cascades: float
    invalid_rate: float


def masked_argmax(values: list[float], legal_actions: list[int]) -> int:
    if not legal_actions:
        return max(range(len(values)), key=values.__getitem__)
    return max(legal_actions, key=lambda action: values[action])


def evaluate_match3_policy(
    env: Match3Env,
    network: MLPQNetwork,
    episodes: int,
    seed_start: int = 40000,
    max_steps: int = 1000,
) -> Match3EvalStats:
    scores = []
    steps_arr = []
    tiles_arr = []
    cascades_arr = []
    invalid_arr = []

    for ep in range(episodes):
        state = env.reset(seed=seed_start + ep)
        done = False
        steps = 0
        info = {"score": 0, "total_tiles_cleared": 0, "total_cascades": 0, "invalid": False}
        invalid_steps = 0

        while not done and steps < max_steps:
            q_values = network.predict_one(state)
            action = masked_argmax(q_values, env.legal_actions())
            state, _, done, info = env.step(action)
            steps += 1
            invalid_steps += 1 if bool(info["invalid"]) else 0

        scores.append(float(info["score"]))
        steps_arr.append(float(steps))
        tiles_arr.append(float(info["total_tiles_cleared"]))
        cascades_arr.append(float(info["total_cascades"]))
        invalid_arr.append(float(invalid_steps) / float(max(steps, 1)))

    n = max(len(scores), 1)
    return Match3EvalStats(
        avg_score=float(sum(scores) / n),
        median_score=float(median(scores) if scores else 0.0),
        avg_steps=float(sum(steps_arr) / n),
        avg_tiles_cleared=float(sum(tiles_arr) / n),
        avg_cascades=float(sum(cascades_arr) / n),
        invalid_rate=float(sum(invalid_arr) / n),
    )

