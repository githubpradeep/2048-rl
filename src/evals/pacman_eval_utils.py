from __future__ import annotations

from dataclasses import dataclass
from statistics import median

from ..games.pacman_lite import PacmanLiteEnv
from ..network import MLPQNetwork


@dataclass
class PacmanEvalStats:
    avg_score: float
    median_score: float
    avg_steps: float
    avg_lives_left: float
    avg_pellets_eaten: float
    avg_collisions: float
    clear_rate: float


def masked_argmax(values: list[float], legal_actions: list[int]) -> int:
    if not legal_actions:
        return max(range(len(values)), key=values.__getitem__)
    return max(legal_actions, key=lambda a: values[a])


def evaluate_pacman_policy(env: PacmanLiteEnv, network: MLPQNetwork, episodes: int, seed_start: int = 60000, max_steps: int = 2000) -> PacmanEvalStats:
    scores: list[float] = []
    steps_arr: list[float] = []
    lives_arr: list[float] = []
    pellets_arr: list[float] = []
    collisions_arr: list[float] = []
    clears = 0

    for ep in range(episodes):
        state = env.reset(seed=seed_start + ep)
        done = False
        steps = 0
        info = {"score": 0, "lives": 0, "total_pellets_eaten": 0, "total_collisions": 0, "cleared": False}
        while not done and steps < max_steps:
            q_values = network.predict_one(state)
            action = masked_argmax(q_values, env.legal_actions())
            state, _, done, info = env.step(action)
            steps += 1
        scores.append(float(info["score"]))
        steps_arr.append(float(steps))
        lives_arr.append(float(info["lives"]))
        pellets_arr.append(float(info["total_pellets_eaten"]))
        collisions_arr.append(float(info["total_collisions"]))
        clears += 1 if bool(info.get("cleared", False)) else 0

    n = max(len(scores), 1)
    return PacmanEvalStats(
        avg_score=float(sum(scores) / n),
        median_score=float(median(scores) if scores else 0.0),
        avg_steps=float(sum(steps_arr) / n),
        avg_lives_left=float(sum(lives_arr) / n),
        avg_pellets_eaten=float(sum(pellets_arr) / n),
        avg_collisions=float(sum(collisions_arr) / n),
        clear_rate=float(clears / n),
    )
