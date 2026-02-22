from __future__ import annotations

from dataclasses import dataclass
from statistics import median

from ..games.breakout import BreakoutEnv
from ..network import MLPQNetwork


@dataclass
class BreakoutEvalStats:
    avg_score: float
    median_score: float
    avg_steps: float
    avg_lives_left: float
    avg_bricks_broken: float
    clear_rate: float


def masked_argmax(values: list[float], legal_actions: list[int]) -> int:
    if not legal_actions:
        return max(range(len(values)), key=values.__getitem__)
    return max(legal_actions, key=lambda action: values[action])


def evaluate_breakout_policy(
    env: BreakoutEnv,
    network: MLPQNetwork,
    episodes: int,
    seed_start: int = 40000,
    max_steps: int = 2000,
) -> BreakoutEvalStats:
    scores = []
    steps_arr = []
    lives_arr = []
    bricks_arr = []
    clears = []

    for ep in range(episodes):
        state = env.reset(seed=seed_start + ep)
        done = False
        steps = 0
        info = {"score": 0, "lives": 0, "bricks_broken": 0, "cleared": False}

        while not done and steps < max_steps:
            q_values = network.predict_one(state)
            action = masked_argmax(q_values, env.legal_actions())
            state, _, done, info = env.step(action)
            steps += 1

        scores.append(float(info["score"]))
        steps_arr.append(float(steps))
        lives_arr.append(float(info["lives"]))
        bricks_arr.append(float(info["bricks_broken"]))
        clears.append(1.0 if bool(info["cleared"]) else 0.0)

    n = max(len(scores), 1)
    return BreakoutEvalStats(
        avg_score=float(sum(scores) / n),
        median_score=float(median(scores) if scores else 0.0),
        avg_steps=float(sum(steps_arr) / n),
        avg_lives_left=float(sum(lives_arr) / n),
        avg_bricks_broken=float(sum(bricks_arr) / n),
        clear_rate=float(sum(clears) / n),
    )

