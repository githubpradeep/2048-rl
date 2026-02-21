from __future__ import annotations

from dataclasses import dataclass
from statistics import median
from typing import Protocol

from .network import MLPQNetwork


class TetrisLikeEnv(Protocol):
    def reset(self, seed: int | None = None): ...
    def legal_actions(self) -> list[int]: ...
    def step(self, action: int): ...


@dataclass
class TetrisEvalStats:
    avg_score: float
    median_score: float
    avg_lines: float
    avg_steps: float


def masked_argmax(values: list[float], legal_actions: list[int]) -> int:
    if not legal_actions:
        return max(range(len(values)), key=values.__getitem__)
    return max(legal_actions, key=lambda action: values[action])


def evaluate_tetris_policy(
    env: TetrisLikeEnv,
    network: MLPQNetwork,
    episodes: int,
    seed_start: int = 50000,
    max_steps: int = 2000,
) -> TetrisEvalStats:
    scores = []
    lines_arr = []
    steps_arr = []

    for ep in range(episodes):
        state = env.reset(seed=seed_start + ep)
        done = False
        steps = 0
        info = {"score": 0, "lines": 0}

        while not done and steps < max_steps:
            q_values = network.predict_one(state)
            action = masked_argmax(q_values, env.legal_actions())
            state, _, done, info = env.step(action)
            steps += 1

        scores.append(float(info["score"]))
        lines_arr.append(float(info["lines"]))
        steps_arr.append(float(steps))

    n = max(len(scores), 1)
    return TetrisEvalStats(
        avg_score=float(sum(scores) / n),
        median_score=float(median(scores) if scores else 0.0),
        avg_lines=float(sum(lines_arr) / n),
        avg_steps=float(sum(steps_arr) / n),
    )
