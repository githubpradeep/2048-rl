from __future__ import annotations

from dataclasses import dataclass
from statistics import median

from typing import Protocol

from ..games.craftax_classic import CraftaxClassicEnv


class _CraftaxPolicy(Protocol):
    def predict_one(self, state: object) -> list[float]: ...


@dataclass
class CraftaxEvalStats:
    avg_score: float
    median_score: float
    avg_steps: float
    avg_health: float
    avg_reward: float


def masked_argmax(values: list[float], legal_actions: list[int]) -> int:
    if not legal_actions:
        return max(range(len(values)), key=values.__getitem__)
    return max(legal_actions, key=lambda action: values[action])


def evaluate_craftax_policy(
    env: CraftaxClassicEnv,
    network: _CraftaxPolicy,
    episodes: int,
    seed_start: int = 50000,
    max_steps: int = 10000,
) -> CraftaxEvalStats:
    scores: list[float] = []
    steps_arr: list[float] = []
    health_arr: list[float] = []
    rewards: list[float] = []

    for ep in range(episodes):
        state = env.reset(seed=seed_start + ep)
        done = False
        steps = 0
        ep_reward = 0.0
        info = {"score": 0, "health": 0}

        while not done and steps < max_steps:
            q_values = network.predict_one(state)
            action = masked_argmax(q_values, env.legal_actions())
            state, reward, done, info = env.step(action)
            ep_reward += float(reward)
            steps += 1

        scores.append(float(info["score"]))
        steps_arr.append(float(steps))
        health_arr.append(float(info.get("health", 0)))
        rewards.append(ep_reward)

    n = max(len(scores), 1)
    return CraftaxEvalStats(
        avg_score=float(sum(scores) / n),
        median_score=float(median(scores) if scores else 0.0),
        avg_steps=float(sum(steps_arr) / n),
        avg_health=float(sum(health_arr) / n),
        avg_reward=float(sum(rewards) / n),
    )
