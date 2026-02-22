from __future__ import annotations

from dataclasses import dataclass
from statistics import median

from ..games.flappy import FlappyEnv
from ..network import MLPQNetwork


@dataclass
class FlappyEvalStats:
    avg_score: float
    median_score: float
    avg_steps: float
    avg_reward: float


def masked_argmax(values: list[float], legal_actions: list[int]) -> int:
    if not legal_actions:
        return max(range(len(values)), key=values.__getitem__)
    return max(legal_actions, key=lambda action: values[action])


def evaluate_flappy_policy(
    env: FlappyEnv,
    network: MLPQNetwork,
    episodes: int,
    seed_start: int = 40000,
    max_steps: int = 2000,
) -> FlappyEvalStats:
    scores = []
    steps_arr = []
    rewards = []

    for ep in range(episodes):
        state = env.reset(seed=seed_start + ep)
        done = False
        steps = 0
        info = {"score": 0}
        total_reward = 0.0

        while not done and steps < max_steps:
            q_values = network.predict_one(state)
            action = masked_argmax(q_values, env.legal_actions())
            state, reward, done, info = env.step(action)
            total_reward += float(reward)
            steps += 1

        scores.append(float(info["score"]))
        steps_arr.append(float(steps))
        rewards.append(float(total_reward))

    n = max(len(scores), 1)
    return FlappyEvalStats(
        avg_score=float(sum(scores) / n),
        median_score=float(median(scores) if scores else 0.0),
        avg_steps=float(sum(steps_arr) / n),
        avg_reward=float(sum(rewards) / n),
    )
