from __future__ import annotations

from dataclasses import dataclass
from statistics import median

from .games.pong import PongEnv
from .network import MLPQNetwork


@dataclass
class PongEvalStats:
    avg_score: float
    median_score: float
    avg_steps: float
    avg_lives_left: float
    avg_opponent_score: float
    avg_player_hits: float


def masked_argmax(values: list[float], legal_actions: list[int]) -> int:
    if not legal_actions:
        return max(range(len(values)), key=values.__getitem__)
    return max(legal_actions, key=lambda action: values[action])


def evaluate_pong_policy(
    env: PongEnv,
    network: MLPQNetwork,
    episodes: int,
    seed_start: int = 40000,
    max_steps: int = 2000,
) -> PongEvalStats:
    scores = []
    steps_arr = []
    lives_arr = []
    opp_scores = []
    player_hits_arr = []

    for ep in range(episodes):
        state = env.reset(seed=seed_start + ep)
        done = False
        steps = 0
        info = {"score": 0, "lives": 0, "opponent_score": 0, "player_hits": 0}

        while not done and steps < max_steps:
            q_values = network.predict_one(state)
            action = masked_argmax(q_values, env.legal_actions())
            state, _, done, info = env.step(action)
            steps += 1

        scores.append(float(info["score"]))
        steps_arr.append(float(steps))
        lives_arr.append(float(info["lives"]))
        opp_scores.append(float(info["opponent_score"]))
        player_hits_arr.append(float(info["player_hits"]))

    n = max(len(scores), 1)
    return PongEvalStats(
        avg_score=float(sum(scores) / n),
        median_score=float(median(scores) if scores else 0.0),
        avg_steps=float(sum(steps_arr) / n),
        avg_lives_left=float(sum(lives_arr) / n),
        avg_opponent_score=float(sum(opp_scores) / n),
        avg_player_hits=float(sum(player_hits_arr) / n),
    )

