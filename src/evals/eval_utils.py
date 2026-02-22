from __future__ import annotations

from dataclasses import dataclass
from statistics import median
from typing import Dict

from ..games.env2048 import Game2048Env
from ..network import MLPQNetwork


@dataclass
class EvalStats:
    avg_score: float
    median_score: float
    avg_steps: float
    tile_distribution: Dict[int, int]
    reach_512: float
    reach_1024: float
    reach_2048: float


def argmax(values: list[float]) -> int:
    return max(range(len(values)), key=values.__getitem__)


def masked_argmax(values: list[float], legal_actions: list[int]) -> int:
    if not legal_actions:
        return argmax(values)
    return max(legal_actions, key=lambda action: values[action])


def evaluate_policy(
    env: Game2048Env,
    network: MLPQNetwork,
    episodes: int,
    seed_start: int = 10000,
    max_steps: int = 2000,
) -> EvalStats:
    scores = []
    max_tiles = []
    steps_per_episode = []

    for ep in range(episodes):
        state = env.reset(seed=seed_start + ep)
        done = False
        steps = 0
        info = {"score": 0, "max_tile": 0}

        while not done and steps < max_steps:
            q_values = network.predict_one(state)
            action = masked_argmax(q_values, env.game.legal_actions())
            state, _, done, info = env.step(action)
            steps += 1

        scores.append(float(info["score"]))
        max_tiles.append(int(info["max_tile"]))
        steps_per_episode.append(float(steps))

    tile_distribution: Dict[int, int] = {}
    for tile in max_tiles:
        tile_distribution[tile] = tile_distribution.get(tile, 0) + 1

    reach_512_count = sum(1 for tile in max_tiles if tile >= 512)
    reach_1024_count = sum(1 for tile in max_tiles if tile >= 1024)
    reach_2048_count = sum(1 for tile in max_tiles if tile >= 2048)

    n = max(len(scores), 1)

    return EvalStats(
        avg_score=float(sum(scores) / n),
        median_score=float(median(scores) if scores else 0.0),
        avg_steps=float(sum(steps_per_episode) / n),
        tile_distribution=tile_distribution,
        reach_512=float(reach_512_count / n),
        reach_1024=float(reach_1024_count / n),
        reach_2048=float(reach_2048_count / n),
    )
