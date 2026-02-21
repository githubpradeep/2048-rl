from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np

from .env import Game2048Env
from .network import MLPQNetwork


@dataclass
class EvalStats:
    avg_score: float
    median_score: float
    avg_steps: float
    tile_distribution: Dict[int, int]
    reach_512: float
    reach_1024: float
    reach_2048: float


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
        while not done and steps < max_steps:
            q_values = network.predict(state[None, :])[0]
            action = int(np.argmax(q_values))
            state, _, done, info = env.step(action)
            steps += 1

        scores.append(float(info["score"]))
        max_tiles.append(int(info["max_tile"]))
        steps_per_episode.append(float(steps))

    tile_distribution: Dict[int, int] = {}
    for tile in max_tiles:
        tile_distribution[tile] = tile_distribution.get(tile, 0) + 1

    max_tiles_arr = np.array(max_tiles, dtype=np.int32)
    reach_512 = float(np.mean(max_tiles_arr >= 512))
    reach_1024 = float(np.mean(max_tiles_arr >= 1024))
    reach_2048 = float(np.mean(max_tiles_arr >= 2048))

    return EvalStats(
        avg_score=float(np.mean(scores)),
        median_score=float(np.median(scores)),
        avg_steps=float(np.mean(steps_per_episode)),
        tile_distribution=tile_distribution,
        reach_512=reach_512,
        reach_1024=reach_1024,
        reach_2048=reach_2048,
    )
