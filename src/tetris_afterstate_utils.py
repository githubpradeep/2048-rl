from __future__ import annotations

from dataclasses import dataclass
from statistics import median

import numpy as np

from .games.tetris import TetrisPlacementEnv
from .network import MLPQNetwork


@dataclass
class TetrisAfterstateEvalStats:
    avg_score: float
    median_score: float
    avg_lines: float
    avg_steps: float


def pick_afterstate_action(env: TetrisPlacementEnv, network: MLPQNetwork, rng: np.random.Generator | None = None) -> int:
    include_max_height = network.input_dim >= 5
    candidates = env.legal_afterstates(include_max_height=include_max_height)
    if not candidates:
        legal = env.legal_actions()
        if legal:
            if rng is None:
                return int(legal[0])
            return int(rng.choice(legal))
        return 0

    feats = np.stack([feat for _, feat in candidates], axis=0).astype(np.float32)
    values = network.predict_batch(feats)
    best_idx = int(np.argmax([v[0] for v in values]))
    return int(candidates[best_idx][0])


def evaluate_afterstate_policy(
    env: TetrisPlacementEnv,
    network: MLPQNetwork,
    episodes: int,
    seed_start: int = 50000,
    max_steps: int = 2000,
) -> TetrisAfterstateEvalStats:
    scores: list[float] = []
    lines_arr: list[float] = []
    steps_arr: list[float] = []

    for ep in range(episodes):
        env.reset(seed=seed_start + ep)
        done = False
        steps = 0
        info = {"score": 0, "lines": 0}

        while not done and steps < max_steps:
            action = pick_afterstate_action(env, network)
            _, _, done, info = env.step(action)
            steps += 1

        scores.append(float(info["score"]))
        lines_arr.append(float(info["lines"]))
        steps_arr.append(float(steps))

    n = max(len(scores), 1)
    return TetrisAfterstateEvalStats(
        avg_score=float(sum(scores) / n),
        median_score=float(median(scores) if scores else 0.0),
        avg_lines=float(sum(lines_arr) / n),
        avg_steps=float(sum(steps_arr) / n),
    )
