from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np


DiscreteState = Tuple[int, int, int]


@dataclass(frozen=True)
class FlappyDiscretizerConfig:
    dx_bins: int = 24
    dy_bins: int = 31
    vel_bins: int = 15
    dx_min: float = -0.25
    dx_max: float = 1.25
    dy_min: float = -1.0
    dy_max: float = 1.0
    vel_min: float = -1.0
    vel_max: float = 1.0


class FlappyStateDiscretizer:
    """Discretize low-dimensional Flappy state into a compact tabular key.

    Expected state layout (from FlappyEnv):
    [bird_y, bird_vel, p1_dx, p1_gap_y, bird_minus_p1_gap, p2_dx, p2_gap_y, bird_minus_p2_gap]
    """

    def __init__(self, config: FlappyDiscretizerConfig | None = None) -> None:
        self.config = config or FlappyDiscretizerConfig()

    @staticmethod
    def _bin_index(value: float, vmin: float, vmax: float, nbins: int) -> int:
        if nbins <= 1:
            return 0
        if value <= vmin:
            return 0
        if value >= vmax:
            return nbins - 1
        frac = (value - vmin) / (vmax - vmin)
        idx = int(frac * nbins)
        return min(max(idx, 0), nbins - 1)

    def encode(self, state: np.ndarray | Iterable[float]) -> DiscreteState:
        s = np.asarray(state, dtype=np.float32)
        if s.shape[0] < 5:
            raise ValueError(f"Expected flappy state with at least 5 features, got shape {s.shape}")
        cfg = self.config
        dx = self._bin_index(float(s[2]), cfg.dx_min, cfg.dx_max, cfg.dx_bins)
        dy = self._bin_index(float(s[4]), cfg.dy_min, cfg.dy_max, cfg.dy_bins)
        vel = self._bin_index(float(s[1]), cfg.vel_min, cfg.vel_max, cfg.vel_bins)
        return (dx, dy, vel)

    def to_dict(self) -> dict:
        cfg = self.config
        return {
            "dx_bins": cfg.dx_bins,
            "dy_bins": cfg.dy_bins,
            "vel_bins": cfg.vel_bins,
            "dx_min": cfg.dx_min,
            "dx_max": cfg.dx_max,
            "dy_min": cfg.dy_min,
            "dy_max": cfg.dy_max,
            "vel_min": cfg.vel_min,
            "vel_max": cfg.vel_max,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "FlappyStateDiscretizer":
        cfg = FlappyDiscretizerConfig(
            dx_bins=int(payload["dx_bins"]),
            dy_bins=int(payload["dy_bins"]),
            vel_bins=int(payload["vel_bins"]),
            dx_min=float(payload["dx_min"]),
            dx_max=float(payload["dx_max"]),
            dy_min=float(payload["dy_min"]),
            dy_max=float(payload["dy_max"]),
            vel_min=float(payload["vel_min"]),
            vel_max=float(payload["vel_max"]),
        )
        return cls(config=cfg)


class FlappyTabularQAgent:
    """Tabular Q-learning agent over discretized Flappy state."""

    MODEL_KIND = "flappy_tabular_q_v1"

    def __init__(
        self,
        action_size: int = 2,
        discretizer: FlappyStateDiscretizer | None = None,
        alpha: float = 0.1,
        gamma: float = 0.99,
        optimistic_init: float = 0.0,
    ) -> None:
        if action_size <= 0:
            raise ValueError("action_size must be positive")
        self.action_size = int(action_size)
        self.discretizer = discretizer or FlappyStateDiscretizer()
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.optimistic_init = float(optimistic_init)
        self.q_table: Dict[DiscreteState, np.ndarray] = {}

    def _ensure(self, key: DiscreteState) -> np.ndarray:
        values = self.q_table.get(key)
        if values is None:
            values = np.full((self.action_size,), self.optimistic_init, dtype=np.float32)
            self.q_table[key] = values
        return values

    def q_values(self, key: DiscreteState) -> np.ndarray:
        return self._ensure(key)

    def greedy_action_for_key(self, key: DiscreteState, legal_actions: list[int]) -> int:
        values = self._ensure(key)
        if not legal_actions:
            return int(np.argmax(values))
        return max(legal_actions, key=lambda action: float(values[action]))

    def select_action(
        self,
        state: np.ndarray,
        legal_actions: list[int],
        epsilon: float,
        rng: np.random.Generator,
    ) -> tuple[int, DiscreteState]:
        key = self.discretizer.encode(state)
        if legal_actions and rng.random() < float(epsilon):
            action = int(rng.choice(np.asarray(legal_actions, dtype=np.int64)))
            return action, key
        return self.greedy_action_for_key(key, legal_actions), key

    def predict_one(self, state: np.ndarray) -> list[float]:
        key = self.discretizer.encode(state)
        return [float(v) for v in self._ensure(key).tolist()]

    def update_q_learning(
        self,
        state_key: DiscreteState,
        action: int,
        reward: float,
        next_state_key: DiscreteState,
        done: bool,
        next_legal_actions: list[int] | None = None,
    ) -> float:
        q = self._ensure(state_key)
        next_q = self._ensure(next_state_key)

        if done:
            target = float(reward)
        else:
            if next_legal_actions:
                next_best = max(float(next_q[a]) for a in next_legal_actions)
            else:
                next_best = float(np.max(next_q))
            target = float(reward) + self.gamma * next_best

        td_error = target - float(q[action])
        q[action] = float(q[action]) + self.alpha * td_error
        return float(abs(td_error))

    @staticmethod
    def _encode_key(key: DiscreteState) -> str:
        return f"{key[0]},{key[1]},{key[2]}"

    @staticmethod
    def _decode_key(raw: str) -> DiscreteState:
        a, b, c = raw.split(",")
        return (int(a), int(b), int(c))

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "kind": self.MODEL_KIND,
            "action_size": self.action_size,
            "alpha": self.alpha,
            "gamma": self.gamma,
            "optimistic_init": self.optimistic_init,
            "discretizer": self.discretizer.to_dict(),
            "q_table": {
                self._encode_key(key): [float(v) for v in values.tolist()]
                for key, values in self.q_table.items()
            },
        }
        path.write_text(json.dumps(payload), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "FlappyTabularQAgent":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get("kind") != cls.MODEL_KIND:
            raise ValueError(f"Unsupported model kind: {payload.get('kind')}")

        agent = cls(
            action_size=int(payload["action_size"]),
            discretizer=FlappyStateDiscretizer.from_dict(payload["discretizer"]),
            alpha=float(payload.get("alpha", 0.1)),
            gamma=float(payload.get("gamma", 0.99)),
            optimistic_init=float(payload.get("optimistic_init", 0.0)),
        )
        raw_table = payload.get("q_table", {})
        for raw_key, raw_values in raw_table.items():
            key = cls._decode_key(raw_key)
            values = np.asarray(raw_values, dtype=np.float32)
            if values.shape != (agent.action_size,):
                raise ValueError(f"Invalid q-values shape for key {raw_key}: {values.shape}")
            agent.q_table[key] = values
        return agent

