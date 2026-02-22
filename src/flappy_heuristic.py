from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass
class FlappyHeuristicPolicy:
    """Simple state-based heuristic that mimics a threshold controller.

    State layout (FlappyEnv):
    [bird_y, bird_vel, p1_dx, p1_gap_y, bird_minus_p1_gap, ...]
    """

    bird_y_threshold: float = 0.62
    dy_flap_threshold: float = 0.08
    vel_flap_threshold: float = 0.70
    bottom_emergency_y: float = 0.90

    def predict_one(self, state: Sequence[float]) -> list[float]:
        bird_y = float(state[0])
        bird_vel = float(state[1])
        bird_minus_gap = float(state[4])

        flap = False
        if bird_y >= self.bottom_emergency_y:
            flap = True
        elif bird_minus_gap > self.dy_flap_threshold:
            flap = True
        elif bird_vel > self.vel_flap_threshold:
            flap = True
        elif bird_y > self.bird_y_threshold:
            flap = True

        # Return Q-like scores so we can reuse existing eval/play code.
        return [0.0, 1.0] if flap else [1.0, 0.0]

