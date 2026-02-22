from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Batch:
    states: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    next_states: np.ndarray
    dones: np.ndarray
    next_legal_masks: np.ndarray | None = None


class ReplayBuffer:
    def __init__(self, capacity: int, state_dim: int, action_dim: int | None = None) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.capacity = capacity
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
        self.next_legal_masks = (
            np.zeros((capacity, action_dim), dtype=np.float32) if action_dim is not None else None
        )

        self.ptr = 0
        self.size = 0

    def add(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        next_legal_actions: list[int] | np.ndarray | None = None,
    ) -> None:
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_states[self.ptr] = next_state
        self.dones[self.ptr] = float(done)
        if self.next_legal_masks is not None:
            mask = self.next_legal_masks[self.ptr]
            mask.fill(0.0)
            if next_legal_actions is not None:
                legal = np.asarray(next_legal_actions, dtype=np.int64)
                valid = legal[(legal >= 0) & (legal < self.next_legal_masks.shape[1])]
                if valid.size > 0:
                    mask[valid] = 1.0

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, rng: np.random.Generator) -> Batch:
        if batch_size > self.size:
            raise ValueError("batch_size cannot be larger than current buffer size")
        idx = rng.integers(0, self.size, size=batch_size)
        return Batch(
            states=self.states[idx],
            actions=self.actions[idx],
            rewards=self.rewards[idx],
            next_states=self.next_states[idx],
            dones=self.dones[idx],
            next_legal_masks=self.next_legal_masks[idx] if self.next_legal_masks is not None else None,
        )

    def __len__(self) -> int:
        return self.size
