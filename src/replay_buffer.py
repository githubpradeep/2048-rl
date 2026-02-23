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
    indices: np.ndarray | None = None
    weights: np.ndarray | None = None


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
            indices=idx.astype(np.int64),
            weights=None,
        )

    def __len__(self) -> int:
        return self.size


class PrioritizedReplayBuffer(ReplayBuffer):
    def __init__(self, capacity: int, state_dim: int, action_dim: int | None = None, alpha: float = 0.6) -> None:
        super().__init__(capacity=capacity, state_dim=state_dim, action_dim=action_dim)
        if alpha < 0.0:
            raise ValueError("alpha must be >= 0")
        self.alpha = float(alpha)
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.max_priority = 1.0

    def add(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        next_legal_actions: list[int] | np.ndarray | None = None,
    ) -> None:
        insert_idx = self.ptr
        super().add(state, action, reward, next_state, done, next_legal_actions=next_legal_actions)
        self.priorities[insert_idx] = float(self.max_priority)

    def sample(self, batch_size: int, rng: np.random.Generator, beta: float = 0.4) -> Batch:
        if batch_size > self.size:
            raise ValueError("batch_size cannot be larger than current buffer size")
        beta = float(beta)
        pri = self.priorities[: self.size]
        if self.alpha == 0.0:
            probs = np.full((self.size,), 1.0 / self.size, dtype=np.float64)
        else:
            scaled = np.power(np.maximum(pri, 1e-8), self.alpha, dtype=np.float64)
            total = float(np.sum(scaled))
            if total <= 0.0 or not np.isfinite(total):
                probs = np.full((self.size,), 1.0 / self.size, dtype=np.float64)
            else:
                probs = scaled / total
        idx = rng.choice(self.size, size=batch_size, replace=True, p=probs).astype(np.int64)
        samples = probs[idx]
        weights = np.power(self.size * np.maximum(samples, 1e-12), -beta, dtype=np.float64)
        max_w = float(np.max(weights)) if weights.size else 1.0
        if max_w > 0:
            weights = weights / max_w
        return Batch(
            states=self.states[idx],
            actions=self.actions[idx],
            rewards=self.rewards[idx],
            next_states=self.next_states[idx],
            dones=self.dones[idx],
            next_legal_masks=self.next_legal_masks[idx] if self.next_legal_masks is not None else None,
            indices=idx,
            weights=np.asarray(weights, dtype=np.float32),
        )

    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray, eps: float = 1e-6) -> None:
        if indices is None:
            return
        idx = np.asarray(indices, dtype=np.int64)
        pri = np.asarray(priorities, dtype=np.float32)
        if idx.shape[0] != pri.shape[0]:
            raise ValueError("indices and priorities must have same length")
        pri = np.maximum(pri, float(eps))
        self.priorities[idx] = pri
        max_pri = float(np.max(pri)) if pri.size else 0.0
        if max_pri > self.max_priority:
            self.max_priority = max_pri
