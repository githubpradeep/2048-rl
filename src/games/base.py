from __future__ import annotations

from typing import Protocol


class GridGame(Protocol):
    """Minimal interface shared by game engines used by RL env wrappers."""

    def reset(self, seed: int | None = None): ...

    def step(self, action: int): ...

    def legal_actions(self) -> list[int]: ...

    def render(self) -> str: ...
