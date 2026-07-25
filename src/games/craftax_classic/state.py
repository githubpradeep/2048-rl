"""Mutable, NumPy-backed state containers for Craftax-Classic.

Ported from ``reference/craftax/craftax/craftax_classic/envs/craftax_state.py``.
The original uses immutable ``flax.struct`` dataclasses (functional-update
style, required for JAX tracing). Since this port runs eagerly with plain
Python control flow, we use ordinary *mutable* ``@dataclass`` containers with
NumPy array fields instead; game logic mutates these arrays/fields in place.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

from .constants import Action, BlockType


@dataclass
class Inventory:
    wood: int = 0
    stone: int = 0
    coal: int = 0
    iron: int = 0
    diamond: int = 0
    sapling: int = 0
    wood_pickaxe: int = 0
    stone_pickaxe: int = 0
    iron_pickaxe: int = 0
    wood_sword: int = 0
    stone_sword: int = 0
    iron_sword: int = 0


@dataclass
class Mobs:
    """A fixed-size population of homogeneous mobs (zombies, cows, etc.)."""

    position: np.ndarray  # (N, 2) int32
    health: np.ndarray  # (N,) int32
    mask: np.ndarray  # (N,) bool -- whether the slot holds a living mob
    attack_cooldown: np.ndarray  # (N,) int32

    @classmethod
    def empty(cls, n: int) -> "Mobs":
        return cls(
            position=np.zeros((n, 2), dtype=np.int32),
            health=np.zeros((n,), dtype=np.int32),
            mask=np.zeros((n,), dtype=bool),
            attack_cooldown=np.zeros((n,), dtype=np.int32),
        )

    def copy(self) -> "Mobs":
        return Mobs(
            position=self.position.copy(),
            health=self.health.copy(),
            mask=self.mask.copy(),
            attack_cooldown=self.attack_cooldown.copy(),
        )


@dataclass
class EnvParams:
    max_timesteps: int = 10000
    day_length: int = 300

    always_diamond: bool = True

    zombie_health: int = 5
    cow_health: int = 3
    skeleton_health: int = 3

    mob_despawn_distance: int = 14

    spawn_cow_chance: float = 0.1
    spawn_zombie_base_chance: float = 0.02
    spawn_zombie_night_chance: float = 0.1
    spawn_skeleton_chance: float = 0.05


@dataclass
class StaticEnvParams:
    map_size: Tuple[int, int] = (64, 64)

    # Mobs
    max_zombies: int = 3
    max_cows: int = 3
    max_growing_plants: int = 10
    max_skeletons: int = 2
    max_arrows: int = 3


@dataclass
class EnvState:
    map: np.ndarray  # (H, W) int32, BlockType values
    mob_map: np.ndarray  # (H, W) bool -- True where any mob currently stands

    player_position: np.ndarray  # (2,) int32
    player_direction: int  # last directional Action id (indexes DIRECTIONS)

    # Intrinsics
    player_health: int
    player_food: int
    player_drink: int
    player_energy: int
    is_sleeping: bool

    # Second order intrinsics
    player_recover: float
    player_hunger: float
    player_thirst: float
    player_fatigue: float

    inventory: Inventory

    zombies: Mobs
    cows: Mobs
    skeletons: Mobs
    arrows: Mobs
    arrow_directions: np.ndarray  # (max_arrows, 2) int32

    growing_plants_positions: np.ndarray  # (max_growing_plants, 2) int32
    growing_plants_age: np.ndarray  # (max_growing_plants,) int32
    growing_plants_mask: np.ndarray  # (max_growing_plants,) bool

    light_level: float

    achievements: np.ndarray  # (NUM_ACHIEVEMENTS,) bool

    timestep: int

    @classmethod
    def create_default(
        cls,
        static_params: StaticEnvParams | None = None,
        rng: np.random.Generator | None = None,
    ) -> "EnvState":
        """Build a minimal, valid EnvState for testing/bootstrapping.

        This is *not* a port of the original world generator (world_gen.py
        was out of scope for this port) -- it just produces an all-grass
        map with the player centered and no mobs/plants, which is enough to
        exercise `craftax_step` end-to-end.
        """
        static_params = static_params or StaticEnvParams()
        rng = rng or np.random.default_rng()

        h, w = static_params.map_size
        game_map = np.full((h, w), BlockType.GRASS, dtype=np.int32)
        mob_map = np.zeros((h, w), dtype=bool)

        player_position = np.array([h // 2, w // 2], dtype=np.int32)

        max_arrows = static_params.max_arrows
        max_plants = static_params.max_growing_plants

        return cls(
            map=game_map,
            mob_map=mob_map,
            player_position=player_position,
            player_direction=int(Action.UP),
            player_health=9,
            player_food=9,
            player_drink=9,
            player_energy=9,
            is_sleeping=False,
            player_recover=0.0,
            player_hunger=0.0,
            player_thirst=0.0,
            player_fatigue=0.0,
            inventory=Inventory(),
            zombies=Mobs.empty(static_params.max_zombies),
            cows=Mobs.empty(static_params.max_cows),
            skeletons=Mobs.empty(static_params.max_skeletons),
            arrows=Mobs.empty(max_arrows),
            arrow_directions=np.zeros((max_arrows, 2), dtype=np.int32),
            growing_plants_positions=np.zeros((max_plants, 2), dtype=np.int32),
            growing_plants_age=np.zeros((max_plants,), dtype=np.int32),
            growing_plants_mask=np.zeros((max_plants,), dtype=bool),
            light_level=1.0,
            achievements=np.zeros((22,), dtype=bool),
            timestep=0,
        )
