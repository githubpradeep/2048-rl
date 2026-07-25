"""Craftax-Classic package exports."""

from .constants import Achievement, Action, BlockType, NUM_ACTIONS, NUM_ACHIEVEMENTS
from .game import CraftaxClassicConfig, CraftaxClassicEnv, CraftaxClassicGame
from .logic import (
    calculate_light_level,
    craftax_step,
    get_distance_map,
    is_game_over,
    legal_actions_for_state,
)
from .obs import OBS_SIZE, encode_symbolic
from .state import EnvParams, EnvState, Inventory, Mobs, StaticEnvParams
from .world_gen import generate_world

# Optional: pixel renderer is imported by play; keep package importable without pygame.

__all__ = [
    "Achievement",
    "Action",
    "BlockType",
    "CraftaxClassicConfig",
    "CraftaxClassicEnv",
    "CraftaxClassicGame",
    "EnvParams",
    "EnvState",
    "Inventory",
    "Mobs",
    "NUM_ACTIONS",
    "NUM_ACHIEVEMENTS",
    "OBS_SIZE",
    "StaticEnvParams",
    "calculate_light_level",
    "craftax_step",
    "encode_symbolic",
    "generate_world",
    "get_distance_map",
    "is_game_over",
    "legal_actions_for_state",
]
