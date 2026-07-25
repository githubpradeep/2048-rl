"""Game constants for the pure-Python/NumPy port of Craftax-Classic.

Ported from ``reference/craftax/craftax/craftax_classic/constants.py``. All
JAX arrays have been replaced with NumPy arrays; texture loading and
rendering-only constants (which depend on JAX/PIL/imageio) have been
dropped since this port only concerns itself with game *logic*.
"""

from enum import IntEnum

import numpy as np

# GAME CONSTANTS
OBS_DIM = (7, 9)
MAX_OBS_DIM = max(OBS_DIM)
assert OBS_DIM[0] % 2 == 1 and OBS_DIM[1] % 2 == 1

NUM_BLOCK_TYPES = 17
NUM_ACTIONS = 17
NUM_ACHIEVEMENTS = 22


class BlockType(IntEnum):
    INVALID = 0
    OUT_OF_BOUNDS = 1
    GRASS = 2
    WATER = 3
    STONE = 4
    TREE = 5
    WOOD = 6
    PATH = 7
    COAL = 8
    IRON = 9
    DIAMOND = 10
    CRAFTING_TABLE = 11
    FURNACE = 12
    SAND = 13
    LAVA = 14
    PLANT = 15
    RIPE_PLANT = 16


class Action(IntEnum):
    NOOP = 0  #
    LEFT = 1  # a
    RIGHT = 2  # d
    UP = 3  # w
    DOWN = 4  # s
    DO = 5  # space
    SLEEP = 6  # tab
    PLACE_STONE = 7  # r
    PLACE_TABLE = 8  # t
    PLACE_FURNACE = 9  # f
    PLACE_PLANT = 10  # p
    MAKE_WOOD_PICKAXE = 11  # 1
    MAKE_STONE_PICKAXE = 12  # 2
    MAKE_IRON_PICKAXE = 13  # 3
    MAKE_WOOD_SWORD = 14  # 4
    MAKE_STONE_SWORD = 15  # 5
    MAKE_IRON_SWORD = 16  # 6


# Opposite cardinal moves — used to break DQN UP/DOWN and LEFT/RIGHT oscillation.
MOVE_OPPOSITE: dict[int, int] = {
    int(Action.LEFT): int(Action.RIGHT),
    int(Action.RIGHT): int(Action.LEFT),
    int(Action.UP): int(Action.DOWN),
    int(Action.DOWN): int(Action.UP),
}


class Achievement(IntEnum):
    COLLECT_WOOD = 0
    PLACE_TABLE = 1
    EAT_COW = 2
    COLLECT_SAPLING = 3
    COLLECT_DRINK = 4
    MAKE_WOOD_PICKAXE = 5
    MAKE_WOOD_SWORD = 6
    PLACE_PLANT = 7
    DEFEAT_ZOMBIE = 8
    COLLECT_STONE = 9
    PLACE_STONE = 10
    EAT_PLANT = 11
    DEFEAT_SKELETON = 12
    MAKE_STONE_PICKAXE = 13
    MAKE_STONE_SWORD = 14
    WAKE_UP = 15
    PLACE_FURNACE = 16
    COLLECT_COAL = 17
    COLLECT_IRON = 18
    COLLECT_DIAMOND = 19
    MAKE_IRON_PICKAXE = 20
    MAKE_IRON_SWORD = 21


# GAME MECHANICS

# Indexed by Action value (or by `player_direction`, which stores the last
# directional Action taken). Shape is (NUM_ACTIONS, 2) so that any valid
# action id can be used to safely index this array in plain NumPy (unlike
# JAX's clipped fancy-indexing, NumPy raises on out-of-range indices).
#
# 0: NOOP -> [0, 0]
# 1: LEFT -> [0, -1]
# 2: RIGHT -> [0, 1]
# 3: UP -> [-1, 0]
# 4: DOWN -> [1, 0]
# 5..16: unused (zero vectors)
DIRECTIONS = np.concatenate(
    (
        np.array([[0, 0], [0, -1], [0, 1], [-1, 0], [1, 0]], dtype=np.int32),
        np.zeros((NUM_ACTIONS - 5, 2), dtype=np.int32),
    ),
    axis=0,
)

# The 8 cells directly surrounding the player (used to check "near block"
# conditions for crafting, e.g. "is there a crafting table adjacent?").
CLOSE_BLOCKS = np.array(
    [
        [0, -1],
        [0, 1],
        [-1, 0],
        [1, 0],
        [-1, -1],
        [-1, 1],
        [1, -1],
        [1, 1],
    ],
    dtype=np.int32,
)

# Can't walk through these.
SOLID_BLOCKS = frozenset(
    {
        BlockType.WATER,
        BlockType.STONE,
        BlockType.TREE,
        BlockType.COAL,
        BlockType.IRON,
        BlockType.DIAMOND,
        BlockType.CRAFTING_TABLE,
        BlockType.FURNACE,
        BlockType.PLANT,
        BlockType.RIPE_PLANT,
    }
)
