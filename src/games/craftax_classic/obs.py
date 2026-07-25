"""Symbolic observation encoder for Craftax-Classic.

Layout matches ``render_craftax_symbolic`` in the JAX reference:
``7 * 9 * 21 + 12 + 4 + 4 + 2 = 1345`` float features.
"""

from __future__ import annotations

import numpy as np

from .constants import MAX_OBS_DIM, NUM_BLOCK_TYPES, OBS_DIM
from .state import EnvState

OBS_MAP_CHANNELS = NUM_BLOCK_TYPES + 4  # blocks + zombie/cow/skel/arrow
OBS_SIZE = int(OBS_DIM[0] * OBS_DIM[1] * OBS_MAP_CHANNELS + 12 + 4 + 4 + 2)


def encode_symbolic(state: EnvState) -> np.ndarray:
    obs_h, obs_w = OBS_DIM
    pad = MAX_OBS_DIM + 2
    padded = np.pad(
        state.map,
        ((pad, pad), (pad, pad)),
        constant_values=1,  # BlockType.OUT_OF_BOUNDS
    )

    pr, pc = int(state.player_position[0]), int(state.player_position[1])
    tl_r = pr - obs_h // 2 + pad
    tl_c = pc - obs_w // 2 + pad
    map_view = padded[tl_r : tl_r + obs_h, tl_c : tl_c + obs_w]

    map_one_hot = np.eye(NUM_BLOCK_TYPES, dtype=np.float32)[map_view]

    mob_map = np.zeros((obs_h, obs_w, 4), dtype=np.float32)
    half = np.array([obs_h // 2, obs_w // 2], dtype=np.int32)

    def _add_mobs(mobs, channel: int) -> None:
        for i in range(len(mobs.mask)):
            if not mobs.mask[i]:
                continue
            local = mobs.position[i] - state.player_position + half
            r, c = int(local[0]), int(local[1])
            if 0 <= r < obs_h and 0 <= c < obs_w:
                mob_map[r, c, channel] = 1.0

    _add_mobs(state.zombies, 0)
    _add_mobs(state.cows, 1)
    _add_mobs(state.skeletons, 2)
    _add_mobs(state.arrows, 3)

    all_map = np.concatenate([map_one_hot, mob_map], axis=-1)

    inv = state.inventory
    inventory = (
        np.array(
            [
                inv.wood,
                inv.stone,
                inv.coal,
                inv.iron,
                inv.diamond,
                inv.sapling,
                inv.wood_pickaxe,
                inv.stone_pickaxe,
                inv.iron_pickaxe,
                inv.wood_sword,
                inv.stone_sword,
                inv.iron_sword,
            ],
            dtype=np.float32,
        )
        / 10.0
    )

    intrinsics = (
        np.array(
            [
                state.player_health,
                state.player_food,
                state.player_drink,
                state.player_energy,
            ],
            dtype=np.float32,
        )
        / 10.0
    )

    direction = np.zeros(4, dtype=np.float32)
    dir_idx = int(state.player_direction) - 1
    if 0 <= dir_idx < 4:
        direction[dir_idx] = 1.0

    extras = np.array(
        [float(state.light_level), float(state.is_sleeping)],
        dtype=np.float32,
    )

    return np.concatenate(
        [all_map.reshape(-1), inventory, intrinsics, direction, extras],
        axis=0,
    ).astype(np.float32, copy=False)
