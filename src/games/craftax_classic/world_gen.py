"""NumPy world generator for Craftax-Classic.

Ported from ``reference/craftax/.../craftax_classic/world_gen.py``.
"""

from __future__ import annotations

import numpy as np

from .constants import Action, BlockType, NUM_ACHIEVEMENTS
from .logic import calculate_light_level, get_distance_map
from .noise import generate_fractal_noise_2d
from .state import EnvParams, EnvState, Inventory, Mobs, StaticEnvParams


def generate_world(
    rng: np.random.Generator,
    params: EnvParams | None = None,
    static_params: StaticEnvParams | None = None,
) -> EnvState:
    params = params or EnvParams()
    static_params = static_params or StaticEnvParams()
    h, w = static_params.map_size
    map_size = (h, w)

    player_position = np.array([h // 2, w // 2], dtype=np.int32)

    player_proximity_map = get_distance_map(player_position, map_size).astype(np.float32)
    player_proximity_map = np.clip(player_proximity_map / 5.0, 0.0, 1.0)

    larger_res = (h // 4, w // 4)
    small_res = (h // 16, w // 16)
    x_res = (h // 8, w // 2)

    water = generate_fractal_noise_2d(rng, map_size, small_res, octaves=1)
    water = water + player_proximity_map - 1.0

    game_map = np.where(water > 0.7, BlockType.WATER, BlockType.GRASS).astype(np.int32)

    sand_map = (water < 0.75) & (water > 0.6) & (game_map != BlockType.WATER)
    game_map = np.where(sand_map, BlockType.SAND, game_map)

    mountain_threshold = 0.7
    mountain = generate_fractal_noise_2d(rng, map_size, small_res, octaves=1) + 0.05
    mountain = mountain + player_proximity_map - 1.0
    game_map = np.where(mountain > mountain_threshold, BlockType.STONE, game_map)

    path_x = generate_fractal_noise_2d(rng, map_size, x_res, octaves=1)
    path = (mountain > mountain_threshold) & (path_x > 0.8)
    game_map = np.where(path, BlockType.PATH, game_map)

    path_y = path_x.T
    path = (mountain > mountain_threshold) & (path_y > 0.8)
    game_map = np.where(path, BlockType.PATH, game_map)

    caves = (mountain > 0.85) & (water > 0.4)
    game_map = np.where(caves, BlockType.PATH, game_map)

    coal_map = (game_map == BlockType.STONE) & (rng.random(map_size) < 0.04)
    game_map = np.where(coal_map, BlockType.COAL, game_map)

    iron_map = (game_map == BlockType.STONE) & (rng.random(map_size) < 0.03)
    game_map = np.where(iron_map, BlockType.IRON, game_map)

    diamond_map = (mountain > 0.8) & (rng.random(map_size) < 0.005) & (game_map == BlockType.STONE)
    game_map = np.where(diamond_map, BlockType.DIAMOND, game_map)

    tree_noise = generate_fractal_noise_2d(rng, map_size, larger_res, octaves=1)
    tree = (tree_noise > 0.5) & (rng.random(map_size) > 0.8) & (game_map == BlockType.GRASS)
    game_map = np.where(tree, BlockType.TREE, game_map)

    lava_map = (mountain > 0.85) & (tree_noise > 0.7)
    game_map = np.where(lava_map, BlockType.LAVA, game_map)

    # Ensure player spawns on grass.
    game_map[player_position[0], player_position[1]] = BlockType.GRASS

    # Place a guaranteed diamond when always_diamond is set.
    stone_cells = np.flatnonzero(game_map.ravel() == BlockType.STONE)
    if len(stone_cells) > 0:
        diamond_index = int(rng.choice(stone_cells))
        diamond_r = diamond_index // w
        diamond_c = diamond_index % w
        if params.always_diamond:
            game_map[diamond_r, diamond_c] = BlockType.DIAMOND

    zombies = Mobs.empty(static_params.max_zombies)
    zombies.health[:] = 1

    cows = Mobs.empty(static_params.max_cows)
    cows.health[:] = params.cow_health

    skeletons = Mobs.empty(static_params.max_skeletons)
    arrows = Mobs.empty(static_params.max_arrows)
    max_plants = static_params.max_growing_plants

    return EnvState(
        map=game_map.astype(np.int32),
        mob_map=np.zeros(map_size, dtype=bool),
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
        zombies=zombies,
        cows=cows,
        skeletons=skeletons,
        arrows=arrows,
        arrow_directions=np.ones((static_params.max_arrows, 2), dtype=np.int32),
        growing_plants_positions=np.zeros((max_plants, 2), dtype=np.int32),
        growing_plants_age=np.zeros((max_plants,), dtype=np.int32),
        growing_plants_mask=np.zeros((max_plants,), dtype=bool),
        light_level=float(calculate_light_level(0, params)),
        achievements=np.zeros((NUM_ACHIEVEMENTS,), dtype=bool),
        timestep=0,
    )
