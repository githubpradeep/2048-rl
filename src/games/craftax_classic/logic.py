"""Pure Python + NumPy port of Craftax-Classic's game logic.

Ported from ``reference/craftax/craftax/craftax_classic/game_logic.py``.
Translation notes:

- ``jax.lax.select(cond, a, b)`` -> plain Python ``a if cond else b`` (or, more
  often, an ``if cond: ...`` block that mutates state directly, since we no
  longer need to keep every branch "alive" for tracing).
- ``jax.lax.scan`` over a fixed-size population (zombies/cows/.../plants)
  -> a plain Python ``for`` loop.
- ``jax.random.PRNGKey`` + ``jax.random.split`` -> a single stateful
  ``numpy.random.Generator`` threaded through every function that needs
  randomness. Since a ``Generator`` mutates its own internal state on every
  draw, there is no need to split/thread keys manually.
- State is represented with mutable dataclasses (see ``state.py``) holding
  NumPy arrays, so subsystems mutate ``state`` (and the arrays/objects it
  holds) in place rather than returning brand-new immutable copies.

Mechanically, the effects of every action, every achievement trigger, the
reward formula, and the subsystem order inside ``craftax_step`` all mirror
the original JAX implementation.
"""

from __future__ import annotations

import math
from typing import Tuple

import numpy as np

from .constants import (
    Achievement,
    Action,
    BlockType,
    CLOSE_BLOCKS,
    DIRECTIONS,
    SOLID_BLOCKS,
)
from .state import EnvParams, EnvState, Mobs, StaticEnvParams


# ---------------------------------------------------------------------------
# Basic predicates
# ---------------------------------------------------------------------------


def is_game_over(state: EnvState, params: EnvParams) -> bool:
    done_steps = state.timestep >= params.max_timesteps
    in_lava = (
        state.map[state.player_position[0], state.player_position[1]]
        == BlockType.LAVA
    )
    is_dead = state.player_health <= 0
    return bool(done_steps or in_lava or is_dead)


def in_bounds(state: EnvState, position: np.ndarray) -> bool:
    in_bounds_x = 0 <= position[0] < state.map.shape[0]
    in_bounds_y = 0 <= position[1] < state.map.shape[1]
    return bool(in_bounds_x and in_bounds_y)


def is_in_wall(state: EnvState, position: np.ndarray) -> bool:
    return int(state.map[position[0], position[1]]) in SOLID_BLOCKS


def is_in_mob(state: EnvState, position: np.ndarray) -> bool:
    on_player = (
        state.player_position[0] == position[0]
        and state.player_position[1] == position[1]
    )
    return bool(state.mob_map[position[0], position[1]] or on_player)


def is_position_in_bounds_not_in_wall_not_in_mob_not_in_lava(
    state: EnvState, position: np.ndarray
) -> bool:
    if not in_bounds(state, position):
        return False
    if is_in_wall(state, position):
        return False
    if is_in_mob(state, position):
        return False
    if state.map[position[0], position[1]] == BlockType.LAVA:
        return False
    return True


def get_player_attack_damage(state: EnvState) -> int:
    damages = (
        1,
        2 * state.inventory.wood_sword,
        3 * state.inventory.stone_sword,
        5 * state.inventory.iron_sword,
    )
    return max(damages)


def update_plants_with_eat(
    state: EnvState, plant_position: np.ndarray, static_params: StaticEnvParams
) -> np.ndarray:
    """Return a copy of ``growing_plants_age`` with the plant living at
    ``plant_position`` reset to age 0 (used when the player eats a ripe
    plant). Mirrors the original's "first match, else index 0" behaviour --
    the caller only commits this result when a plant was actually being
    eaten at a valid position, so the "else index 0" fallback is never
    observable.
    """
    new_age = state.growing_plants_age.copy()
    plant_index = 0
    for i in range(static_params.max_growing_plants):
        if (
            state.growing_plants_positions[i][0] == plant_position[0]
            and state.growing_plants_positions[i][1] == plant_position[1]
        ):
            plant_index = i
            break
    new_age[plant_index] = 0
    return new_age


# ---------------------------------------------------------------------------
# Interact (mine / attack / drink / eat)
# ---------------------------------------------------------------------------


def do_action(
    rng: np.random.Generator,
    state: EnvState,
    action: int,
    static_params: StaticEnvParams,
) -> EnvState:
    """Handle the DO action: attacking mobs takes priority over mining the
    block in front of the player; if a mob was attacked, mining/drinking/
    eating is skipped entirely this step."""
    if action != Action.DO:
        return state

    block_position = state.player_position + DIRECTIONS[state.player_direction]
    bx, by = int(block_position[0]), int(block_position[1])

    def _attack(mobs: Mobs, achievement: Achievement) -> Tuple[bool, bool]:
        target_index = None
        for idx in range(len(mobs.mask)):
            if (
                mobs.mask[idx]
                and mobs.position[idx][0] == bx
                and mobs.position[idx][1] == by
            ):
                target_index = idx
                break
        if target_index is None:
            return False, False

        was_alive = bool(mobs.mask[target_index])
        mobs.health[target_index] -= get_player_attack_damage(state)
        mobs.mask[target_index] = mobs.health[target_index] > 0
        did_kill = was_alive and not bool(mobs.mask[target_index])
        if did_kill:
            state.achievements[achievement] = True
        return True, did_kill

    is_attacking_zombie, did_kill_zombie = _attack(
        state.zombies, Achievement.DEFEAT_ZOMBIE
    )

    is_attacking_cow, did_kill_cow = _attack(state.cows, Achievement.EAT_COW)
    if did_kill_cow:
        state.player_food = min(9, state.player_food + 6)
        state.player_hunger = 0.0

    is_attacking_skeleton, did_kill_skeleton = _attack(
        state.skeletons, Achievement.DEFEAT_SKELETON
    )

    did_attack_mob = is_attacking_zombie or is_attacking_cow or is_attacking_skeleton
    did_kill_mob = did_kill_zombie or did_kill_cow or did_kill_skeleton

    if did_kill_mob and in_bounds(state, block_position):
        state.mob_map[bx, by] = False

    # A random draw is always consumed here (whether or not it ends up
    # mattering) to mirror the structure of the original, which always
    # rolls for the 10% "found a sapling" chance whenever action == DO.
    sapling_roll = rng.random()

    in_bounds_block = in_bounds(state, block_position)
    action_block_in_bounds = in_bounds_block and not did_attack_mob

    if action_block_in_bounds:
        block_type = BlockType(int(state.map[bx, by]))
        new_block_type = block_type

        if block_type == BlockType.TREE:
            new_block_type = BlockType.GRASS
            state.inventory.wood += 1
            state.achievements[Achievement.COLLECT_WOOD] = True

        if block_type == BlockType.STONE and state.inventory.wood_pickaxe > 0:
            new_block_type = BlockType.PATH
            state.inventory.stone += 1
            state.achievements[Achievement.COLLECT_STONE] = True

        if block_type == BlockType.COAL and state.inventory.wood_pickaxe > 0:
            new_block_type = BlockType.PATH
            state.inventory.coal += 1
            state.achievements[Achievement.COLLECT_COAL] = True

        if block_type == BlockType.IRON and state.inventory.stone_pickaxe > 0:
            new_block_type = BlockType.PATH
            state.inventory.iron += 1
            state.achievements[Achievement.COLLECT_IRON] = True

        if block_type == BlockType.DIAMOND and state.inventory.iron_pickaxe > 0:
            new_block_type = BlockType.PATH
            state.inventory.diamond += 1
            state.achievements[Achievement.COLLECT_DIAMOND] = True

        if block_type == BlockType.GRASS and sapling_roll < 0.1:
            state.inventory.sapling += 1
            state.achievements[Achievement.COLLECT_SAPLING] = True

        if block_type == BlockType.WATER:
            state.player_drink = min(9, state.player_drink + 1)
            state.player_thirst = 0.0
            state.achievements[Achievement.COLLECT_DRINK] = True

        if block_type == BlockType.RIPE_PLANT:
            new_block_type = BlockType.PLANT
            state.player_food = min(9, state.player_food + 4)
            state.player_hunger = 0.0
            state.achievements[Achievement.EAT_PLANT] = True
            state.growing_plants_age = update_plants_with_eat(
                state, block_position, static_params
            )

        state.map[bx, by] = new_block_type

    return state


def is_near_block(state: EnvState, block_type: BlockType) -> bool:
    px, py = int(state.player_position[0]), int(state.player_position[1])
    h, w = state.map.shape
    for dx, dy in CLOSE_BLOCKS:
        x, y = px + int(dx), py + int(dy)
        if 0 <= x < h and 0 <= y < w and state.map[x, y] == block_type:
            return True
    return False


def _facing_cell(state: EnvState) -> tuple[np.ndarray, bool]:
    pos = state.player_position + DIRECTIONS[state.player_direction]
    return pos, in_bounds(state, pos)


def _facing_has_mob(state: EnvState, pos: np.ndarray) -> bool:
    bx, by = int(pos[0]), int(pos[1])
    return bool(state.mob_map[bx, by])


def do_action_is_useful(state: EnvState) -> bool:
    """True when DO would attack, mine, drink (if thirsty), or harvest."""
    block_position, ok = _facing_cell(state)
    if not ok:
        return False
    if _facing_has_mob(state, block_position):
        return True
    bx, by = int(block_position[0]), int(block_position[1])
    block_type = BlockType(int(state.map[bx, by]))
    inv = state.inventory
    if block_type == BlockType.TREE:
        return True
    if block_type == BlockType.WATER and state.player_drink < 9:
        return True
    if block_type == BlockType.RIPE_PLANT:
        return True
    if block_type in (BlockType.STONE, BlockType.COAL) and inv.wood_pickaxe > 0:
        return True
    if block_type == BlockType.IRON and inv.stone_pickaxe > 0:
        return True
    if block_type == BlockType.DIAMOND and inv.iron_pickaxe > 0:
        return True
    return False


def legal_actions_for_state(state: EnvState) -> list[int]:
    """Return actions that are not known no-ops in the current state.

    Classic exposes all 17 actions every step; most craft/place/DO choices do
    nothing and greedy DQN collapses onto them (player appears frozen). Masking
    keeps movement/interact available for training and demos.
    """
    if state.is_sleeping:
        # Sleep overrides every action to NOOP until energy recovers.
        return [int(Action.NOOP)]

    legal = [int(Action.LEFT), int(Action.RIGHT), int(Action.UP), int(Action.DOWN)]

    if do_action_is_useful(state):
        legal.append(int(Action.DO))

    if state.player_energy < 9:
        legal.append(int(Action.SLEEP))

    inv = state.inventory
    place_pos, place_ok = _facing_cell(state)
    can_place = place_ok and not is_in_mob(state, place_pos)
    if can_place:
        px, py = int(place_pos[0]), int(place_pos[1])
        original = BlockType(int(state.map[px, py]))
        in_wall = is_in_wall(state, place_pos)
        if not in_wall and inv.wood >= 2:
            legal.append(int(Action.PLACE_TABLE))
        if not in_wall and inv.stone > 0:
            legal.append(int(Action.PLACE_FURNACE))
        if (original == BlockType.WATER or not in_wall) and inv.stone > 0:
            legal.append(int(Action.PLACE_STONE))
        if original == BlockType.GRASS and inv.sapling > 0:
            legal.append(int(Action.PLACE_PLANT))

    at_table = is_near_block(state, BlockType.CRAFTING_TABLE)
    at_furnace = is_near_block(state, BlockType.FURNACE)
    if at_table and inv.wood >= 1:
        legal.append(int(Action.MAKE_WOOD_PICKAXE))
        legal.append(int(Action.MAKE_WOOD_SWORD))
    if at_table and inv.wood >= 1 and inv.stone >= 1:
        legal.append(int(Action.MAKE_STONE_PICKAXE))
        legal.append(int(Action.MAKE_STONE_SWORD))
    if (
        at_table
        and at_furnace
        and inv.wood >= 1
        and inv.stone >= 1
        and inv.iron >= 1
        and inv.coal >= 1
    ):
        legal.append(int(Action.MAKE_IRON_PICKAXE))
        legal.append(int(Action.MAKE_IRON_SWORD))

    return legal


# ---------------------------------------------------------------------------
# Crafting
# ---------------------------------------------------------------------------


def do_crafting(state: EnvState, action: int) -> EnvState:
    is_at_crafting_table = is_near_block(state, BlockType.CRAFTING_TABLE)
    is_at_furnace = is_near_block(state, BlockType.FURNACE)
    inv = state.inventory

    if action == Action.MAKE_WOOD_PICKAXE and is_at_crafting_table and inv.wood >= 1:
        inv.wood -= 1
        inv.wood_pickaxe += 1
        state.achievements[Achievement.MAKE_WOOD_PICKAXE] = True

    if (
        action == Action.MAKE_STONE_PICKAXE
        and is_at_crafting_table
        and inv.wood >= 1
        and inv.stone >= 1
    ):
        inv.wood -= 1
        inv.stone -= 1
        inv.stone_pickaxe += 1
        state.achievements[Achievement.MAKE_STONE_PICKAXE] = True

    if (
        action == Action.MAKE_IRON_PICKAXE
        and is_at_furnace
        and is_at_crafting_table
        and inv.wood >= 1
        and inv.stone >= 1
        and inv.iron >= 1
        and inv.coal >= 1
    ):
        inv.wood -= 1
        inv.stone -= 1
        inv.iron -= 1
        inv.coal -= 1
        inv.iron_pickaxe += 1
        state.achievements[Achievement.MAKE_IRON_PICKAXE] = True

    if action == Action.MAKE_WOOD_SWORD and is_at_crafting_table and inv.wood >= 1:
        inv.wood -= 1
        inv.wood_sword += 1
        state.achievements[Achievement.MAKE_WOOD_SWORD] = True

    if (
        action == Action.MAKE_STONE_SWORD
        and is_at_crafting_table
        and inv.stone >= 1
        and inv.wood >= 1
    ):
        inv.wood -= 1
        inv.stone -= 1
        inv.stone_sword += 1
        state.achievements[Achievement.MAKE_STONE_SWORD] = True

    if (
        action == Action.MAKE_IRON_SWORD
        and is_at_furnace
        and is_at_crafting_table
        and inv.iron >= 1
        and inv.wood >= 1
        and inv.stone >= 1
        and inv.coal >= 1
    ):
        inv.wood -= 1
        inv.iron -= 1
        inv.stone -= 1
        inv.coal -= 1
        inv.iron_sword += 1
        state.achievements[Achievement.MAKE_IRON_SWORD] = True

    return state


# ---------------------------------------------------------------------------
# Placing blocks / saplings
# ---------------------------------------------------------------------------


def add_new_growing_plant(
    state: EnvState,
    position: np.ndarray,
    is_placing_sapling: bool,
    static_params: StaticEnvParams,
) -> None:
    """Register a newly-placed sapling in the first free growing-plant slot
    (a no-op if there is no free slot, or if a sapling isn't being placed)."""
    if not is_placing_sapling:
        return
    for i in range(static_params.max_growing_plants):
        if not state.growing_plants_mask[i]:
            state.growing_plants_positions[i] = position
            state.growing_plants_age[i] = 0
            state.growing_plants_mask[i] = True
            return


def place_block(state: EnvState, action: int, static_params: StaticEnvParams) -> EnvState:
    placing_block_position = (
        state.player_position + DIRECTIONS[state.player_direction]
    )
    px, py = int(placing_block_position[0]), int(placing_block_position[1])

    in_bounds_place = in_bounds(state, placing_block_position)
    action_block_in_bounds = in_bounds_place and not is_in_mob(
        state, placing_block_position
    )

    is_placing_sapling = False

    if action_block_in_bounds:
        inv = state.inventory
        original_block = BlockType(int(state.map[px, py]))
        currently_in_wall = is_in_wall(state, placing_block_position)

        if action == Action.PLACE_TABLE and not currently_in_wall and inv.wood >= 2:
            state.map[px, py] = BlockType.CRAFTING_TABLE
            inv.wood -= 2
            state.achievements[Achievement.PLACE_TABLE] = True

        if action == Action.PLACE_FURNACE and not currently_in_wall and inv.stone > 0:
            state.map[px, py] = BlockType.FURNACE
            inv.stone -= 1
            state.achievements[Achievement.PLACE_FURNACE] = True

        is_placing_on_valid_block = (
            original_block == BlockType.WATER or not currently_in_wall
        )
        if action == Action.PLACE_STONE and is_placing_on_valid_block and inv.stone > 0:
            state.map[px, py] = BlockType.STONE
            inv.stone -= 1
            state.achievements[Achievement.PLACE_STONE] = True

        if (
            action == Action.PLACE_PLANT
            and original_block == BlockType.GRASS
            and inv.sapling > 0
        ):
            state.map[px, py] = BlockType.PLANT
            inv.sapling -= 1
            state.achievements[Achievement.PLACE_PLANT] = True
            is_placing_sapling = True

    add_new_growing_plant(state, placing_block_position, is_placing_sapling, static_params)

    return state


# ---------------------------------------------------------------------------
# Time / light
# ---------------------------------------------------------------------------


def calculate_light_level(timestep: int, params: EnvParams) -> float:
    progress = (timestep / params.day_length) % 1 + 0.3
    return 1 - abs(math.cos(math.pi * progress)) ** 3


# ---------------------------------------------------------------------------
# Mobs
# ---------------------------------------------------------------------------


def _clear_and_register_mob_map(
    state: EnvState,
    old_position: np.ndarray,
    new_position: np.ndarray,
    was_alive: bool,
    new_mask: bool,
) -> None:
    if was_alive:
        state.mob_map[old_position[0], old_position[1]] = False
    if new_mask:
        state.mob_map[new_position[0], new_position[1]] = True


def _choose_axis_towards_player(
    rng: np.random.Generator, player_position: np.ndarray, mob_position: np.ndarray
) -> np.ndarray:
    """Pick a single-axis direction vector stepping the mob towards the
    player, along whichever axis (x or y) currently has the larger absolute
    distance (ties broken uniformly at random)."""
    abs_diff = np.abs(player_position - mob_position)
    max_diff = abs_diff.max()
    candidate_axes = np.flatnonzero(abs_diff == max_diff)
    axis = int(rng.choice(candidate_axes))
    direction = np.zeros(2, dtype=np.int32)
    direction[axis] = np.sign(player_position[axis] - mob_position[axis]).astype(np.int32)
    return direction


def _move_zombies(
    rng: np.random.Generator,
    state: EnvState,
    params: EnvParams,
    static_params: StaticEnvParams,
) -> None:
    for i in range(static_params.max_zombies):
        zombies = state.zombies
        old_position = zombies.position[i].copy()
        was_alive = bool(zombies.mask[i])

        random_move_proposed = old_position + DIRECTIONS[1 + rng.integers(4)]

        player_move_direction = _choose_axis_towards_player(
            rng, state.player_position, old_position
        )
        player_move_proposed = old_position + player_move_direction

        close_to_player = int(np.abs(old_position - state.player_position).sum()) < 10
        close_to_player = close_to_player and rng.random() < 0.75

        proposed_position = (
            player_move_proposed if close_to_player else random_move_proposed
        )

        is_attacking_player = (
            int(np.abs(old_position - state.player_position).sum()) == 1
            and zombies.attack_cooldown[i] <= 0
            and was_alive
        )
        if is_attacking_player:
            proposed_position = old_position

        zombie_damage = 7 if state.is_sleeping else 2
        new_cooldown = 5 if is_attacking_player else zombies.attack_cooldown[i] - 1
        is_waking_player = state.is_sleeping and is_attacking_player

        if is_attacking_player:
            state.player_health -= zombie_damage
        state.is_sleeping = state.is_sleeping and not is_attacking_player
        if is_waking_player:
            state.achievements[Achievement.WAKE_UP] = True

        valid_move = is_position_in_bounds_not_in_wall_not_in_mob_not_in_lava(
            state, proposed_position
        )
        new_position = proposed_position if valid_move else old_position

        should_not_despawn = (
            int(np.abs(old_position - state.player_position).sum())
            < params.mob_despawn_distance
        )
        new_mask = was_alive and should_not_despawn

        _clear_and_register_mob_map(state, old_position, new_position, was_alive, new_mask)

        zombies.position[i] = new_position
        zombies.attack_cooldown[i] = new_cooldown
        zombies.mask[i] = new_mask


def _move_cows(
    rng: np.random.Generator,
    state: EnvState,
    params: EnvParams,
    static_params: StaticEnvParams,
) -> None:
    for i in range(static_params.max_cows):
        cows = state.cows
        old_position = cows.position[i].copy()
        was_alive = bool(cows.mask[i])

        # DIRECTIONS[1:9] includes 4 real directions + 4 zero vectors, i.e.
        # a 50% chance of not moving at all.
        proposed_position = old_position + DIRECTIONS[1 + rng.integers(8)]

        valid_move = is_position_in_bounds_not_in_wall_not_in_mob_not_in_lava(
            state, proposed_position
        )
        new_position = proposed_position if valid_move else old_position

        should_not_despawn = (
            int(np.abs(old_position - state.player_position).sum())
            < params.mob_despawn_distance
        )
        new_mask = was_alive and should_not_despawn

        _clear_and_register_mob_map(state, old_position, new_position, was_alive, new_mask)

        cows.position[i] = new_position
        cows.mask[i] = new_mask


def _move_skeletons(
    rng: np.random.Generator,
    state: EnvState,
    params: EnvParams,
    static_params: StaticEnvParams,
) -> None:
    for i in range(static_params.max_skeletons):
        skeletons = state.skeletons
        old_position = skeletons.position[i].copy()
        was_alive = bool(skeletons.mask[i])

        random_move_proposed = old_position + DIRECTIONS[1 + rng.integers(4)]

        player_move_direction = _choose_axis_towards_player(
            rng, state.player_position, old_position
        )
        player_move_towards_proposed = old_position + player_move_direction
        player_move_away_proposed = old_position - player_move_direction

        distance_to_player = int(np.abs(old_position - state.player_position).sum())
        far_from_player = distance_to_player >= 10
        too_close_to_player = distance_to_player <= 3

        proposed_position = (
            player_move_towards_proposed if far_from_player else random_move_proposed
        )
        if too_close_to_player:
            proposed_position = player_move_away_proposed

        # 85% chance to override with pure random movement regardless of the
        # far/close logic above.
        if rng.random() <= 0.85:
            proposed_position = random_move_proposed

        is_attacking_player = 4 <= distance_to_player <= 5
        # If we want to flee (too close) but are blocked, shoot instead.
        if too_close_to_player and not is_position_in_bounds_not_in_wall_not_in_mob_not_in_lava(
            state, proposed_position
        ):
            is_attacking_player = True
        is_attacking_player = (
            is_attacking_player and skeletons.attack_cooldown[i] <= 0 and was_alive
        )

        can_spawn_arrow = int(state.arrows.mask.sum()) < static_params.max_arrows
        is_spawning_arrow = is_attacking_player and can_spawn_arrow
        if is_spawning_arrow:
            new_arrow_index = int(np.argmax(~state.arrows.mask))
            state.arrows.position[new_arrow_index] = old_position
            state.arrows.mask[new_arrow_index] = True
            state.arrow_directions[new_arrow_index] = player_move_direction

        if is_attacking_player:
            proposed_position = old_position

        new_cooldown = 4 if is_attacking_player else skeletons.attack_cooldown[i] - 1

        valid_move = is_position_in_bounds_not_in_wall_not_in_mob_not_in_lava(
            state, proposed_position
        )
        new_position = proposed_position if valid_move else old_position

        should_not_despawn = (
            int(np.abs(old_position - state.player_position).sum())
            < params.mob_despawn_distance
        )
        new_mask = was_alive and should_not_despawn

        _clear_and_register_mob_map(state, old_position, new_position, was_alive, new_mask)

        skeletons.position[i] = new_position
        skeletons.attack_cooldown[i] = new_cooldown
        skeletons.mask[i] = new_mask


def _move_arrows(
    rng: np.random.Generator,
    state: EnvState,
    params: EnvParams,
    static_params: StaticEnvParams,
) -> None:
    del rng, params  # Arrow movement is fully deterministic given direction.

    for i in range(static_params.max_arrows):
        arrows = state.arrows
        if not arrows.mask[i]:
            continue

        proposed_position = arrows.position[i] + state.arrow_directions[i]

        hit_player = bool(
            proposed_position[0] == state.player_position[0]
            and proposed_position[1] == state.player_position[1]
        )

        in_bounds_arrow = in_bounds(state, proposed_position)
        if in_bounds_arrow:
            # Arrows can fly over water.
            in_wall = is_in_wall(state, proposed_position) and not (
                state.map[proposed_position[0], proposed_position[1]]
                == BlockType.WATER
            )
            in_mob = is_in_mob(state, proposed_position)
        else:
            in_wall = True
            in_mob = False

        continue_move = in_bounds_arrow and not in_wall and not in_mob
        position = proposed_position
        new_mask = continue_move  # arrows.mask[i] is already True here.

        if hit_player:
            state.player_health -= 2
            state.is_sleeping = False

        if in_bounds_arrow:
            block_here = int(state.map[position[0], position[1]])
            hit_bench_or_furnace = block_here in (
                BlockType.FURNACE,
                BlockType.CRAFTING_TABLE,
            )
            if hit_bench_or_furnace:
                state.map[position[0], position[1]] = BlockType.PATH

        arrows.position[i] = position
        arrows.mask[i] = new_mask


def update_mobs(
    rng: np.random.Generator,
    state: EnvState,
    params: EnvParams,
    static_params: StaticEnvParams,
) -> EnvState:
    _move_zombies(rng, state, params, static_params)
    _move_cows(rng, state, params, static_params)
    _move_skeletons(rng, state, params, static_params)
    _move_arrows(rng, state, params, static_params)
    return state


def get_distance_map(position: np.ndarray, map_size: Tuple[int, int]) -> np.ndarray:
    h, w = map_size
    dist_x = np.abs(np.arange(h) - int(position[0]))[:, None]
    dist_x = np.tile(dist_x, (1, w))

    dist_y = np.abs(np.arange(w) - int(position[1]))[None, :]
    dist_y = np.tile(dist_y, (h, 1))

    return dist_x + dist_y


def _spawn_one(
    rng: np.random.Generator,
    state: EnvState,
    mobs: Mobs,
    can_spawn_base: bool,
    spawn_chance: float,
    candidate_map: np.ndarray,
    health: int,
) -> None:
    can_spawn = can_spawn_base and rng.random() < spawn_chance
    if not can_spawn:
        return

    coords = np.argwhere(candidate_map)
    if len(coords) == 0:
        return

    position = coords[rng.integers(len(coords))]
    slot = int(np.argmax(~mobs.mask))

    mobs.position[slot] = position
    mobs.health[slot] = health
    mobs.mask[slot] = True
    state.mob_map[position[0], position[1]] = True


def spawn_mobs(
    state: EnvState,
    rng: np.random.Generator,
    params: EnvParams,
    static_params: StaticEnvParams,
) -> EnvState:
    player_distance_map = get_distance_map(state.player_position, static_params.map_size)

    cows_can_spawn_map = (
        (state.map == BlockType.GRASS)
        & (player_distance_map > 3)
        & (player_distance_map < params.mob_despawn_distance)
        & (~state.mob_map)
    )
    can_spawn_cow = int(state.cows.mask.sum()) < static_params.max_cows
    _spawn_one(
        rng,
        state,
        state.cows,
        can_spawn_cow,
        params.spawn_cow_chance,
        cows_can_spawn_map,
        params.cow_health,
    )

    zombie_spawn_chance = (
        params.spawn_zombie_base_chance
        + params.spawn_zombie_night_chance * (1 - state.light_level) ** 2
    )
    zombies_can_spawn_map = (
        ((state.map == BlockType.GRASS) | (state.map == BlockType.PATH))
        & (player_distance_map > 9)
        & (player_distance_map < params.mob_despawn_distance)
        & (~state.mob_map)
    )
    can_spawn_zombie = int(state.zombies.mask.sum()) < static_params.max_zombies
    _spawn_one(
        rng,
        state,
        state.zombies,
        can_spawn_zombie,
        zombie_spawn_chance,
        zombies_can_spawn_map,
        params.zombie_health,
    )

    skeletons_can_spawn_map = (
        (state.map == BlockType.PATH)
        & (player_distance_map > 9)
        & (player_distance_map < params.mob_despawn_distance)
        & (~state.mob_map)
    )
    can_spawn_skeleton = int(state.skeletons.mask.sum()) < static_params.max_skeletons
    _spawn_one(
        rng,
        state,
        state.skeletons,
        can_spawn_skeleton,
        params.spawn_skeleton_chance,
        skeletons_can_spawn_map,
        params.skeleton_health,
    )

    return state


# ---------------------------------------------------------------------------
# Plants, intrinsics, health, movement
# ---------------------------------------------------------------------------


def update_plants(state: EnvState, static_params: StaticEnvParams) -> EnvState:
    for i in range(static_params.max_growing_plants):
        if state.growing_plants_mask[i]:
            state.growing_plants_age[i] += 1
            if state.growing_plants_age[i] >= 600:
                pos = state.growing_plants_positions[i]
                state.map[pos[0], pos[1]] = BlockType.RIPE_PLANT
        else:
            state.growing_plants_age[i] = 0
    return state


def update_player_intrinsics(state: EnvState, action: int) -> EnvState:
    # Start sleeping?
    if action == Action.SLEEP and state.player_energy < 9:
        state.is_sleeping = True

    # Wake up?
    if state.player_energy >= 9 and state.is_sleeping:
        state.is_sleeping = False
        state.achievements[Achievement.WAKE_UP] = True

    # Hunger
    hunger_add = 0.5 if state.is_sleeping else 1.0
    new_hunger = state.player_hunger + hunger_add
    if new_hunger > 25:
        state.player_food = max(state.player_food - 1, 0)
        new_hunger = 0.0
    state.player_hunger = new_hunger

    # Thirst
    thirst_add = 0.5 if state.is_sleeping else 1.0
    new_thirst = state.player_thirst + thirst_add
    if new_thirst > 20:
        state.player_drink = max(state.player_drink - 1, 0)
        new_thirst = 0.0
    state.player_thirst = new_thirst

    # Fatigue
    if state.is_sleeping:
        new_fatigue = min(state.player_fatigue - 1, 0)
    else:
        new_fatigue = state.player_fatigue + 1

    if new_fatigue > 30:
        state.player_energy = max(state.player_energy - 1, 0)
        new_fatigue = 0.0
    if new_fatigue < -10:
        state.player_energy = min(state.player_energy + 1, 9)
        new_fatigue = 0.0
    state.player_fatigue = new_fatigue

    # Health
    all_necessities = (
        state.player_food > 0
        and state.player_drink > 0
        and (state.player_energy > 0 or state.is_sleeping)
    )
    if state.is_sleeping:
        recover_add = 2.0 if all_necessities else -0.5
    else:
        recover_add = 1.0 if all_necessities else -1.0

    new_recover = state.player_recover + recover_add
    if new_recover > 25:
        state.player_health = min(state.player_health + 1, 9)
        new_recover = 0.0
    if new_recover < -15:
        state.player_health = state.player_health - 1
        new_recover = 0.0
    state.player_recover = new_recover

    return state


def move_player(state: EnvState, action: int) -> EnvState:
    proposed_position = state.player_position + DIRECTIONS[action]

    valid_move = is_position_in_bounds_not_in_wall_not_in_mob_not_in_lava(
        state, proposed_position
    )
    if not valid_move and in_bounds(state, proposed_position):
        # Walking into lava is allowed (it's instantly lethal via
        # `update_health`, but movement itself isn't blocked).
        if state.map[proposed_position[0], proposed_position[1]] == BlockType.LAVA:
            valid_move = True

    if valid_move:
        state.player_position = proposed_position.astype(np.int32)

    is_new_direction = bool(np.abs(DIRECTIONS[action]).sum() != 0)
    if is_new_direction:
        state.player_direction = int(action)

    return state


def cap_inventory(state: EnvState) -> EnvState:
    inv = state.inventory
    inv.wood = min(inv.wood, 9)
    inv.stone = min(inv.stone, 9)
    inv.coal = min(inv.coal, 9)
    inv.iron = min(inv.iron, 9)
    inv.diamond = min(inv.diamond, 9)
    inv.sapling = min(inv.sapling, 9)
    inv.wood_pickaxe = min(inv.wood_pickaxe, 9)
    inv.stone_pickaxe = min(inv.stone_pickaxe, 9)
    inv.iron_pickaxe = min(inv.iron_pickaxe, 9)
    inv.wood_sword = min(inv.wood_sword, 9)
    inv.stone_sword = min(inv.stone_sword, 9)
    inv.iron_sword = min(inv.iron_sword, 9)
    return state


def update_health(state: EnvState) -> EnvState:
    in_lava = (
        state.map[state.player_position[0], state.player_position[1]]
        == BlockType.LAVA
    )
    if in_lava:
        state.player_health = 0
    # Cap health at 0 (avoids a spurious negative-health reward penalty; the
    # player dies either way once health reaches 0).
    state.player_health = max(0, state.player_health)
    return state


# ---------------------------------------------------------------------------
# Top-level step function
# ---------------------------------------------------------------------------


def craftax_step(
    state: EnvState,
    action: int,
    params: EnvParams,
    static_params: StaticEnvParams,
    rng: np.random.Generator,
) -> Tuple[EnvState, float]:
    init_achievement_count = int(state.achievements.sum())
    init_health = state.player_health

    # Interrupt action if sleeping.
    if state.is_sleeping:
        action = Action.NOOP

    # Crafting
    state = do_crafting(state, action)

    # Interact (mining, attacking, eating plants, drinking water)
    state = do_action(rng, state, action, static_params)

    # Placing
    state = place_block(state, action, static_params)

    # Movement
    state = move_player(state, action)

    # Mobs
    state = update_mobs(rng, state, params, static_params)
    state = spawn_mobs(state, rng, params, static_params)

    # Plants
    state = update_plants(state, static_params)

    # Intrinsics
    state = update_player_intrinsics(state, action)

    # Cap inventory
    state = cap_inventory(state)

    # Cap and manage health
    state = update_health(state)

    # Reward
    achievement_reward = float(int(state.achievements.sum()) - init_achievement_count)
    health_reward = (state.player_health - init_health) * 0.1
    reward = achievement_reward + health_reward

    state.timestep += 1
    state.light_level = calculate_light_level(state.timestep, params)

    return state, reward
