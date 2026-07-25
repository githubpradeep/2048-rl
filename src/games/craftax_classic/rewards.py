"""Optional training-time reward shaping for Craftax-Classic.

The Classic env reward is sparse: ``Δachievements + 0.1·Δhealth``. That is fine
as an evaluation metric but makes vanilla DQN collapse into random walks that
starve. These helpers add dense survival / resource signals used only when
``CraftaxClassicConfig.reward_shaping`` is enabled. Evaluation of skill should
still use achievement count (``info['score']``).
"""

from __future__ import annotations

from dataclasses import dataclass

from .constants import MOVE_OPPOSITE
from .state import EnvState, Inventory


@dataclass(frozen=True)
class CraftaxRewardShaping:
    """Scales for dense auxiliary rewards (added on top of Classic reward)."""

    survive_bonus: float = 0.0002
    death_penalty: float = 1.0
    resource_scale: float = 0.1
    vital_gain_scale: float = 0.05
    low_vital_penalty: float = 0.01
    # Penalize immediate cardinal reversals (UP↔DOWN / LEFT↔RIGHT ping-pong).
    reverse_penalty: float = 0.02


@dataclass
class VitalSnapshot:
    """Pre-step vitals/inventory (needed because ``craftax_step`` mutates in place)."""

    health: int
    food: int
    drink: int
    energy: int
    is_sleeping: bool
    inventory: Inventory
    last_move: int | None = None


def snapshot_vitals(state: EnvState, last_move: int | None = None) -> VitalSnapshot:
    inv = state.inventory
    return VitalSnapshot(
        health=int(state.player_health),
        food=int(state.player_food),
        drink=int(state.player_drink),
        energy=int(state.player_energy),
        is_sleeping=bool(state.is_sleeping),
        inventory=Inventory(
            wood=int(inv.wood),
            stone=int(inv.stone),
            coal=int(inv.coal),
            iron=int(inv.iron),
            diamond=int(inv.diamond),
            sapling=int(inv.sapling),
            wood_pickaxe=int(inv.wood_pickaxe),
            stone_pickaxe=int(inv.stone_pickaxe),
            iron_pickaxe=int(inv.iron_pickaxe),
            wood_sword=int(inv.wood_sword),
            stone_sword=int(inv.stone_sword),
            iron_sword=int(inv.iron_sword),
        ),
        last_move=last_move,
    )


def _inv_delta(prev: Inventory, cur: Inventory) -> float:
    return float(
        (cur.wood - prev.wood)
        + (cur.stone - prev.stone)
        + (cur.coal - prev.coal)
        + (cur.iron - prev.iron)
        + (cur.diamond - prev.diamond)
        + (cur.sapling - prev.sapling)
        + (cur.wood_pickaxe - prev.wood_pickaxe)
        + (cur.stone_pickaxe - prev.stone_pickaxe)
        + (cur.iron_pickaxe - prev.iron_pickaxe)
        + (cur.wood_sword - prev.wood_sword)
        + (cur.stone_sword - prev.stone_sword)
        + (cur.iron_sword - prev.iron_sword)
    )


def shape_craftax_reward(
    prev: VitalSnapshot,
    cur: EnvState,
    base_reward: float,
    done: bool,
    shaping: CraftaxRewardShaping,
    action: int | None = None,
) -> float:
    """Return Classic reward plus dense survival/resource terms."""
    reward = float(base_reward)

    # Repeatable resource / tool gains (first-time unlocks already in base_reward).
    gained = _inv_delta(prev.inventory, cur.inventory)
    if gained > 0:
        reward += shaping.resource_scale * gained

    # Recovering food/drink/energy is critical; ignore decreases (already pressured via HP).
    food_gain = max(0, int(cur.player_food) - int(prev.food))
    drink_gain = max(0, int(cur.player_drink) - int(prev.drink))
    energy_gain = max(0, int(cur.player_energy) - int(prev.energy))
    reward += shaping.vital_gain_scale * float(food_gain + drink_gain + energy_gain)

    # Soft pressure when vitals are empty (encourages drink/food/sleep before HP drops).
    if int(cur.player_food) <= 0:
        reward -= shaping.low_vital_penalty
    if int(cur.player_drink) <= 0:
        reward -= shaping.low_vital_penalty
    if int(cur.player_energy) <= 0 and not cur.is_sleeping:
        reward -= shaping.low_vital_penalty

    # Immediate reverse of the previous cardinal move (ping-pong).
    if (
        action is not None
        and prev.last_move is not None
        and MOVE_OPPOSITE.get(int(prev.last_move)) == int(action)
    ):
        reward -= shaping.reverse_penalty

    if not done:
        reward += shaping.survive_bonus
    elif int(cur.player_health) <= 0:
        reward -= shaping.death_penalty

    return float(reward)
