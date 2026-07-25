"""Craftax-Classic game engine and RL env wrapper (pure NumPy)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .constants import MOVE_OPPOSITE, NUM_ACTIONS, BlockType
from .logic import craftax_step, is_game_over, legal_actions_for_state
from .obs import OBS_SIZE, encode_symbolic
from .rewards import CraftaxRewardShaping, shape_craftax_reward, snapshot_vitals
from .state import EnvParams, EnvState, StaticEnvParams
from .world_gen import generate_world

BLOCK_CHARS = {
    BlockType.INVALID: "?",
    BlockType.OUT_OF_BOUNDS: "#",
    BlockType.GRASS: ".",
    BlockType.WATER: "~",
    BlockType.STONE: "#",
    BlockType.TREE: "T",
    BlockType.WOOD: "=",
    BlockType.PATH: ",",
    BlockType.COAL: "c",
    BlockType.IRON: "i",
    BlockType.DIAMOND: "*",
    BlockType.CRAFTING_TABLE: "C",
    BlockType.FURNACE: "F",
    BlockType.SAND: ":",
    BlockType.LAVA: "!",
    BlockType.PLANT: "p",
    BlockType.RIPE_PLANT: "P",
}


@dataclass
class CraftaxClassicConfig:
    map_size: tuple[int, int] = (64, 64)
    max_timesteps: int = 10000
    day_length: int = 300
    always_diamond: bool = True
    # Training-time dense rewards (eval skill still uses achievement count).
    reward_shaping: bool = False
    survive_bonus: float = 0.0002
    death_penalty: float = 1.0
    resource_scale: float = 0.1
    vital_gain_scale: float = 0.05
    low_vital_penalty: float = 0.01
    reverse_penalty: float = 0.02
    # Drop the opposite of the last cardinal move from legal_actions (anti-oscillation).
    block_reverse_moves: bool = True


class CraftaxClassicGame:
    """Pure Craftax-Classic simulation with deterministic RNG support."""

    def __init__(
        self,
        config: CraftaxClassicConfig | None = None,
        seed: int | None = None,
    ) -> None:
        self.config = config or CraftaxClassicConfig()
        self.rng = np.random.default_rng(seed)
        self.params = EnvParams(
            max_timesteps=self.config.max_timesteps,
            day_length=self.config.day_length,
            always_diamond=self.config.always_diamond,
        )
        self.static_params = StaticEnvParams(map_size=self.config.map_size)
        self.state: EnvState | None = None
        self.last_reward = 0.0
        self.last_raw_reward = 0.0
        self.game_over = False
        self._last_move: int | None = None
        self._shaping = CraftaxRewardShaping(
            survive_bonus=float(self.config.survive_bonus),
            death_penalty=float(self.config.death_penalty),
            resource_scale=float(self.config.resource_scale),
            vital_gain_scale=float(self.config.vital_gain_scale),
            low_vital_penalty=float(self.config.low_vital_penalty),
            reverse_penalty=float(self.config.reverse_penalty),
        )

    @property
    def action_size(self) -> int:
        return NUM_ACTIONS

    def reset(self, seed: int | None = None) -> EnvState:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.state = generate_world(self.rng, self.params, self.static_params)
        self.last_reward = 0.0
        self.last_raw_reward = 0.0
        self.game_over = False
        self._last_move = None
        return self.state

    def legal_actions(self) -> list[int]:
        if self.state is None:
            return list(range(NUM_ACTIONS))
        legal = legal_actions_for_state(self.state)
        if self.config.block_reverse_moves and self._last_move is not None:
            opposite = MOVE_OPPOSITE.get(int(self._last_move))
            if opposite is not None:
                filtered = [a for a in legal if a != opposite]
                if filtered:
                    return filtered
        return legal

    def step(self, action: int) -> tuple[EnvState, float, bool]:
        if self.state is None:
            self.reset()
        assert self.state is not None
        if self.game_over:
            return self.state, 0.0, True

        action_i = int(action)
        if action_i < 0 or action_i >= NUM_ACTIONS:
            raise ValueError(f"Invalid action {action}")

        prev = snapshot_vitals(self.state, last_move=self._last_move)
        self.state, raw_reward = craftax_step(
            self.state, action_i, self.params, self.static_params, self.rng
        )
        done = bool(is_game_over(self.state, self.params))
        self.game_over = done
        reward = float(raw_reward)
        self.last_raw_reward = reward
        if self.config.reward_shaping:
            reward = shape_craftax_reward(
                prev, self.state, reward, done, self._shaping, action=action_i
            )
        if action_i in MOVE_OPPOSITE:
            self._last_move = action_i
        self.last_reward = float(reward)
        return self.state, float(reward), done

    def achievement_count(self) -> int:
        if self.state is None:
            return 0
        return int(self.state.achievements.sum())

    def render(self, view: int = 7) -> str:
        if self.state is None:
            return "<uninitialized>"
        st = self.state
        pr, pc = int(st.player_position[0]), int(st.player_position[1])
        half = view // 2
        h, w = st.map.shape
        lines: list[str] = []
        for r in range(pr - half, pr + half + 1):
            row: list[str] = []
            for c in range(pc - half, pc + half + 1):
                if r == pr and c == pc:
                    row.append("@")
                    continue
                if not (0 <= r < h and 0 <= c < w):
                    row.append(" ")
                    continue
                if st.mob_map[r, c]:
                    ch = "Z"
                    for i in range(len(st.cows.mask)):
                        if st.cows.mask[i] and st.cows.position[i][0] == r and st.cows.position[i][1] == c:
                            ch = "o"
                    for i in range(len(st.skeletons.mask)):
                        if (
                            st.skeletons.mask[i]
                            and st.skeletons.position[i][0] == r
                            and st.skeletons.position[i][1] == c
                        ):
                            ch = "S"
                    for i in range(len(st.arrows.mask)):
                        if (
                            st.arrows.mask[i]
                            and st.arrows.position[i][0] == r
                            and st.arrows.position[i][1] == c
                        ):
                            ch = "-"
                    row.append(ch)
                    continue
                bt = BlockType(int(st.map[r, c]))
                row.append(BLOCK_CHARS.get(bt, "?"))
            lines.append("".join(row))

        inv = st.inventory
        lines.append(
            f"HP={st.player_health} Food={st.player_food} Drink={st.player_drink} "
            f"Energy={st.player_energy} Sleep={int(st.is_sleeping)}"
        )
        lines.append(
            f"Inv wood={inv.wood} stone={inv.stone} coal={inv.coal} iron={inv.iron} "
            f"dia={inv.diamond} sap={inv.sapling} "
            f"picks={inv.wood_pickaxe}/{inv.stone_pickaxe}/{inv.iron_pickaxe} "
            f"swords={inv.wood_sword}/{inv.stone_sword}/{inv.iron_sword}"
        )
        lines.append(
            f"Achievements={self.achievement_count()}/22  "
            f"t={st.timestep}/{self.params.max_timesteps}  "
            f"light={st.light_level:.2f}  reward={self.last_reward:.2f}"
        )
        return "\n".join(lines)


class CraftaxClassicEnv:
    """RL wrapper: symbolic float32 obs (1345,), 17 actions."""

    def __init__(
        self,
        config: CraftaxClassicConfig | None = None,
        seed: int | None = None,
    ) -> None:
        self.game = CraftaxClassicGame(config=config, seed=seed)
        self.action_size = NUM_ACTIONS
        self.state_dim = OBS_SIZE

    def reset(self, seed: int | None = None) -> np.ndarray:
        self.game.reset(seed=seed)
        return self.get_state()

    def legal_actions(self) -> list[int]:
        return self.game.legal_actions()

    def get_state(self) -> np.ndarray:
        if self.game.state is None:
            self.game.reset()
        assert self.game.state is not None
        return encode_symbolic(self.game.state)

    def step(self, action: int) -> tuple[np.ndarray, float, bool, dict[str, Any]]:
        _, reward, done = self.game.step(action)
        st = self.game.state
        assert st is not None
        inv = st.inventory
        info = {
            "score": int(self.game.achievement_count()),
            "achievements": int(self.game.achievement_count()),
            "health": int(st.player_health),
            "food": int(st.player_food),
            "drink": int(st.player_drink),
            "energy": int(st.player_energy),
            "timestep": int(st.timestep),
            "is_sleeping": bool(st.is_sleeping),
            "light_level": float(st.light_level),
            "raw_reward": float(self.game.last_raw_reward),
            "wood": int(inv.wood),
            "stone": int(inv.stone),
            "coal": int(inv.coal),
            "iron": int(inv.iron),
            "diamond": int(inv.diamond),
        }
        return self.get_state(), float(reward), bool(done), info

    def render(self) -> str:
        return self.game.render()
