"""Pixel renderer for Craftax-Classic using the official tile/sprite assets.

Mirrors the layout of ``reference/craftax/.../craftax_classic/renderer.py``
(map view + 2-row inventory HUD) without JAX. Textures are loaded via pygame.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .constants import OBS_DIM, BlockType
from .state import EnvState

ASSETS_DIR = Path(__file__).resolve().parent / "assets"
INVENTORY_OBS_HEIGHT = 2
BLOCK_PIXEL_SIZE_HUMAN = 64

BLOCK_TEXTURE_NAMES = [
    "debug_tile.png",  # INVALID
    "debug_tile.png",  # OUT_OF_BOUNDS (replaced with grey)
    "grass.png",
    "water.png",
    "stone.png",
    "tree.png",
    "wood.png",
    "path.png",
    "coal.png",
    "iron.png",
    "diamond.png",
    "table.png",
    "furnace.png",
    "sand.png",
    "lava.png",
    "plant_on_grass.png",
    "ripe_plant_on_grass.png",
]


def _require_pygame():
    try:
        import pygame
    except ImportError as exc:
        raise RuntimeError(
            "pygame is required for Craftax pixel rendering. "
            "Install with `pip install pygame`."
        ) from exc
    return pygame


def _load_surface(pygame: Any, name: str, size: int) -> Any:
    path = ASSETS_DIR / name
    if not path.is_file():
        raise FileNotFoundError(f"Missing Craftax asset: {path}")
    surf = pygame.image.load(str(path)).convert_alpha()
    if surf.get_width() != size or surf.get_height() != size:
        surf = pygame.transform.scale(surf, (size, size))
    return surf


def _scale_nearest(pygame: Any, surf: Any, size: int) -> Any:
    if surf.get_width() == size and surf.get_height() == size:
        return surf
    return pygame.transform.scale(surf, (size, size))


class CraftaxTextures:
    """Cached textures at a fixed block pixel size."""

    def __init__(self, block_pixel_size: int = BLOCK_PIXEL_SIZE_HUMAN) -> None:
        pygame = _require_pygame()
        if not pygame.get_init():
            pygame.init()
        if pygame.display.get_surface() is None:
            # convert_alpha() needs an active display; prefer a hidden window.
            flags = getattr(pygame, "HIDDEN", 0)
            pygame.display.set_mode((1, 1), flags=flags)

        self.block_pixel_size = block_pixel_size
        small = max(1, int(block_pixel_size * 0.8))
        number_size = max(1, int(block_pixel_size * 0.6))

        self.blocks: list[Any] = []
        for i, name in enumerate(BLOCK_TEXTURE_NAMES):
            if i == int(BlockType.OUT_OF_BOUNDS):
                grey = pygame.Surface((block_pixel_size, block_pixel_size))
                grey.fill((128, 128, 128))
                self.blocks.append(grey)
            else:
                self.blocks.append(_load_surface(pygame, name, block_pixel_size))

        self.blocks_small: list[Any] = [
            _scale_nearest(pygame, b, small) for b in self.blocks
        ]

        self.player = [
            _load_surface(pygame, "player-left.png", block_pixel_size),
            _load_surface(pygame, "player-right.png", block_pixel_size),
            _load_surface(pygame, "player-up.png", block_pixel_size),
            _load_surface(pygame, "player-down.png", block_pixel_size),
            _load_surface(pygame, "player-sleep.png", block_pixel_size),
        ]
        self.zombie = _load_surface(pygame, "zombie.png", block_pixel_size)
        self.cow = _load_surface(pygame, "cow.png", block_pixel_size)
        self.skeleton = _load_surface(pygame, "skeleton.png", block_pixel_size)
        self.arrow_up = _load_surface(pygame, "arrow-up.png", block_pixel_size)

        self.health = _load_surface(pygame, "health.png", small)
        self.food = _load_surface(pygame, "food.png", small)
        self.drink = _load_surface(pygame, "drink.png", small)
        self.energy = _load_surface(pygame, "energy.png", small)
        self.sapling = _load_surface(pygame, "sapling.png", small)
        self.wood_pickaxe = _load_surface(pygame, "wood_pickaxe.png", small)
        self.stone_pickaxe = _load_surface(pygame, "stone_pickaxe.png", small)
        self.iron_pickaxe = _load_surface(pygame, "iron_pickaxe.png", small)
        self.wood_sword = _load_surface(pygame, "wood_sword.png", small)
        self.stone_sword = _load_surface(pygame, "stone_sword.png", small)
        self.iron_sword = _load_surface(pygame, "iron_sword.png", small)

        self.numbers: list[Any | None] = [None]
        for n in range(1, 10):
            self.numbers.append(_load_surface(pygame, f"{n}.png", number_size))

        self.small = small
        self.number_size = number_size
        self.empty_small = pygame.Surface((small, small), pygame.SRCALPHA)


class CraftaxPixelRenderer:
    """Renders EnvState to a pygame Surface matching Classic pixel obs layout."""

    def __init__(self, block_pixel_size: int = BLOCK_PIXEL_SIZE_HUMAN) -> None:
        self.pygame = _require_pygame()
        self.bps = block_pixel_size
        self.textures = CraftaxTextures(block_pixel_size)
        self.map_h = OBS_DIM[0] * block_pixel_size
        self.map_w = OBS_DIM[1] * block_pixel_size
        self.inv_h = INVENTORY_OBS_HEIGHT * block_pixel_size
        self.width = self.map_w
        self.height = self.map_h + self.inv_h

    def render(self, state: EnvState) -> Any:
        pygame = self.pygame
        bps = self.bps
        tex = self.textures
        obs_h, obs_w = OBS_DIM
        half = np.array([obs_h // 2, obs_w // 2], dtype=np.int32)

        canvas = pygame.Surface((self.width, self.height))
        canvas.fill((0, 0, 0))

        pr, pc = int(state.player_position[0]), int(state.player_position[1])
        mh, mw = state.map.shape

        # Map tiles
        for r in range(obs_h):
            for c in range(obs_w):
                mr, mc = pr - half[0] + r, pc - half[1] + c
                if 0 <= mr < mh and 0 <= mc < mw:
                    block = int(state.map[mr, mc])
                else:
                    block = int(BlockType.OUT_OF_BOUNDS)
                block = max(0, min(block, len(tex.blocks) - 1))
                canvas.blit(tex.blocks[block], (c * bps, r * bps))

        # Mobs (drawn under player)
        self._blit_mobs(canvas, state, state.zombies, tex.zombie, half)
        self._blit_mobs(canvas, state, state.cows, tex.cow, half)
        self._blit_mobs(canvas, state, state.skeletons, tex.skeleton, half)
        self._blit_arrows(canvas, state, half)

        # Player at center
        if state.is_sleeping:
            player_idx = 4
        else:
            # player_direction is Action id: LEFT=1 .. DOWN=4
            player_idx = int(np.clip(state.player_direction - 1, 0, 3))
        canvas.blit(tex.player[player_idx], (half[1] * bps, half[0] * bps))

        # Night / sleep overlays on map region
        daylight = float(np.clip(state.light_level, 0.0, 1.0))
        if daylight < 0.5:
            night = pygame.Surface((self.map_w, self.map_h), pygame.SRCALPHA)
            alpha = int(140 * (0.5 - daylight) / 0.5)
            night.fill((0, 16, 64, alpha))
            canvas.blit(night, (0, 0))
        if state.is_sleeping:
            sleep = pygame.Surface((self.map_w, self.map_h), pygame.SRCALPHA)
            sleep.fill((0, 0, 16, 110))
            canvas.blit(sleep, (0, 0))

        # Inventory HUD
        inv_y0 = self.map_h
        pygame.draw.rect(canvas, (20, 20, 24), (0, inv_y0, self.map_w, self.inv_h))
        self._draw_inventory(canvas, state, inv_y0)

        return canvas

    def _local_pos(self, world_pos: np.ndarray, player_pos: np.ndarray, half: np.ndarray) -> np.ndarray:
        return world_pos.astype(np.int32) - player_pos.astype(np.int32) + half

    def _blit_mobs(self, canvas: Any, state: EnvState, mobs: Any, texture: Any, half: np.ndarray) -> None:
        obs_h, obs_w = OBS_DIM
        bps = self.bps
        for i in range(mobs.mask.shape[0]):
            if not mobs.mask[i]:
                continue
            local = self._local_pos(mobs.position[i], state.player_position, half)
            if 0 <= local[0] < obs_h and 0 <= local[1] < obs_w:
                canvas.blit(texture, (int(local[1]) * bps, int(local[0]) * bps))

    def _blit_arrows(self, canvas: Any, state: EnvState, half: np.ndarray) -> None:
        pygame = self.pygame
        obs_h, obs_w = OBS_DIM
        bps = self.bps
        tex = self.textures.arrow_up
        for i in range(state.arrows.mask.shape[0]):
            if not state.arrows.mask[i]:
                continue
            local = self._local_pos(state.arrows.position[i], state.player_position, half)
            if not (0 <= local[0] < obs_h and 0 <= local[1] < obs_w):
                continue
            dr, dc = int(state.arrow_directions[i, 0]), int(state.arrow_directions[i, 1])
            arrow = tex
            if dc != 0:
                arrow = pygame.transform.rotate(tex, -90 if dc > 0 else 90)
            elif dr > 0:
                arrow = pygame.transform.flip(tex, False, True)
            canvas.blit(arrow, (int(local[1]) * bps, int(local[0]) * bps))

    def _blit_icon_with_count(
        self,
        canvas: Any,
        icon: Any | None,
        count: int,
        cell_x: int,
        cell_y: int,
        inv_y0: int,
    ) -> None:
        pygame = self.pygame
        bps = self.bps
        tex = self.textures
        pad = (bps - tex.small) // 2
        x = cell_x * bps + pad
        y = inv_y0 + cell_y * bps + pad
        if icon is not None and count > 0:
            canvas.blit(icon, (x, y))
        # Digit overlay (Classic clamps display to 1..9)
        n = int(np.clip(count, 0, 9))
        if n > 0 and tex.numbers[n] is not None:
            number = tex.numbers[n]
            nx = (cell_x + 1) * bps - tex.number_size - 1
            ny = inv_y0 + (cell_y + 1) * bps - tex.number_size - 1
            canvas.blit(number, (nx, ny))

    def _draw_inventory(self, canvas: Any, state: EnvState, inv_y0: int) -> None:
        tex = self.textures
        inv = state.inventory

        # Row 0: health, food, drink, energy, sapling, wood, stone, coal, iron
        self._blit_icon_with_count(canvas, tex.health, state.player_health, 0, 0, inv_y0)
        self._blit_icon_with_count(canvas, tex.food, state.player_food, 1, 0, inv_y0)
        self._blit_icon_with_count(canvas, tex.drink, state.player_drink, 2, 0, inv_y0)
        self._blit_icon_with_count(canvas, tex.energy, state.player_energy, 3, 0, inv_y0)
        self._blit_icon_with_count(canvas, tex.sapling, inv.sapling, 4, 0, inv_y0)
        self._blit_icon_with_count(canvas, tex.blocks_small[int(BlockType.WOOD)], inv.wood, 5, 0, inv_y0)
        self._blit_icon_with_count(canvas, tex.blocks_small[int(BlockType.STONE)], inv.stone, 6, 0, inv_y0)
        self._blit_icon_with_count(canvas, tex.blocks_small[int(BlockType.COAL)], inv.coal, 7, 0, inv_y0)
        self._blit_icon_with_count(canvas, tex.blocks_small[int(BlockType.IRON)], inv.iron, 8, 0, inv_y0)

        # Row 1: diamond, pickaxes, swords
        self._blit_icon_with_count(
            canvas, tex.blocks_small[int(BlockType.DIAMOND)], inv.diamond, 0, 1, inv_y0
        )
        self._blit_icon_with_count(canvas, tex.wood_pickaxe, inv.wood_pickaxe, 1, 1, inv_y0)
        self._blit_icon_with_count(canvas, tex.stone_pickaxe, inv.stone_pickaxe, 2, 1, inv_y0)
        self._blit_icon_with_count(canvas, tex.iron_pickaxe, inv.iron_pickaxe, 3, 1, inv_y0)
        self._blit_icon_with_count(canvas, tex.wood_sword, inv.wood_sword, 4, 1, inv_y0)
        self._blit_icon_with_count(canvas, tex.stone_sword, inv.stone_sword, 5, 1, inv_y0)
        self._blit_icon_with_count(canvas, tex.iron_sword, inv.iron_sword, 6, 1, inv_y0)


class CraftaxPygameViewer:
    """Windowed viewer around CraftaxPixelRenderer with a status header."""

    def __init__(self, block_pixel_size: int = 48, header_h: int = 80, scale: int = 1) -> None:
        pygame = _require_pygame()
        self.pygame = pygame
        if not pygame.get_init():
            pygame.init()
        self.header_h = header_h
        self.scale = max(1, scale)
        # Open the real window before loading textures (convert_alpha needs a display).
        map_w = OBS_DIM[1] * block_pixel_size
        map_h = (OBS_DIM[0] + INVENTORY_OBS_HEIGHT) * block_pixel_size
        w = map_w * self.scale
        h = header_h + map_h * self.scale
        self.screen = pygame.display.set_mode((w, h))
        pygame.display.set_caption("Craftax-Classic")
        self.renderer = CraftaxPixelRenderer(block_pixel_size=block_pixel_size)
        self.bg = (18, 18, 22)
        self.text = (235, 235, 240)
        self.title_font = pygame.font.SysFont("arial", 22, bold=True)
        self.info_font = pygame.font.SysFont("arial", 16)

    def draw(
        self,
        state: EnvState,
        *,
        score: int,
        health: int,
        steps: int,
        done: bool,
        action_name: str | None = None,
        delay: float | None = None,
    ) -> None:
        pygame = self.pygame
        self.screen.fill(self.bg)
        title = self.title_font.render("Craftax-Classic", True, self.text)
        self.screen.blit(title, (12, 8))
        info = self.info_font.render(
            f"Achievements {score}/22   HP {health}   Steps {steps}"
            + ("   DONE" if done else ""),
            True,
            self.text,
        )
        self.screen.blit(info, (12, 34))
        action_bits: list[str] = []
        if action_name:
            action_bits.append(f"Action: {action_name}")
        if delay is not None:
            action_bits.append(f"{delay:.2f}s/step  [ / ] slower/faster  Space pause")
        if action_bits:
            action_line = self.info_font.render("   ".join(action_bits), True, self.text)
            self.screen.blit(action_line, (12, 54))

        frame = self.renderer.render(state)
        if self.scale != 1:
            frame = pygame.transform.scale(
                frame,
                (self.renderer.width * self.scale, self.renderer.height * self.scale),
            )
        self.screen.blit(frame, (0, self.header_h))
        pygame.display.flip()
