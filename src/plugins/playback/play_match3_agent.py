from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

from ...games.match3 import Match3Config, Match3Env
from ...network import MLPQNetwork


@dataclass
class PlayResult:
    score: int
    steps: int
    tiles: int
    cascades: int


def pick_action(network: MLPQNetwork, state: object, legal_actions: list[int]) -> int:
    q_values = network.predict_one(state)
    if not legal_actions:
        return max(range(len(q_values)), key=q_values.__getitem__)
    return max(legal_actions, key=lambda action: q_values[action])


def run_terminal(env: Match3Env, network: MLPQNetwork, seed: int, delay: float, max_steps: int) -> PlayResult:
    state = env.reset(seed=seed)
    done = False
    steps = 0
    info = {"score": 0, "total_tiles_cleared": 0, "total_cascades": 0}

    print("Starting Match-3 autoplay in terminal mode...\n")
    while not done and steps < max_steps:
        legal = env.legal_actions()
        action = pick_action(network, state, legal)
        swap = env.action_to_swap(action)
        state, reward, done, info = env.step(action)
        steps += 1

        print(f"Step {steps} | Action {action} swap={swap} | Reward {reward:.2f} | Legal={len(legal)}")
        print(env.render())
        print("=" * 70)
        if delay > 0:
            time.sleep(delay)

    return PlayResult(
        score=int(info["score"]),
        steps=steps,
        tiles=int(info["total_tiles_cleared"]),
        cascades=int(info["total_cascades"]),
    )


class Match3PygameRenderer:
    def __init__(self, width: int = 6, height: int = 6, num_colors: int = 5, cell_size: int = 56, margin: int = 8, header_h: int = 96) -> None:
        import pygame

        self.pygame = pygame
        self.w = width
        self.h = height
        self.num_colors = num_colors
        self.cell_size = cell_size
        self.margin = margin
        self.header_h = header_h

        canvas_w = margin + width * (cell_size + margin)
        canvas_h = header_h + margin + height * (cell_size + margin)
        self.screen = pygame.display.set_mode((canvas_w, canvas_h))
        pygame.display.set_caption("Match-3 RL Autoplay")

        self.bg = (246, 248, 252)
        self.board_bg = (226, 233, 244)
        self.text = (36, 48, 66)
        self.palette = [
            (230, 88, 86),
            (63, 121, 230),
            (74, 177, 99),
            (246, 196, 73),
            (164, 95, 220),
            (240, 125, 54),
            (72, 196, 202),
            (194, 106, 130),
        ]

        self.title_font = pygame.font.SysFont("arial", 32, bold=True)
        self.info_font = pygame.font.SysFont("arial", 20, bold=True)

    def _cell_rect(self, r: int, c: int):
        pygame = self.pygame
        x = self.margin + c * (self.cell_size + self.margin)
        y = self.header_h + self.margin + r * (self.cell_size + self.margin)
        return pygame.Rect(x, y, self.cell_size, self.cell_size)

    def draw(
        self,
        env: Match3Env,
        score: int,
        steps: int,
        tiles: int,
        cascades: int,
        done: bool,
        *,
        board_override: np.ndarray | None = None,
        highlight_cells: Sequence[tuple[int, int]] | None = None,
        pop_mask: np.ndarray | None = None,
        banner: str | None = None,
    ) -> None:
        pygame = self.pygame
        board = env.game.board_arr if board_override is None else board_override

        self.screen.fill(self.bg)
        title = self.title_font.render("Match-3 RL", True, self.text)
        self.screen.blit(title, (self.margin, 12))
        line1 = self.info_font.render(f"Score: {score}   Steps: {steps}/{env.game.config.max_steps}", True, self.text)
        line2 = self.info_font.render(f"Tiles: {tiles}   Cascades: {cascades}   Legal: {len(env.legal_actions())}", True, self.text)
        self.screen.blit(line1, (self.margin, 48))
        self.screen.blit(line2, (self.margin, 72))
        if banner:
            label = self.info_font.render(banner, True, (110, 60, 30))
            self.screen.blit(label, (self.margin + 320, 48))
        if done:
            over = self.info_font.render("Done (Q/ESC to quit)", True, (150, 40, 40))
            self.screen.blit(over, (self.margin + 320, 72))

        arena = pygame.Rect(0, self.header_h, self.screen.get_width(), self.screen.get_height() - self.header_h)
        pygame.draw.rect(self.screen, self.board_bg, arena)

        for r in range(self.h):
            for c in range(self.w):
                rect = self._cell_rect(r, c)
                v = int(board[r, c])
                base = (236, 242, 249)
                pygame.draw.rect(self.screen, base, rect, border_radius=8)
                if v > 0:
                    color = self.palette[(v - 1) % len(self.palette)]
                    inner = rect.inflate(-8, -8)
                    pygame.draw.rect(self.screen, color, inner, border_radius=10)
                if pop_mask is not None and 0 <= r < pop_mask.shape[0] and 0 <= c < pop_mask.shape[1] and bool(pop_mask[r, c]):
                    glow = rect.inflate(-4, -4)
                    pygame.draw.rect(self.screen, (255, 255, 255), glow, border_radius=12)
                    pygame.draw.rect(self.screen, (250, 210, 80), glow, width=3, border_radius=12)

        if highlight_cells:
            for r, c in highlight_cells:
                if not (0 <= r < self.h and 0 <= c < self.w):
                    continue
                rect = self._cell_rect(r, c).inflate(-2, -2)
                pygame.draw.rect(self.screen, (40, 40, 40), rect, width=4, border_radius=10)
                pygame.draw.rect(self.screen, (255, 255, 255), rect.inflate(-4, -4), width=2, border_radius=8)

        pygame.display.flip()


def _pump_events_for_step(pygame, close_on_end_keys: bool = True) -> bool:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            return False
        if close_on_end_keys and event.type == pygame.KEYDOWN and event.key in (pygame.K_q, pygame.K_ESCAPE):
            return False
    return True


def _animate_match3_step(
    renderer: Match3PygameRenderer,
    env: Match3Env,
    *,
    pre_board: np.ndarray,
    swap_cells: Sequence[tuple[int, int]],
    score: int,
    steps: int,
    tiles: int,
    cascades: int,
    done: bool,
    select_flash: float,
    move_flash: float,
    pop_flash: float,
) -> bool:
    pygame = renderer.pygame
    running = True

    def draw_and_wait(
        duration: float,
        *,
        board_override: np.ndarray | None = None,
        highlight_cells: Sequence[tuple[int, int]] | None = None,
        pop_mask: np.ndarray | None = None,
        banner: str | None = None,
    ) -> bool:
        nonlocal running
        renderer.draw(
            env,
            score,
            steps,
            tiles,
            cascades,
            done,
            board_override=board_override,
            highlight_cells=highlight_cells,
            pop_mask=pop_mask,
            banner=banner,
        )
        end_t = time.monotonic() + max(0.0, duration)
        while time.monotonic() < end_t:
            if not _pump_events_for_step(pygame):
                running = False
                return False
            pygame.time.delay(8)
        return True

    if swap_cells:
        if not draw_and_wait(select_flash, board_override=pre_board, highlight_cells=swap_cells, banner="select"):
            return False
        swapped_board = env.game.last_swapped_board
        if swapped_board is not None:
            if not draw_and_wait(move_flash, board_override=swapped_board, highlight_cells=swap_cells, banner="moved"):
                return False

    for idx, frame in enumerate(env.game.last_cascade_frames, start=1):
        board_before, mask = frame
        if not draw_and_wait(pop_flash, board_override=board_before, pop_mask=mask, banner=f"pop x{idx}"):
            return False

    return running


def run_pygame(
    env: Match3Env,
    network: MLPQNetwork,
    seed: int,
    delay: float,
    max_steps: int,
    close_on_end: bool,
    *,
    select_delay: float | None = None,
    move_delay: float | None = None,
    clear_delay: float | None = None,
) -> PlayResult:
    try:
        import pygame
    except ImportError as exc:
        raise RuntimeError("pygame is not installed. Install dependencies with `pip install -r requirements.txt`.") from exc

    pygame.init()
    try:
        renderer = Match3PygameRenderer(width=env.game.config.width, height=env.game.config.height, num_colors=env.game.config.num_colors)
        clock = pygame.time.Clock()
        env.game.capture_step_traces = True
        select_delay_eff = float(select_delay) if select_delay is not None else min(0.18, max(0.06, delay * 0.7))
        move_delay_eff = float(move_delay) if move_delay is not None else min(0.18, max(0.06, delay * 0.7))
        clear_delay_eff = float(clear_delay) if clear_delay is not None else min(0.14, max(0.04, delay * 0.5))

        state = env.reset(seed=seed)
        done = False
        steps = 0
        info = {"score": 0, "total_tiles_cleared": 0, "total_cascades": 0}

        next_step_ts = time.monotonic()
        end_ts: float | None = None
        running = True

        print("Starting Match-3 autoplay in pygame mode. Press Q or ESC to quit.")

        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN and event.key in (pygame.K_q, pygame.K_ESCAPE):
                    running = False

            now = time.monotonic()
            if not done and steps < max_steps and now >= next_step_ts:
                legal = env.legal_actions()
                action = pick_action(network, state, legal)
                swap_cells = list(env.action_to_swap(action))
                pre_board = env.game.board_arr.copy()
                state, _, done, info = env.step(action)
                steps += 1
                if renderer is not None:
                    running = _animate_match3_step(
                        renderer,
                        env,
                        pre_board=pre_board,
                        swap_cells=swap_cells,
                        score=int(info["score"]),
                        steps=steps,
                        tiles=int(info["total_tiles_cleared"]),
                        cascades=int(info["total_cascades"]),
                        done=done or steps >= max_steps,
                        select_flash=select_delay_eff,
                        move_flash=move_delay_eff,
                        pop_flash=clear_delay_eff,
                    )
                    if not running:
                        break
                next_step_ts = time.monotonic() + max(0.0, delay)
                if done or steps >= max_steps:
                    end_ts = time.monotonic()

            renderer.draw(
                env,
                int(info["score"]),
                steps,
                int(info["total_tiles_cleared"]),
                int(info["total_cascades"]),
                done or steps >= max_steps,
            )

            if close_on_end and end_ts is not None and now - end_ts > 1.0:
                running = False

            clock.tick(60)

        return PlayResult(
            score=int(info["score"]),
            steps=steps,
            tiles=int(info["total_tiles_cleared"]),
            cascades=int(info["total_cascades"]),
        )
    finally:
        pygame.quit()


def main() -> None:
    parser = argparse.ArgumentParser(description="Autoplay Match-3 with a trained model")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--width", type=int, default=6)
    parser.add_argument("--height", type=int, default=6)
    parser.add_argument("--num-colors", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=150)
    parser.add_argument("--step-reward", type=float, default=-0.01)
    parser.add_argument("--tile-reward", type=float, default=0.20)
    parser.add_argument("--combo-bonus", type=float, default=0.15)
    parser.add_argument("--invalid-penalty", type=float, default=-0.50)
    parser.add_argument("--no-reshuffle-on-stuck", dest="reshuffle_on_stuck", action="store_false")
    parser.set_defaults(reshuffle_on_stuck=True)
    parser.add_argument("--delay", type=float, default=0.10)
    parser.add_argument("--select-delay", type=float, default=None, help="Seconds to show selected tiles before swap.")
    parser.add_argument("--move-delay", type=float, default=None, help="Seconds to show swapped tiles before clears.")
    parser.add_argument("--clear-delay", type=float, default=None, help="Seconds to flash cleared tiles per cascade.")
    parser.add_argument("--mode", choices=["terminal", "pygame"], default="terminal")
    parser.add_argument("--close-on-end", action="store_true")
    args = parser.parse_args()

    env = Match3Env(
        config=Match3Config(
            width=args.width,
            height=args.height,
            num_colors=args.num_colors,
            max_steps=args.max_steps,
            step_reward=args.step_reward,
            tile_reward=args.tile_reward,
            combo_bonus=args.combo_bonus,
            invalid_penalty=args.invalid_penalty,
            reshuffle_on_stuck=args.reshuffle_on_stuck,
        ),
        seed=args.seed,
    )
    network = MLPQNetwork.load(args.model)

    if args.mode == "pygame":
        result = run_pygame(
            env,
            network,
            args.seed,
            args.delay,
            args.max_steps,
            args.close_on_end,
            select_delay=args.select_delay,
            move_delay=args.move_delay,
            clear_delay=args.clear_delay,
        )
    else:
        result = run_terminal(env, network, args.seed, args.delay, args.max_steps)

    print("Match-3 game finished")
    print(f"Final score: {result.score}")
    print(f"Tiles cleared: {result.tiles}")
    print(f"Cascades: {result.cascades}")
    print(f"Steps: {result.steps}")


if __name__ == "__main__":
    main()
