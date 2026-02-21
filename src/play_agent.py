from __future__ import annotations

import argparse
import time
from dataclasses import dataclass

import numpy as np

from .env import Game2048Env
from .network import MLPQNetwork


TILE_COLORS = {
    0: (205, 193, 180),
    2: (238, 228, 218),
    4: (237, 224, 200),
    8: (242, 177, 121),
    16: (245, 149, 99),
    32: (246, 124, 95),
    64: (246, 94, 59),
    128: (237, 207, 114),
    256: (237, 204, 97),
    512: (237, 200, 80),
    1024: (237, 197, 63),
    2048: (237, 194, 46),
}


@dataclass
class PlayResult:
    score: int
    max_tile: int
    steps: int


def pick_action(network: MLPQNetwork, state: np.ndarray) -> int:
    q_values = network.predict(state[None, :])[0]
    return int(np.argmax(q_values))


def run_terminal(
    env: Game2048Env,
    network: MLPQNetwork,
    seed: int,
    delay: float,
    max_steps: int,
    debug: bool,
) -> PlayResult:
    state = env.reset(seed=seed)
    done = False
    steps = 0
    info = {"score": 0, "max_tile": 0}

    print("Starting autoplay in terminal mode...\n")
    while not done and steps < max_steps:
        before_board = env.game.board.copy()
        action = pick_action(network, state)
        state, reward, done, info = env.step(action)

        print(f"Step {steps + 1} | Action {action} | Reward {reward:.1f}")
        if debug:
            print("Before:")
            print(before_board)
            print("After:")
            print(env.game.board)
            print(f"Moved: {info.get('moved')} | Gain: {info.get('gain')}")
        print(env.render())
        print("=" * 50)

        steps += 1
        if delay > 0:
            time.sleep(delay)

    return PlayResult(score=int(info["score"]), max_tile=int(info["max_tile"]), steps=steps)


class PygameRenderer:
    def __init__(self, grid_size: int = 4, tile_size: int = 120, margin: int = 12, header_h: int = 110) -> None:
        import pygame

        self.pygame = pygame
        self.grid_size = grid_size
        self.tile_size = tile_size
        self.margin = margin
        self.header_h = header_h

        self.width = margin * (grid_size + 1) + tile_size * grid_size
        self.height = self.width + header_h
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("2048 RL Autoplay")

        self.bg_color = (250, 248, 239)
        self.board_color = (187, 173, 160)
        self.text_dark = (119, 110, 101)
        self.text_light = (249, 246, 242)

        self.title_font = pygame.font.SysFont("arial", 36, bold=True)
        self.info_font = pygame.font.SysFont("arial", 24, bold=True)
        self.tile_font = pygame.font.SysFont("arial", 40, bold=True)

    def _tile_color(self, value: int) -> tuple[int, int, int]:
        if value in TILE_COLORS:
            return TILE_COLORS[value]
        return (60, 58, 50)

    def _tile_text_color(self, value: int) -> tuple[int, int, int]:
        return self.text_dark if value in (0, 2, 4) else self.text_light

    def draw(self, board: np.ndarray, score: int, max_tile: int, steps: int, done: bool) -> None:
        pygame = self.pygame
        self.screen.fill(self.bg_color)

        title_surface = self.title_font.render("2048 RL", True, self.text_dark)
        self.screen.blit(title_surface, (self.margin, 16))

        info_y = 58
        info = f"Score: {score}   Max Tile: {max_tile}   Steps: {steps}"
        info_surface = self.info_font.render(info, True, self.text_dark)
        self.screen.blit(info_surface, (self.margin, info_y))

        if done:
            done_surface = self.info_font.render("Game over (Q or ESC to quit)", True, (160, 60, 60))
            self.screen.blit(done_surface, (self.margin, info_y + 30))

        board_rect = pygame.Rect(0, self.header_h, self.width, self.width)
        pygame.draw.rect(self.screen, self.board_color, board_rect, border_radius=8)

        for r in range(self.grid_size):
            for c in range(self.grid_size):
                value = int(board[r, c])
                x = self.margin + c * (self.tile_size + self.margin)
                y = self.header_h + self.margin + r * (self.tile_size + self.margin)
                rect = pygame.Rect(x, y, self.tile_size, self.tile_size)
                pygame.draw.rect(self.screen, self._tile_color(value), rect, border_radius=8)

                if value != 0:
                    label = str(value)
                    font_size = 40 if value < 1000 else 30
                    font = pygame.font.SysFont("arial", font_size, bold=True)
                    text = font.render(label, True, self._tile_text_color(value))
                    text_rect = text.get_rect(center=rect.center)
                    self.screen.blit(text, text_rect)

        pygame.display.flip()


def run_pygame(
    env: Game2048Env,
    network: MLPQNetwork,
    seed: int,
    delay: float,
    max_steps: int,
    close_on_end: bool,
) -> PlayResult:
    try:
        import pygame
    except ImportError as exc:
        raise RuntimeError("pygame is not installed. Install dependencies with `pip install -r requirements.txt`.") from exc

    pygame.init()
    try:
        renderer = PygameRenderer(grid_size=env.game.size)
        clock = pygame.time.Clock()

        state = env.reset(seed=seed)
        done = False
        steps = 0
        info = {"score": 0, "max_tile": 0}

        next_step_ts = time.monotonic()
        end_ts: float | None = None
        running = True

        print("Starting autoplay in pygame mode. Press Q or ESC to quit.")

        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN and event.key in (pygame.K_q, pygame.K_ESCAPE):
                    running = False

            now = time.monotonic()
            if not done and steps < max_steps and now >= next_step_ts:
                action = pick_action(network, state)
                state, _, done, info = env.step(action)
                steps += 1
                next_step_ts = now + max(0.0, delay)
                if done or steps >= max_steps:
                    end_ts = now

            renderer.draw(env.game.board, int(info["score"]), int(info["max_tile"]), steps, done or steps >= max_steps)

            if close_on_end and end_ts is not None and now - end_ts > 1.0:
                running = False

            clock.tick(60)

        return PlayResult(score=int(info["score"]), max_tile=int(info["max_tile"]), steps=steps)
    finally:
        pygame.quit()


def main() -> None:
    parser = argparse.ArgumentParser(description="Autoplay 2048 with a trained model")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--delay", type=float, default=0.1)
    parser.add_argument("--max-steps", type=int, default=2000)
    parser.add_argument("--mode", choices=["terminal", "pygame"], default="terminal")
    parser.add_argument("--debug", action="store_true", help="Terminal mode only: print board before/after each step.")
    parser.add_argument(
        "--close-on-end",
        action="store_true",
        help="Only for pygame mode: close window automatically shortly after game end.",
    )
    args = parser.parse_args()

    env = Game2048Env(seed=args.seed)
    network = MLPQNetwork.load(args.model)

    if args.mode == "pygame":
        result = run_pygame(
            env=env,
            network=network,
            seed=args.seed,
            delay=args.delay,
            max_steps=args.max_steps,
            close_on_end=args.close_on_end,
        )
    else:
        result = run_terminal(
            env=env,
            network=network,
            seed=args.seed,
            delay=args.delay,
            max_steps=args.max_steps,
            debug=args.debug,
        )

    print("Game finished")
    print(f"Final score: {result.score}")
    print(f"Max tile: {result.max_tile}")
    print(f"Steps: {result.steps}")


if __name__ == "__main__":
    main()
