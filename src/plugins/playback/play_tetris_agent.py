from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from typing import Union

from ...games.tetris import TetrisConfig, TetrisEnv, TetrisPlacementEnv
from ...network import MLPQNetwork


@dataclass
class PlayResult:
    score: int
    lines: int
    steps: int


TetrisAnyEnv = Union[TetrisEnv, TetrisPlacementEnv]


def pick_action(network: MLPQNetwork, state: object, legal_actions: list[int]) -> int:
    q_values = network.predict_one(state)
    if not legal_actions:
        return max(range(len(q_values)), key=q_values.__getitem__)
    return max(legal_actions, key=lambda action: q_values[action])


def run_terminal(env: TetrisAnyEnv, network: MLPQNetwork, seed: int, delay: float, max_steps: int) -> PlayResult:
    state = env.reset(seed=seed)
    done = False
    steps = 0
    info = {"score": 0, "lines": 0}

    print("Starting Tetris autoplay in terminal mode...\n")
    while not done and steps < max_steps:
        action = pick_action(network, state, env.legal_actions())
        state, reward, done, info = env.step(action)
        steps += 1

        print(f"Step {steps} | Action {action} | Reward {reward:.2f}")
        print(env.render())
        print("=" * 50)

        if delay > 0:
            time.sleep(delay)

    return PlayResult(score=int(info["score"]), lines=int(info["lines"]), steps=steps)


class TetrisPygameRenderer:
    def __init__(self, height: int = 20, width: int = 10, cell_size: int = 28, margin: int = 8, header_h: int = 90) -> None:
        import pygame

        self.pygame = pygame
        self.height = height
        self.width = width
        self.cell_size = cell_size
        self.margin = margin
        self.header_h = header_h

        self.screen_w = margin * (width + 1) + cell_size * width
        self.screen_h = header_h + margin * (height + 1) + cell_size * height
        self.screen = pygame.display.set_mode((self.screen_w, self.screen_h))
        pygame.display.set_caption("Tetris RL Autoplay")

        self.bg_color = (245, 245, 252)
        self.grid_bg = (220, 225, 240)
        self.locked_color = (76, 86, 180)
        self.active_color = (86, 196, 230)
        self.text_color = (38, 46, 76)

        self.title_font = pygame.font.SysFont("arial", 32, bold=True)
        self.info_font = pygame.font.SysFont("arial", 20, bold=True)

    def draw(self, board: object, score: int, lines: int, steps: int, done: bool) -> None:
        pygame = self.pygame
        self.screen.fill(self.bg_color)

        title = self.title_font.render("Tetris RL", True, self.text_color)
        self.screen.blit(title, (self.margin, 14))
        info = self.info_font.render(f"Score: {score}   Lines: {lines}   Steps: {steps}", True, self.text_color)
        self.screen.blit(info, (self.margin, 52))

        if done:
            over = self.info_font.render("Game over (Q/ESC to quit)", True, (150, 40, 40))
            self.screen.blit(over, (self.margin, 76))

        board_y0 = self.header_h
        bg_rect = pygame.Rect(0, board_y0, self.screen_w, self.screen_h - board_y0)
        pygame.draw.rect(self.screen, self.grid_bg, bg_rect)

        for r in range(self.height):
            for c in range(self.width):
                x = self.margin + c * (self.cell_size + self.margin)
                y = board_y0 + self.margin + r * (self.cell_size + self.margin)
                rect = pygame.Rect(x, y, self.cell_size, self.cell_size)

                value = int(board[r, c])
                if value == 0:
                    color = (236, 239, 250)
                elif value == 1:
                    color = self.locked_color
                else:
                    color = self.active_color

                pygame.draw.rect(self.screen, color, rect, border_radius=5)

        pygame.display.flip()


def run_pygame(
    env: TetrisAnyEnv,
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
        renderer = TetrisPygameRenderer(height=env.game.height, width=env.game.width)
        clock = pygame.time.Clock()

        state = env.reset(seed=seed)
        done = False
        steps = 0
        info = {"score": 0, "lines": 0}

        next_step_ts = time.monotonic()
        end_ts: float | None = None
        running = True

        print("Starting Tetris autoplay in pygame mode. Press Q or ESC to quit.")

        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN and event.key in (pygame.K_q, pygame.K_ESCAPE):
                    running = False

            now = time.monotonic()
            if not done and steps < max_steps and now >= next_step_ts:
                action = pick_action(network, state, env.legal_actions())
                state, _, done, info = env.step(action)
                steps += 1
                next_step_ts = now + max(0.0, delay)
                if done or steps >= max_steps:
                    end_ts = now

            renderer.draw(env.game.board_with_piece(), int(info["score"]), int(info["lines"]), steps, done or steps >= max_steps)

            if close_on_end and end_ts is not None and now - end_ts > 1.0:
                running = False

            clock.tick(60)

        return PlayResult(score=int(info["score"]), lines=int(info["lines"]), steps=steps)
    finally:
        pygame.quit()


def main() -> None:
    parser = argparse.ArgumentParser(description="Autoplay Tetris with a trained model")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--height", type=int, default=20)
    parser.add_argument("--width", type=int, default=10)
    parser.add_argument("--delay", type=float, default=0.08)
    parser.add_argument("--max-steps", type=int, default=1200)
    parser.add_argument("--mode", choices=["terminal", "pygame"], default="terminal")
    parser.add_argument("--placement-actions", action="store_true")
    parser.add_argument("--close-on-end", action="store_true")
    args = parser.parse_args()

    if args.placement_actions:
        env: TetrisAnyEnv = TetrisPlacementEnv(
            config=TetrisConfig(height=args.height, width=args.width, max_steps=args.max_steps), seed=args.seed
        )
    else:
        env = TetrisEnv(config=TetrisConfig(height=args.height, width=args.width, max_steps=args.max_steps), seed=args.seed)
    network = MLPQNetwork.load(args.model)

    if args.mode == "pygame":
        result = run_pygame(env, network, args.seed, args.delay, args.max_steps, args.close_on_end)
    else:
        result = run_terminal(env, network, args.seed, args.delay, args.max_steps)

    print("Tetris game finished")
    print(f"Final score: {result.score}")
    print(f"Lines cleared: {result.lines}")
    print(f"Steps: {result.steps}")


if __name__ == "__main__":
    main()
