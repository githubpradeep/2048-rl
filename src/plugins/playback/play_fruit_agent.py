from __future__ import annotations

import argparse
import time
from dataclasses import dataclass

from ...games.fruit_cutter import FruitCutterConfig, FruitCutterEnv
from ...network import MLPQNetwork


@dataclass
class PlayResult:
    score: int
    steps: int


def pick_action(network: MLPQNetwork, state: object, legal_actions: list[int]) -> int:
    q_values = network.predict_one(state)
    if not legal_actions:
        return max(range(len(q_values)), key=q_values.__getitem__)
    return max(legal_actions, key=lambda action: q_values[action])


def run_terminal(env: FruitCutterEnv, network: MLPQNetwork, seed: int, delay: float, max_steps: int) -> PlayResult:
    state = env.reset(seed=seed)
    done = False
    steps = 0
    info = {"score": 0}

    print("Starting Fruit Cutter autoplay in terminal mode...\n")
    while not done and steps < max_steps:
        action = pick_action(network, state, env.legal_actions())
        state, reward, done, info = env.step(action)
        steps += 1

        print(f"Step {steps} | Action {action} | Reward {reward:.2f}")
        print(env.render())
        print("=" * 50)

        if delay > 0:
            time.sleep(delay)

    return PlayResult(score=int(info["score"]), steps=steps)


class FruitPygameRenderer:
    def __init__(self, grid_size: int = 10, cell_size: int = 42, margin: int = 10, header_h: int = 90) -> None:
        import pygame

        self.pygame = pygame
        self.grid_size = grid_size
        self.cell_size = cell_size
        self.margin = margin
        self.header_h = header_h

        self.width = margin * (grid_size + 1) + cell_size * grid_size
        self.height = self.width + header_h
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Fruit Cutter RL Autoplay")

        self.bg_color = (250, 244, 236)
        self.grid_bg = (232, 221, 203)
        self.fruit_color = (232, 98, 67)
        self.bomb_color = (50, 50, 58)
        self.player_color = (50, 120, 220)
        self.text_color = (45, 45, 45)

        self.title_font = pygame.font.SysFont("arial", 34, bold=True)
        self.info_font = pygame.font.SysFont("arial", 22, bold=True)

    def draw(self, board: object, score: int, steps: int, done: bool) -> None:
        pygame = self.pygame
        self.screen.fill(self.bg_color)

        title = self.title_font.render("Fruit Cutter RL", True, self.text_color)
        self.screen.blit(title, (self.margin, 14))
        info = self.info_font.render(f"Score: {score}   Steps: {steps}", True, self.text_color)
        self.screen.blit(info, (self.margin, 52))

        if done:
            over = self.info_font.render("Game over (Q/ESC to quit)", True, (150, 40, 40))
            self.screen.blit(over, (self.margin, 76))

        board_rect = pygame.Rect(0, self.header_h, self.width, self.width)
        pygame.draw.rect(self.screen, self.grid_bg, board_rect)

        for r in range(self.grid_size):
            for c in range(self.grid_size):
                x = self.margin + c * (self.cell_size + self.margin)
                y = self.header_h + self.margin + r * (self.cell_size + self.margin)
                rect = pygame.Rect(x, y, self.cell_size, self.cell_size)

                value = int(board[r, c])
                if value == 0:
                    color = (244, 235, 222)
                elif value == 1:
                    color = self.fruit_color
                elif value == 2:
                    color = self.bomb_color
                else:
                    color = self.player_color

                pygame.draw.rect(self.screen, color, rect, border_radius=7)

        pygame.display.flip()


def run_pygame(
    env: FruitCutterEnv,
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
        renderer = FruitPygameRenderer(grid_size=env.game.grid_size)
        clock = pygame.time.Clock()

        state = env.reset(seed=seed)
        done = False
        steps = 0
        info = {"score": 0}

        next_step_ts = time.monotonic()
        end_ts: float | None = None
        running = True

        print("Starting Fruit Cutter autoplay in pygame mode. Press Q or ESC to quit.")

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

            renderer.draw(env.game.board(), int(info["score"]), steps, done or steps >= max_steps)

            if close_on_end and end_ts is not None and now - end_ts > 1.0:
                running = False

            clock.tick(60)

        return PlayResult(score=int(info["score"]), steps=steps)
    finally:
        pygame.quit()


def main() -> None:
    parser = argparse.ArgumentParser(description="Autoplay Fruit Cutter with a trained model")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--grid-size", type=int, default=10)
    parser.add_argument("--state-grid-size", type=int, default=0)
    parser.add_argument("--spawn-prob", type=float, default=0.45)
    parser.add_argument("--bomb-prob", type=float, default=0.18)
    parser.add_argument("--delay", type=float, default=0.08)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--mode", choices=["terminal", "pygame"], default="terminal")
    parser.add_argument("--close-on-end", action="store_true")
    args = parser.parse_args()

    state_grid_size = args.state_grid_size if args.state_grid_size > 0 else args.grid_size
    env = FruitCutterEnv(
        config=FruitCutterConfig(grid_size=args.grid_size, spawn_prob=args.spawn_prob, bomb_prob=args.bomb_prob),
        seed=args.seed,
        state_grid_size=state_grid_size,
    )
    network = MLPQNetwork.load(args.model)

    if args.mode == "pygame":
        result = run_pygame(env, network, args.seed, args.delay, args.max_steps, args.close_on_end)
    else:
        result = run_terminal(env, network, args.seed, args.delay, args.max_steps)

    print("Fruit Cutter game finished")
    print(f"Final score: {result.score}")
    print(f"Steps: {result.steps}")


if __name__ == "__main__":
    main()
