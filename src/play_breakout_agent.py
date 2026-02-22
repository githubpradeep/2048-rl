from __future__ import annotations

import argparse
import time
from dataclasses import dataclass

from .games.breakout import BreakoutConfig, BreakoutEnv
from .network import MLPQNetwork


@dataclass
class PlayResult:
    score: int
    steps: int
    lives: int
    bricks_left: int


def pick_action(network: MLPQNetwork, state: object, legal_actions: list[int]) -> int:
    q_values = network.predict_one(state)
    if not legal_actions:
        return max(range(len(q_values)), key=q_values.__getitem__)
    return max(legal_actions, key=lambda action: q_values[action])


def run_terminal(env: BreakoutEnv, network: MLPQNetwork, seed: int, delay: float, max_steps: int) -> PlayResult:
    state = env.reset(seed=seed)
    done = False
    steps = 0
    info = {"score": 0, "lives": env.game.lives, "bricks_left": env.game.bricks_left()}

    print("Starting Breakout autoplay in terminal mode...\n")
    while not done and steps < max_steps:
        action = pick_action(network, state, env.legal_actions())
        state, reward, done, info = env.step(action)
        steps += 1

        print(f"Step {steps} | Action {action} | Reward {reward:.2f}")
        print(env.render())
        print("=" * 60)

        if delay > 0:
            time.sleep(delay)

    return PlayResult(
        score=int(info["score"]),
        steps=steps,
        lives=int(info["lives"]),
        bricks_left=int(info["bricks_left"]),
    )


class BreakoutPygameRenderer:
    def __init__(self, width: int = 12, height: int = 14, cell_size: int = 34, margin: int = 8, header_h: int = 88) -> None:
        import pygame

        self.pygame = pygame
        self.grid_w = width
        self.grid_h = height
        self.cell_size = cell_size
        self.margin = margin
        self.header_h = header_h

        canvas_w = margin + width * (cell_size + margin)
        canvas_h = header_h + margin + height * (cell_size + margin)
        self.screen = pygame.display.set_mode((canvas_w, canvas_h))
        pygame.display.set_caption("Breakout RL Autoplay")

        self.bg = (244, 247, 252)
        self.arena_bg = (224, 233, 244)
        self.brick = (219, 88, 78)
        self.ball = (246, 204, 58)
        self.paddle = (59, 116, 224)
        self.text = (36, 48, 66)

        self.title_font = pygame.font.SysFont("arial", 34, bold=True)
        self.info_font = pygame.font.SysFont("arial", 21, bold=True)

    def _cell_rect(self, row: int, col: int):
        pygame = self.pygame
        x = self.margin + col * (self.cell_size + self.margin)
        y = self.header_h + self.margin + row * (self.cell_size + self.margin)
        return pygame.Rect(x, y, self.cell_size, self.cell_size)

    def draw(self, board: object, score: int, lives: int, steps: int, bricks_left: int, done: bool) -> None:
        pygame = self.pygame
        self.screen.fill(self.bg)

        title = self.title_font.render("Breakout RL", True, self.text)
        self.screen.blit(title, (self.margin, 12))
        info = self.info_font.render(
            f"Score: {score}   Lives: {lives}   Bricks: {bricks_left}   Steps: {steps}",
            True,
            self.text,
        )
        self.screen.blit(info, (self.margin, 52))
        if done:
            over = self.info_font.render("Game over (Q/ESC to quit)", True, (150, 40, 40))
            self.screen.blit(over, (self.margin + 360, 52))

        arena_y = self.header_h
        arena_rect = pygame.Rect(0, arena_y, self.screen.get_width(), self.screen.get_height() - arena_y)
        pygame.draw.rect(self.screen, self.arena_bg, arena_rect)

        for r in range(self.grid_h):
            for c in range(self.grid_w):
                rect = self._cell_rect(r, c)
                v = int(board[r, c])
                if v == 1:
                    color = self.brick
                    pygame.draw.rect(self.screen, color, rect, border_radius=6)
                elif v == 2:
                    center = (rect.x + rect.w // 2, rect.y + rect.h // 2)
                    pygame.draw.circle(self.screen, self.ball, center, max(4, self.cell_size // 4))
                elif v == 3:
                    pygame.draw.rect(self.screen, self.paddle, rect, border_radius=6)
                else:
                    pygame.draw.rect(self.screen, (234, 240, 248), rect, border_radius=4)

        pygame.display.flip()


def run_pygame(
    env: BreakoutEnv,
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
        renderer = BreakoutPygameRenderer(width=env.game.config.width, height=env.game.config.height)
        clock = pygame.time.Clock()

        state = env.reset(seed=seed)
        done = False
        steps = 0
        info = {"score": 0, "lives": env.game.lives, "bricks_left": env.game.bricks_left()}

        next_step_ts = time.monotonic()
        end_ts: float | None = None
        running = True

        print("Starting Breakout autoplay in pygame mode. Press Q or ESC to quit.")

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

            renderer.draw(
                env.game.board(),
                int(info["score"]),
                int(info["lives"]),
                steps,
                int(info["bricks_left"]),
                done or steps >= max_steps,
            )

            if close_on_end and end_ts is not None and now - end_ts > 1.0:
                running = False

            clock.tick(60)

        return PlayResult(
            score=int(info["score"]),
            steps=steps,
            lives=int(info["lives"]),
            bricks_left=int(info["bricks_left"]),
        )
    finally:
        pygame.quit()


def main() -> None:
    parser = argparse.ArgumentParser(description="Autoplay Breakout with a trained model")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--width", type=int, default=12)
    parser.add_argument("--height", type=int, default=14)
    parser.add_argument("--brick-rows", type=int, default=4)
    parser.add_argument("--brick-top", type=int, default=1)
    parser.add_argument("--paddle-width", type=int, default=3)
    parser.add_argument("--start-lives", type=int, default=3)
    parser.add_argument("--step-reward", type=float, default=0.02)
    parser.add_argument("--brick-reward", type=float, default=1.0)
    parser.add_argument("--paddle-hit-reward", type=float, default=0.03)
    parser.add_argument("--life-loss-penalty", type=float, default=-3.0)
    parser.add_argument("--clear-bonus", type=float, default=5.0)
    parser.add_argument("--delay", type=float, default=0.08)
    parser.add_argument("--max-steps", type=int, default=1200)
    parser.add_argument("--mode", choices=["terminal", "pygame"], default="terminal")
    parser.add_argument("--close-on-end", action="store_true")
    args = parser.parse_args()

    env = BreakoutEnv(
        config=BreakoutConfig(
            width=args.width,
            height=args.height,
            brick_rows=args.brick_rows,
            brick_top=args.brick_top,
            paddle_width=args.paddle_width,
            start_lives=args.start_lives,
            step_reward=args.step_reward,
            brick_reward=args.brick_reward,
            paddle_hit_reward=args.paddle_hit_reward,
            life_loss_penalty=args.life_loss_penalty,
            clear_bonus=args.clear_bonus,
            max_steps=args.max_steps,
        ),
        seed=args.seed,
    )
    network = MLPQNetwork.load(args.model)

    if args.mode == "pygame":
        result = run_pygame(env, network, args.seed, args.delay, args.max_steps, args.close_on_end)
    else:
        result = run_terminal(env, network, args.seed, args.delay, args.max_steps)

    print("Breakout game finished")
    print(f"Final score: {result.score}")
    print(f"Lives left: {result.lives}")
    print(f"Bricks left: {result.bricks_left}")
    print(f"Steps: {result.steps}")


if __name__ == "__main__":
    main()

