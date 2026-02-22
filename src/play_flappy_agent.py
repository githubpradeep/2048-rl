from __future__ import annotations

import argparse
import time
from dataclasses import dataclass

from .games.flappy import FlappyConfig, FlappyEnv, Pipe
from .network import MLPQNetwork


@dataclass
class PlayResult:
    score: int
    steps: int


def pick_action(network: MLPQNetwork, state: object, legal_actions: list[int]) -> int:
    q_values = network.predict_one(state)
    if not legal_actions:
        return max(range(len(q_values)), key=q_values.__getitem__)
    return max(legal_actions, key=lambda action: q_values[action])


def run_terminal(env: FlappyEnv, network: MLPQNetwork, seed: int, delay: float, max_steps: int) -> PlayResult:
    state = env.reset(seed=seed)
    done = False
    steps = 0
    info = {"score": 0}

    print("Starting Flappy autoplay in terminal mode...\n")
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


class FlappyPygameRenderer:
    def __init__(self, width: int = 84, height: int = 84, scale: int = 8, header_h: int = 80) -> None:
        import pygame

        self.pygame = pygame
        self.width = width
        self.height = height
        self.scale = scale
        self.header_h = header_h

        self.screen = pygame.display.set_mode((width * scale, height * scale + header_h))
        pygame.display.set_caption("Flappy Bird RL Autoplay")

        self.sky = (147, 214, 252)
        self.pipe_color = (52, 152, 74)
        self.bird_color = (245, 200, 55)
        self.text_color = (36, 48, 66)
        self.bg_header = (231, 244, 252)

        self.title_font = pygame.font.SysFont("arial", 32, bold=True)
        self.info_font = pygame.font.SysFont("arial", 21, bold=True)

    def _draw_pipe(self, pipe: Pipe, pipe_width: float, gap_size: float) -> None:
        pygame = self.pygame
        x = int(pipe.x * self.scale)
        w = int(pipe_width * self.scale)
        half_gap = gap_size / 2.0
        gap_top = int((pipe.gap_y - half_gap) * self.scale)
        gap_bot = int((pipe.gap_y + half_gap) * self.scale)
        y0 = self.header_h
        total_h = self.height * self.scale

        if gap_top > 0:
            rect_top = pygame.Rect(x, y0, w, gap_top)
            pygame.draw.rect(self.screen, self.pipe_color, rect_top, border_radius=4)
        if gap_bot < total_h:
            rect_bot = pygame.Rect(x, y0 + gap_bot, w, total_h - gap_bot)
            pygame.draw.rect(self.screen, self.pipe_color, rect_bot, border_radius=4)

    def draw(self, env: FlappyEnv, score: int, steps: int, done: bool) -> None:
        pygame = self.pygame
        cfg = env.game.config

        self.screen.fill(self.sky)
        header_rect = pygame.Rect(0, 0, cfg.width * self.scale, self.header_h)
        pygame.draw.rect(self.screen, self.bg_header, header_rect)

        title = self.title_font.render("Flappy RL", True, self.text_color)
        self.screen.blit(title, (12, 10))
        info = self.info_font.render(f"Score: {score}   Steps: {steps}", True, self.text_color)
        self.screen.blit(info, (12, 46))
        if done:
            over = self.info_font.render("Game over (Q/ESC to quit)", True, (150, 40, 40))
            self.screen.blit(over, (260, 46))

        for pipe in env.game.pipes:
            self._draw_pipe(pipe, cfg.pipe_width, cfg.gap_size)

        bx = int(cfg.bird_x * self.scale)
        by = self.header_h + int(env.game.bird_y * self.scale)
        pygame.draw.circle(self.screen, self.bird_color, (bx, by), max(4, self.scale // 2))

        pygame.display.flip()


def run_pygame(
    env: FlappyEnv,
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
        renderer = FlappyPygameRenderer(width=env.game.config.width, height=env.game.config.height)
        clock = pygame.time.Clock()

        state = env.reset(seed=seed)
        done = False
        steps = 0
        info = {"score": 0}

        next_step_ts = time.monotonic()
        end_ts: float | None = None
        running = True

        print("Starting Flappy autoplay in pygame mode. Press Q or ESC to quit.")

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

            renderer.draw(env, int(info["score"]), steps, done or steps >= max_steps)

            if close_on_end and end_ts is not None and now - end_ts > 1.0:
                running = False

            clock.tick(60)

        return PlayResult(score=int(info["score"]), steps=steps)
    finally:
        pygame.quit()


def main() -> None:
    parser = argparse.ArgumentParser(description="Autoplay Flappy Bird with a trained model")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--width", type=int, default=84)
    parser.add_argument("--height", type=int, default=84)
    parser.add_argument("--pipe-gap", type=float, default=30.0)
    parser.add_argument("--pipe-speed", type=float, default=1.3)
    parser.add_argument("--pipe-spacing", type=float, default=44.0)
    parser.add_argument("--initial-pipe-offset", type=float, default=20.0)
    parser.add_argument("--floor-height", type=float, default=10.0)
    parser.add_argument("--max-gap-delta", type=float, default=8.0)
    parser.add_argument("--gravity", type=float, default=0.18)
    parser.add_argument("--flap-velocity", type=float, default=-2.2)
    parser.add_argument("--delay", type=float, default=0.08)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--mode", choices=["terminal", "pygame"], default="terminal")
    parser.add_argument("--close-on-end", action="store_true")
    args = parser.parse_args()

    env = FlappyEnv(
        config=FlappyConfig(
            width=args.width,
            height=args.height,
            gap_size=args.pipe_gap,
            pipe_speed=args.pipe_speed,
            pipe_spacing=args.pipe_spacing,
            initial_pipe_offset=args.initial_pipe_offset,
            floor_height=args.floor_height,
            max_gap_delta=args.max_gap_delta,
            gravity=args.gravity,
            flap_velocity=args.flap_velocity,
            max_steps=args.max_steps,
        ),
        seed=args.seed,
    )
    network = MLPQNetwork.load(args.model)

    if args.mode == "pygame":
        result = run_pygame(env, network, args.seed, args.delay, args.max_steps, args.close_on_end)
    else:
        result = run_terminal(env, network, args.seed, args.delay, args.max_steps)

    print("Flappy game finished")
    print(f"Final score: {result.score}")
    print(f"Steps: {result.steps}")


if __name__ == "__main__":
    main()
