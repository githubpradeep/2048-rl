from __future__ import annotations

import argparse
import time
from dataclasses import dataclass

from ...games.pong import PongConfig, PongEnv
from ...network import MLPQNetwork


@dataclass
class PlayResult:
    player_score: int
    opponent_score: int
    steps: int
    lives: int


def pick_action(network: MLPQNetwork, state: object, legal_actions: list[int]) -> int:
    q_values = network.predict_one(state)
    if not legal_actions:
        return max(range(len(q_values)), key=q_values.__getitem__)
    return max(legal_actions, key=lambda action: q_values[action])


def run_terminal(env: PongEnv, network: MLPQNetwork, seed: int, delay: float, max_steps: int) -> PlayResult:
    state = env.reset(seed=seed)
    done = False
    steps = 0
    info = {"score": 0, "opponent_score": 0, "lives": env.game.lives}

    print("Starting Pong autoplay in terminal mode...\n")
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
        player_score=int(info["score"]),
        opponent_score=int(info["opponent_score"]),
        steps=steps,
        lives=int(info["lives"]),
    )


class PongPygameRenderer:
    def __init__(self, width: int = 12, height: int = 16, cell_size: int = 34, margin: int = 8, header_h: int = 88) -> None:
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
        pygame.display.set_caption("Pong RL Autoplay")

        self.bg = (244, 247, 252)
        self.arena = (227, 236, 246)
        self.player = (59, 116, 224)
        self.opp = (214, 82, 92)
        self.ball = (245, 205, 72)
        self.text = (36, 48, 66)

        self.title_font = pygame.font.SysFont("arial", 34, bold=True)
        self.info_font = pygame.font.SysFont("arial", 21, bold=True)

    def _cell_rect(self, row: int, col: int):
        pygame = self.pygame
        x = self.margin + col * (self.cell_size + self.margin)
        y = self.header_h + self.margin + row * (self.cell_size + self.margin)
        return pygame.Rect(x, y, self.cell_size, self.cell_size)

    def draw(self, board: object, player_score: int, opp_score: int, lives: int, steps: int, done: bool) -> None:
        pygame = self.pygame
        self.screen.fill(self.bg)

        title = self.title_font.render("Pong RL", True, self.text)
        self.screen.blit(title, (self.margin, 12))
        info = self.info_font.render(
            f"Score: {player_score}-{opp_score}   Lives: {lives}   Steps: {steps}",
            True,
            self.text,
        )
        self.screen.blit(info, (self.margin, 52))
        if done:
            over = self.info_font.render("Game over (Q/ESC to quit)", True, (150, 40, 40))
            self.screen.blit(over, (self.margin + 320, 52))

        arena_y = self.header_h
        arena_rect = pygame.Rect(0, arena_y, self.screen.get_width(), self.screen.get_height() - arena_y)
        pygame.draw.rect(self.screen, self.arena, arena_rect)

        for r in range(self.grid_h):
            for c in range(self.grid_w):
                rect = self._cell_rect(r, c)
                v = int(board[r, c])
                if v == 1:
                    pygame.draw.rect(self.screen, self.player, rect, border_radius=5)
                elif v == 2:
                    pygame.draw.rect(self.screen, self.opp, rect, border_radius=5)
                elif v == 3:
                    center = (rect.x + rect.w // 2, rect.y + rect.h // 2)
                    pygame.draw.circle(self.screen, self.ball, center, max(4, self.cell_size // 4))
                else:
                    pygame.draw.rect(self.screen, (236, 242, 249), rect, border_radius=3)

        pygame.display.flip()


def run_pygame(env: PongEnv, network: MLPQNetwork, seed: int, delay: float, max_steps: int, close_on_end: bool) -> PlayResult:
    try:
        import pygame
    except ImportError as exc:
        raise RuntimeError("pygame is not installed. Install dependencies with `pip install -r requirements.txt`.") from exc

    pygame.init()
    try:
        renderer = PongPygameRenderer(width=env.game.config.width, height=env.game.config.height)
        clock = pygame.time.Clock()

        state = env.reset(seed=seed)
        done = False
        steps = 0
        info = {"score": 0, "opponent_score": 0, "lives": env.game.lives}

        next_step_ts = time.monotonic()
        end_ts: float | None = None
        running = True

        print("Starting Pong autoplay in pygame mode. Press Q or ESC to quit.")

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

            renderer.draw(env.game.board(), int(info["score"]), int(info["opponent_score"]), int(info["lives"]), steps, done or steps >= max_steps)

            if close_on_end and end_ts is not None and now - end_ts > 1.0:
                running = False

            clock.tick(60)

        return PlayResult(
            player_score=int(info["score"]),
            opponent_score=int(info["opponent_score"]),
            steps=steps,
            lives=int(info["lives"]),
        )
    finally:
        pygame.quit()


def main() -> None:
    parser = argparse.ArgumentParser(description="Autoplay Pong with a trained model")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--width", type=int, default=12)
    parser.add_argument("--height", type=int, default=16)
    parser.add_argument("--paddle-height", type=int, default=4)
    parser.add_argument("--paddle-speed", type=int, default=1)
    parser.add_argument("--opponent-track-prob", type=float, default=0.85)
    parser.add_argument("--start-lives", type=int, default=3)
    parser.add_argument("--step-reward", type=float, default=0.01)
    parser.add_argument("--paddle-hit-reward", type=float, default=0.05)
    parser.add_argument("--score-reward", type=float, default=2.0)
    parser.add_argument("--concede-penalty", type=float, default=-2.0)
    parser.add_argument("--delay", type=float, default=0.08)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--mode", choices=["terminal", "pygame"], default="terminal")
    parser.add_argument("--close-on-end", action="store_true")
    args = parser.parse_args()

    env = PongEnv(
        config=PongConfig(
            width=args.width,
            height=args.height,
            paddle_height=args.paddle_height,
            paddle_speed=args.paddle_speed,
            opponent_track_prob=args.opponent_track_prob,
            start_lives=args.start_lives,
            step_reward=args.step_reward,
            paddle_hit_reward=args.paddle_hit_reward,
            score_reward=args.score_reward,
            concede_penalty=args.concede_penalty,
            max_steps=args.max_steps,
        ),
        seed=args.seed,
    )
    network = MLPQNetwork.load(args.model)

    if args.mode == "pygame":
        result = run_pygame(env, network, args.seed, args.delay, args.max_steps, args.close_on_end)
    else:
        result = run_terminal(env, network, args.seed, args.delay, args.max_steps)

    print("Pong game finished")
    print(f"Final score: {result.player_score}-{result.opponent_score}")
    print(f"Lives left: {result.lives}")
    print(f"Steps: {result.steps}")


if __name__ == "__main__":
    main()

