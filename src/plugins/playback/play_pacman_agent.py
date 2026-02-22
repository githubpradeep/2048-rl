from __future__ import annotations

import argparse
import time
from dataclasses import dataclass

from ...games.pacman_lite import PacmanLiteConfig, PacmanLiteEnv
from ...network import MLPQNetwork


@dataclass
class PlayResult:
    score: int
    steps: int
    lives: int
    pellets_left: int


def pick_action(network: MLPQNetwork, state: object, legal_actions: list[int]) -> int:
    q_values = network.predict_one(state)
    if not legal_actions:
        return max(range(len(q_values)), key=q_values.__getitem__)
    return max(legal_actions, key=lambda action: q_values[action])


def run_terminal(env: PacmanLiteEnv, network: MLPQNetwork, seed: int, delay: float, max_steps: int) -> PlayResult:
    state = env.reset(seed=seed)
    done = False
    steps = 0
    info = {"score": 0, "lives": env.game.lives, "pellets_left": env.game.pellets_left()}

    print("Starting Pacman-lite autoplay in terminal mode...\n")
    while not done and steps < max_steps:
        action = pick_action(network, state, env.legal_actions())
        state, reward, done, info = env.step(action)
        steps += 1
        print(f"Step {steps} | Action {action} | Reward {reward:.2f}")
        print(env.render())
        print("=" * 70)
        if delay > 0:
            time.sleep(delay)

    return PlayResult(score=int(info["score"]), steps=steps, lives=int(info["lives"]), pellets_left=int(info["pellets_left"]))


class PacmanPygameRenderer:
    def __init__(self, grid_size: int = 11, cell_size: int = 34, margin: int = 8, header_h: int = 88) -> None:
        import pygame

        self.pygame = pygame
        self.grid_size = grid_size
        self.cell_size = cell_size
        self.margin = margin
        self.header_h = header_h

        canvas_w = margin + grid_size * (cell_size + margin)
        canvas_h = header_h + margin + grid_size * (cell_size + margin)
        self.screen = pygame.display.set_mode((canvas_w, canvas_h))
        pygame.display.set_caption("Pacman-lite RL Autoplay")

        self.bg = (248, 247, 240)
        self.arena = (234, 232, 220)
        self.wall = (52, 84, 196)
        self.pellet = (245, 220, 115)
        self.pacman = (245, 196, 38)
        self.ghost = (214, 78, 96)
        self.text = (40, 47, 60)

        self.title_font = pygame.font.SysFont("arial", 30, bold=True)
        self.info_font = pygame.font.SysFont("arial", 19, bold=True)

    def _cell_rect(self, r: int, c: int):
        return self.pygame.Rect(
            self.margin + c * (self.cell_size + self.margin),
            self.header_h + self.margin + r * (self.cell_size + self.margin),
            self.cell_size,
            self.cell_size,
        )

    def draw(self, board: object, score: int, lives: int, pellets_left: int, steps: int, done: bool) -> None:
        pygame = self.pygame
        self.screen.fill(self.bg)
        self.screen.blit(self.title_font.render("Pacman-lite RL", True, self.text), (self.margin, 10))
        line = self.info_font.render(f"Score: {score}  Lives: {lives}  Pellets: {pellets_left}  Steps: {steps}", True, self.text)
        self.screen.blit(line, (self.margin, 46))
        if done:
            self.screen.blit(self.info_font.render("Game over (Q/ESC to quit)", True, (150, 40, 40)), (self.margin, 68))

        arena = pygame.Rect(0, self.header_h, self.screen.get_width(), self.screen.get_height() - self.header_h)
        pygame.draw.rect(self.screen, self.arena, arena)

        for r in range(self.grid_size):
            for c in range(self.grid_size):
                rect = self._cell_rect(r, c)
                v = int(board[r, c])
                if v == 1:
                    pygame.draw.rect(self.screen, self.wall, rect, border_radius=5)
                else:
                    pygame.draw.rect(self.screen, (243, 241, 231), rect, border_radius=5)
                    if v == 2:
                        pygame.draw.circle(self.screen, self.pellet, rect.center, max(2, self.cell_size // 10))
                    elif v == 3:
                        pygame.draw.circle(self.screen, self.pacman, rect.center, max(6, self.cell_size // 3))
                    elif v == 4:
                        pygame.draw.circle(self.screen, self.ghost, rect.center, max(6, self.cell_size // 3))
                        eye_r = max(1, self.cell_size // 12)
                        pygame.draw.circle(self.screen, (255, 255, 255), (rect.centerx - 4, rect.centery - 2), eye_r)
                        pygame.draw.circle(self.screen, (255, 255, 255), (rect.centerx + 4, rect.centery - 2), eye_r)
        pygame.display.flip()


def run_pygame(env: PacmanLiteEnv, network: MLPQNetwork, seed: int, delay: float, max_steps: int, close_on_end: bool) -> PlayResult:
    try:
        import pygame
    except ImportError as exc:
        raise RuntimeError("pygame is not installed. Install dependencies with `pip install -r requirements.txt`.") from exc

    pygame.init()
    try:
        renderer = PacmanPygameRenderer(grid_size=env.game.grid_size)
        clock = pygame.time.Clock()
        state = env.reset(seed=seed)
        done = False
        steps = 0
        info = {"score": 0, "lives": env.game.lives, "pellets_left": env.game.pellets_left()}
        next_step_ts = time.monotonic()
        end_ts: float | None = None
        running = True

        print("Starting Pacman-lite autoplay in pygame mode. Press Q or ESC to quit.")
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

            renderer.draw(env.game.board(), int(info["score"]), int(info["lives"]), int(info["pellets_left"]), steps, done or steps >= max_steps)
            if close_on_end and end_ts is not None and now - end_ts > 1.0:
                running = False
            clock.tick(60)

        return PlayResult(score=int(info["score"]), steps=steps, lives=int(info["lives"]), pellets_left=int(info["pellets_left"]))
    finally:
        pygame.quit()


def main() -> None:
    parser = argparse.ArgumentParser(description="Autoplay Pacman-lite with a trained model")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--grid-size", type=int, default=11)
    parser.add_argument("--state-grid-size", type=int, default=0)
    parser.add_argument("--num-ghosts", type=int, default=2)
    parser.add_argument("--ghost-chase-prob", type=float, default=0.75)
    parser.add_argument("--delay", type=float, default=0.08)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--mode", choices=["terminal", "pygame"], default="terminal")
    parser.add_argument("--close-on-end", action="store_true")
    args = parser.parse_args()

    state_grid_size = args.state_grid_size if args.state_grid_size > 0 else args.grid_size
    env = PacmanLiteEnv(
        config=PacmanLiteConfig(grid_size=args.grid_size, num_ghosts=args.num_ghosts, ghost_chase_prob=args.ghost_chase_prob, max_steps=args.max_steps),
        seed=args.seed,
        state_grid_size=state_grid_size,
    )
    network = MLPQNetwork.load(args.model)

    if args.mode == "pygame":
        result = run_pygame(env, network, args.seed, args.delay, args.max_steps, args.close_on_end)
    else:
        result = run_terminal(env, network, args.seed, args.delay, args.max_steps)

    print("Pacman-lite game finished")
    print(f"Final score: {result.score}")
    print(f"Lives left: {result.lives}")
    print(f"Pellets left: {result.pellets_left}")
    print(f"Steps: {result.steps}")


if __name__ == "__main__":
    main()
