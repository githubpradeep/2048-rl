from __future__ import annotations

import argparse
import time

from .games.tetris import TetrisConfig, TetrisPlacementEnv
from .network import MLPQNetwork
from .play_tetris_agent import PlayResult, TetrisPygameRenderer
from .tetris_afterstate_utils import pick_afterstate_action


def run_terminal(env: TetrisPlacementEnv, network: MLPQNetwork, seed: int, delay: float, max_steps: int) -> PlayResult:
    env.reset(seed=seed)
    done = False
    steps = 0
    info = {"score": 0, "lines": 0}

    print("Starting Tetris afterstate autoplay in terminal mode...\n")
    while not done and steps < max_steps:
        action = pick_afterstate_action(env, network)
        _, reward, done, info = env.step(action)
        steps += 1

        print(f"Step {steps} | Action {action} | Reward {reward:.2f}")
        print(env.render())
        print("=" * 50)

        if delay > 0:
            time.sleep(delay)

    return PlayResult(score=int(info["score"]), lines=int(info["lines"]), steps=steps)


def run_pygame(
    env: TetrisPlacementEnv,
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

        env.reset(seed=seed)
        done = False
        steps = 0
        info = {"score": 0, "lines": 0}

        next_step_ts = time.monotonic()
        end_ts: float | None = None
        running = True

        print("Starting Tetris afterstate autoplay in pygame mode. Press Q or ESC to quit.")

        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN and event.key in (pygame.K_q, pygame.K_ESCAPE):
                    running = False

            now = time.monotonic()
            if not done and steps < max_steps and now >= next_step_ts:
                action = pick_afterstate_action(env, network)
                _, _, done, info = env.step(action)
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
    parser = argparse.ArgumentParser(description="Autoplay Tetris with a trained afterstate model")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--height", type=int, default=20)
    parser.add_argument("--width", type=int, default=10)
    parser.add_argument("--delay", type=float, default=0.08)
    parser.add_argument("--max-steps", type=int, default=1200)
    parser.add_argument("--mode", choices=["terminal", "pygame"], default="terminal")
    parser.add_argument("--close-on-end", action="store_true")
    args = parser.parse_args()

    env = TetrisPlacementEnv(
        config=TetrisConfig(height=args.height, width=args.width, max_steps=args.max_steps),
        seed=args.seed,
    )
    network = MLPQNetwork.load(args.model)

    if args.mode == "pygame":
        result = run_pygame(env, network, args.seed, args.delay, args.max_steps, args.close_on_end)
    else:
        result = run_terminal(env, network, args.seed, args.delay, args.max_steps)

    print("Tetris afterstate game finished")
    print(f"Final score: {result.score}")
    print(f"Lines cleared: {result.lines}")
    print(f"Steps: {result.steps}")


if __name__ == "__main__":
    main()
