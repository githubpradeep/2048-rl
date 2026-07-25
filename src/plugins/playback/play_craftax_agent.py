from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from typing import Any

from ...games.craftax_classic import Action, CraftaxClassicConfig, CraftaxClassicEnv
from ...games.craftax_classic.renderer import CraftaxPygameViewer
from ...network import MLPQNetwork

# Slow enough to read movement / DO / craft on a textured map (~3 steps/sec).
DEFAULT_DELAY = 0.35
MIN_DELAY = 0.05
MAX_DELAY = 2.0


@dataclass
class PlayResult:
    score: int
    steps: int
    health: int
    achievements: int


def pick_action(network: MLPQNetwork, state: object, legal_actions: list[int]) -> int:
    q_values = network.predict_one(state)
    if not legal_actions:
        return max(range(len(q_values)), key=q_values.__getitem__)
    return max(legal_actions, key=lambda action: q_values[action])


def action_label(action: int) -> str:
    try:
        return Action(action).name
    except ValueError:
        return str(action)


def run_terminal(
    env: CraftaxClassicEnv,
    network: MLPQNetwork,
    seed: int,
    delay: float,
    max_steps: int,
) -> PlayResult:
    state = env.reset(seed=seed)
    done = False
    steps = 0
    info: dict[str, Any] = {"score": 0, "health": 9, "achievements": 0}

    print("Starting Craftax-Classic autoplay in terminal mode...\n")
    while not done and steps < max_steps:
        action = pick_action(network, state, env.legal_actions())
        state, reward, done, info = env.step(action)
        steps += 1
        print(
            f"Step {steps} | Action {action_label(action)} ({action}) | "
            f"Reward {reward:.2f} | Ach={info['score']}"
        )
        print(env.render())
        print("=" * 60)
        if delay > 0:
            time.sleep(delay)

    return PlayResult(
        score=int(info["score"]),
        steps=steps,
        health=int(info.get("health", 0)),
        achievements=int(info.get("achievements", info["score"])),
    )


def run_pygame(
    env: CraftaxClassicEnv,
    network: MLPQNetwork,
    seed: int,
    delay: float,
    max_steps: int,
    close_on_end: bool,
    block_pixel_size: int,
) -> PlayResult:
    try:
        import pygame
    except ImportError as exc:
        raise RuntimeError(
            "pygame is not installed. Install dependencies with `pip install -r requirements.txt`."
        ) from exc

    pygame.init()
    try:
        viewer = CraftaxPygameViewer(block_pixel_size=block_pixel_size)
        clock = pygame.time.Clock()
        state = env.reset(seed=seed)
        done = False
        steps = 0
        info: dict[str, Any] = {"score": 0, "health": 9, "achievements": 0}
        next_step_ts = time.monotonic()
        end_ts: float | None = None
        running = True
        paused = False
        step_delay = max(MIN_DELAY, float(delay))
        last_action_name = "—"
        print(
            "Craftax-Classic autoplay. Q/ESC quit | Space pause | "
            "[ slower | ] faster"
        )

        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key in (pygame.K_q, pygame.K_ESCAPE):
                        running = False
                    elif event.key == pygame.K_SPACE:
                        paused = not paused
                        if not paused:
                            next_step_ts = time.monotonic()
                    elif event.key == pygame.K_LEFTBRACKET:
                        step_delay = min(MAX_DELAY, step_delay + 0.05)
                    elif event.key == pygame.K_RIGHTBRACKET:
                        step_delay = max(MIN_DELAY, step_delay - 0.05)

            now = time.monotonic()
            if (
                not paused
                and not done
                and steps < max_steps
                and now >= next_step_ts
            ):
                action = pick_action(network, state, env.legal_actions())
                last_action_name = action_label(action)
                state, _, done, info = env.step(action)
                steps += 1
                next_step_ts = now + step_delay
                if done or steps >= max_steps:
                    end_ts = now

            game_state = env.game.state
            if game_state is not None:
                status = last_action_name + ("  (PAUSED)" if paused else "")
                viewer.draw(
                    game_state,
                    score=int(info["score"]),
                    health=int(info.get("health", 0)),
                    steps=steps,
                    done=done or steps >= max_steps,
                    action_name=status,
                    delay=step_delay,
                )
            if close_on_end and end_ts is not None and now - end_ts > 1.0:
                running = False
            clock.tick(60)

        return PlayResult(
            score=int(info["score"]),
            steps=steps,
            health=int(info.get("health", 0)),
            achievements=int(info.get("achievements", info["score"])),
        )
    finally:
        pygame.quit()


def main() -> None:
    parser = argparse.ArgumentParser(description="Autoplay Craftax-Classic with a trained model")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--delay",
        type=float,
        default=DEFAULT_DELAY,
        help=f"Seconds between steps (default {DEFAULT_DELAY}).",
    )
    parser.add_argument("--max-steps", type=int, default=10000)
    parser.add_argument("--mode", choices=["terminal", "pygame"], default="terminal")
    parser.add_argument("--close-on-end", action="store_true")
    parser.add_argument("--map-size", type=int, default=64)
    parser.add_argument("--max-timesteps", type=int, default=10000)
    parser.add_argument(
        "--block-pixel-size",
        type=int,
        default=48,
        help="Tile size in pixels (Classic human default is 64).",
    )
    args = parser.parse_args()

    env = CraftaxClassicEnv(
        config=CraftaxClassicConfig(
            map_size=(args.map_size, args.map_size),
            max_timesteps=args.max_timesteps,
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
            args.block_pixel_size,
        )
    else:
        result = run_terminal(env, network, args.seed, args.delay, args.max_steps)

    print("Craftax-Classic game finished")
    print(f"Achievements: {result.achievements}/22")
    print(f"Health: {result.health}")
    print(f"Steps: {result.steps}")


if __name__ == "__main__":
    main()
