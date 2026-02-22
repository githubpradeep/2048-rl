from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any


SUPPORTED_ENVS = (
    "2048",
    "breakout",
    "flappy",
    "flappy_heuristic",
    "flappy_tabular",
    "fruit",
    "match3",
    "pacman",
    "pong",
    "shooter",
    "snake",
    "tetris",
    "tetris_afterstate",
)


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml  # type: ignore
    except ImportError as exc:
        raise RuntimeError("PyYAML is required. Install with `pip install -r requirements.txt`.") from exc

    raw = path.read_text(encoding="utf-8")
    data = yaml.safe_load(raw) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config file must contain a top-level mapping: {path}")
    return data


def _normalize_section(value: Any, section_name: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"Config section `{section_name}` must be a mapping/object")
    normalized: dict[str, Any] = {}
    for key, item in value.items():
        normalized[str(key).replace("-", "_")] = item
    return normalized


def _cfg(params: dict[str, Any], key: str, default: Any) -> Any:
    return params.get(key, default)


def _print_json_params(params: dict[str, Any]) -> None:
    print(json.dumps(params, sort_keys=True, indent=2))


def _parse_overrides(tokens: list[str]) -> dict[str, Any]:
    overrides: dict[str, Any] = {}
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if not tok.startswith("--"):
            raise ValueError(f"Unexpected positional override token: {tok}")
        key = tok[2:].replace("-", "_")
        if key.startswith("no_"):
            overrides[key[3:]] = False
            i += 1
            continue
        if i + 1 < len(tokens) and not tokens[i + 1].startswith("--"):
            raw = tokens[i + 1]
            low = raw.lower()
            if low in {"true", "false"}:
                overrides[key] = (low == "true")
            else:
                overrides[key] = raw
            i += 2
            continue
        overrides[key] = True
        i += 1
    return overrides


def _eval_2048(params: dict[str, Any]) -> None:
    from .games.env2048 import Game2048Env
    from .evals.eval_utils import evaluate_policy
    from .network import MLPQNetwork

    episodes = int(_cfg(params, "episodes", 100))
    seed = int(_cfg(params, "seed", 123))
    model = str(params["model"])

    env = Game2048Env(seed=seed)
    network = MLPQNetwork.load(model)
    stats = evaluate_policy(env, network, episodes=episodes, seed_start=seed)

    print("Evaluation Results")
    print("------------------")
    print(f"Episodes: {episodes}")
    print(f"Average score: {stats.avg_score:.2f}")
    print(f"Median score: {stats.median_score:.2f}")
    print(f"Average steps: {stats.avg_steps:.2f}")
    print(f"Reach >= 512:  {stats.reach_512:.2%}")
    print(f"Reach >= 1024: {stats.reach_1024:.2%}")
    print(f"Reach >= 2048: {stats.reach_2048:.2%}")
    print("Max tile distribution:")
    for tile in sorted(stats.tile_distribution):
        print(f"  {tile:4d}: {stats.tile_distribution[tile]}")


def _eval_snake(params: dict[str, Any]) -> None:
    from .games.snake import SnakeConfig, SnakeEnv, SnakeFeatureEnv
    from .network import MLPQNetwork
    from .evals.snake_eval_utils import evaluate_snake_policy

    episodes = int(_cfg(params, "episodes", 100))
    seed = int(_cfg(params, "seed", 123))
    grid_size = int(_cfg(params, "grid_size", 10))
    state_grid_size = int(_cfg(params, "state_grid_size", 0)) or grid_size
    state_mode = str(_cfg(params, "state_mode", "board"))
    model = str(params["model"])

    if state_mode == "features":
        env = SnakeFeatureEnv(config=SnakeConfig(grid_size=grid_size), seed=seed)
    else:
        env = SnakeEnv(config=SnakeConfig(grid_size=grid_size), seed=seed, state_grid_size=state_grid_size)
    network = MLPQNetwork.load(model)
    stats = evaluate_snake_policy(env, network, episodes=episodes, seed_start=seed)

    print("Snake Evaluation Results")
    print("------------------------")
    print(f"Episodes: {episodes}")
    print(f"Average score: {stats.avg_score:.3f}")
    print(f"Median score: {stats.median_score:.3f}")
    print(f"Average steps: {stats.avg_steps:.2f}")
    print(f"Average length: {stats.avg_length:.2f}")
    print(f"Food rate (foods/step): {stats.food_rate:.5f}")


def _eval_fruit(params: dict[str, Any]) -> None:
    from .evals.fruit_eval_utils import evaluate_fruit_policy
    from .games.fruit_cutter import FruitCutterConfig, FruitCutterEnv
    from .network import MLPQNetwork

    episodes = int(_cfg(params, "episodes", 100))
    seed = int(_cfg(params, "seed", 123))
    grid_size = int(_cfg(params, "grid_size", 10))
    state_grid_size = int(_cfg(params, "state_grid_size", 0)) or grid_size
    spawn_prob = float(_cfg(params, "spawn_prob", 0.45))
    bomb_prob = float(_cfg(params, "bomb_prob", 0.18))
    model = str(params["model"])

    env = FruitCutterEnv(
        config=FruitCutterConfig(grid_size=grid_size, spawn_prob=spawn_prob, bomb_prob=bomb_prob),
        seed=seed,
        state_grid_size=state_grid_size,
    )
    network = MLPQNetwork.load(model)
    stats = evaluate_fruit_policy(env, network, episodes=episodes, seed_start=seed)

    print("Fruit Cutter Evaluation Results")
    print("-------------------------------")
    print(f"Episodes: {episodes}")
    print(f"Average score: {stats.avg_score:.3f}")
    print(f"Median score: {stats.median_score:.3f}")
    print(f"Average steps: {stats.avg_steps:.2f}")
    print(f"Average sliced/episode: {stats.avg_sliced_per_episode:.2f}")
    print(f"Average missed/episode: {stats.avg_missed_per_episode:.2f}")


def _eval_shooter(params: dict[str, Any]) -> None:
    from .games.shooter import ShooterConfig, ShooterEnv
    from .network import MLPQNetwork
    from .evals.shooter_eval_utils import evaluate_shooter_policy

    episodes = int(_cfg(params, "episodes", 100))
    seed = int(_cfg(params, "seed", 123))
    grid_size = int(_cfg(params, "grid_size", 10))
    state_grid_size = int(_cfg(params, "state_grid_size", 0)) or grid_size
    spawn_prob = float(_cfg(params, "spawn_prob", 0.35))
    max_steps = int(_cfg(params, "max_steps", 1000))
    model = str(params["model"])

    env = ShooterEnv(
        config=ShooterConfig(grid_size=grid_size, spawn_prob=spawn_prob, max_steps=max_steps),
        seed=seed,
        state_grid_size=state_grid_size,
    )
    network = MLPQNetwork.load(model)
    stats = evaluate_shooter_policy(env, network, episodes=episodes, seed_start=seed)

    print("Shooter Evaluation Results")
    print("--------------------------")
    print(f"Episodes: {episodes}")
    print(f"Average score: {stats.avg_score:.3f}")
    print(f"Median score: {stats.median_score:.3f}")
    print(f"Average steps: {stats.avg_steps:.2f}")
    print(f"Average lives left: {stats.avg_lives_left:.2f}")
    print(f"Average kills/episode: {stats.avg_kills_per_episode:.2f}")
    print(f"Average escaped/episode: {stats.avg_escaped_per_episode:.2f}")


def _eval_tetris(params: dict[str, Any]) -> None:
    from .games.tetris import TetrisConfig, TetrisEnv, TetrisPlacementEnv
    from .network import MLPQNetwork
    from .evals.tetris_eval_utils import evaluate_tetris_policy

    episodes = int(_cfg(params, "episodes", 100))
    seed = int(_cfg(params, "seed", 123))
    height = int(_cfg(params, "height", 20))
    width = int(_cfg(params, "width", 10))
    max_steps = int(_cfg(params, "max_steps", 1200))
    placement_actions = bool(_cfg(params, "placement_actions", False))
    model = str(params["model"])

    cfg = TetrisConfig(height=height, width=width, max_steps=max_steps)
    env = TetrisPlacementEnv(config=cfg, seed=seed) if placement_actions else TetrisEnv(config=cfg, seed=seed)
    network = MLPQNetwork.load(model)
    stats = evaluate_tetris_policy(env, network, episodes=episodes, seed_start=seed, max_steps=max_steps)

    print("Tetris Evaluation Results")
    print("-------------------------")
    print(f"Episodes: {episodes}")
    print(f"Average score: {stats.avg_score:.3f}")
    print(f"Median score: {stats.median_score:.3f}")
    print(f"Average lines: {stats.avg_lines:.2f}")
    print(f"Average steps: {stats.avg_steps:.2f}")


def _eval_tetris_afterstate(params: dict[str, Any]) -> None:
    from .games.tetris import TetrisConfig, TetrisPlacementEnv
    from .network import MLPQNetwork
    from .plugins.tetris.afterstate_utils import evaluate_afterstate_policy

    episodes = int(_cfg(params, "episodes", 100))
    seed = int(_cfg(params, "seed", 123))
    height = int(_cfg(params, "height", 20))
    width = int(_cfg(params, "width", 10))
    max_steps = int(_cfg(params, "max_steps", 1200))
    model = str(params["model"])

    env = TetrisPlacementEnv(config=TetrisConfig(height=height, width=width, max_steps=max_steps), seed=seed)
    network = MLPQNetwork.load(model)
    stats = evaluate_afterstate_policy(env, network, episodes=episodes, seed_start=seed, max_steps=max_steps)

    print("Tetris Afterstate Evaluation Results")
    print("------------------------------------")
    print(f"Episodes: {episodes}")
    print(f"Average score: {stats.avg_score:.3f}")
    print(f"Median score: {stats.median_score:.3f}")
    print(f"Average lines: {stats.avg_lines:.2f}")
    print(f"Average steps: {stats.avg_steps:.2f}")


def _build_flappy_namespace(params: dict[str, Any]) -> SimpleNamespace:
    return SimpleNamespace(
        width=int(_cfg(params, "width", 84)),
        height=int(_cfg(params, "height", 84)),
        pipe_gap=float(_cfg(params, "pipe_gap", 30.0)),
        pipe_speed=float(_cfg(params, "pipe_speed", 1.3)),
        pipe_spacing=float(_cfg(params, "pipe_spacing", 44.0)),
        initial_pipe_offset=float(_cfg(params, "initial_pipe_offset", 20.0)),
        floor_height=float(_cfg(params, "floor_height", 10.0)),
        max_gap_delta=_cfg(params, "max_gap_delta", 8.0),
        gravity=float(_cfg(params, "gravity", 0.18)),
        flap_velocity=float(_cfg(params, "flap_velocity", -2.2)),
        step_reward=float(_cfg(params, "step_reward", 0.02)),
        pass_reward=float(_cfg(params, "pass_reward", 5.0)),
        crash_penalty=float(_cfg(params, "crash_penalty", -3.0)),
        max_steps=int(_cfg(params, "max_steps", 1000)),
        env_preset=_cfg(params, "env_preset", None),
        allow_env_mismatch=bool(_cfg(params, "allow_env_mismatch", False)),
        print_model_env=bool(_cfg(params, "print_model_env", False)),
    )


def _eval_flappy(params: dict[str, Any]) -> None:
    from .plugins.flappy.env_config import apply_flappy_preset, build_flappy_config_from_args, validate_model_env_or_raise
    from .evals.flappy_eval_utils import evaluate_flappy_policy
    from .games.flappy import FlappyEnv
    from .network import MLPQNetwork

    episodes = int(_cfg(params, "episodes", 100))
    seed = int(_cfg(params, "seed", 123))
    model = str(params["model"])

    ns = _build_flappy_namespace(params)
    preset_name = apply_flappy_preset(ns)
    config = build_flappy_config_from_args(ns, include_rewards=False)
    validate_model_env_or_raise(
        model,
        config,
        allow_mismatch=bool(ns.allow_env_mismatch),
        print_model_env=bool(ns.print_model_env),
    )
    env = FlappyEnv(config=config, seed=seed)
    network = MLPQNetwork.load(model)
    stats = evaluate_flappy_policy(env, network, episodes=episodes, seed_start=seed, max_steps=int(ns.max_steps))

    print("Flappy Evaluation Results")
    print("-------------------------")
    print(f"Episodes: {episodes}")
    print(f"Average score: {stats.avg_score:.3f}")
    print(f"Median score: {stats.median_score:.3f}")
    print(f"Average steps: {stats.avg_steps:.2f}")
    print(f"Average reward: {stats.avg_reward:.2f}")
    print(f"Env preset: {preset_name or 'custom'}")


def _eval_flappy_tabular(params: dict[str, Any]) -> None:
    from .plugins.flappy.env_config import apply_flappy_preset, build_flappy_config_from_args, validate_model_env_or_raise
    from .evals.flappy_eval_utils import evaluate_flappy_policy
    from .plugins.flappy.tabular import FlappyTabularQAgent
    from .games.flappy import FlappyEnv

    episodes = int(_cfg(params, "episodes", 100))
    seed = int(_cfg(params, "seed", 123))
    model = str(params["model"])

    ns = _build_flappy_namespace(params)
    # Tabular eval defaults differ from DQN reward defaults; override builder inputs from params if provided.
    ns.step_reward = float(_cfg(params, "step_reward", 1.0))
    ns.pass_reward = float(_cfg(params, "pass_reward", 100.0))
    ns.crash_penalty = float(_cfg(params, "crash_penalty", -100.0))
    preset_name = apply_flappy_preset(ns)
    config = build_flappy_config_from_args(ns, include_rewards=True)
    validate_model_env_or_raise(
        model,
        config,
        allow_mismatch=bool(ns.allow_env_mismatch),
        print_model_env=bool(ns.print_model_env),
    )
    env = FlappyEnv(config=config, seed=seed)
    agent = FlappyTabularQAgent.load(model)
    stats = evaluate_flappy_policy(env, agent, episodes=episodes, seed_start=seed, max_steps=int(ns.max_steps))

    print("Flappy Tabular Evaluation Results")
    print("--------------------------------")
    print(f"Episodes: {episodes}")
    print(f"Average score: {stats.avg_score:.3f}")
    print(f"Median score: {stats.median_score:.3f}")
    print(f"Average steps: {stats.avg_steps:.2f}")
    print(f"Average reward: {stats.avg_reward:.2f}")
    print(f"Env preset: {preset_name or 'custom'}")


def _eval_flappy_heuristic(params: dict[str, Any]) -> None:
    from .plugins.flappy.env_config import apply_flappy_preset, build_flappy_config_from_args
    from .evals.flappy_eval_utils import evaluate_flappy_policy
    from .plugins.flappy.heuristic import FlappyHeuristicPolicy
    from .games.flappy import FlappyEnv

    episodes = int(_cfg(params, "episodes", 100))
    seed = int(_cfg(params, "seed", 123))

    ns = _build_flappy_namespace(params)
    preset_name = apply_flappy_preset(ns)
    config = build_flappy_config_from_args(ns, include_rewards=False)
    env = FlappyEnv(config=config, seed=seed)
    policy = FlappyHeuristicPolicy(
        bird_y_threshold=float(_cfg(params, "bird_y_threshold", 0.62)),
        dy_flap_threshold=float(_cfg(params, "dy_flap_threshold", 0.08)),
        vel_flap_threshold=float(_cfg(params, "vel_flap_threshold", 0.70)),
        bottom_emergency_y=float(_cfg(params, "bottom_emergency_y", 0.90)),
    )
    stats = evaluate_flappy_policy(env, policy, episodes=episodes, seed_start=seed, max_steps=int(ns.max_steps))

    print("Flappy Heuristic Evaluation Results")
    print("----------------------------------")
    print(f"Episodes: {episodes}")
    print(f"Average score: {stats.avg_score:.3f}")
    print(f"Median score: {stats.median_score:.3f}")
    print(f"Average steps: {stats.avg_steps:.2f}")
    print(f"Average reward: {stats.avg_reward:.2f}")
    print(f"Env preset: {preset_name or 'custom'}")


def _eval_breakout(params: dict[str, Any]) -> None:
    from .evals.breakout_eval_utils import evaluate_breakout_policy
    from .games.breakout import BreakoutConfig, BreakoutEnv
    from .network import MLPQNetwork

    episodes = int(_cfg(params, "episodes", 100))
    seed = int(_cfg(params, "seed", 123))
    max_steps = int(_cfg(params, "max_steps", 1200))
    model = str(params["model"])
    env = BreakoutEnv(
        config=BreakoutConfig(
            width=int(_cfg(params, "width", 12)),
            height=int(_cfg(params, "height", 14)),
            brick_rows=int(_cfg(params, "brick_rows", 4)),
            brick_top=int(_cfg(params, "brick_top", 1)),
            paddle_width=int(_cfg(params, "paddle_width", 3)),
            start_lives=int(_cfg(params, "start_lives", 3)),
            step_reward=float(_cfg(params, "step_reward", 0.02)),
            brick_reward=float(_cfg(params, "brick_reward", 1.0)),
            paddle_hit_reward=float(_cfg(params, "paddle_hit_reward", 0.03)),
            life_loss_penalty=float(_cfg(params, "life_loss_penalty", -3.0)),
            clear_bonus=float(_cfg(params, "clear_bonus", 5.0)),
            max_steps=max_steps,
        ),
        seed=seed,
    )
    network = MLPQNetwork.load(model)
    stats = evaluate_breakout_policy(env, network, episodes=episodes, seed_start=seed, max_steps=max_steps)

    print("Breakout Evaluation Results")
    print("--------------------------")
    print(f"Episodes: {episodes}")
    print(f"Average score: {stats.avg_score:.3f}")
    print(f"Median score: {stats.median_score:.3f}")
    print(f"Average steps: {stats.avg_steps:.2f}")
    print(f"Average lives left: {stats.avg_lives_left:.2f}")
    print(f"Average bricks broken: {stats.avg_bricks_broken:.2f}")
    print(f"Clear rate: {100.0 * stats.clear_rate:.2f}%")


def _eval_pong(params: dict[str, Any]) -> None:
    from .games.pong import PongConfig, PongEnv
    from .network import MLPQNetwork
    from .evals.pong_eval_utils import evaluate_pong_policy

    episodes = int(_cfg(params, "episodes", 100))
    seed = int(_cfg(params, "seed", 123))
    max_steps = int(_cfg(params, "max_steps", 1000))
    model = str(params["model"])

    env = PongEnv(
        config=PongConfig(
            width=int(_cfg(params, "width", 12)),
            height=int(_cfg(params, "height", 16)),
            paddle_height=int(_cfg(params, "paddle_height", 4)),
            paddle_speed=int(_cfg(params, "paddle_speed", 1)),
            opponent_track_prob=float(_cfg(params, "opponent_track_prob", 0.85)),
            start_lives=int(_cfg(params, "start_lives", 3)),
            step_reward=float(_cfg(params, "step_reward", 0.01)),
            paddle_hit_reward=float(_cfg(params, "paddle_hit_reward", 0.05)),
            score_reward=float(_cfg(params, "score_reward", 2.0)),
            concede_penalty=float(_cfg(params, "concede_penalty", -2.0)),
            max_steps=max_steps,
        ),
        seed=seed,
    )
    network = MLPQNetwork.load(model)
    stats = evaluate_pong_policy(env, network, episodes=episodes, seed_start=seed, max_steps=max_steps)

    print("Pong Evaluation Results")
    print("-----------------------")
    print(f"Episodes: {episodes}")
    print(f"Average score: {stats.avg_score:.3f}")
    print(f"Median score: {stats.median_score:.3f}")
    print(f"Average steps: {stats.avg_steps:.2f}")
    print(f"Average lives left: {stats.avg_lives_left:.2f}")
    print(f"Average opponent score: {stats.avg_opponent_score:.2f}")
    print(f"Average player hits: {stats.avg_player_hits:.2f}")


def _eval_match3(params: dict[str, Any]) -> None:
    from .games.match3 import Match3Config, Match3Env
    from .evals.match3_eval_utils import evaluate_match3_policy
    from .network import MLPQNetwork

    episodes = int(_cfg(params, "episodes", 100))
    seed = int(_cfg(params, "seed", 123))
    max_steps = int(_cfg(params, "max_steps", 150))
    model = str(params["model"])

    env = Match3Env(
        config=Match3Config(
            width=int(_cfg(params, "width", 6)),
            height=int(_cfg(params, "height", 6)),
            num_colors=int(_cfg(params, "num_colors", 5)),
            max_steps=max_steps,
            step_reward=float(_cfg(params, "step_reward", -0.01)),
            tile_reward=float(_cfg(params, "tile_reward", 0.20)),
            combo_bonus=float(_cfg(params, "combo_bonus", 0.15)),
            invalid_penalty=float(_cfg(params, "invalid_penalty", -0.50)),
            reshuffle_on_stuck=bool(_cfg(params, "reshuffle_on_stuck", True)),
        ),
        seed=seed,
    )
    network = MLPQNetwork.load(model)
    stats = evaluate_match3_policy(env, network, episodes=episodes, seed_start=seed, max_steps=max_steps)

    print("Match-3 Evaluation Results")
    print("--------------------------")
    print(f"Episodes: {episodes}")
    print(f"Average score: {stats.avg_score:.3f}")
    print(f"Median score: {stats.median_score:.3f}")
    print(f"Average steps: {stats.avg_steps:.2f}")
    print(f"Average tiles cleared: {stats.avg_tiles_cleared:.2f}")
    print(f"Average cascades: {stats.avg_cascades:.2f}")
    print(f"Invalid rate: {stats.invalid_rate:.4f}")


def _eval_pacman(params: dict[str, Any]) -> None:
    from .evals.pacman_eval_utils import evaluate_pacman_policy
    from .games.pacman_lite import PacmanLiteConfig, PacmanLiteEnv
    from .network import MLPQNetwork

    episodes = int(_cfg(params, "episodes", 100))
    seed = int(_cfg(params, "seed", 123))
    max_steps = int(_cfg(params, "max_steps", 500))
    grid_size = int(_cfg(params, "grid_size", 11))
    state_grid_size = int(_cfg(params, "state_grid_size", 0)) or grid_size
    model = str(params["model"])

    env = PacmanLiteEnv(
        config=PacmanLiteConfig(
            grid_size=grid_size,
            num_ghosts=int(_cfg(params, "num_ghosts", 2)),
            ghost_chase_prob=float(_cfg(params, "ghost_chase_prob", 0.75)),
            pellet_reward=float(_cfg(params, "pellet_reward", 1.0)),
            step_reward=float(_cfg(params, "step_reward", -0.02)),
            ghost_collision_penalty=float(_cfg(params, "ghost_collision_penalty", -5.0)),
            clear_bonus=float(_cfg(params, "clear_bonus", 8.0)),
            max_steps=max_steps,
            start_lives=int(_cfg(params, "start_lives", 3)),
        ),
        seed=seed,
        state_grid_size=state_grid_size,
    )
    network = MLPQNetwork.load(model)
    stats = evaluate_pacman_policy(env, network, episodes=episodes, seed_start=seed, max_steps=max_steps)

    print("Pacman-lite Evaluation Results")
    print("------------------------------")
    print(f"Episodes: {episodes}")
    print(f"Average score: {stats.avg_score:.3f}")
    print(f"Median score: {stats.median_score:.3f}")
    print(f"Average steps: {stats.avg_steps:.2f}")
    print(f"Average lives left: {stats.avg_lives_left:.2f}")
    print(f"Average pellets eaten: {stats.avg_pellets_eaten:.2f}")
    print(f"Average collisions: {stats.avg_collisions:.2f}")
    print(f"Clear rate: {100.0 * stats.clear_rate:.2f}%")


_RUNNERS = {
    "2048": _eval_2048,
    "snake": _eval_snake,
    "fruit": _eval_fruit,
    "shooter": _eval_shooter,
    "tetris": _eval_tetris,
    "tetris_afterstate": _eval_tetris_afterstate,
    "flappy": _eval_flappy,
    "flappy_tabular": _eval_flappy_tabular,
    "flappy_heuristic": _eval_flappy_heuristic,
    "breakout": _eval_breakout,
    "pong": _eval_pong,
    "match3": _eval_match3,
    "pacman": _eval_pacman,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Unified generic evaluator (YAML-driven)")
    parser.add_argument("--env", required=True, choices=sorted(SUPPORTED_ENVS))
    parser.add_argument("--config", type=str, default=None, help="YAML config file (defaults to sample_configs/<env>.yaml)")
    parser.add_argument("--dry-run", action="store_true", help="Print merged eval params and exit")
    args, rest = parser.parse_known_args()

    config_path = Path(args.config) if args.config else Path("sample_configs") / f"{args.env}.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    config = _load_yaml(config_path)
    declared_env = config.get("env")
    if declared_env is not None and str(declared_env) != args.env:
        raise ValueError(f"Config env mismatch: config has env={declared_env!r}, CLI requested env={args.env!r}")

    common = _normalize_section(config.get("common"), "common")
    eval_cfg = _normalize_section(config.get("eval"), "eval")
    if not eval_cfg:
        raise ValueError(f"Config has no `eval` section: {config_path}")
    params = {**common, **eval_cfg}
    if rest:
        params.update(_parse_overrides(rest))

    print(f"Config: {config_path}")
    if args.dry_run:
        _print_json_params(params)
        return

    runner = _RUNNERS[args.env]
    runner(params)


if __name__ == "__main__":
    main()
