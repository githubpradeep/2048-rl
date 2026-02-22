from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Sequence

import numpy as np

from .games.env2048 import Game2048Env
from .plugins.flappy.env_config import apply_flappy_preset, build_flappy_config_from_args, validate_model_env_or_raise
from .plugins.flappy.heuristic import FlappyHeuristicPolicy
from .plugins.flappy.tabular import FlappyTabularQAgent
from .games.breakout import BreakoutConfig, BreakoutEnv
from .games.flappy import FlappyEnv
from .games.fruit_cutter import FruitCutterConfig, FruitCutterEnv
from .games.match3 import Match3Config, Match3Env
from .games.pong import PongConfig, PongEnv
from .games.shooter import ShooterConfig, ShooterEnv
from .games.snake import SnakeConfig, SnakeEnv, SnakeFeatureEnv
from .games.tetris import TetrisConfig, TetrisEnv, TetrisPlacementEnv
from .network import MLPQNetwork
from .plugins.tetris.afterstate_utils import pick_afterstate_action

SUPPORTED_ENVS = (
    "2048",
    "breakout",
    "flappy",
    "flappy_heuristic",
    "flappy_tabular",
    "fruit",
    "match3",
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
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config file must contain a top-level mapping: {path}")
    return data


def _normalize_section(value: Any, section_name: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"Config section `{section_name}` must be a mapping/object")
    return {str(k).replace('-', '_'): v for k, v in value.items()}


def _parse_overrides(tokens: list[str]) -> dict[str, Any]:
    overrides: dict[str, Any] = {}
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if not tok.startswith("--"):
            raise ValueError(f"Unexpected positional override token: {tok}")
        key = tok[2:].replace('-', '_')
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


def _fallback_play_from_eval(eval_cfg: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in eval_cfg.items() if k not in {"episodes"}}


def _cfg(params: dict[str, Any], key: str, default: Any) -> Any:
    return params.get(key, default)


def _print_json_params(params: dict[str, Any]) -> None:
    print(json.dumps(params, sort_keys=True, indent=2))


def _flappy_ns(params: dict[str, Any]) -> SimpleNamespace:
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
        env_preset=_cfg(params, "env_preset", None),
        allow_env_mismatch=bool(_cfg(params, "allow_env_mismatch", False)),
        print_model_env=bool(_cfg(params, "print_model_env", False)),
        max_steps=int(_cfg(params, "max_steps", 1000)),
    )


def _build_env_and_policy(env_name: str, params: dict[str, Any]):
    seed = int(_cfg(params, "seed", 2026))
    model = params.get("model")
    kind = "dqn"

    if env_name == "2048":
        env = Game2048Env(seed=seed)
        policy = MLPQNetwork.load(str(model))
    elif env_name == "snake":
        grid = int(_cfg(params, "grid_size", 10))
        state_mode = str(_cfg(params, "state_mode", "board"))
        if state_mode == "features":
            env = SnakeFeatureEnv(config=SnakeConfig(grid_size=grid), seed=seed)
        else:
            state_grid = int(_cfg(params, "state_grid_size", 0)) or grid
            env = SnakeEnv(config=SnakeConfig(grid_size=grid), seed=seed, state_grid_size=state_grid)
        policy = MLPQNetwork.load(str(model))
    elif env_name == "fruit":
        grid = int(_cfg(params, "grid_size", 10))
        state_grid = int(_cfg(params, "state_grid_size", 0)) or grid
        env = FruitCutterEnv(
            config=FruitCutterConfig(grid_size=grid, spawn_prob=float(_cfg(params, "spawn_prob", 0.45)), bomb_prob=float(_cfg(params, "bomb_prob", 0.18))),
            seed=seed,
            state_grid_size=state_grid,
        )
        policy = MLPQNetwork.load(str(model))
    elif env_name == "shooter":
        grid = int(_cfg(params, "grid_size", 10))
        state_grid = int(_cfg(params, "state_grid_size", 0)) or grid
        env = ShooterEnv(
            config=ShooterConfig(grid_size=grid, spawn_prob=float(_cfg(params, "spawn_prob", 0.35)), max_steps=int(_cfg(params, "max_steps", 1000))),
            seed=seed,
            state_grid_size=state_grid,
        )
        policy = MLPQNetwork.load(str(model))
    elif env_name == "tetris":
        cfg = TetrisConfig(height=int(_cfg(params, "height", 20)), width=int(_cfg(params, "width", 10)), max_steps=int(_cfg(params, "max_steps", 1200)))
        env = TetrisPlacementEnv(config=cfg, seed=seed) if bool(_cfg(params, "placement_actions", False)) else TetrisEnv(config=cfg, seed=seed)
        policy = MLPQNetwork.load(str(model))
    elif env_name == "tetris_afterstate":
        cfg = TetrisConfig(height=int(_cfg(params, "height", 20)), width=int(_cfg(params, "width", 10)), max_steps=int(_cfg(params, "max_steps", 1200)))
        env = TetrisPlacementEnv(config=cfg, seed=seed)
        policy = MLPQNetwork.load(str(model))
        kind = "tetris_afterstate"
    elif env_name in {"flappy", "flappy_tabular", "flappy_heuristic"}:
        ns = _flappy_ns(params)
        if env_name == "flappy_tabular":
            ns.step_reward = float(_cfg(params, "step_reward", 1.0))
            ns.pass_reward = float(_cfg(params, "pass_reward", 100.0))
            ns.crash_penalty = float(_cfg(params, "crash_penalty", -100.0))
        apply_flappy_preset(ns)
        include_rewards = env_name == "flappy_tabular"
        cfg = build_flappy_config_from_args(ns, include_rewards=include_rewards)
        if env_name in {"flappy", "flappy_tabular"} and model:
            validate_model_env_or_raise(str(model), cfg, allow_mismatch=bool(ns.allow_env_mismatch), print_model_env=bool(ns.print_model_env))
        env = FlappyEnv(config=cfg, seed=seed)
        if env_name == "flappy":
            policy = MLPQNetwork.load(str(model))
        elif env_name == "flappy_tabular":
            policy = FlappyTabularQAgent.load(str(model))
            kind = "tabular"
        else:
            policy = FlappyHeuristicPolicy(
                bird_y_threshold=float(_cfg(params, "bird_y_threshold", 0.62)),
                dy_flap_threshold=float(_cfg(params, "dy_flap_threshold", 0.08)),
                vel_flap_threshold=float(_cfg(params, "vel_flap_threshold", 0.70)),
                bottom_emergency_y=float(_cfg(params, "bottom_emergency_y", 0.90)),
            )
            kind = "heuristic"
    elif env_name == "breakout":
        env = BreakoutEnv(
            config=BreakoutConfig(
                width=int(_cfg(params, "width", 12)), height=int(_cfg(params, "height", 14)), brick_rows=int(_cfg(params, "brick_rows", 4)),
                brick_top=int(_cfg(params, "brick_top", 1)), paddle_width=int(_cfg(params, "paddle_width", 3)), start_lives=int(_cfg(params, "start_lives", 3)),
                step_reward=float(_cfg(params, "step_reward", 0.02)), brick_reward=float(_cfg(params, "brick_reward", 1.0)),
                paddle_hit_reward=float(_cfg(params, "paddle_hit_reward", 0.03)), life_loss_penalty=float(_cfg(params, "life_loss_penalty", -3.0)),
                clear_bonus=float(_cfg(params, "clear_bonus", 5.0)), max_steps=int(_cfg(params, "max_steps", 1200)),
            ), seed=seed,
        )
        policy = MLPQNetwork.load(str(model))
    elif env_name == "pong":
        env = PongEnv(
            config=PongConfig(
                width=int(_cfg(params, "width", 12)), height=int(_cfg(params, "height", 16)), paddle_height=int(_cfg(params, "paddle_height", 4)),
                paddle_speed=int(_cfg(params, "paddle_speed", 1)), opponent_track_prob=float(_cfg(params, "opponent_track_prob", 0.85)),
                start_lives=int(_cfg(params, "start_lives", 3)), step_reward=float(_cfg(params, "step_reward", 0.01)),
                paddle_hit_reward=float(_cfg(params, "paddle_hit_reward", 0.05)), score_reward=float(_cfg(params, "score_reward", 2.0)),
                concede_penalty=float(_cfg(params, "concede_penalty", -2.0)), max_steps=int(_cfg(params, "max_steps", 1000)),
            ), seed=seed,
        )
        policy = MLPQNetwork.load(str(model))
    elif env_name == "match3":
        env = Match3Env(
            config=Match3Config(
                width=int(_cfg(params, "width", 6)), height=int(_cfg(params, "height", 6)), num_colors=int(_cfg(params, "num_colors", 5)),
                max_steps=int(_cfg(params, "max_steps", 150)), step_reward=float(_cfg(params, "step_reward", -0.01)), tile_reward=float(_cfg(params, "tile_reward", 0.20)),
                combo_bonus=float(_cfg(params, "combo_bonus", 0.15)), invalid_penalty=float(_cfg(params, "invalid_penalty", -0.50)),
                reshuffle_on_stuck=bool(_cfg(params, "reshuffle_on_stuck", True)),
            ), seed=seed,
        )
        policy = MLPQNetwork.load(str(model))
    else:
        raise ValueError(f"Unsupported env: {env_name}")

    return env, policy, kind


def _legal_actions(env) -> list[int]:
    if hasattr(env, "legal_actions"):
        return list(env.legal_actions())
    if hasattr(env, "game") and hasattr(env.game, "legal_actions"):
        return list(env.game.legal_actions())
    return []


def _pick_action(env_name: str, env, policy, state: np.ndarray) -> int:
    if env_name == "tetris_afterstate":
        return int(pick_afterstate_action(env, policy))
    q_values = policy.predict_one(state)
    legal = _legal_actions(env)
    if not legal:
        return int(max(range(len(q_values)), key=q_values.__getitem__))
    return int(max(legal, key=lambda a: q_values[a]))


def _board_matrix(env_name: str, env):
    if env_name == "2048":
        return env.game.board
    game = getattr(env, "game", None)
    if game is None:
        return None
    if env_name in {"tetris", "tetris_afterstate"} and hasattr(game, "board_with_piece"):
        return game.board_with_piece()
    if hasattr(game, "board"):
        board_attr = game.board
        if callable(board_attr):
            return board_attr()
        return board_attr
    if hasattr(game, "board_arr"):
        return game.board_arr
    return None


class GridPygameRenderer:
    def __init__(self, rows: int, cols: int, title: str, cell_size: int = 34, margin: int = 8, header_h: int = 96) -> None:
        import pygame

        self.pygame = pygame
        self.rows = rows
        self.cols = cols
        self.title = title
        self.cell = cell_size
        self.margin = margin
        self.header_h = header_h
        w = margin + cols * (cell_size + margin)
        h = header_h + margin + rows * (cell_size + margin)
        self.screen = pygame.display.set_mode((w, h))
        pygame.display.set_caption(title)
        self.bg = (245, 248, 252)
        self.board_bg = (229, 236, 246)
        self.text = (35, 47, 64)
        self.title_font = pygame.font.SysFont("arial", 28, bold=True)
        self.info_font = pygame.font.SysFont("arial", 18, bold=True)
        self.tile_font = pygame.font.SysFont("arial", 18, bold=True)

    def _rect(self, r: int, c: int):
        return self.pygame.Rect(
            self.margin + c * (self.cell + self.margin),
            self.header_h + self.margin + r * (self.cell + self.margin),
            self.cell,
            self.cell,
        )

    def _color(self, v: int) -> tuple[int, int, int]:
        palette = [
            (236, 242, 249), (88, 140, 242), (232, 90, 88), (90, 191, 122), (246, 198, 84),
            (160, 110, 220), (98, 186, 192), (240, 133, 70), (122, 130, 150), (60, 70, 90),
        ]
        if v == 0:
            return palette[0]
        if v in {2,4,8,16,32,64,128,256,512,1024,2048}:
            # 2048-ish warm palette mapping by log2
            idx = min(9, int(np.log2(max(v, 2))))
            warm = [
                (238,228,218),(237,224,200),(242,177,121),(245,149,99),(246,124,95),
                (246,94,59),(237,207,114),(237,204,97),(237,200,80),(237,194,46),
            ]
            return warm[min(idx-1, len(warm)-1)] if idx > 0 else palette[0]
        return palette[(abs(int(v)) % (len(palette)-1)) + 1]

    def draw(self, board: np.ndarray, info_lines: list[str], banner: str | None = None) -> None:
        pygame = self.pygame
        self.screen.fill(self.bg)
        self.screen.blit(self.title_font.render(self.title, True, self.text), (self.margin, 10))
        y = 42
        for line in info_lines[:3]:
            self.screen.blit(self.info_font.render(line, True, self.text), (self.margin, y))
            y += 20
        if banner:
            self.screen.blit(self.info_font.render(banner, True, (140, 70, 40)), (self.margin + 260, 42))
        arena = pygame.Rect(0, self.header_h, self.screen.get_width(), self.screen.get_height() - self.header_h)
        pygame.draw.rect(self.screen, self.board_bg, arena)
        rows, cols = board.shape
        for r in range(rows):
            for c in range(cols):
                rect = self._rect(r, c)
                v = int(board[r, c])
                pygame.draw.rect(self.screen, self._color(v), rect, border_radius=6)
                if v != 0 and self.cell >= 24:
                    txt = str(v)
                    surf = self.tile_font.render(txt, True, (32, 32, 32))
                    self.screen.blit(surf, surf.get_rect(center=rect.center))
        pygame.display.flip()


def _match3_animate(renderer: GridPygameRenderer, env: Match3Env, info: dict[str, Any], steps: int, done: bool, *, pre_board: np.ndarray, swap_cells, select_delay: float, move_delay: float, clear_delay: float) -> bool:
    pygame = renderer.pygame
    score = int(info.get("score", 0))
    tiles = int(info.get("total_tiles_cleared", 0))
    casc = int(info.get("total_cascades", 0))

    def pump_until(sec: float, board: np.ndarray, banner: str | None = None):
        end_t = time.monotonic() + max(0.0, sec)
        while time.monotonic() < end_t:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return False
                if event.type == pygame.KEYDOWN and event.key in (pygame.K_q, pygame.K_ESCAPE):
                    return False
            renderer.draw(board, [f"Score: {score}", f"Steps: {steps}", f"Tiles: {tiles} Cascades: {casc}"], banner=banner)
            pygame.time.delay(8)
        return True

    if swap_cells:
        if not pump_until(select_delay, pre_board, banner="select"):
            return False
        swapped = env.game.last_swapped_board
        if swapped is not None and not pump_until(move_delay, swapped, banner="moved"):
            return False
    for idx, frame in enumerate(env.game.last_cascade_frames, start=1):
        board_before, mask = frame
        # crude flash by zeroing popped cells in view
        flash = board_before.copy()
        flash[mask.astype(bool)] = 0
        if not pump_until(clear_delay, flash, banner=f"pop x{idx}"):
            return False
    return True


def run_terminal(env_name: str, env, policy, params: dict[str, Any]) -> dict[str, Any]:
    seed = int(_cfg(params, "seed", 2026))
    delay = float(_cfg(params, "delay", 0.08))
    max_steps = int(_cfg(params, "max_steps", 2000))
    state = env.reset(seed=seed)
    done = False
    steps = 0
    info: dict[str, Any] = {}
    print(f"Starting {env_name} autoplay in terminal mode...\n")
    while not done and steps < max_steps:
        action = _pick_action(env_name, env, policy, state)
        state, reward, done, info = env.step(action)
        steps += 1
        print(f"Step {steps} | Action {action} | Reward {float(reward):.2f}")
        print(env.render())
        print("=" * 70)
        if delay > 0:
            time.sleep(delay)
    return {"steps": steps, **{k: v for k, v in info.items() if isinstance(v, (int, float, bool))}}


def run_pygame(env_name: str, env, policy, params: dict[str, Any]) -> dict[str, Any]:
    try:
        import pygame
    except ImportError as exc:
        raise RuntimeError("pygame is not installed. Install dependencies with `pip install -r requirements.txt`.") from exc

    seed = int(_cfg(params, "seed", 2026))
    delay = float(_cfg(params, "delay", 0.08))
    max_steps = int(_cfg(params, "max_steps", 2000))
    close_on_end = bool(_cfg(params, "close_on_end", False))

    pygame.init()
    try:
        state = env.reset(seed=seed)
        if env_name == "match3" and hasattr(env, "game"):
            env.game.capture_step_traces = True
        board = _board_matrix(env_name, env)
        if board is None:
            raise RuntimeError(f"No board renderer available for env {env_name}")
        rows, cols = int(board.shape[0]), int(board.shape[1])
        renderer = GridPygameRenderer(rows, cols, title=f"{env_name} autoplay")
        clock = pygame.time.Clock()
        done = False
        steps = 0
        info: dict[str, Any] = {}
        next_step_ts = time.monotonic()
        end_ts: float | None = None
        running = True

        select_delay = float(_cfg(params, "select_delay", min(0.18, max(0.06, delay * 0.7))))
        move_delay = float(_cfg(params, "move_delay", min(0.18, max(0.06, delay * 0.7))))
        clear_delay = float(_cfg(params, "clear_delay", min(0.14, max(0.04, delay * 0.5))))

        print(f"Starting {env_name} autoplay in pygame mode. Press Q or ESC to quit.")

        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN and event.key in (pygame.K_q, pygame.K_ESCAPE):
                    running = False

            now = time.monotonic()
            if not done and steps < max_steps and now >= next_step_ts:
                pre_board = _board_matrix(env_name, env)
                swap_cells = None
                if env_name == "match3":
                    legal = env.legal_actions()
                    action = _pick_action(env_name, env, policy, state)
                    try:
                        swap_cells = list(env.action_to_swap(action))
                    except Exception:
                        swap_cells = None
                else:
                    action = _pick_action(env_name, env, policy, state)
                state, _, done, info = env.step(action)
                steps += 1
                if env_name == "match3" and swap_cells is not None and pre_board is not None:
                    if not _match3_animate(renderer, env, info, steps, done or steps >= max_steps, pre_board=pre_board.copy(), swap_cells=swap_cells, select_delay=select_delay, move_delay=move_delay, clear_delay=clear_delay):
                        running = False
                        break
                next_step_ts = time.monotonic() + max(0.0, delay)
                if done or steps >= max_steps:
                    end_ts = time.monotonic()

            board = _board_matrix(env_name, env)
            info_lines = [
                f"Steps: {steps}/{max_steps}",
                f"Score: {int(info.get('score', 0))}",
                " ".join(f"{k}:{int(v)}" for k, v in info.items() if k in {"max_tile","lines","length","lives","opponent_score"}),
            ]
            renderer.draw(np.asarray(board), [line for line in info_lines if line], banner=("done" if (done or steps >= max_steps) else None))

            if close_on_end and end_ts is not None and now - end_ts > 1.0:
                running = False
            clock.tick(60)

        return {"steps": steps, **{k: v for k, v in info.items() if isinstance(v, (int, float, bool))}}
    finally:
        pygame.quit()


def main() -> None:
    parser = argparse.ArgumentParser(description="Unified generic player (YAML-driven)")
    parser.add_argument("--env", required=True, choices=sorted(SUPPORTED_ENVS))
    parser.add_argument("--config", type=str, default=None, help="YAML config file (defaults to sample_configs/<env>.yaml)")
    parser.add_argument("--dry-run", action="store_true", help="Print merged play params and exit")
    args, rest = parser.parse_known_args()

    config_path = Path(args.config) if args.config else Path("sample_configs") / f"{args.env}.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    config = _load_yaml(config_path)
    declared_env = config.get("env")
    if declared_env is not None and str(declared_env) != args.env:
        raise ValueError(f"Config env mismatch: config has env={declared_env!r}, CLI requested env={args.env!r}")

    common = _normalize_section(config.get("common"), "common")
    play_cfg = _normalize_section(config.get("play"), "play") or _fallback_play_from_eval(_normalize_section(config.get("eval"), "eval"))
    if not play_cfg:
        raise ValueError(f"Config has neither `play` nor usable `eval` section: {config_path}")
    params = {**common, **play_cfg}
    if rest:
        params.update(_parse_overrides(rest))

    print(f"Config: {config_path}")
    if args.dry_run:
        _print_json_params(params)
        return

    env, policy, _ = _build_env_and_policy(args.env, params)
    mode = str(_cfg(params, "mode", "terminal"))
    result = run_pygame(args.env, env, policy, params) if mode == "pygame" else run_terminal(args.env, env, policy, params)

    print(f"{args.env} game finished")
    for key in ("score", "max_tile", "lines", "length", "lives", "opponent_score", "passed_pipes", "total_tiles_cleared", "total_cascades"):
        if key in result:
            label = key.replace("_", " ").title()
            print(f"{label}: {result[key]}")
    print(f"Steps: {result.get('steps', 0)}")


if __name__ == "__main__":
    main()
