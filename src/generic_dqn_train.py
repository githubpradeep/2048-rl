from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

import numpy as np

from .games.env2048 import EnvConfig, Game2048Env
from .evals.eval_utils import evaluate_policy
from .plugins.flappy.env_config import apply_flappy_preset, build_flappy_config_from_args, flappy_model_metadata, write_model_metadata
from .evals.flappy_eval_utils import evaluate_flappy_policy
from .games.breakout import BreakoutConfig, BreakoutEnv
from .games.flappy import FlappyEnv
from .games.fruit_cutter import FruitCutterConfig, FruitCutterEnv
from .games.match3 import Match3Config, Match3Env
from .games.pacman_lite import PacmanLiteConfig, PacmanLiteEnv
from .games.pong import PongConfig, PongEnv
from .games.shooter import ShooterConfig, ShooterEnv
from .games.snake import SnakeConfig, SnakeEnv, SnakeFeatureEnv
from .games.tetris import TetrisConfig, TetrisEnv, TetrisPlacementEnv
from .evals.match3_eval_utils import evaluate_match3_policy
from .evals.pacman_eval_utils import evaluate_pacman_policy
from .model_env_metadata import dqn_model_metadata_from_params, write_model_metadata as write_generic_model_metadata
from .network import AdamOptimizer, MLPQNetwork
from .evals.pong_eval_utils import evaluate_pong_policy
from .replay_buffer import PrioritizedReplayBuffer, ReplayBuffer
from .evals.shooter_eval_utils import evaluate_shooter_policy
from .evals.snake_eval_utils import evaluate_snake_policy
from .evals.tetris_eval_utils import evaluate_tetris_policy
from .evals.fruit_eval_utils import evaluate_fruit_policy
from .plugins.tetris.expert import choose_expert_action
from .evals.breakout_eval_utils import evaluate_breakout_policy


@dataclass
class TrainHooks:
    env_name: str
    save_best_name: str
    save_final_name: str
    build_episode_env: Callable[[int], Any]
    build_eval_env: Callable[[], Any]
    eval_once: Callable[[Any, MLPQNetwork, int, int], Any]
    eval_summary: Callable[[Any], str]
    eval_metric: Callable[[Any], float]
    use_legal_masks: bool = False
    reward_shaper: Callable[[np.ndarray, np.ndarray, float, dict[str, Any]], float] | None = None
    on_episode_start: Callable[[Any], dict[str, Any]] | None = None
    on_step: Callable[[dict[str, Any], dict[str, Any]], None] | None = None
    format_episode: Callable[[int, int, dict[str, Any], float, float, float, float], str] | None = None
    write_metadata: Callable[[Path], None] | None = None
    action_override: Callable[[Any, np.random.Generator, int, float, np.ndarray, MLPQNetwork, dict[str, Any]], int] | None = None
    target_clip: float | None = None
    loss_name: str = "mse"
    huber_delta: float = 1.0
    optimizer_max_grad_norm: float | None = None


@dataclass
class _StepTransition:
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool
    next_legal_actions: list[int] | None


class NStepAccumulator:
    def __init__(self, n_step: int) -> None:
        self.n_step = max(int(n_step), 1)
        self.buf: deque[_StepTransition] = deque()

    def _emit_one(self, gamma: float, force_done: bool = False) -> _StepTransition:
        k = min(self.n_step, len(self.buf))
        first = self.buf[0]
        reward_sum = 0.0
        next_state = first.next_state
        done = first.done
        next_legal = first.next_legal_actions
        for i in range(k):
            t = self.buf[i]
            reward_sum += (gamma**i) * float(t.reward)
            next_state = t.next_state
            done = bool(t.done)
            next_legal = t.next_legal_actions
            if done:
                break
        out = _StepTransition(
            state=first.state,
            action=int(first.action),
            reward=float(reward_sum),
            next_state=next_state,
            done=bool(done or force_done),
            next_legal_actions=next_legal,
        )
        self.buf.popleft()
        return out

    def append_and_pop_ready(
        self,
        transition: _StepTransition,
        *,
        gamma: float,
        flush: bool = False,
        force_terminal_flush: bool = False,
    ) -> list[_StepTransition]:
        self.buf.append(transition)
        out: list[_StepTransition] = []
        while len(self.buf) >= self.n_step:
            out.append(self._emit_one(gamma=gamma, force_done=False))
        if flush:
            while self.buf:
                out.append(self._emit_one(gamma=gamma, force_done=force_terminal_flush))
        return out

    def flush_remaining(self, *, gamma: float, force_terminal: bool) -> list[_StepTransition]:
        out: list[_StepTransition] = []
        while self.buf:
            out.append(self._emit_one(gamma=gamma, force_done=force_terminal))
        return out


def parse_hidden_sizes(raw: str) -> tuple[int, ...]:
    values = [v.strip() for v in str(raw).split(",") if v.strip()]
    if not values:
        raise ValueError("hidden sizes cannot be empty")
    return tuple(int(v) for v in values)


def parse_int_list(raw: str | None) -> list[int]:
    if raw is None:
        return []
    values = [v.strip() for v in str(raw).split(",") if v.strip()]
    return [int(v) for v in values]


def epsilon_by_step(step: int, start: float, end: float, decay_steps: int) -> float:
    if step >= decay_steps:
        return float(end)
    frac = step / max(decay_steps, 1)
    return float(start + frac * (end - start))


def argmax(values: list[float]) -> int:
    return int(max(range(len(values)), key=values.__getitem__))


def masked_argmax(values: list[float], legal_actions: list[int]) -> int:
    if not legal_actions:
        return argmax(values)
    return int(max(legal_actions, key=lambda action: values[action]))


def masked_max_q(target_q: np.ndarray, legal_masks: np.ndarray) -> np.ndarray:
    legal = legal_masks > 0.5
    masked_target = np.where(legal, target_q, -1e9)
    has_legal = np.any(legal, axis=1)
    max_next_q = np.max(masked_target, axis=1)
    return np.where(has_legal, max_next_q, 0.0).astype(np.float32)


def masked_double_q(online_q: np.ndarray, target_q: np.ndarray, legal_masks: np.ndarray) -> np.ndarray:
    legal = legal_masks > 0.5
    masked_online = np.where(legal, online_q, -1e9)
    next_actions = np.argmax(masked_online, axis=1)
    has_legal = np.any(legal, axis=1)
    selected = target_q[np.arange(target_q.shape[0]), next_actions]
    return np.where(has_legal, selected, 0.0).astype(np.float32)


def _cfg(params: dict[str, Any], key: str, default: Any) -> Any:
    return params.get(key, default)


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
        max_steps=int(_cfg(params, "max_steps", 1000)),
    )


def flappy_gap_alignment_shaping(state: np.ndarray, next_state: np.ndarray, scale: float, info: dict[str, Any]) -> float:
    if scale == 0.0 or int(info.get("passed_pipes", 0)) > 0:
        return 0.0
    curr_dx = float(state[2])
    next_dx = float(next_state[2])
    curr_abs_dy = abs(float(state[4]))
    next_abs_dy = abs(float(next_state[4]))
    if curr_dx < -0.1 or next_dx < -0.2:
        return 0.0
    return float(scale * (curr_abs_dy - next_abs_dy))


def curriculum_grid_size(episode: int, total_episodes: int, stages: list[int]) -> int:
    if len(stages) == 1:
        return stages[0]
    idx = min(len(stages) - 1, ((episode - 1) * len(stages)) // max(total_episodes, 1))
    return int(stages[idx])


def _make_snake_env(grid_size: int, params: dict[str, Any], seed: int):
    state_mode = str(_cfg(params, "state_mode", "board"))
    distance_reward_scale = float(_cfg(params, "distance_reward_scale", 0.2))
    config = SnakeConfig(grid_size=grid_size, distance_reward_scale=distance_reward_scale)
    if state_mode == "features":
        return SnakeFeatureEnv(config=config, seed=seed)
    state_grid_size = int(_cfg(params, "state_grid_size", 0))
    if state_grid_size <= 0:
        state_grid_size = grid_size
    return SnakeEnv(config=config, seed=seed, state_grid_size=state_grid_size)


def _tetris_action_override(env, rng, global_step, epsilon, state, online, ctx):
    legal_actions = env.legal_actions()
    if not legal_actions:
        return 0
    expert_start = ctx["expert_start"]
    expert_end = ctx["expert_end"]
    expert_decay_steps = ctx["expert_decay_steps"]
    expert_warmup_steps = ctx["expert_warmup_steps"]
    placement_actions = ctx["placement_actions"]
    expert_prob = epsilon_by_step(global_step, expert_start, expert_end, expert_decay_steps)
    if global_step < expert_warmup_steps:
        return choose_expert_action(env, placement_actions=placement_actions)
    if rng.random() < epsilon:
        if rng.random() < expert_prob:
            return choose_expert_action(env, placement_actions=placement_actions)
        return int(rng.choice(np.asarray(legal_actions, dtype=np.int64)))
    q_values = online.predict_one(state)
    return masked_argmax(q_values, legal_actions)


def build_dqn_hooks(env_name: str, params: dict[str, Any], seed: int, episodes: int) -> tuple[Any, TrainHooks, dict[str, Any]]:
    extra: dict[str, Any] = {}

    if env_name == "2048":
        env = Game2048Env(seed=seed, config=EnvConfig())
        def ep_env(ep: int): return Game2048Env(seed=seed + ep, config=EnvConfig())
        def ev_env(): return Game2048Env(seed=seed + 999)
        def ev(eval_env, net, n, seed_start): return evaluate_policy(eval_env, net, episodes=n, seed_start=seed_start)
        def ev_sum(stats):
            return (
                f"avg_score={stats.avg_score:.1f} median={stats.median_score:.1f} "
                f"reach_512={stats.reach_512:.2%} reach_1024={stats.reach_1024:.2%} reach_2048={stats.reach_2048:.2%}"
            )
        def ev_metric(stats): return float(stats.avg_score)
        def start(env): return {"invalid_moves": 0}
        def step(track, info):
            if not bool(info.get("moved", True)):
                track["invalid_moves"] += 1
        def fmt(ep, steps, info, reward, eps, loss, td):
            invalid_ratio = track_invalid = 0.0
            return (
                f"ep={ep:4d} steps={steps:4d} score={int(info.get('score',0)):6d} max_tile={int(info.get('max_tile',0)):4d} "
                f"reward={reward:8.1f} eps={eps:.3f} loss={loss:.5f} td_abs={td:.5f}"
            )
        hooks = TrainHooks(env_name, "dqn_2048_best.json", "dqn_2048_final.json", ep_env, ev_env, ev, ev_sum, ev_metric, on_episode_start=start, on_step=step)
        return env, hooks, extra

    if env_name == "snake":
        curriculum = parse_int_list(str(_cfg(params, "curriculum_grid_sizes", "")))
        if curriculum:
            if any(s < 4 for s in curriculum):
                raise ValueError("All curriculum grid sizes must be >= 4")
            grid_stages = curriculum
        else:
            grid_stages = [int(_cfg(params, "grid_size", 10))]
        max_grid = max(grid_stages)
        if str(_cfg(params, "state_mode", "board")) == "board":
            state_grid_size = int(_cfg(params, "state_grid_size", 0)) or max_grid
            if state_grid_size < max_grid:
                raise ValueError("state-grid-size must be >= max curriculum grid size")
            extra["state_grid_size"] = state_grid_size
        extra["grid_stages"] = grid_stages
        env = _make_snake_env(grid_stages[0], params, seed)
        def ep_env(ep: int):
            g = curriculum_grid_size(ep, episodes, grid_stages)
            extra["last_episode_grid"] = g
            return _make_snake_env(g, params, seed + ep)
        def ev_env():
            return _make_snake_env(grid_stages[-1], params, seed + 999)
        def ev(eval_env, net, n, seed_start): return evaluate_snake_policy(eval_env, net, episodes=n, seed_start=seed_start)
        def ev_sum(stats):
            return f"avg_score={stats.avg_score:.2f} median={stats.median_score:.2f} avg_len={stats.avg_length:.2f} avg_steps={stats.avg_steps:.2f} food_rate={stats.food_rate:.4f}"
        hooks = TrainHooks(
            env_name, "snake_dqn_best.json", "snake_dqn_final.json", ep_env, ev_env, ev, ev_sum, lambda s: float(s.avg_score),
            use_legal_masks=True,
            optimizer_max_grad_norm=float(_cfg(params, "max_grad_norm", 5.0)),
            loss_name=str(_cfg(params, "loss", "huber")),
            huber_delta=float(_cfg(params, "huber_delta", 1.0)),
        )
        return env, hooks, extra

    if env_name == "fruit":
        grid = int(_cfg(params, "grid_size", 10))
        state_grid = int(_cfg(params, "state_grid_size", 0)) or grid
        cfg = FruitCutterConfig(grid_size=grid, spawn_prob=float(_cfg(params, "spawn_prob", 0.45)), bomb_prob=float(_cfg(params, "bomb_prob", 0.18)))
        env = FruitCutterEnv(config=cfg, seed=seed, state_grid_size=state_grid)
        def ep_env(ep: int): return FruitCutterEnv(config=cfg, seed=seed + ep, state_grid_size=state_grid)
        def ev_env(): return FruitCutterEnv(config=cfg, seed=seed + 999, state_grid_size=state_grid)
        def ev(eval_env, net, n, seed_start): return evaluate_fruit_policy(eval_env, net, episodes=n, seed_start=seed_start)
        def ev_sum(stats): return f"avg_score={stats.avg_score:.2f} median={stats.median_score:.2f} avg_steps={stats.avg_steps:.2f} avg_sliced={stats.avg_sliced_per_episode:.2f} avg_missed={stats.avg_missed_per_episode:.2f}"
        return env, TrainHooks(env_name, "fruit_dqn_best.json", "fruit_dqn_final.json", ep_env, ev_env, ev, ev_sum, lambda s: float(s.avg_score)), extra

    if env_name == "shooter":
        grid = int(_cfg(params, "grid_size", 10))
        state_grid = int(_cfg(params, "state_grid_size", 0)) or grid
        cfg = ShooterConfig(grid_size=grid, spawn_prob=float(_cfg(params, "spawn_prob", 0.35)), max_steps=int(_cfg(params, "max_steps", 800)))
        env = ShooterEnv(config=cfg, seed=seed, state_grid_size=state_grid)
        def ep_env(ep: int): return ShooterEnv(config=cfg, seed=seed + ep, state_grid_size=state_grid)
        def ev_env(): return ShooterEnv(config=cfg, seed=seed + 999, state_grid_size=state_grid)
        def ev(eval_env, net, n, seed_start): return evaluate_shooter_policy(eval_env, net, episodes=n, seed_start=seed_start)
        def ev_sum(stats): return f"avg_score={stats.avg_score:.2f} median={stats.median_score:.2f} avg_steps={stats.avg_steps:.2f} avg_lives={stats.avg_lives_left:.2f} avg_kills={stats.avg_kills_per_episode:.2f}"
        return env, TrainHooks(env_name, "shooter_dqn_best.json", "shooter_dqn_final.json", ep_env, ev_env, ev, ev_sum, lambda s: float(s.avg_score)), extra

    if env_name == "breakout":
        cfg = BreakoutConfig(
            width=int(_cfg(params, "width", 12)), height=int(_cfg(params, "height", 14)),
            brick_rows=int(_cfg(params, "brick_rows", 4)), brick_top=int(_cfg(params, "brick_top", 1)),
            paddle_width=int(_cfg(params, "paddle_width", 3)), start_lives=int(_cfg(params, "start_lives", 3)),
            step_reward=float(_cfg(params, "step_reward", 0.02)), brick_reward=float(_cfg(params, "brick_reward", 1.0)),
            paddle_hit_reward=float(_cfg(params, "paddle_hit_reward", 0.03)), life_loss_penalty=float(_cfg(params, "life_loss_penalty", -3.0)),
            clear_bonus=float(_cfg(params, "clear_bonus", 5.0)), max_steps=int(_cfg(params, "max_steps", 1200)),
        )
        env = BreakoutEnv(config=cfg, seed=seed)
        def ep_env(ep: int): return BreakoutEnv(config=cfg, seed=seed + ep)
        def ev_env(): return BreakoutEnv(config=cfg, seed=seed + 999)
        def ev(eval_env, net, n, seed_start): return evaluate_breakout_policy(eval_env, net, episodes=n, seed_start=seed_start)
        def ev_sum(stats): return f"avg_score={stats.avg_score:.2f} median={stats.median_score:.2f} avg_steps={stats.avg_steps:.2f} avg_bricks={stats.avg_bricks_broken:.2f} clear_rate={100.0*stats.clear_rate:.1f}%"
        return env, TrainHooks(env_name, "breakout_dqn_best.json", "breakout_dqn_final.json", ep_env, ev_env, ev, ev_sum, lambda s: float(s.avg_score)), extra

    if env_name == "pong":
        cfg = PongConfig(
            width=int(_cfg(params, "width", 12)), height=int(_cfg(params, "height", 16)), paddle_height=int(_cfg(params, "paddle_height", 4)),
            paddle_speed=int(_cfg(params, "paddle_speed", 1)), opponent_track_prob=float(_cfg(params, "opponent_track_prob", 0.85)),
            start_lives=int(_cfg(params, "start_lives", 3)), step_reward=float(_cfg(params, "step_reward", 0.01)),
            paddle_hit_reward=float(_cfg(params, "paddle_hit_reward", 0.05)), score_reward=float(_cfg(params, "score_reward", 2.0)),
            concede_penalty=float(_cfg(params, "concede_penalty", -2.0)), max_steps=int(_cfg(params, "max_steps", 1000)),
        )
        env = PongEnv(config=cfg, seed=seed)
        def ep_env(ep: int): return PongEnv(config=cfg, seed=seed + ep)
        def ev_env(): return PongEnv(config=cfg, seed=seed + 999)
        def ev(eval_env, net, n, seed_start): return evaluate_pong_policy(eval_env, net, episodes=n, seed_start=seed_start)
        def ev_sum(stats): return f"avg_score={stats.avg_score:.2f} median={stats.median_score:.2f} avg_steps={stats.avg_steps:.2f} avg_lives={stats.avg_lives_left:.2f} avg_hits={stats.avg_player_hits:.2f}"
        return env, TrainHooks(env_name, "pong_dqn_best.json", "pong_dqn_final.json", ep_env, ev_env, ev, ev_sum, lambda s: float(s.avg_score)), extra

    if env_name == "flappy":
        ns = _flappy_ns(params)
        preset_name = apply_flappy_preset(ns)
        config = build_flappy_config_from_args(ns, include_rewards=True)
        align_scale = float(_cfg(params, "align_shaping_scale", 2.0))
        env = FlappyEnv(config=config, seed=seed)
        def ep_env(ep: int): return FlappyEnv(config=config, seed=seed + ep)
        def ev_env(): return FlappyEnv(config=config, seed=seed + 999)
        def ev(eval_env, net, n, seed_start): return evaluate_flappy_policy(eval_env, net, episodes=n, seed_start=seed_start, max_steps=int(_cfg(params, "max_steps", 1000)))
        def ev_sum(stats): return f"avg_score={stats.avg_score:.2f} median={stats.median_score:.2f} avg_steps={stats.avg_steps:.2f} avg_reward={stats.avg_reward:.2f}"
        def shaper(s, ns2, r, info): return r + flappy_gap_alignment_shaping(s, ns2, align_scale, info)
        def write_meta(path: Path): write_model_metadata(path, flappy_model_metadata(config, preset_name, algo="dqn"))
        return env, TrainHooks(env_name, "flappy_dqn_best.json", "flappy_dqn_final.json", ep_env, ev_env, ev, ev_sum, lambda s: float(s.avg_score), reward_shaper=shaper, write_metadata=write_meta, optimizer_max_grad_norm=float(_cfg(params, "max_grad_norm", 5.0)), loss_name=str(_cfg(params, "loss", "huber")), huber_delta=float(_cfg(params, "huber_delta", 1.0))), extra

    if env_name == "match3":
        cfg = Match3Config(
            width=int(_cfg(params, "width", 6)), height=int(_cfg(params, "height", 6)), num_colors=int(_cfg(params, "num_colors", 5)),
            max_steps=int(_cfg(params, "max_steps", 150)), step_reward=float(_cfg(params, "step_reward", -0.01)),
            tile_reward=float(_cfg(params, "tile_reward", 0.20)), combo_bonus=float(_cfg(params, "combo_bonus", 0.15)),
            invalid_penalty=float(_cfg(params, "invalid_penalty", -0.50)), reshuffle_on_stuck=bool(_cfg(params, "reshuffle_on_stuck", True)),
        )
        env = Match3Env(config=cfg, seed=seed)
        def ep_env(ep: int): return Match3Env(config=cfg, seed=seed + ep)
        def ev_env(): return Match3Env(config=cfg, seed=seed + 999)
        def ev(eval_env, net, n, seed_start): return evaluate_match3_policy(eval_env, net, episodes=n, seed_start=seed_start, max_steps=cfg.max_steps)
        def ev_sum(stats): return f"avg_score={stats.avg_score:.2f} median={stats.median_score:.2f} avg_tiles={stats.avg_tiles_cleared:.2f} avg_cascades={stats.avg_cascades:.2f} invalid_rate={stats.invalid_rate:.3f}"
        return env, TrainHooks(env_name, "match3_dqn_best.json", "match3_dqn_final.json", ep_env, ev_env, ev, ev_sum, lambda s: float(s.avg_score), use_legal_masks=True), extra

    if env_name == "pacman":
        grid = int(_cfg(params, "grid_size", 11))
        state_grid = int(_cfg(params, "state_grid_size", 0)) or grid
        cfg = PacmanLiteConfig(
            grid_size=grid,
            num_ghosts=int(_cfg(params, "num_ghosts", 2)),
            ghost_chase_prob=float(_cfg(params, "ghost_chase_prob", 0.75)),
            pellet_reward=float(_cfg(params, "pellet_reward", 1.0)),
            step_reward=float(_cfg(params, "step_reward", -0.02)),
            ghost_collision_penalty=float(_cfg(params, "ghost_collision_penalty", -5.0)),
            clear_bonus=float(_cfg(params, "clear_bonus", 8.0)),
            max_steps=int(_cfg(params, "max_steps", 500)),
            start_lives=int(_cfg(params, "start_lives", 3)),
        )
        env = PacmanLiteEnv(config=cfg, seed=seed, state_grid_size=state_grid)
        def ep_env(ep: int): return PacmanLiteEnv(config=cfg, seed=seed + ep, state_grid_size=state_grid)
        def ev_env(): return PacmanLiteEnv(config=cfg, seed=seed + 999, state_grid_size=state_grid)
        def ev(eval_env, net, n, seed_start): return evaluate_pacman_policy(eval_env, net, episodes=n, seed_start=seed_start, max_steps=cfg.max_steps)
        def ev_sum(stats): return f"avg_score={stats.avg_score:.2f} median={stats.median_score:.2f} avg_steps={stats.avg_steps:.2f} avg_lives={stats.avg_lives_left:.2f} clears={100.0*stats.clear_rate:.1f}%"
        return env, TrainHooks(env_name, "pacman_dqn_best.json", "pacman_dqn_final.json", ep_env, ev_env, ev, ev_sum, lambda s: float(s.avg_score), use_legal_masks=True), extra

    if env_name == "tetris":
        placement_actions = bool(_cfg(params, "placement_actions", False))
        cfg = TetrisConfig(height=int(_cfg(params, "height", 20)), width=int(_cfg(params, "width", 10)), max_steps=int(_cfg(params, "max_steps", 1200)))
        env = TetrisPlacementEnv(config=cfg, seed=seed) if placement_actions else TetrisEnv(config=cfg, seed=seed)
        def ep_env(ep: int):
            return TetrisPlacementEnv(config=cfg, seed=seed + ep) if placement_actions else TetrisEnv(config=cfg, seed=seed + ep)
        def ev_env():
            return TetrisPlacementEnv(config=cfg, seed=seed + 999) if placement_actions else TetrisEnv(config=cfg, seed=seed + 999)
        def ev(eval_env, net, n, seed_start): return evaluate_tetris_policy(eval_env, net, episodes=n, seed_start=seed_start, max_steps=cfg.max_steps)
        def ev_sum(stats): return f"avg_score={stats.avg_score:.2f} median={stats.median_score:.2f} avg_lines={stats.avg_lines:.2f} avg_steps={stats.avg_steps:.2f}"
        extra.update({
            "expert_start": float(_cfg(params, "expert_start", 0.30)),
            "expert_end": float(_cfg(params, "expert_end", 0.05)),
            "expert_decay_steps": int(_cfg(params, "expert_decay_steps", 40_000)),
            "expert_warmup_steps": int(_cfg(params, "expert_warmup_steps", 12_000)),
            "placement_actions": placement_actions,
        })
        grad_clip = float(_cfg(params, "grad_clip", 5.0))
        target_clip = float(_cfg(params, "target_clip", 30.0))
        return env, TrainHooks(env_name, "tetris_dqn_best.json", "tetris_dqn_final.json", ep_env, ev_env, ev, ev_sum, lambda s: float(s.avg_score), use_legal_masks=True, action_override=_tetris_action_override, target_clip=target_clip if target_clip > 0 else None, optimizer_max_grad_norm=grad_clip), extra

    raise ValueError(f"Unsupported DQN env: {env_name}")


def train_dqn(env_name: str, params: dict[str, Any]) -> None:
    episodes = int(_cfg(params, "episodes", 1000))
    max_steps = int(_cfg(params, "max_steps", 1000))
    seed = int(_cfg(params, "seed", 42))
    buffer_size = int(_cfg(params, "buffer_size", 100_000))
    batch_size = int(_cfg(params, "batch_size", 128))
    gamma = float(_cfg(params, "gamma", 0.99))
    lr = float(_cfg(params, "lr", 1e-3))
    target_update = int(_cfg(params, "target_update", 1000))
    double_dqn = bool(_cfg(params, "double_dqn", False))
    dueling = bool(_cfg(params, "dueling", False))
    train_every = int(_cfg(params, "train_every", 4))
    warmup_steps = int(_cfg(params, "warmup_steps", 2000))
    n_step = max(int(_cfg(params, "n_step", 1)), 1)
    eps_start = float(_cfg(params, "eps_start", 1.0))
    eps_end = float(_cfg(params, "eps_end", 0.05))
    eps_decay_steps = int(_cfg(params, "eps_decay_steps", 120_000))
    hidden_sizes = parse_hidden_sizes(_cfg(params, "hidden_sizes", "256,256"))
    eval_every = int(_cfg(params, "eval_every", 50))
    eval_episodes = int(_cfg(params, "eval_episodes", 30))
    save_dir = Path(str(_cfg(params, "save_dir", f"models/{env_name}")))
    save_dir.mkdir(parents=True, exist_ok=True)
    per = bool(_cfg(params, "per", False))
    per_alpha = float(_cfg(params, "per_alpha", 0.6))
    per_beta_start = float(_cfg(params, "per_beta_start", 0.4))
    per_beta_end = float(_cfg(params, "per_beta_end", 1.0))
    per_beta_decay_steps = int(_cfg(params, "per_beta_decay_steps", max(eps_decay_steps, 1)))
    per_priority_eps = float(_cfg(params, "per_priority_eps", 1e-6))

    rng = np.random.default_rng(seed)
    env0, hooks, extra = build_dqn_hooks(env_name, params, seed, episodes)
    if hooks.write_metadata is None:
        generic_metadata = dqn_model_metadata_from_params(env_name, params)
        if generic_metadata is not None:
            def _write_generic_metadata(path: Path) -> None:
                write_generic_model_metadata(path, generic_metadata)
            hooks.write_metadata = _write_generic_metadata
    state_dim = int(env0.reset(seed=seed).shape[0])

    online = MLPQNetwork(input_dim=state_dim, output_dim=env0.action_size, hidden_sizes=hidden_sizes, seed=seed, dueling=dueling)
    target = MLPQNetwork(input_dim=state_dim, output_dim=env0.action_size, hidden_sizes=hidden_sizes, seed=seed + 1, dueling=dueling)
    target.copy_from(online)

    optimizer = AdamOptimizer(lr=lr, max_grad_norm=hooks.optimizer_max_grad_norm)
    if per:
        buffer: ReplayBuffer | PrioritizedReplayBuffer = PrioritizedReplayBuffer(
            capacity=buffer_size,
            state_dim=state_dim,
            action_dim=(env0.action_size if hooks.use_legal_masks else None),
            alpha=per_alpha,
        )
    else:
        buffer = ReplayBuffer(capacity=buffer_size, state_dim=state_dim, action_dim=(env0.action_size if hooks.use_legal_masks else None))

    global_step = 0
    best_eval = -float("inf")
    print(
        f"Training {env_name} | double_dqn={double_dqn} dueling={dueling} hidden={hidden_sizes} "
        f"n_step={n_step} per={per}"
    )

    for episode in range(1, episodes + 1):
        env = hooks.build_episode_env(episode)
        state = env.reset(seed=seed + episode)
        done = False
        steps = 0
        episode_reward = 0.0
        losses: list[float] = []
        td_abs_vals: list[float] = []
        trackers = hooks.on_episode_start(env) if hooks.on_episode_start else {}
        info: dict[str, Any] = {}
        nstep_acc = NStepAccumulator(n_step=n_step)

        while not done and steps < max_steps:
            legal_actions = env.legal_actions() if hasattr(env, "legal_actions") else []
            epsilon = epsilon_by_step(global_step, eps_start, eps_end, eps_decay_steps)

            if hooks.action_override is not None:
                action = hooks.action_override(env, rng, global_step, epsilon, state, online, extra)
            elif rng.random() < epsilon:
                action = int(rng.choice(np.asarray(legal_actions, dtype=np.int64))) if legal_actions else int(rng.integers(0, env.action_size))
            else:
                q_values = online.predict_one(state)
                action = masked_argmax(q_values, legal_actions)

            next_state, reward, done, info = env.step(action)
            reward_val = float(reward)
            if hooks.reward_shaper is not None:
                reward_val = float(hooks.reward_shaper(np.asarray(state, dtype=np.float32), np.asarray(next_state, dtype=np.float32), reward_val, info))

            next_legal = env.legal_actions() if hooks.use_legal_masks else None
            ready = nstep_acc.append_and_pop_ready(
                _StepTransition(
                    state=np.asarray(state, dtype=np.float32).copy(),
                    action=int(action),
                    reward=float(reward_val),
                    next_state=np.asarray(next_state, dtype=np.float32).copy(),
                    done=bool(done),
                    next_legal_actions=(list(next_legal) if next_legal is not None else None),
                ),
                gamma=gamma,
                flush=bool(done),
                force_terminal_flush=False,
            )
            for t in ready:
                buffer.add(t.state, t.action, t.reward, t.next_state, t.done, next_legal_actions=t.next_legal_actions)

            if hooks.on_step is not None:
                hooks.on_step(trackers, info)
            episode_reward += reward_val
            steps += 1
            global_step += 1
            state = next_state

            if len(buffer) >= batch_size and global_step >= warmup_steps and global_step % train_every == 0:
                if per and isinstance(buffer, PrioritizedReplayBuffer):
                    beta = epsilon_by_step(global_step, per_beta_start, per_beta_end, per_beta_decay_steps)
                    batch = buffer.sample(batch_size, rng, beta=beta)
                else:
                    batch = buffer.sample(batch_size, rng)
                target_q = np.asarray(target.predict_batch(batch.next_states), dtype=np.float32)
                if hooks.use_legal_masks:
                    legal_masks = batch.next_legal_masks
                    if legal_masks is None:
                        raise RuntimeError(f"{env_name} requires next_legal_masks")
                    if double_dqn:
                        online_q = np.asarray(online.predict_batch(batch.next_states), dtype=np.float32)
                        max_next_q = masked_double_q(online_q, target_q, legal_masks)
                    else:
                        max_next_q = masked_max_q(target_q, legal_masks)
                elif double_dqn:
                    online_q = online.predict_batch(batch.next_states)
                    next_actions = [argmax(v) for v in online_q]
                    max_next_q = np.asarray([float(target_q[i][next_actions[i]]) for i in range(len(next_actions))], dtype=np.float32)
                else:
                    max_next_q = np.max(target_q, axis=1).astype(np.float32)

                if hooks.target_clip is not None and hooks.target_clip > 0:
                    max_next_q = np.clip(max_next_q, -hooks.target_clip, hooks.target_clip)

                gamma_bootstrap = float(gamma**n_step)
                td_target = batch.rewards + gamma_bootstrap * (1.0 - batch.dones) * max_next_q
                if hooks.target_clip is not None and hooks.target_clip > 0:
                    td_target = np.clip(td_target, -hooks.target_clip, hooks.target_clip)
                if per and isinstance(buffer, PrioritizedReplayBuffer):
                    loss, td_abs, td_vec = online.train_batch(
                        batch.states,
                        batch.actions,
                        td_target,
                        optimizer,
                        loss=hooks.loss_name,
                        huber_delta=hooks.huber_delta,
                        sample_weights=batch.weights,
                        return_td_errors=True,
                    )
                    if batch.indices is not None:
                        buffer.update_priorities(batch.indices, td_vec + per_priority_eps, eps=per_priority_eps)
                else:
                    loss, td_abs = online.train_batch(
                        batch.states,
                        batch.actions,
                        td_target,
                        optimizer,
                        loss=hooks.loss_name,
                        huber_delta=hooks.huber_delta,
                    )
                losses.append(float(loss))
                td_abs_vals.append(float(td_abs))

            if global_step % target_update == 0:
                target.copy_from(online)

        # Flush any remaining short n-step transitions on time-limit truncation.
        if not done and n_step > 1:
            for t in nstep_acc.flush_remaining(gamma=gamma, force_terminal=True):
                buffer.add(t.state, t.action, t.reward, t.next_state, t.done, next_legal_actions=t.next_legal_actions)

        mean_loss = float(np.mean(losses)) if losses else 0.0
        mean_td_abs = float(np.mean(td_abs_vals)) if td_abs_vals else 0.0
        eps_now = epsilon_by_step(global_step, eps_start, eps_end, eps_decay_steps)

        # Compact generic log with optional common fields.
        score = int(info.get("score", 0))
        extra_fields: list[str] = []
        for key in ("max_tile", "lines", "length", "lives", "passed_pipes"):
            if key in info:
                extra_fields.append(f"{key}={int(info[key]):4d}")
        print(
            f"ep={episode:4d} steps={steps:4d} score={score:4d} "
            + (" ".join(extra_fields) + " " if extra_fields else "")
            + f"reward={episode_reward:8.2f} eps={eps_now:.3f} loss={mean_loss:.5f} td_abs={mean_td_abs:.5f}"
        )

        if episode % eval_every == 0:
            eval_env = hooks.build_eval_env()
            stats = hooks.eval_once(eval_env, online, eval_episodes, seed + 7000)
            print(f"  eval: {hooks.eval_summary(stats)}")
            metric = hooks.eval_metric(stats)
            if metric > best_eval:
                best_eval = metric
                best_path = save_dir / hooks.save_best_name
                online.save(best_path)
                if hooks.write_metadata is not None:
                    hooks.write_metadata(best_path)
                print(f"  saved new best checkpoint: {best_path}")

    final_path = save_dir / hooks.save_final_name
    online.save(final_path)
    if hooks.write_metadata is not None:
        hooks.write_metadata(final_path)
    print(f"Training complete. Final model saved to: {final_path}")
