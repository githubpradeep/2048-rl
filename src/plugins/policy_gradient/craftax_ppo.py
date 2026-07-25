"""Craftax-Classic PPO training / eval / play (pure NumPy, vectorized envs)."""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import numpy as np

from ...evals.craftax_eval_utils import evaluate_craftax_policy
from ...games.craftax_classic import Action, CraftaxClassicConfig, CraftaxClassicEnv, NUM_ACTIONS
from ...games.craftax_classic.renderer import CraftaxPygameViewer
from ...network import AdamOptimizer
from .actor_critic import apply_cli_overrides  # shared YAML override helpers
from .ppo_network import PPOPolicyNetwork, compute_gae


def _normalize(v: Any) -> Any:
    if isinstance(v, dict):
        return {str(k).replace("-", "_"): _normalize(x) for k, x in v.items()}
    if isinstance(v, list):
        return [_normalize(x) for x in v]
    return v


def _cfg(params: dict[str, Any], key: str, default: Any) -> Any:
    return params.get(key, default)


def _parse_hidden_sizes(raw: Any) -> tuple[int, ...]:
    vals = [v.strip() for v in str(raw).split(",") if v.strip()]
    if not vals:
        raise ValueError("hidden sizes cannot be empty")
    return tuple(int(v) for v in vals)


def _build_craftax_config(params: dict[str, Any], *, reward_shaping: bool | None = None) -> CraftaxClassicConfig:
    map_size = int(_cfg(params, "map_size", 64))
    shaping = bool(_cfg(params, "reward_shaping", False)) if reward_shaping is None else bool(reward_shaping)
    return CraftaxClassicConfig(
        map_size=(map_size, map_size),
        max_timesteps=int(_cfg(params, "max_steps", _cfg(params, "max_timesteps", 10000))),
        day_length=int(_cfg(params, "day_length", 300)),
        always_diamond=bool(_cfg(params, "always_diamond", True)),
        reward_shaping=shaping,
        survive_bonus=float(_cfg(params, "survive_bonus", 0.0002)),
        death_penalty=float(_cfg(params, "death_penalty", 1.0)),
        resource_scale=float(_cfg(params, "resource_scale", 0.1)),
        vital_gain_scale=float(_cfg(params, "vital_gain_scale", 0.05)),
        low_vital_penalty=float(_cfg(params, "low_vital_penalty", 0.01)),
        reverse_penalty=float(_cfg(params, "reverse_penalty", 0.02)),
        block_reverse_moves=bool(_cfg(params, "block_reverse_moves", True)),
    )


def legal_mask_for_env(env: CraftaxClassicEnv) -> np.ndarray:
    mask = np.zeros((NUM_ACTIONS,), dtype=np.float32)
    for a in env.legal_actions():
        if 0 <= int(a) < NUM_ACTIONS:
            mask[int(a)] = 1.0
    if float(mask.sum()) <= 0.0:
        mask[:] = 1.0
    return mask


class CraftaxVecEnv:
    """Synchronous vector of ``CraftaxClassicEnv`` with auto-reset on done."""

    def __init__(self, num_envs: int, config: CraftaxClassicConfig, seed: int) -> None:
        self.num_envs = int(num_envs)
        self.config = config
        self.envs = [
            CraftaxClassicEnv(config=config, seed=seed + i) for i in range(self.num_envs)
        ]
        self._episode_returns = np.zeros((self.num_envs,), dtype=np.float32)
        self._episode_scores = np.zeros((self.num_envs,), dtype=np.float32)
        self._completed_returns: list[float] = []
        self._completed_scores: list[float] = []

    def reset(self, seed: int) -> tuple[np.ndarray, np.ndarray]:
        states = []
        masks = []
        self._episode_returns[:] = 0.0
        self._episode_scores[:] = 0.0
        for i, env in enumerate(self.envs):
            states.append(np.asarray(env.reset(seed=seed + i), dtype=np.float32))
            masks.append(legal_mask_for_env(env))
        return np.stack(states, axis=0), np.stack(masks, axis=0)

    def step(
        self, actions: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]]]:
        next_states: list[np.ndarray] = []
        rewards = np.zeros((self.num_envs,), dtype=np.float32)
        dones = np.zeros((self.num_envs,), dtype=np.float32)
        masks: list[np.ndarray] = []
        infos: list[dict[str, Any]] = []
        acts = np.asarray(actions, dtype=np.int64).reshape(self.num_envs)
        for i, env in enumerate(self.envs):
            state, reward, done, info = env.step(int(acts[i]))
            rewards[i] = float(reward)
            dones[i] = 1.0 if done else 0.0
            self._episode_returns[i] += float(reward)
            self._episode_scores[i] = float(info.get("score", 0))
            if done:
                self._completed_returns.append(float(self._episode_returns[i]))
                self._completed_scores.append(float(self._episode_scores[i]))
                self._episode_returns[i] = 0.0
                self._episode_scores[i] = 0.0
                # Auto-reset so the next observation is a fresh episode start.
                state = env.reset()
            next_states.append(np.asarray(state, dtype=np.float32))
            masks.append(legal_mask_for_env(env))
            infos.append(info)
        return (
            np.stack(next_states, axis=0),
            rewards,
            dones,
            np.stack(masks, axis=0),
            infos,
        )

    def pop_completed_stats(self) -> tuple[list[float], list[float]]:
        returns = list(self._completed_returns)
        scores = list(self._completed_scores)
        self._completed_returns.clear()
        self._completed_scores.clear()
        return returns, scores


def action_label(action: int) -> str:
    try:
        return Action(action).name
    except ValueError:
        return str(action)


def pick_greedy_action(model: PPOPolicyNetwork, state: np.ndarray, legal: list[int]) -> int:
    scores = model.predict_one(state)
    if not legal:
        return int(max(range(len(scores)), key=scores.__getitem__))
    return int(max(legal, key=lambda a: scores[a]))


def run_train_from_config(full_cfg: dict[str, Any]) -> None:
    cfg = _normalize(full_cfg)
    common = dict(cfg.get("common") or {})
    train_cfg = dict(cfg.get("train") or {})
    eval_cfg = dict(cfg.get("eval") or {})

    seed = int(common.get("seed", 42))
    rng = np.random.default_rng(seed)

    num_envs = int(_cfg(train_cfg, "num_envs", 8))
    horizon = int(_cfg(train_cfg, "horizon", 128))
    num_updates = int(_cfg(train_cfg, "num_updates", 2000))
    epochs = int(_cfg(train_cfg, "epochs", 4))
    minibatch_size = int(_cfg(train_cfg, "minibatch_size", 256))
    gamma = float(_cfg(train_cfg, "gamma", 0.995))
    gae_lambda = float(_cfg(train_cfg, "gae_lambda", 0.95))
    clip_eps = float(_cfg(train_cfg, "clip_eps", 0.2))
    value_coef = float(_cfg(train_cfg, "value_coef", 0.5))
    entropy_coef = float(_cfg(train_cfg, "entropy_coef", 0.01))
    normalize_advantages = bool(_cfg(train_cfg, "normalize_advantages", True))
    eval_every = int(_cfg(train_cfg, "eval_every", 25))
    eval_episodes = int(_cfg(train_cfg, "eval_episodes", _cfg(eval_cfg, "episodes", 20)))
    max_steps = int(_cfg(train_cfg, "max_steps", 10000))
    save_dir = Path(str(_cfg(train_cfg, "save_dir", "models/craftax_ppo")))
    save_dir.mkdir(parents=True, exist_ok=True)
    best_path = save_dir / "craftax_ppo_best.json"
    final_path = save_dir / "craftax_ppo_final.json"

    train_env_cfg = _build_craftax_config({**common, **train_cfg}, reward_shaping=None)
    eval_env_cfg = _build_craftax_config(
        {**common, **train_cfg, **eval_cfg},
        reward_shaping=False,
    )

    vec = CraftaxVecEnv(num_envs, train_env_cfg, seed=seed)
    states, legal_masks = vec.reset(seed=seed)

    model = PPOPolicyNetwork(
        input_dim=int(states.shape[1]),
        output_dim=NUM_ACTIONS,
        hidden_sizes=_parse_hidden_sizes(_cfg(train_cfg, "hidden_sizes", "512,512")),
        seed=seed,
    )
    optimizer = AdamOptimizer(
        lr=float(_cfg(train_cfg, "lr", 3e-4)),
        max_grad_norm=float(_cfg(train_cfg, "max_grad_norm", 0.5)),
    )

    steps_per_update = num_envs * horizon
    print(
        f"Training craftax_ppo | envs={num_envs} horizon={horizon} updates={num_updates} "
        f"epochs={epochs} mb={minibatch_size} gamma={gamma:.3f} λ={gae_lambda:.2f} "
        f"clip={clip_eps:.2f} hidden={model.hidden_sizes} shaping={train_env_cfg.reward_shaping}"
    )

    best_metric = -float("inf")
    rolling_scores: list[float] = []

    for update in range(1, num_updates + 1):
        buf_states = np.zeros((horizon, num_envs, states.shape[1]), dtype=np.float32)
        buf_actions = np.zeros((horizon, num_envs), dtype=np.int64)
        buf_logp = np.zeros((horizon, num_envs), dtype=np.float32)
        buf_rewards = np.zeros((horizon, num_envs), dtype=np.float32)
        buf_dones = np.zeros((horizon, num_envs), dtype=np.float32)
        buf_values = np.zeros((horizon, num_envs), dtype=np.float32)
        buf_masks = np.zeros((horizon, num_envs, NUM_ACTIONS), dtype=np.float32)

        for t in range(horizon):
            buf_states[t] = states
            buf_masks[t] = legal_masks
            actions, logp, values = model.sample_actions(states, legal_masks, rng)
            next_states, rewards, dones, next_masks, _infos = vec.step(actions)
            buf_actions[t] = actions
            buf_logp[t] = logp
            buf_rewards[t] = rewards
            buf_dones[t] = dones
            buf_values[t] = values
            states = next_states
            legal_masks = next_masks

        # Bootstrap values for GAE (0 on terminal last step).
        _p, bootstrap_v, _c = model.forward_masked(states, legal_masks)
        next_values = bootstrap_v.reshape(num_envs).astype(np.float32)
        # If the final transition was done, vec already reset; bootstrap with V(new).
        # Standard: when dones[T-1], next_value contribution is zeroed in GAE via done flag.

        advantages, returns = compute_gae(
            buf_rewards,
            buf_values,
            buf_dones,
            next_values,
            gamma=gamma,
            gae_lambda=gae_lambda,
        )

        flat_states = buf_states.reshape(steps_per_update, -1)
        flat_actions = buf_actions.reshape(steps_per_update)
        flat_logp = buf_logp.reshape(steps_per_update)
        flat_adv = advantages.reshape(steps_per_update)
        flat_ret = returns.reshape(steps_per_update)
        flat_masks = buf_masks.reshape(steps_per_update, NUM_ACTIONS)

        indices = np.arange(steps_per_update)
        metrics_acc = {
            "loss": 0.0,
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "entropy": 0.0,
            "clip_frac": 0.0,
            "approx_kl": 0.0,
        }
        n_mb = 0
        for _epoch in range(epochs):
            rng.shuffle(indices)
            for start in range(0, steps_per_update, minibatch_size):
                mb_idx = indices[start : start + minibatch_size]
                if mb_idx.size == 0:
                    continue
                m = model.ppo_train_batch(
                    states=flat_states[mb_idx],
                    actions=flat_actions[mb_idx],
                    old_log_probs=flat_logp[mb_idx],
                    advantages=flat_adv[mb_idx],
                    returns=flat_ret[mb_idx],
                    legal_masks=flat_masks[mb_idx],
                    optimizer=optimizer,
                    clip_eps=clip_eps,
                    value_coef=value_coef,
                    entropy_coef=entropy_coef,
                    normalize_advantages=normalize_advantages,
                )
                for k in metrics_acc:
                    metrics_acc[k] += float(m[k])
                n_mb += 1
        if n_mb > 0:
            for k in metrics_acc:
                metrics_acc[k] /= n_mb

        completed_returns, completed_scores = vec.pop_completed_stats()
        if completed_scores:
            rolling_scores.extend(completed_scores)
            rolling_scores = rolling_scores[-100:]

        mean_roll = float(np.mean(rolling_scores)) if rolling_scores else 0.0
        mean_ret = float(np.mean(completed_returns)) if completed_returns else 0.0
        print(
            f"upd={update:4d} steps={update * steps_per_update:8d} "
            f"ep_score≈{mean_roll:5.2f} ep_ret≈{mean_ret:6.2f} "
            f"loss={metrics_acc['loss']:.4f} pi={metrics_acc['policy_loss']:.4f} "
            f"v={metrics_acc['value_loss']:.4f} ent={metrics_acc['entropy']:.3f} "
            f"clip={metrics_acc['clip_frac']:.3f} kl={metrics_acc['approx_kl']:.4f}"
        )

        if update % eval_every == 0 or update == num_updates:
            eval_env = CraftaxClassicEnv(config=eval_env_cfg, seed=seed + 999)
            stats = evaluate_craftax_policy(
                eval_env,
                model,  # type: ignore[arg-type]
                episodes=eval_episodes,
                seed_start=seed + 50_000,
                max_steps=max_steps,
            )
            summary = (
                f"avg_achievements={stats.avg_score:.2f} median={stats.median_score:.2f} "
                f"avg_steps={stats.avg_steps:.1f} avg_health={stats.avg_health:.2f} "
                f"avg_reward={stats.avg_reward:.2f}"
            )
            print(f"  eval: {summary}")
            metric = float(stats.avg_score)
            if metric > best_metric:
                best_metric = metric
                model.save(best_path)
                print(f"  saved new best checkpoint: {best_path}")

    model.save(final_path)
    print(f"saved final checkpoint: {final_path}")


def run_eval_from_config(full_cfg: dict[str, Any]) -> None:
    cfg = _normalize(full_cfg)
    common = dict(cfg.get("common") or {})
    eval_cfg = dict(cfg.get("eval") or {})
    seed = int(eval_cfg.get("seed", common.get("seed", 42)))
    episodes = int(eval_cfg.get("episodes", 50))
    max_steps = int(eval_cfg.get("max_steps", 10000))
    model_path = str(eval_cfg.get("model", ""))
    if not model_path:
        raise ValueError("craftax_ppo eval requires eval.model")
    model = PPOPolicyNetwork.load(model_path)
    env_cfg = _build_craftax_config({**common, **eval_cfg}, reward_shaping=False)
    env = CraftaxClassicEnv(config=env_cfg, seed=seed)
    stats = evaluate_craftax_policy(
        env, model, episodes=episodes, seed_start=seed, max_steps=max_steps  # type: ignore[arg-type]
    )
    print(
        f"craftax_ppo eval | avg_achievements={stats.avg_score:.2f} median={stats.median_score:.2f} "
        f"avg_steps={stats.avg_steps:.1f} avg_health={stats.avg_health:.2f} avg_reward={stats.avg_reward:.2f}"
    )


def run_play_from_config(full_cfg: dict[str, Any]) -> None:
    cfg = _normalize(full_cfg)
    common = dict(cfg.get("common") or {})
    play_cfg = dict(cfg.get("play") or {})
    seed = int(play_cfg.get("seed", common.get("seed", 2026)))
    delay = float(play_cfg.get("delay", 0.35))
    max_steps = int(play_cfg.get("max_steps", 10000))
    mode = str(play_cfg.get("mode", "pygame"))
    block_pixel_size = int(play_cfg.get("block_pixel_size", 48))
    close_on_end = bool(play_cfg.get("close_on_end", False))
    model_path = str(play_cfg.get("model", ""))
    if not model_path:
        raise ValueError("craftax_ppo play requires play.model")
    model = PPOPolicyNetwork.load(model_path)
    env_cfg = _build_craftax_config({**common, **play_cfg}, reward_shaping=False)
    env = CraftaxClassicEnv(config=env_cfg, seed=seed)

    if mode == "terminal":
        state = env.reset(seed=seed)
        done = False
        steps = 0
        info: dict[str, Any] = {"score": 0, "health": 9}
        print("Starting Craftax-Classic PPO autoplay (terminal)...\n")
        while not done and steps < max_steps:
            action = pick_greedy_action(model, state, env.legal_actions())
            state, reward, done, info = env.step(action)
            steps += 1
            print(
                f"Step {steps} | {action_label(action)} | reward={reward:.2f} | ach={info['score']}"
            )
            print(env.render())
            print("=" * 60)
            if delay > 0:
                time.sleep(delay)
        print(f"Done | achievements={info['score']}/22 health={info.get('health', 0)} steps={steps}")
        return

    try:
        import pygame
    except ImportError as exc:
        raise RuntimeError("pygame is required for pygame play mode") from exc

    pygame.init()
    try:
        viewer = CraftaxPygameViewer(block_pixel_size=block_pixel_size)
        clock = pygame.time.Clock()
        state = env.reset(seed=seed)
        done = False
        steps = 0
        info = {"score": 0, "health": 9, "achievements": 0}
        next_step_ts = time.monotonic()
        end_ts: float | None = None
        running = True
        paused = False
        step_delay = max(0.05, float(delay))
        last_action_name = "—"
        print("Craftax PPO play. Q/ESC quit | Space pause | [ slower | ] faster")

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
                        step_delay = min(2.0, step_delay + 0.05)
                    elif event.key == pygame.K_RIGHTBRACKET:
                        step_delay = max(0.05, step_delay - 0.05)

            now = time.monotonic()
            if not paused and not done and steps < max_steps and now >= next_step_ts:
                action = pick_greedy_action(model, state, env.legal_actions())
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

        print(
            f"Done | achievements={info['score']}/22 health={info.get('health', 0)} steps={steps}"
        )
    finally:
        pygame.quit()


def main() -> None:
    parser = argparse.ArgumentParser(description="Craftax-Classic PPO")
    parser.add_argument("--config", type=str, default="sample_configs/craftax_ppo.yaml")
    parser.add_argument("--mode", choices=["train", "eval", "play"], default="train")
    args, rest = parser.parse_known_args()

    try:
        import yaml  # type: ignore
    except ImportError as exc:
        raise RuntimeError("PyYAML is required") from exc

    raw = yaml.safe_load(Path(args.config).read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError("Config must be a mapping")
    section = {"train": "train", "eval": "eval", "play": "play"}[args.mode]
    from .actor_critic import _parse_overrides

    cfg = apply_cli_overrides(raw, _parse_overrides(rest), section=section) if rest else raw
    cfg["env"] = "craftax_ppo"
    if args.mode == "train":
        run_train_from_config(cfg)
    elif args.mode == "eval":
        run_eval_from_config(cfg)
    else:
        run_play_from_config(cfg)


if __name__ == "__main__":
    main()
