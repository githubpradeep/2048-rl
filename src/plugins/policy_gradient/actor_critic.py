from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from ...evals.pong_eval_utils import evaluate_pong_policy
from ...games.pong import PongConfig, PongEnv
from ...network import AdamOptimizer
from ..playback import play_pong_agent


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml  # type: ignore
    except ImportError as exc:
        raise RuntimeError("PyYAML is required. Install with `pip install -r requirements.txt`.") from exc
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config file must contain a top-level mapping: {path}")
    return data


def _normalize(v: Any) -> Any:
    if isinstance(v, dict):
        return {str(k).replace("-", "_"): _normalize(x) for k, x in v.items()}
    if isinstance(v, list):
        return [_normalize(x) for x in v]
    return v


def _parse_overrides(tokens: list[str]) -> dict[str, Any]:
    def _coerce(raw: str) -> Any:
        low = raw.lower()
        if low in {"true", "false"}:
            return (low == "true")
        if raw[:1] in {"[", "{"}:
            try:
                import yaml  # type: ignore

                return yaml.safe_load(raw)
            except Exception:
                return raw
        return raw

    out: dict[str, Any] = {}
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if not tok.startswith("--"):
            raise ValueError(f"Unexpected positional override token: {tok}")
        key = tok[2:].replace("-", "_")
        if i + 1 < len(tokens) and not tokens[i + 1].startswith("--"):
            out[key] = _coerce(tokens[i + 1])
            i += 2
        else:
            out[key] = True
            i += 1
    return out


def _deep_set(cfg: dict[str, Any], path: str, value: Any) -> None:
    parts = path.split(".")
    cur = cfg
    for part in parts[:-1]:
        nxt = cur.get(part)
        if not isinstance(nxt, dict):
            nxt = {}
            cur[part] = nxt
        cur = nxt
    cur[parts[-1]] = value


def apply_cli_overrides(full_cfg: dict[str, Any], overrides: dict[str, Any], *, section: str) -> dict[str, Any]:
    cfg = _normalize(full_cfg)
    for k, v in overrides.items():
        if "." in k:
            _deep_set(cfg, k, v)
        else:
            _deep_set(cfg, f"{section}.{k}", v)
    return cfg


def _cfg(params: dict[str, Any], key: str, default: Any) -> Any:
    return params.get(key, default)


def _build_pong_env(params: dict[str, Any], *, seed: int) -> PongEnv:
    max_steps = int(_cfg(params, "max_steps", 1000))
    return PongEnv(
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


def _softmax_last(x: np.ndarray) -> np.ndarray:
    shifted = x - np.max(x, axis=-1, keepdims=True)
    exps = np.exp(shifted, dtype=np.float32)
    denom = np.maximum(np.sum(exps, axis=-1, keepdims=True), 1e-12)
    return (exps / denom).astype(np.float32)


class ActorCriticPolicyNetwork:
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        *,
        hidden_sizes: tuple[int, ...] = (128, 128),
        seed: int | None = None,
    ) -> None:
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.hidden_sizes = tuple(int(v) for v in hidden_sizes)
        self.rng = np.random.default_rng(seed)

        self.trunk_w: list[np.ndarray] = []
        self.trunk_b: list[np.ndarray] = []
        layer_sizes = [self.input_dim, *self.hidden_sizes]
        for in_dim, out_dim in zip(layer_sizes[:-1], layer_sizes[1:]):
            scale = np.sqrt(2.0 / max(in_dim, 1))
            w = (self.rng.standard_normal((in_dim, out_dim), dtype=np.float32) * scale).astype(np.float32)
            b = np.zeros((1, out_dim), dtype=np.float32)
            self.trunk_w.append(w)
            self.trunk_b.append(b)

        feature_dim = self.hidden_sizes[-1] if self.hidden_sizes else self.input_dim
        scale = np.sqrt(2.0 / max(feature_dim, 1))
        self.policy_w = (self.rng.standard_normal((feature_dim, self.output_dim), dtype=np.float32) * scale).astype(np.float32)
        self.policy_b = np.zeros((1, self.output_dim), dtype=np.float32)
        self.value_w = (self.rng.standard_normal((feature_dim, 1), dtype=np.float32) * scale).astype(np.float32)
        self.value_b = np.zeros((1, 1), dtype=np.float32)

    @staticmethod
    def _relu(x: np.ndarray) -> np.ndarray:
        return np.maximum(0.0, x)

    @staticmethod
    def _relu_grad(x: np.ndarray) -> np.ndarray:
        return (x > 0).astype(np.float32)

    def forward(self, states: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        x = np.asarray(states, dtype=np.float32)
        if x.ndim == 1:
            x = x.reshape(1, -1)

        activations = [x]
        pre_activations: list[np.ndarray] = []
        current = x
        for w, b in zip(self.trunk_w, self.trunk_b):
            z = current @ w + b
            pre_activations.append(z)
            current = self._relu(z)
            activations.append(current)
        features = current if self.trunk_w else x
        logits = features @ self.policy_w + self.policy_b
        probs = _softmax_last(logits)
        values = features @ self.value_w + self.value_b
        cache = {
            "activations": activations,
            "pre_activations": pre_activations,
            "features": features,
            "logits": logits,
            "probs": probs,
            "values": values,
        }
        return probs, values, cache

    def predict_one(self, state: Any) -> list[float]:
        probs, _values, _cache = self.forward(np.asarray(state, dtype=np.float32))
        return [float(v) for v in probs[0].tolist()]

    def sample_action(self, state: Any, legal_actions: list[int], rng: np.random.Generator) -> int:
        probs, _values, _cache = self.forward(np.asarray(state, dtype=np.float32))
        p = probs[0].astype(np.float64)
        if legal_actions:
            mask = np.zeros_like(p, dtype=np.float64)
            mask[np.asarray(legal_actions, dtype=np.int64)] = 1.0
            p = p * mask
            total = float(np.sum(p))
            if total <= 0.0:
                p = mask / max(float(np.sum(mask)), 1.0)
            else:
                p = p / total
        return int(rng.choice(len(p), p=p))

    def train_batch(
        self,
        *,
        states: np.ndarray,
        actions: np.ndarray,
        returns: np.ndarray,
        optimizer: AdamOptimizer,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        normalize_advantages: bool = True,
    ) -> dict[str, float]:
        probs, values, cache = self.forward(states)
        B = probs.shape[0]
        actions_arr = np.asarray(actions, dtype=np.int64).reshape(B)
        returns_arr = np.asarray(returns, dtype=np.float32).reshape(B, 1)
        value_pred = np.asarray(values, dtype=np.float32).reshape(B, 1)
        advantages = returns_arr - value_pred
        if normalize_advantages and B > 1:
            adv_mean = float(np.mean(advantages))
            adv_std = float(np.std(advantages))
            advantages_norm = (advantages - adv_mean) / max(adv_std, 1e-6)
        else:
            advantages_norm = advantages

        targets = np.zeros_like(probs, dtype=np.float32)
        targets[np.arange(B), actions_arr] = 1.0
        log_probs = np.log(np.maximum(probs, 1e-12))
        selected_logp = log_probs[np.arange(B), actions_arr].reshape(B, 1)

        entropy = float(-np.mean(np.sum(probs * log_probs, axis=1)))
        policy_loss = float(-np.mean(advantages_norm * selected_logp))
        value_errors = value_pred - returns_arr
        value_loss = float(0.5 * np.mean(value_errors**2))
        loss = float(policy_loss + float(value_coef) * value_loss - float(entropy_coef) * entropy)

        grad_logits = (probs - targets) * advantages_norm
        grad_logits /= max(B, 1)
        if entropy_coef != 0.0:
            entropy_term = log_probs + 1.0
            entropy_center = np.sum(probs * entropy_term, axis=1, keepdims=True)
            entropy_grad = probs * (entropy_term - entropy_center)
            grad_logits += float(entropy_coef) * entropy_grad / max(B, 1)

        grad_value = float(value_coef) * value_errors / max(B, 1)

        features = np.asarray(cache["features"], dtype=np.float32)
        grad_policy_w = features.T @ grad_logits
        grad_policy_b = np.sum(grad_logits, axis=0, keepdims=True)
        grad_value_w = features.T @ grad_value
        grad_value_b = np.sum(grad_value, axis=0, keepdims=True)

        grad_features = grad_logits @ self.policy_w.T
        grad_features += grad_value @ self.value_w.T

        grad_trunk_w = [np.zeros_like(w, dtype=np.float32) for w in self.trunk_w]
        grad_trunk_b = [np.zeros_like(b, dtype=np.float32) for b in self.trunk_b]
        grad = grad_features
        activations: list[np.ndarray] = cache["activations"]
        pre_activations: list[np.ndarray] = cache["pre_activations"]
        for li in reversed(range(len(self.trunk_w))):
            grad *= self._relu_grad(pre_activations[li])
            a_prev = activations[li]
            grad_trunk_w[li] = a_prev.T @ grad
            grad_trunk_b[li] = np.sum(grad, axis=0, keepdims=True)
            if li > 0:
                grad = grad @ self.trunk_w[li].T

        weights = [*self.trunk_w, self.policy_w, self.value_w]
        biases = [*self.trunk_b, self.policy_b, self.value_b]
        grad_w = [*grad_trunk_w, grad_policy_w.astype(np.float32), grad_value_w.astype(np.float32)]
        grad_b = [*grad_trunk_b, grad_policy_b.astype(np.float32), grad_value_b.astype(np.float32)]
        optimizer.step(weights, biases, grad_w, grad_b)

        return {
            "loss": loss,
            "policy_loss": policy_loss,
            "value_loss": value_loss,
            "entropy": entropy,
            "avg_value": float(np.mean(value_pred)),
        }

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "format": "actor_critic_policy_mlp_v1",
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
            "hidden_sizes": list(self.hidden_sizes),
            "trunk_w": [w.tolist() for w in self.trunk_w],
            "trunk_b": [b.tolist() for b in self.trunk_b],
            "policy_w": self.policy_w.tolist(),
            "policy_b": self.policy_b.tolist(),
            "value_w": self.value_w.tolist(),
            "value_b": self.value_b.tolist(),
        }
        p.write_text(json.dumps(payload), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "ActorCriticPolicyNetwork":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if str(payload.get("format")) != "actor_critic_policy_mlp_v1":
            raise ValueError("Unsupported actor-critic model format")
        model = cls(
            input_dim=int(payload["input_dim"]),
            output_dim=int(payload["output_dim"]),
            hidden_sizes=tuple(int(v) for v in payload.get("hidden_sizes", [])),
            seed=0,
        )
        model.trunk_w = [np.asarray(w, dtype=np.float32) for w in payload["trunk_w"]]
        model.trunk_b = [np.asarray(b, dtype=np.float32) for b in payload["trunk_b"]]
        model.policy_w = np.asarray(payload["policy_w"], dtype=np.float32)
        model.policy_b = np.asarray(payload["policy_b"], dtype=np.float32)
        model.value_w = np.asarray(payload["value_w"], dtype=np.float32)
        model.value_b = np.asarray(payload["value_b"], dtype=np.float32)
        return model


def _discounted_returns(rewards: list[float], gamma: float) -> np.ndarray:
    out = np.zeros((len(rewards),), dtype=np.float32)
    running = 0.0
    for i in range(len(rewards) - 1, -1, -1):
        running = float(rewards[i]) + gamma * running
        out[i] = running
    return out


def _bootstrapped_returns(rewards: list[float], gamma: float, bootstrap_value: float) -> np.ndarray:
    out = np.zeros((len(rewards),), dtype=np.float32)
    running = float(bootstrap_value)
    for i in range(len(rewards) - 1, -1, -1):
        running = float(rewards[i]) + gamma * running
        out[i] = running
    return out


def _parse_hidden_sizes(raw: Any) -> tuple[int, ...]:
    vals = [v.strip() for v in str(raw).split(",") if v.strip()]
    if not vals:
        raise ValueError("hidden sizes cannot be empty")
    return tuple(int(v) for v in vals)


def _eval_model(cfg: dict[str, Any], *, model: ActorCriticPolicyNetwork | None = None) -> tuple[Any, str, float]:
    cfg = _normalize(cfg)
    common = dict(cfg.get("common") or {})
    eval_cfg = dict(cfg.get("eval") or {})
    seed = int(eval_cfg.get("seed", common.get("seed", 42)))
    episodes = int(eval_cfg.get("episodes", 50))
    if model is None:
        model_path = str(eval_cfg.get("model", ""))
        if not model_path:
            raise ValueError("pong_actor_critic eval requires eval.model")
        model = ActorCriticPolicyNetwork.load(model_path)

    params = {**common, **eval_cfg}
    env = _build_pong_env(params, seed=seed)
    stats = evaluate_pong_policy(env, model, episodes=episodes, seed_start=seed, max_steps=int(_cfg(params, "max_steps", 1000)))
    summary = (
        f"avg_score={stats.avg_score:.2f} median={stats.median_score:.2f} avg_steps={stats.avg_steps:.2f} "
        f"avg_lives={stats.avg_lives_left:.2f} avg_hits={stats.avg_player_hits:.2f}"
    )
    return stats, summary, float(stats.avg_score)


def run_train_from_config(full_cfg: dict[str, Any]) -> None:
    cfg = _normalize(full_cfg)
    common = dict(cfg.get("common") or {})
    train_cfg = dict(cfg.get("train") or {})
    eval_cfg = dict(cfg.get("eval") or {})

    seed = int(common.get("seed", 42))
    rng = np.random.default_rng(seed)
    env = _build_pong_env({**common, **train_cfg}, seed=seed)
    state0 = np.asarray(env.reset(seed=seed), dtype=np.float32)

    model = ActorCriticPolicyNetwork(
        input_dim=int(state0.shape[0]),
        output_dim=int(env.action_size),
        hidden_sizes=_parse_hidden_sizes(_cfg(train_cfg, "hidden_sizes", "128,128")),
        seed=seed,
    )
    optimizer = AdamOptimizer(
        lr=float(_cfg(train_cfg, "lr", 5e-4)),
        max_grad_norm=float(_cfg(train_cfg, "max_grad_norm", 5.0)),
    )

    episodes = int(_cfg(train_cfg, "episodes", 3000))
    gamma = float(_cfg(train_cfg, "gamma", 0.99))
    rollout_steps = int(_cfg(train_cfg, "rollout_steps", _cfg(train_cfg, "update_every", 4)))
    value_coef = float(_cfg(train_cfg, "value_coef", 0.5))
    entropy_coef = float(_cfg(train_cfg, "entropy_coef", 0.01))
    normalize_advantages = bool(_cfg(train_cfg, "normalize_advantages", True))
    eval_every = int(_cfg(train_cfg, "eval_every", 50))
    eval_episodes = int(_cfg(train_cfg, "eval_episodes", _cfg(eval_cfg, "episodes", 50)))
    max_steps = int(_cfg(train_cfg, "max_steps", 1000))
    save_dir = Path(str(_cfg(train_cfg, "save_dir", "models/pong_actor_critic")))
    save_dir.mkdir(parents=True, exist_ok=True)
    best_path = save_dir / "pong_actor_critic_best.json"
    final_path = save_dir / "pong_actor_critic_final.json"

    rolling_returns: list[float] = []
    best_metric = -float("inf")
    last_loss = 0.0
    last_policy_loss = 0.0
    last_value_loss = 0.0
    last_entropy = 0.0

    print(
        f"Training pong_actor_critic | episodes={episodes} rollout_steps={rollout_steps} gamma={gamma:.3f} "
        f"hidden={','.join(str(v) for v in model.hidden_sizes)} value_coef={value_coef:.3f} entropy_coef={entropy_coef:.4f}"
    )

    for ep in range(1, episodes + 1):
        env = _build_pong_env({**common, **train_cfg}, seed=seed + ep)
        state = np.asarray(env.reset(seed=seed + ep), dtype=np.float32)
        done = False
        steps = 0
        info = {"score": 0, "opponent_score": 0, "player_hits": 0, "lives": env.game.lives}
        rollout_states: list[np.ndarray] = []
        rollout_actions: list[int] = []
        rollout_rewards: list[float] = []
        ep_total_reward = 0.0
        ep_loss_acc = 0.0
        ep_pi_acc = 0.0
        ep_v_acc = 0.0
        ep_ent_acc = 0.0
        ep_updates = 0

        while not done and steps < max_steps:
            action = model.sample_action(state, env.legal_actions(), rng)
            next_state, reward, done, info = env.step(action)
            rollout_states.append(state.copy())
            rollout_actions.append(int(action))
            rollout_rewards.append(float(reward))
            ep_total_reward += float(reward)
            state = np.asarray(next_state, dtype=np.float32)
            steps += 1
            if len(rollout_states) >= rollout_steps or done:
                if done:
                    bootstrap_value = 0.0
                else:
                    _p, bootstrap_values, _c = model.forward(state)
                    bootstrap_value = float(bootstrap_values[0, 0])
                returns_arr = _bootstrapped_returns(rollout_rewards, gamma, bootstrap_value)
                states_arr = np.stack(rollout_states, axis=0).astype(np.float32)
                actions_arr = np.asarray(rollout_actions, dtype=np.int64)
                metrics = model.train_batch(
                    states=states_arr,
                    actions=actions_arr,
                    returns=returns_arr,
                    optimizer=optimizer,
                    value_coef=value_coef,
                    entropy_coef=entropy_coef,
                    normalize_advantages=normalize_advantages,
                )
                last_loss = float(metrics["loss"])
                last_policy_loss = float(metrics["policy_loss"])
                last_value_loss = float(metrics["value_loss"])
                last_entropy = float(metrics["entropy"])
                ep_loss_acc += last_loss
                ep_pi_acc += last_policy_loss
                ep_v_acc += last_value_loss
                ep_ent_acc += last_entropy
                ep_updates += 1
                rollout_states = []
                rollout_actions = []
                rollout_rewards = []

        ep_return = float(ep_total_reward)
        rolling_returns.append(ep_return)
        if len(rolling_returns) > 50:
            rolling_returns.pop(0)
        if ep_updates > 0:
            last_loss = ep_loss_acc / ep_updates
            last_policy_loss = ep_pi_acc / ep_updates
            last_value_loss = ep_v_acc / ep_updates
            last_entropy = ep_ent_acc / ep_updates

        print(
            f"ep={ep:4d} steps={steps:4d} score={int(info.get('score', 0)):4d} "
            f"opp={int(info.get('opponent_score', 0)):4d} hits={int(info.get('player_hits', 0)):4d} "
            f"return={ep_return:8.2f} avg_return={float(np.mean(rolling_returns)):8.2f} "
            f"loss={last_loss:.5f} pi={last_policy_loss:.5f} v={last_value_loss:.5f} ent={last_entropy:.4f}"
        )

        if ep % eval_every == 0:
            eval_cfg_local = dict(cfg)
            eval_cfg_local["eval"] = dict(eval_cfg)
            eval_cfg_local["eval"]["episodes"] = eval_episodes
            stats, summary, metric = _eval_model(eval_cfg_local, model=model)
            print(f"  eval(pong): {summary}")
            if metric > best_metric:
                best_metric = metric
                model.save(best_path)
                print(f"  saved new best checkpoint: {best_path}")

    if not best_path.exists():
        model.save(best_path)
        print(f"Saved fallback best checkpoint: {best_path}")
    model.save(final_path)
    print(f"Training complete. Final model saved to: {final_path}")


def run_eval_from_config(full_cfg: dict[str, Any]) -> None:
    cfg = _normalize(full_cfg)
    episodes = int(((cfg.get("eval") or {}).get("episodes") or 50))
    stats, _summary, _metric = _eval_model(cfg)
    print("Pong Actor-Critic Evaluation Results")
    print("-----------------------------------")
    print(f"Episodes: {episodes}")
    print(f"Average score: {stats.avg_score:.3f}")
    print(f"Median score: {stats.median_score:.3f}")
    print(f"Average steps: {stats.avg_steps:.2f}")
    print(f"Average lives left: {stats.avg_lives_left:.2f}")
    print(f"Average opponent score: {stats.avg_opponent_score:.2f}")
    print(f"Average player hits: {stats.avg_player_hits:.2f}")


def run_play_from_config(full_cfg: dict[str, Any]) -> None:
    cfg = _normalize(full_cfg)
    common = dict(cfg.get("common") or {})
    play_cfg = dict(cfg.get("play") or {})
    eval_cfg = dict(cfg.get("eval") or {})
    merged = {**eval_cfg, **play_cfg}
    if "episodes" in merged:
        merged.pop("episodes")

    seed = int(merged.get("seed", common.get("seed", 42)))
    model_path = str(merged.get("model", ""))
    if not model_path:
        raise ValueError("pong_actor_critic play requires play.model")
    mode = str(merged.get("mode", "terminal"))
    delay = float(merged.get("delay", 0.08))
    max_steps = int(merged.get("max_steps", 1000))
    close_on_end = bool(merged.get("close_on_end", False))

    env = _build_pong_env({**common, **merged}, seed=seed)
    model = ActorCriticPolicyNetwork.load(model_path)

    print("Starting Pong actor-critic autoplay. Press Q or ESC to quit.")
    if mode == "pygame":
        result = play_pong_agent.run_pygame(env, model, seed, delay, max_steps, close_on_end)
    else:
        result = play_pong_agent.run_terminal(env, model, seed, delay, max_steps)

    print("Pong actor-critic play finished")
    print(f"Final score: {result.player_score}-{result.opponent_score}")
    print(f"Lives left: {result.lives}")
    print(f"Steps: {result.steps}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Pong actor-critic workflow")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--mode", choices=["train", "eval", "play"], default="train")
    parser.add_argument("--dry-run", action="store_true")
    args, rest = parser.parse_known_args()

    cfg_path = Path(args.config)
    cfg = _load_yaml(cfg_path)
    cfg = apply_cli_overrides(cfg, _parse_overrides(rest), section=args.mode)

    if args.dry_run:
        print(json.dumps(_normalize(cfg), indent=2, sort_keys=True))
        return

    if args.mode == "train":
        run_train_from_config(cfg)
    elif args.mode == "eval":
        run_eval_from_config(cfg)
    else:
        run_play_from_config(cfg)


if __name__ == "__main__":
    main()
