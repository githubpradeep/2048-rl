"""Pure-NumPy PPO policy/value network with legal-action masking."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from ...network import AdamOptimizer


def _softmax_last(x: np.ndarray) -> np.ndarray:
    shifted = x - np.max(x, axis=-1, keepdims=True)
    exps = np.exp(shifted, dtype=np.float32)
    denom = np.maximum(np.sum(exps, axis=-1, keepdims=True), 1e-12)
    return (exps / denom).astype(np.float32)


def compute_gae(
    rewards: np.ndarray,
    values: np.ndarray,
    dones: np.ndarray,
    next_values: np.ndarray,
    *,
    gamma: float,
    gae_lambda: float,
) -> tuple[np.ndarray, np.ndarray]:
    """GAE(λ) over a rollout of shape ``(T, N)``.

    ``dones[t]`` is True if the step that produced ``rewards[t]`` ended the episode.
    ``next_values`` is V(s_{T}) per env (0 if that env's last step was terminal).
    """
    rewards = np.asarray(rewards, dtype=np.float32)
    values = np.asarray(values, dtype=np.float32)
    dones = np.asarray(dones, dtype=np.float32)
    next_values = np.asarray(next_values, dtype=np.float32)
    t_steps, n_envs = rewards.shape
    advantages = np.zeros((t_steps, n_envs), dtype=np.float32)
    last_gae = np.zeros((n_envs,), dtype=np.float32)
    for t in reversed(range(t_steps)):
        if t == t_steps - 1:
            next_nonterminal = 1.0 - dones[t]
            next_v = next_values
        else:
            next_nonterminal = 1.0 - dones[t]
            next_v = values[t + 1]
        delta = rewards[t] + float(gamma) * next_v * next_nonterminal - values[t]
        last_gae = delta + float(gamma) * float(gae_lambda) * next_nonterminal * last_gae
        advantages[t] = last_gae
    returns = advantages + values
    return advantages, returns


class PPOPolicyNetwork:
    """Shared-trunk MLP → categorical policy + scalar value (PPO)."""

    FORMAT = "ppo_policy_mlp_v1"

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        *,
        hidden_sizes: tuple[int, ...] = (512, 512),
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
        self.policy_w = (self.rng.standard_normal((feature_dim, self.output_dim), dtype=np.float32) * scale).astype(
            np.float32
        )
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

    def forward_masked(
        self,
        states: np.ndarray,
        legal_masks: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        probs, values, cache = self.forward(states)
        masks = np.asarray(legal_masks, dtype=np.float32)
        if masks.ndim == 1:
            masks = masks.reshape(1, -1)
        logits = np.asarray(cache["logits"], dtype=np.float32)
        masked_logits = np.where(masks > 0.5, logits, np.float32(-1e9))
        probs = _softmax_last(masked_logits)
        cache["logits"] = masked_logits
        cache["probs"] = probs
        cache["legal_masks"] = masks
        return probs, values, cache

    def predict_one(self, state: Any) -> list[float]:
        """Return action scores for greedy play/eval (masked later by caller)."""
        probs, _values, _cache = self.forward(np.asarray(state, dtype=np.float32))
        return [float(v) for v in probs[0].tolist()]

    def sample_actions(
        self,
        states: np.ndarray,
        legal_masks: np.ndarray,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Sample actions; returns ``(actions, log_probs, values)`` each shaped ``(N,)`` / ``(N,)`` / ``(N,)``."""
        probs, values, _cache = self.forward_masked(states, legal_masks)
        n = probs.shape[0]
        actions = np.zeros((n,), dtype=np.int64)
        log_probs = np.zeros((n,), dtype=np.float32)
        for i in range(n):
            p = probs[i].astype(np.float64)
            total = float(np.sum(p))
            if total <= 0.0:
                mask = legal_masks[i] > 0.5
                idx = np.flatnonzero(mask)
                a = int(rng.choice(idx)) if len(idx) else 0
            else:
                a = int(rng.choice(self.output_dim, p=p / total))
            actions[i] = a
            log_probs[i] = float(np.log(max(float(probs[i, a]), 1e-12)))
        return actions, log_probs, values.reshape(n).astype(np.float32)

    def ppo_train_batch(
        self,
        *,
        states: np.ndarray,
        actions: np.ndarray,
        old_log_probs: np.ndarray,
        advantages: np.ndarray,
        returns: np.ndarray,
        legal_masks: np.ndarray,
        optimizer: AdamOptimizer,
        clip_eps: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        normalize_advantages: bool = True,
    ) -> dict[str, float]:
        probs, values, cache = self.forward_masked(states, legal_masks)
        b = int(probs.shape[0])
        actions_arr = np.asarray(actions, dtype=np.int64).reshape(b)
        old_logp = np.asarray(old_log_probs, dtype=np.float32).reshape(b, 1)
        adv = np.asarray(advantages, dtype=np.float32).reshape(b, 1)
        ret = np.asarray(returns, dtype=np.float32).reshape(b, 1)
        value_pred = np.asarray(values, dtype=np.float32).reshape(b, 1)

        if normalize_advantages and b > 1:
            adv_mean = float(np.mean(adv))
            adv_std = float(np.std(adv))
            adv = (adv - adv_mean) / max(adv_std, 1e-6)

        log_probs = np.log(np.maximum(probs, 1e-12))
        new_logp = log_probs[np.arange(b), actions_arr].reshape(b, 1)
        ratio = np.exp(new_logp - old_logp)

        surr1 = ratio * adv
        surr2 = np.clip(ratio, 1.0 - float(clip_eps), 1.0 + float(clip_eps)) * adv
        policy_obj = np.minimum(surr1, surr2)
        policy_loss = float(-np.mean(policy_obj))

        value_errors = value_pred - ret
        value_loss = float(0.5 * np.mean(value_errors**2))

        entropy = float(-np.mean(np.sum(probs * log_probs, axis=1)))
        loss = float(policy_loss + float(value_coef) * value_loss - float(entropy_coef) * entropy)

        # Policy gradient: flow through unclipped surrogate when it is the min.
        active = (surr1 <= surr2).astype(np.float32)
        grad_logp = (-active * ratio * adv) / max(b, 1)  # (B, 1)

        one_hot = np.zeros_like(probs, dtype=np.float32)
        one_hot[np.arange(b), actions_arr] = 1.0
        grad_logits = (one_hot - probs) * grad_logp

        if entropy_coef != 0.0:
            entropy_term = log_probs + 1.0
            entropy_center = np.sum(probs * entropy_term, axis=1, keepdims=True)
            entropy_grad = probs * (entropy_term - entropy_center)
            grad_logits += float(entropy_coef) * entropy_grad / max(b, 1)

        grad_value = float(value_coef) * value_errors / max(b, 1)

        features = np.asarray(cache["features"], dtype=np.float32)
        grad_policy_w = features.T @ grad_logits
        grad_policy_b = np.sum(grad_logits, axis=0, keepdims=True)
        grad_value_w = features.T @ grad_value
        grad_value_b = np.sum(grad_value, axis=0, keepdims=True)

        grad_features = grad_logits @ self.policy_w.T
        grad_features += grad_value @ self.value_w.T

        grad_trunk_w = [np.zeros_like(w, dtype=np.float32) for w in self.trunk_w]
        grad_trunk_b = [np.zeros_like(b_, dtype=np.float32) for b_ in self.trunk_b]
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

        clip_frac = float(np.mean(np.abs(ratio - 1.0) > float(clip_eps)))
        return {
            "loss": loss,
            "policy_loss": policy_loss,
            "value_loss": value_loss,
            "entropy": entropy,
            "clip_frac": clip_frac,
            "approx_kl": float(np.mean(old_logp - new_logp)),
            "avg_value": float(np.mean(value_pred)),
        }

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "format": self.FORMAT,
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
    def load(cls, path: str | Path) -> "PPOPolicyNetwork":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        fmt = str(payload.get("format"))
        if fmt not in {cls.FORMAT, "actor_critic_policy_mlp_v1"}:
            raise ValueError(f"Unsupported PPO model format: {fmt}")
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
