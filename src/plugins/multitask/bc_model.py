from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from ...network import AdamOptimizer


def _relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, x)


def _relu_grad(x: np.ndarray) -> np.ndarray:
    return (x > 0).astype(np.float32)


def _softmax_last(x: np.ndarray) -> np.ndarray:
    shifted = x - np.max(x, axis=-1, keepdims=True)
    exps = np.exp(shifted, dtype=np.float32)
    denom = np.maximum(np.sum(exps, axis=-1, keepdims=True), 1e-12)
    return (exps / denom).astype(np.float32)


class MultitaskBCOptimizers:
    def __init__(self, *, lr: float, games: list[str], max_grad_norm: float | None = None) -> None:
        self.shared = AdamOptimizer(lr=lr, max_grad_norm=max_grad_norm)
        self.adapter: dict[str, AdamOptimizer] = {g: AdamOptimizer(lr=lr, max_grad_norm=max_grad_norm) for g in games}
        self.head: dict[str, AdamOptimizer] = {g: AdamOptimizer(lr=lr, max_grad_norm=max_grad_norm) for g in games}


class MultitaskBCNetwork:
    """Simple shared MLP for behavior cloning across games.

    Architecture:
    - per-game linear adapter -> shared hidden width
    - shared ReLU MLP trunk
    - per-game linear action head
    """

    def __init__(
        self,
        game_input_dims: dict[str, int],
        game_action_dims: dict[str, int],
        *,
        shared_width: int = 128,
        trunk_hidden_sizes: tuple[int, ...] = (128, 128),
        seed: int | None = None,
    ) -> None:
        self.rng = np.random.default_rng(seed)
        self.games = sorted(str(g) for g in game_input_dims.keys())
        self.game_input_dims = {g: int(game_input_dims[g]) for g in self.games}
        self.game_action_dims = {g: int(game_action_dims[g]) for g in self.games}
        self.shared_width = int(shared_width)
        self.trunk_hidden_sizes = tuple(int(v) for v in trunk_hidden_sizes)

        def init_w(in_dim: int, out_dim: int) -> np.ndarray:
            scale = np.sqrt(2.0 / max(in_dim, 1))
            return (self.rng.standard_normal((in_dim, out_dim), dtype=np.float32) * scale).astype(np.float32)

        self.adapter_w: dict[str, np.ndarray] = {}
        self.adapter_b: dict[str, np.ndarray] = {}
        for game in self.games:
            self.adapter_w[game] = init_w(self.game_input_dims[game], self.shared_width)
            self.adapter_b[game] = np.zeros((1, self.shared_width), dtype=np.float32)

        layer_sizes = [self.shared_width, *self.trunk_hidden_sizes]
        self.shared_w: list[np.ndarray] = []
        self.shared_b: list[np.ndarray] = []
        for in_dim, out_dim in zip(layer_sizes[:-1], layer_sizes[1:]):
            self.shared_w.append(init_w(in_dim, out_dim))
            self.shared_b.append(np.zeros((1, out_dim), dtype=np.float32))
        self.feature_dim = self.trunk_hidden_sizes[-1] if self.trunk_hidden_sizes else self.shared_width

        self.head_w: dict[str, np.ndarray] = {}
        self.head_b: dict[str, np.ndarray] = {}
        for game in self.games:
            self.head_w[game] = init_w(self.feature_dim, self.game_action_dims[game])
            self.head_b[game] = np.zeros((1, self.game_action_dims[game]), dtype=np.float32)

    def _forward_game(self, game: str, states: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
        x = np.asarray(states, dtype=np.float32)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        if x.shape[1] != self.game_input_dims[game]:
            raise ValueError(f"{game}: expected input_dim={self.game_input_dims[game]}, got {x.shape[1]}")

        ad_pre = x @ self.adapter_w[game] + self.adapter_b[game]
        h = _relu(ad_pre)

        acts = [h]
        pres: list[np.ndarray] = []
        current = h
        for w, b in zip(self.shared_w, self.shared_b):
            z = current @ w + b
            pres.append(z)
            current = _relu(z)
            acts.append(current)

        features = current
        logits = features @ self.head_w[game] + self.head_b[game]
        cache = {
            "x": x,
            "ad_pre": ad_pre,
            "adapter_out": h,
            "shared_acts": acts,
            "shared_pres": pres,
            "features": features,
        }
        return logits.astype(np.float32), cache

    def predict_action_scores(
        self,
        *,
        game: str,
        state: np.ndarray,
        legal_actions: list[int] | None = None,
    ) -> np.ndarray:
        logits, _ = self._forward_game(game, np.asarray(state, dtype=np.float32).reshape(1, -1))
        scores = logits[0].copy()
        if legal_actions is not None and len(legal_actions) > 0:
            mask = np.zeros_like(scores, dtype=np.float32)
            mask[np.asarray(legal_actions, dtype=np.int64)] = 1.0
            scores = np.where(mask > 0.5, scores, -1e9)
        return scores.astype(np.float32)

    def train_game_batch(
        self,
        *,
        game: str,
        states: np.ndarray,
        actions: np.ndarray,
        legal_masks: np.ndarray,
        optimizers: MultitaskBCOptimizers,
        label_smoothing: float = 0.0,
        teacher_q_values: np.ndarray | None = None,
        distill: bool = False,
        distill_alpha: float = 0.6,
        distill_temperature: float = 2.0,
    ) -> dict[str, float]:
        logits, cache = self._forward_game(game, states)
        B, A = logits.shape
        actions_arr = np.asarray(actions, dtype=np.int64).reshape(B)
        legal_arr = np.asarray(legal_masks, dtype=np.float32).reshape(B, A)

        masked_logits = np.where(legal_arr > 0.5, logits, -1e9)
        probs = _softmax_last(masked_logits)

        targets = np.zeros_like(probs, dtype=np.float32)
        targets[np.arange(B), actions_arr] = 1.0
        if label_smoothing > 0.0:
            ls = float(label_smoothing)
            legal_norm = np.maximum(np.sum(legal_arr, axis=1, keepdims=True), 1.0)
            smooth = legal_arr / legal_norm
            targets = (1.0 - ls) * targets + ls * smooth

        log_probs = np.log(np.maximum(probs, 1e-12))
        ce_loss = float(-np.sum(targets * log_probs) / max(B, 1))
        pred = np.argmax(probs, axis=1)
        acc = float(np.mean(pred == actions_arr)) if B > 0 else 0.0

        grad_ce = (probs - targets) / max(B, 1)
        grad_ce = np.where(legal_arr > 0.5, grad_ce, 0.0)

        use_distill = bool(distill and teacher_q_values is not None)
        distill_loss = 0.0
        grad_logits = grad_ce
        if use_distill:
            temp = max(float(distill_temperature), 1e-4)
            alpha = float(np.clip(distill_alpha, 0.0, 1.0))
            teacher_q = np.asarray(teacher_q_values, dtype=np.float32).reshape(B, A)
            masked_teacher_q = np.where(legal_arr > 0.5, teacher_q, -1e9)
            student_probs_t = _softmax_last(masked_logits / temp)
            teacher_probs_t = _softmax_last(masked_teacher_q / temp)
            distill_log_probs = np.log(np.maximum(student_probs_t, 1e-12))
            distill_loss = float(-np.sum(teacher_probs_t * distill_log_probs) / max(B, 1))
            grad_distill = (student_probs_t - teacher_probs_t) / (max(B, 1) * temp)
            grad_distill = np.where(legal_arr > 0.5, grad_distill, 0.0)
            grad_logits = alpha * grad_ce + (1.0 - alpha) * grad_distill
            loss = alpha * ce_loss + (1.0 - alpha) * distill_loss
        else:
            loss = ce_loss

        # Head gradients
        features = np.asarray(cache["features"], dtype=np.float32)
        grad_head_w = features.T @ grad_logits
        grad_head_b = np.sum(grad_logits, axis=0, keepdims=True)
        grad_features = grad_logits @ self.head_w[game].T

        # Shared trunk gradients
        grad_shared_w = [np.zeros_like(w, dtype=np.float32) for w in self.shared_w]
        grad_shared_b = [np.zeros_like(b, dtype=np.float32) for b in self.shared_b]
        grad = grad_features
        shared_acts = cache["shared_acts"]
        shared_pres = cache["shared_pres"]
        for li in reversed(range(len(self.shared_w))):
            a_prev = shared_acts[li]
            grad *= _relu_grad(shared_pres[li])
            grad_shared_w[li] = a_prev.T @ grad
            grad_shared_b[li] = np.sum(grad, axis=0, keepdims=True)
            grad = grad @ self.shared_w[li].T

        # Adapter gradients
        grad_adapter_post = grad if self.shared_w else grad_features
        grad_adapter_pre = grad_adapter_post * _relu_grad(cache["ad_pre"])
        grad_adapter_w = cache["x"].T @ grad_adapter_pre
        grad_adapter_b = np.sum(grad_adapter_pre, axis=0, keepdims=True)

        if self.shared_w:
            optimizers.shared.step(self.shared_w, self.shared_b, grad_shared_w, grad_shared_b)
        optimizers.adapter[game].step(
            [self.adapter_w[game]],
            [self.adapter_b[game]],
            [grad_adapter_w.astype(np.float32)],
            [grad_adapter_b.astype(np.float32)],
        )
        optimizers.head[game].step(
            [self.head_w[game]],
            [self.head_b[game]],
            [grad_head_w.astype(np.float32)],
            [grad_head_b.astype(np.float32)],
        )

        metrics = {"loss": loss, "acc": acc, "ce_loss": ce_loss}
        if use_distill:
            metrics["distill_loss"] = distill_loss
        return metrics

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "format": "multitask_bc_mlp_v1",
            "games": self.games,
            "game_input_dims": self.game_input_dims,
            "game_action_dims": self.game_action_dims,
            "shared_width": self.shared_width,
            "trunk_hidden_sizes": list(self.trunk_hidden_sizes),
            "adapter_w": {g: self.adapter_w[g].tolist() for g in self.games},
            "adapter_b": {g: self.adapter_b[g].tolist() for g in self.games},
            "shared_w": [w.tolist() for w in self.shared_w],
            "shared_b": [b.tolist() for b in self.shared_b],
            "head_w": {g: self.head_w[g].tolist() for g in self.games},
            "head_b": {g: self.head_b[g].tolist() for g in self.games},
        }
        p.write_text(json.dumps(payload), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "MultitaskBCNetwork":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if str(payload.get("format")) != "multitask_bc_mlp_v1":
            raise ValueError("Unsupported multitask BC model format")
        model = cls(
            game_input_dims={str(k): int(v) for k, v in dict(payload["game_input_dims"]).items()},
            game_action_dims={str(k): int(v) for k, v in dict(payload["game_action_dims"]).items()},
            shared_width=int(payload["shared_width"]),
            trunk_hidden_sizes=tuple(int(v) for v in payload.get("trunk_hidden_sizes", [])),
            seed=0,
        )
        for g in model.games:
            model.adapter_w[g] = np.asarray(payload["adapter_w"][g], dtype=np.float32)
            model.adapter_b[g] = np.asarray(payload["adapter_b"][g], dtype=np.float32)
            model.head_w[g] = np.asarray(payload["head_w"][g], dtype=np.float32)
            model.head_b[g] = np.asarray(payload["head_b"][g], dtype=np.float32)
        model.shared_w = [np.asarray(w, dtype=np.float32) for w in payload["shared_w"]]
        model.shared_b = [np.asarray(b, dtype=np.float32) for b in payload["shared_b"]]
        model.feature_dim = model.shared_w[-1].shape[1] if model.shared_w else model.shared_width
        return model
