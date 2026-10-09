"""Mono-Forward local learning for the Q-network.

Gong, Li, and Abdulla, arXiv:2501.09238. Each hidden layer has its own
readout. That readout is trained on the TD target, and the resulting
error updates only that layer. Nothing is sent backward into earlier
layers. The last layer is the Q-function used for acting, targets, eval,
and play, so a saved network loads as an ordinary MLP.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from .network import AdamOptimizer, MLPQNetwork, action_effect_grads


class MonoForwardQNetwork(MLPQNetwork):
    """Non-dueling MLP plus one local action readout per hidden layer."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_sizes: tuple[int, ...] = (128, 128),
        seed: int | None = None,
        action_effect: bool = False,
    ) -> None:
        super().__init__(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_sizes=hidden_sizes,
            seed=seed,
            dueling=False,
            action_effect=action_effect,
        )
        self.projections: list[np.ndarray] = []
        self.proj_biases: list[np.ndarray] = []
        for width in self.hidden_sizes:
            scale = np.sqrt(2.0 / float(width))
            self.projections.append(
                (self.rng.standard_normal((width, self.output_dim), dtype=np.float32) * scale).astype(np.float32)
            )
            self.proj_biases.append(np.zeros((1, self.output_dim), dtype=np.float32))


class MonoForwardTrainer:
    """Layer-wise TD updates. One Adam state per layer, no cross-layer backward pass."""

    def __init__(self, net: MonoForwardQNetwork, lr: float, max_grad_norm: float | None = None) -> None:
        self.lr = float(lr)
        clip = None if max_grad_norm is None else float(max_grad_norm)
        self.hidden_adams = [
            AdamOptimizer(lr=self.lr, max_grad_norm=clip) for _ in net.projections
        ]
        self.output_adam = AdamOptimizer(lr=self.lr, max_grad_norm=clip)
        self.steps = 0
        # After a weight-only resume the local readouts are new. Fit those
        # readouts first so their random matrices do not drag the loaded layers.
        self.hidden_weight_freeze_steps = 0

    def step(
        self,
        net: MonoForwardQNetwork,
        states: Sequence[Sequence[float]] | np.ndarray,
        actions: Sequence[int] | np.ndarray,
        targets: Sequence[float] | np.ndarray,
        loss: str = "mse",
        huber_delta: float = 1.0,
        sample_weights: Sequence[float] | np.ndarray | None = None,
        return_td_errors: bool = False,
        effect_labels: Sequence[float] | np.ndarray | None = None,
        effect_coef: float = 1.0,
    ) -> tuple[float, float] | tuple[float, float, np.ndarray]:
        if not isinstance(net, MonoForwardQNetwork):
            raise TypeError("Mono-Forward training requires MonoForwardQNetwork")
        states_arr = MLPQNetwork._as_batch(states)
        actions_arr = np.asarray(actions, dtype=np.int64).reshape(-1)
        targets_arr = np.asarray(targets, dtype=np.float32).reshape(-1)
        weights_arr = None if sample_weights is None else np.asarray(sample_weights, dtype=np.float32).reshape(-1)
        grad_scale = _grad_scale(states_arr.shape[0], weights_arr)

        effect_labels_arr = None if effect_labels is None else np.asarray(effect_labels, dtype=np.float32).reshape(-1)
        hidden = states_arr
        fit_readouts_only = self.hidden_weight_freeze_steps > 0
        last_hidden = len(self.hidden_adams) - 1
        for layer, adam in enumerate(self.hidden_adams):
            weight = net.weights[layer]
            bias = net.biases[layer]
            proj = net.projections[layer]
            proj_bias = net.proj_biases[layer]
            preact = hidden @ weight + bias
            activated = np.maximum(preact, 0.0)
            local_q = activated @ proj + proj_bias
            grad_q = _output_grad(local_q, actions_arr, targets_arr, grad_scale, loss, huber_delta)
            grad_h = grad_q @ proj.T
            grad_z = grad_h * (preact > 0.0)
            if fit_readouts_only:
                step_w = [proj]
                step_b = [proj_bias]
                step_gw = [activated.T @ grad_q]
                step_gb = [np.sum(grad_q, axis=0, keepdims=True)]
            else:
                step_w = [weight, proj]
                step_b = [bias, proj_bias]
                step_gw = [hidden.T @ grad_z, activated.T @ grad_q]
                step_gb = [np.sum(grad_z, axis=0, keepdims=True), np.sum(grad_q, axis=0, keepdims=True)]
            if (
                net.action_effect
                and layer == last_hidden
                and effect_labels_arr is not None
                and net.effect_w is not None
                and net.effect_b is not None
            ):
                logits = activated @ net.effect_w + net.effect_b
                grad_logits, accuracy = action_effect_grads(
                    logits, actions_arr, effect_labels_arr, grad_scale, effect_coef
                )
                net.last_effect_acc = accuracy
                if not fit_readouts_only:
                    grad_z = grad_z + (grad_logits @ net.effect_w.T) * (preact > 0.0)
                    step_gw[0] = hidden.T @ grad_z
                    step_gb[0] = np.sum(grad_z, axis=0, keepdims=True)
                step_w.append(net.effect_w)
                step_b.append(net.effect_b)
                step_gw.append(activated.T @ grad_logits)
                step_gb.append(np.sum(grad_logits, axis=0, keepdims=True))
            adam.step(step_w, step_b, step_gw, step_gb)
            hidden = activated
        if fit_readouts_only:
            self.hidden_weight_freeze_steps -= 1
        if net.action_effect and not self.hidden_adams and effect_labels_arr is not None:
            net.update_action_effect_head(states_arr, actions_arr, effect_labels_arr, coef=effect_coef)

        out_w = net.weights[-1]
        out_b = net.biases[-1]
        q_values = hidden @ out_w + out_b
        objective, td_abs, grad_q = _objective_and_grad(
            q_values, actions_arr, targets_arr, weights_arr, grad_scale, loss, huber_delta
        )
        self.output_adam.step(
            [out_w],
            [out_b],
            [hidden.T @ grad_q],
            [np.sum(grad_q, axis=0, keepdims=True)],
        )
        self.steps += 1
        if return_td_errors:
            return objective, float(np.mean(td_abs)), td_abs
        return objective, float(np.mean(td_abs))


def _grad_scale(batch: int, sample_weights: np.ndarray | None) -> np.ndarray:
    if sample_weights is None:
        return np.full((batch,), 1.0 / batch, dtype=np.float32)
    weights = np.asarray(sample_weights, dtype=np.float32).reshape(-1)
    if weights.shape[0] != batch:
        raise ValueError("sample_weights length must match batch size")
    total = float(np.sum(weights))
    if total <= 0.0:
        return np.full((batch,), 1.0 / batch, dtype=np.float32)
    return (weights / total).astype(np.float32)


def _output_grad(
    q_values: np.ndarray,
    actions: np.ndarray,
    targets: np.ndarray,
    grad_scale: np.ndarray,
    loss: str,
    huber_delta: float,
) -> np.ndarray:
    _, _, grad_q = _objective_and_grad(q_values, actions, targets, None, grad_scale, loss, huber_delta)
    return grad_q


def _objective_and_grad(
    q_values: np.ndarray,
    actions: np.ndarray,
    targets: np.ndarray,
    sample_weights: np.ndarray | None,
    grad_scale: np.ndarray,
    loss: str,
    huber_delta: float,
) -> tuple[float, np.ndarray, np.ndarray]:
    batch = q_values.shape[0]
    taken = q_values[np.arange(batch), actions]
    td_errors = taken - targets
    td_abs = np.abs(td_errors).astype(np.float32)
    if loss == "huber":
        delta = float(huber_delta)
        quadratic = td_abs <= delta
        per_sample = np.where(quadratic, 0.5 * td_errors**2, delta * (td_abs - 0.5 * delta))
        residual = np.where(quadratic, td_errors, delta * np.sign(td_errors))
    else:
        per_sample = 0.5 * td_errors**2
        residual = td_errors
    if sample_weights is None:
        objective = float(np.mean(per_sample))
    else:
        total = float(np.sum(sample_weights))
        objective = float(np.sum(per_sample * sample_weights) / max(total, 1e-8))
    grad_q = np.zeros_like(q_values, dtype=np.float32)
    grad_q[np.arange(batch), actions] = residual.astype(np.float32) * grad_scale
    return objective, td_abs, grad_q
