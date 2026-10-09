"""Dust: activation-space zeroth-order gradients for the Q-network.

Implements the MLP case of Dust (Dahal et al., 2026, https://qlabs.sh/research/dust).
Weight-space evolution, including EGGROLL, draws one population member per
forward pass. Dust adds Gaussian noise to a linear layer's output, independently
on every batch row, and rewards that noise by how much it lowers the TD loss.
The reward-weighted noise is the estimated output error. Its outer product with
the layer's input is the weight gradient, which Adam applies.

Each layer is jittered in its own forward pass so one layer's noise is not
credited to another. There is no attention over future rows, so a row is
rewarded only by its own loss (the paper's ``gamma = 0``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from .network import AdamOptimizer, MLPQNetwork


@dataclass
class DustConfig:
    """Hyperparameters for :class:`DustOptimizer`.

    Attributes:
        population: Draws ``K`` per layer. Each draw jitters every batch row.
        sigma: Noise scale as a fraction of that layer's clean pre-activation
            standard deviation. The ``1/sigma`` in the estimator puts the
            gradient back in loss units.
        lr: Adam learning rate applied to the estimated gradient.
        seed: Seed for the activation noise.
    """

    population: int = 32
    sigma: float = 0.1
    lr: float = 1e-3
    seed: int = 0

    def validate(self) -> None:
        if int(self.population) < 1:
            raise ValueError("dust population must be >= 1")
        if float(self.sigma) <= 0.0:
            raise ValueError("dust sigma must be > 0")
        if float(self.lr) < 0.0:
            raise ValueError("dust lr must be >= 0")


def dust_config_from_params(params: dict[str, Any]) -> DustConfig:
    """Build a :class:`DustConfig` from merged YAML/CLI params."""
    defaults = DustConfig()
    return DustConfig(
        population=int(params.get("dust_population", defaults.population)),
        sigma=float(params.get("dust_sigma", defaults.sigma)),
        lr=float(params.get("dust_lr", params.get("lr", defaults.lr))),
        seed=int(params.get("seed", 42)),
    )


def _per_sample_losses(
    q_values: np.ndarray,
    actions: np.ndarray,
    targets: np.ndarray,
    loss: str,
    huber_delta: float,
) -> np.ndarray:
    """Per-row TD loss. ``q_values`` is ``(K, batch, actions)``."""
    batch = q_values.shape[1]
    pred = q_values[:, np.arange(batch), actions]
    errors = pred - targets.reshape(1, -1)
    if loss == "huber":
        delta = float(huber_delta)
        absolute = np.abs(errors)
        quadratic = absolute <= delta
        return np.where(quadratic, 0.5 * errors**2, delta * (absolute - 0.5 * delta)).astype(np.float32)
    return (0.5 * errors**2).astype(np.float32)


def _grad_scale(batch: int, sample_weights: np.ndarray | None) -> np.ndarray:
    """Same per-row multiplier the backprop trainer applies to the TD error."""
    if sample_weights is None:
        return np.full((batch,), 1.0 / batch, dtype=np.float32)
    weights = np.asarray(sample_weights, dtype=np.float32).reshape(-1)
    if weights.shape[0] != batch:
        raise ValueError("sample_weights length must match batch size")
    total = float(np.sum(weights))
    if total <= 0.0:
        return np.full((batch,), 1.0 / batch, dtype=np.float32)
    return (weights / total).astype(np.float32)


def _output_error(losses: np.ndarray, noise: np.ndarray, sigma: float) -> np.ndarray:
    """Equation 2 with ``gamma = 0``.

    ``losses`` is ``(K, batch)`` and ``noise`` is ``(K, batch, width)``.
    A draw whose loss is below the draw mean gets a positive reward, and the
    leading minus sign makes the estimate a descent direction.
    """
    reward = losses.mean(axis=0, keepdims=True) - losses
    weighted = reward[..., None] * noise
    return (-np.mean(weighted, axis=0) / np.float32(sigma)).astype(np.float32)


def _layer_sigma(clean: np.ndarray, fraction: float) -> float:
    scale = float(np.std(clean))
    if scale < 1e-4:
        scale = 1e-4
    return max(float(fraction) * scale, 1e-4)


def _q_from_non_dueling(net: MLPQNetwork, layer: int, preact: np.ndarray) -> np.ndarray:
    """Q-values from a noisy pre-activation, with later layers left clean."""
    current = preact
    last = len(net.weights) - 1
    if layer < last:
        current = np.maximum(current, 0.0)
    for index in range(layer + 1, len(net.weights)):
        current = np.einsum("kbi,io->kbo", current, net.weights[index], optimize=True) + net.biases[index]
        if index < last:
            current = np.maximum(current, 0.0)
    return current


def _q_from_trunk(net: MLPQNetwork, layer: int, preact: np.ndarray) -> np.ndarray:
    """Dueling Q-values from a noisy trunk pre-activation."""
    current = np.maximum(preact, 0.0)
    for index in range(layer + 1, len(net.weights)):
        current = np.einsum("kbi,io->kbo", current, net.weights[index], optimize=True) + net.biases[index]
        current = np.maximum(current, 0.0)
    return _dueling_q(net, current)


def _dueling_q(net: MLPQNetwork, features: np.ndarray) -> np.ndarray:
    assert net.value_w is not None and net.value_b is not None
    assert net.advantage_w is not None and net.advantage_b is not None
    value = np.einsum("kbf,fo->kbo", features, net.value_w, optimize=True) + net.value_b
    advantage = np.einsum("kbf,fo->kbo", features, net.advantage_w, optimize=True) + net.advantage_b
    return value + (advantage - np.mean(advantage, axis=-1, keepdims=True))


def _grads_from_error(inputs: np.ndarray, error: np.ndarray, grad_scale: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    scaled = error * grad_scale.reshape(-1, 1)
    return inputs.T @ scaled, np.sum(scaled, axis=0, keepdims=True)


def _estimate_layer(
    rng: np.random.Generator,
    inputs: np.ndarray,
    clean: np.ndarray,
    width: int,
    population: int,
    sigma_fraction: float,
    q_of_preact,
    actions: np.ndarray,
    targets: np.ndarray,
    loss: str,
    huber_delta: float,
    grad_scale: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    sigma = _layer_sigma(clean, sigma_fraction)
    noise = rng.standard_normal((population, inputs.shape[0], width)).astype(np.float32)
    preact = clean.reshape(1, *clean.shape) + np.float32(sigma) * noise
    q_values = q_of_preact(preact)
    losses = _per_sample_losses(q_values, actions, targets, loss, huber_delta)
    error = _output_error(losses, noise, sigma)
    return _grads_from_error(inputs, error, grad_scale)


def estimate_gradients(
    net: MLPQNetwork,
    states: np.ndarray,
    actions: np.ndarray,
    targets: np.ndarray,
    rng: np.random.Generator,
    population: int,
    sigma: float,
    loss: str,
    huber_delta: float,
    sample_weights: np.ndarray | None,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Dust weight and bias gradients in the order Adam expects."""
    q_values, cache = net.forward(states)
    del q_values
    batch = states.shape[0]
    grad_scale = _grad_scale(batch, sample_weights)

    if not net.dueling:
        activations: list[np.ndarray] = cache["activations"]
        preacts: list[np.ndarray] = cache["pre_activations"]
        grad_w: list[np.ndarray] = []
        grad_b: list[np.ndarray] = []
        for layer, (weight, bias) in enumerate(zip(net.weights, net.biases)):
            del bias
            gw, gb = _estimate_layer(
                rng,
                activations[layer],
                preacts[layer],
                weight.shape[1],
                population,
                sigma,
                lambda preact, layer=layer: _q_from_non_dueling(net, layer, preact),
                actions,
                targets,
                loss,
                huber_delta,
                grad_scale,
            )
            grad_w.append(gw.astype(np.float32))
            grad_b.append(gb.astype(np.float32))
        return grad_w, grad_b

    trunk_in: list[np.ndarray] = cache["trunk_activations"]
    trunk_pre: list[np.ndarray] = cache["trunk_pre_activations"]
    features: np.ndarray = cache["features"]
    value: np.ndarray = cache["value"]
    advantage: np.ndarray = cache["advantage"]
    assert net.value_w is not None and net.value_b is not None
    assert net.advantage_w is not None and net.advantage_b is not None

    grad_w = []
    grad_b = []
    for layer, weight in enumerate(net.weights):
        gw, gb = _estimate_layer(
            rng,
            trunk_in[layer],
            trunk_pre[layer],
            weight.shape[1],
            population,
            sigma,
            lambda preact, layer=layer: _q_from_trunk(net, layer, preact),
            actions,
            targets,
            loss,
            huber_delta,
            grad_scale,
        )
        grad_w.append(gw.astype(np.float32))
        grad_b.append(gb.astype(np.float32))

    def q_from_value(preact: np.ndarray) -> np.ndarray:
        centered = advantage - np.mean(advantage, axis=1, keepdims=True)
        return preact + centered.reshape(1, *centered.shape)

    def q_from_advantage(preact: np.ndarray) -> np.ndarray:
        centered = preact - np.mean(preact, axis=-1, keepdims=True)
        return value.reshape(1, *value.shape) + centered

    value_w, value_b = _estimate_layer(
        rng,
        features,
        value,
        net.value_w.shape[1],
        population,
        sigma,
        q_from_value,
        actions,
        targets,
        loss,
        huber_delta,
        grad_scale,
    )
    advantage_w, advantage_b = _estimate_layer(
        rng,
        features,
        advantage,
        net.advantage_w.shape[1],
        population,
        sigma,
        q_from_advantage,
        actions,
        targets,
        loss,
        huber_delta,
        grad_scale,
    )
    grad_w.extend([value_w.astype(np.float32), advantage_w.astype(np.float32)])
    grad_b.extend([value_b.astype(np.float32), advantage_b.astype(np.float32)])
    return grad_w, grad_b


class DustOptimizer:
    """Replace backprop with a Dust gradient estimate, then step Adam."""

    def __init__(self, config: DustConfig | None = None, max_grad_norm: float | None = None, **kwargs: Any) -> None:
        if config is None:
            config = DustConfig(**kwargs)
        elif kwargs:
            raise TypeError("pass either a config or keyword overrides, not both")
        config.validate()
        self.config = config
        self.rng = np.random.default_rng(config.seed)
        self.adam = AdamOptimizer(lr=config.lr, max_grad_norm=max_grad_norm)
        self.steps = 0

    def step(
        self,
        net: MLPQNetwork,
        states: Sequence[Sequence[float]] | np.ndarray,
        actions: Sequence[int] | np.ndarray,
        targets: Sequence[float] | np.ndarray,
        loss: str = "mse",
        huber_delta: float = 1.0,
        sample_weights: Sequence[float] | np.ndarray | None = None,
        return_td_errors: bool = False,
    ) -> tuple[float, float] | tuple[float, float, np.ndarray]:
        """One Dust update. The reported loss is the clean network before the step."""
        states_arr = MLPQNetwork._as_batch(states)
        actions_arr = np.asarray(actions, dtype=np.int64).reshape(-1)
        targets_arr = np.asarray(targets, dtype=np.float32).reshape(-1)
        weights_arr = None if sample_weights is None else np.asarray(sample_weights, dtype=np.float32).reshape(-1)

        clean_q, _ = net.forward(states_arr)
        clean_losses = _per_sample_losses(clean_q.reshape(1, *clean_q.shape), actions_arr, targets_arr, loss, huber_delta)[0]
        if weights_arr is None:
            objective = float(np.mean(clean_losses))
        else:
            total = float(np.sum(weights_arr))
            objective = float(np.sum(clean_losses * weights_arr) / max(total, 1e-8))
        taken = clean_q[np.arange(states_arr.shape[0]), actions_arr]
        td_abs = np.abs(taken - targets_arr).astype(np.float32)

        grad_w, grad_b = estimate_gradients(
            net,
            states_arr,
            actions_arr,
            targets_arr,
            self.rng,
            int(self.config.population),
            float(self.config.sigma),
            loss,
            huber_delta,
            weights_arr,
        )
        weights, biases = net._parameter_lists()
        self.adam.step(weights, biases, grad_w, grad_b)
        self.steps += 1
        if return_td_errors:
            return objective, float(np.mean(td_abs)), td_abs
        return objective, float(np.mean(td_abs))
