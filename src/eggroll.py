"""EGGROLL: gradient-free low-rank Evolution Strategies.

Implements *Evolution Strategies at the Hyperscale* (Sarkar et al., ICML 2026,
arXiv:2511.16652). ``--optimizer eggroll`` keeps the DQN loop and replaces
backprop: each perturbed network is scored by TD loss on the replay minibatch,
and that search direction is applied by Adam. ``eggroll_objective: return``
is the paper's episode-return controller instead.

Core idea
---------
Sample a population of *perturbed* networks, score them, and step along the
fitness-weighted average of the perturbations. Naive Gaussian ES perturbs
each weight matrix with a full-rank
``E ~ N(0, I_m, I_n)``. EGGROLL samples thin factors instead,

    E = (1 / sqrt(r)) @ A @ B.T,    A in R^{m x r},  B in R^{n x r},

which costs ``r * (m + n)`` instead of ``m * n`` numbers per member. Individual
members are rank-``r``. The update sums the independent members, so its rank
is ``min(N_eff * r, m, n)``. Antithetic pairs ``(E, -E)`` span one direction,
so ``N_eff`` is ``N / 2`` when antithetic sampling is on and ``N`` otherwise.
Forward passes never materialise ``E`` thanks to

    u @ (W + sigma * E) = u @ W + (sigma / sqrt(r)) * (u @ A) @ B.T

Fitness
-------
DQN training scores each perturbed network by ``-TD loss`` on the shared
replay minibatch, then hands the resulting direction to Adam. That is the
same objective the working Adam trainer differentiates. Episode return is
available for the paper's control setup, and it is not what this Pong uses.

Practical departures from the paper
-----------------------------------
The paper's Algorithm 1 is a minimal statement of the estimator. The additions
below keep one ``sigma`` and one ``lr`` usable across a heterogeneous MLP.
The applied step is still ``lr * mean_i(w_i E_i)``.

* **Fitness shaping.** Raw fitnesses cannot be used directly: the mean
  ``E[f_i]`` term is a direction-independent constant that swamps the signal.
  Default ``"standardized"`` (subtract mean, divide by std). ``"centered"`` and
  rank-based ``"rank"`` (Wierstra et al., 2014) are also available.
* **Antithetic sampling.** Pairs ``(E, -E)`` halve the estimator variance.
* **Per-tensor relative sigma.** ``sigma_mode="relative"`` scales sigma per
  tensor by that tensor's own weight std, because this repo's He init gives
  very different per-coordinate scales across layers.
* **Per-tensor update cap.** ``normalize_layer_updates`` (default on) limits
  each tensor's step to ``lr`` times its own Frobenius norm. It does not
  inflate a weak estimate up to that size: the step is the paper's mean
  ``lr * mean_i(w_i E_i)``, and the cap only binds when that mean would move
  a layer by more than ``lr`` relative. Forcing every update up to the cap
  turns a noisy minibatch into a fixed-size random walk, which is what made
  long DQN runs fail to improve. Set the flag false for the uncapped mean.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from .network import MLPQNetwork

FITNESS_SHAPINGS = ("standardized", "centered", "rank", "raw")
SIGMA_MODES = ("relative", "absolute")
LR_SCHEDULES = ("constant", "cosine")
UPDATE_RULES = ("mean", "adam")


@dataclass
class EggrollConfig:
    """Hyperparameters for :class:`EggrollOptimizer`.

    Attributes:
        rank: Perturbation rank ``r``. Must be ``< min(m, n)`` for a weight
            tensor to actually be low-rank. ``1`` is the cheapest and often
            sufficient choice; ``2``-``4`` tracks full-rank ES more closely.
        population: Number of perturbed members evaluated per update. Must be
            even when ``antithetic`` is enabled.
        sigma: Perturbation radius. Under ``sigma_mode="relative"`` this is a
            fraction of each tensor's weight std; under ``"absolute"`` it is
            used as-is.
        lr: Learning rate ``alpha_t``. The paper absorbs the ``1/sigma`` factor
            of the true ES gradient into ``alpha_t``, so this is not directly
            comparable to an Adam learning rate.
        fitness_shaping: See module docstring.
        antithetic: Pair each member with its negation.
        sigma_mode: ``"relative"`` or ``"absolute"``.
        lr_schedule: ``"constant"`` or ``"cosine"`` (requires ``total_steps``).
        lr_final: Learning rate floor for the cosine schedule.
        sigma_final: Optional sigma floor; ``None`` keeps sigma constant.
        normalize_layer_updates: Cap each tensor's update at ``lr`` times its
            own Frobenius norm. The applied step is still the mean
            ``lr * mean_i(w_i E_i)``; the cap only shrinks updates that would
            otherwise exceed that relative size. Defaults to ``True``. Set
            ``False`` for the paper's uncapped mean.
        max_update_norm: Optional cap on each tensor's update Frobenius norm
            (the ES analogue of Adam's ``max_grad_norm``). Used by the
            ``"mean"`` update rule.
        update_rule: ``"mean"`` applies ``lr * mean_i(w_i E_i)`` directly.
            ``"adam"`` treats that direction as a gradient and steps with
            Adam, which is what the reference implementation feeds to optax.
            Adam keeps a consistent TD signal moving after the raw average
            has become small.
        total_steps: Updates expected over training, used by the schedules.
        seed: Seed for the perturbation RNG.
    """

    rank: int = 1
    population: int = 32
    sigma: float = 0.1
    lr: float = 0.01
    fitness_shaping: str = "standardized"
    antithetic: bool = True
    sigma_mode: str = "relative"
    lr_schedule: str = "constant"
    lr_final: float = 0.0
    sigma_final: float | None = None
    normalize_layer_updates: bool = True
    max_update_norm: float | None = None
    update_rule: str = "mean"
    total_steps: int | None = None
    seed: int = 0

    def validate(self) -> None:
        if int(self.rank) < 1:
            raise ValueError("eggroll rank must be >= 1")
        if int(self.population) < 2:
            raise ValueError("eggroll population must be >= 2")
        if float(self.sigma) < 0.0:
            raise ValueError("eggroll sigma must be >= 0")
        if self.fitness_shaping not in FITNESS_SHAPINGS:
            raise ValueError(f"fitness_shaping must be one of {FITNESS_SHAPINGS}")
        if self.sigma_mode not in SIGMA_MODES:
            raise ValueError(f"sigma_mode must be one of {SIGMA_MODES}")
        if self.lr_schedule not in LR_SCHEDULES:
            raise ValueError(f"lr_schedule must be one of {LR_SCHEDULES}")
        if self.update_rule not in UPDATE_RULES:
            raise ValueError(f"update_rule must be one of {UPDATE_RULES}")
        if self.lr_schedule == "cosine" and not self.total_steps:
            raise ValueError("lr_schedule='cosine' requires total_steps")
        if self.antithetic and int(self.population) % 2 != 0:
            raise ValueError("antithetic sampling needs an even population")


def shape_fitnesses(fitnesses: np.ndarray, shaping: str) -> np.ndarray:
    """Map raw fitnesses to ES weights.

    The result is the ``w_i`` in ``sum_i w_i E_i``. For every mode except
    ``"raw"`` the weights are zero-mean, which is what makes the estimator an
    unbiased direction estimate rather than a fitness-magnitude offset.
    """
    values = np.asarray(fitnesses, dtype=np.float64).reshape(-1)
    if values.size == 0:
        raise ValueError("cannot shape an empty fitness vector")
    if shaping == "raw":
        return values

    centered = values - float(np.mean(values))
    if shaping == "centered":
        return centered
    if shaping == "standardized":
        std = float(np.std(values))
        if std <= 0.0:
            return np.zeros_like(centered)
        return centered / std

    # Rank-based shaping (Wierstra et al., 2014): weights span [-0.5, 0.5] by
    # rank, so they are invariant to the scale and skew of the fitnesses.
    order = np.argsort(values, kind="stable")
    weights = np.empty(values.size, dtype=np.float64)
    weights[order] = np.linspace(-0.5, 0.5, values.size)
    return weights


class TensorPerturbation:
    """Low-rank (or full-rank) search perturbation for one parameter tensor.

    Holds a *reference* to the live parameter array, so updates are written
    straight into the network. Low-rank tensors store only the thin factors
    ``A`` of shape ``(N, m, r)`` and ``B`` of shape ``(N, n, r)``; the dense
    ``E_i = A_i B_i.T / sqrt(r)`` is never materialised.
    """

    def __init__(self, param: np.ndarray, rank: int) -> None:
        self.param = param
        self.shape = tuple(int(v) for v in param.shape)
        m, n = self.shape
        # A row vector (e.g. a bias of shape (1, n)) has no width to compress,
        # so it falls back to a full-rank perturbation.
        self.low_rank = len(self.shape) == 2 and m > 1 and n > 1 and rank < min(m, n)
        self.rank = rank if self.low_rank else 0
        self.inv_sqrt_r = 1.0 / np.sqrt(float(self.rank)) if self.low_rank else 1.0
        # Biases are added to pre-activations rather than contracted with the
        # input, so they need an additive offset instead of a matmul.
        self.is_bias = len(self.shape) == 2 and self.shape[0] == 1
        self.population = 0
        self.A: np.ndarray | None = None
        self.B: np.ndarray | None = None
        self.E: np.ndarray | None = None

    def configure(self, population: int, rng: np.random.Generator) -> None:
        """(Re)allocate storage for ``population`` members."""
        m, n = self.shape
        self.population = int(population)
        if self.low_rank:
            self.A = rng.standard_normal((population, m, self.rank)).astype(np.float32)
            self.B = rng.standard_normal((population, n, self.rank)).astype(np.float32)
            self.E = None
        else:
            self.A = None
            self.B = None
            self.E = rng.standard_normal((population, m, n)).astype(np.float32)

    def sigma_for(self, relative: bool, sigma: float, fallback_scale: float) -> float:
        """Effective sigma for this tensor under the configured ``sigma_mode``.

        ``fallback_scale`` covers tensors whose own std is zero -- biases start
        at zero, so a purely relative sigma would freeze them forever.
        """
        if not relative:
            return float(sigma)
        scale = float(np.std(self.param))
        if scale <= 1e-12:
            scale = float(fallback_scale)
        return float(sigma) * scale

    def sample(self, rng: np.random.Generator, antithetic: bool) -> None:
        """Redraw every member, negating the second half when antithetic."""
        population = self.population
        if self.low_rank:
            assert self.A is not None and self.B is not None
            self.A[:] = rng.standard_normal(self.A.shape).astype(np.float32)
            self.B[:] = rng.standard_normal(self.B.shape).astype(np.float32)
            if antithetic:
                half = population // 2
                # Negate exactly one factor: flipping both would leave A B.T
                # unchanged and reproduce the member instead of mirroring it.
                self.A[half:] = -self.A[:half]
                self.B[half:] = self.B[:half]
        else:
            assert self.E is not None
            self.E[:] = rng.standard_normal(self.E.shape).astype(np.float32)
            if antithetic:
                half = population // 2
                self.E[half:] = -self.E[:half]

    def forward_offset(self, cur: np.ndarray, sigma: float) -> np.ndarray:
        """Forward-pass contribution of every member's perturbation.

        Args:
            cur: Either ``(B, m)`` (shared input, used by the first layer) or
                ``(N, B, m)`` (per-member input from a previous layer).
            sigma: Perturbation radius for this tensor.

        Returns:
            ``(N, B, n)`` holding ``sigma * cur @ E_i`` for each member, i.e.
            exactly the ``(sigma / sqrt(r)) (cur @ A_i) @ B_i.T`` term.
        """
        if self.low_rank:
            assert self.A is not None and self.B is not None
            if cur.ndim == 2:
                proj = np.einsum("bi,nir->nbr", cur, self.A, optimize=True)
            else:
                proj = np.einsum("nbi,nir->nbr", cur, self.A, optimize=True)
            return np.einsum("nbr,njr->nbj", proj, self.B, optimize=True) * np.float32(sigma * self.inv_sqrt_r)
        assert self.E is not None
        if cur.ndim == 2:
            return np.einsum("bi,nij->nbj", cur, self.E, optimize=True) * np.float32(sigma)
        return np.einsum("nbi,nij->nbj", cur, self.E, optimize=True) * np.float32(sigma)

    def additive_offset(self, sigma: float) -> np.ndarray:
        """``(N, 1, n)`` perturbation for a bias, broadcast over the batch."""
        if self.is_bias:
            assert self.E is not None
            return self.E * np.float32(sigma)
        raise ValueError("additive_offset is only valid for bias-shaped tensors")

    def materialize_member(self, index: int) -> np.ndarray:
        """Dense perturbation ``E_i`` for one member (debugging / tests)."""
        if self.low_rank:
            assert self.A is not None and self.B is not None
            return (self.A[index] @ self.B[index].T * self.inv_sqrt_r).astype(np.float32)
        assert self.E is not None
        return self.E[index].copy()

    def update_direction(self, weights: np.ndarray) -> np.ndarray:
        """Return ``sum_i w_i E_i`` for this tensor.

        The low-rank case contracts the population axis into the rank axis and
        evaluates a single ``(m, N*r) @ (N*r, n)`` matmul -- the rank-1
        ``(diag(f) A).T @ B`` trick from the paper, generalised to any ``r``.
        """
        m, n = self.shape
        if self.low_rank:
            assert self.A is not None and self.B is not None
            scaled = self.A * weights[:, None, None].astype(np.float32)
            left = np.ascontiguousarray(scaled.transpose(1, 0, 2)).reshape(m, -1)
            right = np.ascontiguousarray(self.B.transpose(1, 0, 2)).reshape(n, -1)
            return (left @ right.T) * np.float32(self.inv_sqrt_r)
        assert self.E is not None
        return np.einsum("n,nij->ij", weights.astype(np.float32), self.E, optimize=True)

    def storage_elements(self) -> int:
        """Floats held per member (the auxiliary storage the paper saves)."""
        m, n = self.shape
        return self.rank * (m + n) if self.low_rank else m * n


def _reference_tensors(net: MLPQNetwork) -> list[np.ndarray]:
    """Live parameter arrays in perturbation-plan order.

    Trunk ``(weight, bias)`` pairs first, then the dueling ``value`` and
    ``advantage`` heads as ``(weight, bias)`` pairs. Biases are included
    because leaving them out would freeze them at their initial values.
    """
    tensors: list[np.ndarray] = []
    for w, b in zip(net.weights, net.biases):
        tensors.extend([w, b])
    if net.dueling:
        assert net.value_w is not None and net.value_b is not None
        assert net.advantage_w is not None and net.advantage_b is not None
        tensors.extend([net.value_w, net.value_b, net.advantage_w, net.advantage_b])
    return tensors


def _population_forward(
    net: MLPQNetwork,
    perturbations: Sequence[TensorPerturbation],
    states: np.ndarray,
    sigmas: Sequence[float],
) -> np.ndarray:
    """Q-values of a whole perturbed population in one batched pass.

    Layers are evaluated sequentially because each member's layer-``l`` input
    depends on its own perturbed layer ``l-1``; only the first layer shares its
    input across members.

    Args:
        net: The unperturbed network, supplying the base weights.
        perturbations: Plan built by :func:`_reference_tensors` order.
        states: ``(B, input_dim)`` shared batch, or ``(N, B, input_dim)`` when
            each member has its own observation.
        sigmas: Per-tensor perturbation radii.

    Returns:
        ``(N, B, action_dim)`` Q-values.
    """
    cur = states
    idx = 0

    for w, b in zip(net.weights, net.biases):
        weight_pert = perturbations[idx]
        bias_pert = perturbations[idx + 1]
        base = cur @ w if cur.ndim == 2 else np.einsum("nbi,ij->nbj", cur, w, optimize=True)
        z = (
            base
            + b
            + weight_pert.forward_offset(cur, sigmas[idx])
            + bias_pert.additive_offset(sigmas[idx + 1])
        )
        idx += 2
        # Mirror the network's own activation pattern: a dueling trunk ReLUs its
        # final layer too (the value/advantage heads sit after it), whereas a
        # plain MLP keeps its output layer linear.
        is_last_trunk = idx >= 2 * len(net.weights)
        cur = np.maximum(0.0, z) if (net.dueling or not is_last_trunk) else z

    if not net.dueling:
        return cur

    features = cur if len(net.weights) > 0 else states
    value_pert = perturbations[idx]
    value_bias_pert = perturbations[idx + 1]
    adv_pert = perturbations[idx + 2]
    adv_bias_pert = perturbations[idx + 3]

    assert net.value_w is not None and net.value_b is not None
    assert net.advantage_w is not None and net.advantage_b is not None

    base_value = (
        features @ net.value_w if features.ndim == 2 else np.einsum("nbf,fo->nbo", features, net.value_w, optimize=True)
    )
    base_adv = (
        features @ net.advantage_w
        if features.ndim == 2
        else np.einsum("nbf,fo->nbo", features, net.advantage_w, optimize=True)
    )

    value = (
        base_value
        + net.value_b
        + value_pert.forward_offset(features, sigmas[idx])
        + value_bias_pert.additive_offset(sigmas[idx + 1])
    )
    advantage = (
        base_adv
        + net.advantage_b
        + adv_pert.forward_offset(features, sigmas[idx + 2])
        + adv_bias_pert.additive_offset(sigmas[idx + 3])
    )
    return value + (advantage - np.mean(advantage, axis=2, keepdims=True))


def member_q_values(
    net: MLPQNetwork,
    perturbations: Sequence[TensorPerturbation],
    states: np.ndarray,
    sigmas: Sequence[float],
) -> np.ndarray:
    """Q-values when each population member has its own state.

    Args:
        states: ``(N, input_dim)``, one observation per member.

    Returns:
        ``(N, action_dim)``.
    """
    if states.ndim != 2:
        raise ValueError("states must have shape (population, input_dim)")
    if states.shape[0] != perturbations[0].population:
        raise ValueError("states and perturbations disagree on the population size")
    q_values = _population_forward(net, perturbations, states[:, None, :], sigmas)
    return q_values[:, 0, :]


def rollout_population(
    net: MLPQNetwork,
    optimizer: "EggrollOptimizer",
    envs: Sequence[Any],
    seeds: Sequence[int],
    max_steps: int,
    use_legal_masks: bool = False,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Play one episode per perturbed member and return the total reward.

    Members act greedily on their own perturbed Q-values, which is the
    deterministic policy used at eval time. Finished members stay finished.
    The mean network is not modified.
    """
    population = len(envs)
    if population != optimizer.config.population:
        raise ValueError("env count must equal the EGGROLL population")
    if len(seeds) != population:
        raise ValueError("seeds must have one entry per population member")
    if optimizer.sigmas is None:
        raise RuntimeError("sample_population must run before a rollout")

    states = np.stack([env.reset(seed=int(seed)) for env, seed in zip(envs, seeds)]).astype(np.float32)
    returns = np.zeros(population, dtype=np.float64)
    alive = np.ones(population, dtype=bool)
    infos: list[dict[str, Any]] = [{} for _ in range(population)]
    for _ in range(int(max_steps)):
        if not bool(alive.any()):
            break
        q_values = member_q_values(net, optimizer._perturbations, states, optimizer.sigmas)
        actions = np.argmax(q_values, axis=1)
        if use_legal_masks:
            for index in range(population):
                if not alive[index]:
                    continue
                legal = list(envs[index].legal_actions())
                if legal:
                    actions[index] = legal[int(np.argmax(q_values[index, legal]))]
        for index, env in enumerate(envs):
            if not alive[index]:
                continue
            next_state, reward, done, info = env.step(int(actions[index]))
            returns[index] += float(reward)
            states[index] = np.asarray(next_state, dtype=np.float32)
            infos[index] = dict(info)
            alive[index] = not bool(done)
    return returns, infos


def _td_losses(
    q_values: np.ndarray,
    actions: np.ndarray,
    targets: np.ndarray,
    loss: str,
    huber_delta: float,
    sample_weights: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-sample absolute TD error and per-member loss.

    Args:
        q_values: ``(..., B, action_dim)``. Leading axes are population members
            and are averaged away, so the same helper scores one network and a
            whole population.
        actions: ``(B,)`` taken actions.
        targets: ``(B,)`` TD targets.
        loss: ``"mse"`` or ``"huber"``.
        huber_delta: Huber knee.
        sample_weights: Optional ``(B,)`` priorities (prioritised replay).

    Returns:
        ``(td_abs, mean_loss)``. ``td_abs`` keeps every leading axis; ``mean_loss``
        is reduced over the batch axis only, giving one value per member (or a
        scalar for an unbatched network).
    """
    actions_arr = np.asarray(actions, dtype=np.int64).reshape(-1)
    targets_arr = np.asarray(targets, dtype=np.float32).reshape(-1)
    batch = int(actions_arr.shape[0])
    if q_values.shape[-2] != batch:
        raise ValueError("q_values batch dimension does not match actions")

    rows = np.arange(batch)
    pred = q_values[..., rows, actions_arr]
    td_errors = pred - targets_arr
    td_abs = np.abs(td_errors)

    if loss == "huber":
        delta = float(huber_delta)
        quadratic = td_abs <= delta
        losses = np.where(quadratic, 0.5 * (td_errors**2), delta * (td_abs - 0.5 * delta))
    else:
        losses = 0.5 * (td_errors**2)

    if sample_weights is None:
        return td_abs.astype(np.float32), np.mean(losses, axis=-1)

    weights = np.asarray(sample_weights, dtype=np.float32).reshape(-1)
    if weights.shape[0] != batch:
        raise ValueError("sample_weights length must match batch size")
    total = float(np.sum(weights))
    if total <= 0.0:
        return td_abs.astype(np.float32), np.mean(losses, axis=-1)
    return td_abs.astype(np.float32), (losses * weights).sum(axis=-1) / total


class _TensorAdam:
    """Adam over the network's parameter tensors, in perturbation-plan order.

    The EGGROLL estimate is a noisy gradient. Adam's second moment rescales
    each coordinate, so a small but repeated TD signal still produces a step
    of size ``lr`` instead of vanishing with the raw average.
    """

    def __init__(self, beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8) -> None:
        self.beta1 = float(beta1)
        self.beta2 = float(beta2)
        self.eps = float(eps)
        self.t = 0
        self._m: list[np.ndarray] = []
        self._v: list[np.ndarray] = []
        self._ids: tuple[int, ...] | None = None

    def apply(self, tensors: Sequence[np.ndarray], grads: Sequence[np.ndarray], lr: float) -> None:
        ids = tuple(id(tensor) for tensor in tensors)
        if ids != self._ids or len(self._m) != len(tensors):
            self._m = [np.zeros_like(tensor) for tensor in tensors]
            self._v = [np.zeros_like(tensor) for tensor in tensors]
            self.t = 0
            self._ids = ids
        self.t += 1
        bias1 = 1.0 - self.beta1**self.t
        bias2 = 1.0 - self.beta2**self.t
        step = np.float32(lr)
        for index, (tensor, grad) in enumerate(zip(tensors, grads)):
            g = np.asarray(grad, dtype=tensor.dtype)
            self._m[index] = self.beta1 * self._m[index] + np.float32(1.0 - self.beta1) * g
            self._v[index] = self.beta2 * self._v[index] + np.float32(1.0 - self.beta2) * (g * g)
            m_hat = self._m[index] / np.float32(bias1)
            v_hat = self._v[index] / np.float32(bias2)
            tensor -= (step * m_hat / (np.sqrt(v_hat) + np.float32(self.eps))).astype(tensor.dtype, copy=False)


class EggrollOptimizer:
    """Gradient-free EGGROLL optimizer for :class:`MLPQNetwork`.

    Drop-in for :class:`~src.network.AdamOptimizer` at the DQN update site:
    :meth:`step` takes the same replay minibatch and returns the same
    ``(loss, mean_td_abs, td_abs_vector)`` triple that
    ``MLPQNetwork.train_batch`` returns with ``return_td_errors=True``.

    ``update_rule="adam"`` sends the search direction through :class:`_TensorAdam`.
    ``update_rule="mean"`` applies ``lr * mean_i(w_i E_i)`` directly.
    """

    def __init__(self, config: EggrollConfig | None = None, **kwargs: Any) -> None:
        if config is None:
            config = EggrollConfig(**kwargs)
        elif kwargs:
            raise TypeError("pass either a config or keyword overrides, not both")
        config.validate()
        self.config = config
        self.rng = np.random.default_rng(config.seed)
        self.steps = 0
        self._perturbations: list[TensorPerturbation] = []
        self._signature: tuple[Any, ...] | None = None
        self.sigmas: list[float] | None = None
        self._fallback_scale = 1.0
        self._adam = _TensorAdam()

    # -- parameter plan -----------------------------------------------------

    def _ensure_plan(self, net: MLPQNetwork) -> None:
        """Build or rebuild the perturbation plan if the network changed."""
        tensors = _reference_tensors(net)
        signature = (tuple(t.shape for t in tensors), tuple(id(t) for t in tensors))
        if self._signature == signature:
            return
        self._perturbations = [TensorPerturbation(param, self.config.rank) for param in tensors]
        for perturbation in self._perturbations:
            perturbation.configure(self.config.population, self.rng)
        self._signature = signature

    def _weight_scale(self) -> float:
        """RMS over the weight-matrix entries, used as a sigma floor.

        Biases initialise to exactly zero, so ``std(param)`` would give them a
        zero relative sigma and freeze them out of the search entirely.
        """
        squares = [float(np.mean(p.param * p.param)) for p in self._perturbations if not p.is_bias]
        if not squares:
            return 1.0
        return float(np.sqrt(np.mean(squares)))

    # -- schedules ----------------------------------------------------------

    def _progress(self) -> float:
        total = self.config.total_steps
        if not total:
            return 0.0
        return min(1.0, self.steps / float(total))

    def current_lr(self) -> float:
        """Learning rate for the current step, honouring the schedule."""
        if self.config.lr_schedule == "constant":
            return float(self.config.lr)
        lo, hi = float(self.config.lr_final), float(self.config.lr)
        return float(lo + 0.5 * (hi - lo) * (1.0 + np.cos(np.pi * self._progress())))

    def current_sigma(self) -> float:
        """Sigma for the current step, optionally annealed to ``sigma_final``."""
        if self.config.sigma_final is None:
            return float(self.config.sigma)
        hi, lo = float(self.config.sigma), float(self.config.sigma_final)
        return float(lo + 0.5 * (hi - lo) * (1.0 + np.cos(np.pi * self._progress())))

    # -- update -------------------------------------------------------------

    def _apply_update(
        self,
        perturbations: Sequence[TensorPerturbation],
        tensors: Sequence[np.ndarray],
        weights: np.ndarray,
        lr: float,
        fitness_scale: float,
        fallback_scale: float,
    ) -> None:
        """Apply ``W += lr * mean_i(w_i E_i)``, optionally capped.

        ``update_direction`` returns the sum. Dividing by the population is
        the ``1/N`` in Algorithm 1; without it the step grows with ``N`` and
        no longer matches the configured ``lr``.

        ``fitness_scale`` puts units back on standardized weights. Those
        weights have std 1 even when every member has almost the same loss,
        and a later rescale up to ``lr * ||W||`` then walks the parameters
        by a fixed fraction on pure noise.
        """
        population = max(int(weights.shape[0]), 1)
        coeff = float(fitness_scale) / float(population)
        if coeff == 0.0 or lr == 0.0:
            return
        cap_relative = self.config.normalize_layer_updates
        for perturbation, tensor in zip(perturbations, tensors):
            direction = perturbation.update_direction(weights) * np.float32(coeff)
            direction_norm = float(np.linalg.norm(direction))
            if direction_norm <= 0.0:
                continue
            scale = float(lr)
            if cap_relative:
                reference = float(np.linalg.norm(tensor))
                if reference <= 1e-12:
                    # Biases initialise at 0, so ``lr * ||bias||`` would either
                    # skip the cap or, if we fell through to the raw sum, take
                    # one huge step. Use the same weight scale that relative
                    # sigma uses, spread across this tensor's entries.
                    reference = float(fallback_scale) * np.sqrt(float(tensor.size))
                limit = float(lr) * reference
                if limit >= 0.0 and scale * direction_norm > limit:
                    scale = limit / direction_norm
            hard_cap = self.config.max_update_norm
            if hard_cap is not None and hard_cap > 0.0:
                scale = min(scale, float(hard_cap) / direction_norm)
            tensor += direction * np.float32(scale)

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
        """Run one EGGROLL update on ``net`` from a replay minibatch.

        Samples ``population`` perturbations, evaluates every perturbed network
        on the same minibatch, then applies ``W += lr * mean_i(w_i E_i)`` where
        ``w_i`` are the shaped fitnesses.

        Returns:
            ``(loss, mean_abs_td_error)``, plus the per-sample absolute TD error
            when ``return_td_errors`` is set. The reported loss is always that
            of the *unperturbed* network (measured after the update), so it is
            directly comparable with the Adam path and safe to feed back into
            prioritised replay.
        """
        self._ensure_plan(net)
        states_arr = MLPQNetwork._as_batch(states)
        actions_arr = np.asarray(actions, dtype=np.int64).reshape(-1)
        targets_arr = np.asarray(targets, dtype=np.float32).reshape(-1)
        weights_arr = None if sample_weights is None else np.asarray(sample_weights, dtype=np.float32).reshape(-1)

        relative_sigma = self.config.sigma_mode == "relative"
        sigma_now = self.current_sigma()
        fallback = self._weight_scale()
        sigmas = [p.sigma_for(relative_sigma, sigma_now, fallback) for p in self._perturbations]

        for perturbation in self._perturbations:
            perturbation.sample(self.rng, self.config.antithetic)

        q_population = _population_forward(net, self._perturbations, states_arr, sigmas)
        _, member_losses = _td_losses(q_population, actions_arr, targets_arr, loss, huber_delta, weights_arr)

        fitnesses = -np.asarray(member_losses, dtype=np.float64).reshape(-1)
        self._fallback_scale = fallback
        self.sigmas = sigmas
        self._step_from_fitness(net, fitnesses, restore_standardized_scale=True)
        self.steps += 1

        base_q, _ = net.forward(states_arr)
        td_abs, base_losses = _td_losses(base_q, actions_arr, targets_arr, loss, huber_delta, weights_arr)
        base_loss = float(base_losses)
        td_mean = float(np.mean(td_abs))
        if return_td_errors:
            return base_loss, td_mean, np.asarray(td_abs, dtype=np.float32).reshape(-1)
        return base_loss, td_mean

    def sample_population(self, net: MLPQNetwork) -> None:
        """Draw the perturbations used for one generation of episode rollouts."""
        self._ensure_plan(net)
        relative_sigma = self.config.sigma_mode == "relative"
        sigma_now = self.current_sigma()
        self._fallback_scale = self._weight_scale()
        self.sigmas = [p.sigma_for(relative_sigma, sigma_now, self._fallback_scale) for p in self._perturbations]
        for perturbation in self._perturbations:
            perturbation.sample(self.rng, self.config.antithetic)

    def apply_fitness(self, net: MLPQNetwork, fitnesses: np.ndarray) -> None:
        """Update the mean network from episode returns.

        Standardized fitnesses stay unit-scale. Multiplying the return spread
        back in would make the step depend on the reward units of the game
        rather than on ``lr``.
        """
        if self.sigmas is None:
            raise RuntimeError("sample_population must run before apply_fitness")
        self._step_from_fitness(net, np.asarray(fitnesses, dtype=np.float64), restore_standardized_scale=False)
        self.steps += 1

    def _apply_adam(self, net: MLPQNetwork, shaped: np.ndarray, lr: float) -> None:
        """Descend on ``-sqrt(N) * mean_i(w_i E_i)`` with Adam.

        ``update_direction`` returns the sum. Multiplying by ``1/sqrt(N)`` is
        the reference implementation's ``sqrt(N) / N`` rescaling before optax.
        Standardized weights stay unit-scale; Adam, not the loss spread,
        chooses the step length.
        """
        if lr == 0.0 or float(np.max(np.abs(shaped))) == 0.0:
            return
        population = max(int(shaped.shape[0]), 1)
        coeff = np.float32(1.0 / np.sqrt(float(population)))
        tensors = _reference_tensors(net)
        grads = [-(perturbation.update_direction(shaped) * coeff) for perturbation in self._perturbations]
        self._adam.apply(tensors, grads, lr)

    def _step_from_fitness(self, net: MLPQNetwork, fitnesses: np.ndarray, restore_standardized_scale: bool) -> None:
        shaped = shape_fitnesses(fitnesses, self.config.fitness_shaping)
        if self.config.update_rule == "adam":
            self._apply_adam(net, shaped, self.current_lr())
            return
        fitness_scale = 1.0
        if restore_standardized_scale and self.config.fitness_shaping == "standardized":
            fitness_scale = float(np.std(np.asarray(fitnesses, dtype=np.float64)))
        self._apply_update(
            self._perturbations,
            _reference_tensors(net),
            shaped,
            self.current_lr(),
            fitness_scale,
            self._fallback_scale,
        )

    # -- introspection ------------------------------------------------------

    def memory_report(self, net: MLPQNetwork) -> dict[str, int]:
        """Compare EGGROLL's perturbation storage against full-rank ES.

        Returns element counts for one member and for the whole population,
        which is the ``O(mn) -> O(r(m+n))`` saving from the abstract.
        """
        self._ensure_plan(net)
        eggroll = sum(p.storage_elements() for p in self._perturbations)
        full_rank = sum(p.shape[0] * p.shape[1] for p in self._perturbations)
        population = int(self.config.population)
        return {
            "tensors": len(self._perturbations),
            "population": population,
            "rank": int(self.config.rank),
            "low_rank_tensors": sum(1 for p in self._perturbations if p.low_rank),
            "full_rank_tensors": sum(1 for p in self._perturbations if not p.low_rank),
            "eggroll_elements_per_member": int(eggroll),
            "full_rank_elements_per_member": int(full_rank),
            "eggroll_elements_total": int(eggroll * population),
            "full_rank_elements_total": int(full_rank * population),
        }


def eggroll_config_from_params(params: dict[str, Any]) -> EggrollConfig:
    """Build an :class:`EggrollConfig` from merged YAML/CLI params.

    Keys are prefixed ``eggroll_`` so nothing collides with the DQN settings
    that share names (``lr``, ``rank``, ...).
    """

    def get(key: str, default: Any) -> Any:
        return params.get(f"eggroll_{key}", default)

    defaults = EggrollConfig()
    sigma_final = get("sigma_final", None)
    max_update_norm = get("max_update_norm", None)
    total_steps = get("total_steps", None)
    return EggrollConfig(
        rank=int(get("rank", defaults.rank)),
        population=int(get("population", defaults.population)),
        sigma=float(get("sigma", defaults.sigma)),
        lr=float(get("lr", defaults.lr)),
        fitness_shaping=str(get("fitness_shaping", defaults.fitness_shaping)),
        antithetic=bool(get("antithetic", defaults.antithetic)),
        sigma_mode=str(get("sigma_mode", defaults.sigma_mode)),
        lr_schedule=str(get("lr_schedule", defaults.lr_schedule)),
        lr_final=float(get("lr_final", defaults.lr_final)),
        sigma_final=None if sigma_final is None else float(sigma_final),
        normalize_layer_updates=bool(get("normalize_layer_updates", defaults.normalize_layer_updates)),
        max_update_norm=None if max_update_norm is None else float(max_update_norm),
        update_rule=str(get("update_rule", defaults.update_rule)),
        total_steps=None if total_steps is None else int(total_steps),
        seed=int(params.get("seed", 42)),
    )
