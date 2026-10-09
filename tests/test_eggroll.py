import unittest

import numpy as np

from src.eggroll import (
    EggrollConfig,
    EggrollOptimizer,
    TensorPerturbation,
    _population_forward,
    _reference_tensors,
    _td_losses,
    member_q_values,
    rollout_population,
    shape_fitnesses,
)
from src.network import MLPQNetwork


def _states_and_targets(count: int = 12, input_dim: int = 5, seed: int = 3, n_actions: int = 4):
    rng = np.random.default_rng(seed)
    states = rng.standard_normal((count, input_dim)).astype(np.float32)
    actions = rng.integers(0, n_actions, size=count)
    targets = rng.standard_normal(count).astype(np.float32)
    return states, actions, targets


def _naive_population_q(net, perturbations, states, sigmas):
    """Reference implementation: materialise E_i and run the plain forward.

    Deliberately the slow, obvious thing -- if the batched low-rank path in
    ``_population_forward`` is wrong, this catches it. Walks the same
    ``(weight, bias)`` plan order the optimizer uses.
    """
    population = perturbations[0].population
    tensors = _reference_tensors(net)
    saved = [t.copy() for t in tensors]

    out = []
    try:
        for index in range(population):
            for slot, tensor in enumerate(tensors):
                tensors[slot][...] = saved[slot] + sigmas[slot] * perturbations[slot].materialize_member(index)
            q_values, _ = net.forward(states)
            out.append(q_values)
    finally:
        for slot, tensor in enumerate(tensors):
            tensor[...] = saved[slot]

    return np.stack(out, axis=0)


class TestFitnessShaping(unittest.TestCase):
    def test_non_raw_modes_are_zero_mean(self) -> None:
        fitnesses = np.array([0.1, 5.0, -3.0, 2.5, 2.4, 9.0])
        for mode in ("centered", "standardized", "rank"):
            weights = shape_fitnesses(fitnesses, mode)
            self.assertAlmostEqual(float(np.mean(weights)), 0.0, places=10, msg=mode)

    def test_standardized_has_unit_variance(self) -> None:
        fitnesses = np.array([0.1, 5.0, -3.0, 2.5, 2.4, 9.0])
        weights = shape_fitnesses(fitnesses, "standardized")
        self.assertAlmostEqual(float(np.std(weights)), 1.0, places=10)

    def test_rank_preserves_ordering(self) -> None:
        fitnesses = np.array([0.1, 5.0, -3.0, 2.5])
        weights = shape_fitnesses(fitnesses, "rank")
        self.assertLess(weights[2], weights[0])
        self.assertLess(weights[0], weights[3])
        self.assertLess(weights[3], weights[1])

    def test_constant_fitness_yields_zero_update(self) -> None:
        weights = shape_fitnesses(np.full(6, 2.5), "standardized")
        self.assertTrue(np.allclose(weights, 0.0))

    def test_raw_passes_through(self) -> None:
        fitnesses = np.array([1.0, 2.0, 3.0])
        self.assertTrue(np.allclose(shape_fitnesses(fitnesses, "raw"), fitnesses))


class TestPerturbationSampling(unittest.TestCase):
    def test_low_rank_factors_have_expected_shape(self) -> None:
        perturbation = TensorPerturbation(np.zeros((8, 6), dtype=np.float32), rank=2, )
        perturbation.configure(5, np.random.default_rng(0))
        self.assertTrue(perturbation.low_rank)
        self.assertEqual(perturbation.A.shape, (5, 8, 2))
        self.assertEqual(perturbation.B.shape, (5, 6, 2))
        self.assertIsNone(perturbation.E)

    def test_materialized_member_has_rank_at_most_r(self) -> None:
        perturbation = TensorPerturbation(np.zeros((9, 7), dtype=np.float32), rank=2, )
        perturbation.configure(4, np.random.default_rng(1))
        perturbation.sample(np.random.default_rng(1), antithetic=False)
        member = perturbation.materialize_member(0)
        self.assertEqual(member.shape, (9, 7))
        self.assertLessEqual(int(np.linalg.matrix_rank(member, tol=1e-4)), 2)

    def test_antithetic_second_half_is_negated(self) -> None:
        perturbation = TensorPerturbation(np.zeros((6, 5), dtype=np.float32), rank=1, )
        perturbation.configure(4, np.random.default_rng(2))
        perturbation.sample(np.random.default_rng(2), antithetic=True)
        for index in range(2):
            self.assertTrue(np.allclose(perturbation.materialize_member(index), -perturbation.materialize_member(index + 2)))

    def test_row_vector_falls_back_to_full_rank(self) -> None:
        perturbation = TensorPerturbation(np.zeros((1, 12), dtype=np.float32), rank=1, )
        perturbation.configure(4, np.random.default_rng(3))
        self.assertFalse(perturbation.low_rank)

    def test_rank_at_or_above_min_dim_falls_back_to_full_rank(self) -> None:
        perturbation = TensorPerturbation(np.zeros((5, 3), dtype=np.float32), rank=3, )
        perturbation.configure(4, np.random.default_rng(4))
        self.assertFalse(perturbation.low_rank)

    def test_update_direction_matches_dense_reference(self) -> None:
        rank, population = 2, 6
        perturbation = TensorPerturbation(np.zeros((7, 4), dtype=np.float32), rank=rank)
        perturbation.configure(population, np.random.default_rng(5))
        perturbation.sample(np.random.default_rng(5), antithetic=False)
        weights = np.linspace(-0.5, 0.5, population)

        dense = np.zeros((7, 4), dtype=np.float32)
        for index in range(population):
            dense += weights[index] * perturbation.materialize_member(index)

        self.assertTrue(np.allclose(perturbation.update_direction(weights), dense, atol=1e-4))

    def test_update_reaches_full_rank_when_population_exceeds_dimension(self) -> None:
        perturbation = TensorPerturbation(np.zeros((4, 4), dtype=np.float32), rank=1, )
        perturbation.configure(16, np.random.default_rng(6))
        perturbation.sample(np.random.default_rng(6), antithetic=True)
        weights = shape_fitnesses(np.random.default_rng(7).standard_normal(16), "standardized")
        direction = perturbation.update_direction(weights)
        self.assertEqual(int(np.linalg.matrix_rank(direction, tol=1e-4)), 4)


class TestPopulationForward(unittest.TestCase):
    def test_matches_naive_forward_non_dueling(self) -> None:
        net = MLPQNetwork(input_dim=5, output_dim=4, hidden_sizes=(9, 7), seed=11)
        states, _, _ = _states_and_targets(count=10, input_dim=5)
        optimizer = EggrollOptimizer(EggrollConfig(rank=2, population=6, seed=12, antithetic=False))
        optimizer._ensure_plan(net)

        sigmas = [p.sigma_for(True, 0.3, 0.1) for p in optimizer._perturbations]
        for perturbation in optimizer._perturbations:
            perturbation.sample(optimizer.rng, antithetic=False)

        fast = _population_forward(net, optimizer._perturbations, states, sigmas)
        slow = _naive_population_q(net, optimizer._perturbations, states, sigmas)

        self.assertEqual(fast.shape, (6, 10, 4))
        self.assertTrue(np.allclose(fast, slow, atol=1e-4), f"max diff {np.abs(fast - slow).max()}")

    def test_matches_naive_forward_dueling(self) -> None:
        net = MLPQNetwork(input_dim=6, output_dim=5, hidden_sizes=(8, 6), seed=13, dueling=True)
        states, _, _ = _states_and_targets(count=9, input_dim=6, seed=14)
        optimizer = EggrollOptimizer(EggrollConfig(rank=1, population=4, seed=15, antithetic=False))
        optimizer._ensure_plan(net)

        sigmas = [p.sigma_for(False, 0.05, 0.1) for p in optimizer._perturbations]
        for perturbation in optimizer._perturbations:
            perturbation.sample(optimizer.rng, antithetic=False)

        fast = _population_forward(net, optimizer._perturbations, states, sigmas)
        slow = _naive_population_q(net, optimizer._perturbations, states, sigmas)

        self.assertEqual(fast.shape, (4, 9, 5))
        self.assertTrue(np.allclose(fast, slow, atol=1e-4), f"max diff {np.abs(fast - slow).max()}")

    def test_zero_sigma_reduces_to_base_network(self) -> None:
        net = MLPQNetwork(input_dim=4, output_dim=3, hidden_sizes=(6,), seed=16)
        states, _, _ = _states_and_targets(count=8, input_dim=4, seed=17)
        optimizer = EggrollOptimizer(EggrollConfig(rank=1, population=3, seed=18, antithetic=False))
        optimizer._ensure_plan(net)
        for perturbation in optimizer._perturbations:
            perturbation.sample(optimizer.rng, antithetic=False)

        zeros = [0.0] * len(optimizer._perturbations)
        population_q = _population_forward(net, optimizer._perturbations, states, zeros)
        base_q, _ = net.forward(states)
        self.assertTrue(np.allclose(population_q, np.broadcast_to(base_q, population_q.shape), atol=1e-5))


class TestTdLosses(unittest.TestCase):
    def test_matches_manual_mse(self) -> None:
        states, actions, targets = _states_and_targets(count=8, input_dim=4, seed=19)
        net = MLPQNetwork(input_dim=4, output_dim=4, hidden_sizes=(6,), seed=20)
        q_values, _ = net.forward(states)

        td_abs, loss = _td_losses(q_values, actions, targets, "mse", 1.0, None)
        rows = np.arange(len(actions))
        td = q_values[rows, actions] - targets
        self.assertEqual(float(loss), float(np.mean(0.5 * td**2)))
        self.assertTrue(np.allclose(td_abs, np.abs(td), atol=1e-6))

    def test_batched_and_unbatched_losses_agree(self) -> None:
        states, actions, targets = _states_and_targets(count=8, input_dim=4, seed=21)
        net = MLPQNetwork(input_dim=4, output_dim=4, hidden_sizes=(6,), seed=22)
        population = 3
        optimizer = EggrollOptimizer(EggrollConfig(population=population, seed=23, antithetic=False))
        optimizer._ensure_plan(net)
        sigmas = [p.sigma_for(True, 0.3, 0.1) for p in optimizer._perturbations]
        for perturbation in optimizer._perturbations:
            perturbation.sample(optimizer.rng, antithetic=False)

        q_pop = _population_forward(net, optimizer._perturbations, states, sigmas)
        td_abs, losses = _td_losses(q_pop, actions, targets, "mse", 1.0, None)
        self.assertEqual(losses.shape, (population,))
        self.assertEqual(td_abs.shape, (population, len(actions)))

        # Batched per-member losses must equal scoring each member separately.
        for member in range(population):
            _, member_loss = _td_losses(q_pop[member], actions, targets, "mse", 1.0, None)
            self.assertAlmostEqual(float(member_loss), float(losses[member]), places=6)

    def test_huber_branching(self) -> None:
        q_values = np.array([[3.0, 0.0]], dtype=np.float32)
        actions = np.array([0])
        _, mse_loss = _td_losses(q_values, actions, np.array([1.0]), "mse", 1.0, None)
        _, huber_loss = _td_losses(q_values, actions, np.array([1.0]), "huber", 1.0, None)
        # |td| = 2 > delta = 1, so huber switches to the linear branch:
        # delta * (|td| - 0.5 * delta) = 1.5 instead of 0.5 * 2^2 = 2.0.
        self.assertAlmostEqual(float(mse_loss), 2.0, places=5)
        self.assertAlmostEqual(float(huber_loss), 1.5, places=5)

    def test_sample_weights_rescale(self) -> None:
        q_values = np.array([[2.0], [0.0]], dtype=np.float32)
        actions = np.array([0, 0])
        targets = np.array([0.0, 0.0])
        _, weighted = _td_losses(q_values, actions, targets, "mse", 1.0, np.array([3.0, 1.0]))
        self.assertAlmostEqual(float(weighted), (3.0 * 2.0 + 1.0 * 0.0) / 4.0, places=5)

    def test_sample_weights_rejected_on_length_mismatch(self) -> None:
        q_values = np.array([[2.0], [0.0]], dtype=np.float32)
        with self.assertRaises(ValueError):
            _td_losses(q_values, np.array([0, 0]), np.array([0.0, 0.0]), "mse", 1.0, np.array([1.0]))


class TestEggrollOptimizer(unittest.TestCase):
    def test_reduces_loss_on_linear_problem(self) -> None:
        net = MLPQNetwork(input_dim=4, output_dim=2, hidden_sizes=(16, 16), seed=21)
        states, actions, targets = _states_and_targets(count=32, input_dim=4, seed=22, n_actions=2)
        optimizer = EggrollOptimizer(EggrollConfig(seed=23))

        def mse() -> float:
            q_values, _ = net.forward(states)
            rows = np.arange(len(actions))
            return float(np.mean(0.5 * (q_values[rows, actions] - targets) ** 2))

        before = mse()
        for _ in range(400):
            optimizer.step(net, states, actions, targets, loss="mse")
        after = mse()

        # The step is the fitness mean, not a forced fraction of ||W||, so this
        # is a steady descent rather than a few giant updates.
        self.assertLess(after, 0.75 * before, f"{before:.5f} -> {after:.5f}")

    def test_larger_population_reduces_loss_at_least_as_much(self) -> None:
        """ES variance falls as 1/N, so a bigger population should not do worse."""
        results = {}
        for population in (16, 64):
            net = MLPQNetwork(input_dim=4, output_dim=2, hidden_sizes=(16, 16), seed=21)
            states, actions, targets = _states_and_targets(count=32, input_dim=4, seed=22, n_actions=2)
            optimizer = EggrollOptimizer(EggrollConfig(population=population, seed=23))

            def mse(net=net) -> float:
                q_values, _ = net.forward(states)
                rows = np.arange(len(actions))
                return float(np.mean(0.5 * (q_values[rows, actions] - targets) ** 2))

            before = mse()
            for _ in range(120):
                optimizer.step(net, states, actions, targets, loss="mse")
            results[population] = mse() / before

        self.assertLess(results[64], results[16], f"pop16={results[16]:.4f} pop64={results[64]:.4f}")

    def test_reduces_loss_dueling(self) -> None:
        net = MLPQNetwork(input_dim=4, output_dim=3, hidden_sizes=(16,), seed=24, dueling=True)
        states, actions, targets = _states_and_targets(count=32, input_dim=4, seed=25, n_actions=3)
        optimizer = EggrollOptimizer(EggrollConfig(seed=26))

        def mse() -> float:
            q_values, _ = net.forward(states)
            rows = np.arange(len(actions))
            return float(np.mean(0.5 * (q_values[rows, actions] - targets) ** 2))

        before = mse()
        for _ in range(400):
            optimizer.step(net, states, actions, targets, loss="mse")
        self.assertLess(mse(), 0.75 * before)

    def test_huber_loss_path_runs(self) -> None:
        net = MLPQNetwork(input_dim=4, output_dim=2, hidden_sizes=(8,), seed=27)
        states, actions, targets = _states_and_targets(count=16, input_dim=4, seed=28, n_actions=2)
        optimizer = EggrollOptimizer(EggrollConfig(population=8, seed=29))
        loss, td_mean, td_vec = optimizer.step(
            net, states, actions, targets, loss="huber", huber_delta=1.0, return_td_errors=True
        )
        self.assertGreater(loss, 0.0)
        self.assertGreaterEqual(td_mean, 0.0)
        self.assertEqual(td_vec.shape, (16,))

    def test_sample_weights_accepted(self) -> None:
        net = MLPQNetwork(input_dim=4, output_dim=2, hidden_sizes=(8,), seed=30)
        states, actions, targets = _states_and_targets(count=16, input_dim=4, seed=31, n_actions=2)
        optimizer = EggrollOptimizer(EggrollConfig(population=8, seed=32))
        weights = np.linspace(0.1, 1.0, len(actions)).astype(np.float32)
        loss, _ = optimizer.step(net, states, actions, targets, sample_weights=weights)
        self.assertGreater(loss, 0.0)

    def test_reported_loss_is_unperturbed_network_loss(self) -> None:
        net = MLPQNetwork(input_dim=4, output_dim=2, hidden_sizes=(8,), seed=33)
        states, actions, targets = _states_and_targets(count=16, input_dim=4, seed=34, n_actions=2)
        optimizer = EggrollOptimizer(EggrollConfig(population=8, seed=35))

        reported, _, td_vec = optimizer.step(net, states, actions, targets, return_td_errors=True)

        # `step` mutates net before measuring, so verify against the post-step
        # network: the logged numbers must come from the real weights, not from
        # any perturbed population member.
        q_after, _ = net.forward(states)
        _, expected_loss = _td_losses(q_after, actions, targets, "mse", 1.0, None)
        rows = np.arange(len(actions))
        expected_td = np.abs(q_after[rows, actions] - targets)

        self.assertAlmostEqual(reported, float(expected_loss), places=6)
        self.assertTrue(np.allclose(td_vec, expected_td, atol=1e-5))

    def test_biases_are_perturbed_and_move(self) -> None:
        net = MLPQNetwork(input_dim=4, output_dim=2, hidden_sizes=(8,), seed=36)
        states, actions, targets = _states_and_targets(count=16, input_dim=4, seed=37, n_actions=2)
        optimizer = EggrollOptimizer(EggrollConfig(population=16, lr=0.1, seed=38))

        before = [b.copy() for b in net.biases]
        optimizer.step(net, states, actions, targets)
        for old, new in zip(before, net.biases):
            self.assertGreater(float(np.linalg.norm(new - old)), 0.0)

    def test_biases_in_plan_and_full_rank(self) -> None:
        net = MLPQNetwork(input_dim=4, output_dim=2, hidden_sizes=(8,), seed=39)
        optimizer = EggrollOptimizer(EggrollConfig(rank=1, population=4, seed=40))
        optimizer._ensure_plan(net)
        tensors = _reference_tensors(net)
        self.assertEqual(len(optimizer._perturbations), len(tensors))
        # Trunk (w, b) pairs, then the output layer's own pair.
        self.assertEqual(len(tensors), 2 * len(net.weights))
        for bias in net.biases:
            perturbation = next(p for p in optimizer._perturbations if p.param is bias)
            self.assertFalse(perturbation.low_rank)

    def test_checkpoint_roundtrip_still_works_after_egroll_updates(self) -> None:
        import tempfile
        from pathlib import Path

        net = MLPQNetwork(input_dim=4, output_dim=2, hidden_sizes=(8,), seed=36)
        states, actions, targets = _states_and_targets(count=16, input_dim=4, seed=37, n_actions=2)
        optimizer = EggrollOptimizer(EggrollConfig(population=8, seed=38))
        optimizer.step(net, states, actions, targets)

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "eggroll_model.json"
            net.save(path)
            loaded = MLPQNetwork.load(path)

        for a, b in zip(net.predict_one(states[0]), loaded.predict_one(states[0])):
            self.assertAlmostEqual(a, b, places=5)

    def test_cosine_schedule_decays(self) -> None:
        optimizer = EggrollOptimizer(
            EggrollConfig(population=8, lr=0.1, lr_schedule="cosine", lr_final=0.0, total_steps=100, seed=39)
        )
        start = optimizer.current_lr()
        for _ in range(50):
            optimizer.steps += 1
        middle = optimizer.current_lr()
        optimizer.steps = 100
        end = optimizer.current_lr()
        self.assertAlmostEqual(start, 0.1, places=6)
        self.assertLess(middle, start)
        self.assertAlmostEqual(end, 0.0, places=6)

    def test_sigma_anneal(self) -> None:
        optimizer = EggrollOptimizer(
            EggrollConfig(population=8, sigma=0.5, sigma_final=0.1, total_steps=100, seed=40)
        )
        self.assertAlmostEqual(optimizer.current_sigma(), 0.5, places=6)
        optimizer.steps = 100
        self.assertAlmostEqual(optimizer.current_sigma(), 0.1, places=6)

    def test_normalize_layer_updates_caps_relative_step(self) -> None:
        net = MLPQNetwork(input_dim=32, output_dim=4, hidden_sizes=(64,), seed=41)
        states, actions, targets = _states_and_targets(count=32, input_dim=32, seed=42)
        lr = 0.05
        optimizer = EggrollOptimizer(
            EggrollConfig(population=32, lr=lr, normalize_layer_updates=True, seed=43)
        )

        before = [w.copy() for w in net.weights]
        optimizer.step(net, states, actions, targets)
        for old, new in zip(before, net.weights):
            base_norm = float(np.linalg.norm(old))
            step_norm = float(np.linalg.norm(new - old))
            self.assertGreater(step_norm, 0.0)
            self.assertLessEqual(step_norm / base_norm, lr * 1.001)

    def test_unnormalized_step_is_the_population_mean(self) -> None:
        net = MLPQNetwork(input_dim=4, output_dim=2, hidden_sizes=(8,), seed=50)
        states, actions, targets = _states_and_targets(count=16, input_dim=4, seed=51, n_actions=2)
        lr = 0.05
        optimizer = EggrollOptimizer(
            EggrollConfig(
                population=8,
                lr=lr,
                sigma=0.0,
                sigma_mode="absolute",
                fitness_shaping="raw",
                antithetic=False,
                normalize_layer_updates=False,
                seed=52,
            )
        )
        # sigma 0 makes every member identical, so build the direction from a
        # known weight vector instead of a live step.
        optimizer._ensure_plan(net)
        for perturbation in optimizer._perturbations:
            perturbation.sample(optimizer.rng, antithetic=False)
        weights = np.linspace(-1.0, 1.0, optimizer.config.population)
        before = [tensor.copy() for tensor in _reference_tensors(net)]
        optimizer._apply_update(optimizer._perturbations, _reference_tensors(net), weights, lr, 1.0, 1.0)
        for old, tensor, perturbation in zip(before, _reference_tensors(net), optimizer._perturbations):
            expected = old + lr * perturbation.update_direction(weights) / len(weights)
            self.assertTrue(np.allclose(tensor, expected, atol=1e-5))

    def test_tiny_fitness_spread_stays_well_under_the_cap(self) -> None:
        net = MLPQNetwork(input_dim=8, output_dim=2, hidden_sizes=(16,), seed=53)
        states, actions, targets = _states_and_targets(count=16, input_dim=8, seed=54, n_actions=2)
        lr = 0.05
        optimizer = EggrollOptimizer(
            EggrollConfig(
                population=16,
                lr=lr,
                sigma=1e-4,
                sigma_mode="absolute",
                normalize_layer_updates=True,
                seed=55,
            )
        )
        before = [w.copy() for w in net.weights]
        optimizer.step(net, states, actions, targets)
        for old, new in zip(before, net.weights):
            relative = float(np.linalg.norm(new - old) / np.linalg.norm(old))
            self.assertLess(relative, lr * 0.05)

    def test_max_update_norm_caps_step(self) -> None:
        net = MLPQNetwork(input_dim=32, output_dim=4, hidden_sizes=(64,), seed=44)
        states, actions, targets = _states_and_targets(count=32, input_dim=32, seed=45)
        cap = 1e-4
        optimizer = EggrollOptimizer(
            EggrollConfig(population=32, lr=0.5, max_update_norm=cap, seed=46)
        )

        before = [w.copy() for w in net.weights]
        optimizer.step(net, states, actions, targets)
        for old, new in zip(before, net.weights):
            self.assertLessEqual(float(np.linalg.norm(new - old)), cap * 1.001)

    def test_memory_report_shows_savings(self) -> None:
        net = MLPQNetwork(input_dim=128, output_dim=8, hidden_sizes=(256, 256), seed=47)
        report = EggrollOptimizer(EggrollConfig(rank=1, population=32, seed=48)).memory_report(net)
        self.assertLess(report["eggroll_elements_per_member"], report["full_rank_elements_per_member"])
        self.assertEqual(
            report["eggroll_elements_total"],
            report["eggroll_elements_per_member"] * report["population"],
        )
        self.assertGreater(report["low_rank_tensors"], 0)
        # Bias rows stay full-rank, so some tensors are unavoidable.
        self.assertGreater(report["full_rank_tensors"], 0)

    def test_odd_population_rejected_with_antithetic(self) -> None:
        with self.assertRaises(ValueError):
            EggrollConfig(population=7, antithetic=True).validate()

    def test_cosine_without_total_steps_rejected(self) -> None:
        with self.assertRaises(ValueError):
            EggrollConfig(population=8, lr_schedule="cosine").validate()

    def test_unknown_fitness_shaping_rejected(self) -> None:
        with self.assertRaises(ValueError):
            EggrollConfig(population=8, fitness_shaping="bogus").validate()


class TestEggrollAgainstAdamPath(unittest.TestCase):
    def test_eggroll_optimizer_is_not_adam_optimizer(self) -> None:
        from src.network import AdamOptimizer

        optimizer = EggrollOptimizer(EggrollConfig(population=8))
        self.assertNotIsInstance(optimizer, AdamOptimizer)
        self.assertIsInstance(EggrollOptimizer(EggrollConfig(population=8)), EggrollOptimizer)


class _OneStepBandit:
    """Reward +1 for action 1 and -1 for action 0, then end."""

    action_size = 2

    def reset(self, seed=None):
        del seed
        return np.zeros(4, dtype=np.float32)

    def step(self, action: int):
        reward = 1.0 if int(action) == 1 else -1.0
        return np.zeros(4, dtype=np.float32), reward, True, {"score": reward}


class TestReturnBasedSearch(unittest.TestCase):
    def test_per_member_states_match_materialized_forward(self) -> None:
        population = 6
        net = MLPQNetwork(input_dim=4, output_dim=3, hidden_sizes=(8,), seed=5, dueling=True)
        optimizer = EggrollOptimizer(EggrollConfig(rank=2, population=population, sigma=0.3, seed=6, antithetic=True))
        optimizer.sample_population(net)
        states = np.random.default_rng(7).standard_normal((population, 4)).astype(np.float32)
        batched = member_q_values(net, optimizer._perturbations, states, optimizer.sigmas)

        tensors = _reference_tensors(net)
        saved = [tensor.copy() for tensor in tensors]
        try:
            for index in range(population):
                for slot, tensor in enumerate(tensors):
                    tensor[...] = saved[slot] + optimizer.sigmas[slot] * optimizer._perturbations[slot].materialize_member(index)
                expected, _ = net.forward(states[index : index + 1])
                self.assertTrue(np.allclose(batched[index], expected[0], atol=1e-5))
        finally:
            for slot, tensor in enumerate(tensors):
                tensor[...] = saved[slot]

    def test_rollout_leaves_the_mean_network_unchanged(self) -> None:
        population = 4
        net = MLPQNetwork(input_dim=4, output_dim=2, hidden_sizes=(8,), seed=8)
        optimizer = EggrollOptimizer(EggrollConfig(population=population, seed=9, antithetic=True))
        optimizer.sample_population(net)
        before = [weight.copy() for weight in net.weights]
        returns, infos = rollout_population(
            net,
            optimizer,
            [_OneStepBandit() for _ in range(population)],
            [1, 2, 3, 4],
            max_steps=3,
        )
        self.assertEqual(returns.shape, (population,))
        self.assertEqual(len(infos), population)
        self.assertTrue(np.all(np.isin(returns, [-1.0, 1.0])))
        for old, new in zip(before, net.weights):
            self.assertTrue(np.array_equal(old, new))

    def test_bandit_return_improves(self) -> None:
        population = 16
        net = MLPQNetwork(input_dim=4, output_dim=2, hidden_sizes=(16,), seed=11)
        optimizer = EggrollOptimizer(
            EggrollConfig(
                rank=1,
                population=population,
                sigma=0.25,
                lr=0.08,
                seed=12,
                antithetic=True,
                fitness_shaping="standardized",
                normalize_layer_updates=True,
            )
        )

        def mean_return() -> float:
            optimizer.sample_population(net)
            returns, _ = rollout_population(
                net,
                optimizer,
                [_OneStepBandit() for _ in range(population)],
                list(range(population)),
                max_steps=1,
            )
            return float(np.mean(returns))

        before = mean_return()
        for _ in range(25):
            optimizer.sample_population(net)
            returns, _ = rollout_population(
                net,
                optimizer,
                [_OneStepBandit() for _ in range(population)],
                list(range(100, 100 + population)),
                max_steps=1,
            )
            optimizer.apply_fitness(net, returns)
        after = mean_return()
        self.assertGreater(after, before)
        self.assertGreater(after, 0.5)


class TestAdamUpdateRule(unittest.TestCase):
    def test_unknown_update_rule_rejected(self) -> None:
        with self.assertRaises(ValueError):
            EggrollConfig(population=8, update_rule="sgd").validate()

    def test_flat_population_leaves_weights_unchanged(self) -> None:
        net = MLPQNetwork(input_dim=4, output_dim=2, hidden_sizes=(8,), seed=4)
        optimizer = EggrollOptimizer(
            EggrollConfig(population=8, sigma=0.0, lr=1e-3, update_rule="adam", seed=5, antithetic=True)
        )
        states, actions, targets = _states_and_targets(count=16, input_dim=4, seed=6, n_actions=2)
        before = [weight.copy() for weight in net.weights]
        optimizer.step(net, states, actions, targets)
        for old, new in zip(before, net.weights):
            self.assertTrue(np.array_equal(old, new))

    def test_adam_rule_reduces_td_loss(self) -> None:
        net = MLPQNetwork(input_dim=4, output_dim=2, hidden_sizes=(16, 16), seed=21)
        states, actions, targets = _states_and_targets(count=32, input_dim=4, seed=22, n_actions=2)
        optimizer = EggrollOptimizer(
            EggrollConfig(
                population=32,
                sigma=0.05,
                lr=1e-3,
                rank=4,
                update_rule="adam",
                seed=23,
                normalize_layer_updates=False,
            )
        )

        def mse() -> float:
            q_values, _ = net.forward(states)
            rows = np.arange(len(actions))
            return float(np.mean(0.5 * (q_values[rows, actions] - targets) ** 2))

        before = mse()
        for _ in range(40):
            optimizer.step(net, states, actions, targets, loss="mse")
        after = mse()
        self.assertLess(after, 0.75 * before, f"{before:.5f} -> {after:.5f}")


if __name__ == "__main__":
    unittest.main()
