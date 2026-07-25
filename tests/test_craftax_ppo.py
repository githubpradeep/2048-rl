from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.games.craftax_classic import CraftaxClassicConfig, NUM_ACTIONS
from src.network import AdamOptimizer
from src.plugins.policy_gradient.craftax_ppo import CraftaxVecEnv
from src.plugins.policy_gradient.ppo_network import PPOPolicyNetwork, compute_gae


class TestCraftaxPPO(unittest.TestCase):
    def test_gae_terminal_zeros_bootstrap(self) -> None:
        rewards = np.array([[1.0], [1.0]], dtype=np.float32)
        values = np.array([[0.5], [0.5]], dtype=np.float32)
        dones = np.array([[0.0], [1.0]], dtype=np.float32)
        next_values = np.array([9.0], dtype=np.float32)
        adv, ret = compute_gae(rewards, values, dones, next_values, gamma=0.99, gae_lambda=1.0)
        # Last step done => no bootstrap from next_values.
        self.assertAlmostEqual(float(ret[1, 0]), 1.0, places=5)
        self.assertAlmostEqual(float(ret[0, 0]), 1.0 + 0.99 * 1.0, places=5)

    def test_vec_env_shapes_and_masks(self) -> None:
        cfg = CraftaxClassicConfig(max_timesteps=50, reward_shaping=False)
        vec = CraftaxVecEnv(4, cfg, seed=0)
        states, masks = vec.reset(seed=0)
        self.assertEqual(states.shape, (4, 1345))
        self.assertEqual(masks.shape, (4, NUM_ACTIONS))
        self.assertTrue(np.all(masks.sum(axis=1) >= 4))
        actions = np.zeros((4,), dtype=np.int64)
        for i in range(4):
            legal = np.flatnonzero(masks[i] > 0.5)
            actions[i] = int(legal[0])
        next_states, rewards, dones, next_masks, infos = vec.step(actions)
        self.assertEqual(next_states.shape, (4, 1345))
        self.assertEqual(rewards.shape, (4,))
        self.assertEqual(dones.shape, (4,))
        self.assertEqual(next_masks.shape, (4, NUM_ACTIONS))
        self.assertEqual(len(infos), 4)

    def test_ppo_update_smoke(self) -> None:
        rng = np.random.default_rng(0)
        model = PPOPolicyNetwork(input_dim=1345, output_dim=NUM_ACTIONS, hidden_sizes=(32, 32), seed=0)
        opt = AdamOptimizer(lr=1e-3, max_grad_norm=1.0)
        b = 64
        states = rng.standard_normal((b, 1345), dtype=np.float32)
        masks = np.zeros((b, NUM_ACTIONS), dtype=np.float32)
        masks[:, 1:5] = 1.0  # movement only
        actions, logp, values = model.sample_actions(states, masks, rng)
        adv = rng.standard_normal((b,), dtype=np.float32)
        ret = values + adv
        metrics = model.ppo_train_batch(
            states=states,
            actions=actions,
            old_log_probs=logp,
            advantages=adv,
            returns=ret,
            legal_masks=masks,
            optimizer=opt,
            clip_eps=0.2,
        )
        self.assertIn("loss", metrics)
        self.assertIn("clip_frac", metrics)

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "m.json"
            model.save(path)
            loaded = PPOPolicyNetwork.load(path)
            self.assertEqual(loaded.input_dim, 1345)
            self.assertEqual(loaded.output_dim, NUM_ACTIONS)


if __name__ == "__main__":
    unittest.main()
