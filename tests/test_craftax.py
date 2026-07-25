from __future__ import annotations

import unittest

import numpy as np

from src.games.craftax_classic import (
    Achievement,
    Action,
    BlockType,
    CraftaxClassicConfig,
    CraftaxClassicEnv,
    CraftaxClassicGame,
    OBS_SIZE,
)
from src.games.craftax_classic.logic import craftax_step, is_game_over
from src.games.craftax_classic.state import EnvParams, EnvState, StaticEnvParams


class TestCraftaxClassic(unittest.TestCase):
    def test_obs_shape_and_dtype(self) -> None:
        env = CraftaxClassicEnv(seed=7)
        state = env.reset(seed=7)
        self.assertEqual(state.shape, (OBS_SIZE,))
        self.assertEqual(OBS_SIZE, 1345)
        self.assertEqual(state.dtype, np.float32)
        self.assertEqual(env.action_size, 17)
        legal = env.legal_actions()
        self.assertTrue(legal)
        self.assertTrue(set(legal).issubset(set(range(17))))
        # Movement is always available when awake.
        for move in (Action.LEFT, Action.RIGHT, Action.UP, Action.DOWN):
            self.assertIn(int(move), legal)
        self.assertNotIn(int(Action.NOOP), legal)

    def test_seed_determinism(self) -> None:
        env_a = CraftaxClassicEnv(seed=123)
        env_b = CraftaxClassicEnv(seed=123)
        sa = env_a.reset(seed=123)
        sb = env_b.reset(seed=123)
        self.assertTrue(np.allclose(sa, sb))
        self.assertEqual(
            env_a.game.state.player_position.tolist(),
            env_b.game.state.player_position.tolist(),
        )
        self.assertTrue(np.array_equal(env_a.game.state.map, env_b.game.state.map))

    def test_legal_actions_mask_futile_crafts(self) -> None:
        env = CraftaxClassicEnv(seed=7)
        env.reset(seed=7)
        legal = set(env.legal_actions())
        # Fresh spawn: no wood/stone/sapling -> place/make are illegal.
        for a in (
            Action.PLACE_TABLE,
            Action.PLACE_FURNACE,
            Action.PLACE_STONE,
            Action.PLACE_PLANT,
            Action.MAKE_WOOD_PICKAXE,
            Action.MAKE_STONE_PICKAXE,
            Action.MAKE_IRON_PICKAXE,
            Action.MAKE_WOOD_SWORD,
            Action.MAKE_STONE_SWORD,
            Action.MAKE_IRON_SWORD,
            Action.NOOP,
        ):
            self.assertNotIn(int(a), legal)

    def test_sleep_forces_noop(self) -> None:
        sp = StaticEnvParams()
        st = EnvState.create_default(sp, np.random.default_rng(0))
        st.is_sleeping = True
        st.player_energy = 5
        pr, pc = int(st.player_position[0]), int(st.player_position[1])
        st.map[pr - 1, pc] = BlockType.TREE
        st.player_direction = int(Action.UP)
        wood_before = st.inventory.wood
        st, _ = craftax_step(st, int(Action.DO), EnvParams(), sp, np.random.default_rng(1))
        self.assertEqual(st.inventory.wood, wood_before)

    def test_craft_wood_pickaxe(self) -> None:
        sp = StaticEnvParams()
        st = EnvState.create_default(sp, np.random.default_rng(0))
        pr, pc = int(st.player_position[0]), int(st.player_position[1])
        st.map[pr, pc + 1] = BlockType.CRAFTING_TABLE
        st.inventory.wood = 1
        st, reward = craftax_step(
            st, int(Action.MAKE_WOOD_PICKAXE), EnvParams(), sp, np.random.default_rng(2)
        )
        self.assertEqual(st.inventory.wood_pickaxe, 1)
        self.assertTrue(bool(st.achievements[Achievement.MAKE_WOOD_PICKAXE]))
        self.assertGreater(reward, 0.0)

    def test_max_timesteps_terminates(self) -> None:
        cfg = CraftaxClassicConfig(max_timesteps=5)
        game = CraftaxClassicGame(config=cfg, seed=1)
        game.reset(seed=1)
        done = False
        for _ in range(20):
            _, _, done = game.step(int(Action.NOOP))
            if done:
                break
        self.assertTrue(done)
        self.assertTrue(is_game_over(game.state, game.params))

    def test_smoke_random_rollout(self) -> None:
        env = CraftaxClassicEnv(seed=99)
        env.reset(seed=99)
        rng = np.random.default_rng(0)
        for _ in range(200):
            action = int(rng.integers(0, env.action_size))
            state, reward, done, info = env.step(action)
            self.assertEqual(state.shape, (1345,))
            self.assertIsInstance(reward, float)
            self.assertIn("score", info)
            if done:
                break

    def test_pixel_renderer_draws_map(self) -> None:
        try:
            import pygame
        except ImportError:
            self.skipTest("pygame not installed")

        pygame.init()
        try:
            from src.games.craftax_classic.renderer import CraftaxPixelRenderer

            pygame.display.set_mode((1, 1), flags=getattr(pygame, "HIDDEN", 0))
            env = CraftaxClassicEnv(seed=44)
            env.reset(seed=44)
            assert env.game.state is not None
            frame = CraftaxPixelRenderer(block_pixel_size=16).render(env.game.state)
            arr = pygame.surfarray.array3d(frame)
            # Map region should contain real tile colors, not a blank fill.
            map_h = 7 * 16
            self.assertGreater(float(arr[:, :map_h, :].mean()), 20.0)
            self.assertGreater(int(arr.max()), 100)
        finally:
            pygame.quit()

    def test_block_reverse_moves(self) -> None:
        env = CraftaxClassicEnv(seed=7)
        env.reset(seed=7)
        env.step(int(Action.UP))
        legal = env.legal_actions()
        self.assertNotIn(int(Action.DOWN), legal)
        self.assertIn(int(Action.UP), legal)
        self.assertIn(int(Action.LEFT), legal)

    def test_reverse_penalty_in_shaping(self) -> None:
        from src.games.craftax_classic.rewards import (
            CraftaxRewardShaping,
            shape_craftax_reward,
            snapshot_vitals,
        )

        sp = StaticEnvParams()
        st = EnvState.create_default(sp, np.random.default_rng(0))
        prev = snapshot_vitals(st, last_move=int(Action.UP))
        shaped = CraftaxRewardShaping(survive_bonus=0.0, reverse_penalty=0.02)
        rev = shape_craftax_reward(
            prev, st, base_reward=0.0, done=False, shaping=shaped, action=int(Action.DOWN)
        )
        self.assertAlmostEqual(rev, -0.02, places=6)
        fwd = shape_craftax_reward(
            prev, st, base_reward=0.0, done=False, shaping=shaped, action=int(Action.UP)
        )
        self.assertAlmostEqual(fwd, 0.0, places=6)

    def test_reward_shaping_adds_survive_and_death(self) -> None:
        from src.games.craftax_classic.rewards import (
            CraftaxRewardShaping,
            shape_craftax_reward,
            snapshot_vitals,
        )

        sp = StaticEnvParams()
        st = EnvState.create_default(sp, np.random.default_rng(0))
        prev = snapshot_vitals(st)
        shaped = CraftaxRewardShaping(survive_bonus=0.001, death_penalty=1.0)
        alive = shape_craftax_reward(prev, st, base_reward=0.0, done=False, shaping=shaped)
        self.assertAlmostEqual(alive, 0.001, places=6)

        st.player_health = 0
        dead = shape_craftax_reward(prev, st, base_reward=0.0, done=True, shaping=shaped)
        self.assertAlmostEqual(dead, -1.0, places=6)

        cfg = CraftaxClassicConfig(reward_shaping=True, max_timesteps=3)
        env = CraftaxClassicEnv(config=cfg, seed=1)
        env.reset(seed=1)
        _, reward, done, info = env.step(int(Action.NOOP))
        self.assertIn("raw_reward", info)
        self.assertGreaterEqual(reward, info["raw_reward"])  # survive bonus while alive
        self.assertFalse(done)


if __name__ == "__main__":
    unittest.main()
