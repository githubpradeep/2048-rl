import unittest

import numpy as np

from src.games.snake import LEFT, RIGHT, SnakeConfig, SnakeEnv, SnakeGame


class TestSnakeGame(unittest.TestCase):
    def test_reset_spawns_food_not_on_snake(self) -> None:
        game = SnakeGame(seed=11)
        game.reset(seed=11)
        self.assertEqual(len(game.snake), 3)
        self.assertNotIn(game.food, game.snake)

    def test_eat_food_increases_score_and_length(self) -> None:
        game = SnakeGame(seed=1)
        game.reset(seed=1)

        old_length = len(game.snake)
        head_r, head_c = game.snake[-1]
        game.food = (head_r, head_c + 1)

        result = game.step(RIGHT)
        self.assertFalse(result.done)
        self.assertTrue(result.ate_food)
        self.assertEqual(game.score, 1)
        self.assertEqual(len(game.snake), old_length + 1)

    def test_reverse_action_is_ignored(self) -> None:
        game = SnakeGame(seed=2)
        game.reset(seed=2)

        # Snake starts moving RIGHT, so LEFT is reverse and should be ignored.
        old_head = game.snake[-1]
        result = game.step(LEFT)

        self.assertTrue(result.reversed_input)
        self.assertEqual(game.direction, RIGHT)
        self.assertEqual(game.snake[-1], (old_head[0], old_head[1] + 1))

    def test_collision_with_wall_ends_game(self) -> None:
        cfg = SnakeConfig(grid_size=6)
        game = SnakeGame(config=cfg, seed=3)
        game.reset(seed=3)

        game.snake = [(2, 3), (2, 4), (2, 5)]
        game.direction = RIGHT

        result = game.step(RIGHT)
        self.assertTrue(result.done)
        self.assertTrue(result.collision)

    def test_starvation_ends_game(self) -> None:
        cfg = SnakeConfig(
            grid_size=7,
            step_reward=0.0,
            death_penalty=-5.0,
            distance_reward_scale=0.0,
            max_steps_without_food=2,
        )
        game = SnakeGame(config=cfg, seed=4)
        game.reset(seed=4)

        head_r, head_c = game.snake[-1]
        game.food = (head_r + 3, head_c)

        first = game.step(RIGHT)
        self.assertFalse(first.done)
        second = game.step(RIGHT)
        self.assertTrue(second.done)
        self.assertEqual(second.reward, -5.0)

    def test_distance_reward_shaping(self) -> None:
        cfg = SnakeConfig(
            grid_size=8,
            step_reward=0.0,
            food_reward=0.0,
            death_penalty=-10.0,
            reverse_penalty=0.0,
            distance_reward_scale=1.0,
            max_steps_without_food=100,
        )
        game = SnakeGame(config=cfg, seed=7)
        game.reset(seed=7)

        game.snake = [(3, 1), (3, 2), (3, 3)]
        game.direction = 0  # UP
        game.food = (0, 3)
        closer = game.step(0)  # move up, closer by 1
        self.assertAlmostEqual(closer.reward, 1.0, places=6)

        game.snake = [(3, 1), (3, 2), (3, 3)]
        game.direction = 0  # UP
        game.food = (0, 3)
        farther = game.step(RIGHT)  # move right, farther by 1
        self.assertAlmostEqual(farther.reward, -1.0, places=6)


class TestSnakeEnv(unittest.TestCase):
    def test_state_shape(self) -> None:
        env = SnakeEnv(config=SnakeConfig(grid_size=8), seed=5)
        state = env.reset(seed=5)
        self.assertEqual(state.shape, (3 * 8 * 8,))
        self.assertEqual(state.dtype, np.float32)

    def test_state_shape_with_padding_grid(self) -> None:
        env = SnakeEnv(config=SnakeConfig(grid_size=8), seed=5, state_grid_size=10)
        state = env.reset(seed=5)
        self.assertEqual(state.shape, (3 * 10 * 10,))
        self.assertEqual(state.dtype, np.float32)

    def test_legal_actions_exclude_reverse(self) -> None:
        env = SnakeEnv(config=SnakeConfig(grid_size=8), seed=6)
        env.reset(seed=6)
        legal = env.legal_actions()
        self.assertNotIn(LEFT, legal)


if __name__ == "__main__":
    unittest.main()
