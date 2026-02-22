"""Collection of game engines for the from-scratch RL arcade."""

from .breakout import BreakoutConfig, BreakoutEnv, BreakoutGame
from .fruit_cutter import FruitCutterConfig, FruitCutterEnv, FruitCutterGame
from .flappy import FlappyConfig, FlappyEnv, FlappyGame
from .shooter import ShooterConfig, ShooterEnv, ShooterGame
from .snake import SnakeConfig, SnakeEnv, SnakeGame
from .tetris import TetrisConfig, TetrisEnv, TetrisGame, TetrisPlacementEnv

__all__ = [
    "BreakoutConfig",
    "BreakoutEnv",
    "BreakoutGame",
    "SnakeConfig",
    "SnakeEnv",
    "SnakeGame",
    "FruitCutterConfig",
    "FruitCutterEnv",
    "FruitCutterGame",
    "FlappyConfig",
    "FlappyEnv",
    "FlappyGame",
    "ShooterConfig",
    "ShooterEnv",
    "ShooterGame",
    "TetrisConfig",
    "TetrisEnv",
    "TetrisPlacementEnv",
    "TetrisGame",
]
