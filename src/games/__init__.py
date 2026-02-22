"""Collection of game engines for the from-scratch RL arcade."""

from .breakout import BreakoutConfig, BreakoutEnv, BreakoutGame
from .fruit_cutter import FruitCutterConfig, FruitCutterEnv, FruitCutterGame
from .flappy import FlappyConfig, FlappyEnv, FlappyGame
from .match3 import Match3Config, Match3Env, Match3Game
from .pong import PongConfig, PongEnv, PongGame
from .shooter import ShooterConfig, ShooterEnv, ShooterGame
from .snake import SnakeConfig, SnakeEnv, SnakeGame
from .tetris import TetrisConfig, TetrisEnv, TetrisGame, TetrisPlacementEnv

__all__ = [
    "BreakoutConfig",
    "BreakoutEnv",
    "BreakoutGame",
    "Match3Config",
    "Match3Env",
    "Match3Game",
    "PongConfig",
    "PongEnv",
    "PongGame",
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
