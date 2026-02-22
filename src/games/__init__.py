"""Collection of game engines for the from-scratch RL arcade."""

from .breakout import BreakoutConfig, BreakoutEnv, BreakoutGame
from .env2048 import EnvConfig as Game2048EnvConfig, Game2048Env
from .fruit_cutter import FruitCutterConfig, FruitCutterEnv, FruitCutterGame
from .flappy import FlappyConfig, FlappyEnv, FlappyGame
from .game2048_engine import Game2048
from .match3 import Match3Config, Match3Env, Match3Game
from .pong import PongConfig, PongEnv, PongGame
from .pacman_lite import PacmanLiteConfig, PacmanLiteEnv, PacmanLiteGame
from .shooter import ShooterConfig, ShooterEnv, ShooterGame
from .snake import SnakeConfig, SnakeEnv, SnakeGame
from .tetris import TetrisConfig, TetrisEnv, TetrisGame, TetrisPlacementEnv

__all__ = [
    "BreakoutConfig",
    "BreakoutEnv",
    "BreakoutGame",
    "Game2048",
    "Game2048Env",
    "Game2048EnvConfig",
    "Match3Config",
    "Match3Env",
    "Match3Game",
    "PongConfig",
    "PongEnv",
    "PongGame",
    "PacmanLiteConfig",
    "PacmanLiteEnv",
    "PacmanLiteGame",
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
