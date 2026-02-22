# RL Arcade From Scratch (NumPy)

A from-scratch reinforcement learning game framework in pure Python + NumPy.

This repo started with 2048 and now includes multiple games, custom environments, training/eval/play scripts, pygame visualizations, and several algorithm variants (DQN, DDQN+dueling, tabular Q-learning, afterstate value learning for Tetris).

No `gymnasium`. No `stable-baselines3`.

## What is implemented

### Core RL stack
- Custom environment interfaces (`reset`, `step`, `legal_actions`, `get_state`)
- NumPy MLP Q-network with manual backprop (`src/network.py`)
- Custom Adam optimizer (`src/network.py`)
- Replay buffer with optional legal-action masks (`src/replay_buffer.py`)
- DQN / Double DQN / Dueling options (game-dependent)
- Evaluation scripts + autoplay demos (terminal + pygame)

### Games (current)
- `2048` (DQN)
- `Snake` (DQN, board state + feature state modes)
- `Fruit Cutter` (DQN)
- `Shooter` (DQN)
- `Tetris` (DQN, primitive/placement actions)
- `Tetris Afterstate` (value learning on placement afterstates)
- `Flappy Bird` (DQN + tabular/discretized Q-learning + heuristic baseline)
- `Breakout` (DQN)
- `Pong` (DQN)
- `Match-3` (DQN with legal-action masking + cascade engine)

## Docs
- Framework + 2048 deep dive + multi-game walkthrough: `docs/2048_rl_tutorial.md`

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run tests

```bash
python -m unittest discover -s tests -v
```

## Project layout (main files)

### Shared RL infrastructure
- `src/network.py`: NumPy MLP Q-network + Adam optimizer
- `src/replay_buffer.py`: replay buffer (supports legal-action masks)
- `src/train_game.py`: generic DQN trainer dispatcher
- `src/evaluate_game.py`: generic evaluator dispatcher
- `src/play_game.py`: generic autoplay dispatcher

### 2048
- `src/game_engine.py`, `src/env.py`
- `src/train_dqn.py`, `src/evaluate.py`, `src/play_agent.py`
- `src/eval_utils.py`

### Snake
- `src/games/snake.py`
- `src/train_snake_dqn.py`, `src/evaluate_snake.py`, `src/play_snake_agent.py`
- `src/snake_eval_utils.py`
- `src/benchmark_snake_dqn.py`

### Fruit / Shooter
- `src/games/fruit_cutter.py`, `src/games/shooter.py`
- `src/train_fruit_dqn.py`, `src/evaluate_fruit.py`, `src/play_fruit_agent.py`
- `src/train_shooter_dqn.py`, `src/evaluate_shooter.py`, `src/play_shooter_agent.py`

### Tetris
- `src/games/tetris.py`
- `src/train_tetris_dqn.py`, `src/evaluate_tetris.py`, `src/play_tetris_agent.py`
- `src/train_tetris_afterstate.py`, `src/evaluate_tetris_afterstate.py`, `src/play_tetris_afterstate_agent.py`

### Flappy Bird
- `src/games/flappy.py`
- `src/flappy_env_config.py` (presets + metadata + env mismatch checks)
- `src/flappy_heuristic.py`, `src/flappy_tabular.py`
- `src/train_flappy_dqn.py`, `src/evaluate_flappy.py`, `src/play_flappy_agent.py`
- `src/train_flappy_tabular.py`, `src/evaluate_flappy_tabular.py`, `src/play_flappy_tabular.py`
- `src/evaluate_flappy_heuristic.py`, `src/play_flappy_heuristic.py`

### Breakout / Pong / Match-3
- `src/games/breakout.py`, `src/games/pong.py`, `src/games/match3.py`
- `src/train_breakout_dqn.py`, `src/evaluate_breakout.py`, `src/play_breakout_agent.py`
- `src/train_pong_dqn.py`, `src/evaluate_pong.py`, `src/play_pong_agent.py`
- `src/train_match3_dqn.py`, `src/evaluate_match3.py`, `src/play_match3_agent.py`

## Quick start by game

### 2048 (DQN)

Train:
```bash
python -m src.train_dqn --episodes 3000 --eval-every 50 --eval-episodes 50 --save-dir models/2048
```

Evaluate:
```bash
python -m src.evaluate --model models/2048/dqn_2048_best.json --episodes 200
```

Play (pygame):
```bash
python -m src.play_agent --model models/2048/dqn_2048_best.json --mode pygame --delay 0.08
```

### Snake (DQN)

Recommended: feature-state mode (better learning than flat board MLP on 10x10).

Train (feature state):
```bash
python -m src.train_snake_dqn \
  --state-mode features \
  --episodes 5000 \
  --curriculum-grid-sizes 6,8,10 \
  --double-dqn --dueling \
  --hidden-sizes 128,128 \
  --lr 3e-4 \
  --eps-decay-steps 40000 \
  --eval-every 50 --eval-episodes 50 \
  --save-dir models/snake_features
```

Evaluate (feature state):
```bash
python -m src.evaluate_snake \
  --model models/snake_features/snake_dqn_best.json \
  --episodes 200 --grid-size 10 --state-mode features
```

Play (pygame, feature state):
```bash
python -m src.play_snake_agent \
  --model models/snake_features/snake_dqn_best.json \
  --mode pygame --grid-size 10 --state-mode features --delay 0.06
```

Board-state mode (legacy / baseline) still works:
```bash
python -m src.train_snake_dqn --state-mode board --curriculum-grid-sizes 8,10 --state-grid-size 10 --double-dqn --dueling --save-dir models/snake_board
```

Benchmark baseline vs DDQN+dueling:
```bash
python -m src.benchmark_snake_dqn --episodes 2000 --eval-episodes 100 --save-root models/snake_benchmark
```

### Fruit Cutter (DQN)

Train:
```bash
python -m src.train_fruit_dqn --episodes 3000 --double-dqn --dueling --save-dir models/fruit
```

Evaluate / Play:
```bash
python -m src.evaluate_fruit --model models/fruit/fruit_dqn_best.json --episodes 200
python -m src.play_fruit_agent --model models/fruit/fruit_dqn_best.json --mode pygame --delay 0.08
```

### Shooter (DQN)

Train:
```bash
python -m src.train_shooter_dqn --episodes 3000 --double-dqn --dueling --save-dir models/shooter
```

Evaluate / Play:
```bash
python -m src.evaluate_shooter --model models/shooter/shooter_dqn_best.json --episodes 200
python -m src.play_shooter_agent --model models/shooter/shooter_dqn_best.json --mode pygame --delay 0.08
```

Note: some shooter play runs end at `max_steps` even if lives remain. That is a demo step limit, not necessarily game-over by lives.

### Tetris (DQN and Afterstate)

DQN (placement actions recommended):
```bash
python -m src.train_tetris_dqn \
  --episodes 4000 \
  --height 10 --width 6 --max-steps 500 \
  --placement-actions \
  --double-dqn --dueling \
  --save-dir models/tetris_placement_ddqn
```

Evaluate / Play (placement DQN):
```bash
python -m src.evaluate_tetris --model models/tetris_placement_ddqn/tetris_dqn_best.json --episodes 200 --height 10 --width 6 --max-steps 500 --placement-actions
python -m src.play_tetris_agent --model models/tetris_placement_ddqn/tetris_dqn_best.json --mode pygame --delay 0.08 --height 10 --width 6 --max-steps 500 --placement-actions
```

Afterstate value learning (recommended stronger Tetris path in this repo):
```bash
python -m src.train_tetris_afterstate \
  --episodes 5000 \
  --height 10 --width 6 --max-steps 500 \
  --eval-every 100 --eval-episodes 100 \
  --save-dir models/tetris_afterstate
```

Evaluate / Play (afterstate):
```bash
python -m src.evaluate_tetris_afterstate --model models/tetris_afterstate/tetris_afterstate_best.json --episodes 200 --height 10 --width 6 --max-steps 500
python -m src.play_tetris_afterstate_agent --model models/tetris_afterstate/tetris_afterstate_best.json --mode pygame --delay 0.08 --height 10 --width 6 --max-steps 500
```

### Flappy Bird (heuristic baseline, DQN, tabular Q-learning)

Start with the heuristic baseline to verify env solvability:
```bash
python -m src.evaluate_flappy_heuristic --episodes 200 --env-preset standard
python -m src.play_flappy_heuristic --mode pygame --delay 0.06 --env-preset standard
```

DQN train (recommended baseline uses env presets + metadata):
```bash
python -m src.train_flappy_dqn \
  --episodes 5000 \
  --eval-every 100 --eval-episodes 100 \
  --double-dqn --dueling \
  --env-preset standard \
  --save-dir models/flappy_standard
```

DQN evaluate / play (strict env-model match on by default):
```bash
python -m src.evaluate_flappy --model models/flappy_standard/flappy_dqn_best.json --episodes 200 --env-preset standard
python -m src.play_flappy_agent --model models/flappy_standard/flappy_dqn_best.json --mode pygame --delay 0.06 --env-preset standard
```

Tabular/discretized Q-learning:
```bash
python -m src.train_flappy_tabular --episodes 10000 --eval-every 200 --eval-episodes 100 --env-preset standard --save-dir models/flappy_tabular
python -m src.evaluate_flappy_tabular --model models/flappy_tabular/flappy_tabular_best.json --episodes 200 --env-preset standard
python -m src.play_flappy_tabular --model models/flappy_tabular/flappy_tabular_best.json --mode pygame --delay 0.06 --env-preset standard
```

If you intentionally want to evaluate a Flappy model on different runtime env settings:
```bash
python -m src.evaluate_flappy --model models/flappy_standard/flappy_dqn_best.json --episodes 50 --env-preset hard --allow-env-mismatch --print-model-env
```

### Breakout (DQN)

Train:
```bash
python -m src.train_breakout_dqn --episodes 3000 --eval-every 50 --eval-episodes 50 --double-dqn --dueling --save-dir models/breakout
```

Evaluate / Play:
```bash
python -m src.evaluate_breakout --model models/breakout/breakout_dqn_best.json --episodes 200
python -m src.play_breakout_agent --model models/breakout/breakout_dqn_best.json --mode pygame --delay 0.06
```

### Pong (DQN)

Train:
```bash
python -m src.train_pong_dqn --episodes 3000 --eval-every 50 --eval-episodes 50 --double-dqn --dueling --save-dir models/pong
```

Evaluate / Play:
```bash
python -m src.evaluate_pong --model models/pong/pong_dqn_best.json --episodes 200
python -m src.play_pong_agent --model models/pong/pong_dqn_best.json --mode pygame --delay 0.06
```

### Match-3 (DQN with legal action masking)

Train:
```bash
python -m src.train_match3_dqn \
  --episodes 4000 \
  --eval-every 50 --eval-episodes 50 \
  --double-dqn --dueling \
  --save-dir models/match3
```

Evaluate:
```bash
python -m src.evaluate_match3 --model models/match3/match3_dqn_best.json --episodes 200
```

Play with pygame move/pop animations:
```bash
python -m src.play_match3_agent \
  --model models/match3/match3_dqn_best.json \
  --mode pygame \
  --delay 0.12 \
  --select-delay 0.25 \
  --move-delay 0.25 \
  --clear-delay 0.18
```

## Generic DQN runner (dispatch)

Supported in generic dispatch: `2048`, `snake`, `fruit`, `shooter`, `tetris`, `flappy`, `breakout`, `pong`, `match3`.

Train / evaluate / play examples:
```bash
python -m src.train_game pong --episodes 3000 --double-dqn --dueling --save-dir models/pong_generic
python -m src.evaluate_game pong --model models/pong_generic/pong_dqn_best.json --episodes 200
python -m src.play_game pong --model models/pong_generic/pong_dqn_best.json --mode pygame --delay 0.06
```

## Important compatibility notes

### Flappy env metadata and presets
- Flappy DQN/tabular checkpoints save environment metadata (preset + env config fingerprint).
- Eval/play scripts validate runtime env flags against the saved model metadata by default.
- Use `--allow-env-mismatch` only for explicit cross-env experiments.

### Snake state mode compatibility
- Snake models trained with `--state-mode board` are not compatible with `--state-mode features`, and vice versa.
- Keep train/eval/play settings consistent.

### `best` vs `final`
- `*_best.json` is usually what you want to evaluate/showcase.
- `*_final.json` is just the last checkpoint.

## Common debugging tips

- If training looks stuck early, check warmup and epsilon decay before changing architecture.
- If autoplay is weak but train logs look good, compare the exact eval env config and model checkpoint (`best` vs `final`).
- For Flappy, run the heuristic first to confirm env solvability.
- For Match-3, use pygame animation delays to verify swaps and cascades visually.

## Next improvements (shared infra)
- Prioritized experience replay (PER)
- n-step returns
- Better checkpoint metadata for all games (not just Flappy)
- Benchmark scripts for more games (like Flappy heuristic/tabular/DQN side-by-side)
