# 2048 RL From Scratch (NumPy)

This project implements everything from scratch in Python for training an RL agent on 2048:
- 2048 game engine
- custom RL environment API (`reset/step/get_state`)
- from-scratch DQN (NumPy-backed MLP + manual backprop + Adam + replay buffer + target network)
- evaluation script
- autoplay demo script (terminal + pygame visualization)
- multi-game scaffold with Snake + Fruit Cutter

No `gymnasium` and no `stable-baselines3` are used.

## Full Tutorial

- Detailed walkthrough: `docs/2048_rl_tutorial.md`

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Project layout

- `src/game_engine.py`: pure 2048 logic
- `src/env.py`: custom environment wrapper
- `src/network.py`: MLP Q-network + Adam optimizer
- `src/replay_buffer.py`: replay buffer implementation
- `src/train_dqn.py`: DQN training loop
- `src/evaluate.py`: policy evaluation
- `src/play_agent.py`: automatic gameplay demo
- `src/games/base.py`: shared game interface
- `src/games/snake.py`: Snake engine + Snake env
- `src/train_snake_dqn.py`: Snake DQN training loop
- `src/evaluate_snake.py`: Snake policy evaluation
- `src/play_snake_agent.py`: Snake autoplay demo
- `src/snake_eval_utils.py`: Snake eval helpers
- `src/games/fruit_cutter.py`: Fruit Cutter engine + env
- `src/train_fruit_dqn.py`: Fruit Cutter DQN training loop
- `src/evaluate_fruit.py`: Fruit Cutter policy evaluation
- `src/play_fruit_agent.py`: Fruit Cutter autoplay demo
- `src/fruit_eval_utils.py`: Fruit eval helpers
- `src/train_game.py`: generic train dispatcher (`2048|snake|fruit`)
- `src/evaluate_game.py`: generic evaluate dispatcher
- `src/play_game.py`: generic play dispatcher
- `tests/`: unit tests

## Run tests

```bash
python -m unittest discover -s tests -v
```

## Train

```bash
python -m src.train_dqn --episodes 2000 --eval-every 50 --eval-episodes 30
```

Useful training flags:

```bash
python -m src.train_dqn \
  --episodes 1000 \
  --batch-size 128 \
  --buffer-size 100000 \
  --warmup-steps 5000 \
  --target-update 1000 \
  --lr 0.001 \
  --hidden-sizes 128,128
```

Models are saved in `models/` as:
- `dqn_2048_best.json`
- `dqn_2048_final.json`

## Evaluate

```bash
python -m src.evaluate --model models/dqn_2048_best.json --episodes 100
```

## Autoplay demo

Terminal mode:

```bash
python -m src.play_agent --model models/dqn_2048_best.json --mode terminal --delay 0.1
```

Debug each transition (before/after board):

```bash
python -m src.play_agent --model models/dqn_2048_best.json --mode terminal --delay 0.3 --debug
```

Pygame visualization mode:

```bash
python -m src.play_agent --model models/dqn_2048_best.json --mode pygame --delay 0.08
```

Controls in pygame mode:
- `Q` or `ESC`: quit

Optional pygame flag:

```bash
python -m src.play_agent --model models/dqn_2048_best.json --mode pygame --close-on-end
```

## Snake (Game #2)

Train:

```bash
python -m src.train_snake_dqn --episodes 1500 --eval-every 50 --eval-episodes 25 --save-dir models/snake
```

Double DQN + dueling architecture:

```bash
python -m src.train_snake_dqn \
  --episodes 5000 \
  --curriculum-grid-sizes 8,10 \
  --state-grid-size 10 \
  --distance-reward-scale 0.2 \
  --double-dqn \
  --dueling \
  --eval-every 50 \
  --eval-episodes 50 \
  --save-dir models/snake_ddqn_dueling
```

Curriculum train (8x8 -> 10x10 in one run):

```bash
python -m src.train_snake_dqn \
  --episodes 5000 \
  --curriculum-grid-sizes 8,10 \
  --state-grid-size 10 \
  --distance-reward-scale 0.2 \
  --eval-every 50 \
  --eval-episodes 50 \
  --save-dir models/snake_v2
```

Evaluate:

```bash
python -m src.evaluate_snake --model models/snake/snake_dqn_best.json --episodes 100
```

Autoplay (terminal):

```bash
python -m src.play_snake_agent --model models/snake/snake_dqn_best.json --mode terminal --delay 0.1
```

Autoplay (pygame):

```bash
python -m src.play_snake_agent --model models/snake/snake_dqn_best.json --mode pygame --delay 0.08
```

If the model was trained with padded state grid, pass `--state-grid-size` consistently:

```bash
python -m src.evaluate_snake --model models/snake_v2/snake_dqn_best.json --episodes 200 --grid-size 10 --state-grid-size 10
python -m src.play_snake_agent --model models/snake_v2/snake_dqn_best.json --mode pygame --grid-size 10 --state-grid-size 10 --delay 0.06
```

Benchmark baseline vs Double+Dueling (same seeds/settings):

```bash
python -m src.benchmark_snake_dqn --episodes 2000 --eval-episodes 100 --save-root models/snake_benchmark
```

## Fruit Cutter (Game #3)

Train:

```bash
python -m src.train_fruit_dqn --episodes 2000 --eval-every 50 --eval-episodes 30 --save-dir models/fruit
```

Double DQN + dueling:

```bash
python -m src.train_fruit_dqn \
  --episodes 3000 \
  --double-dqn \
  --dueling \
  --save-dir models/fruit_ddqn_dueling
```

Evaluate:

```bash
python -m src.evaluate_fruit --model models/fruit/fruit_dqn_best.json --episodes 200
```

Autoplay:

```bash
python -m src.play_fruit_agent --model models/fruit/fruit_dqn_best.json --mode pygame --delay 0.08
```

## Generic Runner

Train any game:

```bash
python -m src.train_game snake --episodes 3000 --save-dir models/snake_generic
python -m src.train_game fruit --episodes 2000 --save-dir models/fruit_generic
python -m src.train_game 2048 --episodes 2000 --save-dir models/2048_generic
```

Evaluate:

```bash
python -m src.evaluate_game snake --model models/snake_generic/snake_dqn_best.json --episodes 100
python -m src.evaluate_game fruit --model models/fruit_generic/fruit_dqn_best.json --episodes 100
python -m src.evaluate_game 2048 --model models/2048_generic/dqn_2048_best.json --episodes 100
```

Play:

```bash
python -m src.play_game snake --model models/snake_generic/snake_dqn_best.json --mode pygame
python -m src.play_game fruit --model models/fruit_generic/fruit_dqn_best.json --mode pygame
python -m src.play_game 2048 --model models/2048_generic/dqn_2048_best.json --mode pygame
```
