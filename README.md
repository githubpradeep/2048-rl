# 2048 RL From Scratch (NumPy)

This project implements everything from scratch in Python for training an RL agent on 2048:
- 2048 game engine
- custom RL environment API (`reset/step/get_state`)
- from-scratch DQN (NumPy-backed MLP + manual backprop + Adam + replay buffer + target network)
- evaluation script
- autoplay demo script (terminal + pygame visualization)

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
