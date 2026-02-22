# RL Arcade From Scratch: 2048 + Multi-Game Tutorial

This tutorial explains the project as a reusable reinforcement learning framework built from scratch in Python/NumPy.

It starts with the original 2048 pipeline, then shows how the same ideas are applied across Snake, Tetris, Flappy Bird, Breakout, Pong, Match-3, and other games in this repo.

This project does **not** use `gymnasium` or `stable-baselines3`.

## 1. What This Repo Actually Teaches

This codebase is useful because it combines three things in one place:
- correct game logic (pure engines)
- RL environment wrappers (custom API)
- learning systems (DQN, DDQN+dueling, tabular Q-learning, afterstate learning)

You can study both:
- RL algorithm mechanics
- environment design/debugging (which matters just as much)

## 2. Shared Architecture (Across Games)

Most games in this repo follow the same pattern:

1. Pure game engine (`src/games/...`, e.g. `src/games/game2048_engine.py`)
2. RL env wrapper (`reset`, `step`, `get_state`, `legal_actions`)
3. Unified trainer (`src/train.py`, YAML-driven; internal DQN engine in `src/generic_dqn_train.py`)
4. Evaluator (`src/eval.py`)
5. Unified autoplay demo (`src/play.py`, terminal + pygame)

Core reusable files:
- `src/network.py`: NumPy MLP + manual backprop + Adam
- `src/replay_buffer.py`: replay storage (supports legal-action masks)
- `src/train.py`, `src/eval.py`, `src/play.py`: config-driven unified wrappers (YAML-based)

### Config-driven workflow (recommended)

The repo now also supports a single train/eval/play entry point:

```bash
python -m src.train --env snake --config sample_configs/snake.yaml
python -m src.eval --env snake --config sample_configs/snake.yaml
python -m src.play --env snake --config sample_configs/snake.yaml
```

`sample_configs/*.yaml` store recommended defaults per env. Each config file uses:
- `env`
- `common`
- `train`
- `eval`
- `play` (optional; `src.play` falls back to `eval` if missing)

Use `--dry-run` to inspect the resolved underlying command:

```bash
python -m src.train --env flappy --config sample_configs/flappy.yaml --dry-run
python -m src.play --env flappy --config sample_configs/flappy.yaml --dry-run
```

## 3. The 2048 Baseline (Original Project)

2048 remains the cleanest place to learn the basic DQN flow in this repo.

### 2048 pipeline
- `src/games/game2048_engine.py`: deterministic 2048 rules
- `src/games/env2048.py`: RL wrapper + state encoding + reward shaping
- `src/train.py`: unified YAML-driven trainer (uses `src/generic_dqn_train.py` for DQN)
- `src/eval.py`: greedy policy evaluation
- `src/play.py`: unified autoplay demo (terminal/pygame)

### Why 2048 is a good teaching example
- small action space (`4` actions)
- deterministic transitions (except tile spawn RNG)
- easy to inspect board states
- reward mostly tied to merges

### 2048 commands
Train:
```bash
python -m src.train --env 2048 --episodes 3000 --eval-every 50 --eval-episodes 50 --save-dir models/2048
```

Evaluate:
```bash
python -m src.eval --env 2048 --model models/2048/dqn_2048_best.json --episodes 200
```

Play:
```bash
python -m src.play_agent --model models/2048/dqn_2048_best.json --mode pygame --delay 0.08
```

## 4. DQN in This Codebase (What is Shared)

The DQN trainer loops across games all use the same structure:

1. epsilon-greedy action selection
2. environment step
3. store transition in replay buffer
4. sample mini-batch after warmup
5. compute TD targets with target network
6. manual backprop through MLP
7. periodic target network sync

### Legal-action masking matters
Many games in this repo have invalid actions at a given state (2048, Snake reverse move, Tetris placements, Match-3 swaps).

The project now uses action masking in multiple places:
- action selection (pick among legal actions)
- some target computations (to avoid bootstrapping through illegal actions)

This is a practical improvement, not a cosmetic one. It can materially change training quality.

## 5. Environment Design Is Part of RL (Major Lesson)

A repeated pattern in this project: when results are poor, the issue is often not just “network size” or “more episodes.”

### Examples from this repo
- **Flappy Bird**: physics scaling and pipe generation continuity mattered more than just increasing MLP width.
- **Snake**: board-input flat MLP plateaued; feature-state environment improved learning behavior without changing gameplay.
- **Tetris**: afterstate/value-learning formulation worked better than naive action-value DQN for stronger play.

This is the core engineering lesson: 
- game correctness
- state representation
- reward design
- action space design
are all part of the RL system.

## 6. Game-by-Game Guide (Current)

### 6.1 Snake

#### What changed and why
Snake now supports two state modes:
- `board` (3-channel flattened board tensor)
- `features` (engineered state features, recommended)

Feature mode was added because board-state + flat MLP often plateaus on `10x10` Snake.

#### Recommended Snake training (feature mode)
```bash
python -m src.train --env snake \
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

Evaluate / play:
```bash
python -m src.eval --env snake --model models/snake_features/snake_dqn_best.json --episodes 200 --grid-size 10 --state-mode features
python -m src.play_snake_agent --model models/snake_features/snake_dqn_best.json --mode pygame --grid-size 10 --state-mode features --delay 0.06
```

Important:
- `board` and `features` models are not compatible.
- Keep `--state-mode` consistent in train/eval/play.

### 6.2 Fruit Cutter

Good for learning sparse-ish reward + reactive control on a small grid.

Train / evaluate / play:
```bash
python -m src.train --env fruit --episodes 3000 --double-dqn --dueling --save-dir models/fruit
python -m src.eval --env fruit --model models/fruit/fruit_dqn_best.json --episodes 200
python -m src.play_fruit_agent --model models/fruit/fruit_dqn_best.json --mode pygame --delay 0.08
```

### 6.3 Shooter

A score-and-survival arcade setup with step limits in demos.

Train / evaluate / play:
```bash
python -m src.train --env shooter --episodes 3000 --double-dqn --dueling --save-dir models/shooter
python -m src.eval --env shooter --model models/shooter/shooter_dqn_best.json --episodes 200
python -m src.play_shooter_agent --model models/shooter/shooter_dqn_best.json --mode pygame --delay 0.08
```

Note:
- A play run can end because it hit `--max-steps`, even if lives remain.

### 6.4 Tetris (DQN + Afterstate)

Tetris is much harder for standard DQN in primitive action space.
This repo includes a stronger path: afterstate learning.

#### DQN (placement action mode recommended)
```bash
python -m src.train --env tetris \
  --episodes 4000 \
  --height 10 --width 6 --max-steps 500 \
  --placement-actions \
  --double-dqn --dueling \
  --save-dir models/tetris_placement_ddqn
```

#### Afterstate (recommended for better Tetris learning)
```bash
python -m src.train --env tetris_afterstate \
  --episodes 5000 \
  --height 10 --width 6 --max-steps 500 \
  --eval-every 100 --eval-episodes 100 \
  --save-dir models/tetris_afterstate
```

Evaluate/play afterstate:
```bash
python -m src.eval --env tetris_afterstate --model models/tetris_afterstate/tetris_afterstate_best.json --episodes 200 --height 10 --width 6 --max-steps 500
python -m src.play_tetris_afterstate_agent --model models/tetris_afterstate/tetris_afterstate_best.json --mode pygame --delay 0.08 --height 10 --width 6 --max-steps 500
```

### 6.5 Flappy Bird (Heuristic + DQN + Tabular)

Flappy is the best example in this repo of why environment calibration matters.

#### Key env fixes already in this repo
- board-scaled physics
- floor/ground region
- initial pipe offset/runway
- pipe gap continuity (`max_gap_delta`)
- presets (`easy`, `standard`, `hard`)
- checkpoint env metadata + strict mismatch validation in eval/play

#### Start with heuristic baseline (sanity check)
```bash
python -m src.eval --env flappy_heuristic --episodes 200 --env-preset standard
python -m src.play_flappy_heuristic --mode pygame --delay 0.06 --env-preset standard
```

#### DQN Flappy
```bash
python -m src.train --env flappy \
  --episodes 5000 \
  --eval-every 100 --eval-episodes 100 \
  --double-dqn --dueling \
  --env-preset standard \
  --save-dir models/flappy_standard
```

```bash
python -m src.eval --env flappy --model models/flappy_standard/flappy_dqn_best.json --episodes 200 --env-preset standard
python -m src.play_flappy_agent --model models/flappy_standard/flappy_dqn_best.json --mode pygame --delay 0.06 --env-preset standard
```

#### Tabular/discretized Q-learning Flappy
```bash
python -m src.train --env flappy_tabular --episodes 10000 --eval-every 200 --eval-episodes 100 --env-preset standard --save-dir models/flappy_tabular
python -m src.eval --env flappy_tabular --model models/flappy_tabular/flappy_tabular_best.json --episodes 200 --env-preset standard
python -m src.play_flappy_tabular --model models/flappy_tabular/flappy_tabular_best.json --mode pygame --delay 0.06 --env-preset standard
```

#### Flappy model/env mismatch checks
Flappy eval/play validates the saved checkpoint env metadata by default.
This prevents accidental comparisons across different physics/presets.

Use mismatch override only when doing explicit transfer tests:
```bash
python -m src.eval --env flappy --model models/flappy_standard/flappy_dqn_best.json --episodes 50 --env-preset hard --allow-env-mismatch --print-model-env
```

### 6.6 Breakout

Breakout is a good next-step arcade control task after Flappy/Pong because it has denser reward than Flappy and clearer objective progress.

Train / evaluate / play:
```bash
python -m src.train --env breakout --episodes 3000 --eval-every 50 --eval-episodes 50 --double-dqn --dueling --save-dir models/breakout
python -m src.eval --env breakout --model models/breakout/breakout_dqn_best.json --episodes 200
python -m src.play_breakout_agent --model models/breakout/breakout_dqn_best.json --mode pygame --delay 0.06
```

### 6.7 Pong

Pong is a strong demonstration task for tracking and control with shaped rewards.

Train / evaluate / play:
```bash
python -m src.train --env pong --episodes 3000 --eval-every 50 --eval-episodes 50 --double-dqn --dueling --save-dir models/pong
python -m src.eval --env pong --model models/pong/pong_dqn_best.json --episodes 200
python -m src.play_pong_agent --model models/pong/pong_dqn_best.json --mode pygame --delay 0.06
```

### 6.8 Match-3

Match-3 adds a different RL regime:
- combinatorial action space
- legal swap masking
- cascades / combo rewards

The implementation uses legal-action masking in target bootstrapping and can animate each move in pygame.

Train / evaluate:
```bash
python -m src.train --env match3 --episodes 4000 --eval-every 50 --eval-episodes 50 --double-dqn --dueling --save-dir models/match3
python -m src.eval --env match3 --model models/match3/match3_dqn_best.json --episodes 200
```

Play with visible move and cascade timing:
```bash
python -m src.play_match3_agent \
  --model models/match3/match3_dqn_best.json \
  --mode pygame \
  --delay 0.12 \
  --select-delay 0.25 \
  --move-delay 0.25 \
  --clear-delay 0.18
```

### 6.9 2048 / Fruit / Shooter / Tetris / Flappy via Generic DQN Dispatcher

The generic dispatcher supports these DQN games:
- `2048`, `snake`, `fruit`, `shooter`, `tetris`, `flappy`, `breakout`, `pong`, `match3`

Example:
```bash
python -m src.train pong --episodes 3000 --double-dqn --dueling --save-dir models/pong_generic
python -m src.eval --env pong --model models/pong_generic/pong_dqn_best.json --episodes 200
python -m src.play --env pong --config sample_configs/pong.yaml
```

## 7. How to Read Training Logs (Across Games)

Do not judge a model only from per-episode training logs.

Why:
- epsilon exploration can still be high
- replay warmup delays learning updates
- training reward can be noisy or shaped differently than eval score
- one good autoplay run can be luck

Prefer:
- evaluation metrics on fixed seeds / multiple episodes
- `*_best.json` checkpoints
- median + average, not just max episode score

## 8. Common Failure Modes (and What to Check)

### 8.1 “Training longer did not help”
Common causes:
- environment too hard / poorly calibrated
- bad state representation (not enough useful information)
- plateau in algorithm dynamics (uniform replay, sparse rewards)

Examples in this repo:
- Flappy needed env calibration and presets.
- Snake benefits from feature-state mode.

### 8.2 “Model plays badly but training looked OK”
Check:
- are you using `*_best.json` or `*_final.json`?
- is eval/play using the same env config as training?
- for Snake, did you keep `--state-mode` consistent?
- for Flappy, did env mismatch validation fail / get overridden?

### 8.3 “Game looks wrong in pygame”
Use visualization/debug hooks:
- Match-3: staged animation delays (`--select-delay`, `--move-delay`, `--clear-delay`)
- 2048 terminal debug mode (`--debug`)
- Flappy heuristic baseline for env sanity checks

## 9. Suggested Study Path (Practical)

If you want to understand the codebase efficiently:

1. Start with 2048 (`src/games/game2048_engine.py`, `src/games/env2048.py`, `src/train.py`)
2. Read `src/network.py` and `src/replay_buffer.py`
3. Inspect a masked-action game (Snake or Match-3)
4. Compare Flappy DQN vs tabular + heuristic (env design lesson)
5. Study Tetris afterstate trainer (alternative formulation)

## 10. Recommended Next Improvements (Project-Wide)

High-impact shared upgrades:
- Prioritized Experience Replay (PER)
- n-step returns
- metadata consistency checks for more games (like Flappy)
- benchmark harnesses (heuristic vs tabular vs DQN where applicable)
- CNN support for grid-based board inputs (Snake/Match-3 variants)

## 11. Quick Command Index (Cheat Sheet)

Install / tests:
```bash
pip install -r requirements.txt
python -m unittest discover -s tests -v
```

2048:
```bash
python -m src.train --env 2048 --episodes 3000 --eval-every 50 --eval-episodes 50 --save-dir models/2048
python -m src.eval --env 2048 --model models/2048/dqn_2048_best.json --episodes 200
python -m src.play_agent --model models/2048/dqn_2048_best.json --mode pygame --delay 0.08
```

Snake (feature mode):
```bash
python -m src.train --env snake --state-mode features --episodes 5000 --curriculum-grid-sizes 6,8,10 --double-dqn --dueling --save-dir models/snake_features
python -m src.eval --env snake --model models/snake_features/snake_dqn_best.json --episodes 200 --grid-size 10 --state-mode features
python -m src.play_snake_agent --model models/snake_features/snake_dqn_best.json --mode pygame --grid-size 10 --state-mode features --delay 0.06
```

Flappy (DQN):
```bash
python -m src.train --env flappy --episodes 5000 --double-dqn --dueling --env-preset standard --save-dir models/flappy_standard
python -m src.eval --env flappy --model models/flappy_standard/flappy_dqn_best.json --episodes 200 --env-preset standard
python -m src.play_flappy_agent --model models/flappy_standard/flappy_dqn_best.json --mode pygame --delay 0.06 --env-preset standard
```

Match-3 (animated autoplay):
```bash
python -m src.play_match3_agent --model models/match3/match3_dqn_best.json --mode pygame --delay 0.12 --select-delay 0.25 --move-delay 0.25 --clear-delay 0.18
```
