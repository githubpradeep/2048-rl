# 2048 RL From Scratch: Complete Tutorial

This tutorial explains how this project builds a full 2048 reinforcement learning pipeline from scratch.

You will learn:
- how 2048 mechanics are implemented
- how we turn the game into an RL environment
- how DQN works in this codebase
- how replay buffer, target network, and epsilon-greedy exploration fit together
- how to train, evaluate, debug, and visualize the agent

This project does **not** use `gymnasium` or `stable-baselines3`.

## 1. What We Are Building

Goal: train an agent that plays 2048 automatically.

Pipeline:
1. Pure 2048 game engine (deterministic, testable)
2. Custom environment API (`reset`, `step`, `get_state`)
3. DQN agent implemented from scratch
4. Evaluation metrics and model checkpointing
5. Autoplay visualization (terminal + pygame)

Key files:
- `src/game_engine.py`
- `src/env.py`
- `src/replay_buffer.py`
- `src/network.py`
- `src/train_dqn.py`
- `src/eval_utils.py`
- `src/evaluate.py`
- `src/play_agent.py`

## 2. Setup

```bash
cd /Users/pradeep.borado/misc/2048-rl
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run tests first:

```bash
python -m unittest discover -s tests -v
```

## 3. 2048 Engine Design (`src/game_engine.py`)

The game engine is deliberately independent of RL.

Responsibilities:
- board state (`4x4` integer grid)
- tile spawn (`2` with 90%, `4` with 10%)
- movement logic (`up/down/left/right`)
- merge rules (one merge per pair per move)
- score accumulation
- terminal check (`can_move`)
- legal action detection (`legal_actions`)

### Merge correctness

Important rule in 2048:
- `[2,2,2,0] -> [4,2,0,0]`
- not `[8,0,0,0]`

The implementation uses a `compress_and_merge` flow:
1. remove zeros
2. merge equal adjacent values once
3. pad with zeros

### Why pure logic class first?

Because you can test it in isolation.
Any RL instability is easier to diagnose when game rules are guaranteed correct.

## 4. Environment Wrapper (`src/env.py`)

`Game2048Env` wraps the engine with RL-friendly API:
- `reset(seed=None) -> state`
- `step(action) -> next_state, reward, done, info`
- `get_state() -> encoded flat vector`

### State encoding

Raw board values are powers of two and vary a lot.
So we encode non-zero cells with:

`log2(tile_value) / 16`

This keeps values bounded and easier for the network.

### Reward shaping

Current reward components:
- `+gain` from merges (main signal)
- `invalid_move_penalty` when an action does not change board
- terminal bonus/penalty based on reaching target tile

### Important regression fix

Two key policy-quality fixes are now built in:
1. `max_invalid_streak` is disabled by default (`None`) to preserve normal gameplay flow.
2. Action selection in train/eval/play masks illegal moves, so the policy does not waste turns on invalid actions.

## 5. Replay Buffer (`src/replay_buffer.py`)

Stores transitions:
- `state`
- `action`
- `reward`
- `next_state`
- `done`

DQN uses random mini-batches from this memory.

Why this matters:
- breaks temporal correlation
- improves sample efficiency
- stabilizes training compared to online single-step updates

## 6. Q-Network and Optimizer (`src/network.py`)

`MLPQNetwork`:
- fully-connected MLP
- ReLU hidden layers
- linear output layer with one Q-value per action (`4` actions)

`AdamOptimizer`:
- custom Adam implementation for parameter updates

Training step (`train_batch`):
1. forward pass for batch
2. gather predicted Q for selected actions
3. TD error: `Q(s,a) - target`
4. MSE loss: `0.5 * mean(td_error^2)`
5. manual backprop through layers
6. Adam update

## 7. DQN Training Loop (`src/train_dqn.py`)

Core algorithm each environment step:
1. choose action (epsilon-greedy, legal-action masked)
2. step env, receive transition
3. push transition to replay buffer
4. if warmup done and on train cadence: sample batch and optimize
5. periodically copy online network to target network

### DQN target formula

For sampled transition `(s,a,r,s',done)`:

`target = r + gamma * (1 - done) * max_a' Q_target(s', a')`

Then minimize:

`(Q_online(s,a) - target)^2`

### Why target network?

It reduces moving-target instability by keeping bootstrap targets temporarily fixed.

## 8. Evaluation (`src/evaluate.py`, `src/eval_utils.py`)

Evaluation uses greedy policy (epsilon = 0), with legal-action masking.

Reported metrics:
- average score
- median score
- average episode steps
- max tile distribution
- reach rates for `>=512`, `>=1024`, `>=2048`

Run:

```bash
python -m src.evaluate --model models/dqn_2048_best.json --episodes 200
```

## 9. Autoplay and Visualization (`src/play_agent.py`)

Two modes:

Terminal:
```bash
python -m src.play_agent --model models/dqn_2048_best.json --mode terminal --delay 0.1
```

Pygame:
```bash
python -m src.play_agent --model models/dqn_2048_best.json --mode pygame --delay 0.08
```

Useful debug mode:
```bash
python -m src.play_agent --model models/dqn_2048_best.json --mode terminal --debug --delay 0.3
```

Debug prints per move:
- board before action
- board after action
- whether move was valid
- merge gain

## 10. End-to-End Workflow

### Step A: Train

```bash
python -m src.train_dqn --episodes 3000 --eval-every 50 --eval-episodes 50 --save-dir models
```

### Step B: Evaluate best checkpoint

```bash
python -m src.evaluate --model models/dqn_2048_best.json --episodes 300
```

### Step C: Showcase agent

```bash
python -m src.play_agent --model models/dqn_2048_best.json --mode pygame --delay 0.08
```

## 11. How to Read Training Logs Correctly

A common confusion:
- high per-episode training score does **not** necessarily mean learned policy is good.

Why:
- early episodes may be mostly random exploration
- replay and warmup delay actual learning
- noisy reward can look temporarily strong

Trust evaluation metrics on fixed episodes/checkpoints.

## 12. Common Issues and Fixes

### Issue: Training looks stuck

Check:
- `warmup-steps` may delay learning
- model might still be collecting replay data

Try:
- lower warmup for experiments (not final runs)
- ensure loss is non-zero after warmup

### Issue: Poor scores, many invalid moves

Fix:
- ensure legal-action masking is active (already implemented)
- keep invalid move penalties reasonable

### Issue: `final` model weaker than expected

Use `best` checkpoint for deployment:
- `models/dqn_2048_best.json`

## 13. Hyperparameter Tuning Guide

Good knobs to tune first:
- `--lr` (`1e-3` to `3e-4`)
- `--batch-size` (`64`, `128`, `256`)
- `--target-update` (`500` to `2000`)
- `--eps-decay-steps` (slower decay often helps)
- `--hidden-sizes` (e.g. `128,128` or `256,256`)

Example:

```bash
python -m src.train_dqn \
  --episodes 5000 \
  --batch-size 128 \
  --buffer-size 100000 \
  --warmup-steps 5000 \
  --target-update 1000 \
  --lr 0.0007 \
  --eps-decay-steps 500000 \
  --hidden-sizes 256,256 \
  --eval-every 50 \
  --eval-episodes 50
```

## 14. Suggested Study Path for Learners

If you are teaching or self-learning, use this order:
1. Read and run tests for `src/game_engine.py`
2. Step through `src/env.py` rewards and state encoding
3. Understand replay buffer sampling in `src/replay_buffer.py`
4. Trace forward/backward in `src/network.py`
5. Follow training loop in `src/train_dqn.py`
6. Validate behavior with `src/evaluate.py`
7. Visualize policy with `src/play_agent.py`

## 15. Future Extensions

Natural next improvements:
- Double DQN
- Dueling network heads
- Prioritized replay
- Richer board features (empty count, monotonicity, smoothness)
- Better checkpoint metadata and plotting

## 16. Quick Command Cheat Sheet

Install:
```bash
pip install -r requirements.txt
```

Test:
```bash
python -m unittest discover -s tests -v
```

Train:
```bash
python -m src.train_dqn --episodes 3000 --eval-every 50 --eval-episodes 50
```

Evaluate:
```bash
python -m src.evaluate --model models/dqn_2048_best.json --episodes 200
```

Play (pygame):
```bash
python -m src.play_agent --model models/dqn_2048_best.json --mode pygame --delay 0.08
```

---

If you want, the next tutorial chapter can be a line-by-line derivation of DQN updates directly from Bellman equations and how each term maps to the exact variables in `src/train_dqn.py`.
