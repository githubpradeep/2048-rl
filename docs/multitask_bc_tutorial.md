# Multitask BC Tutorial

This is the clean BC-only generalist path in this repo.

Goal:

- train one model across multiple games
- use offline trajectories from existing specialist DQN agents
- avoid RL instability entirely

The current implementation is intentionally simple:

- per-game state adapters
- shared MLP trunk
- per-game action heads
- supervised cross-entropy on teacher actions
- optional teacher-logit distillation (CE + KL on teacher Q-values)

No transformer. No RTG. No Decision Transformer.

## Why this design

You already have working single-game agents.

That means you can treat them as teachers and export labeled data:

- `state`
- `teacher action`
- `legal action mask`

This turns the problem into ordinary supervised learning.

That is the lowest-risk way to build a single multi-game model.

## Architecture

Implemented in:

- `/Users/pradeep.borado/misc/2048-rl/src/plugins/multitask/bc_model.py`

Structure:

1. Per-game adapter
- one linear layer per game
- maps that game's input dimension into a shared hidden width

2. Shared trunk
- one ReLU MLP shared by all games

3. Per-game head
- one output head per game
- output width matches that game's action count

This gives:

- shared representation learning
- no forced shared input shape
- no forced shared action vocabulary

## Data pipeline

Implemented in:

- `/Users/pradeep.borado/misc/2048-rl/src/plugins/multitask/workflow_bc.py`

Training flow:

1. Load task list from `dataset.tasks`
2. Roll out each teacher model in its own environment
3. Save one compressed shard per game
4. Build mixed-game mini-batches
5. Train the shared BC model

Each exported shard stores:

- `states`
- `actions`
- `legal_masks`
- `teacher_q_values`
- `episode_ids`
- `rewards` (auxiliary metadata)

The BC trainer can use both:

- hard teacher actions
- soft teacher Q-value preferences

That gives better supervision than one-hot labels alone.

## Config schema

Config file:

- `/Users/pradeep.borado/misc/2048-rl/sample_configs/multitask_bc.yaml`

Sections:

### `dataset`

- `manifest`: where the export manifest is written
- `shard-root`: where `.npz` shards are stored
- `auto-export`: export teacher trajectories before training
- `refresh-export`: force rebuild
- `tasks`: list of teachers

Each task supports:

- `game`
- `policy-type` (currently `dqn` only)
- `model-path`
- `sample-weight`
- `max-episodes`
- optional `state-mode`
- optional `env-overrides`

### `model`

- `shared-width`
- `trunk-hidden-sizes`

### `train`

- `epochs`
- `steps-per-epoch`
- `batch-size`
- `lr`
- `grad-clip`
- `label-smoothing`
- `distill`
- `distill-alpha`
- `distill-temperature`
- `distill-overrides`
- `eval-every`
- `save-dir`

`distill-overrides` is a per-game mapping. Each game can override:

- `distill`
- `alpha`
- `temperature`

Example:

```yaml
train:
  distill: true
  distill-alpha: 0.6
  distill-temperature: 2.0
  distill-overrides:
    pong:
      alpha: 0.7
      temperature: 1.5
    flappy:
      alpha: 0.5
      temperature: 2.5
```

Use this when teacher Q-values are calibrated differently across games.

### `eval`

- `model`
- `game`
- `episodes`

### `play`

- `model`
- `game`
- `mode`
- `delay`

## Train

```bash
python -m src.train --env multitask_bc --config sample_configs/multitask_bc.yaml
```

What happens:

1. teacher trajectories are exported (if needed)
2. shared BC model is trained
3. if `distill: true`, the student matches both:
   - teacher chosen action
   - teacher Q-value distribution
   - with optional per-game `alpha` / `temperature` overrides
4. eval runs every `eval-every`
5. checkpoints saved:
   - `multitask_bc_best.json`
   - `multitask_bc_final.json`

## Evaluate

```bash
python -m src.eval --env multitask_bc --config sample_configs/multitask_bc.yaml
```

Override the target game:

```bash
python -m src.eval --env multitask_bc --config sample_configs/multitask_bc.yaml --game breakout
```

## Play

```bash
python -m src.play --env multitask_bc --config sample_configs/multitask_bc.yaml
```

Override the target game:

```bash
python -m src.play --env multitask_bc --config sample_configs/multitask_bc.yaml --game snake
```

## Inline task override

You can override the task list from the CLI.

Use JSON-style quoting in `zsh`:

```bash
python -m src.train --env multitask_bc \
  --config sample_configs/multitask_bc.yaml \
  --dataset.tasks '[{"game":"pong","policy-type":"dqn","model-path":"models/pong/pong_dqn_best.json","max-episodes":5000,"sample-weight":1.0}]' \
  --eval.game pong \
  --play.game pong
```

This is the recommended way to do fast single-game or two-game BC experiments without creating new config files.

## Recommended progression

Start with:

- `pong`
- `breakout`

Then add:

- `snake` (`state-mode: features`)
- `flappy`

Do not start with every game at once.

The model is simple, and early success depends more on clean task scope than on bigger networks.

## Known limitations

- single-step BC only (no temporal memory)
- DQN teachers only
- no mixed modality sequence model

## Best next improvements

1. Better dataset balancing
- more teacher coverage
- per-game caps / weights

2. Stronger benchmarking
- compare specialist vs multitask BC across fixed seeds
