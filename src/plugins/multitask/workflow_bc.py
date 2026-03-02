from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from ...generic_dqn_train import build_dqn_hooks, masked_argmax
from ...network import MLPQNetwork
from .bc_model import MultitaskBCNetwork, MultitaskBCOptimizers


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml  # type: ignore
    except ImportError as exc:
        raise RuntimeError("PyYAML is required. Install with `pip install -r requirements.txt`.") from exc
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config file must contain a top-level mapping: {path}")
    return data


def _normalize(v: Any) -> Any:
    if isinstance(v, dict):
        return {str(k).replace("-", "_"): _normalize(x) for k, x in v.items()}
    if isinstance(v, list):
        return [_normalize(x) for x in v]
    return v


def _parse_overrides(tokens: list[str]) -> dict[str, Any]:
    def _coerce(raw: str) -> Any:
        low = raw.lower()
        if low in {"true", "false"}:
            return (low == "true")
        if raw[:1] in {"[", "{"}:
            try:
                import yaml  # type: ignore

                return yaml.safe_load(raw)
            except Exception:
                return raw
        return raw

    out: dict[str, Any] = {}
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if not tok.startswith("--"):
            raise ValueError(f"Unexpected positional override token: {tok}")
        key = tok[2:].replace("-", "_")
        if i + 1 < len(tokens) and not tokens[i + 1].startswith("--"):
            out[key] = _coerce(tokens[i + 1])
            i += 2
        else:
            out[key] = True
            i += 1
    return out


def _deep_get(cfg: dict[str, Any], path: str, default: Any = None) -> Any:
    cur: Any = cfg
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


def _deep_set(cfg: dict[str, Any], path: str, value: Any) -> None:
    parts = path.split(".")
    cur = cfg
    for part in parts[:-1]:
        nxt = cur.get(part)
        if not isinstance(nxt, dict):
            nxt = {}
            cur[part] = nxt
        cur = nxt
    cur[parts[-1]] = value


def apply_cli_overrides(full_cfg: dict[str, Any], overrides: dict[str, Any], *, section: str) -> dict[str, Any]:
    cfg = _normalize(full_cfg)
    for k, v in overrides.items():
        if "." in k:
            _deep_set(cfg, k, v)
        else:
            _deep_set(cfg, f"{section}.{k}", v)
    return cfg


def _task_params(task: dict[str, Any], *, seed: int) -> dict[str, Any]:
    params: dict[str, Any] = {"seed": int(seed)}
    if "state_mode" in task:
        params["state_mode"] = task["state_mode"]
    if "max_steps" in task:
        params["max_steps"] = int(task["max_steps"])
    env_overrides = task.get("env_overrides", {})
    if isinstance(env_overrides, dict):
        for k, v in env_overrides.items():
            params[str(k).replace("-", "_")] = v
    return params


def _pick_teacher_action(net: MLPQNetwork, state: np.ndarray, legal_actions: list[int]) -> int:
    q_values = net.predict_one(state)
    if legal_actions:
        return masked_argmax(q_values, legal_actions)
    return int(max(range(len(q_values)), key=lambda i: q_values[i]))


def _maybe_export(cfg: dict[str, Any]) -> Path:
    dataset_cfg = dict(cfg.get("dataset") or {})
    manifest_path = Path(str(dataset_cfg.get("manifest", "data/multitask_bc/manifest.json")))
    auto_export = bool(dataset_cfg.get("auto_export", True))
    refresh = bool(dataset_cfg.get("refresh_export", False))
    if manifest_path.exists() and not refresh:
        return manifest_path
    if not auto_export and not manifest_path.exists():
        raise FileNotFoundError(f"Missing dataset manifest and auto_export is disabled: {manifest_path}")
    print(f"Exporting trajectories -> {manifest_path}")
    return export_multitask_bc_dataset(cfg, manifest_path=manifest_path)


def export_multitask_bc_dataset(cfg: dict[str, Any], *, manifest_path: Path | None = None) -> Path:
    cfg = _normalize(cfg)
    dataset_cfg = dict(cfg.get("dataset") or {})
    tasks = list(dataset_cfg.get("tasks") or [])
    if not tasks:
        raise ValueError("multitask_bc requires dataset.tasks")

    manifest = Path(str(manifest_path or dataset_cfg.get("manifest", "data/multitask_bc/manifest.json")))
    root_dir = manifest.parent
    shard_root = Path(str(dataset_cfg.get("shard_root", root_dir / "shards")))
    root_dir.mkdir(parents=True, exist_ok=True)
    shard_root.mkdir(parents=True, exist_ok=True)

    common_seed = int(((cfg.get("common") or {}).get("seed")) or 42)
    shards: list[dict[str, Any]] = []

    for ti, task_raw in enumerate(tasks):
        if not isinstance(task_raw, dict):
            raise ValueError("dataset.tasks must contain mappings")
        task = _normalize(dict(task_raw))
        game = str(task["game"])
        policy_type = str(task.get("policy_type", "dqn"))
        if policy_type != "dqn":
            raise ValueError(f"multitask_bc currently supports only policy_type=dqn (got {policy_type!r} for {game})")
        model_path = Path(str(task.get("model_path", "")))
        if not model_path.exists():
            raise FileNotFoundError(f"Teacher model not found for {game}: {model_path}")
        episodes = int(task.get("max_episodes", 100))
        seed = common_seed + 1000 * (ti + 1)
        params = _task_params(task, seed=seed)

        env0, hooks, _extra = build_dqn_hooks(game, params, seed, episodes)
        teacher = MLPQNetwork.load(model_path)
        del env0

        states: list[np.ndarray] = []
        actions: list[int] = []
        masks: list[np.ndarray] = []
        episode_ids: list[int] = []
        rewards: list[float] = []

        for ep in range(1, episodes + 1):
            env = hooks.build_episode_env(ep)
            state = np.asarray(env.reset(seed=seed + ep), dtype=np.float32)
            done = False
            steps = 0
            max_steps = int(task.get("max_steps", params.get("max_steps", 1000)))
            while not done and steps < max_steps:
                legal_actions = env.legal_actions() if hasattr(env, "legal_actions") else list(range(int(env.action_size)))
                action = _pick_teacher_action(teacher, state, legal_actions)
                mask = np.zeros((int(env.action_size),), dtype=np.float32)
                if legal_actions:
                    mask[np.asarray(legal_actions, dtype=np.int64)] = 1.0
                else:
                    mask[:] = 1.0

                next_state, reward, done, _info = env.step(action)
                states.append(state.copy())
                actions.append(int(action))
                masks.append(mask)
                episode_ids.append(ep - 1)
                rewards.append(float(reward))

                state = np.asarray(next_state, dtype=np.float32)
                steps += 1

        if not states:
            raise RuntimeError(f"No transitions exported for {game}")

        states_arr = np.stack(states, axis=0).astype(np.float32)
        actions_arr = np.asarray(actions, dtype=np.int64)
        masks_arr = np.stack(masks, axis=0).astype(np.float32)
        episode_ids_arr = np.asarray(episode_ids, dtype=np.int64)
        rewards_arr = np.asarray(rewards, dtype=np.float32)

        shard_name = f"{game}_bc_{episodes}ep.npz"
        shard_path = shard_root / shard_name
        np.savez_compressed(
            shard_path,
            game=np.asarray(game),
            states=states_arr,
            actions=actions_arr,
            legal_masks=masks_arr,
            episode_ids=episode_ids_arr,
            rewards=rewards_arr,
            state_dim=np.asarray(states_arr.shape[1], dtype=np.int32),
            action_dim=np.asarray(masks_arr.shape[1], dtype=np.int32),
            task_json=np.asarray(json.dumps(task)),
        )
        shards.append(
            {
                "game": game,
                "shard_path": str(shard_path.relative_to(root_dir)),
                "num_steps": int(states_arr.shape[0]),
                "num_episodes": int(episodes),
                "state_dim": int(states_arr.shape[1]),
                "action_dim": int(masks_arr.shape[1]),
            }
        )

    manifest_data = {"root": str(root_dir), "shards": shards}
    manifest.write_text(json.dumps(manifest_data, indent=2), encoding="utf-8")
    return manifest


class MultitaskBCDataset:
    def __init__(self, manifest: dict[str, Any]) -> None:
        self.manifest = manifest
        self.root = Path(str(manifest["root"]))
        self.shards: dict[str, dict[str, Any]] = {}
        self.games: list[str] = []
        self.game_input_dims: dict[str, int] = {}
        self.game_action_dims: dict[str, int] = {}
        self.game_num_steps: dict[str, int] = {}

        for entry in list(manifest.get("shards") or []):
            game = str(entry["game"])
            shard_path = self.root / str(entry["shard_path"])
            data = np.load(shard_path, allow_pickle=False)
            shard = {
                "states": np.asarray(data["states"], dtype=np.float32),
                "actions": np.asarray(data["actions"], dtype=np.int64),
                "legal_masks": np.asarray(data["legal_masks"], dtype=np.float32),
                "rewards": np.asarray(data["rewards"], dtype=np.float32) if "rewards" in data else None,
                "episode_ids": np.asarray(data["episode_ids"], dtype=np.int64) if "episode_ids" in data else None,
                "state_dim": int(data["state_dim"].item()),
                "action_dim": int(data["action_dim"].item()),
                "num_steps": int(data["states"].shape[0]),
            }
            self.shards[game] = shard
            self.games.append(game)
            self.game_input_dims[game] = int(shard["state_dim"])
            self.game_action_dims[game] = int(shard["action_dim"])
            self.game_num_steps[game] = int(shard["num_steps"])
        self.games.sort()
        if not self.games:
            raise ValueError("No BC shards found")

    @classmethod
    def from_manifest(cls, path: str | Path) -> "MultitaskBCDataset":
        manifest = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(manifest, dict):
            raise ValueError("Invalid multitask BC manifest")
        return cls(manifest)

    def sample_game(self, rng: np.random.Generator, weights: dict[str, float]) -> str:
        probs = np.asarray([max(float(weights.get(g, 1.0)), 0.0) for g in self.games], dtype=np.float64)
        total = float(np.sum(probs))
        if total <= 0.0:
            probs = np.ones_like(probs) / len(probs)
        else:
            probs = probs / total
        idx = int(rng.choice(len(self.games), p=probs))
        return self.games[idx]

    def sample_batch(self, game: str, batch_size: int, rng: np.random.Generator) -> dict[str, np.ndarray]:
        shard = self.shards[game]
        idx = rng.integers(0, int(shard["num_steps"]), size=int(batch_size))
        return {
            "states": shard["states"][idx],
            "actions": shard["actions"][idx],
            "legal_masks": shard["legal_masks"][idx],
        }


def _build_task_sampling_weights(tasks: list[dict[str, Any]]) -> dict[str, float]:
    out: dict[str, float] = {}
    for item in tasks:
        if not isinstance(item, dict):
            continue
        norm = _normalize(item)
        out[str(norm.get("game"))] = float(norm.get("sample_weight", 1.0))
    return out


def _load_multitask_model(model_path: str) -> MultitaskBCNetwork:
    return MultitaskBCNetwork.load(model_path)


def _task_entry_by_game(cfg: dict[str, Any], game: str) -> dict[str, Any] | None:
    tasks = _deep_get(cfg, "dataset.tasks", [])
    if not isinstance(tasks, list):
        return None
    for item in tasks:
        if isinstance(item, dict):
            norm = _normalize(item)
            if str(norm.get("game")) == game:
                return norm
    return None


def _params_for_game(cfg: dict[str, Any], game: str, *, section: str, seed: int) -> dict[str, Any]:
    params: dict[str, Any] = {"seed": seed}
    task = _task_entry_by_game(cfg, game) or {}
    params.update(_task_params(task, seed=seed))
    sec_cfg = dict(_deep_get(cfg, section, {}) or {})
    for k, v in _normalize(sec_cfg).items():
        if k in {"model", "episodes", "game", "mode", "delay", "close_on_end"}:
            continue
        params[k] = v
    return params


class _PolicyProxy:
    def __init__(self, model: MultitaskBCNetwork, game: str) -> None:
        self.model = model
        self.game = game

    def predict_one(self, state: Any) -> list[float]:
        scores = self.model.predict_action_scores(game=self.game, state=np.asarray(state, dtype=np.float32))
        return [float(v) for v in scores.tolist()]

    def predict_batch(self, states: Any) -> list[list[float]]:
        arr = np.asarray(states, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        out: list[list[float]] = []
        for i in range(arr.shape[0]):
            out.append(self.predict_one(arr[i]))
        return out


def _eval_multitask(cfg: dict[str, Any], *, model: MultitaskBCNetwork | None = None, game: str | None = None, episodes: int | None = None, seed: int | None = None) -> tuple[Any, str, float]:
    cfg = _normalize(cfg)
    eval_cfg = dict(cfg.get("eval") or {})
    game_name = str(game or eval_cfg.get("game", "pong"))
    eval_episodes = int(episodes or eval_cfg.get("episodes", 50))
    eval_seed = int(seed if seed is not None else ((cfg.get("common") or {}).get("seed") or 42))
    model_path = str(eval_cfg.get("model", ""))
    if model is None:
        if not model_path:
            raise ValueError("multitask_bc eval requires eval.model")
        model = _load_multitask_model(model_path)

    params = _params_for_game(cfg, game_name, section="eval", seed=eval_seed)
    _env0, hooks, _extra = build_dqn_hooks(game_name, params, eval_seed, 1)
    proxy = _PolicyProxy(model, game_name)
    eval_env = hooks.build_eval_env()
    stats = hooks.eval_once(eval_env, proxy, eval_episodes, eval_seed + 5000)
    return stats, hooks.eval_summary(stats), float(hooks.eval_metric(stats))


def _train(cfg: dict[str, Any]) -> None:
    cfg = _normalize(cfg)
    common = dict(cfg.get("common") or {})
    train_cfg = dict(cfg.get("train") or {})
    dataset_cfg = dict(cfg.get("dataset") or {})

    seed = int(common.get("seed", 42))
    rng = np.random.default_rng(seed)

    manifest_path = _maybe_export(cfg)
    ds = MultitaskBCDataset.from_manifest(manifest_path)
    task_weights = _build_task_sampling_weights(list(dataset_cfg.get("tasks") or []))

    shared_width = int(_deep_get(cfg, "model.shared_width", 128))
    trunk_hidden_sizes_raw = str(_deep_get(cfg, "model.trunk_hidden_sizes", "128,128"))
    trunk_hidden_sizes = tuple(int(v.strip()) for v in trunk_hidden_sizes_raw.split(",") if v.strip())
    lr = float(train_cfg.get("lr", 3e-4))
    grad_clip = train_cfg.get("grad_clip", 1.0)
    grad_clip_f = None if grad_clip is None else float(grad_clip)
    label_smoothing = float(train_cfg.get("label_smoothing", 0.0))
    epochs = int(train_cfg.get("epochs", 20))
    steps_per_epoch = int(train_cfg.get("steps_per_epoch", 200))
    batch_size = int(train_cfg.get("batch_size", 64))
    eval_every = int(train_cfg.get("eval_every", 1))
    eval_episodes = int(_deep_get(cfg, "eval.episodes", 25))
    eval_game = str(_deep_get(cfg, "eval.game", ds.games[0]))
    save_dir = Path(str(train_cfg.get("save_dir", "models/multitask_bc")))
    save_dir.mkdir(parents=True, exist_ok=True)
    best_path = save_dir / "multitask_bc_best.json"
    final_path = save_dir / "multitask_bc_final.json"

    model = MultitaskBCNetwork(
        game_input_dims=ds.game_input_dims,
        game_action_dims=ds.game_action_dims,
        shared_width=shared_width,
        trunk_hidden_sizes=trunk_hidden_sizes,
        seed=seed,
    )
    optim = MultitaskBCOptimizers(lr=lr, games=ds.games, max_grad_norm=grad_clip_f)

    best_metric = -float("inf")
    print(
        f"Training multitask_bc (shared MLP BC) | games={','.join(ds.games)} "
        f"| epochs={epochs} steps/epoch={steps_per_epoch} batch={batch_size}"
    )
    for epoch in range(1, epochs + 1):
        losses: list[float] = []
        accs: list[float] = []
        game_counts = {g: 0 for g in ds.games}
        for _ in range(steps_per_epoch):
            game = ds.sample_game(rng, task_weights)
            game_counts[game] += 1
            batch = ds.sample_batch(game, batch_size, rng)
            metrics = model.train_game_batch(
                game=game,
                states=batch["states"],
                actions=batch["actions"],
                legal_masks=batch["legal_masks"],
                optimizers=optim,
                label_smoothing=label_smoothing,
            )
            losses.append(float(metrics["loss"]))
            accs.append(float(metrics["acc"]))

        print(
            f"epoch={epoch:3d} loss={float(np.mean(losses)):.4f} acc={float(np.mean(accs)):.4f} "
            + " ".join(f"{g}:{game_counts[g]}" for g in sorted(game_counts))
        )
        if epoch % eval_every == 0:
            _stats, summary, metric = _eval_multitask(cfg, model=model, game=eval_game, episodes=eval_episodes, seed=seed + 7000)
            print(f"  eval({eval_game}): {summary}")
            if metric > best_metric:
                best_metric = metric
                model.save(best_path)
                print(f"  saved new best checkpoint: {best_path}")
    model.save(final_path)
    print(f"Training complete. Final model saved to: {final_path}")


def run_train_from_config(full_cfg: dict[str, Any]) -> None:
    _train(full_cfg)


def run_eval_from_config(full_cfg: dict[str, Any]) -> None:
    cfg = _normalize(full_cfg)
    game_name = str(_deep_get(cfg, "eval.game", "pong"))
    episodes = int(_deep_get(cfg, "eval.episodes", 50))
    _stats, summary, _metric = _eval_multitask(cfg, game=game_name, episodes=episodes)
    print("Multitask BC Evaluation Results")
    print("------------------------------")
    print(f"Game: {game_name}")
    print(f"Episodes: {episodes}")
    print(summary)


def run_play_from_config(full_cfg: dict[str, Any]) -> None:
    cfg = _normalize(full_cfg)
    common = dict(cfg.get("common") or {})
    play_cfg = dict(cfg.get("play") or {})
    eval_cfg = dict(cfg.get("eval") or {})
    if not play_cfg:
        raise ValueError("multitask_bc play requires a `play` section")

    seed = int(play_cfg.get("seed", common.get("seed", 42)))
    game_name = str(play_cfg.get("game", eval_cfg.get("game", "pong")))
    model_path = str(play_cfg.get("model", eval_cfg.get("model", "")))
    if not model_path:
        raise ValueError("multitask_bc play requires play.model")
    mode = str(play_cfg.get("mode", "terminal"))
    delay = float(play_cfg.get("delay", 0.08))
    max_steps = int(play_cfg.get("max_steps", eval_cfg.get("max_steps", 1000)))
    close_on_end = bool(play_cfg.get("close_on_end", False))

    model = _load_multitask_model(model_path)
    params = _params_for_game(cfg, game_name, section="play", seed=seed)
    params["max_steps"] = max_steps
    _env0, hooks, _extra = build_dqn_hooks(game_name, params, seed, 1)
    env = hooks.build_eval_env()
    proxy = _PolicyProxy(model, game_name)

    from ..playback import (
        play_agent,
        play_breakout_agent,
        play_flappy_agent,
        play_fruit_agent,
        play_match3_agent,
        play_pacman_agent,
        play_pong_agent,
        play_shooter_agent,
        play_snake_agent,
        play_tetris_agent,
    )

    playback_map: dict[str, Any] = {
        "2048": play_agent,
        "pong": play_pong_agent,
        "breakout": play_breakout_agent,
        "snake": play_snake_agent,
        "flappy": play_flappy_agent,
        "fruit": play_fruit_agent,
        "shooter": play_shooter_agent,
        "pacman": play_pacman_agent,
        "match3": play_match3_agent,
        "tetris": play_tetris_agent,
    }
    if game_name not in playback_map:
        raise ValueError(f"multitask_bc play not supported for game: {game_name}")
    mod = playback_map[game_name]

    print(f"Starting multitask BC autoplay | model={Path(model_path).name} | game={game_name} | mode={mode}")
    if mode == "pygame":
        kwargs: dict[str, Any] = {}
        if game_name == "match3":
            if "select_delay" in play_cfg:
                kwargs["select_delay"] = float(play_cfg["select_delay"])
            if "move_delay" in play_cfg:
                kwargs["move_delay"] = float(play_cfg["move_delay"])
            if "clear_delay" in play_cfg:
                kwargs["clear_delay"] = float(play_cfg["clear_delay"])
        result = mod.run_pygame(env, proxy, seed, delay, max_steps, close_on_end, **kwargs) if kwargs else mod.run_pygame(env, proxy, seed, delay, max_steps, close_on_end)
    else:
        if game_name == "2048":
            debug = bool(play_cfg.get("debug", False))
            result = mod.run_terminal(env, proxy, seed, delay, max_steps, debug)
        else:
            result = mod.run_terminal(env, proxy, seed, delay, max_steps)

    print(f"Multitask BC play finished for {game_name}")
    for key, value in vars(result).items():
        print(f"{key}: {value}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Multitask BC workflow (simple shared MLP)")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--mode", choices=["export", "train", "eval", "play"], default="train")
    parser.add_argument("--dry-run", action="store_true")
    args, rest = parser.parse_known_args()

    cfg_path = Path(args.config)
    cfg = _load_yaml(cfg_path)
    cfg = apply_cli_overrides(cfg, _parse_overrides(rest), section=args.mode)

    if args.dry_run:
        print(json.dumps(_normalize(cfg), indent=2, sort_keys=True))
        return

    if args.mode == "export":
        out = export_multitask_bc_dataset(cfg)
        print(f"Exported manifest: {out}")
    elif args.mode == "train":
        run_train_from_config(cfg)
    elif args.mode == "eval":
        run_eval_from_config(cfg)
    elif args.mode == "play":
        run_play_from_config(cfg)


if __name__ == "__main__":
    main()

