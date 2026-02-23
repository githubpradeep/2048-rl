from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


SUPPORTED_GENERIC_ENVS = {"snake", "tetris", "match3", "pacman", "pong", "breakout", "fruit", "shooter"}


def _read_model_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Model JSON payload must be an object")
    return payload


def write_model_metadata(path: str | Path, metadata: dict[str, Any]) -> None:
    p = Path(path)
    payload = _read_model_json(p)
    payload["metadata"] = metadata
    p.write_text(json.dumps(payload), encoding="utf-8")


def read_model_metadata(path: str | Path) -> dict[str, Any] | None:
    payload = _read_model_json(path)
    metadata = payload.get("metadata")
    if isinstance(metadata, dict):
        return metadata
    return None


def _stable_fingerprint(env_config: dict[str, Any]) -> str:
    raw = json.dumps(env_config, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def _as_int(params: dict[str, Any], key: str, default: int) -> int:
    return int(params.get(key, default))


def _as_float(params: dict[str, Any], key: str, default: float) -> float:
    return float(params.get(key, default))


def _as_bool(params: dict[str, Any], key: str, default: bool) -> bool:
    val = params.get(key, default)
    if isinstance(val, bool):
        return val
    if isinstance(val, str):
        return val.lower() == "true"
    return bool(val)


def _parse_int_csv(raw: Any) -> list[int]:
    if raw is None:
        return []
    if isinstance(raw, (list, tuple)):
        return [int(v) for v in raw]
    vals = [v.strip() for v in str(raw).split(",") if v.strip()]
    return [int(v) for v in vals]


def runtime_env_config_from_params(env_name: str, params: dict[str, Any]) -> dict[str, Any] | None:
    if env_name not in SUPPORTED_GENERIC_ENVS:
        return None

    if env_name == "snake":
        state_mode = str(params.get("state_mode", "board"))
        stages = _parse_int_csv(params.get("curriculum_grid_sizes"))
        grid_size = max(stages) if stages else _as_int(params, "grid_size", 10)
        payload: dict[str, Any] = {
            "grid_size": grid_size,
            "state_mode": state_mode,
        }
        if state_mode == "board":
            state_grid = _as_int(params, "state_grid_size", 0) or grid_size
            payload["state_grid_size"] = state_grid
        return payload

    if env_name == "tetris":
        return {
            "height": _as_int(params, "height", 20),
            "width": _as_int(params, "width", 10),
            "max_steps": _as_int(params, "max_steps", 1200),
            "placement_actions": _as_bool(params, "placement_actions", False),
        }

    if env_name == "match3":
        return {
            "width": _as_int(params, "width", 6),
            "height": _as_int(params, "height", 6),
            "num_colors": _as_int(params, "num_colors", 5),
            "max_steps": _as_int(params, "max_steps", 150),
            "reshuffle_on_stuck": _as_bool(params, "reshuffle_on_stuck", True),
        }

    if env_name == "pacman":
        grid_size = _as_int(params, "grid_size", 11)
        return {
            "grid_size": grid_size,
            "state_grid_size": (_as_int(params, "state_grid_size", 0) or grid_size),
            "num_ghosts": _as_int(params, "num_ghosts", 2),
            "ghost_chase_prob": _as_float(params, "ghost_chase_prob", 0.75),
            "max_steps": _as_int(params, "max_steps", 500),
            "start_lives": _as_int(params, "start_lives", 3),
        }

    if env_name == "pong":
        return {
            "width": _as_int(params, "width", 12),
            "height": _as_int(params, "height", 16),
            "paddle_height": _as_int(params, "paddle_height", 4),
            "paddle_speed": _as_int(params, "paddle_speed", 1),
            "opponent_track_prob": _as_float(params, "opponent_track_prob", 0.85),
            "start_lives": _as_int(params, "start_lives", 3),
            "max_steps": _as_int(params, "max_steps", 1000),
        }

    if env_name == "breakout":
        return {
            "width": _as_int(params, "width", 12),
            "height": _as_int(params, "height", 14),
            "brick_rows": _as_int(params, "brick_rows", 4),
            "brick_top": _as_int(params, "brick_top", 1),
            "paddle_width": _as_int(params, "paddle_width", 3),
            "start_lives": _as_int(params, "start_lives", 3),
            "max_steps": _as_int(params, "max_steps", 1200),
        }

    if env_name == "fruit":
        grid_size = _as_int(params, "grid_size", 10)
        return {
            "grid_size": grid_size,
            "state_grid_size": (_as_int(params, "state_grid_size", 0) or grid_size),
            "spawn_prob": _as_float(params, "spawn_prob", 0.45),
            "bomb_prob": _as_float(params, "bomb_prob", 0.18),
            "max_steps": _as_int(params, "max_steps", 500),
        }

    if env_name == "shooter":
        grid_size = _as_int(params, "grid_size", 10)
        return {
            "grid_size": grid_size,
            "state_grid_size": (_as_int(params, "state_grid_size", 0) or grid_size),
            "spawn_prob": _as_float(params, "spawn_prob", 0.35),
            "max_steps": _as_int(params, "max_steps", 600),
            "start_lives": _as_int(params, "start_lives", 3),
        }

    return None


def dqn_model_metadata_from_params(env_name: str, params: dict[str, Any]) -> dict[str, Any] | None:
    env_cfg = runtime_env_config_from_params(env_name, params)
    if env_cfg is None:
        return None
    return {
        "game": env_name,
        "algo": "dqn",
        "env_config": env_cfg,
        "env_fingerprint": _stable_fingerprint(env_cfg),
        "metadata_version": 1,
    }


def _compare_env_cfg(
    expected_env_name: str,
    metadata: dict[str, Any] | None,
    runtime_cfg: dict[str, Any],
    *,
    ignore_keys: set[str] | None = None,
) -> list[str]:
    if metadata is None:
        return ["Model has no environment metadata (older checkpoint format)."]
    model_game = metadata.get("game")
    if model_game is not None and str(model_game) != expected_env_name:
        return [f"Model metadata game mismatch: model={model_game} runtime={expected_env_name}"]
    env_cfg = metadata.get("env_config")
    if not isinstance(env_cfg, dict):
        return ["Model metadata missing env_config."]

    mismatches: list[str] = []
    ignored = ignore_keys or set()
    for key in sorted((set(env_cfg.keys()) | set(runtime_cfg.keys())) - ignored):
        mv = env_cfg.get(key)
        rv = runtime_cfg.get(key)
        if mv != rv:
            mismatches.append(f"  {key}: model={mv} runtime={rv}")
    return mismatches


def validate_model_env_from_params_or_raise(
    model_path: str | Path,
    env_name: str,
    params: dict[str, Any],
    *,
    allow_mismatch: bool = False,
    print_model_env: bool = False,
    ignore_keys: set[str] | None = None,
) -> None:
    runtime_cfg = runtime_env_config_from_params(env_name, params)
    if runtime_cfg is None:
        return
    metadata = read_model_metadata(model_path)
    if print_model_env:
        if metadata is None:
            print("Model metadata: <none>")
        else:
            print(
                "Model metadata env:",
                json.dumps(
                    {
                        "game": metadata.get("game"),
                        "algo": metadata.get("algo"),
                        "env_fingerprint": metadata.get("env_fingerprint"),
                        "env_config": metadata.get("env_config"),
                    },
                    sort_keys=True,
                ),
            )
    mismatches = _compare_env_cfg(env_name, metadata, runtime_cfg, ignore_keys=ignore_keys)
    if mismatches and not allow_mismatch:
        raise ValueError(
            "Model/runtime env mismatch detected.\n"
            + "\n".join(mismatches)
            + "\nUse matching eval/play config or pass --allow-env-mismatch to override."
        )
