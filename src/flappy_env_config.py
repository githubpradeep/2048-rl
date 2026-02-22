from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .games.flappy import FlappyConfig


FLAPPY_ENV_PRESETS: dict[str, dict[str, float | int | None]] = {
    "easy": {
        "gap_size": 34.0,
        "pipe_speed": 1.2,
        "pipe_spacing": 46.0,
        "initial_pipe_offset": 24.0,
        "floor_height": 10.0,
        "max_gap_delta": 6.0,
        "gravity": 0.18,
        "flap_velocity": -2.2,
        "max_velocity": 3.0,
    },
    "standard": {
        "gap_size": 30.0,
        "pipe_speed": 1.3,
        "pipe_spacing": 44.0,
        "initial_pipe_offset": 20.0,
        "floor_height": 10.0,
        "max_gap_delta": 8.0,
        "gravity": 0.18,
        "flap_velocity": -2.2,
        "max_velocity": 3.0,
    },
    "hard": {
        "gap_size": 26.0,
        "pipe_speed": 1.45,
        "pipe_spacing": 42.0,
        "initial_pipe_offset": 20.0,
        "floor_height": 10.0,
        "max_gap_delta": 10.0,
        "gravity": 0.18,
        "flap_velocity": -2.2,
        "max_velocity": 3.0,
    },
}


_ARG_TO_CONFIG_FIELD = {
    "width": "width",
    "height": "height",
    "pipe_gap": "gap_size",
    "pipe_speed": "pipe_speed",
    "pipe_spacing": "pipe_spacing",
    "initial_pipe_offset": "initial_pipe_offset",
    "floor_height": "floor_height",
    "max_gap_delta": "max_gap_delta",
    "gravity": "gravity",
    "flap_velocity": "flap_velocity",
}

_PRESET_TO_ARG_FIELD = {
    "gap_size": "pipe_gap",
    "pipe_speed": "pipe_speed",
    "pipe_spacing": "pipe_spacing",
    "initial_pipe_offset": "initial_pipe_offset",
    "floor_height": "floor_height",
    "max_gap_delta": "max_gap_delta",
    "gravity": "gravity",
    "flap_velocity": "flap_velocity",
}

_FINGERPRINT_KEYS = (
    "width",
    "height",
    "gap_size",
    "pipe_speed",
    "pipe_spacing",
    "initial_pipe_offset",
    "floor_height",
    "max_gap_delta",
    "gravity",
    "flap_velocity",
    "max_velocity",
)


def add_flappy_preset_arg(parser: Any) -> None:
    parser.add_argument(
        "--env-preset",
        choices=sorted(FLAPPY_ENV_PRESETS.keys()),
        default=None,
        help="Apply a named Flappy environment preset (overrides Flappy env flags in this script).",
    )


def add_env_mismatch_args(parser: Any) -> None:
    parser.add_argument(
        "--allow-env-mismatch",
        action="store_true",
        help="Allow eval/play even if model metadata env config differs from runtime env flags.",
    )
    parser.add_argument(
        "--print-model-env",
        action="store_true",
        help="Print model env metadata (if present) before evaluation/play.",
    )


def apply_flappy_preset(args: Any) -> str | None:
    preset = getattr(args, "env_preset", None)
    if not preset:
        return None
    overrides = FLAPPY_ENV_PRESETS[preset]
    for cfg_key, value in overrides.items():
        arg_key = _PRESET_TO_ARG_FIELD.get(cfg_key)
        if arg_key is None:
            continue
        setattr(args, arg_key, value)
    return str(preset)


def build_flappy_config_from_args(args: Any, include_rewards: bool = False) -> FlappyConfig:
    kwargs: dict[str, Any] = {
        "width": int(args.width),
        "height": int(args.height),
        "gap_size": float(args.pipe_gap),
        "pipe_speed": float(args.pipe_speed),
        "pipe_spacing": float(args.pipe_spacing),
        "initial_pipe_offset": float(args.initial_pipe_offset),
        "floor_height": float(args.floor_height),
        "max_gap_delta": None if args.max_gap_delta is None else float(args.max_gap_delta),
        "gravity": float(args.gravity),
        "flap_velocity": float(args.flap_velocity),
        "max_steps": int(args.max_steps),
    }
    if include_rewards:
        if hasattr(args, "step_reward"):
            kwargs["step_reward"] = float(args.step_reward)
        if hasattr(args, "pass_reward"):
            kwargs["pass_reward"] = float(args.pass_reward)
        if hasattr(args, "crash_penalty"):
            kwargs["crash_penalty"] = float(args.crash_penalty)
    return FlappyConfig(**kwargs)


def flappy_env_config_payload(config: FlappyConfig) -> dict[str, float | int | None]:
    return {
        "width": int(config.width),
        "height": int(config.height),
        "gap_size": float(config.gap_size),
        "pipe_speed": float(config.pipe_speed),
        "pipe_spacing": float(config.pipe_spacing),
        "initial_pipe_offset": float(config.initial_pipe_offset),
        "floor_height": float(config.floor_height),
        "max_gap_delta": None if config.max_gap_delta is None else float(config.max_gap_delta),
        "gravity": float(config.gravity),
        "flap_velocity": float(config.flap_velocity),
        "max_velocity": float(config.max_velocity),
    }


def flappy_env_fingerprint(payload: dict[str, Any]) -> str:
    stable = {k: payload.get(k) for k in _FINGERPRINT_KEYS}
    raw = json.dumps(stable, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def flappy_model_metadata(config: FlappyConfig, env_preset: str | None, algo: str) -> dict[str, Any]:
    env_payload = flappy_env_config_payload(config)
    return {
        "game": "flappy",
        "algo": str(algo),
        "env_preset": env_preset,
        "env_config": env_payload,
        "env_fingerprint": flappy_env_fingerprint(env_payload),
        "metadata_version": 1,
    }


def _read_model_json(path: str | Path) -> dict[str, Any]:
    raw = Path(path).read_text(encoding="utf-8")
    payload = json.loads(raw)
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
    if metadata is None:
        return None
    if not isinstance(metadata, dict):
        return None
    return metadata


def format_env_mismatch_report(mismatches: list[str], metadata: dict[str, Any] | None) -> str:
    lines = ["Model/runtime Flappy env mismatch detected."]
    if metadata and "env_preset" in metadata:
        lines.append(f"Model preset: {metadata.get('env_preset')}")
    lines.extend(mismatches)
    lines.append("Use matching env flags or pass --allow-env-mismatch to override.")
    return "\n".join(lines)


def compare_model_env_metadata(
    metadata: dict[str, Any] | None,
    runtime_config: FlappyConfig,
) -> list[str]:
    if metadata is None:
        return ["Model has no Flappy metadata (older checkpoint format)."]
    env_cfg = metadata.get("env_config")
    if not isinstance(env_cfg, dict):
        return ["Model metadata missing env_config."]

    runtime_payload = flappy_env_config_payload(runtime_config)
    mismatches: list[str] = []
    for key in _FINGERPRINT_KEYS:
        mv = env_cfg.get(key)
        rv = runtime_payload.get(key)
        if mv != rv:
            mismatches.append(f"  {key}: model={mv} runtime={rv}")
    return mismatches


def validate_model_env_or_raise(
    model_path: str | Path,
    runtime_config: FlappyConfig,
    *,
    allow_mismatch: bool = False,
    print_model_env: bool = False,
) -> None:
    metadata = read_model_metadata(model_path)
    if print_model_env:
        if metadata is None:
            print("Model metadata: <none>")
        else:
            print(
                "Model metadata env:",
                json.dumps(
                    {
                        "env_preset": metadata.get("env_preset"),
                        "env_fingerprint": metadata.get("env_fingerprint"),
                        "env_config": metadata.get("env_config"),
                    },
                    sort_keys=True,
                ),
            )
    mismatches = compare_model_env_metadata(metadata, runtime_config)
    if mismatches and not allow_mismatch:
        raise ValueError(format_env_mismatch_report(mismatches, metadata))

