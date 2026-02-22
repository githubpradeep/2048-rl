from __future__ import annotations

import argparse
import importlib
import json
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

PLAY_MODULES = {
    "2048": "src.plugins.playback.play_agent",
    "breakout": "src.plugins.playback.play_breakout_agent",
    "flappy": "src.plugins.playback.play_flappy_agent",
    "flappy_heuristic": "src.plugins.playback.play_flappy_heuristic",
    "flappy_tabular": "src.plugins.playback.play_flappy_tabular",
    "fruit": "src.plugins.playback.play_fruit_agent",
    "match3": "src.plugins.playback.play_match3_agent",
    "pong": "src.plugins.playback.play_pong_agent",
    "shooter": "src.plugins.playback.play_shooter_agent",
    "snake": "src.plugins.playback.play_snake_agent",
    "tetris": "src.plugins.playback.play_tetris_agent",
    "tetris_afterstate": "src.plugins.playback.play_tetris_afterstate_agent",
}


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml  # type: ignore
    except ImportError as exc:
        raise RuntimeError("PyYAML is required. Install with `pip install -r requirements.txt`.") from exc
    raw = path.read_text(encoding="utf-8")
    data = yaml.safe_load(raw) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config file must contain a top-level mapping: {path}")
    return data


def _normalize_section(value: Any, section_name: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"Config section `{section_name}` must be a mapping/object")
    return {str(k).replace('-', '_'): v for k, v in value.items()}


def _build_args_from_mapping(params: dict[str, Any]) -> list[str]:
    argv: list[str] = []
    for key, value in params.items():
        if value is None:
            continue
        flag = f"--{str(key).replace('_', '-')}"
        if isinstance(value, bool):
            if value:
                argv.append(flag)
            continue
        if isinstance(value, (list, tuple)):
            argv.extend([flag, ",".join(str(v) for v in value)])
            continue
        argv.extend([flag, str(value)])
    return argv


def _parse_overrides(tokens: list[str]) -> dict[str, Any]:
    overrides: dict[str, Any] = {}
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if not tok.startswith("--"):
            raise ValueError(f"Unexpected positional override token: {tok}")
        key = tok[2:].replace("-", "_")
        if key.startswith("no_"):
            overrides[key[3:]] = False
            i += 1
            continue
        if i + 1 < len(tokens) and not tokens[i + 1].startswith("--"):
            raw = tokens[i + 1]
            low = raw.lower()
            overrides[key] = (low == "true") if low in {"true", "false"} else raw
            i += 2
            continue
        overrides[key] = True
        i += 1
    return overrides


def _fallback_play_from_eval(eval_cfg: dict[str, Any]) -> dict[str, Any]:
    if not eval_cfg:
        return {}
    return {k: v for k, v in eval_cfg.items() if k not in {"episodes"}}


@contextmanager
def _patched_argv(argv: list[str]) -> Iterator[None]:
    old = sys.argv[:]
    sys.argv = argv
    try:
        yield
    finally:
        sys.argv = old


def _print_json_params(params: dict[str, Any]) -> None:
    print(json.dumps(params, sort_keys=True, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Unified generic player (YAML-driven)")
    parser.add_argument("--env", required=True, choices=sorted(PLAY_MODULES))
    parser.add_argument("--config", type=str, default=None, help="YAML config file (defaults to sample_configs/<env>.yaml)")
    parser.add_argument("--dry-run", action="store_true", help="Print merged play params and target module, then exit")
    args, rest = parser.parse_known_args()

    config_path = Path(args.config) if args.config else Path("sample_configs") / f"{args.env}.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    config = _load_yaml(config_path)
    declared_env = config.get("env")
    if declared_env is not None and str(declared_env) != args.env:
        raise ValueError(f"Config env mismatch: config has env={declared_env!r}, CLI requested env={args.env!r}")

    common = _normalize_section(config.get("common"), "common")
    eval_cfg = _normalize_section(config.get("eval"), "eval")
    play_cfg = _normalize_section(config.get("play"), "play")
    # Play should inherit env/model compatibility fields from eval (e.g. state-mode, grid-size),
    # while allowing the play section to override display/runtime settings.
    base_play = _fallback_play_from_eval(eval_cfg)
    merged_play = {**base_play, **play_cfg}
    if not merged_play:
        raise ValueError(f"Config has neither `play` nor usable `eval` section: {config_path}")

    params = {**common, **merged_play}
    if rest:
        params.update(_parse_overrides(rest))

    module_name = PLAY_MODULES[args.env]
    argv_tail = _build_args_from_mapping(params)
    cmd_preview = [sys.executable, "-m", module_name, *argv_tail]

    print(f"Config: {config_path}")
    print("$", " ".join(cmd_preview))
    if args.dry_run:
        _print_json_params(params)
        return

    module = importlib.import_module(module_name)
    if not hasattr(module, "main"):
        raise RuntimeError(f"Play module {module_name} has no main()")
    with _patched_argv([module_name, *argv_tail]):
        module.main()


if __name__ == "__main__":
    main()
