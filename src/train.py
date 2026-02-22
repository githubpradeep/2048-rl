from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .generic_dqn_train import train_dqn

SUPPORTED_ENVS = (
    "2048",
    "breakout",
    "flappy",
    "flappy_tabular",
    "fruit",
    "match3",
    "pacman",
    "pong",
    "shooter",
    "snake",
    "tetris",
    "tetris_afterstate",
)
DQN_ENVS = {"2048", "breakout", "flappy", "fruit", "match3", "pacman", "pong", "shooter", "snake", "tetris"}


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml  # type: ignore
    except ImportError as exc:
        raise RuntimeError("PyYAML is required. Install with `pip install -r requirements.txt`.") from exc
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config file must contain a top-level mapping: {path}")
    return data


def _normalize_section(value: Any, section_name: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"Config section `{section_name}` must be a mapping/object")
    return {str(k).replace('-', '_'): v for k, v in value.items()}


def _parse_overrides(tokens: list[str]) -> dict[str, Any]:
    overrides: dict[str, Any] = {}
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if not tok.startswith("--"):
            raise ValueError(f"Unexpected positional override token: {tok}")
        key = tok[2:].replace('-', '_')
        if key.startswith("no_"):
            overrides[key[3:]] = False
            i += 1
            continue
        if i + 1 < len(tokens) and not tokens[i + 1].startswith("--"):
            raw = tokens[i + 1]
            low = raw.lower()
            if low in {"true", "false"}:
                overrides[key] = (low == "true")
            else:
                overrides[key] = raw
            i += 2
            continue
        overrides[key] = True
        i += 1
    return overrides


def _print_json_params(params: dict[str, Any]) -> None:
    print(json.dumps(params, sort_keys=True, indent=2))


def _run_flappy_tabular(params: dict[str, Any]) -> None:
    from .plugins.flappy.workflow import main as workflow_main
    import sys
    from contextlib import contextmanager

    def build_args(mapping: dict[str, Any]) -> list[str]:
        argv: list[str] = []
        for k, v in mapping.items():
            if v is None:
                continue
            flag = f"--{k.replace('_','-')}"
            if isinstance(v, bool):
                if v:
                    argv.append(flag)
                continue
            if isinstance(v, (list, tuple)):
                argv.extend([flag, ",".join(str(x) for x in v)])
                continue
            argv.extend([flag, str(v)])
        return argv

    @contextmanager
    def patched(argv: list[str]):
        old = sys.argv[:]
        sys.argv = argv
        try:
            yield
        finally:
            sys.argv = old

    argv = ["src.flappy_tabular_workflow", *build_args(params)]
    with patched(argv):
        workflow_main()


def _run_tetris_afterstate(params: dict[str, Any]) -> None:
    from .plugins.tetris.afterstate_workflow import main as workflow_main
    import sys
    from contextlib import contextmanager

    def build_args(mapping: dict[str, Any]) -> list[str]:
        argv: list[str] = []
        for k, v in mapping.items():
            if v is None:
                continue
            flag = f"--{k.replace('_','-')}"
            if isinstance(v, bool):
                if v:
                    argv.append(flag)
                continue
            if isinstance(v, (list, tuple)):
                argv.extend([flag, ",".join(str(x) for x in v)])
                continue
            argv.extend([flag, str(v)])
        return argv

    @contextmanager
    def patched(argv: list[str]):
        old = sys.argv[:]
        sys.argv = argv
        try:
            yield
        finally:
            sys.argv = old

    argv = ["src.tetris_afterstate_workflow", *build_args(params)]
    with patched(argv):
        workflow_main()


def main() -> None:
    parser = argparse.ArgumentParser(description="Unified generic trainer (YAML-driven)")
    parser.add_argument("--env", required=True, choices=sorted(SUPPORTED_ENVS))
    parser.add_argument("--config", type=str, default=None, help="YAML config file (defaults to sample_configs/<env>.yaml)")
    parser.add_argument("--dry-run", action="store_true", help="Print merged train params and exit")
    args, rest = parser.parse_known_args()

    config_path = Path(args.config) if args.config else Path("sample_configs") / f"{args.env}.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    config = _load_yaml(config_path)
    declared_env = config.get("env")
    if declared_env is not None and str(declared_env) != args.env:
        raise ValueError(f"Config env mismatch: config has env={declared_env!r}, CLI requested env={args.env!r}")

    common = _normalize_section(config.get("common"), "common")
    train_cfg = _normalize_section(config.get("train"), "train")
    if not train_cfg:
        raise ValueError(f"Config has no `train` section: {config_path}")
    params = {**common, **train_cfg}
    if rest:
        params.update(_parse_overrides(rest))

    print(f"Config: {config_path}")
    if args.dry_run:
        _print_json_params(params)
        return

    if args.env in DQN_ENVS:
        train_dqn(args.env, params)
    elif args.env == "flappy_tabular":
        _run_flappy_tabular(params)
    elif args.env == "tetris_afterstate":
        _run_tetris_afterstate(params)
    else:
        raise ValueError(f"Unsupported env: {args.env}")


if __name__ == "__main__":
    main()
