from __future__ import annotations

import argparse
import csv
import json
import math
import re
import subprocess
import sys
from pathlib import Path
from typing import Any


DEFAULT_ENVS = [
    "2048",
    "snake",
    "fruit",
    "shooter",
    "tetris",
    "tetris_afterstate",
    "flappy",
    "flappy_tabular",
    "flappy_heuristic",
    "breakout",
    "pong",
    "match3",
    "pacman",
]

PRIMARY_METRIC_KEY = {
    "2048": "average_score",
    "snake": "average_score",
    "fruit": "average_score",
    "shooter": "average_score",
    "tetris": "average_score",
    "tetris_afterstate": "average_score",
    "flappy": "average_score",
    "flappy_tabular": "average_score",
    "flappy_heuristic": "average_score",
    "breakout": "average_score",
    "pong": "average_score",
    "match3": "average_score",
    "pacman": "average_score",
}


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml  # type: ignore
    except ImportError as exc:
        raise RuntimeError("PyYAML is required. Install with `pip install -r requirements.txt`.") from exc
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config file must contain a top-level mapping: {path}")
    return data


def _normalize_section(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError("Config section must be a mapping")
    return {str(k).replace('-', '_'): v for k, v in value.items()}


def _parse_csv_list(raw: str) -> list[str]:
    return [v.strip() for v in raw.split(",") if v.strip()]


def _parse_int_list(raw: str) -> list[int]:
    return [int(v) for v in _parse_csv_list(raw)]


def _slug(text: str) -> str:
    text = text.strip().lower()
    text = text.replace(">=", "ge_")
    text = text.replace("/", "_per_")
    text = text.replace("%", "pct")
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")


def _parse_scalar(raw: str) -> float | int | str:
    s = raw.strip()
    if s.endswith("%"):
        s_num = s[:-1].strip()
        try:
            return float(s_num)
        except ValueError:
            return s
    # Remove commas in numbers (if any)
    s_num = s.replace(",", "")
    try:
        if re.fullmatch(r"[-+]?\d+", s_num):
            return int(s_num)
        if re.fullmatch(r"[-+]?\d*\.\d+(e[-+]?\d+)?", s_num, flags=re.IGNORECASE) or re.fullmatch(r"[-+]?\d+e[-+]?\d+", s_num, flags=re.IGNORECASE):
            return float(s_num)
    except ValueError:
        pass
    return s


def parse_eval_output(stdout: str) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    lines = stdout.splitlines()
    skip_distribution = False
    for line in lines:
        if not line.strip():
            continue
        if line.startswith("Config:") or line.startswith("$"):
            continue
        if set(line.strip()) == {"-"}:
            continue
        if line.strip().endswith("distribution:"):
            skip_distribution = True
            continue
        if skip_distribution:
            # Skip indented histogram rows like "  128: 4"
            if line.startswith(" "):
                continue
            skip_distribution = False
        if ":" not in line:
            continue
        left, right = line.split(":", 1)
        key = _slug(left)
        if not key:
            continue
        # ignore histogram numeric-only keys if any slipped through
        if key.isdigit():
            continue
        metrics[key] = _parse_scalar(right)
    return metrics


def _is_number(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(float(x))


def _mean(values: list[float]) -> float:
    return float(sum(values) / max(len(values), 1))


def _std(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    m = _mean(values)
    return float((sum((v - m) ** 2 for v in values) / len(values)) ** 0.5)


def aggregate_seed_metrics(seed_runs: list[dict[str, Any]]) -> dict[str, Any]:
    keys: set[str] = set()
    for run in seed_runs:
        keys.update(run.keys())
    agg: dict[str, Any] = {}
    for key in sorted(keys):
        vals = [run[key] for run in seed_runs if key in run]
        num_vals = [float(v) for v in vals if _is_number(v)]
        if len(num_vals) == len(vals) and vals:
            agg[f"{key}__mean"] = _mean(num_vals)
            agg[f"{key}__std"] = _std(num_vals)
            agg[f"{key}__min"] = float(min(num_vals))
            agg[f"{key}__max"] = float(max(num_vals))
        else:
            # Keep first non-numeric textual value as reference
            if vals:
                agg[key] = vals[0]
    return agg


def resolve_envs(envs_arg: str) -> list[str]:
    if envs_arg.strip().lower() == "all":
        return list(DEFAULT_ENVS)
    envs = _parse_csv_list(envs_arg)
    unknown = [e for e in envs if e not in DEFAULT_ENVS]
    if unknown:
        raise ValueError(f"Unknown env(s): {unknown}")
    return envs


def config_and_model_status(config_dir: Path, env: str) -> tuple[Path, bool, str | None]:
    config_path = config_dir / f"{env}.yaml"
    if not config_path.exists():
        return config_path, False, "missing_config"
    cfg = _load_yaml(config_path)
    eval_cfg = _normalize_section(cfg.get("eval"))
    if "model" not in eval_cfg:
        return config_path, True, None
    model_path = Path(str(eval_cfg["model"]))
    return config_path, model_path.exists(), (None if model_path.exists() else "missing_model")


def run_eval(env: str, config_path: Path, episodes: int, seed: int, extra_args: list[str]) -> tuple[int, str, str]:
    cmd = [sys.executable, "-m", "src.eval", "--env", env, "--config", str(config_path), "--episodes", str(episodes), "--seed", str(seed), *extra_args]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    return proc.returncode, proc.stdout, proc.stderr


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for k in row.keys():
            if k not in seen:
                seen.add(k)
                fieldnames.append(k)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark all game evaluators across fixed seeds and produce leaderboard outputs")
    parser.add_argument("--envs", type=str, default="all", help="Comma-separated envs or 'all'")
    parser.add_argument("--episodes", type=int, default=100, help="Eval episodes per seed")
    parser.add_argument("--seeds", type=str, default="42,43,44", help="Comma-separated seeds")
    parser.add_argument("--config-dir", type=str, default="sample_configs")
    parser.add_argument("--out-json", type=str, default="benchmarks/leaderboard.json")
    parser.add_argument("--out-csv", type=str, default="benchmarks/leaderboard.csv")
    parser.add_argument("--skip-missing-models", action="store_true", default=True)
    parser.add_argument("--fail-on-missing-model", dest="skip_missing_models", action="store_false")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--pass-through", type=str, default="", help="Extra args passed to src.eval for every run (comma-split tokens not supported; provide raw shell-like tokens after -- is better)")
    args, passthrough_rest = parser.parse_known_args()

    envs = resolve_envs(args.envs)
    seeds = _parse_int_list(args.seeds)
    config_dir = Path(args.config_dir)
    out_json = Path(args.out_json)
    out_csv = Path(args.out_csv)

    extra_args: list[str] = []
    if args.pass_through.strip():
        extra_args.extend(args.pass_through.split())
    extra_args.extend(passthrough_rest)

    results: dict[str, Any] = {
        "episodes_per_seed": args.episodes,
        "seeds": seeds,
        "envs": {},
    }
    leaderboard_rows: list[dict[str, Any]] = []

    print(f"Benchmarking envs={envs} | episodes/seed={args.episodes} | seeds={seeds}")
    for env in envs:
        config_path, ok, reason = config_and_model_status(config_dir, env)
        if not ok:
            msg = f"skip {env}: {reason} ({config_path})"
            if args.skip_missing_models and reason == "missing_model":
                print(msg)
                results["envs"][env] = {"status": "skipped", "reason": reason, "config": str(config_path)}
                continue
            raise FileNotFoundError(msg)

        print(f"\n[{env}] config={config_path}")
        seed_runs: list[dict[str, Any]] = []
        raw_runs: list[dict[str, Any]] = []
        env_failed = False

        for seed in seeds:
            print(f"  seed={seed} ...", end="")
            code, stdout, stderr = run_eval(env, config_path, args.episodes, seed, extra_args)
            if code != 0:
                print(f" fail ({code})")
                raw_runs.append({"seed": seed, "returncode": code, "stdout": stdout, "stderr": stderr})
                if not args.continue_on_error:
                    sys.stderr.write(stdout)
                    sys.stderr.write(stderr)
                    raise RuntimeError(f"Benchmark eval failed for env={env} seed={seed}")
                env_failed = True
                continue
            metrics = parse_eval_output(stdout)
            raw_runs.append({"seed": seed, "returncode": code, "stdout": stdout, "stderr": stderr, "metrics": metrics})
            seed_runs.append(metrics)
            primary = PRIMARY_METRIC_KEY.get(env, "average_score")
            primary_val = metrics.get(primary)
            if _is_number(primary_val):
                print(f" ok ({primary}={float(primary_val):.3f})")
            else:
                print(" ok")

        if not seed_runs and env_failed:
            results["envs"][env] = {"status": "error", "config": str(config_path), "runs": raw_runs}
            continue
        if not seed_runs:
            results["envs"][env] = {"status": "no_runs", "config": str(config_path), "runs": raw_runs}
            continue

        agg = aggregate_seed_metrics(seed_runs)
        primary_key = PRIMARY_METRIC_KEY.get(env, "average_score")
        primary_mean = agg.get(f"{primary_key}__mean")
        steps_mean = agg.get("average_steps__mean")
        median_mean = agg.get("median_score__mean")
        row = {
            "env": env,
            "episodes_per_seed": args.episodes,
            "num_seeds": len(seed_runs),
            "primary_metric": primary_key,
            "primary_mean": primary_mean,
            "primary_std": agg.get(f"{primary_key}__std"),
            "median_score_mean": median_mean,
            "average_steps_mean": steps_mean,
            "status": "ok" if not env_failed else "partial",
        }
        # Keep all aggregated fields in CSV for later analysis.
        row.update(agg)
        leaderboard_rows.append(row)
        results["envs"][env] = {
            "status": row["status"],
            "config": str(config_path),
            "primary_metric": primary_key,
            "aggregate": agg,
            "runs": raw_runs,
        }

    # Sort leaderboard by primary metric mean descending where available.
    leaderboard_rows.sort(key=lambda r: (r.get("primary_mean") is None, -(float(r.get("primary_mean") or -1e18))))
    results["leaderboard"] = leaderboard_rows

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(results, indent=2), encoding="utf-8")
    write_csv(leaderboard_rows, out_csv)

    print("\nLeaderboard")
    print("-----------")
    print(f"{'env':14s} {'primary':>12s} {'std':>10s} {'median':>12s} {'avg_steps':>12s} {'status':>10s}")
    for row in leaderboard_rows:
        pm = row.get("primary_mean")
        ps = row.get("primary_std")
        med = row.get("median_score_mean")
        stp = row.get("average_steps_mean")
        def fmt(v: Any) -> str:
            return f"{float(v):.3f}" if _is_number(v) else "-"
        print(f"{row['env']:14s} {fmt(pm):>12s} {fmt(ps):>10s} {fmt(med):>12s} {fmt(stp):>12s} {str(row['status']):>10s}")

    print(f"\nSaved JSON: {out_json}")
    print(f"Saved CSV:  {out_csv}")


if __name__ == "__main__":
    main()
