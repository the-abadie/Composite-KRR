#!/usr/bin/env python3
from __future__ import annotations

import csv
import os
from pathlib import Path
import re
import subprocess
import sys
import shlex

from run_config import KRRConfig, write_immutable


# Edit these values.
CONFIG_FILE = Path(__file__).with_name("config.json")
CONFIG_OUTPUT_DIR = "sweep_configs"

N_TRAINS = [
    100,
    250,
    500,
]

SEEDS = [
    1,
    2,
    3,
]

TARGETS = [
    {
        "name": "atomization energy",
        "path": "sample/QM7/atomization_energy.npy",
    },
    # {"name": "homo", "path": "sample/QM9/homo.npy"},
    # {"name": "lumo", "path": "sample/QM9/lumo.npy"},
]

RUN_COMMAND = [sys.executable, str(Path(__file__).with_name("main.py"))]
RUN_ENV = {
    # "CUDA_VISIBLE_DEVICES": "0,1,2,3",
}

RUN_NAME_TEMPLATE = "{base_run_name}_n{n_train}_{target_slug}_seed{seed}"
OUTPUT_ROOT = None  # Example: "sample/output/sweeps"
RESULTS_CSV = "sweep_results.csv"

CONTINUE_ON_ERROR = False
DRY_RUN = False


def main() -> int:
    base = KRRConfig.load(CONFIG_FILE)
    env = {**os.environ, **RUN_ENV}
    validate_settings()
    failures = []
    rows = []
    run_index = 0
    for n_train in N_TRAINS:
        for target in TARGETS:
            for seed in SEEDS:
                run_index += 1
                run_name = RUN_NAME_TEMPLATE.format(
                    base_run_name=base.spec.run_name, n_train=n_train, seed=seed,
                    target_name=target["name"], target_slug=slugify(target["name"]),
                    target_stem=Path(target["path"]).stem,
                )
                output_root = Path(OUTPUT_ROOT) if OUTPUT_ROOT else Path(base.spec.output.directory).parent
                config = base.with_overrides({
                    "seed": seed, "split.n_train": n_train, "target.name": target["name"],
                    "target.path": target["path"], "run_name": run_name,
                    "output.directory": str(output_root / str(seed) / run_name),
                    "output.overwrite": False,
                }).resolved()
                config.validate(check_paths=not DRY_RUN)
                config_path = Path(CONFIG_OUTPUT_DIR).resolve() / f"{slugify(run_name)}.json"
                write_immutable(config_path, config.canonical_json())
                command = [*RUN_COMMAND, "--config", str(config_path)]
                print(f"[sweep] {run_index}: {shlex.join(command)}")
                if DRY_RUN:
                    continue
                result = subprocess.run(command, env=env)
                row = build_result_row(n_train=n_train, target=target, seed=seed,
                    run_name=run_name, output_dir=Path(config.spec.output.directory),
                    returncode=result.returncode)
                rows.append(row)
                write_results_csv(rows)
                print(f"[sweep] metrics: MAE={row['mae']} RMSE={row['rmse']} status={row['status']}")
                if result.returncode:
                    failures.append((run_name, result.returncode))
                    if not CONTINUE_ON_ERROR:
                        return result.returncode
    for run_name, returncode in failures:
        print(f"[sweep] failed: {run_name} exit={returncode}", file=sys.stderr)
    print(f"[sweep] prepared {run_index} run(s)" if DRY_RUN else f"[sweep] completed {run_index} run(s)")
    return int(bool(failures))


def validate_settings() -> None:
    if not N_TRAINS:
        raise SystemExit("N_TRAINS is empty.")
    if not SEEDS:
        raise SystemExit("SEEDS is empty.")
    if not TARGETS:
        raise SystemExit("TARGETS is empty.")
    if not RUN_COMMAND:
        raise SystemExit("RUN_COMMAND is empty.")
    for n_train in N_TRAINS:
        if type(n_train) is not int or n_train <= 0:
            raise SystemExit(f"Invalid N_TRAIN value: {n_train!r}")
    for seed in SEEDS:
        if type(seed) is not int:
            raise SystemExit(f"Invalid SEED value: {seed!r}")
    for target in TARGETS:
        if not target.get("name") or not target.get("path"):
            raise SystemExit(f"Invalid target entry: {target!r}")
    if not RESULTS_CSV:
        raise SystemExit("RESULTS_CSV is empty.")


def build_result_row(
    *,
    n_train: int,
    target: dict[str, str],
    seed: int,
    run_name: str,
    output_dir: Path | None,
    returncode: int,
) -> dict[str, object]:
    mae, rmse = collect_metrics(output_dir)
    status = "ok" if returncode == 0 else "failed"
    return {
        "n_train": n_train,
        "seed": seed,
        "target_name": target["name"],
        "target_path": target["path"],
        "run_name": run_name,
        "output_dir": "" if output_dir is None else str(output_dir),
        "returncode": returncode,
        "status": status,
        "mae": "" if mae is None else mae,
        "rmse": "" if rmse is None else rmse,
    }


def collect_metrics(output_dir: Path | None) -> tuple[float | None, float | None]:
    if output_dir is None:
        return None, None

    log_metrics = collect_metrics_from_log(output_dir / "run.log")
    if log_metrics != (None, None):
        return log_metrics

    y_true_path = output_dir / "y_true.npy"
    y_pred_path = output_dir / "y_predictions.npy"
    if not y_true_path.exists() or not y_pred_path.exists():
        return None, None

    import numpy as np

    y_true = np.asarray(np.load(y_true_path), dtype=float)
    y_pred = np.asarray(np.load(y_pred_path), dtype=float)
    diff = y_pred.reshape(y_true.shape) - y_true
    mae = float(np.mean(np.abs(diff)))
    rmse = float(np.sqrt(np.mean(diff * diff)))
    return mae, rmse


def collect_metrics_from_log(log_path: Path) -> tuple[float | None, float | None]:
    if not log_path.exists():
        return None, None

    mae = None
    rmse = None
    for line in log_path.read_text(errors="replace").splitlines():
        mae_match = re.search(r"Held-out test MAE\s*:\s*([-+0-9.eE]+)", line)
        if mae_match:
            mae = float(mae_match.group(1))
        rmse_match = re.search(r"Held-out test RMSE\s*:\s*([-+0-9.eE]+)", line)
        if rmse_match:
            rmse = float(rmse_match.group(1))
    return mae, rmse


def write_results_csv(rows: list[dict[str, object]]) -> None:
    fieldnames = [
        "n_train",
        "seed",
        "target_name",
        "target_path",
        "run_name",
        "output_dir",
        "returncode",
        "status",
        "mae",
        "rmse",
    ]
    with Path(RESULTS_CSV).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def slugify(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip()).strip("._-")
    return slug or "target"


if __name__ == "__main__":
    raise SystemExit(main())
