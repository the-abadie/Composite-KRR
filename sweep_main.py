#!/usr/bin/env python3
from __future__ import annotations

import ast
import csv
import os
from pathlib import Path
import re
import subprocess
import sys


# Edit these values.
CONFIG_FILE = "config.py"

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

RUN_COMMAND = ["uv", "run", "python", "main.py"]
RUN_ENV = {
    # "CUDA_VISIBLE_DEVICES": "0,1,2,3",
}

RUN_NAME_TEMPLATE = "{base_run_name}_n{n_train}_{target_slug}_seed{seed}"
OUTPUT_ROOT = None  # Example: "sample/output/sweeps"
RESULTS_CSV = "sweep_results.csv"

CONTINUE_ON_ERROR = False
RESTORE_CONFIG = True
DRY_RUN = False


def main() -> int:
    config_path = Path(CONFIG_FILE)
    original_config = config_path.read_text()
    base_run_name = str(read_literal_assignment(original_config, "RUN_NAME", "sweep"))
    env = {**os.environ, **RUN_ENV}

    validate_settings()
    failures = []
    rows = []
    run_index = 0

    try:
        for n_train in N_TRAINS:
            for target in TARGETS:
                for seed in SEEDS:
                    run_index += 1
                    run_name = RUN_NAME_TEMPLATE.format(
                        base_run_name=base_run_name,
                        n_train=n_train,
                        seed=seed,
                        target_name=target["name"],
                        target_slug=slugify(target["name"]),
                        target_stem=Path(target["path"]).stem,
                    )
                    replacements = {
                        "SEED": repr(seed),
                        "N_TRAIN": repr(n_train),
                        "Y_NAME": repr(target["name"]),
                        "Y_PATH": repr(target["path"]),
                        "RUN_NAME": repr(run_name),
                        "TRAIN_VAL_SPLIT": "N_TRAIN/N_SAMPLES",
                    }
                    if OUTPUT_ROOT is not None:
                        output_dir = Path(OUTPUT_ROOT) / str(seed) / run_name
                        replacements["OUTPUT_DIR"] = repr(str(output_dir))

                    print(
                        f"[sweep] {run_index}: N_TRAIN={n_train} "
                        f"target={target['name']!r} seed={seed}"
                    )
                    print(f"[sweep] RUN_NAME={run_name!r}")

                    if DRY_RUN:
                        print("[sweep] dry run; not editing config or running main.py")
                        continue

                    updated_config = replace_assignments(original_config, replacements)
                    output_dir = evaluate_output_dir(updated_config)
                    config_path.write_text(updated_config)
                    result = subprocess.run(RUN_COMMAND, env=env)
                    row = build_result_row(
                        n_train=n_train,
                        target=target,
                        seed=seed,
                        run_name=run_name,
                        output_dir=output_dir,
                        returncode=result.returncode,
                    )
                    rows.append(row)
                    write_results_csv(rows)
                    print(
                        f"[sweep] metrics: MAE={row['mae']} RMSE={row['rmse']} "
                        f"status={row['status']}"
                    )

                    if result.returncode != 0:
                        failures.append((n_train, target["name"], seed, result.returncode))
                        if not CONTINUE_ON_ERROR:
                            return result.returncode
    finally:
        if RESTORE_CONFIG and not DRY_RUN:
            config_path.write_text(original_config)
            print(f"[sweep] restored {CONFIG_FILE}")

    if failures:
        for n_train, target_name, seed, returncode in failures:
            print(
                f"[sweep] failed: N_TRAIN={n_train} target={target_name!r} "
                f"seed={seed} exit={returncode}",
                file=sys.stderr,
            )
        return 1

    print(f"[sweep] completed {run_index} run(s)")
    return 0


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


def evaluate_output_dir(config_text: str) -> Path | None:
    namespace = {}
    exec(compile(config_text, CONFIG_FILE, "exec"), namespace)
    output_dir = namespace.get("OUTPUT_DIR")
    if output_dir is None:
        return None
    return Path(output_dir)


def replace_assignments(config_text: str, replacements: dict[str, str]) -> str:
    spans = assignment_line_spans(config_text)
    missing = sorted(name for name in replacements if name not in spans)
    if missing:
        raise SystemExit("Missing assignment(s) in config.py: " + ", ".join(missing))

    lines = config_text.splitlines(keepends=True)
    for name, value in sorted(
        replacements.items(),
        key=lambda item: spans[item[0]][0],
        reverse=True,
    ):
        start_line, end_line = spans[name]
        newline = "\n" if lines[end_line - 1].endswith("\n") else ""
        lines[start_line - 1 : end_line] = [f"{name} = {value}{newline}"]
    return "".join(lines)


def assignment_line_spans(config_text: str) -> dict[str, tuple[int, int]]:
    spans = {}
    for node in ast.parse(config_text).body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if isinstance(target, ast.Name):
            spans[target.id] = (node.lineno, node.end_lineno or node.lineno)
    return spans


def read_literal_assignment(config_text: str, name: str, default):
    for node in ast.parse(config_text).body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if isinstance(target, ast.Name) and target.id == name:
            try:
                return ast.literal_eval(node.value)
            except (SyntaxError, ValueError):
                return default
    return default


def slugify(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip()).strip("._-")
    return slug or "target"


if __name__ == "__main__":
    raise SystemExit(main())
