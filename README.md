# Composite Kernel Ridge Regression

Python package for implementing Kernel Ridge Regression with multiple kernels.
- Parallel: CPU parallel via joblib, and GPU parallel via PyTorch tensor operations.
- Easy config
- Very tunable learning
- Supports K-Fold CV and target-stratification
- Easy to add multiple descriptors and kernels
- Multi-target learning with shared-kernel KRR/CKRR.
- Config validation

## Multi-target learning

Targets may be scalar or multi-output. A `.npy` target file can have shape
`(n_samples,)`, `(n_samples, n_targets)`, or a higher-dimensional shape whose
axes after the first sample axis are flattened into target columns. For `.npz`
targets, `Y_NAME` may be one key containing a target vector/matrix, or a list of
keys whose target columns are concatenated.

Multi-target runs use one shared composite input kernel and solve all target
columns in the same linear system:

```text
dual_coef = solve(K + alpha * I, Y)
Y_pred = K_eval @ dual_coef
```

The CPU and PyTorch cached-scoring paths both use this matrix right-hand side.
Search scores use sklearn's default multi-output aggregation for the selected
metric, and held-out reporting logs aggregate MAE/RMSE plus per-target MAE/RMSE.
When `STRATIFY = True`, multi-output targets are reduced to the first principal
direction of standardized target columns for split stratification.

## Nyström backend

Set `KRR_BACKEND = "nystrom"` in `config.py` to use an approximate streamed
Nyström KRR backend instead of exact dense KRR. The exact backend remains the
default.

Important knobs:

```python
KRR_BACKEND = "nystrom"
KRR_NYSTROM_N_LANDMARKS = 4096
KRR_NYSTROM_BATCH_SIZE = 2048
KRR_CACHED_SCORING_BACKEND = "numpy"  # or "pytorch" when torch is installed
```

The Nyström backend selects landmarks from each training fold, builds the small
landmark kernel, and streams `N x m` kernel features in row batches. During
hyperparameter search it pre-caches fold-local train-to-landmark,
validation-to-landmark, and landmark-to-landmark distances, so candidates reuse
the same `O(Nm)` cache instead of refitting through sklearn. It avoids
materializing the exact `N x N` training kernel.

## Validated single-run configuration

`krr_cli.py` is the stable machine interface for one run. It accepts JSON or
TOML, rejects unknown and inconsistent fields before launching `main.py`,
resolves relative paths from the config file, and writes an immutable
`resolved_config.json` with a stable SHA-256 into the output directory.

```bash
.venv/bin/python krr_cli.py template
.venv/bin/python krr_cli.py schema
.venv/bin/python krr_cli.py validate path/to/run.toml
.venv/bin/python krr_cli.py run path/to/run.toml
```

TOML may place fields at the top level or under `[krr]`. The schema and template
are intended for agents and editors, so scientific settings do not need to be
duplicated as dozens of `argparse` flags. The sibling `../experiments/`
repository owns study matrices, Newton/Stokes profiles, Slurm generation, and
study-specific compatibility code. Do not add named-study runners or Slurm
scripts here.

## Sweeps

Use `sweep_main.py` to run `main.py` over `N_TRAIN`, target, and seed
combinations without permanently editing `config.py`. Edit the constants at
the top of the file:

```python
N_TRAINS = [1000, 5000, 10000]
SEEDS = [1, 2, 3]
TARGETS = [
    {"name": "homo", "path": "sample/QM9/homo.npy"},
    {"name": "lumo", "path": "sample/QM9/lumo.npy"},
]
RUN_COMMAND = [".venv-cu124/bin/python", "main.py"]
RUN_ENV = {"CUDA_VISIBLE_DEVICES": "0,1,2,3"}
RESULTS_CSV = "sweep_results.csv"
```

Then run:

```bash
./sweep_main.py
```

The script restores the original `config.py` when it exits if
`RESTORE_CONFIG = True`. Set `DRY_RUN = True` to inspect generated runs without
launching training. After each run, it writes the accumulated `N_TRAIN`, seed,
target, output directory, return code, MAE, and RMSE rows to `RESULTS_CSV`.

`run_n_train_sweep.sh` is also retained for the existing lightweight workflow
that edits only `N_TRAIN` in `config.py`. It mutates that file and does not
restore it, so prefer `sweep_main.py` for multi-axis local sweeps and the sibling
experiment orchestrator for recorded cluster studies.
