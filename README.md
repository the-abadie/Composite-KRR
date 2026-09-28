# Composite Kernel Ridge Regression

Python package for implementing Kernel Ridge Regression with multiple kernels.
- Parallel: CPU parallel via joblib, and GPU parallel via PyTorch tensor operations.
- Easy config
- Very tunable learning
- Supports K-Fold CV and target-stratification
- Easy to add multiple descriptors and kernels
- Multi-target learning with shared-kernel KRR/CKRR.
- Config validation

## Additive and multiplicative kernels

Each descriptor defines one base kernel with its own bandwidth (`gamma`) and
RBF or Laplacian type. Set `KRR_KERNEL_PRODUCTS` to add elementwise (Hadamard)
products of these kernels. Indices are **zero-based**, in `X_NAMES`/`X_PATHS`
order. The default `[]` preserves additive-only behavior.

For four loaded descriptors, this JSON/TOML-compatible setting adds two products:

```python
KRR_KERNEL = ["rbf", "laplacian", "rbf", "laplacian"]
KRR_KERNEL_PRODUCTS = [[0, 1], [2, 3]]
```

The resulting kernel is

```text
K = w0*K0 + w1*K1 + w2*K2 + w3*K3 + w4*(K0*K1) + w5*(K2*K3)
```

Products use the **unweighted** base kernels. Their weights are independent of
the additive weights: setting `w0 = 0` does not disable `K0*K1`. Factors reuse
the base kernels' bandwidths, kernel types, and fold-fitted preprocessing.
Products may have more than two factors (`[0, 1, 2]`) or repeated factors
(`[0, 0]` for `K0**2`). Invalid indices, single-factor products, and duplicate
products (including reordered duplicates) are rejected.

For direct Python use, both `CompositeKRREstimator` and
`CompositeNystromKRREstimator` accept:

```python
kernel_types=["rbf", "laplacian"]
kernel_products=[[0, 1]]
gammas=[0.1, 0.2]                 # one per base descriptor
kernel_weights=[0.2, 0.3, 0.5]    # base weights first, then product weights
```

`kernel_weights=None` gives every term weight 1; `normalize_kernel_weights=True`
normalizes **all** additive and product weights together. Weights must be finite
and non-negative. The low-level `CompositeKRR` and `CompositeTorchKRR` classes
keep additive weights in `KernelComponent` and accept `kernel_products` plus
`product_weights` (defaulting to 1 per product).

All random-search stages and Bayesian search tune one weight per term and one
gamma per base descriptor. Stage 1 uses equal weights over all terms. Saved
`kernel_weights.npy` follows the ordering above, while `gammas.npy` remains in
descriptor order. The product definition is retained in the saved configuration.
Contribution analysis still measures descriptor removal: dropping a descriptor
also removes every product containing it, and retained products are remapped
to the subset's descriptor indices.

Exact fitting, prediction, cached CV, and streamed Nyström support products on
NumPy and optional PyTorch backends. Select `KRR_CACHED_SCORING_BACKEND =
"pytorch"` and `KRR_PYTORCH_DEVICE = "cuda"` for GPU execution with a suitable
Torch installation. Products need no additional descriptor loads or distance
matrices. Each base kernel is exponentiated once per matrix assembly/candidate
and reused; only factors needed by products are retained. This trades temporary
kernel memory for reuse. Prediction/Nyström row batches and Torch candidate
batches bound that temporary memory. Exact training still uses dense matrices.

Run the regression checks with `.venv/bin/python -m unittest discover -v`.
The optional Torch parity test runs on CPU and also CUDA when available; it is
skipped when Torch is absent.

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
