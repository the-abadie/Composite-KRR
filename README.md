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
RBF or Laplacian type. JSON `kernel_products` uses descriptor IDs, so reordering
the descriptor list does not silently change the product definitions. The default
`[]` preserves additive-only behavior. For four descriptors named `a`, `b`, `c`,
and `d`, set:

```json
"kernel_products": [["a", "b"], ["c", "d"]]
```

The resulting kernel is

```text
K = w0*K0 + w1*K1 + w2*K2 + w3*K3 + w4*(K0*K1) + w5*(K2*K3)
```

Products use the **unweighted** base kernels. Their weights are independent of
the additive weights: setting `w0 = 0` does not disable `K0*K1`. Factors reuse
the base kernels' bandwidths, kernel types, and fold-fitted preprocessing.
Products may have more than two factors (`["a", "b", "c"]`) or repeated factors
(`["a", "a"]` for `Ka**2`). Unknown descriptor IDs, single-factor products, and duplicate
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
NumPy and optional PyTorch backends. Select `execution.backend = "pytorch"` and
`execution.torch.device = "cuda"` for GPU execution with a suitable
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
targets, `target.name` may be one key containing a target vector/matrix, or a list of
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
When `split.stratify = true`, multi-output targets are reduced to the first principal
direction of standardized target columns for split stratification.

## Nyström backend

Set `model.backend` to `"nystrom"` in your JSON config to use an approximate streamed
Nyström KRR backend instead of exact dense KRR. The exact backend remains the
default.

Important knobs:

```json
"model": {
  "backend": "nystrom",
  "nystrom": {"n_landmarks": 4096, "batch_size": 2048}
},
"execution": {"backend": "numpy"}
```

The Nyström backend selects landmarks from each training fold, builds the small
landmark kernel, and streams `N x m` kernel features in row batches. During
hyperparameter search it pre-caches fold-local train-to-landmark,
validation-to-landmark, and landmark-to-landmark distances, so candidates reuse
the same `O(Nm)` cache instead of refitting through sklearn. It avoids
materializing the exact `N x N` training kernel.

## JSON configuration and CLI

Run a saved configuration directly:

```bash
.venv/bin/python main.py --config /absolute/path/to/run.json
```

Without `--config`, `main.py` reads the `config.json` beside the script. Importing
`main` does not start a run. `config.py` is retained only as historical reference;
editing it has no effect. Python configurations are never executed by the loader.

A small complete configuration is:

```json
{
  "schema_version": 1,
  "seed": 12,
  "run_name": "mixed-kernels-n1000",
  "descriptors": [
    {"id": "overlap", "path": "../data/overlap.npy", "kernel": "rbf"},
    {"id": "coulomb", "path": "../data/coulomb.npy", "kernel": "laplacian"}
  ],
  "target": {"path": "../data/energy.npy", "name": "energy"},
  "kernel_products": [["overlap", "coulomb"]],
  "split": {"mode": "random", "n_train": 1000, "n_folds": 5},
  "execution": {"backend": "numpy", "n_jobs": 8, "distance_cache": {"n_jobs": 8}},
  "output": {"directory": "../outputs/seed12"}
}
```

Defaults, strict types, validation, and generated JSON Schema have one source:
`config_schema.py`. Unknown fields at every level, non-finite numbers, duplicate
JSON keys, invalid bounds, and inconsistent product/split settings are rejected.
Error locations identify the offending field. `krr_cli.py` provides agent tools:

```bash
.venv/bin/python krr_cli.py template --output run.json
.venv/bin/python krr_cli.py schema --output config.schema.json
.venv/bin/python krr_cli.py validate run.json
.venv/bin/python krr_cli.py validate run.json --no-check-paths
.venv/bin/python krr_cli.py resolve run.json --set seed=123 \
  --set split.n_train=2000 --set 'output.directory="../outputs/seed123"' \
  --output configs/seed123.json
.venv/bin/python main.py --config configs/seed123.json
```

`resolve` expands defaults and writes a new immutable JSON file; rerunning with
identical content is allowed, replacing different content is refused. Override
values are JSON. Relative input **and output** paths resolve from the original
configuration's directory, including overrides, independently of the shell's
working directory. A resolved file uses absolute paths so moving it is safe.
For cluster portability, paths may start with `@workspace/`, `@krr/`,
`@experiments/`, or `@reports/`; `resolve --portable-workspace /path/to/ml-dev`
retains workspace tokens in the saved file. No implicit environment-variable
expansion or inheritance is performed.

Specify exactly one of `split.n_train` (training plus CV samples) and
`split.train_fraction`. `split.n_samples` defaults to all available samples.
An explicit sample pool larger than the dataset is an error. Setting either
split field with `--set` clears the other inherited field. For predefined splits,
use only `mode: "predefined"`, `train_indices`, `validation_folds`, and
`test_indices`; validation folds may be an NPY array or NPZ with `fold0`, `fold1`,
etc. Cross-field checks also validate bounds, kernel names, and descriptor IDs.

Each run records `resolved_config.json` and `resolved_config.sha256`, including
the actual random seed when `seed` was null. The output directory must be empty
unless `output.overwrite` is true; a different resolved configuration is always
refused. A `.run.lock` prevents simultaneous jobs writing the same directory.
If a process is killed, confirm it has stopped before manually removing its lock.
Use a distinct output directory for each independent job.

TOML remains supported, including legacy `[krr]` wrappers. Existing flat uppercase
JSON/TOML fields remain readable and can be migrated:

```bash
.venv/bin/python krr_cli.py migrate legacy.toml --output migrated.json
```

Legacy product indices are translated into descriptor names. Legacy `N_TRAIN`
and `TRAIN_VAL_SPLIT` must agree when both are supplied; an inconsistent pair is
an error. New configurations should use the structured schema. The Python API
is `run_config.run(KRRConfig.load(path))`; no temporary Python config or global
configuration mutation is involved.

## Sweeps and HPC jobs

Each Slurm job can invoke `main.py --config /path/to/its-run.json`. Keep scientific
settings in these files and scheduler resources in the sibling `experiments/`
repository. That repository owns study matrices, immutable expanded plans,
Newton/Stokes profiles, Slurm rendering, preflight, and submission. It validates
and freezes each KRR config before submission. Cluster submission still requires
a fresh matching preflight and explicit confirmation.

Within the `ml-dev` workspace, follow the root [AGENTS.md](../AGENTS.md), the
[experiment lifecycle](../experiments/README.md#study-lifecycle), and the target
cluster's [Newton](../docs/NEWTON.md) or [Stokes](../docs/STOKES.md) guide.
`main.py --config` is the scientific command used inside a planned job; it does
not replace the workspace's required planning and submission procedure. Set
the compute backend and worker counts explicitly in the JSON: requesting a GPU
or CPUs in Slurm does not change `execution.backend`, `execution.n_jobs`, or
`execution.distance_cache.n_jobs`.

The retained local conveniences also generate independent configs:

- `sweep_main.py`: edit `N_TRAINS`, `SEEDS`, `TARGETS`, and optional `OUTPUT_ROOT`.
  Run with `.venv/bin/python sweep_main.py`. `DRY_RUN = True` writes configs and
  prints commands without fitting. Files live in `CONFIG_OUTPUT_DIR`; local
  summary metrics go to `RESULTS_CSV`.
- `run_n_train_sweep.sh`: edit `N_TRAINS`; `RUN_AFTER_UPDATE=0` generates configs
  without fitting. Override `CONFIG_FILE`, `CONFIG_OUTPUT_DIR`, or `KRR_PYTHON`
  through environment variables as needed.

Neither helper edits the base config. Saved predictions and local CSV summaries
are not authoritative research records: completed studies must still use the
reporting workflow under `reports/`.
