# Composite Kernel Ridge Regression

Python package for implementing Kernel Ridge Regression with multiple kernels.
- Parallel: CPU parallel via joblib, and GPU parallel via PyTorch tensor operations.
- Easy config
- Very tunable learning
- Supports K-Fold CV and target-stratification
- Easy to add multiple descriptors and kernels
- Multi-target learning with shared-kernel KRR/CKRR.
- Config validation
- Centered kernel alignment diagnostics for fitted base and product kernels.

## Kernel alignment analysis

Enable post-fit centered kernel alignment (CKA) in a structured run config:

```json
"reporting": {
  "kernel_alignment": {
    "enabled": true,
    "max_samples": 1000,
    "include_products": true
  }
}
```

| Field | Default | Meaning |
|---|---|---|
| `enabled` | `false` | Generate diagnostics after the final refit, before held-out evaluation. |
| `max_samples` | `1000` | Maximum training/CV rows, integer at least 3. Use all rows when the pool is smaller. |
| `include_products` | `true` | Compare base kernels followed by configured product terms in saved weight order. Set `false` for base kernels only. |

The same fields work in structured TOML under `[reporting.kernel_alignment]`,
and appear in `krr_cli.py template` and `krr_cli.py schema`. For example:

```bash
.venv/bin/python krr_cli.py resolve run.json \
  --set reporting.kernel_alignment.enabled=true \
  --set reporting.kernel_alignment.max_samples=500 \
  --set 'output.directory="../outputs/alignment-seed12"' \
  --output configs/alignment-seed12.json
.venv/bin/python main.py --config configs/alignment-seed12.json
```

This option is independent of `reporting.kernel_contributions`; it performs no
additional model searches. Enabled or otherwise non-default alignment settings
require structured configuration. Conversion to the uppercase legacy view is
rejected instead of silently discarding them. The bundled `config.json` shows
the disabled option; old configs that omit it also leave analysis disabled.

For Gram matrices on the same rows, the statistic is:

```text
H = I - 11^T / n
CKA(K_i, K_j) = <H K_i H, H K_j H>_F / (||H K_i H||_F ||H K_j H||_F)
```

The analysis uses only the outer-training/CV pool, never held-out descriptors or
labels. A local RNG seeded with the recorded run seed samples uniformly without
replacement and retains the original training-pool order. The final model's
fitted normalization, spatial DCT, PCA, kernel types, and bandwidths are reused;
the sampled subset does not refit any transform. Each term is **unweighted**,
including terms whose fitted weight is zero. Products multiply unweighted base
Gram matrices elementwise, including repeated factors, before centering. An
affine linear baseline is not a separate kernel term.

Diagnostics run on CPU for NumPy, PyTorch, exact, and Nyström models. Base kernels
use the fitted descriptor precision; products and alignment calculations use
float64. Nyström reports compare exact base/product kernels on the sampled rows,
not their landmark approximation. With `s` sampled rows, `b` base kernels and `t`
reported terms, temporary kernel storage is `O((b+t)*s^2)` and pairwise alignment
work is `O(t^2*s^2)`, in addition to descriptor transforms and kernel construction.
Increasing `max_samples` can substantially increase memory and runtime, including
on GPU training jobs. Full Gram matrices are not saved.

The output directory receives:

| Artifact | Contents |
|---|---|
| `kernel_alignment.json` | Versioned report with the symmetric CKA matrix, term definitions, fitted weights and bandwidths, exact sampled global indices and local positions, precision, and diagnostic warnings. |
| `kernel_alignment.csv` | Labeled square alignment matrix in the same term order. |
| `kernel_alignment.png` | Heatmap, with undefined entries shown in gray. |
| `kernel_alignment_sample_indices.npy` | Exact global dataset row indices in matrix order. |

`run_metrics.json` includes an alignment summary and elapsed seconds, plus SHA-256
records for all four artifacts. Timings are logged separately from final model
evaluation. Alignment artifacts are diagnostics, not authoritative model result
records; the workspace's `reports/` recording policy still applies to research runs.

Nondegenerate positive-semidefinite kernels have CKA in `[0, 1]`, with 1 indicating
proportional centered geometry on these rows. Symmetric indefinite kernels can
produce negative values; the diagnostic does not perform a PSD test. A constant
or zero kernel has undefined alignment: its row/column is JSON `null`, CSV blank,
and gray in the plot, with a warning and `constant` term status. Numerically zero
centered norms use a scale-relative `64 * float64 epsilon` tolerance. Fewer than
three available rows produce an all-undefined report with `insufficient_samples`
term statuses. Such reports have overall status `partial`.

Reports warn about nearly constant kernels (centered/raw Frobenius norm ratio
below `1e-8`) and nearly identity centered geometry (alignment with `H` above
`1 - 1e-6`). These are numerical diagnostic thresholds, not redundancy cutoffs.
Similarity depends on bandwidth, sample distribution, and sample count. This is
a post-fit description at selected hyperparameters, not an independent estimate
of predictive contribution. Use retrained individual/group CV ablations to
establish whether terms can be removed; keep kernel selection within training/CV
data and reserve held-out test data for final evaluation. No terms are pruned
automatically. See [Cortes et al. (2012)](https://www.jmlr.org/papers/v13/cortes12a.html)
and [Kornblith et al. (2019)](https://proceedings.mlr.press/v97/kornblith19a.html)
for centered alignment and its interpretation.

The Python API compares the unweighted kernel terms of a fitted exact or
Nyström estimator (also accepted inside a fitted `TransformedTargetRegressor`):

```python
from kernel_alignment import analyze_kernel_alignment, centered_kernel_alignment

report = analyze_kernel_alignment(
    fitted_estimator, X_train_val, sample_indices=global_training_indices,
    max_samples=1000, include_products=True, random_state=12,
    output_dir="output/alignment",
)
score = centered_kernel_alignment(K_a, K_b)
```

`X_train_val` is the same raw sample-by-descriptor object matrix accepted by the
estimators. The analysis reuses fitted preprocessing and bandwidths without
refitting. Supply only training/CV rows, with matching global sample indices;
omitting indices uses local row positions. The returned JSON-compatible report
contains the alignment matrix, term definitions, fitted weights, sample indices,
and diagnostic warnings. It writes JSON, matrix CSV, PNG heatmap, and sampled
indices when `output_dir` is provided. `centered_kernel_alignment` accepts two
same-order square Gram matrices and returns `None` for zero centered norm.

High alignment identifies similar geometry; use retrained CV ablations to
establish whether a term can be removed for a particular prediction task.

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
Torch installation. An explicit `device = "cuda"` request fails if CUDA is
unavailable, including when `devices = "auto"` selects the GPU list. Use
`device = "auto"` to permit CPU fallback. Products need no additional descriptor loads or distance
matrices. Each base kernel is exponentiated once per matrix assembly/candidate
and reused; only factors needed by products are retained. This trades temporary
kernel memory for reuse. Prediction/Nyström row batches and Torch candidate
batches bound that temporary memory. Exact training still uses dense matrices.

Run the regression checks with `.venv/bin/python -m unittest discover -v`.
The optional Torch parity test runs on CPU and also CUDA when available; it is
skipped when Torch is absent.

## Fold-fitted linear baseline

Set `model.linear_baseline` to a descriptor ID (for example `"counts"`) to fit
an affine baseline on that block and train exact KRR on its residuals. For
molecular energies, an element-count block can represent a composition
baseline with only a few features. The designated descriptor remains a base
kernel too, with its weight tuned normally.

The baseline uses the raw designated block and centered least squares with an
intercept. It is fitted separately on each CV training fold, after that fold's
target transformation. Validation predictions add its baseline before inverting
the target transformation. The final model refits the baseline on the full
outer-training pool. Held-out labels never enter either fit. NumPy, PyTorch,
cached scoring, scalar targets, and multi-output targets use the same rule.

The default `null` retains existing behavior. Nyström and kernel-contribution
ablations currently reject non-null baselines. Completed runs preserve
`linear_baseline_coefficients.npy` and `linear_baseline_intercept.npy`, in the
transformed target units, with their checksums in `run_metrics.json`.
The direct `CompositeKRREstimator` API accepts `linear_baseline="counts"` too.

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
Feature preprocessing and target transforms (including a separate standard
scaling mean/scale for each target column) are fitted on each CV training fold.
Search scores use sklearn's default multi-output aggregation for the selected
metric. `search.target_space="original"` preserves scoring in original target
units. `search.target_space="transformed"` scores after the target transform
fitted on that fold's training rows, on cached NumPy/Torch and uncached sklearn
paths. With `target.normalization="standard"` this gives equally weighted
standardized per-target errors, so heterogeneous physical units do not dominate
a shared-kernel multi-target objective. Neither validation nor test targets are
used to fit those transformations.

Held-out reporting always logs aggregate MAE/RMSE plus per-target MAE/RMSE in
original units. `run_metrics.json` stores `target_names`, `n_targets`,
`target_mae`, and `target_rmse` in identical column order. For NPY multi-output
targets, `target.name` can be a list with one label per column; NPZ key labels
are expanded when a key contains multiple columns. The metrics record the
`scoring_target_space` and `cv_objective_error`; transformed MAE selection is
stored as `cv_transformed_mae`, while `cv_mae` is reserved for original-unit MAE.
Aggregate RMSE is the arithmetic mean of the per-target RMSEs, matching
`neg_root_mean_squared_error` during search. It is not the square root of a
pooled MSE across targets. Scalar-target results are unchanged. Multi-target
reports produced before this correction used the pooled definition; compare
them only after recomputing RMSE from their saved predictions and true targets.
When `split.stratify = true`, multi-output targets are reduced to the first principal
direction of standardized target columns for split stratification.

### Large held-out complements

NPY descriptors and targets are loaded with read-only memory mapping. Only the
training/CV pool is materialized during search. Final evaluation loads,
preprocesses, and predicts at most `execution.prediction_batch_size` rows at a
time (default 2048), on both NumPy and PyTorch. This outer row limit also bounds
feature preprocessing, while `execution.torch.predict_batch_size` independently
bounds Torch kernel construction. Exact training still uses dense matrices.

Every held-out row is saved in `y_predictions.npy` and `y_true.npy`, in
`test_idx.npy` order. MAE, RMSE, their per-target values, and absolute-error
standard deviation are accumulated in float64 over the complete test set.
Diagnostic scatter/histogram plots use at most `reporting.plot_max_samples`
evenly spaced test rows (default 20000); sampled plots are labeled. The metric,
prediction, and plot row counts are recorded in `run_metrics.json`. NPZ targets
cannot be memory-mapped and are still loaded in memory.

Compressed banks can use a JSON shard manifest as a descriptor `path`, with
`archive_key` selecting a stored component or a virtual concatenation. All
component kernels may reference the same manifest and NPZ parts, so no combined
NPY or duplicate component banks are required. A manifest has this structure:

```json
{
  "schema_version": 1,
  "arrays": {
    "component_0_data": {"shape": [10000, 80], "dtype": "float64"},
    "component_1_data": {"shape": [10000, 120], "dtype": "float64"},
    "combined": {"shape": [10000, 200], "dtype": "float64",
                 "source_keys": ["component_0_data", "component_1_data"]}
  },
  "parts": [
    {"path": "part-00000.npz", "start": 0, "stop": 10000,
     "sha256": "<SHA-256 of these exact compressed archive bytes>"}
  ]
}
```

Part paths are relative to the manifest. Parts must cover the sample axis in
order without gaps or overlaps. Stored arrays match the declared dtype and
feature shape; virtual concatenation flattens and joins only the selected rows.
Rows may be requested in any order, including repeated indices. The loader
verifies each accessed part's SHA-256 and caches one decompressed part, shared
across component views. Gathering multiple blocks from one manifest visits each
selected part once, reads the union of required stored keys, and scatters rows
back into the requested order. Component views and virtual concatenations share
that traversal. Random training rows can still touch most parts of a bank;
grouping avoids a separate archive traversal for each component. It refuses
implicit conversion of the full bank.
Keep shards small enough that one decompressed part fits in memory.

`run_metrics.json` records the selected key, shape, dtype, manifest SHA-256,
and complete manifest including part checksums and any additional generation
metadata. An original generator artifact record retained in `generated_record`
describes the original generation archive, while a repacked component-only
archive must have its own distinct `storage_record` and part SHA-256. These
records must not claim that a repacked archive has the original bytes checksum.

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
  "reporting": {"kernel_alignment": {"enabled": true, "max_samples": 500}},
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
retains workspace tokens in the saved file. By default the workspace is the
engine repository's parent directory. `KRR_WORKSPACE_ROOT` binds those tokens
to a different runtime workspace; the experiment orchestrator sets it when
running frozen engine source. An explicit `resolved(workspace=...)` argument
takes precedence. Relative paths still resolve from the configuration file.
Configuration strings do not expand arbitrary environment variables, and
scientific settings are not inherited from the environment.

Specify exactly one of `split.n_train` (training plus CV samples) and
`split.train_fraction`. `split.n_samples` defaults to all available samples.
An explicit sample pool larger than the dataset is an error. Setting either
split field with `--set` clears the other inherited field. For predefined splits,
use only `mode: "predefined"`, `train_indices`, `validation_folds`, and
`test_indices`; validation folds may be an NPY array or NPZ with `fold0`, `fold1`,
etc. Cross-field checks also validate bounds, kernel names, and descriptor IDs.
These are global dataset indices. `train_indices` are always included in the
training portion of each inner fold; they may be an empty integer array for
ordinary K-fold CV in which every outer-training sample is validated once.
Each validation fold must leave at least one training sample. All train,
validation and held-out index groups must be disjoint.

Descriptors can apply a fixed spatial DCT after fold-fitted normalization:

```json
{"id": "angular", "path": "angular.npy", "kernel": "rbf",
 "normalization": "standard",
 "spatial_dct": {"block_size": 600, "n_components": 32}}
```

The flattened input must contain consecutive channels of `block_size` bins.
Each CV training fold fits its normalization on the original bins, applies
orthonormal DCT-II independently to each channel, and keeps modes
`0..n_components-1`. Validation and prediction reuse that fold's normalization;
the final refit uses the full outer-training pool. By default no standardization
follows the DCT. Set `spatial_dct.post_normalization` to `"standard"` to fit a
second scaler on the retained coefficients of that same training fold; validation
and prediction reuse both fitted scalers. Optional PCA follows these operations. Exact/Nyström caches,
uncached estimators and saved exact inference use the same preprocessing recipe.
With every mode retained and post-normalization `"none"`, RBF distances are
preserved relative to the same normalization without DCT. `spatial_dct` is structured-only; conversion to legacy
fields is rejected rather than dropping this operation.

`run_metrics.json` keeps `features` as the number of stored input features and
also records `kernel_features` and `transformed_feature_counts` for fitted exact
models. A fold-fitted reduction can require larger input arrays than its final
kernel representation; keep these costs separate when reporting compression.
Exact distance-cache logs also record each fold/component's preprocessing seconds,
including row gathering, fitted normalization and dimensional transforms, before
distance computation. These task durations may overlap with multiple cache workers;
their sum is not elapsed wall time. Native metrics record peak process RSS and
Torch peak allocated GPU bytes separately.

Each run records `resolved_config.json` and `resolved_config.sha256`, including
the actual random seed when `seed` was null. The output directory must be empty
unless `output.overwrite` is true; a different resolved configuration is always
refused. A `.run.lock` prevents simultaneous jobs writing the same directory.
If a process is killed, confirm it has stopped before manually removing its lock.
Use a distinct output directory for each independent job.

Completed runs also write `run_metrics.json` directly from the fitted search
object, with CV fold errors, test metrics, best parameters, timings, input and
artifact hashes, and runtime/GPU metadata. `validation_folds.npz` preserves
global training and validation indices for every CV fold. These engine artifacts
are the evidence for downstream authoritative reporting under `reports/`.
Optional [kernel alignment reports](#kernel-alignment-analysis) add training-only
CKA matrices, a heatmap, sampled row indices, and artifact checksums.

For CUDA timing studies, set `execution.torch.warmup_size` (for example 512).
The run warms distance, Gaussian-kernel, Cholesky and solve operations in its
own process and configured precision, synchronizes, then starts the training
timer. Deterministic inputs preserve random state. `run_metrics.json` records
warm-up separately and exposes search-stage durations; total elapsed time still
includes warm-up. The default zero preserves runs without explicit warm-up.

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

`saved_model.SavedExactModel` evaluates saved exact-KRR dual coefficients without
new fitting or hyperparameter search. It rebuilds descriptor and target transforms
using the original training rows only. Callers must verify input/artifact hashes
and reproduce the original saved held-out predictions before external evaluation.
Scalar targets with no PCA or linear baseline are supported, including kernel
products, on NumPy or PyTorch. This inference API does not start a new training run;
new training runs still use `main.py --config` below.

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
