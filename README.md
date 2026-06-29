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
