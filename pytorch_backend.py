from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from typing import Any

import numpy as np
from numpy.typing import NDArray

from kernel_cache import (
    DistanceCache,
    FoldDistanceCache,
    distance_spec_for_kernel,
    inverse_transform_target,
    score_predictions,
)


@dataclass(frozen=True)
class TorchFoldDistanceCache:
    fold_id: int
    train_distances: list[Any]
    validation_train_distances: list[Any]
    y_train_transformed: Any
    y_validation: NDArray
    y_validation_torch: Any
    target_transformer: object | None


@dataclass(frozen=True)
class TorchDistanceCache:
    folds: list[TorchFoldDistanceCache]
    names: list[str]
    kernel_types: list[str]
    normalizations: list[str]
    pca_components: list
    pca_whiten: list[bool]
    device: Any
    dtype: Any

    @property
    def n_components(self) -> int:
        return len(self.names)


def require_torch():
    try:
        return import_module("torch")
    except ImportError as exc:
        raise ImportError(
            "PyTorch is required for the PyTorch KRR backend. Install a PyTorch "
            "build appropriate for your hardware: CUDA builds for NVIDIA GPUs, "
            "or ROCm builds for supported AMD GPUs. PyTorch exposes both as "
            '`device="cuda"` when available.'
        ) from exc


def resolve_torch_device(torch, device: str | None = "auto"):
    if device is None or device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    resolved = torch.device(device)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            'Requested PyTorch device "cuda", but torch.cuda.is_available() is false. '
            "For AMD GPUs, install a ROCm-enabled PyTorch build; it still uses "
            'device type "cuda".'
        )
    return resolved


def resolve_torch_dtype(torch, dtype=None):
    if dtype is None:
        return torch.float64

    dtype = np.dtype(dtype)
    if dtype == np.dtype(np.float64):
        return torch.float64
    if dtype == np.dtype(np.float32):
        return torch.float32
    raise ValueError(f"Unsupported PyTorch floating dtype: {dtype}.")


def as_torch_tensor(x, *, torch, device, dtype):
    if hasattr(torch, "is_tensor") and torch.is_tensor(x):
        return x.to(device=device, dtype=dtype)

    return torch.as_tensor(x, dtype=dtype, device=device)


def distance_cache_to_pytorch(
    cache: DistanceCache,
    *,
    device: str | None = "auto",
    dtype=None,
) -> TorchDistanceCache:
    torch = require_torch()
    resolved_device = resolve_torch_device(torch, device)
    resolved_dtype = resolve_torch_dtype(
        torch,
        cache.folds[0].train_distances[0].dtype if dtype is None else dtype,
    )

    folds = [
        TorchFoldDistanceCache(
            fold_id=fold.fold_id,
            train_distances=[
                as_torch_tensor(
                    distance,
                    torch=torch,
                    device=resolved_device,
                    dtype=resolved_dtype,
                )
                for distance in fold.train_distances
            ],
            validation_train_distances=[
                as_torch_tensor(
                    distance,
                    torch=torch,
                    device=resolved_device,
                    dtype=resolved_dtype,
                )
                for distance in fold.validation_train_distances
            ],
            y_train_transformed=as_torch_tensor(
                fold.y_train_transformed,
                torch=torch,
                device=resolved_device,
                dtype=resolved_dtype,
            ),
            y_validation=np.asarray(fold.y_validation, dtype=float).reshape(-1),
            y_validation_torch=as_torch_tensor(
                fold.y_validation,
                torch=torch,
                device=resolved_device,
                dtype=resolved_dtype,
            ).reshape(-1),
            target_transformer=fold.target_transformer,
        )
        for fold in cache.folds
    ]

    return TorchDistanceCache(
        folds=folds,
        names=list(cache.names),
        kernel_types=list(cache.kernel_types),
        normalizations=list(cache.normalizations),
        pca_components=list(cache.pca_components),
        pca_whiten=list(cache.pca_whiten),
        device=resolved_device,
        dtype=resolved_dtype,
    )


def pairwise_self_lp_distance_pytorch(
    X,
    p: float,
    block_size: int = 1024,
    squared: bool = False,
    dtype=None,
    device: str | None = "auto",
):
    torch = require_torch()
    resolved_device = resolve_torch_device(torch, device)
    resolved_dtype = resolve_torch_dtype(torch, dtype)
    X = as_torch_tensor(X, torch=torch, device=resolved_device, dtype=resolved_dtype)

    if X.ndim != 2:
        raise ValueError(f"X must be a 2D tensor of shape (N, D), got {tuple(X.shape)}.")
    if p < 1:
        raise ValueError("p must be >= 1")
    if squared and p != 2:
        raise ValueError("`squared=True` only valid for p=2.")

    n_samples = X.shape[0]
    if n_samples == 0:
        return torch.empty((0, 0), dtype=resolved_dtype, device=resolved_device)
    if n_samples == 1:
        return torch.zeros((1, 1), dtype=resolved_dtype, device=resolved_device)

    if p == 2:
        sq_norms = torch.sum(X * X, dim=1)
        dist2 = sq_norms[:, None] + sq_norms[None, :] - 2.0 * (X @ X.T)
        dist2 = torch.clamp(dist2, min=0.0)
        if squared:
            return dist2
        dist = torch.sqrt(dist2)
        dist.fill_diagonal_(0.0)
        return dist

    if p == 1:
        dist = torch.cdist(X, X, p=1)
        dist.fill_diagonal_(0.0)
        return dist

    dist = torch.empty(
        (n_samples, n_samples),
        dtype=resolved_dtype,
        device=resolved_device,
    )
    for i0 in range(0, n_samples, block_size):
        i1 = min(i0 + block_size, n_samples)
        xi = X[i0:i1]
        for j0 in range(i0, n_samples, block_size):
            j1 = min(j0 + block_size, n_samples)
            xj = X[j0:j1]
            block = torch.sum(torch.abs(xi[:, None, :] - xj[None, :, :]) ** p, dim=2)
            block = block ** (1.0 / p)
            dist[i0:i1, j0:j1] = block
            if j0 != i0:
                dist[j0:j1, i0:i1] = block.T
    dist.fill_diagonal_(0.0)
    return dist


def pairwise_cross_lp_distance_pytorch(
    X,
    Y,
    p: float,
    block_size: int = 1024,
    squared: bool = False,
    dtype=None,
    device: str | None = "auto",
):
    torch = require_torch()
    resolved_device = resolve_torch_device(torch, device)
    resolved_dtype = resolve_torch_dtype(torch, dtype)
    X = as_torch_tensor(X, torch=torch, device=resolved_device, dtype=resolved_dtype)
    Y = as_torch_tensor(Y, torch=torch, device=resolved_device, dtype=resolved_dtype)

    if X.ndim != 2:
        raise ValueError(f"X must be a 2D tensor of shape (N, D), got {tuple(X.shape)}.")
    if Y.ndim != 2:
        raise ValueError(f"Y must be a 2D tensor of shape (M, D), got {tuple(Y.shape)}.")
    if X.shape[1] != Y.shape[1]:
        raise ValueError(
            f"X and Y must have the same feature dimension, got {X.shape[1]} "
            f"and {Y.shape[1]}."
        )
    if p < 1:
        raise ValueError("p must be >= 1")
    if squared and p != 2:
        raise ValueError("`squared=True` only valid for p=2.")

    n_left = X.shape[0]
    n_right = Y.shape[0]
    if n_left == 0 or n_right == 0:
        return torch.empty(
            (n_left, n_right),
            dtype=resolved_dtype,
            device=resolved_device,
        )

    if p == 2:
        x_sq_norms = torch.sum(X * X, dim=1)
        y_sq_norms = torch.sum(Y * Y, dim=1)
        dist2 = x_sq_norms[:, None] + y_sq_norms[None, :] - 2.0 * (X @ Y.T)
        dist2 = torch.clamp(dist2, min=0.0)
        if squared:
            return dist2
        return torch.sqrt(dist2)

    if p == 1:
        return torch.cdist(X, Y, p=1)

    dist = torch.empty(
        (n_left, n_right),
        dtype=resolved_dtype,
        device=resolved_device,
    )
    for i0 in range(0, n_left, block_size):
        i1 = min(i0 + block_size, n_left)
        xi = X[i0:i1]
        for j0 in range(0, n_right, block_size):
            j1 = min(j0 + block_size, n_right)
            yj = Y[j0:j1]
            block = torch.sum(torch.abs(xi[:, None, :] - yj[None, :, :]) ** p, dim=2)
            dist[i0:i1, j0:j1] = block ** (1.0 / p)
    return dist


def composite_kernel_from_distances_pytorch(
    distances: list[Any],
    *,
    gammas,
    weights,
    kernel_types: list[str],
):
    if not distances:
        raise ValueError("At least one distance matrix is required.")
    if not (len(distances) == len(kernel_types)):
        raise ValueError("distances and kernel_types must have matching lengths.")

    torch = require_torch()
    first_distance = distances[0]
    gammas = torch.as_tensor(
        gammas,
        dtype=first_distance.dtype,
        device=first_distance.device,
    )
    weights = torch.as_tensor(
        weights,
        dtype=first_distance.dtype,
        device=first_distance.device,
    )

    single_candidate = gammas.ndim == 1
    if single_candidate:
        gammas = gammas.reshape(1, -1)
    if weights.ndim == 1:
        weights = weights.reshape(1, -1)

    if gammas.shape != weights.shape:
        raise ValueError(f"gammas and weights must have the same shape, got {gammas.shape} and {weights.shape}.")
    if gammas.shape[1] != len(distances):
        raise ValueError(
            f"Expected {len(distances)} gamma/weight columns, got {gammas.shape[1]}."
        )

    out = None
    for component_index, (distance, kernel_type) in enumerate(
        zip(distances, kernel_types)
    ):
        if distance.shape != first_distance.shape:
            raise ValueError(
                "All distance matrices in one composite kernel must have the "
                f"same shape; got {distance.shape} and {first_distance.shape}."
            )
        distance_spec_for_kernel(kernel_type)
        gamma = gammas[:, component_index].reshape(-1, 1, 1)
        weight = weights[:, component_index].reshape(-1, 1, 1)
        component = torch.exp(-gamma * distance.unsqueeze(0)) * weight
        out = component if out is None else out + component

    return out[0] if single_candidate else out


def predict_fold_from_distances_pytorch(
    fold: FoldDistanceCache | TorchFoldDistanceCache,
    *,
    alpha: float,
    gammas: list[float],
    weights: list[float],
    kernel_types: list[str],
    device: str | None = "auto",
    dtype=None,
) -> NDArray:
    torch_fold = _ensure_torch_fold(fold, device=device, dtype=dtype)
    y_pred_batch = _predict_batch_from_torch_fold(
        torch_fold,
        alphas=np.asarray([alpha], dtype=float),
        gammas=np.asarray([gammas], dtype=float),
        weights=np.asarray([weights], dtype=float),
        kernel_types=kernel_types,
    )
    return _inverse_transform_prediction_batch(torch_fold, y_pred_batch)[0]


def score_fold_from_distances_pytorch(
    fold: FoldDistanceCache | TorchFoldDistanceCache,
    *,
    alpha: float,
    gammas: list[float],
    weights: list[float],
    scoring,
    kernel_types: list[str],
    device: str | None = "auto",
    dtype=None,
) -> float:
    y_pred = predict_fold_from_distances_pytorch(
        fold,
        alpha=alpha,
        gammas=gammas,
        weights=weights,
        kernel_types=kernel_types,
        device=device,
        dtype=dtype,
    )
    y_validation = (
        fold.y_validation
        if isinstance(fold, FoldDistanceCache)
        else fold.y_validation
    )
    return score_predictions(y_validation, y_pred, scoring)


def score_candidates_from_cache_pytorch(
    *,
    alphas,
    gammas,
    weights,
    cache: DistanceCache | TorchDistanceCache,
    scoring,
    device: str | None = "auto",
    dtype=None,
    candidate_batch_size: int = 1,
) -> NDArray:
    torch_cache = (
        cache
        if isinstance(cache, TorchDistanceCache)
        else distance_cache_to_pytorch(cache, device=device, dtype=dtype)
    )
    torch = require_torch()
    alphas, gammas, weights = _validate_candidate_arrays(
        alphas,
        gammas,
        weights,
        n_components=torch_cache.n_components,
    )
    if candidate_batch_size <= 0:
        raise ValueError("candidate_batch_size must be a positive int.")

    n_candidates = alphas.shape[0]
    split_scores = np.empty((n_candidates, len(torch_cache.folds)), dtype=float)
    with torch.no_grad():
        for fold_index, fold in enumerate(torch_cache.folds):
            for start in range(0, n_candidates, candidate_batch_size):
                stop = min(start + candidate_batch_size, n_candidates)
                batch_scores = _score_candidate_batch_for_fold(
                    fold,
                    alphas=alphas[start:stop],
                    gammas=gammas[start:stop],
                    weights=weights[start:stop],
                    kernel_types=torch_cache.kernel_types,
                    scoring=scoring,
                )
                split_scores[start:stop, fold_index] = batch_scores

    return split_scores


def _ensure_torch_fold(
    fold: FoldDistanceCache | TorchFoldDistanceCache,
    *,
    device: str | None,
    dtype,
) -> TorchFoldDistanceCache:
    if isinstance(fold, TorchFoldDistanceCache):
        return fold

    torch_cache = distance_cache_to_pytorch(
        DistanceCache(
            folds=[fold],
            names=[str(index) for index in range(len(fold.train_distances))],
            kernel_types=["rbf"] * len(fold.train_distances),
            normalizations=["none"] * len(fold.train_distances),
            pca_components=[None] * len(fold.train_distances),
            pca_whiten=[False] * len(fold.train_distances),
            estimated_nbytes=0,
            available_memory_nbytes=None,
            memory_budget_nbytes=None,
            n_jobs=1,
        ),
        device=device,
        dtype=dtype,
    )
    return torch_cache.folds[0]


def _score_candidate_batch_for_fold(
    fold: TorchFoldDistanceCache,
    *,
    alphas: NDArray,
    gammas: NDArray,
    weights: NDArray,
    kernel_types: list[str],
    scoring,
) -> NDArray:
    try:
        y_pred_batch = _predict_batch_from_torch_fold(
            fold,
            alphas=alphas,
            gammas=gammas,
            weights=weights,
            kernel_types=kernel_types,
        )
    except RuntimeError as exc:
        if not _is_torch_linalg_error(exc):
            raise
        if len(alphas) == 1:
            return np.asarray([np.nan], dtype=float)

        scores = [
            _score_candidate_batch_for_fold(
                fold,
                alphas=alphas[index : index + 1],
                gammas=gammas[index : index + 1],
                weights=weights[index : index + 1],
                kernel_types=kernel_types,
                scoring=scoring,
            )[0]
            for index in range(len(alphas))
        ]
        return np.asarray(scores, dtype=float)

    torch_scores = _score_prediction_batch_pytorch(fold, y_pred_batch, scoring)
    if torch_scores is not None:
        return torch_scores.detach().cpu().numpy().astype(float, copy=False)

    y_pred_batch_np = _inverse_transform_prediction_batch(fold, y_pred_batch)
    return np.asarray(
        [
            score_predictions(fold.y_validation, y_pred, scoring)
            for y_pred in y_pred_batch_np
        ],
        dtype=float,
    )


def _predict_batch_from_torch_fold(
    fold: TorchFoldDistanceCache,
    *,
    alphas: NDArray,
    gammas: NDArray,
    weights: NDArray,
    kernel_types: list[str],
):
    torch = require_torch()
    alphas_t = torch.as_tensor(
        alphas,
        dtype=fold.y_train_transformed.dtype,
        device=fold.y_train_transformed.device,
    ).reshape(-1)
    gammas_t = torch.as_tensor(
        gammas,
        dtype=fold.y_train_transformed.dtype,
        device=fold.y_train_transformed.device,
    )
    weights_t = torch.as_tensor(
        weights,
        dtype=fold.y_train_transformed.dtype,
        device=fold.y_train_transformed.device,
    )

    K_train = composite_kernel_from_distances_pytorch(
        fold.train_distances,
        gammas=gammas_t,
        weights=weights_t,
        kernel_types=kernel_types,
    )
    diag = torch.arange(K_train.shape[1], device=K_train.device)
    K_train[:, diag, diag] = K_train[:, diag, diag] + alphas_t[:, None]

    rhs = fold.y_train_transformed.reshape(1, -1, 1).expand(K_train.shape[0], -1, -1)
    dual_coef = _cholesky_solve_spd_batch(K_train, rhs).squeeze(-1)

    K_validation = composite_kernel_from_distances_pytorch(
        fold.validation_train_distances,
        gammas=gammas_t,
        weights=weights_t,
        kernel_types=kernel_types,
    )
    return torch.bmm(K_validation, dual_coef.unsqueeze(-1)).squeeze(-1)


def _cholesky_solve_spd_batch(K_train, rhs):
    torch = require_torch()
    factor, info = torch.linalg.cholesky_ex(K_train, upper=False, check_errors=False)
    if bool(torch.any(info != 0).item()):
        raise RuntimeError("Cholesky factorization failed for one or more candidates.")
    return torch.cholesky_solve(rhs, factor, upper=False)


def _score_prediction_batch_pytorch(
    fold: TorchFoldDistanceCache,
    y_pred_batch,
    scoring,
):
    if not isinstance(scoring, str):
        return None

    torch = require_torch()
    y_pred = _inverse_transform_prediction_batch_pytorch(fold, y_pred_batch)
    if y_pred is None:
        return None

    y_true = fold.y_validation_torch.reshape(1, -1)
    diff = y_pred - y_true

    if scoring == "neg_mean_absolute_error":
        return -torch.mean(torch.abs(diff), dim=1)
    if scoring == "neg_mean_squared_error":
        return -torch.mean(diff * diff, dim=1)
    if scoring == "neg_root_mean_squared_error":
        return -torch.sqrt(torch.mean(diff * diff, dim=1))
    if scoring == "r2":
        ss_res = torch.sum(diff * diff, dim=1)
        y_centered = y_true - torch.mean(y_true, dim=1, keepdim=True)
        ss_tot = torch.sum(y_centered * y_centered, dim=1)
        raw_scores = 1.0 - ss_res / ss_tot
        perfect_scores = torch.ones_like(raw_scores)
        imperfect_scores = torch.zeros_like(raw_scores)
        finite_constant_scores = torch.where(
            ss_res == 0.0,
            perfect_scores,
            imperfect_scores,
        )
        return torch.where(ss_tot == 0.0, finite_constant_scores, raw_scores)

    return None


def _inverse_transform_prediction_batch_pytorch(
    fold: TorchFoldDistanceCache,
    y_pred_batch,
):
    return _inverse_transform_tensor_pytorch(fold.target_transformer, y_pred_batch)


def _inverse_transform_tensor_pytorch(transformer, values):
    if transformer is None:
        return values

    class_name = transformer.__class__.__name__
    if class_name == "Pipeline":
        out = values
        for _, step in reversed(transformer.steps):
            out = _inverse_transform_tensor_pytorch(step, out)
            if out is None:
                return None
        return out

    if class_name == "StandardScaler":
        return _inverse_standard_scaler_pytorch(transformer, values)

    if class_name == "FunctionTransformer":
        return _inverse_function_transformer_pytorch(transformer, values)

    return None


def _inverse_standard_scaler_pytorch(transformer, values):
    torch = require_torch()
    scale = getattr(transformer, "scale_", None)
    mean = getattr(transformer, "mean_", None)
    if scale is None or mean is None:
        return None

    scale = np.asarray(scale, dtype=float).reshape(-1)
    mean = np.asarray(mean, dtype=float).reshape(-1)
    if scale.size != 1 or mean.size != 1:
        return None

    scale_t = torch.as_tensor(scale[0], dtype=values.dtype, device=values.device)
    mean_t = torch.as_tensor(mean[0], dtype=values.dtype, device=values.device)
    return values * scale_t + mean_t


def _inverse_function_transformer_pytorch(transformer, values):
    torch = require_torch()
    inverse_func = getattr(transformer, "inverse_func", None)
    if inverse_func is None:
        return values

    func_name = getattr(inverse_func, "__name__", "")
    if func_name in {"_identity", "identity"}:
        return values
    if inverse_func is np.exp or func_name == "exp":
        return torch.exp(values)
    if inverse_func is np.expm1 or func_name == "expm1":
        return torch.expm1(values)

    return None


def _inverse_transform_prediction_batch(
    fold: TorchFoldDistanceCache,
    y_pred_batch,
) -> list[NDArray]:
    y_pred_np = y_pred_batch.detach().cpu().numpy()
    return [
        inverse_transform_target(fold.target_transformer, y_pred)
        for y_pred in y_pred_np
    ]


def _is_torch_linalg_error(exc: RuntimeError) -> bool:
    message = str(exc).lower()
    return "linalg" in message or "singular" in message or "cholesky" in message


def _validate_candidate_arrays(
    alphas,
    gammas,
    weights,
    *,
    n_components: int,
) -> tuple[NDArray, NDArray, NDArray]:
    alphas = np.asarray(alphas, dtype=float).reshape(-1)
    gammas = np.asarray(gammas, dtype=float)
    weights = np.asarray(weights, dtype=float)

    if gammas.ndim == 1 and alphas.shape[0] == 1:
        gammas = gammas.reshape(1, -1)
    if weights.ndim == 1 and alphas.shape[0] == 1:
        weights = weights.reshape(1, -1)

    expected_shape = (alphas.shape[0], n_components)
    if gammas.shape != expected_shape:
        raise ValueError(f"gammas must have shape {expected_shape}, got {gammas.shape}.")
    if weights.shape != expected_shape:
        raise ValueError(f"weights must have shape {expected_shape}, got {weights.shape}.")

    return alphas, gammas, weights
