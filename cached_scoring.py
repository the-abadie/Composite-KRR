from contextlib import nullcontext

import numpy as np
from joblib import Parallel, delayed
from numpy.linalg import LinAlgError
from numpy.typing import NDArray
from sklearn.base import clone
from threadpoolctl import threadpool_limits

from kernel_cache import (
    DistanceCache,
    extract_regressor,
    resolve_cache_n_jobs,
    resolve_kernel_hyperparameters,
    score_fold_from_distances,
)


def resolve_candidate_hyperparameter_arrays(
    estimator,
    candidates: list[dict],
    cache: DistanceCache,
) -> tuple[NDArray, NDArray, NDArray]:
    """
    Resolve sklearn-style candidate parameter dictionaries into plain arrays.

    The returned arrays are the backend-neutral contract used by cached scoring:
    alphas has shape (n_candidates,), while gammas and weights have shape
    (n_candidates, n_components).
    """
    candidates = list(candidates)
    n_candidates = len(candidates)
    n_components = cache.n_components

    alphas = np.empty(n_candidates, dtype=float)
    gammas = np.empty((n_candidates, n_components), dtype=float)
    weights = np.empty((n_candidates, n_components), dtype=float)

    for candidate_index, params in enumerate(candidates):
        alpha, candidate_gammas, candidate_weights = resolve_candidate_params(
            estimator,
            params,
            cache,
        )
        alphas[candidate_index] = alpha
        gammas[candidate_index] = candidate_gammas
        weights[candidate_index] = candidate_weights

    return alphas, gammas, weights


def resolve_candidate_params(
    estimator,
    params: dict,
    cache: DistanceCache,
) -> tuple[float, list[float], list[float]]:
    candidate = clone(estimator)
    if params:
        candidate.set_params(**params)

    regressor = extract_regressor(candidate)
    return resolve_kernel_hyperparameters(
        regressor,
        n_components=cache.n_components,
    )


def score_estimator_params_from_cache_numpy(
    estimator,
    params: dict,
    cache: DistanceCache,
    *,
    scoring,
    n_jobs: int | None = None,
    blas_threads: int | None = None,
) -> NDArray:
    alphas, gammas, weights = resolve_candidate_hyperparameter_arrays(
        estimator,
        [params],
        cache,
    )
    return score_candidates_from_cache_numpy(
        alphas=alphas,
        gammas=gammas,
        weights=weights,
        cache=cache,
        scoring=scoring,
        n_jobs=n_jobs,
        blas_threads=blas_threads,
    )[0]


def score_estimator_params_from_cache_pytorch(
    estimator,
    params: dict,
    cache: DistanceCache,
    *,
    scoring,
    device: str | None = "auto",
    devices=None,
    dtype=None,
    candidate_batch_size: int = 1,
) -> NDArray:
    alphas, gammas, weights = resolve_candidate_hyperparameter_arrays(
        estimator,
        [params],
        cache,
    )
    return score_candidates_from_cache_pytorch(
        alphas=alphas,
        gammas=gammas,
        weights=weights,
        cache=cache,
        scoring=scoring,
        device=device,
        devices=devices,
        dtype=dtype,
        candidate_batch_size=candidate_batch_size,
    )[0]


def score_estimator_params_from_cache(
    estimator,
    params: dict,
    cache: DistanceCache,
    *,
    scoring,
    backend: str = "numpy",
    n_jobs: int | None = None,
    blas_threads: int | None = None,
    pytorch_device: str | None = "auto",
    pytorch_devices=None,
    pytorch_dtype=None,
    pytorch_candidate_batch_size: int = 1,
) -> NDArray:
    backend = normalize_cached_scoring_backend(backend)
    if backend == "numpy":
        return score_estimator_params_from_cache_numpy(
            estimator,
            params,
            cache,
            scoring=scoring,
            n_jobs=n_jobs,
            blas_threads=blas_threads,
        )

    return score_estimator_params_from_cache_pytorch(
        estimator,
        params,
        cache,
        scoring=scoring,
        device=pytorch_device,
        devices=pytorch_devices,
        dtype=pytorch_dtype,
        candidate_batch_size=pytorch_candidate_batch_size,
    )


def score_candidates_from_cache_numpy(
    *,
    alphas,
    gammas,
    weights,
    cache: DistanceCache,
    scoring,
    n_jobs: int | None = None,
    blas_threads: int | None = None,
) -> NDArray:
    alphas, gammas, weights = _validate_candidate_arrays(
        alphas,
        gammas,
        weights,
        n_components=cache.n_components,
    )
    n_candidates = alphas.shape[0]
    n_folds = len(cache.folds)
    split_scores = np.empty((n_candidates, n_folds), dtype=float)

    if n_candidates == 0:
        return split_scores

    if n_jobs in (None, 1) or n_folds == 1:
        for candidate_index in range(n_candidates):
            for fold_index, fold in enumerate(cache.folds):
                split_scores[candidate_index, fold_index] = _safe_score_fold_numpy(
                    fold,
                    alpha=float(alphas[candidate_index]),
                    gammas=gammas[candidate_index].tolist(),
                    weights=weights[candidate_index].tolist(),
                    kernel_types=cache.kernel_types,
                    scoring=scoring,
                )
        return split_scores

    score_n_jobs = resolve_cache_n_jobs(
        n_jobs,
        n_tasks=n_candidates * n_folds,
    )
    with _threadpool_limits_for_blas(blas_threads):
        scored_folds = Parallel(n_jobs=score_n_jobs, prefer="threads")(
            delayed(_score_candidate_fold_numpy)(
                candidate_index,
                fold_index,
                fold,
                float(alphas[candidate_index]),
                gammas[candidate_index].tolist(),
                weights[candidate_index].tolist(),
                cache.kernel_types,
                scoring,
            )
            for candidate_index in range(n_candidates)
            for fold_index, fold in enumerate(cache.folds)
        )

    for candidate_index, fold_index, score in scored_folds:
        split_scores[candidate_index, fold_index] = score

    return split_scores


def score_candidates_from_cache_pytorch(
    *,
    alphas,
    gammas,
    weights,
    cache: DistanceCache,
    scoring,
    device: str | None = "auto",
    devices=None,
    dtype=None,
    candidate_batch_size: int = 1,
) -> NDArray:
    from pytorch_backend import score_candidates_from_cache_pytorch as _impl

    return _impl(
        alphas=alphas,
        gammas=gammas,
        weights=weights,
        cache=cache,
        scoring=scoring,
        device=device,
        devices=devices,
        dtype=dtype,
        candidate_batch_size=candidate_batch_size,
    )


def score_candidates_from_cache(
    *,
    alphas,
    gammas,
    weights,
    cache: DistanceCache,
    scoring,
    backend: str = "numpy",
    n_jobs: int | None = None,
    blas_threads: int | None = None,
    pytorch_device: str | None = "auto",
    pytorch_devices=None,
    pytorch_dtype=None,
    pytorch_candidate_batch_size: int = 1,
) -> NDArray:
    backend = normalize_cached_scoring_backend(backend)
    if backend == "numpy":
        return score_candidates_from_cache_numpy(
            alphas=alphas,
            gammas=gammas,
            weights=weights,
            cache=cache,
            scoring=scoring,
            n_jobs=n_jobs,
            blas_threads=blas_threads,
        )

    return score_candidates_from_cache_pytorch(
        alphas=alphas,
        gammas=gammas,
        weights=weights,
        cache=cache,
        scoring=scoring,
        device=pytorch_device,
        devices=pytorch_devices,
        dtype=pytorch_dtype,
        candidate_batch_size=pytorch_candidate_batch_size,
    )


def normalize_cached_scoring_backend(backend: str) -> str:
    backend = str(backend).lower()
    if backend in {"numpy", "np", "cpu"}:
        return "numpy"
    if backend in {"pytorch", "torch", "gpu", "cuda", "rocm"}:
        return "pytorch"
    raise ValueError(
        'cached scoring backend must be "numpy" or "pytorch", '
        f"got {backend!r}."
    )


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


def _score_candidate_fold_numpy(
    candidate_index: int,
    fold_index: int,
    fold,
    alpha: float,
    gammas: list[float],
    weights: list[float],
    kernel_types: list[str],
    scoring,
) -> tuple[int, int, float]:
    score = _safe_score_fold_numpy(
        fold,
        alpha=alpha,
        gammas=gammas,
        weights=weights,
        kernel_types=kernel_types,
        scoring=scoring,
    )
    return candidate_index, fold_index, score


def _safe_score_fold_numpy(
    fold,
    *,
    alpha: float,
    gammas: list[float],
    weights: list[float],
    kernel_types: list[str],
    scoring,
) -> float:
    try:
        return score_fold_from_distances(
            fold,
            alpha=alpha,
            gammas=gammas,
            weights=weights,
            kernel_types=kernel_types,
            scoring=scoring,
        )
    except LinAlgError:
        return np.nan


def _threadpool_limits_for_blas(blas_threads: int | None):
    if blas_threads is None:
        return nullcontext()

    return threadpool_limits(limits=blas_threads)
