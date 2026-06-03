from dataclasses import dataclass
import logging
import os
import platform
import re
import subprocess

from joblib import Parallel, delayed, effective_n_jobs
import numpy as np
from numpy.typing import NDArray
from scipy.linalg import solve
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.metrics import get_scorer
from threadpoolctl import threadpool_limits

from config import VERBOSITY
from kernels import pairwise_cross_lp_distance, pairwise_self_lp_distance
from preprocess import make_data_preprocessor
from utilities import configure_logging

configure_logging(VERBOSITY)
logger = logging.getLogger("kernel-cache")


@dataclass(frozen=True)
class KernelDistanceSpec:
    kernel_type: str
    p: float
    squared: bool


@dataclass(frozen=True)
class FoldDistanceCache:
    fold_id: int
    train_indices: NDArray
    validation_indices: NDArray
    train_distances: list[NDArray]
    validation_train_distances: list[NDArray]
    y_train_transformed: NDArray
    y_validation: NDArray
    target_transformer: object | None


@dataclass(frozen=True)
class DistanceCache:
    folds: list[FoldDistanceCache]
    names: list[str]
    kernel_types: list[str]
    normalizations: list[str]
    estimated_nbytes: int
    available_memory_nbytes: int | None
    memory_budget_nbytes: int | None
    n_jobs: int

    @property
    def n_components(self) -> int:
        return len(self.names)

    @property
    def nbytes(self) -> int:
        total = 0
        for fold in self.folds:
            total += sum(distance.nbytes for distance in fold.train_distances)
            total += sum(
                distance.nbytes for distance in fold.validation_train_distances
            )
        return total


class UnsupportedDistanceKernelError(ValueError):
    pass


class _PredictionOnlyRegressor(RegressorMixin, BaseEstimator):
    def __init__(self, y_pred: NDArray):
        self.y_pred = np.asarray(y_pred)

    def predict(self, X):
        return self.y_pred


def build_distance_cache(
    X,
    y,
    cv,
    *,
    names: list[str],
    kernel_types: list[str],
    normalizations: list[str],
    target_transformer=None,
    block_size: int = 1024,
    dtype: np.dtype | type = np.float64,
    use_scipy_for_p1_p2: bool = True,
    n_jobs: int | None = -1,
    memory_fraction: float = 0.80,
) -> DistanceCache:
    dtype = np.dtype(dtype)
    X_blocks = unpack_sample_matrix(X, dtype=dtype)
    y = np.asarray(y, dtype=dtype).reshape(-1)
    if not 0 < memory_fraction <= 1:
        raise ValueError("memory_fraction must be in (0, 1].")
    if n_jobs == 0:
        raise ValueError("n_jobs must be None or a non-zero int.")

    if len(X_blocks) != len(names):
        raise ValueError(
            f"Expected {len(names)} descriptor blocks, got {len(X_blocks)}."
        )
    if y.shape[0] != X_blocks[0].shape[0]:
        raise ValueError(f"X has {X_blocks[0].shape[0]} samples, but y has {y.shape[0]}.")
    if len(names) != len(kernel_types) or len(names) != len(normalizations):
        raise ValueError("names, kernel_types, and normalizations must have the same length.")

    specs = [distance_spec_for_kernel(kernel_type) for kernel_type in kernel_types]
    fold_indices = [
        (np.asarray(train_idx, dtype=int), np.asarray(validation_idx, dtype=int))
        for train_idx, validation_idx in cv.split(X, y)
    ]
    estimated_nbytes = estimate_distance_cache_nbytes(
        fold_indices,
        n_components=len(names),
        dtype=dtype,
    )
    available_memory_nbytes = available_memory_bytes()
    memory_budget_nbytes = (
        None
        if available_memory_nbytes is None
        else int(available_memory_nbytes * memory_fraction)
    )
    validate_distance_cache_memory(
        estimated_nbytes=estimated_nbytes,
        available_memory_nbytes=available_memory_nbytes,
        memory_budget_nbytes=memory_budget_nbytes,
        memory_fraction=memory_fraction,
    )

    cache_n_jobs = resolve_cache_n_jobs(n_jobs, n_folds=len(fold_indices))
    logger.info(
        "Estimated distance cache memory: %.2f MiB.",
        estimated_nbytes / (1024**2),
    )
    if memory_budget_nbytes is not None:
        logger.info(
            "Distance cache memory budget: %.2f MiB "
            "(%.0f%% of currently available memory).",
            memory_budget_nbytes / (1024**2),
            100 * memory_fraction,
        )
    logger.info(
        "Precomputing %s fold distance cache with %s worker thread(s).",
        len(fold_indices),
        cache_n_jobs,
    )

    if cache_n_jobs == 1:
        folds = [
            build_fold_distance_cache(
                fold_id=fold_id,
                train_idx=train_idx,
                validation_idx=validation_idx,
                X_blocks=X_blocks,
                y=y,
                specs=specs,
                normalizations=normalizations,
                target_transformer=target_transformer,
                block_size=block_size,
                dtype=dtype,
                use_scipy_for_p1_p2=use_scipy_for_p1_p2,
            )
            for fold_id, (train_idx, validation_idx) in enumerate(
                fold_indices,
                start=1,
            )
        ]
    else:
        with threadpool_limits(limits=1):
            folds = Parallel(n_jobs=cache_n_jobs, prefer="threads")(
                delayed(build_fold_distance_cache)(
                    fold_id=fold_id,
                    train_idx=train_idx,
                    validation_idx=validation_idx,
                    X_blocks=X_blocks,
                    y=y,
                    specs=specs,
                    normalizations=normalizations,
                    target_transformer=target_transformer,
                    block_size=block_size,
                    dtype=dtype,
                    use_scipy_for_p1_p2=use_scipy_for_p1_p2,
                )
                for fold_id, (train_idx, validation_idx) in enumerate(
                    fold_indices,
                    start=1,
                )
            )

    cache = DistanceCache(
        folds=folds,
        names=list(names),
        kernel_types=list(kernel_types),
        normalizations=list(normalizations),
        estimated_nbytes=estimated_nbytes,
        available_memory_nbytes=available_memory_nbytes,
        memory_budget_nbytes=memory_budget_nbytes,
        n_jobs=cache_n_jobs,
    )
    logger.info(
        "Precomputed %s fold distance cache using %.2f MiB.",
        len(cache.folds),
        cache.nbytes / (1024**2),
    )
    return cache


def build_fold_distance_cache(
    *,
    fold_id: int,
    train_idx: NDArray,
    validation_idx: NDArray,
    X_blocks: list[NDArray],
    y: NDArray,
    specs: list[KernelDistanceSpec],
    normalizations: list[str],
    target_transformer,
    block_size: int,
    dtype: np.dtype,
    use_scipy_for_p1_p2: bool,
) -> FoldDistanceCache:
    target_transformer_fold, y_train_transformed = fit_target_transformer(
        target_transformer,
        y[train_idx],
    )
    y_train_transformed = np.asarray(y_train_transformed, dtype=dtype)

    train_distances = []
    validation_train_distances = []

    for X_block, normalization, spec in zip(X_blocks, normalizations, specs):
        preprocessor = make_data_preprocessor(normalization)
        X_train = preprocessor.fit_transform(X_block[train_idx])
        X_validation = preprocessor.transform(X_block[validation_idx])

        train_distances.append(
            pairwise_self_lp_distance(
                X_train,
                p=spec.p,
                squared=spec.squared,
                block_size=block_size,
                dtype=dtype,
                use_scipy_for_p1_p2=use_scipy_for_p1_p2,
            )
        )
        validation_train_distances.append(
            pairwise_cross_lp_distance(
                X_validation,
                X_train,
                p=spec.p,
                squared=spec.squared,
                block_size=block_size,
                dtype=dtype,
                use_scipy_for_p1_p2=use_scipy_for_p1_p2,
            )
        )

    return FoldDistanceCache(
        fold_id=fold_id,
        train_indices=train_idx,
        validation_indices=validation_idx,
        train_distances=train_distances,
        validation_train_distances=validation_train_distances,
        y_train_transformed=y_train_transformed,
        y_validation=np.asarray(y[validation_idx], dtype=dtype),
        target_transformer=target_transformer_fold,
    )


def estimate_distance_cache_nbytes(
    fold_indices: list[tuple[NDArray, NDArray]],
    *,
    n_components: int,
    dtype: np.dtype,
) -> int:
    bytes_per_value = np.dtype(dtype).itemsize
    total_values = 0

    for train_idx, validation_idx in fold_indices:
        n_train = len(train_idx)
        n_validation = len(validation_idx)
        total_values += n_components * (
            n_train * n_train + n_validation * n_train
        )

    return int(total_values * bytes_per_value)


def validate_distance_cache_memory(
    *,
    estimated_nbytes: int,
    available_memory_nbytes: int | None,
    memory_budget_nbytes: int | None,
    memory_fraction: float,
) -> None:
    if available_memory_nbytes is None or memory_budget_nbytes is None:
        logger.warning(
            "Could not determine available system memory; proceeding with "
            "distance cache estimate of %.2f MiB.",
            estimated_nbytes / (1024**2),
        )
        return

    if estimated_nbytes <= memory_budget_nbytes:
        return

    raise MemoryError(
        "Insufficient available memory for distance pre-caching. "
        f"Estimated cache size is {format_nbytes(estimated_nbytes)}, "
        f"available memory budget is {format_nbytes(memory_budget_nbytes)} "
        f"({memory_fraction:.0%} of currently available "
        f"{format_nbytes(available_memory_nbytes)}). "
        "Disable `KRR_USE_DISTANCE_CACHE`, reduce `N_SAMPLES`, reduce the "
        "number of descriptors/folds, lower `KRR_DISTANCE_CACHE_MEMORY_FRACTION`, "
        "or use `KRR_DISTANCE_CACHE_DTYPE = \"float32\"`."
    )


def resolve_cache_n_jobs(n_jobs: int | None, *, n_folds: int) -> int:
    if n_folds <= 0:
        raise ValueError("At least one CV fold is required to build a distance cache.")

    if n_jobs is None:
        requested = effective_n_jobs(-1)
    else:
        requested = effective_n_jobs(n_jobs)

    return max(1, min(n_folds, requested))


def available_memory_bytes() -> int | None:
    psutil_available = _available_memory_from_psutil()
    if psutil_available is not None:
        return psutil_available

    proc_available = _available_memory_from_proc_meminfo()
    if proc_available is not None:
        return proc_available

    sysconf_available = _available_memory_from_sysconf()
    if sysconf_available is not None:
        return sysconf_available

    if platform.system() == "Darwin":
        return _available_memory_from_vm_stat()

    return None


def _available_memory_from_psutil() -> int | None:
    try:
        import psutil
    except ImportError:
        return None

    try:
        return int(psutil.virtual_memory().available)
    except Exception:
        return None


def _available_memory_from_proc_meminfo() -> int | None:
    try:
        with open("/proc/meminfo") as file:
            for line in file:
                if line.startswith("MemAvailable:"):
                    parts = line.split()
                    return int(parts[1]) * 1024
    except OSError:
        return None

    return None


def _available_memory_from_sysconf() -> int | None:
    try:
        pages = os.sysconf("SC_AVPHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
    except (AttributeError, OSError, ValueError):
        return None

    if pages <= 0 or page_size <= 0:
        return None

    return int(pages * page_size)


def _available_memory_from_vm_stat() -> int | None:
    try:
        result = subprocess.run(
            ["vm_stat"],
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None

    page_size_match = re.search(r"page size of (\d+) bytes", result.stdout)
    if page_size_match is None:
        return None

    page_size = int(page_size_match.group(1))
    page_counts = {}
    for line in result.stdout.splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", maxsplit=1)
        value = value.strip().rstrip(".").replace(",", "")
        if value.isdigit():
            page_counts[key.strip()] = int(value)

    available_pages = (
        page_counts.get("Pages free", 0)
        + page_counts.get("Pages inactive", 0)
        + page_counts.get("Pages speculative", 0)
        + page_counts.get("Pages purgeable", 0)
    )
    if available_pages <= 0:
        return None

    return int(available_pages * page_size)


def format_nbytes(nbytes: int) -> str:
    value = float(nbytes)
    for unit in ("bytes", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024 or unit == "TiB":
            if unit == "bytes":
                return f"{int(value)} {unit}"
            return f"{value:.2f} {unit}"
        value /= 1024


def cached_cross_val_scores(
    estimator,
    params: dict,
    cache: DistanceCache,
    *,
    scoring,
) -> NDArray:
    candidate = clone(estimator)
    if params:
        candidate.set_params(**params)

    regressor = extract_regressor(candidate)
    alpha, gammas, weights = resolve_kernel_hyperparameters(
        regressor,
        n_components=cache.n_components,
    )

    scores = [
        score_fold_from_distances(
            fold,
            alpha=alpha,
            gammas=gammas,
            weights=weights,
            scoring=scoring,
            kernel_types=cache.kernel_types,
        )
        for fold in cache.folds
    ]
    return np.asarray(scores, dtype=float)


def score_fold_from_distances(
    fold: FoldDistanceCache,
    *,
    alpha: float,
    gammas: list[float],
    weights: list[float],
    scoring,
    kernel_types: list[str],
) -> float:
    y_pred = predict_fold_from_distances(
        fold,
        alpha=alpha,
        gammas=gammas,
        weights=weights,
        kernel_types=kernel_types,
    )
    return score_predictions(fold.y_validation, y_pred, scoring)


def predict_fold_from_distances(
    fold: FoldDistanceCache,
    *,
    alpha: float,
    gammas: list[float],
    weights: list[float],
    kernel_types: list[str],
) -> NDArray:
    if alpha <= 0:
        raise ValueError(f"alpha must be positive, got {alpha}.")

    K_train = composite_kernel_from_distances(
        fold.train_distances,
        gammas=gammas,
        weights=weights,
        kernel_types=kernel_types,
    )
    K_train[np.diag_indices_from(K_train)] += alpha

    dual_coef = solve(K_train, fold.y_train_transformed, assume_a="pos")
    K_validation = composite_kernel_from_distances(
        fold.validation_train_distances,
        gammas=gammas,
        weights=weights,
        kernel_types=kernel_types,
    )
    y_pred_transformed = K_validation @ dual_coef
    return inverse_transform_target(fold.target_transformer, y_pred_transformed)


def composite_kernel_from_distances(
    distances: list[NDArray],
    *,
    gammas: list[float],
    weights: list[float],
    kernel_types: list[str],
) -> NDArray:
    if not distances:
        raise ValueError("At least one distance matrix is required.")
    if not (len(distances) == len(gammas) == len(weights) == len(kernel_types)):
        raise ValueError("distances, gammas, weights, and kernel_types must have matching lengths.")

    K_total = None
    for distance, gamma, weight, kernel_type in zip(
        distances, gammas, weights, kernel_types
    ):
        if gamma < 0:
            raise ValueError(f"gamma must be non-negative, got {gamma}.")
        if weight < 0:
            raise ValueError(f"kernel weight must be non-negative, got {weight}.")

        distance_spec_for_kernel(kernel_type)
        K = np.exp(-gamma * distance).astype(distance.dtype, copy=False)
        K *= weight
        if K_total is None:
            K_total = K
        else:
            K_total += K

    return K_total


def score_predictions(y_true, y_pred, scoring) -> float:
    if isinstance(scoring, str):
        scorer = get_scorer(scoring)
    else:
        scorer = scoring

    estimator = _PredictionOnlyRegressor(np.asarray(y_pred, dtype=float).reshape(-1))
    X_dummy = np.zeros((len(estimator.y_pred), 1), dtype=float)
    return float(scorer(estimator, X_dummy, np.asarray(y_true, dtype=float).reshape(-1)))


def fit_target_transformer(transformer, y):
    y = np.asarray(y, dtype=float).reshape(-1, 1)
    if transformer is None:
        return None, y.reshape(-1)

    fitted = clone(transformer)
    y_transformed = fitted.fit_transform(y)
    return fitted, np.asarray(y_transformed, dtype=float).reshape(-1)


def inverse_transform_target(transformer, y):
    y = np.asarray(y, dtype=float).reshape(-1, 1)
    if transformer is None:
        return y.reshape(-1)

    return np.asarray(transformer.inverse_transform(y), dtype=float).reshape(-1)


def unpack_sample_matrix(X, dtype: np.dtype | type | str = np.float64) -> list[NDArray]:
    X = np.asarray(X, dtype=object)
    dtype = np.dtype(dtype)
    if X.ndim != 2:
        raise ValueError(
            "Expected X with shape (n_samples, n_descriptors), "
            f"got {X.shape}."
        )
    if X.shape[0] == 0:
        raise ValueError("X must contain at least one sample.")
    if X.shape[1] == 0:
        raise ValueError("X must contain at least one descriptor block.")

    blocks = []
    for block_id in range(X.shape[1]):
        block = np.stack(X[:, block_id]).astype(dtype)
        blocks.append(block.reshape(X.shape[0], -1))

    return blocks


def distance_spec_for_kernel(kernel_type: str) -> KernelDistanceSpec:
    kernel_type = kernel_type.lower()
    if kernel_type == "rbf":
        return KernelDistanceSpec(kernel_type=kernel_type, p=2, squared=True)
    if kernel_type == "laplacian":
        return KernelDistanceSpec(kernel_type=kernel_type, p=1, squared=False)
    raise UnsupportedDistanceKernelError(
        f'Kernel "{kernel_type}" is not supported by the distance cache.'
    )


def extract_regressor(estimator):
    return getattr(estimator, "regressor", estimator)


def extract_target_transformer(estimator):
    return getattr(estimator, "transformer", None)


def resolve_kernel_hyperparameters(regressor, *, n_components: int):
    alpha = float(regressor.alpha)
    gammas = resolve_sequence("gammas", regressor.gammas, n_components, 1.0)
    weights = resolve_sequence("kernel_weights", regressor.kernel_weights, n_components, 1.0)

    gammas = list(np.asarray(gammas, dtype=float))
    weights = np.asarray(weights, dtype=float)
    if getattr(regressor, "normalize_kernel_weights", False):
        weight_sum = weights.sum()
        if weight_sum <= 0:
            raise ValueError("Cannot normalize kernel weights with non-positive sum.")
        weights = weights / weight_sum

    return alpha, gammas, list(weights)


def resolve_sequence(name, values, n_components, default):
    if values is None:
        if name == "names":
            return [f"{default}{i}" for i in range(n_components)]
        return [default] * n_components

    if isinstance(values, str):
        values = [values] * n_components
    elif np.isscalar(values):
        values = [values] * n_components
    else:
        values = list(values)

    if len(values) != n_components:
        raise ValueError(f"{name} must have length {n_components}, got {len(values)}.")

    return values
