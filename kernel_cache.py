from dataclasses import dataclass
import logging
import os
import platform
import re
import subprocess
import threading

from joblib import Parallel, delayed, effective_n_jobs
import numpy as np
from numpy.typing import NDArray
from scipy.linalg import cho_factor, cho_solve
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.metrics import get_scorer
from threadpoolctl import threadpool_limits

from config import VERBOSITY
from kernels import (
    pairwise_cross_lp_distance,
    pairwise_cross_lp_distance_pytorch,
    pairwise_self_lp_distance,
    pairwise_self_lp_distance_pytorch,
)
from preprocess import make_data_preprocessor
from target_utils import (
    align_targets_for_scoring,
    as_target_array,
    as_target_matrix,
)
from utilities import configure_logging

configure_logging(VERBOSITY)
logger = logging.getLogger("kernel-cache")
_kernel_work_buffers = threading.local()


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
    train_distances: list[object]
    validation_train_distances: list[object]
    y_train_transformed: NDArray
    y_validation: NDArray
    target_transformer: object | None


@dataclass(frozen=True)
class _FoldDistanceMetadata:
    fold_id: int
    train_indices: NDArray
    validation_indices: NDArray
    y_train_transformed: NDArray
    y_validation: NDArray
    target_transformer: object | None


@dataclass(frozen=True)
class _ComponentFoldDistanceCache:
    fold_index: int
    component_index: int
    train_distance: object
    validation_train_distance: object


@dataclass(frozen=True)
class DistanceCache:
    folds: list[FoldDistanceCache]
    names: list[str]
    kernel_types: list[str]
    normalizations: list[str]
    pca_components: list
    pca_whiten: list[bool]
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
            total += sum(array_nbytes(distance) for distance in fold.train_distances)
            total += sum(
                array_nbytes(distance)
                for distance in fold.validation_train_distances
            )
        return total


class UnsupportedDistanceKernelError(ValueError):
    pass


def normalize_distance_backend(backend: str) -> str:
    backend = str(backend).lower()
    if backend in {"numpy", "np", "cpu"}:
        return "numpy"
    if backend in {"pytorch", "torch", "gpu", "cuda", "rocm"}:
        return "pytorch"
    raise ValueError(
        'distance backend must be "numpy" or "pytorch", '
        f"got {backend!r}."
    )


def array_nbytes(array) -> int:
    nbytes = getattr(array, "nbytes", None)
    if nbytes is not None:
        return int(nbytes)

    if hasattr(array, "numel") and hasattr(array, "element_size"):
        return int(array.numel() * array.element_size())

    return int(np.asarray(array).nbytes)


def resolve_cache_pytorch_devices(
    *,
    distance_backend: str,
    pytorch_device: str | None,
    pytorch_devices,
    n_folds: int,
):
    if distance_backend != "pytorch":
        return None

    from pytorch_backend import require_torch, resolve_torch_devices

    torch = require_torch()
    return resolve_torch_devices(
        torch,
        pytorch_devices,
        fallback_device=pytorch_device,
        max_devices=n_folds,
    )


def pytorch_device_for_fold(resolved_devices, fold_index: int, fallback_device):
    if not resolved_devices:
        return fallback_device

    return resolved_devices[fold_index % len(resolved_devices)]


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
    pca_components=None,
    pca_whiten=False,
    target_transformer=None,
    block_size: int = 1024,
    dtype: np.dtype | type = np.float64,
    use_scipy_for_p1_p2: bool = True,
    n_jobs: int | None = -1,
    memory_fraction: float = 0.80,
    distance_backend: str = "numpy",
    pytorch_device: str | None = "auto",
    pytorch_devices=None,
) -> DistanceCache:
    dtype = np.dtype(dtype)
    distance_backend = normalize_distance_backend(distance_backend)
    X_blocks = unpack_sample_matrix(X, dtype=dtype)
    y = as_target_array(y, dtype=dtype)
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
    pca_components = resolve_sequence(
        "pca_components",
        pca_components,
        len(names),
        None,
    )
    pca_whiten = resolve_sequence("pca_whiten", pca_whiten, len(names), False)
    if not (
        len(names)
        == len(kernel_types)
        == len(normalizations)
        == len(pca_components)
        == len(pca_whiten)
    ):
        raise ValueError(
            "names, kernel_types, normalizations, pca_components, and "
            "pca_whiten must have the same length."
        )

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

    fold_metadata = [
        build_fold_distance_metadata(
            fold_id=fold_id,
            train_idx=train_idx,
            validation_idx=validation_idx,
            y=y,
            target_transformer=target_transformer,
            dtype=dtype,
        )
        for fold_id, (train_idx, validation_idx) in enumerate(
            fold_indices,
            start=1,
        )
    ]
    n_cache_tasks = len(fold_metadata) * len(X_blocks)
    cache_n_jobs = resolve_cache_n_jobs(n_jobs, n_tasks=n_cache_tasks)
    if distance_backend == "pytorch":
        cache_n_jobs = 1
    resolved_pytorch_devices = resolve_cache_pytorch_devices(
        distance_backend=distance_backend,
        pytorch_device=pytorch_device,
        pytorch_devices=pytorch_devices,
        n_folds=len(fold_metadata),
    )
    logger.debug(
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
        "Precomputing %s fold x %s descriptor distance cache with %s "
        "backend and %s worker thread(s).",
        len(fold_metadata),
        len(X_blocks),
        distance_backend,
        cache_n_jobs,
    )

    if cache_n_jobs == 1:
        component_results = [
            build_component_fold_distance_cache(
                fold_index=fold_index,
                component_index=component_index,
                train_idx=metadata.train_indices,
                validation_idx=metadata.validation_indices,
                X_block=X_block,
                normalization=normalization,
                pca_components=pca_component,
                pca_whiten=pca_whiten_block,
                spec=spec,
                block_size=block_size,
                dtype=dtype,
                use_scipy_for_p1_p2=use_scipy_for_p1_p2,
                distance_backend=distance_backend,
                pytorch_device=pytorch_device_for_fold(
                    resolved_pytorch_devices,
                    fold_index,
                    pytorch_device,
                ),
            )
            for fold_index, metadata in enumerate(fold_metadata)
            for component_index, (
                X_block,
                normalization,
                pca_component,
                pca_whiten_block,
                spec,
            ) in enumerate(
                zip(X_blocks, normalizations, pca_components, pca_whiten, specs)
            )
        ]
    else:
        with threadpool_limits(limits=1):
            component_results = Parallel(n_jobs=cache_n_jobs, prefer="threads")(
                delayed(build_component_fold_distance_cache)(
                    fold_index=fold_index,
                    component_index=component_index,
                    train_idx=metadata.train_indices,
                    validation_idx=metadata.validation_indices,
                    X_block=X_block,
                    normalization=normalization,
                    pca_components=pca_component,
                    pca_whiten=pca_whiten_block,
                    spec=spec,
                    block_size=block_size,
                    dtype=dtype,
                    use_scipy_for_p1_p2=use_scipy_for_p1_p2,
                    distance_backend=distance_backend,
                    pytorch_device=pytorch_device_for_fold(
                        resolved_pytorch_devices,
                        fold_index,
                        pytorch_device,
                    ),
                )
                for fold_index, metadata in enumerate(fold_metadata)
                for component_index, (
                    X_block,
                    normalization,
                    pca_component,
                    pca_whiten_block,
                    spec,
                ) in enumerate(
                    zip(X_blocks, normalizations, pca_components, pca_whiten, specs)
                )
            )

    folds = assemble_fold_distance_caches(
        fold_metadata,
        component_results,
        n_components=len(X_blocks),
    )

    cache = DistanceCache(
        folds=folds,
        names=list(names),
        kernel_types=list(kernel_types),
        normalizations=list(normalizations),
        pca_components=list(pca_components),
        pca_whiten=list(pca_whiten),
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


def build_fold_distance_metadata(
    *,
    fold_id: int,
    train_idx: NDArray,
    validation_idx: NDArray,
    y: NDArray,
    target_transformer,
    dtype: np.dtype,
) -> _FoldDistanceMetadata:
    target_transformer_fold, y_train_transformed = fit_target_transformer(
        target_transformer,
        y[train_idx],
    )

    return _FoldDistanceMetadata(
        fold_id=fold_id,
        train_indices=train_idx,
        validation_indices=validation_idx,
        y_train_transformed=np.asarray(y_train_transformed, dtype=dtype),
        y_validation=np.asarray(y[validation_idx], dtype=dtype),
        target_transformer=target_transformer_fold,
    )


def build_component_fold_distance_cache(
    *,
    fold_index: int,
    component_index: int,
    train_idx: NDArray,
    validation_idx: NDArray,
    X_block: NDArray,
    normalization: str,
    pca_components,
    pca_whiten: bool,
    spec: KernelDistanceSpec,
    block_size: int,
    dtype: np.dtype,
    use_scipy_for_p1_p2: bool,
    distance_backend: str = "numpy",
    pytorch_device: str | None = "auto",
) -> _ComponentFoldDistanceCache:
    preprocessor = make_data_preprocessor(
        normalization,
        pca_components=pca_components,
        pca_whiten=pca_whiten,
    )
    X_train = preprocessor.fit_transform(X_block[train_idx])
    X_validation = preprocessor.transform(X_block[validation_idx])

    if normalize_distance_backend(distance_backend) == "pytorch":
        train_distance = pairwise_self_lp_distance_pytorch(
            X_train,
            p=spec.p,
            squared=spec.squared,
            block_size=block_size,
            dtype=dtype,
            device=pytorch_device,
        )
        validation_train_distance = pairwise_cross_lp_distance_pytorch(
            X_validation,
            X_train,
            p=spec.p,
            squared=spec.squared,
            block_size=block_size,
            dtype=dtype,
            device=pytorch_device,
        )
    else:
        train_distance = pairwise_self_lp_distance(
            X_train,
            p=spec.p,
            squared=spec.squared,
            block_size=block_size,
            dtype=dtype,
            use_scipy_for_p1_p2=use_scipy_for_p1_p2,
        )
        validation_train_distance = pairwise_cross_lp_distance(
            X_validation,
            X_train,
            p=spec.p,
            squared=spec.squared,
            block_size=block_size,
            dtype=dtype,
            use_scipy_for_p1_p2=use_scipy_for_p1_p2,
        )

    return _ComponentFoldDistanceCache(
        fold_index=fold_index,
        component_index=component_index,
        train_distance=train_distance,
        validation_train_distance=validation_train_distance,
    )


def assemble_fold_distance_caches(
    fold_metadata: list[_FoldDistanceMetadata],
    component_results: list[_ComponentFoldDistanceCache],
    *,
    n_components: int,
) -> list[FoldDistanceCache]:
    result_by_key = {
        (result.fold_index, result.component_index): result
        for result in component_results
    }

    folds = []
    for fold_index, metadata in enumerate(fold_metadata):
        train_distances = []
        validation_train_distances = []
        for component_index in range(n_components):
            try:
                result = result_by_key[(fold_index, component_index)]
            except KeyError as exc:
                raise RuntimeError(
                    "Missing distance cache result for "
                    f"fold {metadata.fold_id}, descriptor {component_index}."
                ) from exc

            train_distances.append(result.train_distance)
            validation_train_distances.append(result.validation_train_distance)

        folds.append(
            FoldDistanceCache(
                fold_id=metadata.fold_id,
                train_indices=metadata.train_indices,
                validation_indices=metadata.validation_indices,
                train_distances=train_distances,
                validation_train_distances=validation_train_distances,
                y_train_transformed=metadata.y_train_transformed,
                y_validation=metadata.y_validation,
                target_transformer=metadata.target_transformer,
            )
        )

    return folds


def build_fold_distance_cache(
    *,
    fold_id: int,
    train_idx: NDArray,
    validation_idx: NDArray,
    X_blocks: list[NDArray],
    y: NDArray,
    specs: list[KernelDistanceSpec],
    normalizations: list[str],
    pca_components=None,
    pca_whiten=False,
    target_transformer=None,
    block_size: int,
    dtype: np.dtype,
    use_scipy_for_p1_p2: bool,
) -> FoldDistanceCache:
    pca_components = resolve_sequence(
        "pca_components",
        pca_components,
        len(X_blocks),
        None,
    )
    pca_whiten = resolve_sequence("pca_whiten", pca_whiten, len(X_blocks), False)

    metadata = build_fold_distance_metadata(
        fold_id=fold_id,
        train_idx=train_idx,
        validation_idx=validation_idx,
        y=y,
        target_transformer=target_transformer,
        dtype=dtype,
    )

    component_results = [
        build_component_fold_distance_cache(
            fold_index=0,
            component_index=component_index,
            train_idx=train_idx,
            validation_idx=validation_idx,
            X_block=X_block,
            normalization=normalization,
            pca_components=pca_component,
            pca_whiten=pca_whiten_block,
            spec=spec,
            block_size=block_size,
            dtype=dtype,
            use_scipy_for_p1_p2=use_scipy_for_p1_p2,
        )
        for component_index, (
            X_block,
            normalization,
            pca_component,
            pca_whiten_block,
            spec,
        ) in enumerate(
            zip(X_blocks, normalizations, pca_components, pca_whiten, specs)
        )
    ]

    return assemble_fold_distance_caches(
        [metadata],
        component_results,
        n_components=len(X_blocks),
    )[0]


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


def resolve_cache_n_jobs(
    n_jobs: int | None,
    *,
    n_tasks: int | None = None,
    n_folds: int | None = None,
) -> int:
    if n_tasks is None:
        n_tasks = n_folds
    if n_tasks is None or n_tasks <= 0:
        raise ValueError("At least one task is required to build a distance cache.")

    if n_jobs is None:
        requested = effective_n_jobs(-1)
    else:
        requested = effective_n_jobs(n_jobs)

    return max(1, min(n_tasks, requested))


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


def score_fold_from_distances_pytorch(
    fold: FoldDistanceCache,
    *,
    alpha: float,
    gammas: list[float],
    weights: list[float],
    scoring,
    kernel_types: list[str],
    device: str | None = "auto",
    dtype=None,
) -> float:
    from pytorch_backend import score_fold_from_distances_pytorch as _impl

    return _impl(
        fold,
        alpha=alpha,
        gammas=gammas,
        weights=weights,
        scoring=scoring,
        kernel_types=kernel_types,
        device=device,
        dtype=dtype,
    )


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

    work_buffers = thread_kernel_work_buffers()
    K_train = composite_kernel_from_distances(
        fold.train_distances,
        gammas=gammas,
        weights=weights,
        kernel_types=kernel_types,
        out=work_buffer_for(
            work_buffers,
            fold.train_distances[0].shape,
            fold.train_distances[0].dtype,
            "total",
        ),
        temp=work_buffer_for(
            work_buffers,
            fold.train_distances[0].shape,
            fold.train_distances[0].dtype,
            "component",
        ),
    )
    K_train[np.diag_indices_from(K_train)] += alpha

    cholesky_factor = cho_factor(
        K_train,
        lower=True,
        overwrite_a=True,
        check_finite=False,
    )
    dual_coef = cho_solve(
        cholesky_factor,
        fold.y_train_transformed,
        check_finite=False,
    )
    K_validation = composite_kernel_from_distances(
        fold.validation_train_distances,
        gammas=gammas,
        weights=weights,
        kernel_types=kernel_types,
        out=work_buffer_for(
            work_buffers,
            fold.validation_train_distances[0].shape,
            fold.validation_train_distances[0].dtype,
            "total",
        ),
        temp=work_buffer_for(
            work_buffers,
            fold.validation_train_distances[0].shape,
            fold.validation_train_distances[0].dtype,
            "component",
        ),
    )
    y_pred_transformed = K_validation @ dual_coef
    return inverse_transform_target(fold.target_transformer, y_pred_transformed)


def predict_fold_from_distances_pytorch(
    fold: FoldDistanceCache,
    *,
    alpha: float,
    gammas: list[float],
    weights: list[float],
    kernel_types: list[str],
    device: str | None = "auto",
    dtype=None,
) -> NDArray:
    from pytorch_backend import predict_fold_from_distances_pytorch as _impl

    return _impl(
        fold,
        alpha=alpha,
        gammas=gammas,
        weights=weights,
        kernel_types=kernel_types,
        device=device,
        dtype=dtype,
    )


def composite_kernel_from_distances(
    distances: list[NDArray],
    *,
    gammas: list[float],
    weights: list[float],
    kernel_types: list[str],
    out: NDArray | None = None,
    temp: NDArray | None = None,
) -> NDArray:
    if not distances:
        raise ValueError("At least one distance matrix is required.")
    if not (len(distances) == len(gammas) == len(weights) == len(kernel_types)):
        raise ValueError("distances, gammas, weights, and kernel_types must have matching lengths.")

    first_distance = distances[0]
    if out is None:
        out = np.empty_like(first_distance)
    elif out.shape != first_distance.shape:
        raise ValueError(f"out has shape {out.shape}, expected {first_distance.shape}.")

    if len(distances) > 1:
        if temp is None:
            temp = np.empty_like(first_distance)
        elif temp.shape != first_distance.shape:
            raise ValueError(
                f"temp has shape {temp.shape}, expected {first_distance.shape}."
            )

    for component_index, (distance, gamma, weight, kernel_type) in enumerate(
        zip(distances, gammas, weights, kernel_types)
    ):
        if gamma < 0:
            raise ValueError(f"gamma must be non-negative, got {gamma}.")
        if weight < 0:
            raise ValueError(f"kernel weight must be non-negative, got {weight}.")
        if distance.shape != first_distance.shape:
            raise ValueError(
                "All distance matrices in one composite kernel must have the same "
                f"shape; got {distance.shape} and {first_distance.shape}."
            )

        distance_spec_for_kernel(kernel_type)
        component = out if component_index == 0 else temp
        np.multiply(distance, -gamma, out=component, casting="unsafe")
        np.exp(component, out=component)
        component *= weight

        if component_index > 0:
            out += component

    return out


def composite_kernel_from_distances_pytorch(
    distances,
    *,
    gammas,
    weights,
    kernel_types: list[str],
):
    from pytorch_backend import composite_kernel_from_distances_pytorch as _impl

    return _impl(
        distances,
        gammas=gammas,
        weights=weights,
        kernel_types=kernel_types,
    )


def thread_kernel_work_buffers() -> dict:
    buffers = getattr(_kernel_work_buffers, "buffers", None)
    if buffers is None:
        buffers = {}
        _kernel_work_buffers.buffers = buffers

    return buffers


def work_buffer_for(
    buffers: dict,
    shape: tuple[int, ...],
    dtype: np.dtype | type | str,
    role: str,
) -> NDArray:
    dtype = np.dtype(dtype)
    key = (shape, dtype.str, role)
    buffer = buffers.get(key)
    if buffer is None:
        buffer = np.empty(shape, dtype=dtype)
        buffers[key] = buffer
    return buffer


def score_predictions(y_true, y_pred, scoring) -> float:
    if isinstance(scoring, str):
        scorer = get_scorer(scoring)
    else:
        scorer = scoring

    y_true_aligned, y_pred_aligned = align_targets_for_scoring(y_true, y_pred)
    estimator = _PredictionOnlyRegressor(y_pred_aligned)
    X_dummy = np.zeros((len(y_true_aligned), 1), dtype=float)
    return float(scorer(estimator, X_dummy, y_true_aligned))


def fit_target_transformer(transformer, y):
    y = as_target_matrix(y, dtype=float)
    if transformer is None:
        return None, y

    fitted = clone(transformer)
    y_transformed = fitted.fit_transform(y)
    return fitted, as_target_matrix(y_transformed, dtype=float)


def inverse_transform_target(transformer, y):
    y = as_target_matrix(y, dtype=float)
    if transformer is None:
        return y

    return as_target_matrix(transformer.inverse_transform(y), dtype=float)


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
    if not isinstance(kernel_type, str):
        raise UnsupportedDistanceKernelError(
            f"Kernel type must be a string, got {type(kernel_type).__name__}."
        )

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
