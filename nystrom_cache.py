from dataclasses import dataclass
from contextlib import nullcontext
from concurrent.futures import ThreadPoolExecutor

from joblib import Parallel, delayed
import numpy as np
from numpy.typing import NDArray
from threadpoolctl import threadpool_limits

from kernel_cache import (
    UnsupportedDistanceKernelError,
    array_nbytes,
    available_memory_bytes,
    distance_spec_for_kernel,
    fit_target_transformer,
    format_nbytes,
    inverse_transform_target,
    normalize_distance_backend,
    pytorch_device_for_fold,
    resolve_cache_n_jobs,
    resolve_cache_pytorch_devices,
    resolve_sequence,
    score_predictions,
    unpack_sample_matrix,
    validate_distance_cache_memory,
)
from kernels import (
    pairwise_cross_lp_distance,
    pairwise_cross_lp_distance_pytorch,
    pairwise_self_lp_distance,
    pairwise_self_lp_distance_pytorch,
)
from nystrom_krr import select_landmark_indices
from preprocess import make_data_preprocessor
from target_utils import as_target_array, as_target_matrix


@dataclass(frozen=True)
class NystromFoldDistanceCache:
    fold_id: int
    train_indices: NDArray
    validation_indices: NDArray
    landmark_local_indices: NDArray
    train_landmark_distances: list[object]
    validation_landmark_distances: list[object]
    landmark_distances: list[object]
    y_train_transformed: object
    y_validation: NDArray
    target_transformer: object | None
    batch_size: int
    eigenvalue_floor: float

    @property
    def train_distances(self) -> list[object]:
        return self.train_landmark_distances

    @property
    def validation_train_distances(self) -> list[object]:
        return self.validation_landmark_distances


@dataclass(frozen=True)
class _NystromFoldMetadata:
    fold_id: int
    train_indices: NDArray
    validation_indices: NDArray
    landmark_local_indices: NDArray
    y_train_transformed: NDArray
    y_validation: NDArray
    target_transformer: object | None


@dataclass(frozen=True)
class _ComponentFoldNystromDistanceCache:
    fold_index: int
    component_index: int
    train_landmark_distance: object
    validation_landmark_distance: object
    landmark_distance: object


@dataclass(frozen=True)
class NystromDistanceCache:
    folds: list[NystromFoldDistanceCache]
    names: list[str]
    kernel_types: list[str]
    normalizations: list[str]
    pca_components: list
    pca_whiten: list[bool]
    n_landmarks: int
    landmark_selection: str
    batch_size: int
    eigenvalue_floor: float
    backend: str
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
            total += sum(array_nbytes(distance) for distance in fold.train_landmark_distances)
            total += sum(
                array_nbytes(distance)
                for distance in fold.validation_landmark_distances
            )
            total += sum(array_nbytes(distance) for distance in fold.landmark_distances)
        return total


def is_nystrom_distance_cache(cache) -> bool:
    return isinstance(cache, NystromDistanceCache)


def is_nystrom_regressor(regressor) -> bool:
    return regressor.__class__.__name__ == "CompositeNystromKRREstimator"


def build_nystrom_distance_cache(
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
    n_landmarks: int,
    landmark_selection: str,
    batch_size: int,
    eigenvalue_floor: float,
    random_state=None,
    block_size: int = 1024,
    dtype: np.dtype | type = np.float64,
    use_scipy_for_p1_p2: bool = True,
    n_jobs: int | None = -1,
    memory_fraction: float = 0.80,
    distance_backend: str = "numpy",
    pytorch_device: str | None = "auto",
    pytorch_devices=None,
) -> NystromDistanceCache:
    dtype = np.dtype(dtype)
    distance_backend = normalize_distance_backend(distance_backend)
    X_blocks = unpack_sample_matrix(X, dtype=dtype)
    y = as_target_array(y, dtype=dtype)
    if type(n_landmarks) is not int or n_landmarks <= 0:
        raise ValueError("n_landmarks must be a positive int.")
    if type(batch_size) is not int or batch_size <= 0:
        raise ValueError("batch_size must be a positive int.")
    if eigenvalue_floor < 0:
        raise ValueError("eigenvalue_floor must be non-negative.")
    if not 0 < memory_fraction <= 1:
        raise ValueError("memory_fraction must be in (0, 1].")
    if n_jobs == 0:
        raise ValueError("n_jobs must be None or a non-zero int.")

    pca_components = resolve_sequence("pca_components", pca_components, len(names), None)
    pca_whiten = resolve_sequence("pca_whiten", pca_whiten, len(names), False)
    if not (
        len(X_blocks)
        == len(names)
        == len(kernel_types)
        == len(normalizations)
        == len(pca_components)
        == len(pca_whiten)
    ):
        raise ValueError(
            "descriptor blocks, names, kernel_types, normalizations, "
            "pca_components, and pca_whiten must have matching lengths."
        )
    if y.shape[0] != X_blocks[0].shape[0]:
        raise ValueError(f"X has {X_blocks[0].shape[0]} samples, but y has {y.shape[0]}.")

    specs = [distance_spec_for_kernel(kernel_type) for kernel_type in kernel_types]
    fold_indices = [
        (np.asarray(train_idx, dtype=int), np.asarray(validation_idx, dtype=int))
        for train_idx, validation_idx in cv.split(X, y)
    ]
    fold_metadata = [
        build_nystrom_fold_metadata(
            fold_id=fold_id,
            train_idx=train_idx,
            validation_idx=validation_idx,
            y=y,
            target_transformer=target_transformer,
            dtype=dtype,
            n_landmarks=n_landmarks,
            landmark_selection=landmark_selection,
            random_state=random_state,
        )
        for fold_id, (train_idx, validation_idx) in enumerate(fold_indices, start=1)
    ]
    estimated_nbytes = estimate_nystrom_cache_nbytes(
        fold_metadata,
        n_components=len(names),
        dtype=dtype,
    )
    available_memory_nbytes = available_memory_bytes()
    memory_budget_nbytes = (
        None
        if available_memory_nbytes is None
        else int(available_memory_nbytes * memory_fraction)
    )
    validate_nystrom_cache_memory(
        estimated_nbytes=estimated_nbytes,
        available_memory_nbytes=available_memory_nbytes,
        memory_budget_nbytes=memory_budget_nbytes,
        memory_fraction=memory_fraction,
    )

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

    if cache_n_jobs == 1:
        component_results = [
            build_component_fold_nystrom_distance_cache(
                fold_index=fold_index,
                component_index=component_index,
                train_idx=metadata.train_indices,
                validation_idx=metadata.validation_indices,
                landmark_local_indices=metadata.landmark_local_indices,
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
            ) in enumerate(zip(X_blocks, normalizations, pca_components, pca_whiten, specs))
        ]
    else:
        with threadpool_limits(limits=1):
            component_results = Parallel(n_jobs=cache_n_jobs, prefer="threads")(
                delayed(build_component_fold_nystrom_distance_cache)(
                    fold_index=fold_index,
                    component_index=component_index,
                    train_idx=metadata.train_indices,
                    validation_idx=metadata.validation_indices,
                    landmark_local_indices=metadata.landmark_local_indices,
                    X_block=X_block,
                    normalization=normalization,
                    pca_components=pca_component,
                    pca_whiten=pca_whiten_block,
                    spec=spec,
                    block_size=block_size,
                    dtype=dtype,
                    use_scipy_for_p1_p2=use_scipy_for_p1_p2,
                    distance_backend=distance_backend,
                    pytorch_device=pytorch_device,
                )
                for fold_index, metadata in enumerate(fold_metadata)
                for component_index, (
                    X_block,
                    normalization,
                    pca_component,
                    pca_whiten_block,
                    spec,
                ) in enumerate(zip(X_blocks, normalizations, pca_components, pca_whiten, specs))
            )

    cache = NystromDistanceCache(
        folds=assemble_nystrom_distance_caches(
            fold_metadata,
            component_results,
            n_components=len(X_blocks),
            batch_size=batch_size,
            eigenvalue_floor=float(eigenvalue_floor),
        ),
        names=list(names),
        kernel_types=list(kernel_types),
        normalizations=list(normalizations),
        pca_components=list(pca_components),
        pca_whiten=list(pca_whiten),
        n_landmarks=int(n_landmarks),
        landmark_selection=str(landmark_selection),
        batch_size=int(batch_size),
        eigenvalue_floor=float(eigenvalue_floor),
        backend="numpy",
        estimated_nbytes=estimated_nbytes,
        available_memory_nbytes=available_memory_nbytes,
        memory_budget_nbytes=memory_budget_nbytes,
        n_jobs=cache_n_jobs,
    )
    if distance_backend == "pytorch":
        return nystrom_cache_to_pytorch(
            cache,
            device=pytorch_device,
            devices=pytorch_devices,
            dtype=dtype,
        )
    return cache


def build_nystrom_fold_metadata(
    *,
    fold_id: int,
    train_idx: NDArray,
    validation_idx: NDArray,
    y: NDArray,
    target_transformer,
    dtype: np.dtype,
    n_landmarks: int,
    landmark_selection: str,
    random_state=None,
) -> _NystromFoldMetadata:
    target_transformer_fold, y_train_transformed = fit_target_transformer(
        target_transformer,
        y[train_idx],
    )
    landmark_local_indices = select_landmark_indices(
        len(train_idx),
        n_landmarks=n_landmarks,
        selection=landmark_selection,
        random_state=random_state,
    )

    return _NystromFoldMetadata(
        fold_id=fold_id,
        train_indices=train_idx,
        validation_indices=validation_idx,
        landmark_local_indices=landmark_local_indices,
        y_train_transformed=np.asarray(y_train_transformed, dtype=dtype),
        y_validation=np.asarray(y[validation_idx], dtype=dtype),
        target_transformer=target_transformer_fold,
    )


def build_component_fold_nystrom_distance_cache(
    *,
    fold_index: int,
    component_index: int,
    train_idx: NDArray,
    validation_idx: NDArray,
    landmark_local_indices: NDArray,
    X_block: NDArray,
    normalization: str,
    pca_components,
    pca_whiten: bool,
    spec,
    block_size: int,
    dtype: np.dtype,
    use_scipy_for_p1_p2: bool,
    distance_backend: str = "numpy",
    pytorch_device: str | None = "auto",
) -> _ComponentFoldNystromDistanceCache:
    preprocessor = make_data_preprocessor(
        normalization,
        pca_components=pca_components,
        pca_whiten=pca_whiten,
    )
    X_train = preprocessor.fit_transform(X_block[train_idx])
    X_validation = preprocessor.transform(X_block[validation_idx])
    X_landmarks = X_train[landmark_local_indices]

    if normalize_distance_backend(distance_backend) == "pytorch":
        device = None if pytorch_device is None else str(pytorch_device)
        train_landmark_distance = pairwise_cross_lp_distance_pytorch(
            X_train,
            X_landmarks,
            p=spec.p,
            squared=spec.squared,
            block_size=block_size,
            dtype=dtype,
            device=device,
        )
        validation_landmark_distance = pairwise_cross_lp_distance_pytorch(
            X_validation,
            X_landmarks,
            p=spec.p,
            squared=spec.squared,
            block_size=block_size,
            dtype=dtype,
            device=device,
        )
        landmark_distance = pairwise_self_lp_distance_pytorch(
            X_landmarks,
            p=spec.p,
            squared=spec.squared,
            block_size=block_size,
            dtype=dtype,
            device=device,
        )
    else:
        train_landmark_distance = pairwise_cross_lp_distance(
            X_train,
            X_landmarks,
            p=spec.p,
            squared=spec.squared,
            block_size=block_size,
            dtype=dtype,
            use_scipy_for_p1_p2=use_scipy_for_p1_p2,
        )
        validation_landmark_distance = pairwise_cross_lp_distance(
            X_validation,
            X_landmarks,
            p=spec.p,
            squared=spec.squared,
            block_size=block_size,
            dtype=dtype,
            use_scipy_for_p1_p2=use_scipy_for_p1_p2,
        )
        landmark_distance = pairwise_self_lp_distance(
            X_landmarks,
            p=spec.p,
            squared=spec.squared,
            block_size=block_size,
            dtype=dtype,
            use_scipy_for_p1_p2=use_scipy_for_p1_p2,
        )

    return _ComponentFoldNystromDistanceCache(
        fold_index=fold_index,
        component_index=component_index,
        train_landmark_distance=train_landmark_distance,
        validation_landmark_distance=validation_landmark_distance,
        landmark_distance=landmark_distance,
    )


def assemble_nystrom_distance_caches(
    fold_metadata: list[_NystromFoldMetadata],
    component_results: list[_ComponentFoldNystromDistanceCache],
    *,
    n_components: int,
    batch_size: int,
    eigenvalue_floor: float,
) -> list[NystromFoldDistanceCache]:
    result_by_key = {
        (result.fold_index, result.component_index): result
        for result in component_results
    }

    folds = []
    for fold_index, metadata in enumerate(fold_metadata):
        train_landmark_distances = []
        validation_landmark_distances = []
        landmark_distances = []
        for component_index in range(n_components):
            try:
                result = result_by_key[(fold_index, component_index)]
            except KeyError as exc:
                raise RuntimeError(
                    "Missing Nyström distance cache result for "
                    f"fold {metadata.fold_id}, descriptor {component_index}."
                ) from exc

            train_landmark_distances.append(result.train_landmark_distance)
            validation_landmark_distances.append(result.validation_landmark_distance)
            landmark_distances.append(result.landmark_distance)

        folds.append(
            NystromFoldDistanceCache(
                fold_id=metadata.fold_id,
                train_indices=metadata.train_indices,
                validation_indices=metadata.validation_indices,
                landmark_local_indices=metadata.landmark_local_indices,
                train_landmark_distances=train_landmark_distances,
                validation_landmark_distances=validation_landmark_distances,
                landmark_distances=landmark_distances,
                y_train_transformed=metadata.y_train_transformed,
                y_validation=metadata.y_validation,
                target_transformer=metadata.target_transformer,
                batch_size=batch_size,
                eigenvalue_floor=eigenvalue_floor,
            )
        )
    return folds


def estimate_nystrom_cache_nbytes(
    fold_metadata: list[_NystromFoldMetadata],
    *,
    n_components: int,
    dtype: np.dtype,
) -> int:
    bytes_per_value = np.dtype(dtype).itemsize
    total_values = 0
    for metadata in fold_metadata:
        n_train = len(metadata.train_indices)
        n_validation = len(metadata.validation_indices)
        n_landmarks = len(metadata.landmark_local_indices)
        total_values += n_components * (
            n_train * n_landmarks
            + n_validation * n_landmarks
            + n_landmarks * n_landmarks
        )
    return int(total_values * bytes_per_value)


def validate_nystrom_cache_memory(
    *,
    estimated_nbytes: int,
    available_memory_nbytes: int | None,
    memory_budget_nbytes: int | None,
    memory_fraction: float,
) -> None:
    try:
        validate_distance_cache_memory(
            estimated_nbytes=estimated_nbytes,
            available_memory_nbytes=available_memory_nbytes,
            memory_budget_nbytes=memory_budget_nbytes,
            memory_fraction=memory_fraction,
        )
    except MemoryError as exc:
        raise MemoryError(
            "Insufficient available memory for Nyström landmark pre-caching. "
            f"Estimated cache size is {format_nbytes(estimated_nbytes)}. "
            "Reduce `KRR_NYSTROM_N_LANDMARKS`, reduce descriptors/folds, lower "
            "`KRR_DISTANCE_CACHE_MEMORY_FRACTION`, or use float32."
        ) from exc


def nystrom_cache_to_pytorch(
    cache: NystromDistanceCache,
    *,
    device: str | None = "auto",
    devices=None,
    dtype=None,
) -> NystromDistanceCache:
    from pytorch_backend import require_torch, resolve_torch_devices, resolve_torch_dtype

    torch = require_torch()
    resolved_devices = resolve_torch_devices(
        torch,
        devices,
        fallback_device=device,
        max_devices=len(cache.folds),
    )
    sample_distance = cache.folds[0].train_landmark_distances[0]
    sample_dtype = getattr(sample_distance, "dtype", None)
    resolved_dtype = resolve_torch_dtype(torch, sample_dtype if dtype is None else dtype)

    folds = [
        _nystrom_fold_to_pytorch(
            fold,
            torch=torch,
            device=resolved_devices[fold_index % len(resolved_devices)],
            dtype=resolved_dtype,
        )
        for fold_index, fold in enumerate(cache.folds)
    ]
    return NystromDistanceCache(
        folds=folds,
        names=list(cache.names),
        kernel_types=list(cache.kernel_types),
        normalizations=list(cache.normalizations),
        pca_components=list(cache.pca_components),
        pca_whiten=list(cache.pca_whiten),
        n_landmarks=cache.n_landmarks,
        landmark_selection=cache.landmark_selection,
        batch_size=cache.batch_size,
        eigenvalue_floor=cache.eigenvalue_floor,
        backend="pytorch",
        estimated_nbytes=cache.estimated_nbytes,
        available_memory_nbytes=cache.available_memory_nbytes,
        memory_budget_nbytes=cache.memory_budget_nbytes,
        n_jobs=cache.n_jobs,
    )


def _nystrom_fold_to_pytorch(fold, *, torch, device, dtype):
    from pytorch_backend import as_torch_tensor

    y_validation = as_target_matrix(fold.y_validation, dtype=float)
    return NystromFoldDistanceCache(
        fold_id=fold.fold_id,
        train_indices=fold.train_indices,
        validation_indices=fold.validation_indices,
        landmark_local_indices=fold.landmark_local_indices,
        train_landmark_distances=[
            as_torch_tensor(distance, torch=torch, device=device, dtype=dtype)
            for distance in fold.train_landmark_distances
        ],
        validation_landmark_distances=[
            as_torch_tensor(distance, torch=torch, device=device, dtype=dtype)
            for distance in fold.validation_landmark_distances
        ],
        landmark_distances=[
            as_torch_tensor(distance, torch=torch, device=device, dtype=dtype)
            for distance in fold.landmark_distances
        ],
        y_train_transformed=as_torch_tensor(
            fold.y_train_transformed,
            torch=torch,
            device=device,
            dtype=dtype,
        ),
        y_validation=y_validation,
        target_transformer=fold.target_transformer,
        batch_size=fold.batch_size,
        eigenvalue_floor=fold.eigenvalue_floor,
    )


def score_candidates_from_nystrom_cache(
    *,
    alphas,
    gammas,
    weights,
    cache: NystromDistanceCache,
    scoring,
    backend: str = "numpy",
    n_jobs: int | None = None,
    blas_threads: int | None = None,
    pytorch_device: str | None = "auto",
    pytorch_devices=None,
    pytorch_dtype=None,
    pytorch_candidate_batch_size: int = 1,
) -> NDArray:
    backend = normalize_distance_backend(backend)
    if backend == "pytorch" or cache.backend == "pytorch":
        return score_candidates_from_nystrom_cache_pytorch(
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
    return score_candidates_from_nystrom_cache_numpy(
        alphas=alphas,
        gammas=gammas,
        weights=weights,
        cache=cache,
        scoring=scoring,
        n_jobs=n_jobs,
        blas_threads=blas_threads,
    )


def score_candidates_from_nystrom_cache_numpy(
    *,
    alphas,
    gammas,
    weights,
    cache: NystromDistanceCache,
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

    score_n_jobs = resolve_cache_n_jobs(n_jobs, n_tasks=n_candidates * n_folds)
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


def score_candidates_from_nystrom_cache_pytorch(
    *,
    alphas,
    gammas,
    weights,
    cache: NystromDistanceCache,
    scoring,
    device: str | None = "auto",
    devices=None,
    dtype=None,
    candidate_batch_size: int = 1,
) -> NDArray:
    del candidate_batch_size
    torch_cache = (
        cache
        if cache.backend == "pytorch"
        else nystrom_cache_to_pytorch(cache, device=device, devices=devices, dtype=dtype)
    )
    alphas, gammas, weights = _validate_candidate_arrays(
        alphas,
        gammas,
        weights,
        n_components=torch_cache.n_components,
    )
    n_candidates = alphas.shape[0]
    split_scores = np.empty((n_candidates, len(torch_cache.folds)), dtype=float)

    if _cache_has_multiple_torch_devices(torch_cache):
        with ThreadPoolExecutor(max_workers=_cache_torch_device_count(torch_cache)) as executor:
            futures = [
                executor.submit(
                    _score_all_candidates_for_torch_fold,
                    fold_index,
                    fold,
                    alphas,
                    gammas,
                    weights,
                    torch_cache.kernel_types,
                    scoring,
                )
                for fold_index, fold in enumerate(torch_cache.folds)
            ]
            for future in futures:
                fold_index, fold_scores = future.result()
                split_scores[:, fold_index] = fold_scores
    else:
        for fold_index, fold in enumerate(torch_cache.folds):
            split_scores[:, fold_index] = _score_all_candidates_for_torch_fold(
                fold_index,
                fold,
                alphas,
                gammas,
                weights,
                torch_cache.kernel_types,
                scoring,
            )[1]
    return split_scores


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
        y_pred = predict_fold_from_nystrom_cache_numpy(
            fold,
            alpha=alpha,
            gammas=gammas,
            weights=weights,
            kernel_types=kernel_types,
        )
        return score_predictions(fold.y_validation, y_pred, scoring)
    except np.linalg.LinAlgError:
        return np.nan


def predict_fold_from_nystrom_cache_numpy(
    fold,
    *,
    alpha: float,
    gammas: list[float],
    weights: list[float],
    kernel_types: list[str],
) -> NDArray:
    normalizer = _nystrom_normalizer_numpy(
        fold.landmark_distances,
        gammas=gammas,
        weights=weights,
        kernel_types=kernel_types,
        eigenvalue_floor=fold.eigenvalue_floor,
    )
    rank = normalizer.shape[1]
    y_train = as_target_matrix(fold.y_train_transformed)
    A = np.eye(rank, dtype=normalizer.dtype) * alpha
    B = np.zeros((rank, y_train.shape[1]), dtype=normalizer.dtype)

    for start, stop in _batch_slices(y_train.shape[0], fold.batch_size):
        C = _composite_kernel_from_distance_batch_numpy(
            fold.train_landmark_distances,
            gammas=gammas,
            weights=weights,
            kernel_types=kernel_types,
            row_slice=slice(start, stop),
        )
        Phi = C @ normalizer
        A += Phi.T @ Phi
        B += Phi.T @ y_train[start:stop]

    beta = np.linalg.solve(A, B)
    n_validation = len(fold.validation_indices)
    y_pred = np.empty((n_validation, y_train.shape[1]), dtype=normalizer.dtype)
    for start, stop in _batch_slices(n_validation, fold.batch_size):
        C = _composite_kernel_from_distance_batch_numpy(
            fold.validation_landmark_distances,
            gammas=gammas,
            weights=weights,
            kernel_types=kernel_types,
            row_slice=slice(start, stop),
        )
        y_pred[start:stop] = (C @ normalizer) @ beta

    return inverse_transform_target(fold.target_transformer, y_pred)


def _nystrom_normalizer_numpy(
    landmark_distances: list[NDArray],
    *,
    gammas: list[float],
    weights: list[float],
    kernel_types: list[str],
    eigenvalue_floor: float,
) -> NDArray:
    W = _composite_kernel_from_distance_batch_numpy(
        landmark_distances,
        gammas=gammas,
        weights=weights,
        kernel_types=kernel_types,
        row_slice=None,
    )
    W = 0.5 * (W + W.T)
    eigenvalues, eigenvectors = np.linalg.eigh(W)
    positive = eigenvalues[eigenvalues > 0]
    if positive.size == 0:
        raise np.linalg.LinAlgError("Nyström landmark kernel has no positive modes.")
    threshold = max(float(np.max(positive)) * eigenvalue_floor, 0.0)
    keep = eigenvalues > threshold
    if not np.any(keep):
        raise np.linalg.LinAlgError("Nyström landmark kernel has no retained modes.")
    eigenvalues = eigenvalues[keep]
    eigenvectors = eigenvectors[:, keep]
    return eigenvectors / np.sqrt(eigenvalues)[None, :]


def _composite_kernel_from_distance_batch_numpy(
    distances: list[NDArray],
    *,
    gammas: list[float],
    weights: list[float],
    kernel_types: list[str],
    row_slice: slice | None,
) -> NDArray:
    K_total = None
    for distance, gamma, weight, kernel_type in zip(
        distances,
        gammas,
        weights,
        kernel_types,
    ):
        distance_spec_for_kernel(kernel_type)
        values = np.asarray(distance if row_slice is None else distance[row_slice])
        component = np.exp(-float(gamma) * values) * float(weight)
        K_total = component if K_total is None else K_total + component
    if K_total is None:
        raise ValueError("At least one distance matrix is required.")
    return K_total


def _score_all_candidates_for_torch_fold(
    fold_index: int,
    fold,
    alphas: NDArray,
    gammas: NDArray,
    weights: NDArray,
    kernel_types: list[str],
    scoring,
) -> tuple[int, NDArray]:
    scores = np.empty(alphas.shape[0], dtype=float)
    for candidate_index in range(alphas.shape[0]):
        scores[candidate_index] = _safe_score_fold_pytorch(
            fold,
            alpha=float(alphas[candidate_index]),
            gammas=gammas[candidate_index].tolist(),
            weights=weights[candidate_index].tolist(),
            kernel_types=kernel_types,
            scoring=scoring,
        )
    return fold_index, scores


def _safe_score_fold_pytorch(
    fold,
    *,
    alpha: float,
    gammas: list[float],
    weights: list[float],
    kernel_types: list[str],
    scoring,
) -> float:
    try:
        y_pred = predict_fold_from_nystrom_cache_pytorch(
            fold,
            alpha=alpha,
            gammas=gammas,
            weights=weights,
            kernel_types=kernel_types,
        )
        return score_predictions(fold.y_validation, y_pred, scoring)
    except RuntimeError as exc:
        if _is_torch_linalg_error(exc):
            return np.nan
        raise


def predict_fold_from_nystrom_cache_pytorch(
    fold,
    *,
    alpha: float,
    gammas: list[float],
    weights: list[float],
    kernel_types: list[str],
) -> NDArray:
    from pytorch_backend import require_torch

    torch = require_torch()
    device = fold.y_train_transformed.device
    if getattr(device, "type", None) == "cuda":
        torch.cuda.set_device(device)

    with torch.no_grad():
        normalizer = _nystrom_normalizer_pytorch(
            fold.landmark_distances,
            gammas=gammas,
            weights=weights,
            kernel_types=kernel_types,
            eigenvalue_floor=fold.eigenvalue_floor,
        )
        rank = int(normalizer.shape[1])
        y_train = fold.y_train_transformed
        if y_train.ndim == 1:
            y_train = y_train.reshape(-1, 1)
        A = torch.eye(rank, dtype=normalizer.dtype, device=device) * alpha
        B = torch.zeros((rank, y_train.shape[1]), dtype=normalizer.dtype, device=device)

        for start, stop in _batch_slices(y_train.shape[0], fold.batch_size):
            C = _composite_kernel_from_distance_batch_pytorch(
                fold.train_landmark_distances,
                gammas=gammas,
                weights=weights,
                kernel_types=kernel_types,
                row_slice=slice(start, stop),
            )
            Phi = C @ normalizer
            A += Phi.T @ Phi
            B += Phi.T @ y_train[start:stop]

        beta = torch.linalg.solve(A, B)
        n_validation = len(fold.validation_indices)
        y_pred = np.empty((n_validation, y_train.shape[1]), dtype=float)
        for start, stop in _batch_slices(n_validation, fold.batch_size):
            C = _composite_kernel_from_distance_batch_pytorch(
                fold.validation_landmark_distances,
                gammas=gammas,
                weights=weights,
                kernel_types=kernel_types,
                row_slice=slice(start, stop),
            )
            pred = (C @ normalizer) @ beta
            y_pred[start:stop] = pred.detach().cpu().numpy()

    return inverse_transform_target(fold.target_transformer, y_pred)


def _nystrom_normalizer_pytorch(
    landmark_distances,
    *,
    gammas: list[float],
    weights: list[float],
    kernel_types: list[str],
    eigenvalue_floor: float,
):
    from pytorch_backend import require_torch

    torch = require_torch()
    W = _composite_kernel_from_distance_batch_pytorch(
        landmark_distances,
        gammas=gammas,
        weights=weights,
        kernel_types=kernel_types,
        row_slice=None,
    )
    W = 0.5 * (W + W.T)
    eigenvalues, eigenvectors = torch.linalg.eigh(W)
    positive = eigenvalues[eigenvalues > 0]
    if positive.numel() == 0:
        raise RuntimeError("Nyström landmark kernel has no positive modes.")
    threshold = torch.max(positive) * float(eigenvalue_floor)
    keep = eigenvalues > threshold
    if not bool(torch.any(keep).item()):
        raise RuntimeError("Nyström landmark kernel has no retained modes.")
    eigenvalues = eigenvalues[keep]
    eigenvectors = eigenvectors[:, keep]
    return eigenvectors / torch.sqrt(eigenvalues).reshape(1, -1)


def _composite_kernel_from_distance_batch_pytorch(
    distances,
    *,
    gammas: list[float],
    weights: list[float],
    kernel_types: list[str],
    row_slice: slice | None,
):
    from pytorch_backend import require_torch

    torch = require_torch()
    K_total = None
    for distance, gamma, weight, kernel_type in zip(
        distances,
        gammas,
        weights,
        kernel_types,
    ):
        distance_spec_for_kernel(kernel_type)
        values = distance if row_slice is None else distance[row_slice]
        gamma_t = torch.as_tensor(gamma, dtype=values.dtype, device=values.device)
        weight_t = torch.as_tensor(weight, dtype=values.dtype, device=values.device)
        component = torch.exp(-gamma_t * values) * weight_t
        K_total = component if K_total is None else K_total + component
    if K_total is None:
        raise ValueError("At least one distance matrix is required.")
    return K_total


def _cache_has_multiple_torch_devices(cache: NystromDistanceCache) -> bool:
    return _cache_torch_device_count(cache) > 1


def _cache_torch_device_count(cache: NystromDistanceCache) -> int:
    return len(
        {
            str(fold.y_train_transformed.device)
            for fold in cache.folds
        }
    )


def _threadpool_limits_for_blas(blas_threads: int | None):
    if blas_threads is None:
        return nullcontext()
    return threadpool_limits(limits=blas_threads)


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


def _batch_slices(n_samples: int, batch_size: int):
    for start in range(0, n_samples, batch_size):
        yield start, min(start + batch_size, n_samples)


def _is_torch_linalg_error(exc: RuntimeError) -> bool:
    message = str(exc).lower()
    return "linalg" in message or "singular" in message or "cholesky" in message
