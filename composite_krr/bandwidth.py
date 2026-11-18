"""Estimate per-kernel gamma search bounds from fold distances or descriptor data."""
import logging

import numpy as np

from composite_krr.estimator_inputs import resolve_sequence, unpack_sample_matrix
from kernel_cache import extract_regressor
from kernels import distance_spec_for_kernel, pairwise_self_lp_distance
from preprocess import make_data_preprocessor

logger = logging.getLogger("search")


def estimate_gamma_bounds(
    estimator,
    X,
    *,
    n_components: int,
    legal_gamma_bounds: tuple[float, float] | list[tuple[float, float]],
    decades: float,
    distance_cache=None,
    dtype: np.dtype | type | str = np.float64,
    block_size: int = 1024,
    max_samples: int | None = None,
    random_state=None,
    journal=None,
) -> list[tuple[float, float]]:
    component_legal_bounds = _as_component_bounds(
        legal_gamma_bounds,
        n_components,
    )
    if decades <= 0:
        raise ValueError(f"decades must be positive, got {decades}.")
    if max_samples is not None and (type(max_samples) is not int or max_samples <= 0):
        raise ValueError("max_samples must be None or a positive int.")

    centers = (
        _gamma_centers_from_distance_cache(distance_cache, component_legal_bounds)
        if distance_cache is not None
        else _gamma_centers_from_full_data(
            estimator,
            X,
            n_components=n_components,
            legal_gamma_bounds=component_legal_bounds,
            dtype=dtype,
            block_size=block_size,
            max_samples=max_samples,
            random_state=random_state,
        )
    )

    bounds = [
        _narrow_log_bounds_around(center, legal_bounds, decades)
        for center, legal_bounds in zip(centers, component_legal_bounds)
    ]
    if journal is not None:
        journal.append({"event": "gamma_prior", "centers": centers, "bounds": bounds,
                        "decades": decades, "legal_bounds": component_legal_bounds})
    for component_index, (center, bound) in enumerate(zip(centers, bounds), start=1):
        logger.info(
            "Component %s gamma prior centered at %.6g with bounds "
            "[%.6g, %.6g].",
            component_index,
            center,
            bound[0],
            bound[1],
        )
    return bounds


def _gamma_centers_from_distance_cache(
    distance_cache,
    legal_gamma_bounds: list[tuple[float, float]],
) -> list[float]:
    centers = []
    for component_index, legal_bounds in enumerate(legal_gamma_bounds):
        medians = []
        for fold in distance_cache.folds:
            distance = fold.train_distances[component_index]
            # A Torch scoring fold knows the compute device even when its
            # distances live in host RAM. Only the sampled values need moving.
            device = getattr(getattr(fold, "y_validation_torch", None), "device", None)
            median = _positive_finite_median(distance, device=device)
            if np.isfinite(median) and median > 0:
                medians.append(median)
        centers.append(_gamma_center_from_distance_medians(medians, legal_bounds))

    return centers


def _gamma_centers_from_full_data(
    estimator,
    X,
    *,
    n_components: int,
    legal_gamma_bounds: list[tuple[float, float]],
    dtype: np.dtype | type | str,
    block_size: int,
    max_samples: int | None,
    random_state,
) -> list[float]:
    regressor = extract_regressor(estimator)
    kernel_types = resolve_sequence(
        "kernel_types",
        regressor.kernel_types,
        n_components,
        "rbf",
    )
    normalizations = resolve_sequence(
        "normalizations",
        regressor.normalizations,
        n_components,
        "none",
    )
    pca_components = resolve_sequence(
        "pca_components",
        getattr(regressor, "pca_components", None),
        n_components,
        None,
    )
    pca_whiten = resolve_sequence(
        "pca_whiten",
        getattr(regressor, "pca_whiten", False),
        n_components,
        False,
    )
    X_blocks = unpack_sample_matrix(X, dtype=dtype)
    original_n_samples = X_blocks[0].shape[0]
    if max_samples is not None and original_n_samples > max_samples:
        rng = np.random.default_rng(random_state)
        sample_indices = np.sort(
            rng.choice(original_n_samples, size=max_samples, replace=False)
        )
        X_blocks = [X_block[sample_indices] for X_block in X_blocks]
        logger.info(
            "Estimating gamma priors from %s sampled rows instead of all %s rows.",
            max_samples,
            original_n_samples,
        )
    centers = []

    for (
        X_block,
        normalization,
        pca_component,
        pca_whiten_block,
        kernel_type,
        legal_bounds,
    ) in zip(
        X_blocks,
        normalizations,
        pca_components,
        pca_whiten,
        kernel_types,
        legal_gamma_bounds,
    ):
        spec = distance_spec_for_kernel(kernel_type)
        preprocessor = make_data_preprocessor(
            normalization,
            pca_components=pca_component,
            pca_whiten=pca_whiten_block,
        )
        X_block_t = preprocessor.fit_transform(X_block)
        distances = pairwise_self_lp_distance(
            X_block_t,
            p=spec.p,
            squared=spec.squared,
            block_size=block_size,
            dtype=dtype,
        )
        median = _positive_finite_median(distances)
        centers.append(_gamma_center_from_distance_medians([median], legal_bounds))

    return centers


def _gamma_center_from_distance_medians(
    medians: list[float],
    legal_bounds: tuple[float, float],
) -> float:
    medians = np.asarray(medians, dtype=float)
    medians = medians[np.isfinite(medians) & (medians > 0)]
    if medians.size == 0:
        return _geometric_midpoint(legal_bounds)

    center = 1.0 / float(np.median(medians))
    return float(np.clip(center, legal_bounds[0], legal_bounds[1]))


def _positive_finite_median(values, *, max_values: int = 2_000_000, device=None) -> float:
    if _is_torch_tensor(values):
        return _positive_finite_median_torch(values, max_values=max_values, device=device)

    values = np.asarray(values).ravel()
    if values.size == 0:
        return np.nan

    step = max(1, int(np.ceil(values.size / max_values)))
    sample = values[::step]
    sample = sample[np.isfinite(sample) & (sample > 0)]
    if sample.size == 0:
        return np.nan

    return float(np.median(sample))


def _positive_finite_median_torch(values, *, max_values: int, device=None) -> float:
    from importlib import import_module

    torch = import_module("torch")
    values = values.reshape(-1)
    if values.numel() == 0:
        return np.nan

    step = max(1, int(np.ceil(values.numel() / max_values)))
    sample = values[::step]
    if device is not None:
        sample = sample.to(device)
    sample = sample[torch.isfinite(sample) & (sample > 0)]
    if sample.numel() == 0:
        return np.nan

    return float(torch.median(sample).detach().cpu().item())


def _is_torch_tensor(values) -> bool:
    return (
        values.__class__.__module__.startswith("torch")
        and hasattr(values, "detach")
        and hasattr(values, "numel")
    )


def _narrow_log_bounds_around(
    value: float,
    bounds: tuple[float, float],
    decades: float,
) -> tuple[float, float]:
    low, high = _validate_log_bounds("bounds", bounds)
    if decades <= 0:
        raise ValueError(f"decades must be positive, got {decades}.")
    if not np.isfinite(value) or value <= 0:
        value = _geometric_midpoint((low, high))

    factor = 10.0 ** decades
    new_low = max(low, float(value) / factor)
    new_high = min(high, float(value) * factor)
    if new_high <= new_low:
        return low, high

    return new_low, new_high


def _as_component_bounds(
    bounds: tuple[float, float] | list[tuple[float, float]],
    n_components: int,
) -> list[tuple[float, float]]:
    if n_components <= 0:
        raise ValueError(f"n_components must be positive, got {n_components}.")

    if _looks_like_single_bounds(bounds):
        low, high = bounds
        return [_validate_log_bounds("bounds", (low, high))] * n_components

    component_bounds = list(bounds)
    if len(component_bounds) != n_components:
        raise ValueError(
            f"Expected {n_components} component bounds, "
            f"got {len(component_bounds)}."
        )

    return [
        _validate_log_bounds("bounds", tuple(component_bounds[index]))
        for index in range(n_components)
    ]


def _looks_like_single_bounds(bounds) -> bool:
    if len(bounds) != 2:
        return False
    return np.isscalar(bounds[0]) and np.isscalar(bounds[1])


def _validate_log_bounds(
    name: str,
    bounds: tuple[float, float],
) -> tuple[float, float]:
    low, high = bounds
    low = float(low)
    high = float(high)
    if low <= 0 or high <= low:
        raise ValueError(f"Expected 0 < low < high for {name}, got {bounds}.")

    return low, high


def _geometric_midpoint(bounds: tuple[float, float]) -> float:
    low, high = _validate_log_bounds("bounds", bounds)
    return float(np.sqrt(low * high))
