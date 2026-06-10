import logging
from time import perf_counter

import numpy as np
from sklearn.compose import TransformedTargetRegressor

import postprocess
import preprocess
from search_random import CompositeKRREstimator, staged_random_search_cv


logger = logging.getLogger("kernel-contributions")


def evaluate_kernel_contributions(
    full_result,
    X,
    y,
    *,
    component_names: list[str],
    kernel_types: list[str],
    normalizations: list[str],
    pca_components=None,
    pca_whiten=False,
    target_normalization: str,
    compute_dtype,
    scoring,
    cv,
    search_kwargs: dict,
    output_dir: str | None = None,
    ddof: int = 1,
) -> tuple[list[dict], list[tuple[float, float, str]]]:
    component_names = list(component_names)
    kernel_types = list(kernel_types)
    normalizations = list(normalizations)
    pca_components = _resolve_sequence(
        "pca_components",
        pca_components,
        len(component_names),
        None,
    )
    pca_whiten = _resolve_sequence(
        "pca_whiten",
        pca_whiten,
        len(component_names),
        False,
    )
    n_components = len(component_names)
    timings = []

    if n_components == 0:
        raise ValueError("At least one kernel component is required.")
    if not (
        len(kernel_types)
        == len(normalizations)
        == len(pca_components)
        == len(pca_whiten)
        == n_components
    ):
        raise ValueError(
            "component_names, kernel_types, normalizations, pca_components, "
            "and pca_whiten must have matching lengths."
        )

    X = np.asarray(X, dtype=object)
    if X.ndim != 2 or X.shape[1] != n_components:
        raise ValueError(
            f"X must have shape (n_samples, {n_components}), got {X.shape}."
        )

    single_kernel_results = {}
    leave_one_out_results = {}
    all_component_indices = list(range(n_components))

    if n_components == 1:
        logger.warning(
            "Skipping kernel contribution evaluation: only one kernel "
            "component is present."
        )
        return [], [
            (0.0, 0.0, "Post-Processing: Single-Kernel Contribution Evaluation (Skipped)"),
            (0.0, 0.0, "Post-Processing: Leave-One-Out Kernel Ablations (Skipped)"),
        ]

    time_single_start = perf_counter()
    for component_index, kernel_name in enumerate(component_names):
        single_kernel_results[kernel_name] = _fit_component_subset(
            [component_index],
            label=f"single-kernel [{kernel_name}]",
            X=X,
            y=y,
            component_names=component_names,
            kernel_types=kernel_types,
            normalizations=normalizations,
            pca_components=pca_components,
            pca_whiten=pca_whiten,
            target_normalization=target_normalization,
            compute_dtype=compute_dtype,
            scoring=scoring,
            cv=cv,
            search_kwargs=search_kwargs,
        )
    time_single_end = perf_counter()
    timings.append(
        (
            time_single_start,
            time_single_end,
            "Post-Processing: Single-Kernel Contribution Evaluation",
        )
    )

    time_ablation_start = perf_counter()
    for component_index, kernel_name in enumerate(component_names):
        keep_indices = [
            index
            for index in all_component_indices
            if index != component_index
        ]
        leave_one_out_results[kernel_name] = _fit_component_subset(
            keep_indices,
            label=f"leave-one-out drop [{kernel_name}]",
            X=X,
            y=y,
            component_names=component_names,
            kernel_types=kernel_types,
            normalizations=normalizations,
            pca_components=pca_components,
            pca_whiten=pca_whiten,
            target_normalization=target_normalization,
            compute_dtype=compute_dtype,
            scoring=scoring,
            cv=cv,
            search_kwargs=search_kwargs,
        )
    time_ablation_end = perf_counter()
    timings.append(
        (
            time_ablation_start,
            time_ablation_end,
            "Post-Processing: Leave-One-Out Kernel Ablations",
        )
    )

    rows = postprocess.summarize_kernel_contribution_results(
        full_result,
        single_kernel_results=single_kernel_results,
        leave_one_out_results=leave_one_out_results,
        scoring=scoring,
        output_dir=output_dir,
        ddof=ddof,
    )
    return rows, timings


def _fit_component_subset(
    component_indices,
    *,
    label: str,
    X,
    y,
    component_names: list[str],
    kernel_types: list[str],
    normalizations: list[str],
    pca_components: list,
    pca_whiten: list[bool],
    target_normalization: str,
    compute_dtype,
    scoring,
    cv,
    search_kwargs: dict,
):
    component_indices = list(component_indices)
    if not component_indices:
        raise ValueError("At least one component index is required.")

    subset_names = [component_names[index] for index in component_indices]
    subset_kernel_types = [kernel_types[index] for index in component_indices]
    subset_norms = [normalizations[index] for index in component_indices]
    subset_pca_components = [pca_components[index] for index in component_indices]
    subset_pca_whiten = [pca_whiten[index] for index in component_indices]
    subset_estimator = TransformedTargetRegressor(
        regressor=CompositeKRREstimator(
            names=subset_names,
            kernel_types=subset_kernel_types,
            normalizations=subset_norms,
            pca_components=subset_pca_components,
            pca_whiten=subset_pca_whiten,
            normalize_kernel_weights=True,
            compute_dtype=compute_dtype,
        ),
        transformer=preprocess.make_target_preprocessor(target_normalization),
    )

    logger.warning(
        f"Evaluating {label} kernel contribution model with "
        f"{len(component_indices)} component(s)."
    )
    return staged_random_search_cv(
        subset_estimator,
        X[:, component_indices],
        y,
        n_components=len(component_indices),
        scoring=scoring,
        cv=cv,
        **dict(search_kwargs),
    )


def _resolve_sequence(name: str, values, n_items: int, default):
    if values is None:
        return [default] * n_items
    if isinstance(values, str) or np.isscalar(values):
        return [values] * n_items

    values = list(values)
    if len(values) != n_items:
        raise ValueError(
            f"{name} must have length {n_items}, got {len(values)}."
        )

    return values
