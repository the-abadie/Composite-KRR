import logging

import numpy as np
from numpy.typing import ArrayLike, NDArray

from config import VERBOSITY
from utilities import configure_logging

configure_logging(VERBOSITY)
logger = logging.getLogger("preparation")


def randomized_selection_with_remainder(arr, N, rng) -> tuple[NDArray, NDArray]:
    if rng is None:
        rng = np.random.default_rng()

    selected_idx = rng.choice(len(arr), size=N, replace=False)
    mask = np.ones(len(arr), dtype=bool)
    mask[selected_idx] = False

    selected = arr[selected_idx]
    not_selected = arr[mask]

    return selected, not_selected


def stratified_selection(target, n_strata: int, n_total: int, rng) -> NDArray:
    target = np.asarray(target)
    n = target.shape[0]

    if rng is None:
        rng = np.random.default_rng()
    if not 1 <= n_strata <= n:
        raise ValueError(f"n_strata must be between 1 and {n}, got {n_strata}")
    if not 1 <= n_total <= n:
        raise ValueError(f"n_total must be between 1 and {n}, got {n_total}")

    order = np.argsort(target)
    strata = np.array_split(order, n_strata)

    per = n_total // n_strata
    rem = n_total % n_strata

    idx_parts = []
    for i, s in enumerate(strata):
        s = rng.permutation(s)
        k = per + (1 if i < rem else 0)
        idx_parts.append(s[:k])

    idx = np.concatenate(idx_parts)
    if idx.size != n_total:
        idx = idx[:n_total]
    return idx


def stratified_selection_with_remainder(
    target, n_strata: int, n_total: int, rng
) -> tuple[NDArray, NDArray]:
    idx = stratified_selection(target, n_strata, n_total, rng=rng)
    mask = np.ones(len(target), dtype=bool)
    mask[idx] = False
    rest = np.where(mask)[0]
    return idx, rest


def validate_descriptor_target_lengths(descriptors: list, target: NDArray) -> None:
    n_targets: int = len(target)
    for descriptor in descriptors:
        if descriptor.n_samples != n_targets:
            raise ValueError(
                f"Descriptor {descriptor.name} length ({descriptor.n_samples}) "
                f"does not match the length of the target ({n_targets})."
            )
    logger.info("Validated all descriptor lengths match target length.")
