import logging
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

import config
from target_utils import as_target_matrix
from utilities import configure_logging

configure_logging(config.VERBOSITY)
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
    target = stratification_values(target)
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


def stratification_values(target) -> NDArray:
    target_matrix = as_target_matrix(target, dtype=float)
    if target_matrix.shape[1] == 1:
        return target_matrix[:, 0]

    centered = target_matrix - np.mean(target_matrix, axis=0, keepdims=True)
    scale = np.std(centered, axis=0, keepdims=True)
    scale[scale == 0.0] = 1.0
    standardized = centered / scale
    _, _, vh = np.linalg.svd(standardized, full_matrices=False)
    return standardized @ vh[0]


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


def load_predefined_validation_folds(path: str | Path) -> list[NDArray]:
    path = Path(path)
    if path.suffix != ".npz":
        raise ValueError(f"Predefined validation fold path {path} must be a .npz file.")

    with np.load(path) as npz:
        fold_keys = _sorted_predefined_fold_keys(npz.files)
        folds = [np.asarray(npz[key]).reshape(-1) for key in fold_keys]

    return folds


def validate_predefined_splits(
    predef_idx_train: NDArray,
    predef_idx_val: list[NDArray],
    predef_idx_test: NDArray,
    target_length: int,
) -> None:
    predef_idx_train = _validate_index_array(
        "predefined training indices", predef_idx_train, allow_empty=False
    )
    predef_idx_test = _validate_index_array(
        "predefined testing indices", predef_idx_test, allow_empty=False
    )

    if not predef_idx_val:
        raise ValueError("At least one predefined validation fold is required.")

    validated_val_folds = [
        _validate_index_array(
            f"predefined validation fold {fold_id}",
            fold_idx,
            allow_empty=False,
        )
        for fold_id, fold_idx in enumerate(predef_idx_val)
    ]

    N_PREDEF_TRAIN = len(predef_idx_train)
    N_PREDEF_VAL = sum(len(fold_idx) for fold_idx in validated_val_folds)
    N_PREDEF_TEST = len(predef_idx_test)

    if N_PREDEF_TRAIN + N_PREDEF_VAL + N_PREDEF_TEST > target_length:
        raise ValueError("Number of pre-defined indices is greater than the target length. Please check your splits.")

    all_indices = np.concatenate(
        [predef_idx_train, *validated_val_folds, predef_idx_test]
    )

    if np.any(all_indices < 0) or np.any(all_indices >= target_length):
        raise ValueError(f"Predefined split indices must be between 0 and {target_length - 1}.")

    unique_indices = np.unique(all_indices)
    if len(unique_indices) != len(all_indices):
        raise ValueError(
            "Predefined train, validation, and test indices must be unique and non-overlapping.")

    if config.N_SAMPLES is not None:
        logger.warning(f"WARNING: You are using pre-defined splits but `N_SAMPLES` is not `None`. Config value will be ignored.")
    if config.TRAIN_VAL_SPLIT is not None:
        logger.warning(f"WARNING: You are using pre-defined splits but `TRAIN_VAL_SPLIT` is not `None`. Config value will be ignored." )
    if config.N_KFOLD is not None:
        logger.warning(f"WARNING: You are using pre-defined splits but `N_KFOLD` is not `None`. Config value will be ignored." )
    if config.STRATIFY:
        logger.warning(f"WARNING: You are using pre-defined splits but `STRATIFY` is not `False`. Config value will be ignored.")

    logger.info(
        "Validated predefined splits: "
        f"{N_PREDEF_TRAIN} train, {N_PREDEF_VAL} validation, "
        f"{N_PREDEF_TEST} test samples across {len(validated_val_folds)} folds."
    )


def _sorted_predefined_fold_keys(keys) -> list[str]:
    keys = list(keys)
    if not keys:
        raise ValueError("Predefined validation .npz file contains no folds.")

    for key in keys:
        if not key.startswith("fold") or not key[4:].isdigit():
            raise ValueError(
                f"Predefined validation .npz keys must be named fold0, fold1, ...; got {key!r}.")

    fold_keys = sorted(keys, key=lambda key: int(key[4:]))
    expected_keys = [f"fold{i}" for i in range(len(fold_keys))]
    if fold_keys != expected_keys:
        raise ValueError(
            f"Predefined validation .npz fold keys must be contiguous from fold0; got {fold_keys}.")

    return fold_keys


def _validate_index_array(
    name: str,
    indices,
    *,
    allow_empty: bool,
) -> NDArray:
    indices = np.asarray(indices)
    if indices.ndim != 1:
        raise ValueError(f"{name} must be a 1D array, got shape {indices.shape}.")
    if not allow_empty and len(indices) == 0:
        raise ValueError(f"{name} must not be empty.")
    if not np.issubdtype(indices.dtype, np.integer):
        raise ValueError(f"{name} must contain integer indices, got {indices.dtype}.")

    return indices
