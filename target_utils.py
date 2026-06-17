import numpy as np
from numpy.typing import ArrayLike, NDArray


def as_target_array(y: ArrayLike, dtype=None) -> NDArray:
    """Return y with the first axis as samples and remaining axes as targets."""
    y = np.asarray(y, dtype=dtype)
    if y.ndim == 0:
        raise ValueError("Target data must have at least one sample axis.")
    if y.shape[0] == 0:
        raise ValueError("Target data must contain at least one sample.")
    if y.ndim <= 2:
        return y
    return y.reshape(y.shape[0], -1)


def as_target_matrix(y: ArrayLike, dtype=None) -> NDArray:
    """Return y as a 2D (n_samples, n_targets) matrix."""
    y = as_target_array(y, dtype=dtype)
    if y.ndim == 1:
        return y.reshape(-1, 1)
    if y.shape[1] == 0:
        raise ValueError("Target data must contain at least one target column.")
    return y


def maybe_squeeze_single_target(y: ArrayLike, *, squeeze: bool) -> NDArray:
    y = np.asarray(y)
    if squeeze and y.ndim == 2 and y.shape[1] == 1:
        return y.reshape(-1)
    return y


def align_targets_for_scoring(y_true: ArrayLike, y_pred: ArrayLike) -> tuple[NDArray, NDArray]:
    """Align true/predicted targets for sklearn scorers.

    Single-output targets are returned as 1D arrays for backward-compatible
    sklearn metric behavior. Multi-output targets are returned as 2D matrices.
    """
    y_true_matrix = as_target_matrix(y_true, dtype=float)
    y_pred_matrix = as_target_matrix(y_pred, dtype=float)

    if y_true_matrix.shape != y_pred_matrix.shape:
        raise ValueError(
            "y_true and y_pred must have matching target shapes, got "
            f"{y_true_matrix.shape} and {y_pred_matrix.shape}."
        )

    if y_true_matrix.shape[1] == 1:
        return y_true_matrix.reshape(-1), y_pred_matrix.reshape(-1)
    return y_true_matrix, y_pred_matrix


def target_column_count(y: ArrayLike) -> int:
    return int(as_target_matrix(y).shape[1])

