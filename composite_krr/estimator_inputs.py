"""Block-matrix, hyperparameter and backend conventions shared by fitting and CV."""
import numpy as np
from numpy.typing import NDArray

from kernel_mixing import kernel_term_count, validate_kernel_parameters


def normalize_compute_backend(backend: str, *, name: str) -> str:
    backend = str(backend).lower()
    if backend in {"numpy", "np", "cpu"}:
        return "numpy"
    if backend in {"pytorch", "torch", "gpu", "cuda", "rocm"}:
        return "pytorch"
    raise ValueError(
        f'{name} must be "numpy" or "pytorch", '
        f"got {backend!r}."
    )


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
        block = np.stack(X[:, block_id]).astype(dtype, copy=False)
        blocks.append(block.reshape(X.shape[0], -1))

    return blocks


def resolve_kernel_hyperparameters(regressor, *, n_components: int):
    alpha = float(regressor.alpha)
    gammas = resolve_sequence("gammas", regressor.gammas, n_components, 1.0)
    weights = resolve_sequence("kernel_weights", regressor.kernel_weights, kernel_term_count(regressor, n_components), 1.0)

    validate_kernel_parameters(gammas, weights, n_components, getattr(regressor, "kernel_products", None))
    gammas = list(np.asarray(gammas, dtype=float))
    weights = np.asarray(weights, dtype=float)
    if getattr(regressor, "normalize_kernel_weights", False):
        weight_sum = weights.sum()
        if weight_sum <= 0 or not np.isfinite(weight_sum):
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
