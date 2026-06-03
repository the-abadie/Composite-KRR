import numpy as np
from scipy.spatial.distance import pdist, squareform, cdist

def pairwise_self_lp_distance(
    X: np.ndarray,
    p:float,
    block_size:int = 1024,
    squared:bool = False,
    dtype:np.dtype | type = np.float64,
    use_scipy_for_p1_p2:bool = True
) -> np.ndarray:
    """
    Compute pairwise Lp distances between rows of X.
    Parameters
    ----------
    X:
        Array of shape (N, D).
    p:
        Lp norm order. Must satisfy p >= 1.
    block_size:
        Number of rows per block for the general p implementation.
        Larger is faster but uses more memory.
    dtype:
        Floating dtype used internally.
    use_scipy_for_p1_p2:
        If True, use scipy.spatial.distance.pdist for p=1 and p=2.
    Returns
    -------
    dist:
        Symmetric array of shape (N, N), where dist[i, j] = ||X[i] - X[j]||_p.
    """
    X = np.asarray(X, dtype=dtype)

    if X.ndim != 2:
        raise ValueError(f"X must be a 2D array of shape (N, D), got shape {X.shape}")
    if p < 1:
        raise ValueError("p must be >= 1")

    if squared and p != 2:
        raise ValueError("`squared = True` only valid for p=2 (rbf).")

    N, D = X.shape

    if N == 0:
        return np.empty((0, 0), dtype=dtype)
    if N == 1:
        return np.zeros((1, 1), dtype=dtype)

    # Fast scipy paths for common norms.
    if use_scipy_for_p1_p2 and p in (1, 2):
        if p == 1:
            metric = "cityblock"
        else:
            metric = "sqeuclidean" if squared else "euclidean"
        return squareform(pdist(X, metric=metric)).astype(dtype, copy=False)

    # Fast NumPy-only Euclidean distance path.
    # Uses ||x - y||^2 = ||x||^2 + ||y||^2 - 2 x@y.
    if p == 2:
        sq_norms = np.einsum("ij,ij->i", X, X)
        dist2 = sq_norms[:, None] + sq_norms[None, :] - 2.0 * (X @ X.T)
        # Small negative values can appear from floating point roundoff.
        np.maximum(dist2, 0.0, out=dist2)
        if squared:
            return dist2
        dist = np.sqrt(dist2, out=dist2)
        np.fill_diagonal(dist, 0.0)
        return dist

    # General blockwise implementation for arbitrary p.
    # Avoids materializing X[:, None, :] - X[None, :, :], which is (N, N, D).
    dist = np.empty((N, N), dtype=dtype)
    for i0 in range(0, N, block_size):
        i1 = min(i0 + block_size, N)
        Xi = X[i0:i1]

        for j0 in range(i0, N, block_size):
            j1 = min(j0 + block_size, N)
            Xj = X[j0:j1]

            # Shape: (i_block, j_block, D)
            diff = np.abs(Xi[:, None, :] - Xj[None, :, :])

            # Lp distance: sum(abs(diff)^p)^(1/p)
            block = np.sum(diff**p, axis=2) ** (1.0 / p)
            dist[i0:i1, j0:j1] = block

            if j0 != i0:
                dist[j0:j1, i0:i1] = block.T

    np.fill_diagonal(dist, 0.0)
    return dist


def pairwise_cross_lp_distance(
    X: np.ndarray,
    Y: np.ndarray,
    p: float,
    block_size: int = 1024,
    squared: bool = False,
    dtype: np.dtype | type = np.float64,
    use_scipy_for_p1_p2: bool = True,
) -> np.ndarray:
    """
    Compute pairwise Lp distances between rows of X and rows of Y.

    Returns
    -------
    dist:
        Array of shape (X.shape[0], Y.shape[0]), where
        dist[i, j] = ||X[i] - Y[j]||_p.
    """
    X = np.asarray(X, dtype=dtype)
    Y = np.asarray(Y, dtype=dtype)

    if X.ndim != 2:
        raise ValueError(f"X must be a 2D array of shape (N, D), got {X.shape}")
    if Y.ndim != 2:
        raise ValueError(f"Y must be a 2D array of shape (M, D), got {Y.shape}")
    if X.shape[1] != Y.shape[1]:
        raise ValueError(
            f"X and Y must have the same feature dimension, got "
            f"{X.shape[1]} and {Y.shape[1]}."
        )
    if p < 1:
        raise ValueError("p must be >= 1")
    if squared and p != 2:
        raise ValueError("`squared=True` only valid for p=2.")

    N, D = X.shape
    M = Y.shape[0]

    if N == 0 or M == 0:
        return np.empty((N, M), dtype=dtype)

    if use_scipy_for_p1_p2 and p in (1, 2):
        if p == 1:
            return cdist(X, Y, metric="cityblock").astype(dtype, copy=False)
        metric = "sqeuclidean" if squared else "euclidean"
        return cdist(X, Y, metric=metric).astype(dtype, copy=False)

    if p == 2:
        x_sq_norms = np.einsum("ij,ij->i", X, X)
        y_sq_norms = np.einsum("ij,ij->i", Y, Y)
        dist2 = x_sq_norms[:, None] + y_sq_norms[None, :] - 2.0 * (X @ Y.T)
        np.maximum(dist2, 0.0, out=dist2)
        if squared:
            return dist2
        return np.sqrt(dist2, out=dist2)

    dist = np.empty((N, M), dtype=dtype)
    for i0 in range(0, N, block_size):
        i1 = min(i0 + block_size, N)
        Xi = X[i0:i1]

        for j0 in range(0, M, block_size):
            j1 = min(j0 + block_size, M)
            Yj = Y[j0:j1]

            diff = np.abs(Xi[:, None, :] - Yj[None, :, :])
            dist[i0:i1, j0:j1] = np.sum(diff**p, axis=2) ** (1.0 / p)

    return dist
