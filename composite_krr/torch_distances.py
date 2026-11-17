"""PyTorch pairwise distances shared by exact and Nyström workloads."""
from composite_krr.torch_runtime import as_torch_tensor, require_torch, resolve_torch_device, resolve_torch_dtype


def pairwise_self_lp_distance_pytorch(
    X,
    p: float,
    block_size: int = 1024,
    squared: bool = False,
    dtype=None,
    device: str | None = "auto",
):
    torch = require_torch()
    resolved_device = resolve_torch_device(torch, device)
    resolved_dtype = resolve_torch_dtype(torch, dtype)
    X = as_torch_tensor(X, torch=torch, device=resolved_device, dtype=resolved_dtype)

    if X.ndim != 2:
        raise ValueError(f"X must be a 2D tensor of shape (N, D), got {tuple(X.shape)}.")
    if p < 1:
        raise ValueError("p must be >= 1")
    if squared and p != 2:
        raise ValueError("`squared=True` only valid for p=2.")

    n_samples = X.shape[0]
    if n_samples == 0:
        return torch.empty((0, 0), dtype=resolved_dtype, device=resolved_device)
    if n_samples == 1:
        return torch.zeros((1, 1), dtype=resolved_dtype, device=resolved_device)

    if p == 2:
        sq_norms = torch.sum(X * X, dim=1)
        dist2 = sq_norms[:, None] + sq_norms[None, :] - 2.0 * (X @ X.T)
        dist2 = torch.clamp(dist2, min=0.0)
        if squared:
            return dist2
        dist = torch.sqrt(dist2)
        dist.fill_diagonal_(0.0)
        return dist

    if p == 1:
        # Each slab covers the lower triangle and its diagonal tile. Mirroring
        # the rest avoids computing both orientations of wide L1 distances.
        slab_size = min(256, block_size)
        if X.is_cuda and 0 < slab_size < n_samples and not X.requires_grad:
            dist = torch.empty((n_samples, n_samples), dtype=X.dtype, device=X.device)
            for start in range(0, n_samples, slab_size):
                stop = min(start + slab_size, n_samples)
                slab = torch.cdist(X[start:stop], X[:stop], p=1)
                dist[start:stop, :stop] = slab
                dist[:start, start:stop] = slab[:, :start].T
        else:
            dist = torch.cdist(X, X, p=1)
        dist.fill_diagonal_(0.0)
        return dist

    dist = torch.empty(
        (n_samples, n_samples),
        dtype=resolved_dtype,
        device=resolved_device,
    )
    for i0 in range(0, n_samples, block_size):
        i1 = min(i0 + block_size, n_samples)
        xi = X[i0:i1]
        for j0 in range(i0, n_samples, block_size):
            j1 = min(j0 + block_size, n_samples)
            xj = X[j0:j1]
            block = torch.sum(torch.abs(xi[:, None, :] - xj[None, :, :]) ** p, dim=2)
            block = block ** (1.0 / p)
            dist[i0:i1, j0:j1] = block
            if j0 != i0:
                dist[j0:j1, i0:i1] = block.T
    dist.fill_diagonal_(0.0)
    return dist


def pairwise_cross_lp_distance_pytorch(
    X,
    Y,
    p: float,
    block_size: int = 1024,
    squared: bool = False,
    dtype=None,
    device: str | None = "auto",
):
    torch = require_torch()
    resolved_device = resolve_torch_device(torch, device)
    resolved_dtype = resolve_torch_dtype(torch, dtype)
    X = as_torch_tensor(X, torch=torch, device=resolved_device, dtype=resolved_dtype)
    Y = as_torch_tensor(Y, torch=torch, device=resolved_device, dtype=resolved_dtype)

    if X.ndim != 2:
        raise ValueError(f"X must be a 2D tensor of shape (N, D), got {tuple(X.shape)}.")
    if Y.ndim != 2:
        raise ValueError(f"Y must be a 2D tensor of shape (M, D), got {tuple(Y.shape)}.")
    if X.shape[1] != Y.shape[1]:
        raise ValueError(
            f"X and Y must have the same feature dimension, got {X.shape[1]} "
            f"and {Y.shape[1]}."
        )
    if p < 1:
        raise ValueError("p must be >= 1")
    if squared and p != 2:
        raise ValueError("`squared=True` only valid for p=2.")

    n_left = X.shape[0]
    n_right = Y.shape[0]
    if n_left == 0 or n_right == 0:
        return torch.empty(
            (n_left, n_right),
            dtype=resolved_dtype,
            device=resolved_device,
        )

    if p == 2:
        x_sq_norms = torch.sum(X * X, dim=1)
        y_sq_norms = torch.sum(Y * Y, dim=1)
        dist2 = x_sq_norms[:, None] + y_sq_norms[None, :] - 2.0 * (X @ Y.T)
        dist2 = torch.clamp(dist2, min=0.0)
        if squared:
            return dist2
        return torch.sqrt(dist2)

    if p == 1:
        return torch.cdist(X, Y, p=1)

    dist = torch.empty(
        (n_left, n_right),
        dtype=resolved_dtype,
        device=resolved_device,
    )
    for i0 in range(0, n_left, block_size):
        i1 = min(i0 + block_size, n_left)
        xi = X[i0:i1]
        for j0 in range(0, n_right, block_size):
            j1 = min(j0 + block_size, n_right)
            yj = Y[j0:j1]
            block = torch.sum(torch.abs(xi[:, None, :] - yj[None, :, :]) ** p, dim=2)
            dist[i0:i1, j0:j1] = block ** (1.0 / p)
    return dist
