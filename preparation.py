import random
import numpy as np

def randomized_selection(N):
    pass

def stratified_selection(target, n_strata:int, n_total:int, rng):
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

def stratified_selection_with_remainder(target, n_strata:int, n_total:int, rng):
    idx = stratified_selection(target, n_strata, n_total, rng=rng)
    mask = np.ones(len(target), dtype=bool)
    mask[idx] = False
    rest = np.where(mask)[0]
    return idx, rest

def pad_or_trim_last(X, L:int, fill):
    """
    Make X[..., :] have length L on the last axis by trimming or zero-padding.
    Works for (N,B) or any shape ending in B.
    """
    X = np.asarray(X, dtype=float)
    cur = X.shape[-1]
    if cur == L:
        return X
    if cur > L:
        return X[..., :L]

    pad = [(0, 0)] * (X.ndim - 1) + [(0, L - cur)]
    return np.pad(X, pad, mode="constant", constant_values=fill)
