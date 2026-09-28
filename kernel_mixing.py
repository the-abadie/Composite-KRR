"""Shared additive/product kernel layout and assembly (NumPy or Torch)."""

from numbers import Integral

import numpy as np


def resolve_kernel_products(kernel_products, n_components: int) -> tuple[tuple[int, ...], ...]:
    """Products reference zero-based base-kernel indices; bandwidths are shared."""
    if kernel_products is None:
        return ()
    if not isinstance(kernel_products, (list, tuple)):
        raise ValueError("kernel_products must be a list of index lists.")
    products = []
    for factors in kernel_products:
        if not isinstance(factors, (list, tuple)) or len(factors) < 2:
            raise ValueError("Each kernel product must contain at least two base indices.")
        if any(
            isinstance(index, bool) or not isinstance(index, Integral)
            or not 0 <= index < n_components
            for index in factors
        ):
            raise ValueError(f"Kernel product indices must be integers in [0, {n_components}).")
        product = tuple(sorted(int(index) for index in factors))
        if product in products:
            raise ValueError(f"Duplicate kernel product: {product}.")
        products.append(product)
    return tuple(products)


def kernel_term_count(regressor, n_components: int) -> int:
    return n_components + len(resolve_kernel_products(
        getattr(regressor, "kernel_products", None), n_components
    ))


def validate_kernel_parameters(gammas, weights, n_components: int, kernel_products=()):
    products = resolve_kernel_products(kernel_products, n_components)
    for name, values, size in (
        ("gammas", gammas, n_components),
        ("kernel_weights", weights, n_components + len(products)),
    ):
        array = np.asarray(values, dtype=float)
        if array.shape != (size,):
            raise ValueError(f"{name} must have length {size}, got shape {array.shape}.")
        if not np.all(np.isfinite(array)) or np.any(array < 0):
            raise ValueError(f"{name} must contain finite non-negative values.")
    return products


def mix_kernel_terms(base_kernels, weights, kernel_products=(), *, out=None):
    """Consume unweighted kernels once, retaining only factors needed by products.

    Arrays/tensors stay on their original device. Weights may be scalars or
    broadcastable candidate batches. Neither base kernels nor weights are mutated.
    Products are elementwise, with their own weight independent of factor weights.
    """
    needed = {index for factors in kernel_products for index in factors}
    retained = {}
    total = out
    if total is not None:
        total[...] = 0
    n_base = 0
    for index, kernel in enumerate(base_kernels):
        n_base += 1
        if index in needed:
            retained[index] = kernel
        weighted = kernel * weights[index]
        if total is None:
            total = weighted
        else:
            total += weighted
    if total is None:
        raise ValueError("At least one base kernel is required.")
    for term_index, factors in enumerate(kernel_products):
        product = retained[factors[0]] * retained[factors[1]]
        for index in factors[2:]:
            product *= retained[index]
        product *= weights[n_base + term_index]
        total += product
    return total
