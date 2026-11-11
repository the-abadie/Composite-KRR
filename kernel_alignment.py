"""Centered alignment of fitted, unweighted base and product kernel terms."""
from __future__ import annotations

import csv
import json
import logging
from pathlib import Path

import numpy as np
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.utils.validation import check_is_fitted

logger = logging.getLogger("kernel-alignment")

ARTIFACT_NAMES = (
    "kernel_alignment.json", "kernel_alignment.csv",
    "kernel_alignment.png", "kernel_alignment_sample_indices.npy",
)


def _centered_unit_kernel(kernel):
    """Return a Frobenius unit matrix, or None when centering removes it."""
    if np.iscomplexobj(kernel):
        raise ValueError("Kernel matrices must be real.")
    matrix = np.asarray(kernel, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or len(matrix) < 2:
        raise ValueError("Kernel matrices must be square with at least two samples.")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("Kernel matrices must contain only finite values.")
    # Scaling before centering avoids overflow and makes the tolerance relative.
    scale = float(np.max(np.abs(matrix)))
    if scale == 0:
        return None, 0.0
    scaled = matrix / scale
    if not np.allclose(scaled, scaled.T, rtol=1e-7, atol=1e-12):
        raise ValueError("Kernel matrices must be symmetric.")
    mean = scaled.mean(axis=0)
    centered = scaled - mean[None, :] - mean[:, None] + mean.mean()
    norm = float(np.linalg.norm(centered))
    fraction = norm / float(np.linalg.norm(scaled))
    if fraction <= 64 * np.finfo(np.float64).eps:
        return None, fraction
    centered /= norm
    return centered, fraction


def centered_kernel_alignment(left, right):
    """Frobenius cosine of HKH and HLH; None for a constant/zero kernel.

    H = I - 11^T/n. Nondegenerate PSD kernels have alignment in [0, 1];
    indefinite symmetric kernels can have negative alignment. No PSD check or
    diagonal removal is performed. Inputs are never modified.
    """
    a, _ = _centered_unit_kernel(left)
    b, _ = _centered_unit_kernel(right)
    if np.shape(left) != np.shape(right):
        raise ValueError("Kernel matrices must have the same shape and sample order.")
    if a is None or b is None:
        return None
    return float(np.clip(np.einsum("ij,ij->", a, b), -1.0, 1.0))


def analyze_kernel_alignment(
    estimator, X, *, sample_indices=None, max_samples=1000,
    include_products=True, random_state=0, output_dir=None,
):
    """Analyze fitted CKRR geometry on a deterministic subset of supplied rows.

    Accepts fitted exact/Nyström estimators or a fitted TransformedTargetRegressor.
    X is the raw (n_samples, n_descriptors) object matrix used by those estimators.
    Callers supply training/CV rows only; no labels, refits, or predictions are
    needed. Preprocessors and bandwidths come from the final fitted estimator.
    Nyström runs analyze exact kernel terms, not landmark-projected features.
    All diagnostics run on CPU, with float64 alignment accumulation.
    """
    if type(max_samples) is not int or max_samples < 3:
        raise ValueError("max_samples must be an integer of at least 3.")
    if type(include_products) is not bool:
        raise ValueError("include_products must be a boolean.")
    regressor = getattr(estimator, "regressor_", estimator)
    check_is_fitted(regressor, ["X_preprocessors_", "gammas_", "kernel_types_",
                                "kernel_products_", "kernel_weights_", "names_"])
    X = np.asarray(X, dtype=object)
    n_components = len(regressor.X_preprocessors_)
    if X.ndim != 2 or X.shape[1] != n_components or not len(X):
        raise ValueError(f"X must have shape (n_samples, {n_components}) with nonzero rows.")
    indices = np.arange(len(X)) if sample_indices is None else np.asarray(sample_indices)
    if (indices.shape != (len(X),) or indices.dtype.kind not in "iu"
            or np.any(indices < 0) or len(np.unique(indices)) != len(indices)):
        raise ValueError("sample_indices must contain one unique nonnegative integer per X row.")
    selected = np.arange(len(X))
    if len(selected) > max_samples:
        selected = np.sort(np.random.default_rng(random_state).choice(
            len(X), size=max_samples, replace=False,
        ))
    n_samples = len(selected)
    products = list(regressor.kernel_products_) if include_products else []
    factors = [(i,) for i in range(n_components)] + [tuple(p) for p in products]
    names = list(regressor.names_)
    labels = names + [" * ".join(names[i] for i in product) for product in products]
    weights = list(regressor.kernel_weights_)[:len(factors)]
    report = {
        "schema_version": 1, "method": "centered_kernel_alignment",
        "formula": "<H K_i H, H K_j H>_F / (||H K_i H||_F ||H K_j H||_F)",
        "sample_source": "supplied rows (runner uses training/CV pool only)",
        "n_available": len(X), "n_samples": n_samples, "max_samples": max_samples,
        "sampling": "all rows" if n_samples == len(X) else "uniform without replacement; sorted local positions",
        "random_state": random_state, "sample_indices": indices[selected].tolist(),
        "sample_positions": selected.tolist(), "include_products": include_products,
        "weighted": False, "kernel_evaluation": "exact terms on CPU; fitted preprocessing and bandwidths",
        "kernel_dtype": str(regressor.compute_dtype_), "product_dtype": "float64", "alignment_dtype": "float64",
        "gammas": [float(gamma) for gamma in regressor.gammas_],
        "labels": labels, "terms": [], "warnings": [],
    }
    units = []
    bases = []
    if n_samples >= 3:
        for i, preprocessor in enumerate(regressor.X_preprocessors_):
            block = np.stack(X[selected, i]).astype(regressor.compute_dtype_)
            block = block.reshape(n_samples, -1)
            transformed = np.asarray(preprocessor.transform(block), dtype=regressor.compute_dtype_)
            bases.append(np.asarray(pairwise_kernels(
                transformed, metric=regressor.kernel_types_[i], filter_params=True,
                gamma=regressor.gammas_[i], n_jobs=1,
            ), dtype=np.float64))
    for index, term_factors in enumerate(factors):
        term = {"index": index, "label": labels[index],
                "kind": "base" if index < n_components else "product",
                "descriptor_ids": [names[i] for i in term_factors],
                "kernel_types": [regressor.kernel_types_[i] for i in term_factors],
                "fitted_weight": float(weights[index])}
        if n_samples < 3:
            unit, fraction = None, None
            term["status"] = "insufficient_samples"
        else:
            kernel = bases[term_factors[0]]
            if len(term_factors) > 1:
                kernel = kernel.copy()
                for factor in term_factors[1:]:
                    kernel *= bases[factor]
            unit, fraction = _centered_unit_kernel(kernel)
            term["status"] = "constant" if unit is None else "ok"
            if unit is None:
                report["warnings"].append(f"Term {index} ({labels[index]}) has zero numerical centered norm; alignment is undefined.")
            elif fraction < 1e-8:
                report["warnings"].append(f"Term {index} ({labels[index]}) is nearly constant; inspect its bandwidth.")
        term["centered_norm_fraction"] = fraction
        # Alignment with H diagnoses the uninformative near-identity limit.
        term["identity_alignment"] = None if unit is None else float(np.trace(unit) / np.sqrt(n_samples - 1))
        if unit is not None and term["identity_alignment"] > 1 - 1e-6:
            report["warnings"].append(f"Term {index} ({labels[index]}) is nearly identity after centering; high pairwise alignment may be trivial.")
        units.append(unit)
        report["terms"].append(term)
    if n_samples < 3:
        report["warnings"].append("At least three rows are required for meaningful centered alignment; all entries are undefined.")
    matrix = [[None] * len(factors) for _ in factors]
    for i, left in enumerate(units):
        for j in range(i + 1):
            right = units[j]
            if left is not None and right is not None:
                value = 1.0 if i == j else float(np.clip(np.einsum("ij,ij->", left, right), -1.0, 1.0))
                matrix[i][j] = matrix[j][i] = value
    report["alignment"] = matrix
    report["status"] = "ok" if all(t["status"] == "ok" for t in report["terms"]) else "partial"
    for warning in report["warnings"]:
        logger.warning(warning)
    if output_dir is not None:
        _write_artifacts(report, output_dir)
    return report


def _write_artifacts(report, output_dir):
    from matplotlib import colormaps
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / ARTIFACT_NAMES[0]).write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    with (output / ARTIFACT_NAMES[1]).open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["kernel", *report["labels"]])
        for name, row in zip(report["labels"], report["alignment"]):
            writer.writerow([name, *row])
    np.save(output / ARTIFACT_NAMES[3], np.asarray(report["sample_indices"], dtype=np.int64))
    size = min(20, max(5, 2 + len(report["labels"]) * 0.65))
    figure = Figure(figsize=(size, size), layout="constrained")
    FigureCanvasAgg(figure)
    figure.suptitle("Centered kernel alignment", fontsize=14)
    axis = figure.subplots()
    values = np.array(report["alignment"], dtype=float)
    cmap = colormaps["viridis"].copy()
    cmap.set_bad("lightgray")
    lower = -1 if np.any(values < 0) else 0
    plot = axis.imshow(np.ma.masked_invalid(values), vmin=lower, vmax=1, cmap=cmap)
    positions = np.arange(len(report["labels"]))
    axis.set_xticks(positions, report["labels"], rotation=45, ha="right")
    axis.set_yticks(positions, report["labels"])
    axis.set_title(f"{report['n_samples']} training/CV rows; unweighted terms\nGray = undefined", fontsize=11)
    if len(positions) <= 15:
        for i, row in enumerate(report["alignment"]):
            for j, value in enumerate(row):
                axis.text(j, i, "N/A" if value is None else f"{value:.2f}",
                          ha="center", va="center", color="black" if value is None or value > 0.6 else "white")
    figure.colorbar(plot, ax=axis, label="CKA")
    figure.savefig(output / ARTIFACT_NAMES[2], dpi=150)
