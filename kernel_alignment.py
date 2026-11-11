"""Centered alignment of fitted kernel terms with each other and with targets."""
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
TARGET_ARTIFACT_NAMES = ("kernel_target_alignment.csv", "kernel_target_alignment.png")


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
    y=None, include_target=False, target_space="original", target_names=None,
):
    """Analyze fitted CKRR geometry on a deterministic subset of supplied rows.

    Accepts fitted exact/Nyström estimators or a fitted TransformedTargetRegressor.
    X is the raw (n_samples, n_descriptors) object matrix used by those estimators.
    Callers supply training/CV rows only. Optional kernel-target alignment uses
    matching y rows and one linear target kernel per output column, without
    refitting. Transformed target space requires a fitted target wrapper.
    Preprocessors and bandwidths come from the final fitted estimator.
    Nyström runs analyze exact kernel terms, not landmark-projected features.
    All diagnostics run on CPU, with float64 alignment accumulation.
    """
    if type(max_samples) is not int or max_samples < 3:
        raise ValueError("max_samples must be an integer of at least 3.")
    if type(include_products) is not bool:
        raise ValueError("include_products must be a boolean.")
    if type(include_target) is not bool:
        raise ValueError("include_target must be a boolean.")
    if target_space not in ("original", "transformed"):
        raise ValueError("target_space must be original or transformed.")
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
    target_vectors = None
    target_report = None
    if include_target:
        target_vectors, target_report = _prepare_targets(
            estimator, y, len(X), selected, target_space, target_names,
        )
    products = list(regressor.kernel_products_) if include_products else []
    factors = [(i,) for i in range(n_components)] + [tuple(p) for p in products]
    names = list(regressor.names_)
    labels = names + [" * ".join(names[i] for i in product) for product in products]
    weights = list(regressor.kernel_weights_)[:len(factors)]
    report = {
        "schema_version": 2, "method": "centered_kernel_alignment",
        "formula": "<H K_i H, H K_j H>_F / (||H K_i H||_F ||H K_j H||_F)",
        "sample_source": "supplied rows (runner uses training/CV pool only)",
        "n_available": len(X), "n_samples": n_samples, "max_samples": max_samples,
        "sampling": "all rows" if n_samples == len(X) else "uniform without replacement; sorted local positions",
        "random_state": random_state, "sample_indices": indices[selected].tolist(),
        "sample_positions": selected.tolist(), "include_products": include_products,
        "include_target": include_target,
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
    if target_report is not None:
        target_report["alignment"] = [
            [None if unit is None or vector is None else
             float(np.clip(vector @ unit @ vector, -1.0, 1.0))
             for vector in target_vectors]
            for unit in units
        ]
        report["target_alignment"] = target_report
        for column in target_report["targets"]:
            if column["status"] != "ok":
                report["warnings"].append(
                    f"Target {column['index']} ({column['name']}) has status {column['status']}; target alignment is undefined."
                )
    report["status"] = "ok" if all(t["status"] == "ok" for t in report["terms"]) else "partial"
    if target_report is not None and any(t["status"] != "ok" for t in target_report["targets"]):
        report["status"] = "partial"
    for warning in report["warnings"]:
        logger.warning(warning)
    if output_dir is not None:
        _write_artifacts(report, output_dir)
    return report


def _prepare_targets(estimator, y, n_rows, selected, target_space, target_names):
    from target_utils import as_target_matrix

    if y is None or np.iscomplexobj(y):
        raise ValueError("Kernel-target alignment requires real y with one row per X row.")
    values = as_target_matrix(y, dtype=np.float64)
    if len(values) != n_rows or not np.all(np.isfinite(values)):
        raise ValueError("y must have one row per X row and contain only finite values.")
    values = values[selected]
    transformer = getattr(estimator, "transformer_", None)
    if target_space == "transformed":
        if transformer is None:
            raise ValueError("Transformed target alignment requires a fitted target transformer on the estimator.")
        target_shape = values.shape
        values = as_target_matrix(transformer.transform(values), dtype=np.float64)
        if values.shape != target_shape or not np.all(np.isfinite(values)):
            raise ValueError("The fitted target transformer must return finite values with matching target shape.")
    n_targets = values.shape[1]
    if target_names is None:
        names = ["target"] if n_targets == 1 else [f"target_{i + 1}" for i in range(n_targets)]
    else:
        names = [target_names] if isinstance(target_names, str) else list(target_names)
        if len(names) != n_targets or any(not isinstance(name, str) or not name for name in names):
            raise ValueError("target_names must contain one nonempty name per target column.")
    columns, vectors = [], []
    for index, name in enumerate(names):
        vector = None
        status = "insufficient_samples"
        if len(selected) >= 3:
            column = values[:, index]
            # Bound arithmetic before centering, then normalize only the varying
            # part. This avoids overflow and retains small, representable variation.
            scale = float(np.max(np.abs(column)))
            centered = column / scale if scale else column.copy()
            centered = centered - centered[0]
            centered -= centered.mean()
            spread = float(np.max(np.abs(centered)))
            status = "constant" if spread == 0 else "ok"
            if spread:
                vector = centered / spread
                vector /= np.linalg.norm(vector)
        columns.append({"index": index, "name": name, "status": status})
        vectors.append(vector)
    return vectors, {
        "method": "centered_kernel_target_alignment",
        "target_kernel": "one linear outer-product kernel per target column",
        "formula": "y_c^T (H K H) y_c / (||H K H||_F ||y_c||_2^2)",
        "target_space": target_space,
        "target_transform": type(transformer).__name__ if target_space == "transformed" else None,
        "target_names": names, "targets": columns,
        "label_source": "same sampled rows as the kernel matrix",
        "baseline_residualized": False,
    }


def _write_artifacts(report, output_dir):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / ARTIFACT_NAMES[0]).write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    with (output / ARTIFACT_NAMES[1]).open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["kernel", *report["labels"]])
        for name, row in zip(report["labels"], report["alignment"]):
            writer.writerow([name, *row])
    np.save(output / ARTIFACT_NAMES[3], np.asarray(report["sample_indices"], dtype=np.int64))
    _plot_alignment(report["alignment"], report["labels"], report["labels"],
                    "Centered kernel alignment", report["n_samples"], output / ARTIFACT_NAMES[2])
    if report["include_target"]:
        target = report["target_alignment"]
        with (output / TARGET_ARTIFACT_NAMES[0]).open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["kernel", *target["target_names"]])
            for label, row in zip(report["labels"], target["alignment"]):
                writer.writerow([label, *row])
        _plot_alignment(target["alignment"], report["labels"], target["target_names"],
                        f"Kernel–target alignment ({target['target_space']})",
                        report["n_samples"], output / TARGET_ARTIFACT_NAMES[1])


def _plot_alignment(matrix, row_labels, column_labels, title, n_samples, path):
    from matplotlib import colormaps
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    width = min(20, max(5, 2 + len(column_labels) * 0.65))
    height = min(20, max(5, 2 + len(row_labels) * 0.65))
    figure = Figure(figsize=(width, height), layout="constrained")
    FigureCanvasAgg(figure)
    figure.suptitle(title, fontsize=14)
    axis = figure.subplots()
    values = np.array(matrix, dtype=float)
    cmap = colormaps["viridis"].copy()
    cmap.set_bad("lightgray")
    lower = -1 if np.any(values < 0) else 0
    plot = axis.imshow(np.ma.masked_invalid(values), vmin=lower, vmax=1, cmap=cmap, aspect="auto")
    axis.set_xticks(np.arange(len(column_labels)), column_labels, rotation=45, ha="right")
    axis.set_yticks(np.arange(len(row_labels)), row_labels)
    axis.set_title(f"{n_samples} training/CV rows; unweighted terms\nGray = undefined", fontsize=11)
    if max(len(row_labels), len(column_labels)) <= 15:
        for i, row in enumerate(matrix):
            for j, value in enumerate(row):
                axis.text(j, i, "N/A" if value is None else f"{value:.2f}",
                          ha="center", va="center", color="black" if value is None or value > 0.6 else "white")
    figure.colorbar(plot, ax=axis, label="CKA")
    figure.savefig(path, dpi=150)
