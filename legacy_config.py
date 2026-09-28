"""Translate the former flat JSON/TOML format without importing Python config files."""
from __future__ import annotations

import math
from typing import Any

from config_schema import RunConfig

# Only field names are mapped here; defaults and validation belong to RunConfig.
FIELDS = {
    "SEED": "seed", "RUN_NAME": "run_name", "VERBOSITY": "verbosity",
    "Y_PATH": "target.path", "Y_NAME": "target.name", "Y_NORM": "target.normalization",
    "OUTPUT_DIR": "output.directory", "OVERWRITE_OK": "output.overwrite",
    "KRR_BACKEND": "model.backend",
    "KRR_ALPHA_BOUNDS": "search.alpha_bounds", "KRR_GAMMA_BOUNDS": "search.gamma_bounds",
    "KRR_RANDOM_SEARCH_STAGE1": "search.random.stage1",
    "KRR_RANDOM_SEARCH_STAGE2": "search.random.stage2",
    "KRR_RANDOM_SEARCH_STAGE3": "search.random.stage3",
    "KRR_BAYESIAN_SEARCH_TRIALS": "search.bayesian.trials",
    "KRR_BAYESIAN_SEARCH_TIMEOUT": "search.bayesian.timeout",
    "KRR_BAYESIAN_SEARCH_PATIENCE": "search.bayesian.patience",
    "KRR_BAYESIAN_BATCH_SIZE": "search.bayesian.batch_size",
    "KRR_TOP_K_FRACTION": "search.top_k_fraction",
    "KRR_TOP_K_MIN_CANDIDATES": "search.top_k_min_candidates",
    "KRR_GAMMA_PRIOR_MAX_SAMPLES": "search.gamma_prior_max_samples",
    "KRR_SCORE_METRIC": "search.scoring",
    "KRR_EVALUATE_KERNEL_CONTRIBUTIONS": "reporting.kernel_contributions",
    "KRR_KERNEL_CONTRIBUTION_BAYESIAN_SEARCH_TRIALS": "reporting.contribution_bayesian_trials",
    "KRR_RANDOM_SEARCH_N_JOBS": "execution.n_jobs",
    "KRR_RANDOM_SEARCH_BLAS_THREADS": "execution.blas_threads",
    "KRR_COMPUTE_DTYPE": "execution.dtype",
    "KRR_USE_DISTANCE_CACHE": "execution.distance_cache.enabled",
    "KRR_DISTANCE_BLOCK_SIZE": "execution.distance_cache.block_size",
    "KRR_DISTANCE_CACHE_DTYPE": "execution.distance_cache.dtype",
    "KRR_DISTANCE_CACHE_N_JOBS": "execution.distance_cache.n_jobs",
    "KRR_DISTANCE_CACHE_MEMORY_FRACTION": "execution.distance_cache.memory_fraction",
    "KRR_CACHED_SCORING_BACKEND": "execution.backend",
    "KRR_PYTORCH_DEVICE": "execution.torch.device",
    "KRR_PYTORCH_DEVICES": "execution.torch.devices",
    "KRR_PYTORCH_CANDIDATE_BATCH_SIZE": "execution.torch.candidate_batch_size",
    "KRR_PYTORCH_PREDICT_BATCH_SIZE": "execution.torch.predict_batch_size",
    "KRR_NYSTROM_N_LANDMARKS": "model.nystrom.n_landmarks",
    "KRR_NYSTROM_LANDMARK_SELECTION": "model.nystrom.landmark_selection",
    "KRR_NYSTROM_BATCH_SIZE": "model.nystrom.batch_size",
    "KRR_NYSTROM_EIGENVALUE_FLOOR": "model.nystrom.eigenvalue_floor",
}
SPLIT_FIELDS = {
    "N_SAMPLES": "n_samples", "N_TRAIN": "n_train", "TRAIN_VAL_SPLIT": "train_fraction",
    "N_KFOLD": "n_folds", "STRATIFY": "stratify", "N_STRATA": "n_strata",
}
PREDEFINED_FIELDS = {
    "PREDEF_TRAINING_IDX_PATH": "train_indices",
    "PREDEF_VAL_KFOLD_IDX_PATH": "validation_folds",
    "PREDEF_TESTING_IDX_PATH": "test_indices",
}
SPECIAL_FIELDS = {"X_PATHS", "X_NAMES", "X_NORMS", "X_PCA_COMPONENTS", "X_PCA_WHITEN",
                  "KRR_KERNEL", "KRR_KERNEL_PRODUCTS", "USE_PREDEFINED_SPLITS"}
ALLOWED_FIELDS = set(FIELDS) | set(SPLIT_FIELDS) | set(PREDEFINED_FIELDS) | SPECIAL_FIELDS


def _set_nested(document, path, value):
    keys = path.split(".")
    for key in keys[:-1]:
        document = document.setdefault(key, {})
    document[keys[-1]] = value


def _sequence(values, size, name, default=None):
    value = values.get(name, default)
    if not isinstance(value, list):
        value = [value] * size
    if len(value) != size:
        raise ValueError(f"{name} must have length {size}.")
    return value


def from_flat(values: dict[str, Any]) -> dict[str, Any]:
    unknown = set(values) - ALLOWED_FIELDS
    if unknown:
        raise ValueError(f"Unknown KRR configuration fields: {sorted(unknown)}")
    for key in ("X_PATHS", "X_NAMES", "X_NORMS"):
        if not isinstance(values.get(key), list) or not values[key]:
            raise ValueError(f"{key} must be a non-empty list.")
    size = len(values["X_PATHS"])
    fields = {
        "id": _sequence(values, size, "X_NAMES"),
        "path": values["X_PATHS"],
        "normalization": _sequence(values, size, "X_NORMS"),
        "kernel": _sequence(values, size, "KRR_KERNEL", "rbf"),
        "pca_components": _sequence(values, size, "X_PCA_COMPONENTS"),
        "pca_whiten": _sequence(values, size, "X_PCA_WHITEN", False),
    }
    result = {"schema_version": 1, "descriptors": [
        {name: column[i] for name, column in fields.items()} for i in range(size)
    ]}
    for old, new in FIELDS.items():
        if old in values:
            _set_nested(result, new, values[old])
    predefined = values.get("USE_PREDEFINED_SPLITS", False)
    if type(predefined) is not bool:
        raise ValueError("USE_PREDEFINED_SPLITS must be a boolean.")
    split_fields = PREDEFINED_FIELDS if predefined else SPLIT_FIELDS
    split = {"mode": "predefined" if predefined else "random"}
    split.update({new: values[old] for old, new in split_fields.items() if old in values})
    if not predefined and split.get("n_train") is not None and split.get("train_fraction") is not None:
        count, fraction, total = split["n_train"], split["train_fraction"], split.get("n_samples")
        if type(count) is not int or type(fraction) not in (int, float) or type(total) is not int:
            raise ValueError("N_TRAIN and TRAIN_VAL_SPLIT require numeric, consistent N_SAMPLES.")
        if not math.isfinite(fraction) or (int(total * fraction) != count and not math.isclose(total * fraction, count, abs_tol=1e-10)):
            raise ValueError("N_TRAIN conflicts with N_SAMPLES * TRAIN_VAL_SPLIT.")
        split.pop("train_fraction")
    result["split"] = split
    products = values.get("KRR_KERNEL_PRODUCTS", [])
    if products is None:
        products = []
    if not isinstance(products, list):
        raise ValueError("KRR_KERNEL_PRODUCTS must be a list of index lists.")
    result["kernel_products"] = []
    for factors in products:
        if not isinstance(factors, list) or len(factors) < 2 or any(
            type(i) is not int or not 0 <= i < size for i in factors
        ):
            raise ValueError("KRR_KERNEL_PRODUCTS requires valid zero-based descriptor indices.")
        result["kernel_products"].append([fields["id"][i] for i in factors])
    aliases = {"dense": "exact", "nyström": "nystrom"}
    if "model" in result and isinstance(result["model"].get("backend"), str):
        value = result["model"]["backend"].lower()
        result["model"]["backend"] = aliases.get(value, value)
    aliases = {"np": "numpy", "cpu": "numpy", "torch": "pytorch", "gpu": "pytorch", "cuda": "pytorch", "rocm": "pytorch"}
    if "execution" in result and isinstance(result["execution"].get("backend"), str):
        value = result["execution"]["backend"].lower()
        result["execution"]["backend"] = aliases.get(value, value)
    for descriptor in result["descriptors"]:
        if isinstance(descriptor["kernel"], str):
            descriptor["kernel"] = descriptor["kernel"].lower()
    return result


def to_flat(model: RunConfig) -> dict[str, Any]:
    """Compatibility view for existing Python clients; returns a fresh dictionary."""
    document = model.model_dump(mode="json")
    values = {}
    for old, path in FIELDS.items():
        value = document
        for key in path.split("."):
            value = value[key]
        values[old] = value
    for old, field in (("X_NAMES", "id"), ("X_PATHS", "path"), ("X_NORMS", "normalization"),
                       ("KRR_KERNEL", "kernel"), ("X_PCA_COMPONENTS", "pca_components"), ("X_PCA_WHITEN", "pca_whiten")):
        values[old] = [d[field] for d in document["descriptors"]]
    names = values["X_NAMES"]
    values["KRR_KERNEL_PRODUCTS"] = [[names.index(name) for name in term] for term in model.kernel_products]
    values["USE_PREDEFINED_SPLITS"] = model.split.mode == "predefined"
    for old, key in {**SPLIT_FIELDS, **PREDEFINED_FIELDS}.items():
        values[old] = document["split"].get(key)
    if model.split.mode == "random" and model.split.n_samples is not None:
        values["N_TRAIN"] = model.split.training_count(model.split.n_samples)
    values["KRR_DISTANCE_CACHE_DTYPE"] = model.execution.distance_cache.dtype or model.execution.dtype
    return values
