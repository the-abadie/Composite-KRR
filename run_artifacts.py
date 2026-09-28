"""Machine-readable evidence emitted directly by a completed engine run."""
import hashlib
import importlib.metadata
import json
import os
import platform
import resource
from pathlib import Path

import numpy as np


def array_record(path):
    path = Path(path)
    with path.open("rb") as handle:
        digest = hashlib.file_digest(handle, "sha256").hexdigest()
    data = np.load(path, mmap_mode="r")
    if isinstance(data, np.lib.npyio.NpzFile):
        data.close()
        return {"path": str(path), "sha256": digest}
    return {"path": str(path), "sha256": digest,
            "shape": list(data.shape), "dtype": str(data.dtype)}


def json_value(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Unsupported JSON value: {type(value)}")


def save_run_metrics(spec, search_result, cv, idx_train, idx_test, y_train,
                     test_summary, *, elapsed_seconds, training_seconds):
    from postprocess import best_validation_fold_errors
    from run_config import write_immutable

    output = Path(spec.output.directory)
    folds = {}
    for index, (train, val) in enumerate(cv.split(np.empty(len(idx_train)))):
        folds[f"train_{index}"] = idx_train[train]
        folds[f"validation_{index}"] = idx_train[val]
    np.savez(output / "validation_folds.npz", **folds)
    scoring = "neg_root_mean_squared_error" if spec.search.scoring == "rmse" else spec.search.scoring
    errors = best_validation_fold_errors(search_result, scoring=scoring)
    runtime = {"hostname": platform.node(), "python": platform.python_version(),
               "max_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
               "slurm": {k: v for k, v in os.environ.items() if k.startswith("SLURM_")},
               "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"), "versions": {}}
    for package in ("numpy", "scipy", "scikit-learn", "optuna", "torch", "pydantic"):
        try:
            runtime["versions"][package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            pass
    if spec.execution.backend == "pytorch":
        import torch
        runtime["cuda_runtime"] = torch.version.cuda
        runtime["gpus"] = [{"index": i, "name": torch.cuda.get_device_name(i),
                            "peak_allocated_bytes": torch.cuda.max_memory_allocated(i)}
                           for i in range(torch.cuda.device_count())]
    blocks = [array_record(d.path) for d in spec.descriptors]
    report = {"complete": True, "seed": spec.seed, "n_train": len(idx_train),
              "n_test": len(idx_test), "features": sum(int(np.prod(b["shape"][1:])) for b in blocks),
              "scoring": scoring, "best_score": float(search_result.best_score_),
              "cv_fold_errors": errors.tolist(),
              "cv_mae": float(np.mean(errors)) if scoring == "neg_mean_absolute_error" else None,
              "cv_fold_std": float(np.std(errors, ddof=1)) if len(errors) > 1 else None,
              **test_summary, "best_params": search_result.best_params_,
              "training_target_mean": np.mean(y_train, axis=0),
              "training_target_median": np.median(y_train, axis=0),
              "elapsed_seconds": elapsed_seconds, "training_seconds": training_seconds,
              "descriptor_blocks": blocks, "target": array_record(spec.target.path),
              "runtime": runtime,
              "artifacts": {name: array_record(output / name) for name in
                            ("fold_val_idx.npy", "test_idx.npy", "validation_folds.npz",
                             "y_predictions.npy", "y_true.npy", "alpha.npy", "gammas.npy", "kernel_weights.npy")}}
    write_immutable(output / "run_metrics.json", json.dumps(report, indent=2, default=json_value, allow_nan=False) + "\n")
