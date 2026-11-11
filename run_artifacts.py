"""Machine-readable evidence emitted directly by a completed engine run."""
import hashlib
import importlib.metadata
import json
import os
import platform
import resource
from pathlib import Path

import numpy as np


def file_record(path):
    """Checksum a report or other artifact without interpreting its contents."""
    path = Path(path)
    with path.open("rb") as handle:
        digest = hashlib.file_digest(handle, "sha256").hexdigest()
    return {"path": str(path), "sha256": digest}


def array_record(path, *, archive_key=None):
    path = Path(path)
    digest = file_record(path)["sha256"]
    if path.suffix.lower() == ".json":
        from descriptor_storage import ShardedDescriptorArray
        data = ShardedDescriptorArray(path, archive_key)
        return {"path": str(path), "sha256": digest, "shape": list(data.shape),
                "dtype": str(data.dtype), "archive_key": data.archive_key,
                "storage": "compressed_npz_shards", "manifest": data.manifest}
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
                     test_summary, *, elapsed_seconds, training_seconds, target_names=None,
                     warmup=None, stage_timings=None, kernel_alignment=None):
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
    blocks = [array_record(d.path, archive_key=d.archive_key) for d in spec.descriptors]
    if target_names is None:
        from class_Target import target_names as infer_target_names
        target_names = infer_target_names(spec.target.name, y_train)
    report = {"complete": True, "seed": spec.seed, "n_train": len(idx_train),
              "n_test": len(idx_test), "features": sum(int(np.prod(b["shape"][1:])) for b in blocks),
              "target_names": target_names, "n_targets": len(target_names),
              "scoring": scoring, "scoring_target_space": spec.search.target_space,
              "best_score": float(search_result.best_score_), "cv_objective_error": float(np.mean(errors)),
              "cv_fold_errors": errors.tolist(),
              "cv_mae": float(np.mean(errors)) if scoring == "neg_mean_absolute_error" and spec.search.target_space == "original" else None,
              "cv_transformed_mae": float(np.mean(errors)) if scoring == "neg_mean_absolute_error" and spec.search.target_space == "transformed" else None,
              "cv_fold_std": float(np.std(errors, ddof=1)) if len(errors) > 1 else None,
              **test_summary, "best_params": search_result.best_params_,
              "training_target_mean": np.mean(y_train, axis=0),
              "training_target_median": np.median(y_train, axis=0),
              "elapsed_seconds": elapsed_seconds, "training_seconds": training_seconds,
              "warmup": warmup, "training_stage_seconds": stage_timings,
              "held_out_evaluation": {"prediction_batch_size": spec.execution.prediction_batch_size,
                                      "metric_rows": len(idx_test), "prediction_rows": len(idx_test),
                                      "plot_rows": min(len(idx_test), spec.reporting.plot_max_samples),
                                      "plot_sampling": "deterministic evenly spaced rows"},
              "descriptor_blocks": blocks, "target": array_record(spec.target.path),
              "runtime": runtime,
              "artifacts": {name: array_record(output / name) for name in
                            ("fold_val_idx.npy", "test_idx.npy", "validation_folds.npz",
                             "y_predictions.npy", "y_true.npy", "alpha.npy", "gammas.npy", "kernel_weights.npy")}}
    regressor = getattr(getattr(search_result, "best_estimator_", None), "regressor_", None)
    transformed_counts = getattr(regressor, "transformed_feature_counts_", None)
    if transformed_counts is not None:
        report["kernel_features"] = sum(transformed_counts)
        report["transformed_feature_counts"] = transformed_counts
    if kernel_alignment is not None:
        from kernel_alignment import ARTIFACT_NAMES, TARGET_ARTIFACT_NAMES
        report["kernel_alignment"] = kernel_alignment
        alignment_artifacts = ARTIFACT_NAMES + (TARGET_ARTIFACT_NAMES if kernel_alignment.get("include_target") else ())
        for name in alignment_artifacts:
            path = output / name
            report["artifacts"][name] = array_record(path) if path.suffix == ".npy" else file_record(path)
    baseline = getattr(regressor, "linear_baseline_", None)
    if baseline is not None:
        report["linear_baseline"] = {"descriptor_id": spec.model.linear_baseline,
                                     "target_space": "fold-fitted target transformation",
                                     "fit": "centered least squares with intercept; training samples only"}
        for name, array in zip(("linear_baseline_coefficients.npy", "linear_baseline_intercept.npy"), baseline):
            np.save(output / name, array)
            report["artifacts"][name] = array_record(output / name)
    write_immutable(output / "run_metrics.json", json.dumps(report, indent=2, default=json_value, allow_nan=False) + "\n")
