"""Append-only, JSON-safe evidence from a single independent search."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from time import perf_counter

import numpy as np


def json_safe(value):
    if isinstance(value, np.random.RandomState):
        return {"type": "numpy.RandomState", "state": json_safe(value.get_state())}
    if isinstance(value, np.random.Generator):
        return {"type": "numpy.Generator", "state": json_safe(value.bit_generator.state)}
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_safe(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def describe_distribution(distribution):
    if hasattr(distribution, "bounds"):
        return {"distribution": "loguniform", "bounds": distribution.bounds}
    if hasattr(distribution, "alpha"):
        return {"distribution": "dirichlet", "alpha": distribution.alpha}
    if getattr(getattr(distribution, "dist", None), "name", None) == "loguniform":
        return {"distribution": "loguniform", "bounds": list(distribution.args)}
    if isinstance(distribution, (list, tuple)):
        return {"distribution": "choices", "values": distribution}
    raise TypeError(f"Cannot record search distribution {type(distribution).__name__}")


def candidate_parameters(estimator, candidate, n_components):
    """Include fixed Stage 1 weights as well as sampled parameters."""
    values = estimator.get_params(deep=True)
    values.update(candidate)
    gamma_key = next(k for k in values if k == "gammas" or k.endswith("__gammas"))
    prefix = gamma_key[:-len("gammas")]
    gammas = values.get(gamma_key)
    if gammas is None:
        gammas = [1.] * n_components
    if np.isscalar(gammas):
        gammas = [gammas] * n_components
    from kernel_mixing import kernel_term_count
    from kernel_cache import extract_regressor
    regressor = extract_regressor(estimator)
    weights = values.get(prefix + "kernel_weights")
    if weights is None:
        weights = [1.] * kernel_term_count(regressor, n_components)
    weights = np.asarray(weights, dtype=float)
    if getattr(regressor, "normalize_kernel_weights", False):
        weights = weights / weights.sum()
    return {"alpha": values[prefix + "alpha"], "gammas": gammas,
            "kernel_weights": weights}


class SearchJournal:
    """Flush completed batches/trials; keep partial evidence on exceptions.

    This records a fresh search, not a resumable optimizer checkpoint.
    """
    def __init__(self, output_dir, *, settings, context=None):
        self.output = Path(output_dir)
        self.output.mkdir(parents=True, exist_ok=True)
        self.history = self.output / "search_history.jsonl"
        self.summary = self.output / "search_summary.json"
        if self.summary.exists():
            raise FileExistsError(f"Refusing to replace search evidence: {self.summary}")
        self.history.open("x").close()
        self.settings = json_safe(settings)
        self.context = json_safe(context or {})
        self.started = perf_counter()
        self.stages = []
        self.n_trials = 0
        self.best_score = None
        self.best_params = None
        self.best_trial = None
        self.finished = False
        self.append({"event": "search_started", "schema_version": 1,
                     "settings": self.settings, "context": self.context})

    def __enter__(self):
        return self

    def __exit__(self, kind, error, traceback):
        if not self.finished:
            for stage in self.stages:
                if stage.status == "running":
                    stage.finish(status="failed", error=None if error is None else {
                        "type": type(error).__name__, "message": str(error)})
            self.finish(status="failed", error=None if error is None else {
                "type": type(error).__name__, "message": str(error)})
        return False

    def append(self, record):
        payload = json_safe({"elapsed_seconds": perf_counter() - self.started, **record})
        with self.history.open("a") as handle:
            handle.write(json.dumps(payload, sort_keys=True, allow_nan=False) + "\n")

    def stage(self, name, *, requested_trials, bounds, estimator=None,
              n_components=None):
        record = StageJournal(self, name, requested_trials=requested_trials,
                              bounds=bounds, estimator=estimator,
                              n_components=n_components)
        self.stages.append(record)
        return record

    def finish(self, *, status="complete", error=None, selected=None):
        report = {"schema_version": 1, "status": status, "settings": self.settings,
                  "context": self.context, "elapsed_seconds": perf_counter() - self.started,
                  "n_trials": self.n_trials, "best_score": self.best_score,
                  "best_trial": self.best_trial, "best_params": self.best_params,
                  "stages": [s.summary() for s in self.stages], "error": error, "selected": selected}
        self.append({"event": "search_finished", **report})
        report["history_sha256"] = hashlib.sha256(self.history.read_bytes()).hexdigest()
        with self.summary.open("x") as handle:
            handle.write(json.dumps(json_safe(report), indent=2, allow_nan=False) + "\n")
        self.finished = True
        return {"status": status, "n_trials": self.n_trials,
                "history_path": self.history.name, "summary_path": self.summary.name}


class StageJournal:
    def __init__(self, parent, name, *, requested_trials, bounds, estimator, n_components):
        self.parent, self.name = parent, name
        self.requested_trials = requested_trials
        self.bounds = json_safe(bounds)
        self.estimator, self.n_components = estimator, n_components
        self.started = perf_counter()
        self.n_trials = 0
        self.failed_trials = 0
        self.best_score = None
        self.status = "running"
        self.stop_reason = None
        self.parent.append({"event": "stage_started", "stage": name,
                            "requested_trials": requested_trials, "bounds": self.bounds})

    def trial(self, params, split_scores, *, stage_trial=None, timing=None, **extra):
        scores = np.asarray(split_scores, dtype=float)
        valid = scores.size > 0 and bool(np.all(np.isfinite(scores)))
        score = float(scores.mean()) if valid else None
        self.n_trials += 1
        self.failed_trials += int(not valid)
        self.parent.n_trials += 1
        improved = score is not None and (self.parent.best_score is None or score > self.parent.best_score)
        if score is not None and (self.best_score is None or score > self.best_score):
            self.best_score = score
        if improved:
            self.parent.best_score = score
            self.parent.best_params = json_safe(params)
            self.parent.best_trial = self.parent.n_trials
        record = {"event": "trial", "trial": self.parent.n_trials, "stage": self.name,
                  "stage_trial": stage_trial or self.n_trials, "params": params,
                  "split_scores": scores, "mean_score": score,
                  "std_score": float(scores.std()) if valid else None,
                  "status": "complete" if valid else "invalid",
                  "failed_fold_indices": np.flatnonzero(~np.isfinite(scores)),
                  "improved": improved, "best_score": self.parent.best_score,
                  "timing": timing or {}, **extra}
        self.parent.append(record)

    def random_batch(self, candidates, split_scores, *, seconds, timing=None):
        for candidate, scores in zip(candidates, split_scores):
            self.trial(candidate_parameters(self.estimator, candidate, self.n_components), scores,
                       sampled_params=candidate, timing={"scope": "stage_batch",
                       "batch_wall_seconds": seconds, "batch_size": len(candidates),
                       "amortized_seconds": seconds / len(candidates), **(timing or {})})

    def failed_batch(self, candidates, error, *, seconds):
        for candidate in candidates:
            self.trial(candidate_parameters(self.estimator, candidate, self.n_components), [],
                       sampled_params=candidate, status="failed",
                       error={"type": type(error).__name__, "message": str(error)},
                       timing={"scope": "failed_batch", "batch_wall_seconds": seconds,
                               "batch_size": len(candidates)})

    def finish(self, *, status="complete", stop_reason=None, error=None):
        self.status, self.stop_reason = status, stop_reason
        self.seconds = perf_counter() - self.started
        self.parent.append({"event": "stage_finished", **self.summary(), "error": error})

    def summary(self):
        return {"stage": self.name, "requested_trials": self.requested_trials,
                "recorded_trials": self.n_trials, "invalid_trials": self.failed_trials,
                "best_score": self.best_score, "bounds": self.bounds,
                "elapsed_seconds": getattr(self, "seconds", perf_counter() - self.started),
                "status": self.status, "stop_reason": self.stop_reason}


class MeaningfulProgress:
    """Patience anchor separate from exact best-candidate selection."""
    def __init__(self, absolute=0., relative=0.):
        if not all(math.isfinite(v) and v >= 0 for v in (absolute, relative)):
            raise ValueError("Patience improvement tolerances must be finite and non-negative.")
        self.absolute, self.relative = absolute, relative
        self.anchor = None
        self.last_trial = 0

    def observe(self, trial_number, score):
        if self.anchor is None:
            self.anchor, self.last_trial = score, trial_number
            return True
        tolerance = max(self.absolute, self.relative * abs(self.anchor)) if math.isfinite(self.anchor) else 0.
        if score > self.anchor + tolerance:
            self.anchor, self.last_trial = score, trial_number
            return True
        return False
