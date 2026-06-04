from dataclasses import dataclass

import logging
import numpy as np
from numpy.linalg import LinAlgError
from sklearn.base import clone
from sklearn.model_selection import cross_val_score
from kernel_cache import cached_cross_val_scores
from postprocess import bayesian_search_history
from utilities import configure_logging
from config import VERBOSITY

configure_logging(VERBOSITY)
logger = logging.getLogger("search")

try:
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
except ImportError:
    optuna = None


@dataclass(frozen=True)
class BayesianSearchResult:
    study: object
    best_estimator_: object
    best_params_: dict
    best_score_: float
    best_split_scores_: np.ndarray
    search_history_: list[dict]


def fit_bayesian_search(
    estimator,
    X,
    y,
    *,
    n_components: int,
    alpha_bounds: tuple[float, float],
    gamma_bounds: tuple[float, float] | list[tuple[float, float]],
    kernel_weight_bounds: tuple[float, float],
    initial_params: dict,
    scoring: str,
    cv,
    random_state=None,
    n_jobs=None,
    blas_threads: int | None = 1,
    timeout: float | None = None,
    patience: int | None = None,
    n_trials: int,
    prefix: str,
    stage_name: str = "Bayesian search",
    distance_cache=None,
    kernel_weight_center=None,
    kernel_weight_logit_radius: float = 1.5,
) -> BayesianSearchResult:
    if optuna is None:
        raise ImportError("Optuna is required for the optional Bayesian stage.")
    if n_trials <= 0:
        raise ValueError(f"n_trials must be positive, got {n_trials}.")
    if n_components <= 0:
        raise ValueError(f"n_components must be positive, got {n_components}.")
    if patience is not None and patience <= 0:
        raise ValueError(f"patience must be positive when set, got {patience}.")

    _validate_log_bounds("alpha_bounds", alpha_bounds)
    component_gamma_bounds = _as_component_bounds(gamma_bounds, n_components)
    _validate_kernel_weight_bounds(kernel_weight_bounds)
    if kernel_weight_logit_radius <= 0:
        raise ValueError(
            "kernel_weight_logit_radius must be positive, "
            f"got {kernel_weight_logit_radius}."
        )
    sampler_seed = random_state if isinstance(random_state, int) else None
    sampler = optuna.samplers.TPESampler(seed=sampler_seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)

    unprefixed_initial_params = _unprefix_params(initial_params, prefix)
    if kernel_weight_center is None and n_components > 1:
        kernel_weight_center = unprefixed_initial_params["kernel_weights"]
    kernel_weight_center = (
        None
        if kernel_weight_center is None
        else _normalize_weights(kernel_weight_center, size=n_components)
    )
    _enqueue_initial_bayesian_trial(
        study,
        unprefixed_initial_params,
        n_components=n_components,
    )

    def objective(trial):
        params = _suggest_bayesian_params(
            trial,
            n_components=n_components,
            alpha_bounds=alpha_bounds,
            gamma_bounds=component_gamma_bounds,
            kernel_weight_bounds=kernel_weight_bounds,
            kernel_weight_center=kernel_weight_center,
            kernel_weight_logit_radius=kernel_weight_logit_radius,
        )
        if distance_cache is None:
            candidate = _clone_with_params(estimator, prefix=prefix, **params)
            scores = cross_val_score(
                candidate,
                X,
                y,
                scoring=scoring,
                cv=cv,
                n_jobs=n_jobs,
                error_score="raise",
            )
        else:
            try:
                scores = cached_cross_val_scores(
                    estimator,
                    _prefix_params(params, prefix),
                    distance_cache,
                    scoring=scoring,
                    n_jobs=n_jobs,
                    blas_threads=blas_threads,
                )
            except LinAlgError:
                return -np.inf
            if not np.all(np.isfinite(scores)):
                return -np.inf

        scores = np.asarray(scores, dtype=float)
        if not np.all(np.isfinite(scores)):
            return -np.inf

        trial.set_user_attr("split_scores", scores.tolist())
        return float(np.mean(scores))

    milestones = _progress_milestones(n_trials)

    def log_progress(study, trial):
        completed = len(study.trials)
        if completed in milestones:
            logger.info(
                f"{stage_name} progress: "
                f"{completed}/{n_trials} ({completed / n_trials:.0%})"
            )

    def stop_after_patience(study, trial):
        if patience is None or trial.value is None:
            return

        trials_since_improvement = trial.number - study.best_trial.number
        if trials_since_improvement >= patience:
            logger.warning(
                f"{stage_name} stopped after {trials_since_improvement} "
                "trials without improvement "
                f"(best trial {study.best_trial.number + 1})."
            )
            study.stop()

    study.optimize(
        objective,
        n_trials=n_trials,
        timeout=timeout,
        callbacks=[log_progress, stop_after_patience],
    )

    best_params = _params_from_bayesian_trial(
        study.best_trial.params,
        n_components=n_components,
        kernel_weight_center=kernel_weight_center,
    )
    best_estimator = _clone_with_params(estimator, prefix=prefix, **best_params)
    best_split_scores = np.asarray(
        study.best_trial.user_attrs.get("split_scores", []),
        dtype=float,
    )
    best_estimator.fit(X, y)

    return BayesianSearchResult(
        study=study,
        best_estimator_=best_estimator,
        best_params_=_prefix_params(best_params, prefix),
        best_score_=float(study.best_value),
        best_split_scores_=best_split_scores,
        search_history_=bayesian_search_history(study, scoring=scoring),
    )


def _suggest_bayesian_params(
    trial,
    *,
    n_components: int,
    alpha_bounds: tuple[float, float],
    gamma_bounds: list[tuple[float, float]],
    kernel_weight_bounds: tuple[float, float],
    kernel_weight_center,
    kernel_weight_logit_radius: float,
) -> dict:
    params = {
        "alpha": trial.suggest_float(
            "alpha", alpha_bounds[0], alpha_bounds[1], log=True
        ),
        "gammas": [
            trial.suggest_float(
                f"gamma_{i}", gamma_bounds[i][0], gamma_bounds[i][1], log=True
            )
            for i in range(n_components)
        ],
    }

    if n_components == 1:
        params["kernel_weights"] = [1.0]
    else:
        _validate_kernel_weight_bounds(kernel_weight_bounds)
        center_logits = _center_logits(kernel_weight_center, size=n_components)
        logits = [
            trial.suggest_float(
                f"kernel_weight_logit_{i}",
                center_logits[i] - kernel_weight_logit_radius,
                center_logits[i] + kernel_weight_logit_radius,
            )
            for i in range(n_components)
        ]
        params["kernel_weights"] = _softmax(logits).tolist()

    return params


def _params_from_bayesian_trial(
    trial_params: dict,
    *,
    n_components: int,
    kernel_weight_center=None,
) -> dict:
    params = {
        "alpha": trial_params["alpha"],
        "gammas": [trial_params[f"gamma_{i}"] for i in range(n_components)],
    }

    if n_components == 1:
        params["kernel_weights"] = [1.0]
    elif all(f"kernel_weight_logit_{i}" in trial_params for i in range(n_components)):
        logits = [trial_params[f"kernel_weight_logit_{i}"] for i in range(n_components)]
        params["kernel_weights"] = _softmax(logits).tolist()
    else:
        params["kernel_weights"] = [
            trial_params[f"kernel_weight_{i}"] for i in range(n_components)
        ]

    return params


def _enqueue_initial_bayesian_trial(
    study,
    params: dict,
    *,
    n_components: int,
) -> None:
    trial_params = {
        "alpha": params["alpha"],
        **{f"gamma_{i}": params["gammas"][i] for i in range(n_components)},
    }

    if n_components > 1:
        logits = _center_logits(params["kernel_weights"], size=n_components)
        trial_params.update(
            {
                f"kernel_weight_logit_{i}": logits[i]
                for i in range(n_components)
            }
        )

    study.enqueue_trial(trial_params)


def _clone_with_params(estimator, *, prefix: str, **params):
    fixed_params = {
        f"{prefix}{name}": value
        for name, value in params.items()
        if value is not None
    }
    cloned = clone(estimator)
    if fixed_params:
        cloned.set_params(**fixed_params)
    return cloned


def _unprefix_params(params: dict, prefix: str) -> dict:
    if not prefix:
        return dict(params)

    return {
        key.removeprefix(prefix): value
        for key, value in params.items()
        if key.startswith(prefix)
    }


def _prefix_params(params: dict, prefix: str) -> dict:
    return {f"{prefix}{key}": value for key, value in params.items()}


def _as_component_bounds(
    bounds: tuple[float, float] | list[tuple[float, float]],
    n_components: int,
) -> list[tuple[float, float]]:
    if _looks_like_single_bounds(bounds):
        low, high = bounds
        return [_validate_log_bounds("gamma_bounds", (low, high))] * n_components

    component_bounds = list(bounds)
    if len(component_bounds) != n_components:
        raise ValueError(
            f"Expected {n_components} component gamma bounds, "
            f"got {len(component_bounds)}."
        )

    return [
        _validate_log_bounds("gamma_bounds", tuple(component_bounds[index]))
        for index in range(n_components)
    ]


def _looks_like_single_bounds(bounds) -> bool:
    if len(bounds) != 2:
        return False
    return np.isscalar(bounds[0]) and np.isscalar(bounds[1])


def _normalize_weights(
    weights,
    *,
    size: int,
    min_value: float = 1e-12,
) -> np.ndarray:
    weights = np.asarray(weights, dtype=float)
    if weights.shape != (size,):
        raise ValueError(f"Expected {size} weights, got shape {weights.shape}.")
    weights = np.maximum(weights, min_value)
    weight_sum = float(np.sum(weights))
    if weight_sum <= 0 or not np.isfinite(weight_sum):
        return np.full(size, 1.0 / size, dtype=float)

    return weights / weight_sum


def _center_logits(weights, *, size: int) -> np.ndarray:
    if weights is None:
        return np.zeros(size, dtype=float)

    weights = _normalize_weights(weights, size=size)
    logits = np.log(weights)
    return logits - np.mean(logits)


def _softmax(logits) -> np.ndarray:
    logits = np.asarray(logits, dtype=float)
    logits = logits - np.max(logits)
    exp_logits = np.exp(logits)
    return exp_logits / np.sum(exp_logits)


def _validate_log_bounds(name: str, bounds: tuple[float, float]) -> tuple[float, float]:
    low, high = bounds
    low = float(low)
    high = float(high)
    if low <= 0 or high <= low:
        raise ValueError(f"Expected 0 < low < high for {name}, got {bounds}.")

    return low, high


def _validate_kernel_weight_bounds(bounds: tuple[float, float]) -> tuple[float, float]:
    low, high = bounds
    low = float(low)
    high = float(high)
    if low < 0 or high <= low:
        raise ValueError(
            "Expected 0 <= low < high for kernel_weight_bounds, "
            f"got {bounds}."
        )

    return low, high


def _progress_milestones(n_trials: int) -> set[int]:
    return {
        int(np.ceil(n_trials * fraction / 10))
        for fraction in range(1, 11)
    }
