from dataclasses import dataclass

import logging
import numpy as np
from sklearn.base import clone
from sklearn.model_selection import cross_val_score
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
    search_history_: list[dict]


def fit_bayesian_search(
    estimator,
    X,
    y,
    *,
    n_components: int,
    alpha_bounds: tuple[float, float],
    gamma_bounds: tuple[float, float],
    kernel_weight_bounds: tuple[float, float],
    initial_params: dict,
    scoring: str,
    cv,
    random_state=None,
    n_jobs=None,
    timeout: float | None = None,
    patience: int | None = None,
    n_trials: int,
    prefix: str,
    stage_name: str = "Bayesian search",
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
    _validate_log_bounds("gamma_bounds", gamma_bounds)
    _validate_kernel_weight_bounds(kernel_weight_bounds)

    sampler_seed = random_state if isinstance(random_state, int) else None
    sampler = optuna.samplers.TPESampler(seed=sampler_seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)

    unprefixed_initial_params = _unprefix_params(initial_params, prefix)
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
            gamma_bounds=gamma_bounds,
            kernel_weight_bounds=kernel_weight_bounds,
        )
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
    )
    best_estimator = _clone_with_params(estimator, prefix=prefix, **best_params)
    best_estimator.fit(X, y)

    return BayesianSearchResult(
        study=study,
        best_estimator_=best_estimator,
        best_params_=_prefix_params(best_params, prefix),
        best_score_=float(study.best_value),
        search_history_=bayesian_search_history(study, scoring=scoring),
    )


def _suggest_bayesian_params(
    trial,
    *,
    n_components: int,
    alpha_bounds: tuple[float, float],
    gamma_bounds: tuple[float, float],
    kernel_weight_bounds: tuple[float, float],
) -> dict:
    params = {
        "alpha": trial.suggest_float(
            "alpha", alpha_bounds[0], alpha_bounds[1], log=True
        ),
        "gammas": [
            trial.suggest_float(
                f"gamma_{i}", gamma_bounds[0], gamma_bounds[1], log=True
            )
            for i in range(n_components)
        ],
    }

    if n_components == 1:
        params["kernel_weights"] = [1.0]
    else:
        params["kernel_weights"] = [
            trial.suggest_float(
                f"kernel_weight_{i}",
                kernel_weight_bounds[0],
                kernel_weight_bounds[1],
            )
            for i in range(n_components)
        ]

    return params


def _params_from_bayesian_trial(trial_params: dict, *, n_components: int) -> dict:
    params = {
        "alpha": trial_params["alpha"],
        "gammas": [trial_params[f"gamma_{i}"] for i in range(n_components)],
    }

    if n_components == 1:
        params["kernel_weights"] = [1.0]
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
        trial_params.update(
            {
                f"kernel_weight_{i}": params["kernel_weights"][i]
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


def _validate_log_bounds(name: str, bounds: tuple[float, float]) -> None:
    low, high = bounds
    if low <= 0 or high <= low:
        raise ValueError(f"Expected 0 < low < high for {name}, got {bounds}.")


def _validate_kernel_weight_bounds(bounds: tuple[float, float]) -> None:
    low, high = bounds
    if low < 0 or high <= low:
        raise ValueError(
            "Expected 0 <= low < high for kernel_weight_bounds, "
            f"got {bounds}."
        )


def _progress_milestones(n_trials: int) -> set[int]:
    return {
        int(np.ceil(n_trials * fraction / 10))
        for fraction in range(1, 11)
    }
