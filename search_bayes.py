from dataclasses import dataclass
import gc
from time import monotonic, perf_counter
from search_evidence import MeaningfulProgress, json_safe

import logging
import numpy as np
from sklearn.base import clone
from sklearn.model_selection import cross_val_score
from cached_scoring import (
    normalize_cached_scoring_backend,
    resolve_candidate_hyperparameter_arrays,
    score_candidates_from_cache,
    score_estimator_params_from_cache,
)
from kernel_cache import extract_regressor
from kernel_mixing import kernel_term_count
from nystrom_cache import is_nystrom_distance_cache, nystrom_cache_to_pytorch
from postprocess import bayesian_search_history

logger = logging.getLogger("search")

try:
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
except ImportError:
    optuna = None


@dataclass(frozen=True)
class BayesianSearchResult:
    study: object
    best_estimator_: object | None
    best_params_: dict
    best_score_: float
    best_split_scores_: np.ndarray
    search_history_: list[dict]
    estimator: object | None = None
    round_results_: tuple = ()


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
    refit: bool = True,
    distance_cache=None,
    kernel_weight_center=None,
    kernel_weight_logit_radius: float = 1.5,
    sampler_options: dict | None = None,
    min_delta_absolute: float = 0.0,
    min_delta_relative: float = 0.0,
    journal=None,
    observations=None,
    cached_scoring_backend: str = "numpy",
    pytorch_device: str | None = "auto",
    pytorch_devices=None,
    pytorch_candidate_batch_size: int = 1,
    bayesian_batch_size: int = 1,
) -> BayesianSearchResult:
    if optuna is None:
        raise ImportError("Optuna is required for the optional Bayesian stage.")
    if n_trials <= 0:
        raise ValueError(f"n_trials must be positive, got {n_trials}.")
    if n_components <= 0:
        raise ValueError(f"n_components must be positive, got {n_components}.")
    if patience is not None and patience <= 0:
        raise ValueError(f"patience must be positive when set, got {patience}.")
    if type(bayesian_batch_size) is not int or bayesian_batch_size < 0:
        raise ValueError("bayesian_batch_size must be a non-negative int.")

    n_terms = kernel_term_count(extract_regressor(estimator), n_components)
    _validate_log_bounds("alpha_bounds", alpha_bounds)
    component_gamma_bounds = _as_component_bounds(gamma_bounds, n_components)
    _validate_kernel_weight_bounds(kernel_weight_bounds)
    if kernel_weight_logit_radius <= 0:
        raise ValueError(
            "kernel_weight_logit_radius must be positive, "
            f"got {kernel_weight_logit_radius}."
        )
    distance_cache = _prepare_bayesian_distance_cache(
        distance_cache,
        cached_scoring_backend=cached_scoring_backend,
        pytorch_device=pytorch_device,
        pytorch_devices=pytorch_devices,
    )
    sampler_seed = random_state if isinstance(random_state, int) else None
    from config_schema import TPESamplerConfig
    sampler_options = TPESamplerConfig.model_validate(sampler_options or {}).model_dump()
    progress = MeaningfulProgress(min_delta_absolute, min_delta_relative)
    sampler = optuna.samplers.TPESampler(seed=sampler_seed, **sampler_options)
    study = optuna.create_study(direction="maximize", sampler=sampler)

    unprefixed_initial_params = _unprefix_params(initial_params, prefix)
    if kernel_weight_center is None and n_terms > 1:
        kernel_weight_center = unprefixed_initial_params["kernel_weights"]
    kernel_weight_center = (
        None
        if kernel_weight_center is None
        else _normalize_weights(kernel_weight_center, size=n_terms)
    )
    imported = _import_same_run_observations(
        study, observations or [], n_components=n_components, n_terms=n_terms,
        alpha_bounds=alpha_bounds, gamma_bounds=component_gamma_bounds,
        kernel_weight_center=kernel_weight_center,
        kernel_weight_logit_radius=kernel_weight_logit_radius,
    )
    imported_count = imported['accepted_count']
    if journal is not None and observations is not None:
        journal.append({'event':'observations_imported','stage':stage_name,**imported})
    _enqueue_initial_bayesian_trial(
        study,
        unprefixed_initial_params,
        n_components=n_components,
        n_terms=n_terms,
        kernel_weight_center=kernel_weight_center,
        kernel_weight_logit_radius=kernel_weight_logit_radius,
    )

    stage_recorder = None if journal is None else journal.stage(
        stage_name, requested_trials=n_trials,
        bounds={"alpha": alpha_bounds, "gammas": component_gamma_bounds,
                "weight_logit_center": _center_logits(kernel_weight_center, size=n_terms).tolist(),
                "weight_logit_radius": kernel_weight_logit_radius,
                "weight_logit_bounds": [[float(c - kernel_weight_logit_radius), float(c + kernel_weight_logit_radius)]
                                        for c in _center_logits(kernel_weight_center, size=n_terms)] if n_terms > 1 else [],
                "sampler": sampler_options, "sampler_seed": sampler_seed,
                "reuse_observations": observations is not None,
                "imported_observations": imported_count,
                "excluded_observations": imported['excluded_counts']})

    def record_completed_trial(study, trial):
        meaningful = progress.observe(trial.number, trial.value)
        if stage_recorder is not None:
            stage_recorder.trial(
                _params_from_bayesian_trial(trial.params, n_components=n_components, n_terms=n_terms),
                trial.user_attrs.get("split_scores", []), stage_trial=trial.number + 1 - imported_count,
                timing=trial.user_attrs.get("timing", {}),
                sampler_params=trial.params, meaningful_improvement=meaningful)

    def evaluate_trial(trial):
        started = perf_counter()
        params = _suggest_bayesian_params(
            trial,
            n_components=n_components,
            n_terms=n_terms,
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
            scores = score_estimator_params_from_cache(
                estimator,
                _prefix_params(params, prefix),
                distance_cache,
                scoring=scoring,
                backend=cached_scoring_backend,
                n_jobs=n_jobs,
                blas_threads=blas_threads,
                pytorch_device=pytorch_device,
                pytorch_devices=pytorch_devices,
                pytorch_dtype=_cache_distance_dtype(distance_cache),
                pytorch_candidate_batch_size=pytorch_candidate_batch_size,
            )
        scores = np.asarray(scores, dtype=float)
        trial.set_user_attr("split_scores", scores.tolist())
        trial.set_user_attr("timing", {"scope": "trial", "evaluation_seconds": perf_counter() - started})
        if not np.all(np.isfinite(scores)):
            return -np.inf
        return float(np.mean(scores))

    def record_failed_trial(trial, error, timing):
        if stage_recorder is not None:
            try:
                params = _params_from_bayesian_trial(trial.params, n_components=n_components, n_terms=n_terms)
            except KeyError:
                params = dict(trial.params)
            stage_recorder.trial(params, [], stage_trial=trial.number + 1 - imported_count, timing=timing,
                                 sampler_params=trial.params, status="failed",
                                 error={"type": type(error).__name__, "message": str(error)})

    def objective(trial):
        started = perf_counter()
        try:
            return evaluate_trial(trial)
        except Exception as exc:
            record_failed_trial(trial, exc, {"scope": "trial", "evaluation_seconds": perf_counter() - started})
            raise

    milestones = _progress_milestones(n_trials)

    def log_progress(study, trial):
        completed = len(study.trials) - imported_count
        if completed in milestones:
            logger.info(
                f"{stage_name} progress: "
                f"{completed}/{n_trials} ({completed / n_trials:.0%})"
            )

    def stop_after_patience(study, trial):
        if patience is None or trial.value is None:
            return

        trials_since_improvement = trial.number - progress.last_trial
        if trials_since_improvement >= patience:
            logger.warning(
                f"{stage_name} stopped after {trials_since_improvement} "
                "trials without improvement "
                f"(last meaningful improvement trial {progress.last_trial + 1 - imported_count})."
            )
            study.stop()

    if bayesian_batch_size <= 1:
        study.optimize(
            objective,
            n_trials=n_trials,
            timeout=timeout,
            callbacks=[record_completed_trial, log_progress, stop_after_patience],
        )
    else:
        _optimize_bayesian_search_batched(
            study,
            estimator,
            X,
            y,
            n_components=n_components,
            n_terms=n_terms,
            alpha_bounds=alpha_bounds,
            gamma_bounds=component_gamma_bounds,
            kernel_weight_bounds=kernel_weight_bounds,
            kernel_weight_center=kernel_weight_center,
            kernel_weight_logit_radius=kernel_weight_logit_radius,
            scoring=scoring,
            cv=cv,
            n_jobs=n_jobs,
            blas_threads=blas_threads,
            timeout=timeout,
            patience=patience,
            n_trials=n_trials,
            prefix=prefix,
            stage_name=stage_name,
            distance_cache=distance_cache,
            cached_scoring_backend=cached_scoring_backend,
            pytorch_device=pytorch_device,
            pytorch_devices=pytorch_devices,
            pytorch_candidate_batch_size=pytorch_candidate_batch_size,
            bayesian_batch_size=bayesian_batch_size,
            milestones=milestones,
            trial_callback=record_completed_trial,
            failure_callback=record_failed_trial,
            progress=progress,
            trial_offset=imported_count,
        )

    if stage_recorder is not None:
        reason = "trial_limit" if len(study.trials) - imported_count >= n_trials else (
            "patience" if patience is not None and study.trials[-1].number - progress.last_trial >= patience else "timeout")
        stage_recorder.finish(stop_reason=reason)

    best_params = study.best_trial.user_attrs.get('observed_params') or _params_from_bayesian_trial(
        study.best_trial.params,
        n_components=n_components,
        n_terms=n_terms,
        kernel_weight_center=kernel_weight_center,
    )
    best_split_scores = np.asarray(
        study.best_trial.user_attrs.get("split_scores", []),
        dtype=float,
    )
    distance_cache = None
    _release_cached_scoring_memory(cached_scoring_backend=cached_scoring_backend)
    best_estimator = None
    if refit:
        best_estimator = _clone_with_params(estimator, prefix=prefix, **best_params)
        best_estimator.fit(X, y)

    return BayesianSearchResult(
        study=study,
        best_estimator_=best_estimator,
        best_params_=_prefix_params(best_params, prefix),
        best_score_=float(study.best_value),
        best_split_scores_=best_split_scores,
        search_history_=bayesian_search_history(study, scoring=scoring),
        estimator=estimator,
    )


def _import_same_run_observations(
    study, observations, *, n_components, n_terms, alpha_bounds, gamma_bounds,
    kernel_weight_center, kernel_weight_logit_radius,
):
    """Only exact observed candidates inside this static sampler box are eligible."""
    distributions={'alpha':optuna.distributions.FloatDistribution(*alpha_bounds,log=True)}
    distributions.update({f'gamma_{i}':optuna.distributions.FloatDistribution(*bounds,log=True)
                          for i,bounds in enumerate(gamma_bounds)})
    if n_terms>1:
        center=_center_logits(kernel_weight_center,size=n_terms)
        distributions.update({f'kernel_weight_logit_{i}':optuna.distributions.FloatDistribution(
            float(value-kernel_weight_logit_radius),float(value+kernel_weight_logit_radius))
            for i,value in enumerate(center)})
    accepted=[];excluded={'invalid_scores':0,'invalid_params':0,'outside_bounds':0,'nonrepresentable_weights':0}
    for observation in observations:
        scores=np.asarray(observation['split_scores'],dtype=float)
        if not scores.size or not np.isfinite(scores).all():
            excluded['invalid_scores']+=1;continue
        params=observation['params'];gammas=np.asarray(params['gammas'],dtype=float)
        weights=np.asarray(params['kernel_weights'],dtype=float)
        if (gammas.shape!=(n_components,) or weights.shape!=(n_terms,) or
            not np.isfinite(gammas).all() or not np.isfinite(weights).all() or
            np.any(gammas<=0) or np.any(weights<0) or not np.isclose(weights.sum(),1.,rtol=1e-12,atol=1e-12) or
            not np.isfinite(params['alpha']) or params['alpha']<=0):
            excluded['invalid_params']+=1;continue
        if np.any(weights==0):
            excluded['nonrepresentable_weights']+=1;continue
        sampler_params={'alpha':float(params['alpha']),**{f'gamma_{i}':float(g) for i,g in enumerate(gammas)}}
        if n_terms>1:
            logits=np.log(weights);logits-=logits.mean()
            # Softmax is unchanged by a common offset. Retain any representable
            # weight vector, even if its zero-mean logits alone miss the box.
            low=max(distributions[f'kernel_weight_logit_{i}'].low-value for i,value in enumerate(logits))
            high=min(distributions[f'kernel_weight_logit_{i}'].high-value for i,value in enumerate(logits))
            if low>high:
                excluded['outside_bounds']+=1;continue
            logits+=min(max(0.,low),high)
            sampler_params.update({f'kernel_weight_logit_{i}':float(g) for i,g in enumerate(logits)})
        if any(not distribution.low<=sampler_params[key]<=distribution.high
               for key,distribution in distributions.items()):
            excluded['outside_bounds']+=1;continue
        record=json_safe(dict(params=params,split_scores=scores,mean_score=float(scores.mean()),
            sampler_params=sampler_params,source_stage=observation['source_stage'],source_trial=observation['source_trial']))
        study.add_trial(optuna.trial.create_trial(params=sampler_params,distributions=distributions,
            value=record['mean_score'],user_attrs={'reused_observation':True,'observed_params':record['params'],
            'split_scores':record['split_scores'],'source_stage':record['source_stage'],'source_trial':record['source_trial']}))
        accepted.append(record)
    return dict(accepted_count=len(accepted),excluded_counts=excluded,observations=accepted)


def _prepare_bayesian_distance_cache(
    distance_cache,
    *,
    cached_scoring_backend: str,
    pytorch_device: str | None,
    pytorch_devices,
):
    if distance_cache is None:
        return None
    if is_nystrom_distance_cache(distance_cache):
        if normalize_cached_scoring_backend(cached_scoring_backend) == "numpy":
            return distance_cache
        if distance_cache.backend == "pytorch":
            return distance_cache
        logger.info(
            "Transferring Bayesian Nyström distance cache to PyTorch device %s.",
            pytorch_device,
        )
        return nystrom_cache_to_pytorch(
            distance_cache,
            device=pytorch_device,
            devices=pytorch_devices,
            dtype=_cache_distance_dtype(distance_cache),
        )

    if normalize_cached_scoring_backend(cached_scoring_backend) == "numpy":
        return distance_cache

    from pytorch_backend import TorchDistanceCache, distance_cache_to_pytorch

    if isinstance(distance_cache, TorchDistanceCache):
        return distance_cache

    logger.info(
        "Transferring Bayesian distance cache to PyTorch device %s.",
        pytorch_device,
    )
    return distance_cache_to_pytorch(
        distance_cache,
        device=pytorch_device,
        devices=pytorch_devices,
        dtype=_cache_distance_dtype(distance_cache),
    )


def _optimize_bayesian_search_batched(
    study,
    estimator,
    X,
    y,
    *,
    n_components: int,
    n_terms: int,
    alpha_bounds: tuple[float, float],
    gamma_bounds: list[tuple[float, float]],
    kernel_weight_bounds: tuple[float, float],
    kernel_weight_center,
    kernel_weight_logit_radius: float,
    scoring: str,
    cv,
    n_jobs,
    blas_threads: int | None,
    timeout: float | None,
    patience: int | None,
    n_trials: int,
    prefix: str,
    stage_name: str,
    distance_cache,
    cached_scoring_backend: str,
    pytorch_device: str | None,
    pytorch_devices,
    pytorch_candidate_batch_size: int,
    bayesian_batch_size: int,
    milestones: set[int],
    trial_callback=None,
    failure_callback=None,
    progress=None,
    trial_offset=0,
) -> None:
    start_time = monotonic()
    completed_trials = 0

    while completed_trials < n_trials:
        if timeout is not None and monotonic() - start_time >= timeout:
            logger.warning(
                f"{stage_name} stopped after reaching timeout "
                f"({timeout:.1f} seconds)."
            )
            return

        batch_size = min(bayesian_batch_size, n_trials - completed_trials)
        trials = []
        params_batch = []
        for _ in range(batch_size):
            trial = study.ask()
            params = _suggest_bayesian_params(
                trial,
                n_components=n_components,
                n_terms=n_terms,
                alpha_bounds=alpha_bounds,
                gamma_bounds=gamma_bounds,
                kernel_weight_bounds=kernel_weight_bounds,
                kernel_weight_center=kernel_weight_center,
                kernel_weight_logit_radius=kernel_weight_logit_radius,
            )
            trials.append(trial)
            params_batch.append(params)

        scoring_started = perf_counter()
        try:
            split_scores = _score_bayesian_param_batch(
                estimator,
                X,
                y,
                params_batch,
                scoring=scoring,
                cv=cv,
                n_jobs=n_jobs,
                blas_threads=blas_threads,
                prefix=prefix,
                distance_cache=distance_cache,
                cached_scoring_backend=cached_scoring_backend,
                pytorch_device=pytorch_device,
                pytorch_devices=pytorch_devices,
                pytorch_candidate_batch_size=pytorch_candidate_batch_size,
            )
        except Exception as exc:
            if failure_callback is not None:
                for trial in trials:
                    failure_callback(trial, exc, {"scope": "failed_batch", "batch_size": len(trials),
                                                  "batch_wall_seconds": perf_counter() - scoring_started})
            raise
        batch_seconds = perf_counter() - scoring_started
        mean_scores = _mean_finite_scores(split_scores)

        stop_requested = False
        for trial, mean_score, trial_split_scores in zip(
            trials,
            mean_scores,
            split_scores,
        ):
            value = float(mean_score) if np.isfinite(mean_score) else -np.inf
            trial.set_user_attr("split_scores", trial_split_scores.tolist())
            trial.set_user_attr("timing", {"scope": "bayesian_batch", "batch_wall_seconds": batch_seconds,
                                           "batch_size": len(trials), "amortized_seconds": batch_seconds / len(trials)})

            study.tell(trial, value)
            if trial_callback is not None:
                trial_callback(study, next(t for t in study.get_trials(deepcopy=False) if t.number == trial.number))
            completed_trials += 1
            _log_batched_progress(
                stage_name,
                completed_trials,
                n_trials,
                milestones,
            )
            if not stop_requested:
                stop_requested = _batched_patience_stop_requested(
                    study,
                    patience=patience,
                    stage_name=stage_name,
                    progress=progress,
                    trial_offset=trial_offset,
                )

        if stop_requested:
            return


def _score_bayesian_param_batch(
    estimator,
    X,
    y,
    params_batch: list[dict],
    *,
    scoring: str,
    cv,
    n_jobs,
    blas_threads: int | None,
    prefix: str,
    distance_cache,
    cached_scoring_backend: str,
    pytorch_device: str | None,
    pytorch_devices,
    pytorch_candidate_batch_size: int,
) -> np.ndarray:
    if distance_cache is None:
        return np.asarray(
            [
                cross_val_score(
                    _clone_with_params(estimator, prefix=prefix, **params),
                    X,
                    y,
                    scoring=scoring,
                    cv=cv,
                    n_jobs=n_jobs,
                    error_score="raise",
                )
                for params in params_batch
            ],
            dtype=float,
        )

    prefixed_params = [_prefix_params(params, prefix) for params in params_batch]
    alphas, gammas, weights = resolve_candidate_hyperparameter_arrays(
        estimator,
        prefixed_params,
        distance_cache,
    )
    return score_candidates_from_cache(
        alphas=alphas,
        gammas=gammas,
        weights=weights,
        cache=distance_cache,
        scoring=scoring,
        backend=cached_scoring_backend,
        n_jobs=n_jobs,
        blas_threads=blas_threads,
        pytorch_device=pytorch_device,
        pytorch_devices=pytorch_devices,
        pytorch_dtype=_cache_distance_dtype(distance_cache),
        pytorch_candidate_batch_size=pytorch_candidate_batch_size,
    )


def _mean_finite_scores(split_scores: np.ndarray) -> np.ndarray:
    split_scores = np.asarray(split_scores, dtype=float)
    mean_scores = np.full(split_scores.shape[0], -np.inf, dtype=float)
    valid_rows = np.all(np.isfinite(split_scores), axis=1)
    if np.any(valid_rows):
        mean_scores[valid_rows] = np.mean(split_scores[valid_rows], axis=1)
    return mean_scores


def _log_batched_progress(
    stage_name: str,
    completed_trials: int,
    n_trials: int,
    milestones: set[int],
) -> None:
    if completed_trials in milestones:
        logger.info(
            f"{stage_name} progress: "
            f"{completed_trials}/{n_trials} ({completed_trials / n_trials:.0%})"
        )


def _batched_patience_stop_requested(
    study,
    *,
    patience: int | None,
    stage_name: str,
    progress=None,
    trial_offset=0,
) -> bool:
    if patience is None:
        return False

    completed_trials = [
        trial
        for trial in study.trials
        if trial.state == optuna.trial.TrialState.COMPLETE
    ]
    if not completed_trials:
        return False

    latest_trial = completed_trials[-1]
    if latest_trial.value is None:
        return False

    last_improvement = study.best_trial.number if progress is None else progress.last_trial
    trials_since_improvement = latest_trial.number - last_improvement
    if trials_since_improvement < patience:
        return False

    logger.warning(
        f"{stage_name} stopped after {trials_since_improvement} "
        "trials without improvement "
        f"(last meaningful improvement trial {last_improvement + 1 - trial_offset})."
    )
    return True


def _cache_distance_dtype(cache):
    distance = cache.folds[0].train_distances[0]
    dtype = getattr(distance, "dtype", None)
    if dtype is not None:
        return dtype
    return np.asarray(distance).dtype


def _release_cached_scoring_memory(*, cached_scoring_backend: str) -> None:
    gc.collect()
    if normalize_cached_scoring_backend(cached_scoring_backend) != "pytorch":
        return

    try:
        from pytorch_backend import require_torch

        torch = require_torch()
    except ImportError:
        return

    if torch.cuda.is_available():
        for device_index in range(torch.cuda.device_count()):
            with torch.cuda.device(device_index):
                torch.cuda.empty_cache()


def _suggest_bayesian_params(
    trial,
    *,
    n_components: int,
    n_terms: int | None = None,
    alpha_bounds: tuple[float, float],
    gamma_bounds: list[tuple[float, float]],
    kernel_weight_bounds: tuple[float, float],
    kernel_weight_center,
    kernel_weight_logit_radius: float,
) -> dict:
    n_terms = n_components if n_terms is None else n_terms
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

    if n_terms == 1:
        params["kernel_weights"] = [1.0]
    else:
        _validate_kernel_weight_bounds(kernel_weight_bounds)
        center_logits = _center_logits(kernel_weight_center, size=n_terms)
        logits = [
            trial.suggest_float(
                f"kernel_weight_logit_{i}",
                center_logits[i] - kernel_weight_logit_radius,
                center_logits[i] + kernel_weight_logit_radius,
            )
            for i in range(n_terms)
        ]
        params["kernel_weights"] = _softmax(logits).tolist()

    return params


def _params_from_bayesian_trial(
    trial_params: dict,
    *,
    n_components: int,
    n_terms: int | None = None,
    kernel_weight_center=None,
) -> dict:
    n_terms = n_components if n_terms is None else n_terms
    params = {
        "alpha": trial_params["alpha"],
        "gammas": [trial_params[f"gamma_{i}"] for i in range(n_components)],
    }

    if n_terms == 1:
        params["kernel_weights"] = [1.0]
    elif all(f"kernel_weight_logit_{i}" in trial_params for i in range(n_terms)):
        logits = [trial_params[f"kernel_weight_logit_{i}"] for i in range(n_terms)]
        params["kernel_weights"] = _softmax(logits).tolist()
    else:
        params["kernel_weights"] = [
            trial_params[f"kernel_weight_{i}"] for i in range(n_terms)
        ]

    return params


def _enqueue_initial_bayesian_trial(
    study,
    params: dict,
    *,
    n_components: int,
    n_terms: int | None = None,
    kernel_weight_center=None,
    kernel_weight_logit_radius=None,
) -> None:
    n_terms = n_components if n_terms is None else n_terms
    trial_params = {
        "alpha": params["alpha"],
        **{f"gamma_{i}": params["gammas"][i] for i in range(n_components)},
    }

    if n_terms > 1:
        logits = _center_logits(params["kernel_weights"], size=n_terms)
        if kernel_weight_center is not None and kernel_weight_logit_radius is not None:
            center=_center_logits(kernel_weight_center,size=n_terms)
            if not np.allclose(logits,center,rtol=0.,atol=1e-12):
                weights=np.asarray(params['kernel_weights'],dtype=float)
                if np.any(weights<=0):
                    raise ValueError('A fixed-box incumbent must have positive representable weights.')
                logits=np.log(weights);logits-=logits.mean()
                low=float(np.max(center-kernel_weight_logit_radius-logits))
                high=float(np.min(center+kernel_weight_logit_radius-logits))
                if low>high:
                    raise ValueError('Initial kernel weights cannot be represented in the fixed logit box.')
                logits+=min(max(0.,low),high)
        trial_params.update(
            {
                f"kernel_weight_logit_{i}": logits[i]
                for i in range(n_terms)
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
