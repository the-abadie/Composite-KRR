from dataclasses import dataclass
from contextlib import nullcontext
from time import perf_counter

import numpy as np
import logging
from joblib import Parallel, delayed
from numpy.linalg import LinAlgError
from threadpoolctl import threadpool_limits
from utilities import configure_logging, time_dif
from config import VERBOSITY
from scipy.stats import loguniform
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.model_selection import ParameterSampler, RandomizedSearchCV
from sklearn.utils import check_random_state
from sklearn.utils.validation import check_is_fitted

from class_CompositeKRR import CompositeKRR, KernelComponent
from kernel_cache import (
    UnsupportedDistanceKernelError,
    build_distance_cache,
    distance_spec_for_kernel,
    extract_regressor,
    extract_target_transformer,
    resolve_kernel_hyperparameters,
    resolve_sequence,
    score_fold_from_distances,
    unpack_sample_matrix,
)
from kernels import pairwise_self_lp_distance
from postprocess import (
    attach_random_search_history,
    combined_random_search_history,
    log_random_search_improvements,
    offset_bayesian_search_history,
)
from preprocess import make_data_preprocessor
from search_bayes import BayesianSearchResult, fit_bayesian_search

configure_logging(VERBOSITY)
logger = logging.getLogger("search")

time_log = logging.getLogger("timing")

class LogUniformListBounds:
    def __init__(self, bounds: list[tuple[float, float]]):
        if not bounds:
            raise ValueError("bounds must contain at least one component.")
        for low, high in bounds:
            if low <= 0 or high <= low:
                raise ValueError(
                    "Expected 0 < low < high for log-uniform bounds, "
                    f"got {(low, high)}."
                )

        self.bounds = [(float(low), float(high)) for low, high in bounds]

    def rvs(self, random_state=None):
        rng = check_random_state(random_state)
        return [
            float(loguniform(low, high).rvs(random_state=rng))
            for low, high in self.bounds
        ]


class SimplexWeightDistribution:
    def __init__(
        self,
        size: int,
        *,
        center: list[float] | np.ndarray | None = None,
        concentration: float | None = None,
        min_alpha: float = 1e-3,
    ):
        if size <= 0:
            raise ValueError(f"size must be positive, got {size}.")

        self.size = size
        if center is None:
            alpha = np.ones(size, dtype=float)
        else:
            center = _normalize_weights(center, size=size)
            if concentration is None or concentration <= 0:
                raise ValueError(
                    "concentration must be positive when center is provided."
                )
            alpha = center * float(concentration)

        self.alpha = np.maximum(alpha, min_alpha)

    def rvs(self, random_state=None):
        rng = check_random_state(random_state)
        return rng.dirichlet(self.alpha).tolist()


class CachedRandomSearchCV:
    def __init__(
        self,
        *,
        estimator,
        param_distributions: dict,
        n_iter: int,
        scoring,
        cv,
        random_state=None,
        n_jobs=None,
        blas_threads: int | None = 1,
        refit=True,
        distance_cache,
    ):
        self.estimator = estimator
        self.param_distributions = param_distributions
        self.n_iter = n_iter
        self.scoring = scoring
        self.cv = cv
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.blas_threads = blas_threads
        self.refit = refit
        self.distance_cache = distance_cache

    def fit(self, X, y):
        candidates = list(
            ParameterSampler(
                self.param_distributions,
                n_iter=self.n_iter,
                random_state=self.random_state,
            )
        )
        if not candidates:
            raise ValueError("ParameterSampler produced no candidates.")

        if self.n_jobs in (None, 1):
            split_scores = [
                _score_cached_candidate(
                    self.estimator,
                    params,
                    self.distance_cache,
                    self.scoring,
                )
                for params in candidates
            ]
        else:
            candidate_hyperparameters = [
                _resolve_cached_candidate_hyperparameters(
                    self.estimator,
                    params,
                    self.distance_cache,
                )
                for params in candidates
            ]
            with _threadpool_limits_for_blas(self.blas_threads):
                scored_folds = Parallel(n_jobs=self.n_jobs, prefer="threads")(
                    delayed(_score_cached_candidate_fold)(
                        candidate_index,
                        fold_index,
                        fold,
                        alpha,
                        gammas,
                        weights,
                        self.distance_cache.kernel_types,
                        self.scoring,
                    )
                    for candidate_index, (alpha, gammas, weights) in enumerate(
                        candidate_hyperparameters
                    )
                    for fold_index, fold in enumerate(self.distance_cache.folds)
                )
            split_scores = np.empty(
                (len(candidates), len(self.distance_cache.folds)),
                dtype=float,
            )
            for candidate_index, fold_index, score in scored_folds:
                split_scores[candidate_index, fold_index] = score

        split_scores = np.asarray(split_scores, dtype=float)
        failed_fold_mask = ~np.isfinite(split_scores)
        valid_candidates = ~np.any(failed_fold_mask, axis=1)
        mean_scores = np.full(len(candidates), -np.inf, dtype=float)
        std_scores = np.full(len(candidates), np.nan, dtype=float)
        if np.any(valid_candidates):
            mean_scores[valid_candidates] = np.mean(
                split_scores[valid_candidates],
                axis=1,
            )
            std_scores[valid_candidates] = np.std(
                split_scores[valid_candidates],
                axis=1,
            )
        else:
            raise ValueError(
                "All cached random-search candidates failed during CV scoring. "
                "Try increasing `KRR_ALPHA_BOUNDS`, using `KRR_COMPUTE_DTYPE = "
                "\"float64\"`, or disabling the distance cache."
            )
        best_index = int(np.argmax(mean_scores))

        ranks = np.empty(len(mean_scores), dtype=int)
        ranks[np.argsort(-mean_scores)] = np.arange(1, len(mean_scores) + 1)

        self.cv_results_ = {
            "params": candidates,
            "mean_test_score": mean_scores,
            "std_test_score": std_scores,
            "rank_test_score": ranks,
        }
        for split_id in range(split_scores.shape[1]):
            self.cv_results_[f"split{split_id}_test_score"] = split_scores[:, split_id]

        self.best_index_ = best_index
        self.best_params_ = candidates[best_index]
        self.best_score_ = float(mean_scores[best_index])
        self.n_splits_ = split_scores.shape[1]
        self.failed_fold_scores_ = int(np.count_nonzero(failed_fold_mask))
        self.failed_candidates_ = int(np.count_nonzero(~valid_candidates))
        self.n_jobs_ = self.n_jobs
        self.blas_threads_ = self.blas_threads
        self.parallel_backend_ = (
            "serial" if self.n_jobs in (None, 1) else "threading"
        )
        self.parallel_granularity_ = (
            "candidate" if self.n_jobs in (None, 1) else "candidate_fold"
        )
        if self.failed_fold_scores_ > 0:
            logger.warning(
                "%s cached fold score(s) failed due to singular or invalid "
                "kernel systems; %s candidate(s) were excluded from selection.",
                self.failed_fold_scores_,
                self.failed_candidates_,
            )

        if self.refit:
            self.best_estimator_ = clone(self.estimator)
            self.best_estimator_.set_params(**self.best_params_)
            self.best_estimator_.fit(X, y)

        return self


def _score_cached_candidate(estimator, params, distance_cache, scoring):
    alpha, gammas, weights = _resolve_cached_candidate_hyperparameters(
        estimator,
        params,
        distance_cache,
    )
    scores = [
        _safe_score_cached_fold(
            fold,
            alpha=alpha,
            gammas=gammas,
            weights=weights,
            kernel_types=distance_cache.kernel_types,
            scoring=scoring,
        )
        for fold in distance_cache.folds
    ]
    return np.asarray(scores, dtype=float)


def _resolve_cached_candidate_hyperparameters(estimator, params, distance_cache):
    candidate = clone(estimator)
    if params:
        candidate.set_params(**params)

    regressor = extract_regressor(candidate)
    return resolve_kernel_hyperparameters(
        regressor,
        n_components=distance_cache.n_components,
    )


def _score_cached_candidate_fold(
    candidate_index,
    fold_index,
    fold,
    alpha,
    gammas,
    weights,
    kernel_types,
    scoring,
):
    score = _safe_score_cached_fold(
        fold,
        alpha=alpha,
        gammas=gammas,
        weights=weights,
        kernel_types=kernel_types,
        scoring=scoring,
    )
    return candidate_index, fold_index, score


def _safe_score_cached_fold(
    fold,
    *,
    alpha,
    gammas,
    weights,
    kernel_types,
    scoring,
):
    try:
        return score_fold_from_distances(
            fold,
            alpha=alpha,
            gammas=gammas,
            weights=weights,
            kernel_types=kernel_types,
            scoring=scoring,
        )
    except LinAlgError:
        return np.nan


def _threadpool_limits_for_blas(blas_threads: int | None):
    if blas_threads is None:
        return nullcontext()

    return threadpool_limits(limits=blas_threads)


@dataclass(frozen=True)
class StagedRandomSearchResult:
    stage1: object
    stage2: object
    stage3: object | None
    final_params: dict
    timings: list[tuple[float, float, str]]
    bayesian_stage: BayesianSearchResult | None = None

    @property
    def stages(self) -> tuple[object, ...]:
        return tuple(
            stage
            for stage in (self.stage1, self.stage2, self.stage3)
            if stage is not None
        )

    @property
    def final_stage(self):
        return _best_search_stage(
            [
                stage
                for stage in (*self.stages, self.bayesian_stage)
                if stage is not None
            ]
        )

    @property
    def best_estimator_(self):
        return self.final_stage.best_estimator_

    @property
    def best_params_(self) -> dict:
        return self.final_params

    @property
    def best_score_(self) -> float:
        return float(self.final_stage.best_score_)

    @property
    def random_search_history_(self) -> list[dict]:
        return combined_random_search_history(self.stages)

    @property
    def random_search_improvements_(self) -> list[dict]:
        return [
            record
            for record in self.random_search_history_
            if record["improved"]
        ]

    @property
    def bayesian_search_history_(self) -> list[dict]:
        if self.bayesian_stage is None:
            return []

        random_history = self.random_search_history_
        initial_best_score = -np.inf
        initial_best_validation_error = np.inf
        if random_history:
            initial_best_score = random_history[-1]["best_mean_test_score"]
            initial_best_validation_error = random_history[-1][
                "best_validation_error"
            ]

        return offset_bayesian_search_history(
            self.bayesian_stage.search_history_,
            iteration_offset=len(random_history),
            initial_best_score=initial_best_score,
            initial_best_validation_error=initial_best_validation_error,
        )


class CompositeKRREstimator(BaseEstimator, RegressorMixin):
    def __init__(
        self,
        *,
        alpha=1.0,
        gammas=None,
        kernel_weights=None,
        names=None,
        kernel_types=None,
        normalizations=None,
        pca_components=None,
        pca_whiten=False,
        normalize_kernel_weights=False,
        compute_dtype,
    ):
        self.alpha = alpha
        self.gammas = gammas
        self.kernel_weights = kernel_weights
        self.names = names
        self.kernel_types = kernel_types
        self.normalizations = normalizations
        self.pca_components = pca_components
        self.pca_whiten = pca_whiten
        self.normalize_kernel_weights = normalize_kernel_weights
        self.compute_dtype = compute_dtype

    def fit(self, X, y):
        compute_dtype = np.dtype(self.compute_dtype)
        X_blocks = self._unpack_sample_matrix(X)
        y = np.asarray(y, dtype=compute_dtype).reshape(-1)

        n_blocks = len(X_blocks)
        if y.shape[0] != X_blocks[0].shape[0]:
            raise ValueError(
                f"X has {X_blocks[0].shape[0]} samples, but y has {y.shape[0]}."
            )

        names = self._resolve_sequence("names", self.names, n_blocks, "desc")
        kernel_types = self._resolve_sequence(
            "kernel_types", self.kernel_types, n_blocks, "rbf"
        )
        normalizations = self._resolve_sequence(
            "normalizations", self.normalizations, n_blocks, "none"
        )
        pca_components = self._resolve_sequence(
            "pca_components", self.pca_components, n_blocks, None
        )
        pca_whiten = self._resolve_sequence(
            "pca_whiten", self.pca_whiten, n_blocks, False
        )
        gammas = self._resolve_sequence("gammas", self.gammas, n_blocks, 1.0)
        weights = self._resolve_sequence(
            "kernel_weights", self.kernel_weights, n_blocks, 1.0
        )

        weights = np.asarray(weights, dtype=float)
        if self.normalize_kernel_weights:
            if weights.sum() <= 0:
                raise ValueError(
                    "Cannot normalize kernel weights with non-positive sum."
                )
            weights = weights / weights.sum()

        self.X_preprocessors_ = []
        X_blocks_t = []

        for X_block, normalization, pca_component, pca_whiten_block in zip(
            X_blocks,
            normalizations,
            pca_components,
            pca_whiten,
        ):
            preprocessor = make_data_preprocessor(
                normalization,
                pca_components=pca_component,
                pca_whiten=pca_whiten_block,
            )
            X_block_t = np.asarray(
                preprocessor.fit_transform(X_block),
                dtype=compute_dtype,
            )

            self.X_preprocessors_.append(preprocessor)
            X_blocks_t.append(X_block_t)

        components = [
            KernelComponent(
                name=name,
                gamma=gamma,
                kernel_weight=weight,
                kernel_type=kernel_type,
            )
            for name, gamma, weight, kernel_type in zip(
                names, gammas, weights, kernel_types
            )
        ]

        self.model_ = CompositeKRR(
            components=components,
            alpha=self.alpha,
            dtype=compute_dtype,
        )
        self.model_.fit(X_blocks_t, y)
        self.n_features_in_ = n_blocks
        self.names_ = names
        self.kernel_types_ = kernel_types
        self.normalizations_ = normalizations
        self.pca_components_ = pca_components
        self.pca_whiten_ = pca_whiten
        self.gammas_ = list(np.asarray(gammas, dtype=float))
        self.kernel_weights_ = list(weights)
        self.compute_dtype_ = compute_dtype

        return self

    def predict(self, X):
        check_is_fitted(self, "model_")

        X_blocks = self._unpack_sample_matrix(X)
        if len(X_blocks) != len(self.X_preprocessors_):
            raise ValueError(
                f"Expected {len(self.X_preprocessors_)} descriptor blocks, "
                f"got {len(X_blocks)}."
            )

        X_blocks_t = [
            np.asarray(
                preprocessor.transform(X_block),
                dtype=self.compute_dtype_,
            )
            for preprocessor, X_block in zip(self.X_preprocessors_, X_blocks)
        ]

        return self.model_.predict(X_blocks_t)

    def _unpack_sample_matrix(self, X):
        X = np.asarray(X, dtype=object)

        if X.ndim != 2:
            raise ValueError(
                "Expected X with shape (n_samples, n_descriptors), "
                f"got {X.shape}."
            )
        if X.shape[0] == 0:
            raise ValueError("X must contain at least one sample.")
        if X.shape[1] == 0:
            raise ValueError("X must contain at least one descriptor block.")

        blocks = []
        compute_dtype = np.dtype(self.compute_dtype)
        for j in range(X.shape[1]):
            block = np.stack(X[:, j]).astype(compute_dtype)
            block = block.reshape(X.shape[0], -1)
            blocks.append(block)

        return blocks

    @staticmethod
    def _resolve_sequence(name, values, n_blocks, default):
        if values is None:
            if name == "names":
                return [f"{default}{i}" for i in range(n_blocks)]
            return [default] * n_blocks

        if isinstance(values, str):
            values = [values] * n_blocks
        elif np.isscalar(values):
            values = [values] * n_blocks
        else:
            values = list(values)

        if len(values) != n_blocks:
            raise ValueError(
                f"{name} must have length {n_blocks}, got {len(values)}."
            )

        return values


def make_param_distributions(
    n_components: int,
    *,
    alpha_bounds: tuple[float, float],
    gamma_bounds: tuple[float, float] | list[tuple[float, float]],
    kernel_weight_bounds: tuple[float, float] = (0.0, 1.0),
    prefix: str = "",
    include_alpha: bool = True,
    include_gammas: bool = True,
    include_kernel_weights: bool = True,
    kernel_weight_distribution=None,
) -> dict:
    if n_components <= 0:
        raise ValueError(f"n_components must be positive, got {n_components}.")

    param_distributions = {}
    if include_alpha:
        param_distributions[f"{prefix}alpha"] = loguniform(*alpha_bounds)
    if include_gammas:
        param_distributions[f"{prefix}gammas"] = LogUniformListBounds(
            _as_component_bounds(gamma_bounds, n_components)
        )
    if include_kernel_weights and n_components > 1:
        _validate_kernel_weight_bounds(kernel_weight_bounds)
        param_distributions[f"{prefix}kernel_weights"] = (
            kernel_weight_distribution
            if kernel_weight_distribution is not None
            else SimplexWeightDistribution(n_components)
        )

    return param_distributions


def estimate_gamma_bounds(
    estimator,
    X,
    *,
    n_components: int,
    legal_gamma_bounds: tuple[float, float] | list[tuple[float, float]],
    decades: float,
    distance_cache=None,
    dtype: np.dtype | type | str = np.float64,
    block_size: int = 1024,
) -> list[tuple[float, float]]:
    component_legal_bounds = _as_component_bounds(
        legal_gamma_bounds,
        n_components,
    )
    if decades <= 0:
        raise ValueError(f"decades must be positive, got {decades}.")

    centers = (
        _gamma_centers_from_distance_cache(distance_cache, component_legal_bounds)
        if distance_cache is not None
        else _gamma_centers_from_full_data(
            estimator,
            X,
            n_components=n_components,
            legal_gamma_bounds=component_legal_bounds,
            dtype=dtype,
            block_size=block_size,
        )
    )

    bounds = [
        _narrow_log_bounds_around(center, legal_bounds, decades)
        for center, legal_bounds in zip(centers, component_legal_bounds)
    ]
    for component_index, (center, bound) in enumerate(zip(centers, bounds), start=1):
        logger.info(
            "Component %s gamma prior centered at %.6g with bounds "
            "[%.6g, %.6g].",
            component_index,
            center,
            bound[0],
            bound[1],
        )
    return bounds


def _gamma_centers_from_distance_cache(
    distance_cache,
    legal_gamma_bounds: list[tuple[float, float]],
) -> list[float]:
    centers = []
    for component_index, legal_bounds in enumerate(legal_gamma_bounds):
        medians = []
        for fold in distance_cache.folds:
            median = _positive_finite_median(fold.train_distances[component_index])
            if np.isfinite(median) and median > 0:
                medians.append(median)
        centers.append(_gamma_center_from_distance_medians(medians, legal_bounds))

    return centers


def _gamma_centers_from_full_data(
    estimator,
    X,
    *,
    n_components: int,
    legal_gamma_bounds: list[tuple[float, float]],
    dtype: np.dtype | type | str,
    block_size: int,
) -> list[float]:
    regressor = extract_regressor(estimator)
    kernel_types = resolve_sequence(
        "kernel_types",
        regressor.kernel_types,
        n_components,
        "rbf",
    )
    normalizations = resolve_sequence(
        "normalizations",
        regressor.normalizations,
        n_components,
        "none",
    )
    pca_components = resolve_sequence(
        "pca_components",
        getattr(regressor, "pca_components", None),
        n_components,
        None,
    )
    pca_whiten = resolve_sequence(
        "pca_whiten",
        getattr(regressor, "pca_whiten", False),
        n_components,
        False,
    )
    X_blocks = unpack_sample_matrix(X, dtype=dtype)
    centers = []

    for (
        X_block,
        normalization,
        pca_component,
        pca_whiten_block,
        kernel_type,
        legal_bounds,
    ) in zip(
        X_blocks,
        normalizations,
        pca_components,
        pca_whiten,
        kernel_types,
        legal_gamma_bounds,
    ):
        spec = distance_spec_for_kernel(kernel_type)
        preprocessor = make_data_preprocessor(
            normalization,
            pca_components=pca_component,
            pca_whiten=pca_whiten_block,
        )
        X_block_t = preprocessor.fit_transform(X_block)
        distances = pairwise_self_lp_distance(
            X_block_t,
            p=spec.p,
            squared=spec.squared,
            block_size=block_size,
            dtype=dtype,
        )
        median = _positive_finite_median(distances)
        centers.append(_gamma_center_from_distance_medians([median], legal_bounds))

    return centers


def _gamma_center_from_distance_medians(
    medians: list[float],
    legal_bounds: tuple[float, float],
) -> float:
    medians = np.asarray(medians, dtype=float)
    medians = medians[np.isfinite(medians) & (medians > 0)]
    if medians.size == 0:
        return _geometric_midpoint(legal_bounds)

    center = 1.0 / float(np.median(medians))
    return float(np.clip(center, legal_bounds[0], legal_bounds[1]))


def _positive_finite_median(values, *, max_values: int = 2_000_000) -> float:
    values = np.asarray(values).ravel()
    if values.size == 0:
        return np.nan

    step = max(1, int(np.ceil(values.size / max_values)))
    sample = values[::step]
    sample = sample[np.isfinite(sample) & (sample > 0)]
    if sample.size == 0:
        return np.nan

    return float(np.median(sample))


def _narrow_log_bounds_around(
    value: float,
    bounds: tuple[float, float],
    decades: float,
) -> tuple[float, float]:
    low, high = _validate_log_bounds("bounds", bounds)
    if decades <= 0:
        raise ValueError(f"decades must be positive, got {decades}.")
    if not np.isfinite(value) or value <= 0:
        value = _geometric_midpoint((low, high))

    factor = 10.0 ** decades
    new_low = max(low, float(value) / factor)
    new_high = min(high, float(value) * factor)
    if new_high <= new_low:
        return low, high

    return new_low, new_high


def _top_k_log_bounds(
    params: list[dict],
    value_getter,
    *,
    legal_bounds: tuple[float, float],
    padding_decades: float,
) -> tuple[float, float]:
    low, high = _validate_log_bounds("legal_bounds", legal_bounds)
    values = np.asarray([value_getter(param) for param in params], dtype=float)
    values = values[np.isfinite(values) & (values > 0)]
    if values.size == 0:
        return low, high
    if padding_decades <= 0:
        raise ValueError(
            f"padding_decades must be positive, got {padding_decades}."
        )

    log_values = np.log10(values)
    new_low = max(low, 10.0 ** (float(np.min(log_values)) - padding_decades))
    new_high = min(high, 10.0 ** (float(np.max(log_values)) + padding_decades))
    if new_high <= new_low:
        return low, high

    return new_low, new_high


def _top_candidate_params(
    search,
    *,
    fraction: float,
    min_candidates: int,
) -> list[dict]:
    if not 0 < fraction <= 1:
        raise ValueError(f"fraction must be in (0, 1], got {fraction}.")
    if min_candidates <= 0:
        raise ValueError(f"min_candidates must be positive, got {min_candidates}.")

    scores = np.asarray(search.cv_results_["mean_test_score"], dtype=float)
    finite_indices = np.flatnonzero(np.isfinite(scores))
    if finite_indices.size == 0:
        return [dict(search.best_params_)]

    n_top = min(
        finite_indices.size,
        max(min_candidates, int(np.ceil(finite_indices.size * fraction))),
    )
    ordered = finite_indices[np.argsort(scores[finite_indices])]
    top_indices = ordered[-n_top:]
    return [dict(search.cv_results_["params"][index]) for index in top_indices]


def _average_kernel_weights(
    params: list[dict],
    *,
    key: str,
    n_components: int,
) -> np.ndarray:
    weights = []
    for param in params:
        if key in param:
            weights.append(_normalize_weights(param[key], size=n_components))

    if not weights:
        return _uniform_weights(n_components)

    return _normalize_weights(np.mean(weights, axis=0), size=n_components)


def _as_component_bounds(
    bounds: tuple[float, float] | list[tuple[float, float]],
    n_components: int,
) -> list[tuple[float, float]]:
    if n_components <= 0:
        raise ValueError(f"n_components must be positive, got {n_components}.")

    if _looks_like_single_bounds(bounds):
        low, high = bounds
        return [_validate_log_bounds("bounds", (low, high))] * n_components

    component_bounds = list(bounds)
    if len(component_bounds) != n_components:
        raise ValueError(
            f"Expected {n_components} component bounds, "
            f"got {len(component_bounds)}."
        )

    return [
        _validate_log_bounds("bounds", tuple(component_bounds[index]))
        for index in range(n_components)
    ]


def _looks_like_single_bounds(bounds) -> bool:
    if len(bounds) != 2:
        return False
    return np.isscalar(bounds[0]) and np.isscalar(bounds[1])


def _validate_log_bounds(
    name: str,
    bounds: tuple[float, float],
) -> tuple[float, float]:
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
        return _uniform_weights(size)

    return weights / weight_sum


def _uniform_weights(size: int) -> np.ndarray:
    if size <= 0:
        raise ValueError(f"size must be positive, got {size}.")
    return np.full(size, 1.0 / size, dtype=float)


def _geometric_midpoint(bounds: tuple[float, float]) -> float:
    low, high = _validate_log_bounds("bounds", bounds)
    return float(np.sqrt(low * high))


def staged_random_search_cv(
    estimator,
    X,
    y,
    *,
    n_components: int,
    alpha_bounds: tuple[float, float],
    gamma_bounds: tuple[float, float],
    kernel_weight_bounds: tuple[float, float] = (0.0, 1.0),
    n_iter_stage1: int,
    n_iter_stage2: int,
    n_iter_stage3: int,
    scoring: str,
    cv,
    random_state=None,
    n_jobs=None,
    random_search_blas_threads: int | None = 1,
    refit=True,
    prefix: str = "",
    n_trials_bayesian: int | None = None,
    bayesian_timeout: float | None = None,
    bayesian_patience: int | None = None,
    use_distance_cache: bool = True,
    distance_block_size: int = 1024,
    distance_dtype: np.dtype | type | str = np.float64,
    distance_cache_n_jobs: int | None = -1,
    distance_cache_memory_fraction: float = 0.80,
    gamma_prior_decades: float = 2.5,
    refine_decades: float = 1.0,
    top_k_fraction: float = 0.20,
    top_k_min_candidates: int = 3,
    top_k_padding_decades: float = 0.5,
    stage3_weight_concentration: float = 25.0,
    bayesian_refine_decades: float = 0.3,
    bayesian_weight_logit_radius: float = 1.5,
) -> StagedRandomSearchResult:
    if n_components <= 0:
        raise ValueError(f"n_components must be positive, got {n_components}.")

    for stage, n_iter in {1: n_iter_stage1, 2: n_iter_stage2}.items():
        if n_iter <= 0:
            raise ValueError(f"n_iter_stage{stage} must be positive, got {n_iter}.")
    if n_iter_stage3 < 0:
        raise ValueError(f"n_iter_stage3 must be non-negative, got {n_iter_stage3}.")
    if n_trials_bayesian is not None and n_trials_bayesian < 0:
        raise ValueError(
            f"n_trials_bayesian must be non-negative, got {n_trials_bayesian}."
        )
    if bayesian_patience is not None and bayesian_patience <= 0:
        raise ValueError(
            "bayesian_patience must be positive when set, "
            f"got {bayesian_patience}."
        )

    distance_cache = None
    if use_distance_cache:
        time_cache_start: float = perf_counter()
        distance_cache = _maybe_build_distance_cache(
            estimator,
            X,
            y,
            cv,
            n_components=n_components,
            block_size=distance_block_size,
            dtype=distance_dtype,
            n_jobs=distance_cache_n_jobs,
            memory_fraction=distance_cache_memory_fraction,
        )
        time_cache_end: float = perf_counter()
        cache_timing_label = (
            "Distance Matrix Pre-Caching"
            if distance_cache is not None
            else "Distance Matrix Pre-Caching (Unavailable)"
        )
        time_log.info(
            f"Distance cache preparation completed in "
            f"{time_dif(time_cache_start, time_cache_end)}."
        )
    else:
        time_cache_start = 0.0
        time_cache_end = 0.0
        cache_timing_label = "Distance Matrix Pre-Caching (Skipped)"

    component_gamma_bounds = _as_component_bounds(gamma_bounds, n_components)
    gamma_prior_bounds = estimate_gamma_bounds(
        estimator,
        X,
        n_components=n_components,
        legal_gamma_bounds=component_gamma_bounds,
        decades=gamma_prior_decades,
        distance_cache=distance_cache,
        dtype=distance_dtype,
        block_size=distance_block_size,
    )

    stage1_estimator = _clone_with_params(
        estimator,
        prefix=prefix,
        kernel_weights=_uniform_weights(n_components).tolist(),
    )
    logger.warning(
        "Beginning Stage 1 broad alpha/gamma scale search with equal "
        f"weights. [{n_iter_stage1} iterations]"
    )
    time_stage1_start:float = perf_counter()
    stage1 = _fit_search(
        stage1_estimator,
        X,
        y,
        param_distributions=make_param_distributions(
            n_components,
            alpha_bounds=alpha_bounds,
            gamma_bounds=gamma_prior_bounds,
            prefix=prefix,
            include_kernel_weights=False,
        ),
        n_iter=n_iter_stage1,
        scoring=scoring,
        cv=cv,
        random_state=random_state,
        n_jobs=n_jobs,
        blas_threads=random_search_blas_threads,
        refit=refit,
        stage_name="Stage 1",
        distance_cache=distance_cache,
    )
    time_stage1_end:float = perf_counter()
    time_log.info(f"Stage 1 completed in {time_dif(time_stage1_start, time_stage1_end)}.")

    stage1_best_params = _complete_unprefixed_params_from_estimator(
        stage1.best_estimator_,
        n_components=n_components,
    )
    stage2_alpha_bounds = _narrow_log_bounds_around(
        stage1_best_params["alpha"],
        alpha_bounds,
        refine_decades,
    )
    stage2_gamma_bounds = [
        _narrow_log_bounds_around(gamma, legal_bounds, refine_decades)
        for gamma, legal_bounds in zip(
            stage1_best_params["gammas"],
            component_gamma_bounds,
        )
    ]

    logger.warning(
        "Beginning Stage 2 local alpha/gamma search with simplex weights. "
        f"[{n_iter_stage2} iterations]"
    )
    time_stage2_start:float = perf_counter()
    stage2 = _fit_search(
        estimator,
        X,
        y,
        param_distributions=make_param_distributions(
            n_components,
            alpha_bounds=stage2_alpha_bounds,
            gamma_bounds=stage2_gamma_bounds,
            kernel_weight_bounds=kernel_weight_bounds,
            prefix=prefix,
        ),
        n_iter=n_iter_stage2,
        scoring=scoring,
        cv=cv,
        random_state=random_state,
        n_jobs=n_jobs,
        blas_threads=random_search_blas_threads,
        refit=refit,
        stage_name="Stage 2",
        distance_cache=distance_cache,
    )
    time_stage2_end:float = perf_counter()
    time_log.info(f"Stage 2 completed in {time_dif(time_stage2_start, time_stage2_end)}.")

    if n_iter_stage3 > 0:
        stage2_top_params = _top_candidate_params(
            stage2,
            fraction=top_k_fraction,
            min_candidates=top_k_min_candidates,
        )
        stage3_alpha_bounds = _top_k_log_bounds(
            stage2_top_params,
            lambda params: params[f"{prefix}alpha"],
            legal_bounds=alpha_bounds,
            padding_decades=top_k_padding_decades,
        )
        stage3_gamma_bounds = [
            _top_k_log_bounds(
                stage2_top_params,
                lambda params, component_index=component_index: (
                    params[f"{prefix}gammas"][component_index]
                ),
                legal_bounds=component_gamma_bounds[component_index],
                padding_decades=top_k_padding_decades,
            )
            for component_index in range(n_components)
        ]
        stage3_weight_distribution = None
        if n_components > 1:
            stage3_weight_distribution = SimplexWeightDistribution(
                n_components,
                center=_average_kernel_weights(
                    stage2_top_params,
                    key=f"{prefix}kernel_weights",
                    n_components=n_components,
                ),
                concentration=stage3_weight_concentration,
            )

        logger.warning(
            "Beginning Stage 3 top-k alpha/gamma refinement with centered "
            f"simplex weights. [{n_iter_stage3} iterations]"
        )
        time_stage3_start:float = perf_counter()
        stage3 = _fit_search(
            estimator,
            X,
            y,
            param_distributions=make_param_distributions(
                n_components,
                alpha_bounds=stage3_alpha_bounds,
                gamma_bounds=stage3_gamma_bounds,
                kernel_weight_bounds=kernel_weight_bounds,
                prefix=prefix,
                kernel_weight_distribution=stage3_weight_distribution,
            ),
            n_iter=n_iter_stage3,
            scoring=scoring,
            cv=cv,
            random_state=random_state,
            n_jobs=n_jobs,
            blas_threads=random_search_blas_threads,
            refit=refit,
            stage_name="Stage 3",
            distance_cache=distance_cache,
        )
        time_stage3_end:float = perf_counter()
        time_log.info(f"Stage 3 completed in {time_dif(time_stage3_start, time_stage3_end)}.")
        stage3_timing_label = "Training: Stage III"
    else:
        logger.warning("Skipping Stage 3 because n_iter_stage3 is 0.")
        time_stage3_start = 0.0
        time_stage3_end = 0.0
        stage3 = None
        stage3_timing_label = "Training: Stage III (Skipped)"

    best_random_stage = _best_search_stage([stage1, stage2, stage3])
    final_params = _complete_prefixed_params_from_estimator(
        best_random_stage.best_estimator_,
        prefix=prefix,
        n_components=n_components,
    )

    timings = [
        (time_cache_start, time_cache_end, cache_timing_label),
        (time_stage1_start, time_stage1_end, "Training: Stage I"),
        (time_stage2_start, time_stage2_end, "Training: Stage II"),
        (time_stage3_start, time_stage3_end, stage3_timing_label),
        (0., 0., "Training: Stage IV (Skipped)"),
    ]

    random_result = StagedRandomSearchResult(
        stage1=stage1,
        stage2=stage2,
        stage3=stage3,
        final_params=final_params,
        timings=timings
    )
    log_random_search_improvements(random_result, scoring=scoring, logger=logger)

    bayesian_stage = None
    if n_trials_bayesian is not None and n_trials_bayesian > 0:
        best_random_params = _complete_unprefixed_params_from_estimator(
            best_random_stage.best_estimator_,
            n_components=n_components,
        )
        bayesian_alpha_bounds = _narrow_log_bounds_around(
            best_random_params["alpha"],
            alpha_bounds,
            bayesian_refine_decades,
        )
        bayesian_gamma_bounds = [
            _narrow_log_bounds_around(gamma, legal_bounds, bayesian_refine_decades)
            for gamma, legal_bounds in zip(
                best_random_params["gammas"],
                component_gamma_bounds,
            )
        ]

        logger.warning(
            "Now entering final stage: local Bayesian search over all "
            f"hyperparameters. [{n_trials_bayesian} iterations]"
        )
        time_bayes_start:float = perf_counter()
        bayesian_stage = fit_bayesian_search(
            estimator,
            X,
            y,
            n_components=n_components,
            alpha_bounds=bayesian_alpha_bounds,
            gamma_bounds=bayesian_gamma_bounds,
            kernel_weight_bounds=kernel_weight_bounds,
            initial_params=final_params,
            scoring=scoring,
            cv=cv,
            random_state=random_state,
            n_jobs=n_jobs,
            blas_threads=random_search_blas_threads,
            timeout=bayesian_timeout,
            patience=bayesian_patience,
            n_trials=n_trials_bayesian,
            prefix=prefix,
            stage_name="Bayesian search",
            distance_cache=distance_cache,
            kernel_weight_center=best_random_params["kernel_weights"],
            kernel_weight_logit_radius=bayesian_weight_logit_radius,
        )
        time_bayes_end:float = perf_counter()
        time_log.info(f"Final stage completed in {time_dif(time_bayes_start, time_bayes_end)}.")

        timings = [
            (time_cache_start, time_cache_end, cache_timing_label),
            (time_stage1_start, time_stage1_end, "Training: Stage I"),
            (time_stage2_start, time_stage2_end, "Training: Stage II"),
            (time_stage3_start, time_stage3_end, stage3_timing_label),
            (time_bayes_start , time_bayes_end , "Training: Stage IV"),
        ]

    selected_stage = _best_search_stage(
        [
            stage
            for stage in (stage1, stage2, stage3, bayesian_stage)
            if stage is not None
        ]
    )
    final_params = _complete_prefixed_params_from_estimator(
        selected_stage.best_estimator_,
        prefix=prefix,
        n_components=n_components,
    )
    logger.info(
        "Selected %s as final model with CV score %.6g.",
        getattr(selected_stage, "stage_name", "Bayesian search"),
        selected_stage.best_score_,
    )

    return StagedRandomSearchResult(
        stage1=stage1,
        stage2=stage2,
        stage3=stage3,
        final_params=final_params,
        bayesian_stage=bayesian_stage,
        timings=timings
    )


def _fit_search(
    estimator,
    X,
    y,
    *,
    param_distributions: dict,
    n_iter: int,
    scoring: str,
    cv,
    random_state=None,
    n_jobs=None,
    blas_threads: int | None = 1,
    refit=True,
    stage_name: str = "Random search",
    distance_cache=None,
):
    if distance_cache is None:
        return _fit_random_search(
            estimator,
            X,
            y,
            param_distributions=param_distributions,
            n_iter=n_iter,
            scoring=scoring,
            cv=cv,
            random_state=random_state,
            n_jobs=n_jobs,
            refit=refit,
            stage_name=stage_name,
        )

    return _fit_cached_random_search(
        estimator,
        X,
        y,
        param_distributions=param_distributions,
        n_iter=n_iter,
        scoring=scoring,
        cv=cv,
        random_state=random_state,
        n_jobs=n_jobs,
        blas_threads=blas_threads,
        refit=refit,
        stage_name=stage_name,
        distance_cache=distance_cache,
    )


def _fit_random_search(
    estimator,
    X,
    y,
    *,
    param_distributions: dict,
    n_iter: int,
    scoring: str,
    cv,
    random_state=None,
    n_jobs=None,
    refit=True,
    stage_name: str = "Random search",
) -> RandomizedSearchCV:
    if not param_distributions:
        raise ValueError("param_distributions must contain at least one parameter.")

    search = RandomizedSearchCV(
        estimator=estimator,
        param_distributions=param_distributions,
        n_iter=n_iter,
        scoring=scoring,
        cv=cv,
        random_state=random_state,
        n_jobs=n_jobs,
        refit=refit,
    )
    search.stage_name = stage_name
    search.fit(X, y)
    attach_random_search_history(search, scoring=scoring)
    return search


def _fit_cached_random_search(
    estimator,
    X,
    y,
    *,
    param_distributions: dict,
    n_iter: int,
    scoring: str,
    cv,
    random_state=None,
    n_jobs=None,
    blas_threads: int | None = 1,
    refit=True,
    stage_name: str = "Random search",
    distance_cache,
) -> CachedRandomSearchCV:
    if not param_distributions:
        raise ValueError("param_distributions must contain at least one parameter.")

    search = CachedRandomSearchCV(
        estimator=estimator,
        param_distributions=param_distributions,
        n_iter=n_iter,
        scoring=scoring,
        cv=cv,
        random_state=random_state,
        n_jobs=n_jobs,
        blas_threads=blas_threads,
        refit=refit,
        distance_cache=distance_cache,
    )
    search.stage_name = stage_name
    search.fit(X, y)
    attach_random_search_history(search, scoring=scoring)
    search.distance_cache = None
    return search


def _maybe_build_distance_cache(
    estimator,
    X,
    y,
    cv,
    *,
    n_components: int,
    block_size: int,
    dtype: np.dtype | type | str,
    n_jobs: int | None,
    memory_fraction: float,
):
    regressor = extract_regressor(estimator)
    names = resolve_sequence("names", regressor.names, n_components, "desc")
    kernel_types = resolve_sequence(
        "kernel_types", regressor.kernel_types, n_components, "rbf"
    )
    normalizations = resolve_sequence(
        "normalizations", regressor.normalizations, n_components, "none"
    )
    pca_components = resolve_sequence(
        "pca_components",
        getattr(regressor, "pca_components", None),
        n_components,
        None,
    )
    pca_whiten = resolve_sequence(
        "pca_whiten",
        getattr(regressor, "pca_whiten", False),
        n_components,
        False,
    )
    target_transformer = extract_target_transformer(estimator)

    try:
        return build_distance_cache(
            X,
            y,
            cv,
            names=names,
            kernel_types=kernel_types,
            normalizations=normalizations,
            pca_components=pca_components,
            pca_whiten=pca_whiten,
            target_transformer=target_transformer,
            block_size=block_size,
            dtype=np.dtype(dtype),
            n_jobs=n_jobs,
            memory_fraction=memory_fraction,
        )
    except UnsupportedDistanceKernelError as exc:
        logger.info(f"Distance cache disabled: {exc}")
        return None


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


def _best_search_stage(stages):
    if not stages:
        raise ValueError("At least one search stage is required.")

    finite_stages = [
        stage
        for stage in stages
        if np.isfinite(float(getattr(stage, "best_score_", -np.inf)))
    ]
    if not finite_stages:
        raise ValueError("No search stage has a finite best CV score.")

    return max(finite_stages, key=lambda stage: float(stage.best_score_))


def _complete_prefixed_params_from_estimator(
    estimator,
    *,
    prefix: str,
    n_components: int,
) -> dict:
    params = _complete_unprefixed_params_from_estimator(
        estimator,
        n_components=n_components,
    )
    return {f"{prefix}{name}": value for name, value in params.items()}


def _complete_unprefixed_params_from_estimator(estimator, *, n_components: int) -> dict:
    regressor = extract_regressor(estimator)
    alpha, gammas, weights = resolve_kernel_hyperparameters(
        regressor,
        n_components=n_components,
    )
    return {
        "alpha": float(alpha),
        "gammas": [float(value) for value in np.asarray(gammas, dtype=float)],
        "kernel_weights": [
            float(value)
            for value in np.asarray(weights, dtype=float)
        ],
    }
