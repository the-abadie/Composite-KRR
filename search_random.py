from dataclasses import dataclass
from time import perf_counter

import numpy as np
import logging
from utilities import configure_logging, time_dif
from config import VERBOSITY
from scipy.stats import loguniform, uniform
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.model_selection import RandomizedSearchCV
from sklearn.utils.validation import check_is_fitted

from class_CompositeKRR import CompositeKRR, KernelComponent
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

class LogUniformList:
    def __init__(self, low: float, high: float, size: int):
        if low <= 0 or high <= low:
            raise ValueError(
                f"Expected 0 < low < high for log-uniform bounds, got {low}, {high}."
            )
        if size <= 0:
            raise ValueError(f"size must be positive, got {size}.")

        self.low = low
        self.high = high
        self.size = size

    def rvs(self, random_state=None):
        return loguniform(self.low, self.high).rvs(
            size=self.size, random_state=random_state
        ).tolist()


class UniformList:
    def __init__(self, low: float, high: float, size: int):
        if high <= low:
            raise ValueError(
                f"Expected low < high for uniform bounds, got {low}, {high}."
            )
        if size <= 0:
            raise ValueError(f"size must be positive, got {size}.")

        self.low = low
        self.high = high
        self.size = size

    def rvs(self, random_state=None):
        return uniform(self.low, self.high - self.low).rvs(
            size=self.size, random_state=random_state
        ).tolist()


@dataclass(frozen=True)
class StagedRandomSearchResult:
    stage1: RandomizedSearchCV
    stage2: RandomizedSearchCV
    stage3: RandomizedSearchCV
    final_params: dict
    timings: list[tuple[float, float, str]]
    bayesian_stage: BayesianSearchResult | None = None

    @property
    def stages(self) -> tuple[RandomizedSearchCV, RandomizedSearchCV, RandomizedSearchCV]:
        return self.stage1, self.stage2, self.stage3

    @property
    def final_stage(self):
        return self.bayesian_stage if self.bayesian_stage is not None else self.stage3

    @property
    def best_estimator_(self):
        if self.bayesian_stage is not None:
            return self.bayesian_stage.best_estimator_
        return self.stage3.best_estimator_

    @property
    def best_params_(self) -> dict:
        if self.bayesian_stage is not None:
            return self.bayesian_stage.best_params_
        return self.final_params

    @property
    def best_score_(self) -> float:
        if self.bayesian_stage is not None:
            return self.bayesian_stage.best_score_
        return self.stage3.best_score_

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

        return offset_bayesian_search_history(
            self.bayesian_stage.search_history_,
            iteration_offset=len(self.random_search_history_),
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
        normalize_kernel_weights=False,
    ):
        self.alpha = alpha
        self.gammas = gammas
        self.kernel_weights = kernel_weights
        self.names = names
        self.kernel_types = kernel_types
        self.normalizations = normalizations
        self.normalize_kernel_weights = normalize_kernel_weights

    def fit(self, X, y):
        X_blocks = self._unpack_sample_matrix(X)
        y = np.asarray(y, dtype=float).reshape(-1)

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

        for X_block, normalization in zip(X_blocks, normalizations):
            preprocessor = make_data_preprocessor(normalization)
            X_block_t = preprocessor.fit_transform(X_block)

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

        self.model_ = CompositeKRR(components=components, alpha=self.alpha)
        self.model_.fit(X_blocks_t, y)
        self.n_features_in_ = n_blocks
        self.names_ = names
        self.kernel_types_ = kernel_types
        self.normalizations_ = normalizations
        self.gammas_ = list(np.asarray(gammas, dtype=float))
        self.kernel_weights_ = list(weights)

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
            preprocessor.transform(X_block)
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
        for j in range(X.shape[1]):
            block = np.stack(X[:, j]).astype(float)
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
    gamma_bounds: tuple[float, float],
    kernel_weight_bounds: tuple[float, float] = (0.0, 1.0),
    prefix: str = "",
    include_alpha: bool = True,
    include_gammas: bool = True,
    include_kernel_weights: bool = True,
) -> dict:
    if n_components <= 0:
        raise ValueError(f"n_components must be positive, got {n_components}.")

    param_distributions = {}
    if include_alpha:
        param_distributions[f"{prefix}alpha"] = loguniform(*alpha_bounds)
    if include_gammas:
        param_distributions[f"{prefix}gammas"] = LogUniformList(
            *gamma_bounds, size=n_components
        )
    if include_kernel_weights and n_components > 1:
        param_distributions[f"{prefix}kernel_weights"] = UniformList(
            *kernel_weight_bounds, size=n_components
        )

    return param_distributions


def make_stage_param_distributions(
    stage: int,
    n_components: int,
    *,
    alpha_bounds: tuple[float, float],
    gamma_bounds: tuple[float, float],
    kernel_weight_bounds: tuple[float, float] = (0.0, 1.0),
    prefix: str = "",
) -> dict:
    if stage == 1:
        return make_param_distributions(
            n_components,
            alpha_bounds=alpha_bounds,
            gamma_bounds=gamma_bounds,
            kernel_weight_bounds=kernel_weight_bounds,
            prefix=prefix,
            include_alpha=True,
            include_gammas=False,
            include_kernel_weights=True,
        )

    if stage == 2:
        return make_param_distributions(
            n_components,
            alpha_bounds=alpha_bounds,
            gamma_bounds=gamma_bounds,
            kernel_weight_bounds=kernel_weight_bounds,
            prefix=prefix,
            include_alpha=True,
            include_gammas=True,
            include_kernel_weights=False,
        )

    if stage == 3:
        return make_param_distributions(
            n_components,
            alpha_bounds=alpha_bounds,
            gamma_bounds=gamma_bounds,
            kernel_weight_bounds=kernel_weight_bounds,
            prefix=prefix,
            include_alpha=False,
            include_gammas=True,
            include_kernel_weights=True,
        )

    raise ValueError(f"stage must be 1, 2, or 3, got {stage}.")


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
    refit=True,
    prefix: str = "",
    n_trials_bayesian: int | None = None,
    bayesian_timeout: float | None = None,
    bayesian_patience: int | None = None,
) -> StagedRandomSearchResult:
    if n_components <= 0:
        raise ValueError(f"n_components must be positive, got {n_components}.")

    for stage, n_iter in {
        1: n_iter_stage1,
        2: n_iter_stage2,
        3: n_iter_stage3,
    }.items():
        if n_iter <= 0:
            raise ValueError(f"n_iter_stage{stage} must be positive, got {n_iter}.")
    if n_trials_bayesian is not None and n_trials_bayesian < 0:
        raise ValueError(
            f"n_trials_bayesian must be non-negative, got {n_trials_bayesian}."
        )
    if bayesian_patience is not None and bayesian_patience <= 0:
        raise ValueError(
            "bayesian_patience must be positive when set, "
            f"got {bayesian_patience}."
        )

    stage1_estimator = _clone_with_params(
        estimator,
        prefix=prefix,
        kernel_weights=[1.0] if n_components == 1 else None,
    )
    logger.warning(f"Beginning Stage 1 search in joint alpha/kernel_weight space. [{n_iter_stage1} iterations]")
    time_stage1_start:float = perf_counter()
    stage1 = _fit_random_search(
        stage1_estimator,
        X,
        y,
        param_distributions=make_stage_param_distributions(
            1,
            n_components,
            alpha_bounds=alpha_bounds,
            gamma_bounds=gamma_bounds,
            kernel_weight_bounds=kernel_weight_bounds,
            prefix=prefix,
        ),
        n_iter=n_iter_stage1,
        scoring=scoring,
        cv=cv,
        random_state=random_state,
        n_jobs=n_jobs,
        refit=refit,
        stage_name="Stage 1",
    )
    time_stage1_end:float = perf_counter()
    time_log.info(f"Stage 1 completed in {time_dif(time_stage1_start, time_stage1_end)}.")
    stage1_weights = (
        [1.0]
        if n_components == 1
        else _best_param(stage1, prefix, "kernel_weights")
    )

    stage2_estimator = _clone_with_params(
        estimator,
        prefix=prefix,
        kernel_weights=stage1_weights,
    )
    logger.warning(f"Beginning Stage 2 search in joint alpha/gamma space. [{n_iter_stage2} iterations]")
    time_stage2_start:float = perf_counter()
    stage2 = _fit_random_search(
        stage2_estimator,
        X,
        y,
        param_distributions=make_stage_param_distributions(
            2,
            n_components,
            alpha_bounds=alpha_bounds,
            gamma_bounds=gamma_bounds,
            kernel_weight_bounds=kernel_weight_bounds,
            prefix=prefix,
        ),
        n_iter=n_iter_stage2,
        scoring=scoring,
        cv=cv,
        random_state=random_state,
        n_jobs=n_jobs,
        refit=refit,
        stage_name="Stage 2",
    )
    time_stage2_end:float = perf_counter()
    time_log.info(f"Stage 2 completed in {time_dif(time_stage2_start, time_stage2_end)}.")

    stage3_estimator = _clone_with_params(
        estimator,
        prefix=prefix,
        alpha=_best_param(stage2, prefix, "alpha"),
        kernel_weights=[1.0] if n_components == 1 else None,
    )
    logger.warning(f"Beginning Stage 3 search in joint gamma/kernel_weight space. [{n_iter_stage3} iterations]")
    time_stage3_start:float = perf_counter()
    stage3 = _fit_random_search(
        stage3_estimator,
        X,
        y,
        param_distributions=make_stage_param_distributions(
            3,
            n_components,
            alpha_bounds=alpha_bounds,
            gamma_bounds=gamma_bounds,
            kernel_weight_bounds=kernel_weight_bounds,
            prefix=prefix,
        ),
        n_iter=n_iter_stage3,
        scoring=scoring,
        cv=cv,
        random_state=random_state,
        n_jobs=n_jobs,
        refit=refit,
        stage_name="Stage 3",
    )
    time_stage3_end:float = perf_counter()
    time_log.info(f"Stage 3 completed in {time_dif(time_stage3_start, time_stage3_end)}.")

    final_params = {
        f"{prefix}alpha": _best_param(stage2, prefix, "alpha"),
        f"{prefix}gammas": _best_param(stage3, prefix, "gammas"),
        f"{prefix}kernel_weights": (
            [1.0]
            if n_components == 1
            else _best_param(stage3, prefix, "kernel_weights")
        ),
    }

    timings = [
        (time_stage1_start, time_stage1_end, "Training: Stage I"),
        (time_stage2_start, time_stage2_end, "Training: Stage II"),
        (time_stage3_start, time_stage3_end, "Training: Stage III"),
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
        logger.warning(f"Now entering final stage: Bayesian search over all hyperparameters [{n_trials_bayesian} iterations]")
        time_bayes_start:float = perf_counter()
        bayesian_stage = fit_bayesian_search(
            estimator,
            X,
            y,
            n_components=n_components,
            alpha_bounds=alpha_bounds,
            gamma_bounds=gamma_bounds,
            kernel_weight_bounds=kernel_weight_bounds,
            initial_params=final_params,
            scoring=scoring,
            cv=cv,
            random_state=random_state,
            n_jobs=n_jobs,
            timeout=bayesian_timeout,
            patience=bayesian_patience,
            n_trials=n_trials_bayesian,
            prefix=prefix,
            stage_name="Bayesian search",
        )
        time_bayes_end:float = perf_counter()
        time_log.info(f"Final stage completed in {time_dif(time_bayes_start, time_bayes_end)}.")

        timings = [
            (time_stage1_start, time_stage1_end, "Training: Stage I"),
            (time_stage2_start, time_stage2_end, "Training: Stage II"),
            (time_stage3_start, time_stage3_end, "Training: Stage III"),
            (time_bayes_start , time_bayes_end , "Training: Stage IV"),
        ]

    return StagedRandomSearchResult(
        stage1=stage1,
        stage2=stage2,
        stage3=stage3,
        final_params=final_params,
        bayesian_stage=bayesian_stage,
        timings=timings
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


def _best_param(search: RandomizedSearchCV, prefix: str, name: str):
    return search.best_params_[f"{prefix}{name}"]
