import logging
import random

import numpy as np

import config
import preparation
import preprocess
from class_CompositeDescriptor import CompositeDescriptor
from class_Target import Target
from search_random import (
    CompositeKRREstimator,
    plot_random_search_validation_error,
    staged_random_search_cv,
)
from sklearn.compose import TransformedTargetRegressor
from sklearn.model_selection import KFold
from utilities import configure_logging

# 0) Initialize
configure_logging(config.VERBOSITY)
logger = logging.getLogger("CKRR")

random.seed(config.SEED)
np.random.seed(config.SEED)
rng = np.random.default_rng(config.SEED)

# 1) Load Descriptor
descriptors = CompositeDescriptor(
    names=config.X_NAMES, paths=config.X_PATHS, normalizations=config.X_NORMS
)
descriptors.load_descriptor_blocks_from_npy()
DESCRIPTOR_ORDER: int = len(descriptors.blocks)

# 2) Load Target
target = Target(name=config.Y_NAME, path=config.Y_PATH, normalization=config.Y_NORM)
target.load_target_from_npy()
TARGET_LENGTH: int = target.n_samples

preparation.validate_descriptor_target_lengths(descriptors.blocks, target.data)

# 3) Split/Validate
if config.N_SAMPLES > TARGET_LENGTH:
    logger.warning(
        f"Provided N_SAMPLES ({config.N_SAMPLES}) is greater than the length of the target ({TARGET_LENGTH}). All indicies will be used."
    )
    valid_idx = np.arange(TARGET_LENGTH, dtype=int)
elif config.N_SAMPLES == TARGET_LENGTH:
    valid_idx = np.arange(TARGET_LENGTH, dtype=int)
else:
    valid_idx = rng.choice(a=TARGET_LENGTH, size=config.N_SAMPLES, replace=False)

N_SAMPLES_REMAIN: int = len(valid_idx)
N_TRAIN_VAL: int = int(np.floor(N_SAMPLES_REMAIN * config.TRAIN_VAL_SPLIT))
N_TEST: int = N_SAMPLES_REMAIN - N_TRAIN_VAL

logger.info(f"{N_TRAIN_VAL} samples to be used for training/validation.")
logger.info(f"{N_TEST} samples to be held-out for testing.")

if config.STRATIFY:
    idx_train_local, idx_test_local = preparation.stratified_selection_with_remainder(
        target=target.data[valid_idx],
        n_strata=config.N_STRATA,
        n_total=N_TRAIN_VAL,
        rng=rng,
    )
    idx_train_val = valid_idx[idx_train_local]
    idx_test = valid_idx[idx_test_local]

else:
    idx_train_val, idx_test = preparation.randomized_selection_with_remainder(
        arr=valid_idx, N=N_TRAIN_VAL, rng=rng
    )

# 4) Begin Training

X_train_val = preprocess.descriptor_blocks_to_sample_matrix(descriptors, idx_train_val)
y_train_val = target.data[idx_train_val]
X_test = preprocess.descriptor_blocks_to_sample_matrix(descriptors, idx_test)
y_test = target.data[idx_test]

base_estimator = CompositeKRREstimator(
    names=config.X_NAMES,
    kernel_types=[config.KRR_KERNEL] * DESCRIPTOR_ORDER,
    normalizations=config.X_NORMS,
    normalize_kernel_weights=True,
)
estimator = TransformedTargetRegressor(
    regressor=base_estimator,
    transformer=preprocess.make_target_preprocessor(config.Y_NORM),
)
cv = KFold(n_splits=config.N_KFOLD, shuffle=True, random_state=config.SEED)

scoring = (
    "neg_root_mean_squared_error"
    if config.KRR_SCORE_METRIC == "rmse"
    else config.KRR_SCORE_METRIC
)
search_result = staged_random_search_cv(
    estimator,
    X_train_val,
    y_train_val,
    n_components=DESCRIPTOR_ORDER,
    alpha_bounds=config.KRR_ALPHA_BOUNDS,
    gamma_bounds=config.KRR_GAMMA_BOUNDS,
    n_iter_stage1=config.KRR_RANDOM_SEARCH_STAGE1,
    n_iter_stage2=config.KRR_RANDOM_SEARCH_STAGE2,
    n_iter_stage3=config.KRR_RANDOM_SEARCH_STAGE3,
    scoring=scoring,
    cv=cv,
    random_state=config.SEED,
    n_jobs=config.KRR_RANDOM_SEARCH_N_JOBS,
    prefix="regressor__",
    n_trials_bayesian=config.KRR_BAYESIAN_SEARCH_TRIALS,
    bayesian_timeout=config.KRR_BAYESIAN_SEARCH_TIMEOUT,
    bayesian_patience=config.KRR_BAYESIAN_SEARCH_PATIENCE,
)

best_cv_score = search_result.best_score_
if scoring.startswith("neg_"):
    best_cv_score = -best_cv_score

logger.warning(f"Best CV {config.KRR_SCORE_METRIC}: {best_cv_score:.6g}")
logger.info(f"Best hyperparameters: {search_result.best_params_}")

if config.KRR_RANDOM_SEARCH_PLOT_PATH is not None:
    plot_path = plot_random_search_validation_error(
        search_result,
        config.KRR_RANDOM_SEARCH_PLOT_PATH,
        scoring=scoring,
    )
    logger.warning(f"Search validation plot written to {plot_path}")

if len(idx_test) > 0:
    y_pred = search_result.best_estimator_.predict(X_test)
    test_rmse = np.sqrt(np.mean((y_test - y_pred) ** 2))
    logger.warning(f"Held-out test RMSE: {test_rmse:.6g}")
