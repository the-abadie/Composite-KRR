import logging
import random
from os import makedirs

import numpy as np
from numpy._core import float32
from numpy.typing import NDArray

import config
import preparation
import preprocess
import postprocess
from class_CompositeDescriptor import CompositeDescriptor
from class_Target import Target
from kernel_contributions import evaluate_kernel_contributions
from estimator_factory import make_composite_krr_regressor, normalize_krr_backend
from search_random import staged_random_search_cv
from sklearn.compose import TransformedTargetRegressor
from sklearn.model_selection import KFold, PredefinedSplit
from utilities import configure_logging, time_dif
from time import perf_counter, time
from config_validation import (
    resolve_kernel_types,
    resolve_pca_components,
    resolve_pca_whiten,
    validate_config,
)
from pathlib import Path

# 0) Initialize
time_0:float = perf_counter()
TIMESTAMP:int = int(time())
validate_config()

if config.OUTPUT_DIR is None:
    print(f"[init] `OUTPUT_DIR` is `None`. Setting output directory to `output/{TIMESTAMP}`")
    config.OUTPUT_DIR = Path("output") / str(TIMESTAMP)

if config.OVERWRITE_OK or config.OVERWRITE_OK is not None:
    makedirs(config.OUTPUT_DIR, exist_ok=True)
else:
    makedirs(config.OUTPUT_DIR, exist_ok=False)
print(f"[init] Output will be written to {config.OUTPUT_DIR}")

configure_logging(config.VERBOSITY, log_path=Path(config.OUTPUT_DIR) / "run.log")
logger = logging.getLogger("CKRR")
time_log = logging.getLogger("timing")
logger.warning(f"Logger initialized. Log will be written to {config.OUTPUT_DIR}/run.log")

if config.SEED is None:
    from time import time
    config.SEED = int(time())
    logger.warning(f"`SEED` not set. Setting `SEED` to seconds since epoch ({config.SEED})")

random.seed(config.SEED)
np.random.seed(config.SEED)
rng = np.random.default_rng(config.SEED)



time_end_initialization:float = perf_counter()
time_log.debug(f"Initialization completed in {time_dif(time_0, time_end_initialization)}.")

# 1) Load Descriptor
time_start_prepare_descriptor:float = perf_counter()
descriptors = CompositeDescriptor(
    names=config.X_NAMES, paths=config.X_PATHS, normalizations=config.X_NORMS
)
descriptors.load_descriptor_blocks_from_npy()
DESCRIPTOR_ORDER: int = len(descriptors.blocks)
time_end_prepare_descriptor:float = perf_counter()
time_log.info(f"Descriptor prepared in "
              f"{time_dif(time_start_prepare_descriptor, time_end_prepare_descriptor)}.")

# 2) Load Target
time_start_prepare_target:float = perf_counter()
target = Target(name=config.Y_NAME, path=Path(config.Y_PATH), normalization=config.Y_NORM)
target_data = target.load_target_from_npy()
TARGET_LENGTH: int = target.n_samples

preparation.validate_descriptor_target_lengths(descriptors.blocks, target_data)
time_end_prepare_target:float = perf_counter()

time_log.info(f"Target prepared in {time_dif(time_start_prepare_target, time_end_prepare_target)}.")

# 3) Split/Validate
time_start_prepare_splits:float = perf_counter()
if not config.USE_PREDEFINED_SPLITS:
    if config.N_SAMPLES > TARGET_LENGTH:
        logger.warning(
            f"Provided N_SAMPLES ({config.N_SAMPLES}) is greater than the length of the target \
            ({TARGET_LENGTH}). All indices will be used."
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
            target=target_data[valid_idx],
            n_strata=config.N_STRATA,
            n_total=N_TRAIN_VAL,
            rng=rng,
        )
        idx_train_val = valid_idx[idx_train_local]
        idx_test = valid_idx[idx_test_local]

    else:
        idx_train_val, idx_test = preparation.randomized_selection_with_remainder(
            arr=valid_idx, N=N_TRAIN_VAL, rng=rng)

    cv = KFold(n_splits=config.N_KFOLD, shuffle=True, random_state=config.SEED)

else: # Using pre-defined splits. Separate config validation.
    predef_idx_train: NDArray = np.asarray(np.load(config.PREDEF_TRAINING_IDX_PATH))
    predef_idx_val: list[NDArray] = preparation.load_predefined_validation_folds(
        config.PREDEF_VAL_KFOLD_IDX_PATH
    )
    predef_idx_test: NDArray = np.asarray(np.load(config.PREDEF_TESTING_IDX_PATH))

    preparation.validate_predefined_splits(
        predef_idx_train,
        predef_idx_val,
        predef_idx_test,
        TARGET_LENGTH,
    )

    idx_train_val = np.concatenate([predef_idx_train, *predef_idx_val])
    idx_test = predef_idx_test

    test_fold = np.full(len(idx_train_val), -1, dtype=int)
    offset = len(predef_idx_train)

    for fold_id, fold_val_idx in enumerate(predef_idx_val):
        n_val = len(fold_val_idx)
        test_fold[offset : offset + n_val] = fold_id
        offset += n_val

    cv = PredefinedSplit(test_fold=test_fold)

time_end_prepare_splits:float = perf_counter()

time_log.debug(f"Splits prepared in {time_dif(time_start_prepare_splits, time_end_prepare_splits)}.")

# 4) Begin Training
time_start_training:float = perf_counter()

X_train_val = preprocess.descriptor_blocks_to_sample_matrix(descriptors, idx_train_val)
y_train_val = target_data[idx_train_val]
X_test = preprocess.descriptor_blocks_to_sample_matrix(descriptors, idx_test)
y_test = target_data[idx_test]
kernel_types = resolve_kernel_types(DESCRIPTOR_ORDER)
pca_components = resolve_pca_components(DESCRIPTOR_ORDER)
pca_whiten = resolve_pca_whiten(DESCRIPTOR_ORDER)
krr_backend = normalize_krr_backend(getattr(config, "KRR_BACKEND", "exact"))

base_estimator = make_composite_krr_regressor(
    krr_backend=krr_backend,
    names=config.X_NAMES,
    kernel_types=kernel_types,
    normalizations=config.X_NORMS,
    pca_components=pca_components,
    pca_whiten=pca_whiten,
    normalize_kernel_weights=True,
    compute_dtype=config.KRR_COMPUTE_DTYPE,
    nystrom_n_landmarks=getattr(config, "KRR_NYSTROM_N_LANDMARKS", 2048),
    nystrom_landmark_selection=getattr(
        config,
        "KRR_NYSTROM_LANDMARK_SELECTION",
        "random",
    ),
    random_state=config.SEED,
    nystrom_backend=config.KRR_CACHED_SCORING_BACKEND,
    pytorch_device=config.KRR_PYTORCH_DEVICE,
    nystrom_batch_size=getattr(config, "KRR_NYSTROM_BATCH_SIZE", 2048),
    nystrom_eigenvalue_floor=getattr(
        config,
        "KRR_NYSTROM_EIGENVALUE_FLOOR",
        1e-12,
    ),
)
estimator = TransformedTargetRegressor(
    regressor=base_estimator,
    transformer=preprocess.make_target_preprocessor(config.Y_NORM),
)

scoring = (
    "neg_root_mean_squared_error"
    if config.KRR_SCORE_METRIC == "rmse"
    else config.KRR_SCORE_METRIC
)
use_distance_cache = config.KRR_USE_DISTANCE_CACHE and krr_backend == "exact"
if config.KRR_USE_DISTANCE_CACHE and krr_backend != "exact":
    logger.warning(
        "Ignoring KRR_USE_DISTANCE_CACHE because the Nyström backend uses "
        "streamed landmark features instead of exact dense distance caches."
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
    random_search_blas_threads=config.KRR_RANDOM_SEARCH_BLAS_THREADS,
    prefix="regressor__",
    n_trials_bayesian=config.KRR_BAYESIAN_SEARCH_TRIALS,
    bayesian_timeout=config.KRR_BAYESIAN_SEARCH_TIMEOUT,
    bayesian_patience=config.KRR_BAYESIAN_SEARCH_PATIENCE,
    bayesian_batch_size=config.KRR_BAYESIAN_BATCH_SIZE,
    use_distance_cache=use_distance_cache,
    distance_block_size=config.KRR_DISTANCE_BLOCK_SIZE,
    distance_dtype=config.KRR_DISTANCE_CACHE_DTYPE,
    distance_cache_n_jobs=config.KRR_DISTANCE_CACHE_N_JOBS,
    distance_cache_memory_fraction=config.KRR_DISTANCE_CACHE_MEMORY_FRACTION,
    gamma_prior_max_samples=getattr(config, "KRR_GAMMA_PRIOR_MAX_SAMPLES", 5000),
    cached_scoring_backend=config.KRR_CACHED_SCORING_BACKEND,
    pytorch_device=config.KRR_PYTORCH_DEVICE,
    pytorch_devices=config.KRR_PYTORCH_DEVICES,
    pytorch_candidate_batch_size=config.KRR_PYTORCH_CANDIDATE_BATCH_SIZE,
    top_k_fraction=config.KRR_TOP_K_FRACTION,
    top_k_min_candidates=config.KRR_TOP_K_MIN_CANDIDATES,
)
time_end_training:float = perf_counter()

time_log.info(f"Training completed in {time_dif(time_start_training, time_end_training)}.")

# 5) Postprocessing
time_start_postprocessing:float = perf_counter()
best_cv_score = search_result.best_score_
if scoring.startswith("neg_"):
    best_cv_score = -best_cv_score

logger.warning(f"Best CV {config.KRR_SCORE_METRIC}: {best_cv_score:.6g}")
logger.debug(f"Best hyperparameters: {search_result.best_params_}")

if len(idx_test) > 0:
    y_pred = search_result.best_estimator_.predict(X_test)

    test_summary = postprocess.regression_error_summary(y_true=y_test, y_pred=y_pred)

    fold_mae, fold_std = postprocess.best_validation_mae_and_fold_std(
        search_result=search_result,
        scoring=scoring,
        ddof=1)

    logger.warning(f"Best Validation MAE across folds: {fold_mae:.6f} ± {fold_std:.6f}")
    logger.warning(f"Held-out test MAE : {test_summary['mae']:.6g}")
    logger.warning(f"Held-out test RMSE: {test_summary['rmse']:.6g}")
    if len(test_summary["target_mae"]) > 1:
        for target_index, (target_mae, target_rmse) in enumerate(
            zip(test_summary["target_mae"], test_summary["target_rmse"]),
            start=1,
        ):
            logger.warning(
                f"Held-out target {target_index} MAE/RMSE: "
                f"{target_mae:.6g} / {target_rmse:.6g}"
            )

    postprocess.plot_yy(y_pred=y_pred, y_true=y_test, OUTPUT_DIR=config.OUTPUT_DIR)
    postprocess.plot_error_histogram(y_pred=y_pred, y_true=y_test, bins=250, OUTPUT_DIR=config.OUTPUT_DIR)

postprocess.plot_random_search_validation_error(
    search_result,
    scoring=scoring,
    OUTPUT_DIR=config.OUTPUT_DIR)
time_end_postprocessing_plots:float = perf_counter()

kernel_contribution_timings = []
if config.KRR_EVALUATE_KERNEL_CONTRIBUTIONS:
    contribution_bayesian_trials = (
        config.KRR_BAYESIAN_SEARCH_TRIALS
        if config.KRR_KERNEL_CONTRIBUTION_BAYESIAN_SEARCH_TRIALS is None
        else config.KRR_KERNEL_CONTRIBUTION_BAYESIAN_SEARCH_TRIALS
    )

    _, kernel_contribution_timings = evaluate_kernel_contributions(
        search_result,
        X_train_val,
        y_train_val,
        component_names=config.X_NAMES,
        kernel_types=kernel_types,
        normalizations=config.X_NORMS,
        pca_components=pca_components,
        pca_whiten=pca_whiten,
        target_normalization=config.Y_NORM,
        compute_dtype=config.KRR_COMPUTE_DTYPE,
        scoring=scoring,
        cv=cv,
        search_kwargs={
            "alpha_bounds": config.KRR_ALPHA_BOUNDS,
            "gamma_bounds": config.KRR_GAMMA_BOUNDS,
            "n_iter_stage1": config.KRR_RANDOM_SEARCH_STAGE1,
            "n_iter_stage2": config.KRR_RANDOM_SEARCH_STAGE2,
            "n_iter_stage3": config.KRR_RANDOM_SEARCH_STAGE3,
            "random_state": config.SEED,
            "n_jobs": config.KRR_RANDOM_SEARCH_N_JOBS,
            "random_search_blas_threads": config.KRR_RANDOM_SEARCH_BLAS_THREADS,
            "prefix": "regressor__",
            "n_trials_bayesian": contribution_bayesian_trials,
            "bayesian_timeout": config.KRR_BAYESIAN_SEARCH_TIMEOUT,
            "bayesian_patience": config.KRR_BAYESIAN_SEARCH_PATIENCE,
            "bayesian_batch_size": config.KRR_BAYESIAN_BATCH_SIZE,
            "use_distance_cache": use_distance_cache,
            "distance_block_size": config.KRR_DISTANCE_BLOCK_SIZE,
            "distance_dtype": config.KRR_DISTANCE_CACHE_DTYPE,
            "distance_cache_n_jobs": config.KRR_DISTANCE_CACHE_N_JOBS,
            "distance_cache_memory_fraction": (
                config.KRR_DISTANCE_CACHE_MEMORY_FRACTION
            ),
            "gamma_prior_max_samples": getattr(
                config,
                "KRR_GAMMA_PRIOR_MAX_SAMPLES",
                5000,
            ),
            "cached_scoring_backend": config.KRR_CACHED_SCORING_BACKEND,
            "pytorch_device": config.KRR_PYTORCH_DEVICE,
            "pytorch_devices": config.KRR_PYTORCH_DEVICES,
            "pytorch_candidate_batch_size": config.KRR_PYTORCH_CANDIDATE_BATCH_SIZE,
            "top_k_fraction": config.KRR_TOP_K_FRACTION,
            "top_k_min_candidates": config.KRR_TOP_K_MIN_CANDIDATES,
        },
        krr_backend=krr_backend,
        nystrom_kwargs={
            "nystrom_n_landmarks": getattr(config, "KRR_NYSTROM_N_LANDMARKS", 2048),
            "nystrom_landmark_selection": getattr(
                config,
                "KRR_NYSTROM_LANDMARK_SELECTION",
                "random",
            ),
            "random_state": config.SEED,
            "nystrom_backend": config.KRR_CACHED_SCORING_BACKEND,
            "pytorch_device": config.KRR_PYTORCH_DEVICE,
            "nystrom_batch_size": getattr(config, "KRR_NYSTROM_BATCH_SIZE", 2048),
            "nystrom_eigenvalue_floor": getattr(
                config,
                "KRR_NYSTROM_EIGENVALUE_FLOOR",
                1e-12,
            ),
        },
        output_dir=config.OUTPUT_DIR,
        ddof=1,
    )

time_end_postprocessing:float = perf_counter()

time_log.info(f"Post-processing completed in {time_dif(time_start_postprocessing, time_end_postprocessing)}.")

time_f:float = perf_counter()
time_log.warning(f"CKRR learning stack completed in {time_dif(time_0, time_f)}.")

training_timings = search_result.timings

postprocess.runtime_analysis([
    (time_0, time_end_initialization, "Initialization and Config Validation"),
    (time_start_prepare_descriptor, time_end_prepare_target, "Descriptor/Target Preparation"),
    *training_timings,
    (time_start_postprocessing, time_end_postprocessing_plots, "Post-Processing: Final Model Evaluation"),
    *kernel_contribution_timings,
])
