"""Scientific workflow driven by a validated RunConfig object."""
import logging
import random

import numpy as np
from numpy.typing import NDArray

import preparation
import preprocess
import postprocess
from class_CompositeDescriptor import CompositeDescriptor
from class_Target import Target
from kernel_contributions import evaluate_kernel_contributions
from estimator_factory import make_composite_krr_regressor
from search_random import staged_random_search_cv
from sklearn.compose import TransformedTargetRegressor
from sklearn.model_selection import KFold, PredefinedSplit
from utilities import configure_logging, time_dif
from time import perf_counter
from pathlib import Path

def _execute(spec):
    time_0 = perf_counter()
    logger = logging.getLogger("CKRR")
    time_log = logging.getLogger("timing")
    names = [d.id for d in spec.descriptors]
    paths = [d.path for d in spec.descriptors]
    normalizations = [d.normalization for d in spec.descriptors]
    products = [[names.index(name) for name in factors] for factors in spec.kernel_products]
    predefined = spec.split.mode == "predefined"
    random.seed(spec.seed)
    np.random.seed(spec.seed)
    rng = np.random.default_rng(spec.seed)
    time_end_initialization = perf_counter()

    # 1) Load Descriptor
    time_start_prepare_descriptor:float = perf_counter()
    descriptors = CompositeDescriptor(
        names=names, paths=paths, normalizations=normalizations
    )
    descriptors.load_descriptor_blocks_from_npy()
    DESCRIPTOR_ORDER: int = len(descriptors.blocks)
    time_end_prepare_descriptor:float = perf_counter()
    time_log.info(f"Descriptor prepared in "
                  f"{time_dif(time_start_prepare_descriptor, time_end_prepare_descriptor)}.")

    # 2) Load Target
    time_start_prepare_target:float = perf_counter()
    target = Target(name=spec.target.name, path=Path(spec.target.path), normalization=spec.target.normalization)
    target_data = (target.load_target_from_npz() if Path(spec.target.path).suffix.lower() == ".npz"
                   else target.load_target_from_npy())
    TARGET_LENGTH: int = target.n_samples

    preparation.validate_descriptor_target_lengths(descriptors.blocks, target_data)
    time_end_prepare_target:float = perf_counter()

    time_log.info(f"Target prepared in {time_dif(time_start_prepare_target, time_end_prepare_target)}.")

    # 3) Split/Validate
    time_start_prepare_splits:float = perf_counter()
    if not predefined:
        pool_size = spec.split.n_samples or TARGET_LENGTH
        if pool_size > TARGET_LENGTH:
            raise ValueError(f"split.n_samples={pool_size} exceeds the target length {TARGET_LENGTH}.")
        valid_idx = (np.arange(TARGET_LENGTH, dtype=int) if pool_size == TARGET_LENGTH
                     else rng.choice(TARGET_LENGTH, size=pool_size, replace=False))
        N_SAMPLES_REMAIN = len(valid_idx)
        N_TRAIN_VAL = spec.split.training_count(N_SAMPLES_REMAIN)
        if not spec.split.n_folds <= N_TRAIN_VAL < N_SAMPLES_REMAIN:
            raise ValueError("Training count must be at least n_folds and leave held-out test samples.")
        if spec.split.stratify and spec.split.n_strata > N_TRAIN_VAL:
            raise ValueError("split.n_strata cannot exceed the training count.")
        N_TEST: int = N_SAMPLES_REMAIN - N_TRAIN_VAL

        logger.info(f"{N_TRAIN_VAL} samples to be used for training/validation.")
        logger.info(f"{N_TEST} samples to be held-out for testing.")

        if spec.split.stratify:
            idx_train_local, idx_test_local = preparation.stratified_selection_with_remainder(
                target=target_data[valid_idx],
                n_strata=spec.split.n_strata,
                n_total=N_TRAIN_VAL,
                rng=rng,
            )
            idx_train_val = valid_idx[idx_train_local]
            idx_test = valid_idx[idx_test_local]

        else:
            idx_train_val, idx_test = preparation.randomized_selection_with_remainder(
                arr=valid_idx, N=N_TRAIN_VAL, rng=rng)

        cv = KFold(n_splits=spec.split.n_folds, shuffle=True, random_state=spec.seed)

    else: # Using pre-defined splits. Separate config validation.
        predef_idx_train: NDArray = np.asarray(np.load(spec.split.train_indices))
        predef_idx_val: list[NDArray] = preparation.load_predefined_validation_folds(
            spec.split.validation_folds
        )
        predef_idx_test: NDArray = np.asarray(np.load(spec.split.test_indices))

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
    kernel_types = [d.kernel for d in spec.descriptors]
    pca_components = [d.pca_components for d in spec.descriptors]
    pca_whiten = [d.pca_whiten for d in spec.descriptors]
    krr_backend = spec.model.backend

    base_estimator = make_composite_krr_regressor(
        krr_backend=krr_backend,
        names=names,
        kernel_types=kernel_types,
        kernel_products=products,
        normalizations=normalizations,
        pca_components=pca_components,
        pca_whiten=pca_whiten,
        normalize_kernel_weights=True,
        compute_dtype=spec.execution.dtype,
        nystrom_n_landmarks=spec.model.nystrom.n_landmarks,
        nystrom_landmark_selection=spec.model.nystrom.landmark_selection,
        random_state=spec.seed,
        nystrom_backend=spec.execution.backend,
        pytorch_device=spec.execution.torch.device,
        pytorch_predict_batch_size=spec.execution.torch.predict_batch_size,
        nystrom_batch_size=spec.model.nystrom.batch_size,
        nystrom_eigenvalue_floor=spec.model.nystrom.eigenvalue_floor,
    )
    estimator = TransformedTargetRegressor(
        regressor=base_estimator,
        transformer=preprocess.make_target_preprocessor(spec.target.normalization),
    )

    scoring = (
        "neg_root_mean_squared_error"
        if spec.search.scoring == "rmse"
        else spec.search.scoring
    )
    use_distance_cache = spec.execution.distance_cache.enabled
    if not spec.execution.distance_cache.enabled and krr_backend == "nystrom":
        logger.warning(
            "Nyström backend is running without landmark distance caching; "
            "hyperparameter search will refit every candidate/fold through sklearn."
        )

    search_result = staged_random_search_cv(
        estimator,
        X_train_val,
        y_train_val,
        n_components=DESCRIPTOR_ORDER,
        alpha_bounds=spec.search.alpha_bounds,
        gamma_bounds=spec.search.gamma_bounds,
        n_iter_stage1=spec.search.random.stage1,
        n_iter_stage2=spec.search.random.stage2,
        n_iter_stage3=spec.search.random.stage3,
        scoring=scoring,
        cv=cv,
        random_state=spec.seed,
        n_jobs=spec.execution.n_jobs,
        random_search_blas_threads=spec.execution.blas_threads,
        prefix="regressor__",
        n_trials_bayesian=spec.search.bayesian.trials,
        bayesian_timeout=spec.search.bayesian.timeout,
        bayesian_patience=spec.search.bayesian.patience,
        bayesian_batch_size=spec.search.bayesian.batch_size,
        use_distance_cache=use_distance_cache,
        distance_block_size=spec.execution.distance_cache.block_size,
        distance_dtype=(spec.execution.distance_cache.dtype or spec.execution.dtype),
        distance_cache_n_jobs=spec.execution.distance_cache.n_jobs,
        distance_cache_memory_fraction=spec.execution.distance_cache.memory_fraction,
        gamma_prior_max_samples=spec.search.gamma_prior_max_samples,
        cached_scoring_backend=spec.execution.backend,
        pytorch_device=spec.execution.torch.device,
        pytorch_devices=spec.execution.torch.devices,
        pytorch_candidate_batch_size=spec.execution.torch.candidate_batch_size,
        top_k_fraction=spec.search.top_k_fraction,
        top_k_min_candidates=spec.search.top_k_min_candidates,
    )
    time_end_training:float = perf_counter()

    time_log.info(f"Training completed in {time_dif(time_start_training, time_end_training)}.")

    # 5) Postprocessing
    time_start_postprocessing:float = perf_counter()
    best_cv_score = search_result.best_score_
    if scoring.startswith("neg_"):
        best_cv_score = -best_cv_score

    logger.warning(f"Best CV {spec.search.scoring}: {best_cv_score:.6g}")
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


        postprocess.plot_yy(y_pred=y_pred, y_true=y_test, OUTPUT_DIR=spec.output.directory)
        postprocess.plot_error_histogram(y_pred=y_pred, y_true=y_test, bins=250, OUTPUT_DIR=spec.output.directory)

        final = search_result.best_estimator_   # TransformedTargetRegressor
        reg = final.regressor_                  # fitted inner KRR estimator
        model = getattr(reg, "model_", reg)

        if hasattr(model, "dual_coef_"):
            sample_weights = model.dual_coef_   # NumPy, shape (n_train, n_targets)
            np.save(file=f"{spec.output.directory}/sample_weights.npy", arr=sample_weights)
        elif hasattr(model, "dual_coef_tensor_"):
            sample_weights = model.dual_coef_tensor_.detach().cpu().numpy()
            np.save(file=f"{spec.output.directory}/sample_weights.npy", arr=sample_weights)

        np.save(file=f"{spec.output.directory}/fold_val_idx.npy", arr=idx_train_val)
        np.save(file=f"{spec.output.directory}/test_idx.npy", arr=idx_test)
        np.save(file=f"{spec.output.directory}/y_predictions.npy", arr=y_pred)
        np.save(file=f"{spec.output.directory}/y_true.npy", arr=y_test)
        np.save(file=f"{spec.output.directory}/gammas.npy", arr=search_result.best_params_["regressor__gammas"])
        np.save(file=f"{spec.output.directory}/kernel_weights.npy", arr=search_result.best_params_["regressor__kernel_weights"])
        np.save(file=f"{spec.output.directory}/alpha.npy", arr=search_result.best_params_["regressor__alpha"])

    postprocess.plot_random_search_validation_error(
        search_result,
        scoring=scoring,
        OUTPUT_DIR=spec.output.directory)
    time_end_postprocessing_plots:float = perf_counter()

    kernel_contribution_timings = []
    if spec.reporting.kernel_contributions:
        contribution_bayesian_trials = (
            spec.search.bayesian.trials
            if spec.reporting.contribution_bayesian_trials is None
            else spec.reporting.contribution_bayesian_trials
        )

        _, kernel_contribution_timings = evaluate_kernel_contributions(
            search_result,
            X_train_val,
            y_train_val,
            component_names=names,
            kernel_types=kernel_types,
            kernel_products=products,
            normalizations=normalizations,
            pca_components=pca_components,
            pca_whiten=pca_whiten,
            target_normalization=spec.target.normalization,
            compute_dtype=spec.execution.dtype,
            scoring=scoring,
            cv=cv,
            search_kwargs={
                "alpha_bounds": spec.search.alpha_bounds,
                "gamma_bounds": spec.search.gamma_bounds,
                "n_iter_stage1": spec.search.random.stage1,
                "n_iter_stage2": spec.search.random.stage2,
                "n_iter_stage3": spec.search.random.stage3,
                "random_state": spec.seed,
                "n_jobs": spec.execution.n_jobs,
                "random_search_blas_threads": spec.execution.blas_threads,
                "prefix": "regressor__",
                "n_trials_bayesian": contribution_bayesian_trials,
                "bayesian_timeout": spec.search.bayesian.timeout,
                "bayesian_patience": spec.search.bayesian.patience,
                "bayesian_batch_size": spec.search.bayesian.batch_size,
                "use_distance_cache": use_distance_cache,
                "distance_block_size": spec.execution.distance_cache.block_size,
                "distance_dtype": (spec.execution.distance_cache.dtype or spec.execution.dtype),
                "distance_cache_n_jobs": spec.execution.distance_cache.n_jobs,
                "distance_cache_memory_fraction": (
                    spec.execution.distance_cache.memory_fraction
                ),
                "gamma_prior_max_samples": spec.search.gamma_prior_max_samples,
                "cached_scoring_backend": spec.execution.backend,
                "pytorch_device": spec.execution.torch.device,
                "pytorch_devices": spec.execution.torch.devices,
                "pytorch_candidate_batch_size": spec.execution.torch.candidate_batch_size,
                "top_k_fraction": spec.search.top_k_fraction,
                "top_k_min_candidates": spec.search.top_k_min_candidates,
            },
            krr_backend=krr_backend,
            nystrom_kwargs={
                "nystrom_n_landmarks": spec.model.nystrom.n_landmarks,
                "nystrom_landmark_selection": spec.model.nystrom.landmark_selection,
                "random_state": spec.seed,
                "nystrom_backend": spec.execution.backend,
                "pytorch_device": spec.execution.torch.device,
                "pytorch_predict_batch_size": spec.execution.torch.predict_batch_size,
                "nystrom_batch_size": spec.model.nystrom.batch_size,
                "nystrom_eigenvalue_floor": spec.model.nystrom.eigenvalue_floor,
            },
            output_dir=spec.output.directory,
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


def run(configuration):
    """Validate and record one run, then execute it with no shared configuration state."""
    from secrets import randbits
    from run_config import RunExecution, write_immutable

    configuration.validate()
    resolved = configuration.resolved()
    if resolved.spec.seed is None:
        resolved = resolved.with_overrides({"seed": randbits(32)})
    output = Path(resolved.spec.output.directory)
    output.mkdir(parents=True, exist_ok=True)
    lock = output / ".run.lock"
    with lock.open("x") as handle:
        import os
        handle.write(f"pid={os.getpid()}\n")
    root = logging.getLogger()
    previous_level = root.level
    previous_handlers = list(root.handlers)
    try:
        existing = [path for path in output.iterdir() if path != lock]
        if existing and not resolved.spec.output.overwrite:
            raise FileExistsError(f"Output directory is not empty: {output}; choose a new directory or set output.overwrite.")
        write_immutable(output / "resolved_config.json", resolved.canonical_json())
        write_immutable(output / "resolved_config.sha256", resolved.sha256() + "\n")
        configure_logging(resolved.spec.verbosity, log_path=output / "run.log")
        _execute(resolved.spec)
        return RunExecution(0, output, resolved.sha256())
    finally:
        for handler in list(root.handlers):
            if handler not in previous_handlers:
                root.removeHandler(handler)
                handler.close()
        root.setLevel(previous_level)
        lock.unlink()
