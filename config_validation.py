import config
import numpy as np


def resolve_kernel_types(n_components: int) -> list[str]:
    kernel_types = resolve_config_sequence(
        "KRR_KERNEL",
        config.KRR_KERNEL,
        n_components,
    )

    invalid_kernel_types = [
        kernel_type
        for kernel_type in kernel_types
        if not isinstance(kernel_type, str)
    ]
    if invalid_kernel_types:
        raise ValueError("`KRR_KERNEL` must contain only strings.")

    return [kernel_type.lower() for kernel_type in kernel_types]


def resolve_pca_components(n_components: int) -> list:
    return resolve_optional_config_sequence(
        "X_PCA_COMPONENTS",
        getattr(config, "X_PCA_COMPONENTS", None),
        n_components,
        None,
    )


def resolve_pca_whiten(n_components: int) -> list[bool]:
    return resolve_optional_config_sequence(
        "X_PCA_WHITEN",
        getattr(config, "X_PCA_WHITEN", False),
        n_components,
        False,
    )


def resolve_config_sequence(name: str, values, n_items: int) -> list:
    if isinstance(values, str):
        return [values] * n_items

    try:
        resolved_values = list(values)
    except TypeError as exc:
        raise ValueError(
            f"`{name}` must be a string or a sequence with length {n_items}."
        ) from exc

    if len(resolved_values) != n_items:
        raise ValueError(
            f"`{name}` must be a string or have length {n_items}, "
            f"got {len(resolved_values)}."
        )

    return resolved_values


def resolve_optional_config_sequence(name: str, values, n_items: int, default) -> list:
    if values is None:
        return [default] * n_items
    if isinstance(values, str) or np.isscalar(values):
        return [values] * n_items

    try:
        resolved_values = list(values)
    except TypeError as exc:
        raise ValueError(
            f"`{name}` must be a scalar or a sequence with length {n_items}."
        ) from exc

    if len(resolved_values) != n_items:
        raise ValueError(
            f"`{name}` must be a scalar or have length {n_items}, "
            f"got {len(resolved_values)}."
        )

    return resolved_values


def validate_config() -> None:
    """
    Validates your configuration file to make sure your settings are sane before CKRR runs.
    """

    if type(config.SEED) is not int and config.SEED is not None:
        raise ValueError(
            "`SEED` must be of type `int` for set-seed runs "
            "or of type `None` for the seed to be set to the number of seconds since epoch."
        )

    # if type(config.VERBOSITY) not in {0, 1, 2} or config.VERBOSITY is not None:
    #     raise ValueError(
    #         "`VERBOSITY` must be in {0, 1, 2} for increasing verbosity of logs "
    #         "or of type `None` for the verbosity to be set to 1."
    #     )

    n_descriptors = len(config.X_PATHS)
    resolve_kernel_types(n_descriptors)

    krr_backend = getattr(config, "KRR_BACKEND", "exact")
    if not isinstance(krr_backend, str):
        raise ValueError('`KRR_BACKEND` must be "exact", "nystrom", or "cg".')
    if krr_backend.lower() not in {
        "exact",
        "dense",
        "nystrom",
        "nyström",
        "cg",
        "exact_cg",
        "matrix_free_cg",
        "torch_cg",
    }:
        raise ValueError('`KRR_BACKEND` must be "exact", "nystrom", or "cg".')

    for component in resolve_pca_components(n_descriptors):
        _validate_pca_components(component)

    for whiten in resolve_pca_whiten(n_descriptors):
        if type(whiten) is not bool:
            raise ValueError("`X_PCA_WHITEN` values must be bools.")

    if not isinstance(config.KRR_USE_DISTANCE_CACHE, bool):
        raise ValueError("`KRR_USE_DISTANCE_CACHE` must be a bool.")

    cached_scoring_backend = getattr(config, "KRR_CACHED_SCORING_BACKEND", "numpy")
    if not isinstance(cached_scoring_backend, str):
        raise ValueError("`KRR_CACHED_SCORING_BACKEND` must be a string.")
    if cached_scoring_backend.lower() not in {
        "numpy",
        "np",
        "cpu",
        "pytorch",
        "torch",
        "gpu",
        "cuda",
        "rocm",
    }:
        raise ValueError('`KRR_CACHED_SCORING_BACKEND` must be "numpy" or "pytorch".')

    pytorch_device = getattr(config, "KRR_PYTORCH_DEVICE", "auto")
    if pytorch_device is not None and not isinstance(pytorch_device, str):
        raise ValueError("`KRR_PYTORCH_DEVICE` must be `None` or a string.")

    pytorch_devices = getattr(config, "KRR_PYTORCH_DEVICES", None)
    _validate_optional_device_sequence("KRR_PYTORCH_DEVICES", pytorch_devices)

    pytorch_candidate_batch_size = getattr(
        config,
        "KRR_PYTORCH_CANDIDATE_BATCH_SIZE",
        1,
    )
    if (
        type(pytorch_candidate_batch_size) is not int
        or pytorch_candidate_batch_size <= 0
    ):
        raise ValueError("`KRR_PYTORCH_CANDIDATE_BATCH_SIZE` must be a positive int.")

    bayesian_batch_size = getattr(config, "KRR_BAYESIAN_BATCH_SIZE", 1)
    if type(bayesian_batch_size) is not int or bayesian_batch_size < 0:
        raise ValueError("`KRR_BAYESIAN_BATCH_SIZE` must be a non-negative int.")

    if (
        type(config.KRR_DISTANCE_BLOCK_SIZE) is not int
        or config.KRR_DISTANCE_BLOCK_SIZE <= 0
    ):
        raise ValueError("`KRR_DISTANCE_BLOCK_SIZE` must be a positive int.")

    for name in ("KRR_RANDOM_SEARCH_STAGE1", "KRR_RANDOM_SEARCH_STAGE2"):
        value = getattr(config, name)
        if type(value) is not int or value <= 0:
            raise ValueError(f"`{name}` must be a positive int.")

    if (
        type(config.KRR_RANDOM_SEARCH_STAGE3) is not int
        or config.KRR_RANDOM_SEARCH_STAGE3 < 0
    ):
        raise ValueError("`KRR_RANDOM_SEARCH_STAGE3` must be a non-negative int.")

    try:
        np.dtype(config.KRR_COMPUTE_DTYPE)
    except TypeError as exc:
        raise ValueError("`KRR_COMPUTE_DTYPE` must be a valid NumPy dtype.") from exc

    if not np.issubdtype(np.dtype(config.KRR_COMPUTE_DTYPE), np.floating):
        raise ValueError("`KRR_COMPUTE_DTYPE` must be a floating NumPy dtype.")

    try:
        np.dtype(config.KRR_DISTANCE_CACHE_DTYPE)
    except TypeError as exc:
        raise ValueError("`KRR_DISTANCE_CACHE_DTYPE` must be a valid NumPy dtype.") from exc

    if not np.issubdtype(np.dtype(config.KRR_DISTANCE_CACHE_DTYPE), np.floating):
        raise ValueError("`KRR_DISTANCE_CACHE_DTYPE` must be a floating NumPy dtype.")

    if (
        config.KRR_DISTANCE_CACHE_N_JOBS is not None
        and (
            type(config.KRR_DISTANCE_CACHE_N_JOBS) is not int
            or config.KRR_DISTANCE_CACHE_N_JOBS == 0
        )
    ):
        raise ValueError("`KRR_DISTANCE_CACHE_N_JOBS` must be `None` or a non-zero int.")

    if (
        config.KRR_RANDOM_SEARCH_BLAS_THREADS is not None
        and (
            type(config.KRR_RANDOM_SEARCH_BLAS_THREADS) is not int
            or config.KRR_RANDOM_SEARCH_BLAS_THREADS <= 0
        )
    ):
        raise ValueError(
            "`KRR_RANDOM_SEARCH_BLAS_THREADS` must be `None` or a positive int."
        )

    if not (
        isinstance(config.KRR_DISTANCE_CACHE_MEMORY_FRACTION, float)
        or isinstance(config.KRR_DISTANCE_CACHE_MEMORY_FRACTION, int)
    ):
        raise ValueError("`KRR_DISTANCE_CACHE_MEMORY_FRACTION` must be numeric.")
    if not 0 < config.KRR_DISTANCE_CACHE_MEMORY_FRACTION <= 1:
        raise ValueError("`KRR_DISTANCE_CACHE_MEMORY_FRACTION` must be in (0, 1].")

    gamma_prior_max_samples = getattr(config, "KRR_GAMMA_PRIOR_MAX_SAMPLES", None)
    if gamma_prior_max_samples is not None and (
        type(gamma_prior_max_samples) is not int or gamma_prior_max_samples <= 0
    ):
        raise ValueError("`KRR_GAMMA_PRIOR_MAX_SAMPLES` must be `None` or a positive int.")

    if not (
        isinstance(config.KRR_TOP_K_FRACTION, float)
        or isinstance(config.KRR_TOP_K_FRACTION, int)
    ):
        raise ValueError("`KRR_TOP_K_FRACTION` must be numeric.")
    if not 0 < config.KRR_TOP_K_FRACTION <= 1:
        raise ValueError("`KRR_TOP_K_FRACTION` must be in (0, 1].")

    if (
        type(config.KRR_TOP_K_MIN_CANDIDATES) is not int
        or config.KRR_TOP_K_MIN_CANDIDATES <= 0
    ):
        raise ValueError("`KRR_TOP_K_MIN_CANDIDATES` must be a positive int.")

    nystrom_n_landmarks = getattr(config, "KRR_NYSTROM_N_LANDMARKS", 2048)
    if type(nystrom_n_landmarks) is not int or nystrom_n_landmarks <= 0:
        raise ValueError("`KRR_NYSTROM_N_LANDMARKS` must be a positive int.")

    nystrom_landmark_selection = getattr(
        config,
        "KRR_NYSTROM_LANDMARK_SELECTION",
        "random",
    )
    if not isinstance(nystrom_landmark_selection, str) or (
        nystrom_landmark_selection.lower() not in {"random", "first"}
    ):
        raise ValueError(
            '`KRR_NYSTROM_LANDMARK_SELECTION` must be "random" or "first".'
        )

    nystrom_batch_size = getattr(config, "KRR_NYSTROM_BATCH_SIZE", 2048)
    if type(nystrom_batch_size) is not int or nystrom_batch_size <= 0:
        raise ValueError("`KRR_NYSTROM_BATCH_SIZE` must be a positive int.")

    nystrom_eigenvalue_floor = getattr(
        config,
        "KRR_NYSTROM_EIGENVALUE_FLOOR",
        1e-12,
    )
    if not isinstance(nystrom_eigenvalue_floor, (float, int)) or (
        nystrom_eigenvalue_floor < 0
    ):
        raise ValueError("`KRR_NYSTROM_EIGENVALUE_FLOOR` must be non-negative.")

    cg_tol = getattr(config, "KRR_CG_TOL", 1e-6)
    if not isinstance(cg_tol, (float, int)) or cg_tol <= 0:
        raise ValueError("`KRR_CG_TOL` must be positive.")

    cg_max_iter = getattr(config, "KRR_CG_MAX_ITER", 1000)
    if type(cg_max_iter) is not int or cg_max_iter <= 0:
        raise ValueError("`KRR_CG_MAX_ITER` must be a positive int.")

    cg_block_size = getattr(config, "KRR_CG_BLOCK_SIZE", 2048)
    if type(cg_block_size) is not int or cg_block_size <= 0:
        raise ValueError("`KRR_CG_BLOCK_SIZE` must be a positive int.")

    cg_pytorch_devices = getattr(
        config,
        "KRR_CG_PYTORCH_DEVICES",
        getattr(config, "KRR_PYTORCH_DEVICES", None),
    )
    _validate_optional_device_sequence("KRR_CG_PYTORCH_DEVICES", cg_pytorch_devices)

    cg_log_interval = getattr(config, "KRR_CG_LOG_INTERVAL", 25)
    if type(cg_log_interval) is not int or cg_log_interval < 0:
        raise ValueError("`KRR_CG_LOG_INTERVAL` must be a non-negative int.")

    if not isinstance(config.KRR_EVALUATE_KERNEL_CONTRIBUTIONS, bool):
        raise ValueError("`KRR_EVALUATE_KERNEL_CONTRIBUTIONS` must be a bool.")

    if (
        config.KRR_KERNEL_CONTRIBUTION_BAYESIAN_SEARCH_TRIALS is not None
        and (
            type(config.KRR_KERNEL_CONTRIBUTION_BAYESIAN_SEARCH_TRIALS) is not int
            or config.KRR_KERNEL_CONTRIBUTION_BAYESIAN_SEARCH_TRIALS < 0
        )
    ):
        raise ValueError(
            "`KRR_KERNEL_CONTRIBUTION_BAYESIAN_SEARCH_TRIALS` must be "
            "`None` or a non-negative int."
        )


def _validate_pca_components(value) -> None:
    if value is None:
        return

    if isinstance(value, np.integer):
        value = int(value)

    if isinstance(value, np.floating):
        value = float(value)

    if type(value) is int:
        if value <= 0:
            raise ValueError("Integer `X_PCA_COMPONENTS` values must be positive.")
        return

    if type(value) is float:
        if not 0.0 < value < 1.0:
            raise ValueError(
                "Float `X_PCA_COMPONENTS` values must be in the open interval (0, 1)."
            )
        return

    if isinstance(value, str):
        if value != "mle":
            raise ValueError('String `X_PCA_COMPONENTS` values must be "mle".')
        return

    raise ValueError(
        "`X_PCA_COMPONENTS` values must be None, a positive int, "
        'a float in (0, 1), or "mle".'
    )


def _validate_optional_device_sequence(name: str, devices) -> None:
    if devices is None or isinstance(devices, str):
        return

    try:
        devices = list(devices)
    except TypeError as exc:
        raise ValueError(
            f"`{name}` must be `None`, a string, or a sequence of strings."
        ) from exc
    if not devices:
        raise ValueError(f"`{name}` cannot be an empty sequence.")
    if not all(isinstance(device, str) for device in devices):
        raise ValueError(f"`{name}` must contain only strings.")
