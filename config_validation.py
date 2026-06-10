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
