import config
import numpy as np

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

    if not isinstance(config.KRR_USE_DISTANCE_CACHE, bool):
        raise ValueError("`KRR_USE_DISTANCE_CACHE` must be a bool.")

    if (
        type(config.KRR_DISTANCE_BLOCK_SIZE) is not int
        or config.KRR_DISTANCE_BLOCK_SIZE <= 0
    ):
        raise ValueError("`KRR_DISTANCE_BLOCK_SIZE` must be a positive int.")

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
