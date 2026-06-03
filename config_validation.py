import config

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
