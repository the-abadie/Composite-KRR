"""Compatibility entry point for explicit, side-effect-free configuration validation."""
from run_config import KRRConfig


def validate_config(configuration, *, check_paths=True):
    if not isinstance(configuration, KRRConfig):
        configuration = KRRConfig.from_mapping(configuration)
    configuration.validate(check_paths=check_paths)
    return configuration
