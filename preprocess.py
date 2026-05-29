import logging

import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from config import VERBOSITY
from utilities import configure_logging

configure_logging(VERBOSITY)
logger = logging.getLogger("pre-processing")


def _identity(x):
    return x


def make_target_preprocesser(transform: str):
    if transform == "none":
        logger.info("Target Transform: Identity")
        return FunctionTransformer(
            func=_identity, inverse_func=_identity, validate=False
        )
    if transform == "log":
        logger.info("Target Transform: log")
        return FunctionTransformer(func=np.log, inverse_func=np.exp, validate=False)
    if transform == "log1p":
        logger.info("Target Transform: log1p")
        return FunctionTransformer(func=np.log1p, inverse_func=np.expm1, validate=False)
    if transform == "standard":
        logger.info("Target Transform: StandardScaler")
        return StandardScaler()
    if transform == "log_standard":
        logger.info("Target Transform: logStandardScaler")
        return Pipeline(
            [
                (
                    "log",
                    FunctionTransformer(
                        func=np.log, inverse_func=np.exp, validate=False
                    ),
                ),
                ("scaler", StandardScaler()),
            ]
        )
    if transform == "log1p_standard":
        logger.info("Target Transform: log1pStandardScaler")
        return Pipeline(
            [
                (
                    "log1p",
                    FunctionTransformer(
                        func=np.log1p, inverse_func=np.expm1, validate=False
                    ),
                ),
                ("scaler", StandardScaler()),
            ]
        )
    raise ValueError(f'Unknown target transform "{transform}"')


def make_data_preprocessor(transform: str):
    if transform == "none" or transform == "passthrough":
        logger.info("Data Preprocesor: none")
        return "passthrough"
    if transform == "standard":
        logger.info("Data Preprocesor: StandardScaler()")
        return StandardScaler()
    if transform == "log_standard":
        logger.info("Data Preprocesor: logStandardScaler()")
        log1p = FunctionTransformer(
            np.log1p,
            feature_names_out="one-to-one",
            validate=False,
        )
        return Pipeline([("log1p", log1p), ("scaler", StandardScaler())])
    raise ValueError(f'Unknown data transform: "{transform}"')
