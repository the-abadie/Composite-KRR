from class_CompositeDescriptor import CompositeDescriptor
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


def make_target_preprocessor(transform: str):
    if transform == "none":
        logger.debug("Target Transform: Identity")
        return FunctionTransformer(
            func=_identity, inverse_func=_identity, validate=False
        )
    if transform == "log":
        logger.debug("Target Transform: log")
        return FunctionTransformer(func=np.log, inverse_func=np.exp, validate=False)
    if transform == "log1p":
        logger.debug("Target Transform: log1p")
        return FunctionTransformer(func=np.log1p, inverse_func=np.expm1, validate=False)
    if transform == "standard":
        logger.debug("Target Transform: StandardScaler")
        return StandardScaler()
    if transform == "log_standard":
        logger.debug("Target Transform: logStandardScaler")
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
        logger.debug("Data Preprocesor: none")
        return FunctionTransformer(
            func=_identity, inverse_func=_identity, validate=False
        )
    if transform == "standard":
        # logger.debug("Data Preprocesor: StandardScaler")
        return StandardScaler()
    if transform == "log_standard":
        logger.debug("Data Preprocesor: logStandardScaler")
        log1p = FunctionTransformer(
            np.log1p,
            feature_names_out="one-to-one",
            validate=False,
        )
        return Pipeline([("log1p", log1p), ("scaler", StandardScaler())])
    raise ValueError(f'Unknown data transform: "{transform}"')


def transform_descriptors_for_split(descriptors:CompositeDescriptor, train_idx, eval_idx):
    X_train_blocks = []
    X_eval_blocks = []
    preprocessors = []

    for descriptor in descriptors.blocks:
        transformer = make_data_preprocessor(descriptor.normalization)

        x_train = descriptor.values[train_idx]
        x_eval = descriptor.values[eval_idx]
        descriptor_shape = descriptor.values.shape[1:]

        x_train_2d = x_train.reshape(len(train_idx), -1)
        x_eval_2d = x_eval.reshape(len(eval_idx), -1)

        x_train_t = transformer.fit_transform(x_train_2d)
        x_eval_t = transformer.transform(x_eval_2d)

        X_train_blocks.append(x_train_t.reshape((len(train_idx),) + descriptor_shape))
        X_eval_blocks.append(x_eval_t.reshape((len(eval_idx),) + descriptor_shape))
        preprocessors.append(transformer)

    return X_train_blocks, X_eval_blocks, preprocessors


def descriptor_blocks_to_sample_matrix(
    descriptors: CompositeDescriptor, indices=None
) -> np.ndarray:
    if not descriptors.blocks:
        raise ValueError("Descriptor blocks have not been loaded.")

    if indices is None:
        indices = np.arange(descriptors.blocks[0].n_samples)
    indices = np.asarray(indices, dtype=int)

    X = np.empty((len(indices), len(descriptors.blocks)), dtype=object)
    for block_idx, descriptor in enumerate(descriptors.blocks):
        values = descriptor.values[indices]
        for sample_idx, value in enumerate(values):
            X[sample_idx, block_idx] = value

    return X
