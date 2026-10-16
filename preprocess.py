from class_CompositeDescriptor import CompositeDescriptor
from descriptor_storage import gather_descriptor_rows
import logging
from collections.abc import Mapping

import numpy as np
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted, validate_data
from scipy.fft import dct


logger = logging.getLogger("pre-processing")


def _identity(x):
    return x


class SpatialDCT(TransformerMixin, BaseEstimator):
    """Orthogonal spatial transform for consecutive fixed-width channels."""

    def __init__(self, *, block_size, n_components):
        self.block_size = block_size
        self.n_components = n_components

    def fit(self, X, y=None):
        if type(self.block_size) is not int or type(self.n_components) is not int or not 0 < self.n_components <= self.block_size:
            raise ValueError("Spatial DCT requires 0 < n_components <= block_size integers")
        X = validate_data(self, X, dtype=[np.float64, np.float32])
        if X.shape[1] % self.block_size:
            raise ValueError("Descriptor width must be a multiple of spatial_dct.block_size")
        self.n_features_out_ = X.shape[1] // self.block_size * self.n_components
        return self

    def transform(self, X):
        check_is_fitted(self, "n_features_out_")
        X = validate_data(self, X, reset=False, dtype=[np.float64, np.float32])
        channels = X.reshape(len(X), -1, self.block_size)
        return dct(channels, type=2, norm="ortho", axis=-1)[..., :self.n_components].reshape(len(X), -1)


def descriptor_preprocessing(descriptor):
    """Keep legacy string recipes; carry structured operations through all caches."""
    if descriptor.spatial_dct is None:
        return descriptor.normalization
    return {"normalization": descriptor.normalization,
            "spatial_dct": descriptor.spatial_dct.model_dump(mode="json")}


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
        # logger.debug("Target Transform: StandardScaler")
        return StandardScaler()
    if transform == "log_standard":
        # logger.debug("Target Transform: logStandardScaler")
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
        # logger.info("Target Transform: log1pStandardScaler")
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


def make_data_preprocessor(
    transform: str | Mapping,
    pca_components=None,
    *,
    pca_whiten: bool = False,
):
    steps = []
    spatial_dct = None
    if isinstance(transform, Mapping):
        if set(transform) != {"normalization", "spatial_dct"}:
            raise ValueError("Unknown descriptor preprocessing recipe fields")
        spatial_dct = transform["spatial_dct"]
        transform = transform["normalization"]

    if transform == "none" or transform == "passthrough":
        # logger.debug("Data Preprocesor: none")
        pass
    elif transform == "standard":
        # logger.debug("Data Preprocesor: StandardScaler")
        steps.append(("scaler", StandardScaler()))
    elif transform == "log_standard":
        # logger.debug("Data Preprocesor: logStandardScaler")
        log1p = FunctionTransformer(
            np.log1p,
            feature_names_out="one-to-one",
            validate=False,
        )
        steps.extend([("log1p", log1p), ("scaler", StandardScaler())])
    else:
        raise ValueError(f'Unknown data transform: "{transform}"')

    if spatial_dct is not None:
        steps.append(("spatial_dct", SpatialDCT(**spatial_dct)))

    if pca_components is not None:
        steps.append(
            (
                "pca",
                PCA(
                    n_components=pca_components,
                    whiten=pca_whiten,
                    svd_solver="auto",
                ),
            )
        )

    if not steps:
        return FunctionTransformer(
            func=_identity, inverse_func=_identity, validate=False
        )

    return Pipeline(steps)


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

        shape = descriptor_shape if x_train_t.shape[1] == int(np.prod(descriptor_shape)) else (x_train_t.shape[1],)
        X_train_blocks.append(x_train_t.reshape((len(train_idx),) + shape))
        X_eval_blocks.append(x_eval_t.reshape((len(eval_idx),) + shape))
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
    values_by_block = gather_descriptor_rows([descriptor.values for descriptor in descriptors.blocks], indices)
    for block_idx, values in enumerate(values_by_block):
        for sample_idx, value in enumerate(values):
            X[sample_idx, block_idx] = value

    return X
