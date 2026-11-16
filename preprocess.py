from class_CompositeDescriptor import CompositeDescriptor
from descriptor_storage import gather_descriptor_rows
import logging
from collections.abc import Mapping

import numpy as np
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, PowerTransformer, StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted, validate_data
from scipy.fft import dct, rfft


logger = logging.getLogger("pre-processing")


def _identity(x):
    return x


def _negate(x):
    return -x


class AsinhIQRTransformer(TransformerMixin, BaseEstimator):
    """Apply asinh(y / training IQR), without centering the physical target."""

    def fit(self, X, y=None):
        X = validate_data(self, X, dtype=np.float64)
        self.q25_, self.q75_ = np.quantile(X, [.25, .75], axis=0, method="linear")
        self.iqr_ = self.q75_ - self.q25_
        std = X.std(axis=0)
        self.scale_ = np.where(self.iqr_ > 0, self.iqr_, np.where(std > 0, std, 1.))
        self.scale_source_ = np.where(self.iqr_ > 0, "iqr", np.where(std > 0, "std", "unit"))
        if not np.isfinite(self.scale_).all():
            raise ValueError("Asinh target scale must be finite")
        return self

    def transform(self, X):
        check_is_fitted(self, "scale_")
        X = validate_data(self, X, reset=False, dtype=np.float64)
        return np.arcsinh(X / self.scale_)

    def inverse_transform(self, X):
        check_is_fitted(self, "scale_")
        X = validate_data(self, X, reset=False, dtype=np.float64)
        with np.errstate(over="ignore"):
            return np.sinh(X) * self.scale_


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


class SpatialFFT(TransformerMixin, BaseEstimator):
    """Channel-wise Fourier projection with isometric real/imaginary packing."""

    def __init__(self, *, block_size, n_frequencies):
        self.block_size = block_size
        self.n_frequencies = n_frequencies

    def fit(self, X, y=None):
        if (type(self.block_size) is not int or type(self.n_frequencies) is not int
                or self.block_size < 1 or not 0 < self.n_frequencies <= self.block_size // 2 + 1):
            raise ValueError("Spatial FFT requires integer bins and 0 < n_frequencies <= block_size//2+1")
        X = validate_data(self, X, dtype=[np.float64, np.float32])
        if X.shape[1] % self.block_size:
            raise ValueError("Descriptor width must be a multiple of spatial_fft.block_size")
        self.channel_width_ = min(2 * self.n_frequencies - 1, self.block_size)
        self.n_features_out_ = X.shape[1] // self.block_size * self.channel_width_
        return self

    def transform(self, X):
        check_is_fitted(self, "n_features_out_")
        X = validate_data(self, X, reset=False, dtype=[np.float64, np.float32])
        channels = X.reshape(len(X), -1, self.block_size)
        coefficients = rfft(channels, norm="ortho", axis=-1)[..., :self.n_frequencies]
        packed = np.empty(channels.shape[:-1] + (self.channel_width_,), dtype=X.dtype)
        packed[..., 0] = coefficients[..., 0].real
        stop = min(self.n_frequencies, (self.block_size + 1) // 2)
        packed[..., 1:2*stop-1:2] = np.sqrt(2) * coefficients[..., 1:stop].real
        packed[..., 2:2*stop:2] = np.sqrt(2) * coefficients[..., 1:stop].imag
        if self.block_size % 2 == 0 and self.n_frequencies == self.block_size // 2 + 1:
            packed[..., -1] = coefficients[..., -1].real
        return packed.reshape(len(X), -1)


def descriptor_preprocessing(descriptor):
    """Keep legacy string recipes; carry structured operations through all caches."""
    for name in ("spatial_dct", "spatial_fft"):
        operation = getattr(descriptor, name, None)
        if operation is not None:
            return {"normalization": descriptor.normalization,
                    name: operation.model_dump(mode="json", exclude_defaults=True)}
    return descriptor.normalization


def make_target_preprocessor(transform: str):
    if transform == "asinh_iqr_standard":
        return Pipeline([("asinh", AsinhIQRTransformer()), ("scaler", StandardScaler())])
    if transform in {"negate_log_standard", "negate_boxcox_standard"}:
        return Pipeline([
            ("negate", FunctionTransformer(func=_negate, inverse_func=_negate, validate=False)),
            ("transform", make_target_preprocessor(transform.removeprefix("negate_"))),
        ])
    if transform in {"boxcox_standard", "yeojohnson_standard"}:
        method = "box-cox" if transform == "boxcox_standard" else "yeo-johnson"
        return PowerTransformer(method=method, standardize=True)
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
    spatial_operation = None
    if isinstance(transform, Mapping):
        operations = set(transform) - {"normalization"}
        if "normalization" not in transform or len(operations) != 1 or not operations <= {"spatial_dct", "spatial_fft"}:
            raise ValueError("Unknown descriptor preprocessing recipe fields")
        spatial_operation = next(iter(operations))
        spatial_options = transform[spatial_operation]
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

    if spatial_operation is not None and spatial_options is not None:
        options = dict(spatial_options)
        post_normalization = options.pop("post_normalization", "none")
        if post_normalization not in ("none", "standard"):
            raise ValueError(f"Unknown spatial transform post-normalization: {post_normalization}")
        transform_type = SpatialDCT if spatial_operation == "spatial_dct" else SpatialFFT
        steps.append((spatial_operation, transform_type(**options)))
        if post_normalization == "standard":
            steps.append(("post_scaler", StandardScaler()))

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
