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
from scipy.optimize import minimize_scalar
from scipy.stats import yeojohnson_llf


logger = logging.getLogger("pre-processing")


def _identity(x):
    return x


def _negate(x):
    return -x


def _signed_sqrt(x):
    return np.sign(x) * np.sqrt(np.abs(x))


def _signed_square(x):
    return np.sign(x) * np.square(x)


def _cube(x):
    return np.power(x, 3)


class BoundedYeoJohnsonTransformer(PowerTransformer):
    """Yeo–Johnson training MLE on the closed interval [0, 2].

    Bounds constrain the likelihood optimization, not predictions or labels.
    Inherited fit/transform handling fits standardization after the power and
    preserves sklearn's constant-column and multi-target behavior.
    """

    lambda_bounds = (0.0, 2.0)

    def __init__(self, *, standardize=True, copy=True):
        super().__init__(method="yeo-johnson", standardize=standardize, copy=copy)

    def _yeo_johnson_optimize(self, x):
        # sklearn's optimizer receives one training column at a time. Retain
        # its NaN handling; the engine separately requires finite target labels.
        x = x[~np.isnan(x)]

        def objective(power):
            with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                score = -float(yeojohnson_llf(power, x))
            return score if np.isfinite(score) else np.inf

        optimum = minimize_scalar(objective, bounds=self.lambda_bounds,
                                  method="bounded", options={"xatol": 1e-10})
        if not optimum.success:
            raise ValueError("Bounded Yeo-Johnson maximum-likelihood fit did not converge")
        # A bounded solver samples the open interval: test both endpoints too.
        candidates = (self.lambda_bounds[0], float(optimum.x), self.lambda_bounds[1])
        scores = [objective(power) for power in candidates]
        best = int(np.argmin(scores))
        if not np.isfinite(scores[best]):
            raise ValueError("Bounded Yeo-Johnson maximum-likelihood fit is nonfinite")
        return candidates[best]


class _IQRTargetTransformer(TransformerMixin, BaseEstimator):
    """Fit column-wise target scale and optional median on training rows only."""

    def __init__(self, *, center=False):
        self.center = center

    def fit(self, X, y=None):
        X = validate_data(self, X, dtype=np.float64)
        self.q25_, self.q75_ = np.quantile(X, [.25, .75], axis=0, method="linear")
        self.iqr_ = self.q75_ - self.q25_
        std = X.std(axis=0)
        self.scale_ = np.where(self.iqr_ > 0, self.iqr_, np.where(std > 0, std, 1.))
        self.scale_source_ = np.where(self.iqr_ > 0, "iqr", np.where(std > 0, "std", "unit"))
        if not np.isfinite(self.scale_).all():
            raise ValueError("IQR target scale must be finite")
        self.center_ = np.median(X, axis=0) if self.center else np.zeros(X.shape[1])
        return self


class AsinhIQRTransformer(_IQRTargetTransformer):
    """Apply asinh((y - optional training median) / training IQR)."""

    def transform(self, X):
        check_is_fitted(self, "scale_")
        X = validate_data(self, X, reset=False, dtype=np.float64)
        if self.center:
            X = X - self.center_
        return np.arcsinh(X / self.scale_)

    def inverse_transform(self, X):
        check_is_fitted(self, "scale_")
        X = validate_data(self, X, reset=False, dtype=np.float64)
        with np.errstate(over="ignore"):
            result = np.sinh(X) * self.scale_
        return result + self.center_ if self.center else result


class SignedLogIQRTransformer(_IQRTargetTransformer):
    """Signed log1p with a training-only IQR scale."""

    def transform(self, X):
        check_is_fitted(self, "scale_")
        X = validate_data(self, X, reset=False, dtype=np.float64)
        if self.center:
            X = X - self.center_
        return np.sign(X) * np.log1p(np.abs(X) / self.scale_)

    def inverse_transform(self, X):
        check_is_fitted(self, "scale_")
        X = validate_data(self, X, reset=False, dtype=np.float64)
        with np.errstate(over="ignore"):
            result = np.sign(X) * np.expm1(np.abs(X)) * self.scale_
        return result + self.center_ if self.center else result


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
    if transform == "signed_sqrt_standard":
        return Pipeline([("signed_sqrt", FunctionTransformer(_signed_sqrt, inverse_func=_signed_square)),
                         ("scaler", StandardScaler())])
    if transform == "cbrt_standard":
        return Pipeline([("cbrt", FunctionTransformer(np.cbrt, inverse_func=_cube)),
                         ("scaler", StandardScaler())])
    if transform == "signed_log_iqr_standard":
        return Pipeline([("signed_log", SignedLogIQRTransformer()), ("scaler", StandardScaler())])
    if transform == "asinh_median_iqr_standard":
        return Pipeline([("asinh", AsinhIQRTransformer(center=True)), ("scaler", StandardScaler())])
    if transform == "asinh_iqr_standard":
        return Pipeline([("asinh", AsinhIQRTransformer()), ("scaler", StandardScaler())])
    if transform in {"negate_log_standard", "negate_boxcox_standard"}:
        return Pipeline([
            ("negate", FunctionTransformer(func=_negate, inverse_func=_negate, validate=False)),
            ("transform", make_target_preprocessor(transform.removeprefix("negate_"))),
        ])
    if transform == "yeojohnson_bounded_standard":
        return BoundedYeoJohnsonTransformer(standardize=True)
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
