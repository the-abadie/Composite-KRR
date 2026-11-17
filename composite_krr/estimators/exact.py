"""Scikit-learn adapter for exact composite KRR on NumPy or PyTorch."""
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_is_fitted

from class_CompositeKRR import CompositeKRR, KernelComponent
from composite_krr.estimator_inputs import (
    normalize_compute_backend,
    resolve_kernel_hyperparameters,
    resolve_sequence,
    unpack_sample_matrix,
)
from kernel_mixing import resolve_kernel_products
from linear_baseline import fit_linear_baseline, predict_linear_baseline
from preprocess import make_data_preprocessor
from target_utils import as_target_array, as_target_matrix
from torch_exact_krr import CompositeTorchKRR


class CompositeKRREstimator(BaseEstimator, RegressorMixin):
    def __init__(
        self,
        *,
        alpha=1.0,
        gammas=None,
        kernel_weights=None,
        kernel_products=None,
        names=None,
        kernel_types=None,
        normalizations=None,
        pca_components=None,
        pca_whiten=False,
        normalize_kernel_weights=False,
        compute_dtype,
        backend="numpy",
        pytorch_device: str | None = "auto",
        pytorch_predict_batch_size: int = 2048,
        linear_baseline=None,
    ):
        self.alpha = alpha
        self.gammas = gammas
        self.kernel_weights = kernel_weights
        self.kernel_products = kernel_products
        self.names = names
        self.kernel_types = kernel_types
        self.normalizations = normalizations
        self.pca_components = pca_components
        self.pca_whiten = pca_whiten
        self.normalize_kernel_weights = normalize_kernel_weights
        self.compute_dtype = compute_dtype
        self.backend = backend
        self.pytorch_device = pytorch_device
        self.pytorch_predict_batch_size = pytorch_predict_batch_size
        self.linear_baseline = linear_baseline

    def fit(self, X, y):
        compute_dtype = np.dtype(self.compute_dtype)
        backend = normalize_compute_backend(self.backend, name="cached scoring backend")
        if (
            type(self.pytorch_predict_batch_size) is not int
            or self.pytorch_predict_batch_size <= 0
        ):
            raise ValueError("pytorch_predict_batch_size must be a positive int.")
        X_blocks = unpack_sample_matrix(X, dtype=self.compute_dtype)
        y_array = as_target_array(y, dtype=compute_dtype)
        y_matrix = as_target_matrix(y_array, dtype=compute_dtype)

        n_blocks = len(X_blocks)
        if y_matrix.shape[0] != X_blocks[0].shape[0]:
            raise ValueError(
                f"X has {X_blocks[0].shape[0]} samples, but y has "
                f"{y_matrix.shape[0]}."
            )

        names = resolve_sequence("names", self.names, n_blocks, "desc")
        self.linear_baseline_ = None
        if self.linear_baseline is not None:
            if self.linear_baseline not in names:
                raise ValueError("linear_baseline must name a descriptor block")
            self.linear_baseline_index_ = names.index(self.linear_baseline)
            self.linear_baseline_ = fit_linear_baseline(X_blocks[self.linear_baseline_index_], y_matrix)
            residual = y_matrix - predict_linear_baseline(X_blocks[self.linear_baseline_index_], self.linear_baseline_)
            y_array = residual[:, 0] if y_array.ndim == 1 else residual
        kernel_types = resolve_sequence(
            "kernel_types", self.kernel_types, n_blocks, "rbf"
        )
        normalizations = resolve_sequence(
            "normalizations", self.normalizations, n_blocks, "none"
        )
        pca_components = resolve_sequence(
            "pca_components", self.pca_components, n_blocks, None
        )
        pca_whiten = resolve_sequence(
            "pca_whiten", self.pca_whiten, n_blocks, False
        )
        products = resolve_kernel_products(self.kernel_products, n_blocks)
        _, gammas, weights = resolve_kernel_hyperparameters(self, n_components=n_blocks)

        self.X_preprocessors_ = []
        X_blocks_t = []

        for X_block, normalization, pca_component, pca_whiten_block in zip(
            X_blocks,
            normalizations,
            pca_components,
            pca_whiten,
        ):
            preprocessor = make_data_preprocessor(
                normalization,
                pca_components=pca_component,
                pca_whiten=pca_whiten_block,
            )
            X_block_t = np.asarray(
                preprocessor.fit_transform(X_block),
                dtype=compute_dtype,
            )

            self.X_preprocessors_.append(preprocessor)
            X_blocks_t.append(X_block_t)

        components = [
            KernelComponent(
                name=name,
                gamma=gamma,
                kernel_weight=weight,
                kernel_type=kernel_type,
            )
            for name, gamma, weight, kernel_type in zip(
                names, gammas, weights, kernel_types
            )
        ]

        if backend == "pytorch":
            self.model_ = CompositeTorchKRR(
                components=components,
                kernel_products=products,
                product_weights=weights[n_blocks:],
                alpha=self.alpha,
                dtype=compute_dtype,
                device=self.pytorch_device,
                predict_batch_size=self.pytorch_predict_batch_size,
            )
        else:
            self.model_ = CompositeKRR(
                components=components,
                kernel_products=products,
                product_weights=weights[n_blocks:],
                alpha=self.alpha,
                dtype=compute_dtype,
            )
        self.model_.fit(X_blocks_t, y_array)
        self.transformed_feature_counts_ = [int(b.shape[1]) for b in X_blocks_t]
        self.n_features_in_ = n_blocks
        self.names_ = names
        self.kernel_types_ = kernel_types
        self.kernel_products_ = products
        self.normalizations_ = normalizations
        self.pca_components_ = pca_components
        self.pca_whiten_ = pca_whiten
        self.gammas_ = list(np.asarray(gammas, dtype=float))
        self.kernel_weights_ = list(weights)
        self.compute_dtype_ = compute_dtype
        self.backend_ = backend

        return self

    def predict(self, X):
        check_is_fitted(self, "model_")

        X_blocks = unpack_sample_matrix(X, dtype=self.compute_dtype)
        if len(X_blocks) != len(self.X_preprocessors_):
            raise ValueError(
                f"Expected {len(self.X_preprocessors_)} descriptor blocks, "
                f"got {len(X_blocks)}."
            )

        X_blocks_t = [
            np.asarray(
                preprocessor.transform(X_block),
                dtype=self.compute_dtype_,
            )
            for preprocessor, X_block in zip(self.X_preprocessors_, X_blocks)
        ]

        prediction = self.model_.predict(X_blocks_t)
        if self.linear_baseline_ is not None:
            baseline = predict_linear_baseline(X_blocks[self.linear_baseline_index_], self.linear_baseline_)
            prediction = prediction + (baseline[:, 0] if prediction.ndim == 1 else baseline)
        return prediction
