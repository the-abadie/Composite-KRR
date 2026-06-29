import logging

import numpy as np
from numpy.typing import NDArray
from sklearn.base import BaseEstimator, RegressorMixin

from kernel_cache import (
    distance_spec_for_kernel,
    normalize_distance_backend,
    resolve_sequence,
    unpack_sample_matrix,
)
from kernels import (
    pairwise_cross_lp_distance,
    pairwise_cross_lp_distance_pytorch,
    pairwise_self_lp_distance,
    pairwise_self_lp_distance_pytorch,
)
from preprocess import make_data_preprocessor
from target_utils import as_target_array, as_target_matrix, maybe_squeeze_single_target

logger = logging.getLogger("nystrom-krr")


class CompositeNystromKRREstimator(BaseEstimator, RegressorMixin):
    def __init__(
        self,
        *,
        alpha=1.0,
        gammas=None,
        kernel_weights=None,
        names=None,
        kernel_types=None,
        normalizations=None,
        pca_components=None,
        pca_whiten=False,
        normalize_kernel_weights=False,
        compute_dtype="float64",
        n_landmarks=2048,
        landmark_selection="random",
        random_state=None,
        backend="numpy",
        pytorch_device="auto",
        batch_size=2048,
        eigenvalue_floor=1e-12,
    ):
        self.alpha = alpha
        self.gammas = gammas
        self.kernel_weights = kernel_weights
        self.names = names
        self.kernel_types = kernel_types
        self.normalizations = normalizations
        self.pca_components = pca_components
        self.pca_whiten = pca_whiten
        self.normalize_kernel_weights = normalize_kernel_weights
        self.compute_dtype = compute_dtype
        self.n_landmarks = n_landmarks
        self.landmark_selection = landmark_selection
        self.random_state = random_state
        self.backend = backend
        self.pytorch_device = pytorch_device
        self.batch_size = batch_size
        self.eigenvalue_floor = eigenvalue_floor

    def fit(self, X, y):
        if self.alpha <= 0:
            raise ValueError(f"alpha must be positive, got {self.alpha}.")

        compute_dtype = np.dtype(self.compute_dtype)
        if not np.issubdtype(compute_dtype, np.floating):
            raise ValueError("compute_dtype must be a floating dtype.")
        if type(self.n_landmarks) is not int or self.n_landmarks <= 0:
            raise ValueError("n_landmarks must be a positive int.")
        if type(self.batch_size) is not int or self.batch_size <= 0:
            raise ValueError("batch_size must be a positive int.")
        if self.eigenvalue_floor < 0:
            raise ValueError("eigenvalue_floor must be non-negative.")

        backend = normalize_distance_backend(self.backend)
        X_blocks = unpack_sample_matrix(X, dtype=compute_dtype)
        y_array = as_target_array(y, dtype=compute_dtype)
        y_matrix = as_target_matrix(y_array, dtype=compute_dtype)
        n_samples = y_matrix.shape[0]
        n_blocks = len(X_blocks)
        if X_blocks[0].shape[0] != n_samples:
            raise ValueError(
                f"X has {X_blocks[0].shape[0]} samples, but y has {n_samples}."
            )

        names = resolve_sequence("names", self.names, n_blocks, "desc")
        kernel_types = resolve_sequence("kernel_types", self.kernel_types, n_blocks, "rbf")
        normalizations = resolve_sequence(
            "normalizations",
            self.normalizations,
            n_blocks,
            "none",
        )
        pca_components = resolve_sequence(
            "pca_components",
            self.pca_components,
            n_blocks,
            None,
        )
        pca_whiten = resolve_sequence("pca_whiten", self.pca_whiten, n_blocks, False)
        gammas = resolve_sequence("gammas", self.gammas, n_blocks, 1.0)
        weights = resolve_sequence("kernel_weights", self.kernel_weights, n_blocks, 1.0)

        weights = np.asarray(weights, dtype=float)
        if self.normalize_kernel_weights:
            weight_sum = float(np.sum(weights))
            if weight_sum <= 0 or not np.isfinite(weight_sum):
                raise ValueError(
                    "Cannot normalize kernel weights with non-positive sum."
                )
            weights = weights / weight_sum

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

        landmark_indices = select_landmark_indices(
            n_samples,
            n_landmarks=self.n_landmarks,
            selection=self.landmark_selection,
            random_state=self.random_state,
        )
        landmark_blocks = [X_block[landmark_indices] for X_block in X_blocks_t]

        if backend == "numpy":
            normalizer, beta = _fit_nystrom_numpy(
                X_blocks_t,
                landmark_blocks,
                y_matrix,
                alpha=float(self.alpha),
                gammas=np.asarray(gammas, dtype=float),
                weights=np.asarray(weights, dtype=float),
                kernel_types=list(kernel_types),
                batch_size=self.batch_size,
                dtype=compute_dtype,
                eigenvalue_floor=float(self.eigenvalue_floor),
            )
        else:
            normalizer, beta = _fit_nystrom_pytorch(
                X_blocks_t,
                landmark_blocks,
                y_matrix,
                alpha=float(self.alpha),
                gammas=np.asarray(gammas, dtype=float),
                weights=np.asarray(weights, dtype=float),
                kernel_types=list(kernel_types),
                batch_size=self.batch_size,
                dtype=compute_dtype,
                eigenvalue_floor=float(self.eigenvalue_floor),
                device=self.pytorch_device,
            )

        self.landmark_blocks_ = landmark_blocks
        self.landmark_indices_ = landmark_indices
        self.nystrom_normalizer_ = normalizer
        self.coef_ = beta
        self.target_was_1d_ = y_array.ndim == 1
        self.names_ = names
        self.kernel_types_ = list(kernel_types)
        self.normalizations_ = normalizations
        self.pca_components_ = pca_components
        self.pca_whiten_ = pca_whiten
        self.gammas_ = list(np.asarray(gammas, dtype=float))
        self.kernel_weights_ = list(np.asarray(weights, dtype=float))
        self.compute_dtype_ = compute_dtype
        self.backend_ = backend
        self.n_features_in_ = n_blocks
        self.n_landmarks_ = len(landmark_indices)
        self.n_nystrom_features_ = int(normalizer.shape[1])

        return self

    def predict(self, X):
        if not hasattr(self, "coef_"):
            raise ValueError("CompositeNystromKRREstimator has not been fitted.")

        X_blocks = unpack_sample_matrix(X, dtype=self.compute_dtype_)
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

        if self.backend_ == "numpy":
            y_pred = _predict_nystrom_numpy(
                X_blocks_t,
                self.landmark_blocks_,
                self.nystrom_normalizer_,
                self.coef_,
                gammas=np.asarray(self.gammas_, dtype=float),
                weights=np.asarray(self.kernel_weights_, dtype=float),
                kernel_types=self.kernel_types_,
                batch_size=self.batch_size,
                dtype=self.compute_dtype_,
            )
        else:
            y_pred = _predict_nystrom_pytorch(
                X_blocks_t,
                self.landmark_blocks_,
                self.nystrom_normalizer_,
                self.coef_,
                gammas=np.asarray(self.gammas_, dtype=float),
                weights=np.asarray(self.kernel_weights_, dtype=float),
                kernel_types=self.kernel_types_,
                batch_size=self.batch_size,
                dtype=self.compute_dtype_,
                device=self.pytorch_device,
            )

        return maybe_squeeze_single_target(y_pred, squeeze=self.target_was_1d_)


def select_landmark_indices(
    n_samples: int,
    *,
    n_landmarks: int,
    selection: str,
    random_state=None,
) -> NDArray:
    if n_samples <= 0:
        raise ValueError("Cannot select landmarks from zero samples.")
    n_landmarks = min(int(n_landmarks), int(n_samples))
    selection = str(selection).lower()

    if selection == "first":
        return np.arange(n_landmarks, dtype=int)
    if selection == "random":
        rng = np.random.default_rng(random_state)
        return np.sort(rng.choice(n_samples, size=n_landmarks, replace=False))

    raise ValueError(
        'landmark_selection must be "random" or "first", '
        f"got {selection!r}."
    )


def _fit_nystrom_numpy(
    X_blocks,
    landmark_blocks,
    y_matrix,
    *,
    alpha: float,
    gammas,
    weights,
    kernel_types: list[str],
    batch_size: int,
    dtype,
    eigenvalue_floor: float,
) -> tuple[NDArray, NDArray]:
    W = _composite_self_kernel_numpy(
        landmark_blocks,
        gammas=gammas,
        weights=weights,
        kernel_types=kernel_types,
        dtype=dtype,
    )
    normalizer = _nystrom_normalizer_numpy(
        W,
        eigenvalue_floor=eigenvalue_floor,
        dtype=dtype,
    )
    rank = normalizer.shape[1]
    A = np.eye(rank, dtype=dtype) * alpha
    B = np.zeros((rank, y_matrix.shape[1]), dtype=dtype)

    for start, stop in _batch_slices(y_matrix.shape[0], batch_size):
        C = _composite_cross_kernel_numpy(
            [X_block[start:stop] for X_block in X_blocks],
            landmark_blocks,
            gammas=gammas,
            weights=weights,
            kernel_types=kernel_types,
            dtype=dtype,
        )
        Phi = C @ normalizer
        A += Phi.T @ Phi
        B += Phi.T @ y_matrix[start:stop]

    beta = np.linalg.solve(A, B)
    return normalizer, beta


def _predict_nystrom_numpy(
    X_blocks,
    landmark_blocks,
    normalizer,
    beta,
    *,
    gammas,
    weights,
    kernel_types: list[str],
    batch_size: int,
    dtype,
) -> NDArray:
    n_samples = X_blocks[0].shape[0]
    y_pred = np.empty((n_samples, beta.shape[1]), dtype=dtype)
    for start, stop in _batch_slices(n_samples, batch_size):
        C = _composite_cross_kernel_numpy(
            [X_block[start:stop] for X_block in X_blocks],
            landmark_blocks,
            gammas=gammas,
            weights=weights,
            kernel_types=kernel_types,
            dtype=dtype,
        )
        y_pred[start:stop] = (C @ normalizer) @ beta

    return y_pred


def _fit_nystrom_pytorch(
    X_blocks,
    landmark_blocks,
    y_matrix,
    *,
    alpha: float,
    gammas,
    weights,
    kernel_types: list[str],
    batch_size: int,
    dtype,
    eigenvalue_floor: float,
    device,
) -> tuple[NDArray, NDArray]:
    torch, torch_dtype, torch_device = _resolve_torch(dtype=dtype, device=device)
    landmark_tensors = [
        torch.as_tensor(block, dtype=torch_dtype, device=torch_device)
        for block in landmark_blocks
    ]
    gammas_t = torch.as_tensor(gammas, dtype=torch_dtype, device=torch_device)
    weights_t = torch.as_tensor(weights, dtype=torch_dtype, device=torch_device)
    with torch.no_grad():
        W = _composite_self_kernel_pytorch(
            landmark_tensors,
            gammas=gammas_t,
            weights=weights_t,
            kernel_types=kernel_types,
            dtype=torch_dtype,
            device=torch_device,
        )
        normalizer_t = _nystrom_normalizer_pytorch(
            W,
            eigenvalue_floor=eigenvalue_floor,
        )
        rank = int(normalizer_t.shape[1])
        A = torch.eye(rank, dtype=torch_dtype, device=torch_device) * alpha
        B = torch.zeros(
            (rank, y_matrix.shape[1]),
            dtype=torch_dtype,
            device=torch_device,
        )
        y_t = torch.as_tensor(y_matrix, dtype=torch_dtype, device=torch_device)

        for start, stop in _batch_slices(y_matrix.shape[0], batch_size):
            left_tensors = [
                torch.as_tensor(
                    X_block[start:stop],
                    dtype=torch_dtype,
                    device=torch_device,
                )
                for X_block in X_blocks
            ]
            C = _composite_cross_kernel_pytorch(
                left_tensors,
                landmark_tensors,
                gammas=gammas_t,
                weights=weights_t,
                kernel_types=kernel_types,
                dtype=torch_dtype,
                device=torch_device,
            )
            Phi = C @ normalizer_t
            A += Phi.T @ Phi
            B += Phi.T @ y_t[start:stop]

        beta_t = torch.linalg.solve(A, B)
        normalizer = normalizer_t.detach().cpu().numpy().astype(dtype, copy=False)
        beta = beta_t.detach().cpu().numpy().astype(dtype, copy=False)

    return normalizer, beta


def _predict_nystrom_pytorch(
    X_blocks,
    landmark_blocks,
    normalizer,
    beta,
    *,
    gammas,
    weights,
    kernel_types: list[str],
    batch_size: int,
    dtype,
    device,
) -> NDArray:
    torch, torch_dtype, torch_device = _resolve_torch(dtype=dtype, device=device)
    landmark_tensors = [
        torch.as_tensor(block, dtype=torch_dtype, device=torch_device)
        for block in landmark_blocks
    ]
    normalizer_t = torch.as_tensor(normalizer, dtype=torch_dtype, device=torch_device)
    beta_t = torch.as_tensor(beta, dtype=torch_dtype, device=torch_device)
    gammas_t = torch.as_tensor(gammas, dtype=torch_dtype, device=torch_device)
    weights_t = torch.as_tensor(weights, dtype=torch_dtype, device=torch_device)
    n_samples = X_blocks[0].shape[0]
    y_pred = np.empty((n_samples, beta.shape[1]), dtype=dtype)

    with torch.no_grad():
        for start, stop in _batch_slices(n_samples, batch_size):
            left_tensors = [
                torch.as_tensor(
                    X_block[start:stop],
                    dtype=torch_dtype,
                    device=torch_device,
                )
                for X_block in X_blocks
            ]
            C = _composite_cross_kernel_pytorch(
                left_tensors,
                landmark_tensors,
                gammas=gammas_t,
                weights=weights_t,
                kernel_types=kernel_types,
                dtype=torch_dtype,
                device=torch_device,
            )
            pred_t = (C @ normalizer_t) @ beta_t
            y_pred[start:stop] = pred_t.detach().cpu().numpy()

    return y_pred


def _composite_self_kernel_numpy(
    X_blocks,
    *,
    gammas,
    weights,
    kernel_types: list[str],
    dtype,
) -> NDArray:
    K_total = None
    for X_block, gamma, weight, kernel_type in zip(
        X_blocks,
        gammas,
        weights,
        kernel_types,
    ):
        spec = distance_spec_for_kernel(kernel_type)
        distance = pairwise_self_lp_distance(
            X_block,
            p=spec.p,
            squared=spec.squared,
            dtype=dtype,
        )
        component = _kernel_from_distance_numpy(distance, gamma=gamma, weight=weight)
        K_total = component if K_total is None else K_total + component

    if K_total is None:
        raise ValueError("At least one descriptor block is required.")
    return K_total


def _composite_cross_kernel_numpy(
    X_left_blocks,
    X_right_blocks,
    *,
    gammas,
    weights,
    kernel_types: list[str],
    dtype,
) -> NDArray:
    K_total = None
    for X_left, X_right, gamma, weight, kernel_type in zip(
        X_left_blocks,
        X_right_blocks,
        gammas,
        weights,
        kernel_types,
    ):
        spec = distance_spec_for_kernel(kernel_type)
        distance = pairwise_cross_lp_distance(
            X_left,
            X_right,
            p=spec.p,
            squared=spec.squared,
            dtype=dtype,
        )
        component = _kernel_from_distance_numpy(distance, gamma=gamma, weight=weight)
        K_total = component if K_total is None else K_total + component

    if K_total is None:
        raise ValueError("At least one descriptor block is required.")
    return K_total


def _kernel_from_distance_numpy(distance, *, gamma: float, weight: float) -> NDArray:
    component = np.asarray(distance)
    np.multiply(component, -float(gamma), out=component, casting="unsafe")
    np.exp(component, out=component)
    component *= float(weight)
    return component


def _nystrom_normalizer_numpy(
    W,
    *,
    eigenvalue_floor: float,
    dtype,
) -> NDArray:
    W = 0.5 * (W + W.T)
    eigenvalues, eigenvectors = np.linalg.eigh(W)
    keep = _eigenvalue_keep_mask(eigenvalues, eigenvalue_floor=eigenvalue_floor)
    if not np.any(keep):
        raise np.linalg.LinAlgError("Nyström landmark kernel has no positive modes.")

    eigenvalues = eigenvalues[keep]
    eigenvectors = eigenvectors[:, keep]
    return np.asarray(eigenvectors / np.sqrt(eigenvalues)[None, :], dtype=dtype)


def _eigenvalue_keep_mask(eigenvalues, *, eigenvalue_floor: float) -> NDArray:
    eigenvalues = np.asarray(eigenvalues, dtype=float)
    positive = eigenvalues[eigenvalues > 0]
    if positive.size == 0:
        return np.zeros(eigenvalues.shape, dtype=bool)

    threshold = max(float(np.max(positive)) * eigenvalue_floor, 0.0)
    return eigenvalues > threshold


def _composite_self_kernel_pytorch(
    X_blocks,
    *,
    gammas,
    weights,
    kernel_types: list[str],
    dtype,
    device,
):
    K_total = None
    for X_block, gamma, weight, kernel_type in zip(
        X_blocks,
        gammas,
        weights,
        kernel_types,
    ):
        spec = distance_spec_for_kernel(kernel_type)
        distance = pairwise_self_lp_distance_pytorch(
            X_block,
            p=spec.p,
            squared=spec.squared,
            dtype=dtype,
            device=str(device),
        )
        component = _kernel_from_distance_pytorch(
            distance,
            gamma=gamma,
            weight=weight,
        )
        K_total = component if K_total is None else K_total + component

    if K_total is None:
        raise ValueError("At least one descriptor block is required.")
    return K_total


def _composite_cross_kernel_pytorch(
    X_left_blocks,
    X_right_blocks,
    *,
    gammas,
    weights,
    kernel_types: list[str],
    dtype,
    device,
):
    K_total = None
    for X_left, X_right, gamma, weight, kernel_type in zip(
        X_left_blocks,
        X_right_blocks,
        gammas,
        weights,
        kernel_types,
    ):
        spec = distance_spec_for_kernel(kernel_type)
        distance = pairwise_cross_lp_distance_pytorch(
            X_left,
            X_right,
            p=spec.p,
            squared=spec.squared,
            dtype=dtype,
            device=str(device),
        )
        component = _kernel_from_distance_pytorch(
            distance,
            gamma=gamma,
            weight=weight,
        )
        K_total = component if K_total is None else K_total + component

    if K_total is None:
        raise ValueError("At least one descriptor block is required.")
    return K_total


def _kernel_from_distance_pytorch(distance, *, gamma, weight):
    return torch_exp_scaled(distance, gamma=gamma, weight=weight)


def torch_exp_scaled(distance, *, gamma, weight):
    return (-gamma * distance).exp() * weight


def _nystrom_normalizer_pytorch(W, *, eigenvalue_floor: float):
    torch = _torch_module()
    W = 0.5 * (W + W.T)
    eigenvalues, eigenvectors = torch.linalg.eigh(W)
    positive = eigenvalues[eigenvalues > 0]
    if positive.numel() == 0:
        raise RuntimeError("Nyström landmark kernel has no positive modes.")

    threshold = torch.max(positive) * float(eigenvalue_floor)
    keep = eigenvalues > threshold
    eigenvalues = eigenvalues[keep]
    eigenvectors = eigenvectors[:, keep]
    return eigenvectors / torch.sqrt(eigenvalues).reshape(1, -1)


def _resolve_torch(*, dtype, device):
    from pytorch_backend import require_torch, resolve_torch_device, resolve_torch_dtype

    torch = require_torch()
    torch_device = resolve_torch_device(torch, device)
    torch_dtype = resolve_torch_dtype(torch, dtype)
    return torch, torch_dtype, torch_device


def _torch_module():
    from pytorch_backend import require_torch

    return require_torch()


def _batch_slices(n_samples: int, batch_size: int):
    for start in range(0, n_samples, batch_size):
        yield start, min(start + batch_size, n_samples)
