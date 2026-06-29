from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import logging

import numpy as np
from numpy.typing import NDArray
from sklearn.base import BaseEstimator, RegressorMixin

from kernel_cache import distance_spec_for_kernel, resolve_sequence, unpack_sample_matrix
from preprocess import make_data_preprocessor
from target_utils import as_target_array, as_target_matrix, maybe_squeeze_single_target


logger = logging.getLogger("torch-cg-krr")


class CompositeTorchCGKRREstimator(BaseEstimator, RegressorMixin):
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
        pytorch_device: str | None = "auto",
        pytorch_devices="auto",
        cg_tol: float = 1e-6,
        cg_max_iter: int = 1000,
        cg_block_size: int = 2048,
        cg_log_interval: int = 25,
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
        self.pytorch_device = pytorch_device
        self.pytorch_devices = pytorch_devices
        self.cg_tol = cg_tol
        self.cg_max_iter = cg_max_iter
        self.cg_block_size = cg_block_size
        self.cg_log_interval = cg_log_interval

    def fit(self, X, y):
        if self.alpha <= 0:
            raise ValueError(f"alpha must be positive, got {self.alpha}.")
        if self.cg_tol <= 0:
            raise ValueError(f"cg_tol must be positive, got {self.cg_tol}.")
        if type(self.cg_max_iter) is not int or self.cg_max_iter <= 0:
            raise ValueError("cg_max_iter must be a positive int.")
        if type(self.cg_block_size) is not int or self.cg_block_size <= 0:
            raise ValueError("cg_block_size must be a positive int.")
        if type(self.cg_log_interval) is not int or self.cg_log_interval < 0:
            raise ValueError("cg_log_interval must be a non-negative int.")

        torch, torch_dtype, devices = _resolve_torch_runtime(
            compute_dtype=self.compute_dtype,
            pytorch_device=self.pytorch_device,
            pytorch_devices=self.pytorch_devices,
        )
        compute_dtype = np.dtype(self.compute_dtype)
        if not np.issubdtype(compute_dtype, np.floating):
            raise ValueError("compute_dtype must be a floating dtype.")

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
        kernel_types = [
            _normalize_supported_kernel_type(kernel_type)
            for kernel_type in resolve_sequence(
                "kernel_types",
                self.kernel_types,
                n_blocks,
                "rbf",
            )
        ]
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
        gammas = np.asarray(
            resolve_sequence("gammas", self.gammas, n_blocks, 1.0),
            dtype=float,
        )
        weights = np.asarray(
            resolve_sequence("kernel_weights", self.kernel_weights, n_blocks, 1.0),
            dtype=float,
        )
        if np.any(gammas < 0):
            raise ValueError("All gammas must be non-negative.")
        if np.any(weights < 0):
            raise ValueError("All kernel weights must be non-negative.")
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

        operator = _TorchCompositeKernelOperator(
            X_train_blocks=X_blocks_t,
            gammas=gammas,
            weights=weights,
            kernel_types=kernel_types,
            torch=torch,
            torch_dtype=torch_dtype,
            devices=devices,
            block_size=self.cg_block_size,
        )
        y_cpu = torch.as_tensor(y_matrix, dtype=torch_dtype, device="cpu")
        dual_coef_cpu, info = _conjugate_gradient_solve(
            lambda vector: operator.train_matmul(vector)
            + float(self.alpha) * vector,
            y_cpu,
            tol=float(self.cg_tol),
            max_iter=int(self.cg_max_iter),
            log_interval=int(self.cg_log_interval),
        )
        if not info.converged:
            logger.warning(
                "CG did not converge within %s iterations; max relative residual "
                "is %.6g.",
                info.n_iter,
                info.max_relative_residual,
            )

        self._operator_ = operator
        self.dual_coef_ = dual_coef_cpu.numpy().astype(compute_dtype, copy=False)
        self.target_was_1d_ = y_array.ndim == 1
        self.names_ = names
        self.kernel_types_ = kernel_types
        self.normalizations_ = normalizations
        self.pca_components_ = pca_components
        self.pca_whiten_ = pca_whiten
        self.gammas_ = gammas.tolist()
        self.kernel_weights_ = weights.tolist()
        self.compute_dtype_ = compute_dtype
        self.torch_dtype_ = torch_dtype
        self.devices_ = [str(device) for device in devices]
        self.n_features_in_ = n_blocks
        self.cg_n_iter_ = info.n_iter
        self.cg_converged_ = info.converged
        self.cg_relative_residual_ = info.relative_residuals
        self.cg_max_relative_residual_ = info.max_relative_residual

        return self

    def predict(self, X):
        if not hasattr(self, "dual_coef_"):
            raise ValueError("CompositeTorchCGKRREstimator has not been fitted.")

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

        torch = self._operator_.torch
        coef_cpu = torch.as_tensor(
            self.dual_coef_,
            dtype=self.torch_dtype_,
            device="cpu",
        )
        y_pred_cpu = self._operator_.cross_matmul(X_blocks_t, coef_cpu)
        y_pred = y_pred_cpu.numpy().astype(self.compute_dtype_, copy=False)
        return maybe_squeeze_single_target(y_pred, squeeze=self.target_was_1d_)


def is_torch_cg_regressor(regressor) -> bool:
    return regressor.__class__.__name__ == "CompositeTorchCGKRREstimator"


@dataclass(frozen=True)
class _CGInfo:
    n_iter: int
    converged: bool
    relative_residuals: list[float]
    max_relative_residual: float


class _TorchCompositeKernelOperator:
    def __init__(
        self,
        *,
        X_train_blocks: list[NDArray],
        gammas: NDArray,
        weights: NDArray,
        kernel_types: list[str],
        torch,
        torch_dtype,
        devices,
        block_size: int,
    ):
        if not X_train_blocks:
            raise ValueError("At least one descriptor block is required.")
        if not (
            len(X_train_blocks)
            == len(gammas)
            == len(weights)
            == len(kernel_types)
        ):
            raise ValueError(
                "Descriptor blocks, gammas, weights, and kernel_types must "
                "have matching lengths."
            )

        self.torch = torch
        self.torch_dtype = torch_dtype
        self.devices = list(devices)
        self.block_size = int(block_size)
        self.gammas = [float(gamma) for gamma in gammas]
        self.weights = [float(weight) for weight in weights]
        self.kernel_types = list(kernel_types)
        self.n_train = int(X_train_blocks[0].shape[0])
        self.n_components = len(X_train_blocks)

        self.train_blocks_by_device = [
            [
                torch.as_tensor(block, dtype=torch_dtype, device=device)
                for block in X_train_blocks
            ]
            for device in self.devices
        ]
        self.row_shards = _split_rows(self.n_train, len(self.devices))

    def train_matmul(self, vector_cpu):
        if vector_cpu.ndim == 1:
            vector_cpu = vector_cpu.reshape(-1, 1)
        if vector_cpu.shape[0] != self.n_train:
            raise ValueError(
                f"Expected vector with {self.n_train} rows, got "
                f"{vector_cpu.shape[0]}."
            )
        return self._matmul(
            left_blocks_cpu=None,
            n_left=self.n_train,
            vector_cpu=vector_cpu,
            left_is_train=True,
        )

    def cross_matmul(self, X_left_blocks: list[NDArray], vector_cpu):
        if vector_cpu.ndim == 1:
            vector_cpu = vector_cpu.reshape(-1, 1)
        if vector_cpu.shape[0] != self.n_train:
            raise ValueError(
                f"Expected vector with {self.n_train} rows, got "
                f"{vector_cpu.shape[0]}."
            )
        if len(X_left_blocks) != self.n_components:
            raise ValueError(
                f"Expected {self.n_components} descriptor blocks, "
                f"got {len(X_left_blocks)}."
            )
        n_left = int(X_left_blocks[0].shape[0])
        for block in X_left_blocks:
            if block.shape[0] != n_left:
                raise ValueError("All descriptor blocks must have the same row count.")

        return self._matmul(
            left_blocks_cpu=X_left_blocks,
            n_left=n_left,
            vector_cpu=vector_cpu,
            left_is_train=False,
        )

    def _matmul(
        self,
        *,
        left_blocks_cpu,
        n_left: int,
        vector_cpu,
        left_is_train: bool,
    ):
        row_shards = _split_rows(n_left, len(self.devices))
        out_cpu = self.torch.empty(
            (n_left, vector_cpu.shape[1]),
            dtype=self.torch_dtype,
            device="cpu",
        )
        tasks = [
            (device_index, row_start, row_stop)
            for device_index, (row_start, row_stop) in enumerate(row_shards)
            if row_stop > row_start
        ]

        if len(tasks) == 1:
            results = [
                self._compute_shard(
                    device_index=tasks[0][0],
                    row_start=tasks[0][1],
                    row_stop=tasks[0][2],
                    left_blocks_cpu=left_blocks_cpu,
                    vector_cpu=vector_cpu,
                    left_is_train=left_is_train,
                )
            ]
        else:
            with ThreadPoolExecutor(max_workers=len(tasks)) as executor:
                futures = [
                    executor.submit(
                        self._compute_shard,
                        device_index=device_index,
                        row_start=row_start,
                        row_stop=row_stop,
                        left_blocks_cpu=left_blocks_cpu,
                        vector_cpu=vector_cpu,
                        left_is_train=left_is_train,
                    )
                    for device_index, row_start, row_stop in tasks
                ]
                results = [future.result() for future in futures]

        for row_start, row_stop, shard_cpu in results:
            out_cpu[row_start:row_stop] = shard_cpu
        return out_cpu

    def _compute_shard(
        self,
        *,
        device_index: int,
        row_start: int,
        row_stop: int,
        left_blocks_cpu,
        vector_cpu,
        left_is_train: bool,
    ):
        torch = self.torch
        device = self.devices[device_index]
        if getattr(device, "type", None) == "cuda":
            torch.cuda.set_device(device)

        train_blocks = self.train_blocks_by_device[device_index]
        vector_device = vector_cpu.to(device=device, dtype=self.torch_dtype)
        out_device = torch.empty(
            (row_stop - row_start, vector_cpu.shape[1]),
            dtype=self.torch_dtype,
            device=device,
        )

        with torch.no_grad():
            for block_start in range(row_start, row_stop, self.block_size):
                block_stop = min(block_start + self.block_size, row_stop)
                if left_is_train:
                    left_blocks = [
                        train_block[block_start:block_stop]
                        for train_block in train_blocks
                    ]
                else:
                    left_blocks = [
                        torch.as_tensor(
                            block[block_start:block_stop],
                            dtype=self.torch_dtype,
                            device=device,
                        )
                        for block in left_blocks_cpu
                    ]

                block_out = torch.zeros(
                    (block_stop - block_start, vector_cpu.shape[1]),
                    dtype=self.torch_dtype,
                    device=device,
                )
                for col_start in range(0, self.n_train, self.block_size):
                    col_stop = min(col_start + self.block_size, self.n_train)
                    vector_block = vector_device[col_start:col_stop]
                    for left_block, right_train_block, gamma, weight, kernel_type in zip(
                        left_blocks,
                        train_blocks,
                        self.gammas,
                        self.weights,
                        self.kernel_types,
                    ):
                        kernel_block = _kernel_component(
                            left_block,
                            right_train_block[col_start:col_stop],
                            gamma=gamma,
                            weight=weight,
                            kernel_type=kernel_type,
                            torch=torch,
                        )
                        block_out += kernel_block @ vector_block

                local_start = block_start - row_start
                local_stop = block_stop - row_start
                out_device[local_start:local_stop] = block_out

        return row_start, row_stop, out_device.cpu()


def _conjugate_gradient_solve(
    matvec,
    rhs,
    *,
    tol: float,
    max_iter: int,
    log_interval: int,
) -> tuple[object, _CGInfo]:
    x = rhs.new_zeros(rhs.shape)
    residual = rhs.clone()
    direction = residual.clone()
    residual_sq = _column_dot(residual, residual)
    rhs_norm = residual_sq.sqrt()
    normalizer = rhs_norm.clamp_min(1.0)
    relative = rhs_norm / normalizer
    max_relative = float(relative.max().item())
    if max_relative <= tol:
        return x, _CGInfo(
            n_iter=0,
            converged=True,
            relative_residuals=relative.detach().cpu().numpy().tolist(),
            max_relative_residual=max_relative,
        )

    active = relative > tol
    for iteration in range(1, max_iter + 1):
        operator_direction = matvec(direction)
        direction_operator_direction = _column_dot(direction, operator_direction)
        if bool((direction_operator_direction[active] <= 0).any().item()):
            raise np.linalg.LinAlgError(
                "CG encountered a non-positive search direction curvature."
            )

        step = residual_sq.new_zeros(residual_sq.shape)
        step[active] = residual_sq[active] / direction_operator_direction[active]
        step = step.reshape(1, -1)
        x = x + direction * step
        residual = residual - operator_direction * step

        new_residual_sq = _column_dot(residual, residual)
        relative = new_residual_sq.sqrt() / normalizer
        max_relative = float(relative.max().item())
        if log_interval and (
            iteration == 1
            or iteration % log_interval == 0
            or max_relative <= tol
        ):
            logger.info(
                "CG iteration %s: max relative residual %.6g.",
                iteration,
                max_relative,
            )
        if max_relative <= tol:
            return x, _CGInfo(
                n_iter=iteration,
                converged=True,
                relative_residuals=relative.detach().cpu().numpy().tolist(),
                max_relative_residual=max_relative,
            )

        new_active = relative > tol
        beta_values = residual_sq.new_zeros(residual_sq.shape)
        beta_values[new_active] = (
            new_residual_sq[new_active] / residual_sq[new_active]
        )
        beta = beta_values.reshape(1, -1)
        direction = residual + direction * beta
        direction[:, ~new_active] = 0
        residual_sq = new_residual_sq
        active = new_active

    return x, _CGInfo(
        n_iter=max_iter,
        converged=False,
        relative_residuals=relative.detach().cpu().numpy().tolist(),
        max_relative_residual=max_relative,
    )


def _column_dot(left, right):
    return (left * right).sum(dim=0)


def _kernel_component(
    left,
    right,
    *,
    gamma: float,
    weight: float,
    kernel_type: str,
    torch,
):
    if kernel_type == "rbf":
        distances = _squared_euclidean_distance(left, right, torch=torch)
    elif kernel_type == "laplacian":
        distances = torch.cdist(left, right, p=1)
    else:
        raise ValueError(f'Unsupported CG kernel type "{kernel_type}".')

    kernel = torch.exp(-float(gamma) * distances)
    if weight != 1.0:
        kernel = kernel * float(weight)
    return kernel


def _squared_euclidean_distance(left, right, *, torch):
    left_norms = torch.sum(left * left, dim=1, keepdim=True)
    right_norms = torch.sum(right * right, dim=1).reshape(1, -1)
    distances = left_norms + right_norms - 2.0 * (left @ right.T)
    return torch.clamp(distances, min=0.0)


def _normalize_supported_kernel_type(kernel_type: str) -> str:
    spec = distance_spec_for_kernel(kernel_type)
    return spec.kernel_type


def _resolve_torch_runtime(
    *,
    compute_dtype,
    pytorch_device: str | None,
    pytorch_devices,
):
    from pytorch_backend import require_torch, resolve_torch_devices, resolve_torch_dtype

    torch = require_torch()
    torch_dtype = resolve_torch_dtype(torch, np.dtype(compute_dtype))
    devices = resolve_torch_devices(
        torch,
        pytorch_devices,
        fallback_device=pytorch_device,
    )
    return torch, torch_dtype, devices


def _split_rows(n_rows: int, n_parts: int) -> list[tuple[int, int]]:
    if n_parts <= 0:
        raise ValueError("n_parts must be positive.")
    n_active = min(int(n_rows), int(n_parts)) if n_rows > 0 else 1
    boundaries = np.linspace(0, n_rows, n_active + 1, dtype=int)
    shards = [
        (int(boundaries[index]), int(boundaries[index + 1]))
        for index in range(n_active)
    ]
    return shards
