from __future__ import annotations

import logging

import numpy as np
from numpy.typing import ArrayLike, NDArray

from class_CompositeKRR import KernelComponent
from kernel_cache import distance_spec_for_kernel
from target_utils import as_target_array, as_target_matrix, maybe_squeeze_single_target


logger = logging.getLogger("torch-exact-krr")


class CompositeTorchKRR:
    def __init__(
        self,
        *,
        components: list[KernelComponent],
        alpha: float,
        dtype: np.dtype | type | str = np.float64,
        device: str | None = "auto",
        predict_batch_size: int = 2048,
    ):
        if alpha <= 0:
            raise ValueError(f"alpha must be positive, got {alpha}.")
        if type(predict_batch_size) is not int or predict_batch_size <= 0:
            raise ValueError("predict_batch_size must be a positive int.")

        self.components = list(components) if components is not None else []
        self.alpha = alpha
        self.dtype = np.dtype(dtype)
        self.device = device
        self.predict_batch_size = predict_batch_size

    def fit(self, X_blocks: list[NDArray], y: ArrayLike):
        torch, torch_dtype, torch_device = _resolve_torch(
            dtype=self.dtype,
            device=self.device,
        )
        y_array = as_target_array(y, dtype=self.dtype)
        y_matrix = as_target_matrix(y_array, dtype=self.dtype)
        n_samples = y_matrix.shape[0]
        if n_samples == 0:
            raise ValueError("Cannot fit CompositeTorchKRR with zero samples.")

        X_train_blocks = self._prepare_blocks(X_blocks, n_samples=n_samples)
        X_train_tensors = [
            torch.as_tensor(block, dtype=torch_dtype, device=torch_device)
            for block in X_train_blocks
        ]
        y_tensor = torch.as_tensor(y_matrix, dtype=torch_dtype, device=torch_device)

        with torch.no_grad():
            K = _composite_kernel_torch(
                X_train_tensors,
                X_train_tensors,
                self.components,
                torch=torch,
            )
            diag = torch.arange(K.shape[0], device=torch_device)
            K[diag, diag] = K[diag, diag] + float(self.alpha)
            factor, info = torch.linalg.cholesky_ex(
                K,
                upper=False,
                check_errors=False,
            )
            if bool(torch.any(info != 0).item()):
                raise np.linalg.LinAlgError(
                    "PyTorch Cholesky factorization failed for final exact KRR fit."
                )
            dual_coef = torch.cholesky_solve(y_tensor, factor, upper=False)
            del K, factor

        self.X_train_tensors_ = X_train_tensors
        self.dual_coef_tensor_ = dual_coef
        self.target_was_1d_ = y_array.ndim == 1
        self.torch_dtype_ = torch_dtype
        self.torch_device_ = torch_device
        self.torch_ = torch

        return self

    def predict(self, X_blocks: list[NDArray]) -> NDArray:
        if not hasattr(self, "dual_coef_tensor_"):
            raise ValueError("CompositeTorchKRR has not been fitted.")

        X_eval_blocks = self._prepare_blocks(X_blocks)
        n_eval = X_eval_blocks[0].shape[0]
        y_pred = np.empty(
            (n_eval, int(self.dual_coef_tensor_.shape[1])),
            dtype=self.dtype,
        )

        torch = self.torch_
        with torch.no_grad():
            for start in range(0, n_eval, self.predict_batch_size):
                stop = min(start + self.predict_batch_size, n_eval)
                X_batch_tensors = [
                    torch.as_tensor(
                        block[start:stop],
                        dtype=self.torch_dtype_,
                        device=self.torch_device_,
                    )
                    for block in X_eval_blocks
                ]
                K_eval = _composite_kernel_torch(
                    X_batch_tensors,
                    self.X_train_tensors_,
                    self.components,
                    torch=torch,
                )
                batch_pred = K_eval @ self.dual_coef_tensor_
                y_pred[start:stop] = batch_pred.detach().cpu().numpy()

        return maybe_squeeze_single_target(y_pred, squeeze=self.target_was_1d_)

    def _prepare_blocks(
        self,
        X_blocks: list[NDArray],
        n_samples: int | None = None,
    ) -> list[NDArray]:
        if X_blocks is None:
            raise ValueError("X_blocks must be a sequence of descriptor blocks.")

        X_blocks = list(X_blocks)
        if len(X_blocks) != len(self.components):
            raise ValueError(
                f"Expected {len(self.components)} descriptor blocks, "
                f"got {len(X_blocks)}."
            )
        if not X_blocks:
            raise ValueError("At least one descriptor block is required.")

        prepared_blocks = []
        expected_n_samples = n_samples
        for component, X in zip(self.components, X_blocks):
            X = np.asarray(X, dtype=self.dtype)
            if X.ndim == 0:
                raise ValueError(
                    f"Descriptor block {component.name} must have a sample axis."
                )
            if expected_n_samples is None:
                expected_n_samples = X.shape[0]
            elif X.shape[0] != expected_n_samples:
                raise ValueError(
                    f"Descriptor block {component.name} has {X.shape[0]} samples, "
                    f"but expected {expected_n_samples}."
                )
            prepared_blocks.append(X.reshape(X.shape[0], -1))

        return prepared_blocks


def _composite_kernel_torch(
    X_left_blocks,
    X_right_blocks,
    components: list[KernelComponent],
    *,
    torch,
):
    K_total = None
    for component, X_left, X_right in zip(
        components,
        X_left_blocks,
        X_right_blocks,
    ):
        if X_left.shape[1] != X_right.shape[1]:
            raise ValueError(
                f"Descriptor block {component.name} has incompatible feature "
                f"counts: {X_left.shape[1]} and {X_right.shape[1]}."
            )

        K_component = _kernel_component_torch(
            X_left,
            X_right,
            component=component,
            torch=torch,
        )
        if K_total is None:
            K_total = K_component
        else:
            K_total.add_(K_component)

    if K_total is None:
        raise ValueError("At least one kernel component is required.")
    return K_total


def _kernel_component_torch(X_left, X_right, *, component: KernelComponent, torch):
    spec = distance_spec_for_kernel(component.kernel_type)
    if spec.kernel_type == "rbf":
        kernel = _squared_euclidean_distance(X_left, X_right, torch=torch)
    elif spec.kernel_type == "laplacian":
        kernel = torch.cdist(X_left, X_right, p=1)
    else:
        raise ValueError(
            f'Unsupported PyTorch exact kernel "{component.kernel_type}".'
        )

    kernel.mul_(-float(component.gamma))
    kernel.exp_()
    kernel.mul_(float(component.kernel_weight))
    return kernel


def _squared_euclidean_distance(X_left, X_right, *, torch):
    left_norms = torch.sum(X_left * X_left, dim=1, keepdim=True)
    right_norms = torch.sum(X_right * X_right, dim=1).reshape(1, -1)
    distances = X_left @ X_right.T
    distances.mul_(-2.0)
    distances.add_(left_norms)
    distances.add_(right_norms)
    return distances.clamp_(min=0.0)


def _resolve_torch(*, dtype, device):
    from pytorch_backend import require_torch, resolve_torch_device, resolve_torch_dtype

    torch = require_torch()
    torch_dtype = resolve_torch_dtype(torch, dtype)
    torch_device = resolve_torch_device(torch, device)
    return torch, torch_dtype, torch_device
