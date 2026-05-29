import logging
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.linalg import solve
from sklearn.metrics.pairwise import pairwise_kernels

from config import VERBOSITY
from utilities import configure_logging

configure_logging(VERBOSITY)
logger = logging.getLogger("class_CompositeKRR")


@dataclass(frozen=True)
class KernelComponent:
    name: str
    gamma: float
    kernel_weight: float
    kernel_type: str

    def __post_init__(self) -> None:
        if self.kernel_weight < 0:
            raise ValueError(
                f"Kernel component {self.name} has negative weight "
                f"{self.kernel_weight}."
            )
        if self.gamma < 0:
            raise ValueError(
                f"Kernel component {self.name} has negative gamma {self.gamma}."
            )


class CompositeKRR:
    alpha: float
    components: list[KernelComponent]

    def __init__(self, components: list[KernelComponent], alpha: float):
        if alpha <= 0:
            raise ValueError(f"alpha must be positive, got {alpha}.")

        self.components = list(components) if components is not None else []
        self.alpha = alpha

    def fit(self, X_blocks: list[NDArray], y: ArrayLike):
        y = np.asarray(y, dtype=float).reshape(-1)
        n_samples = y.shape[0]
        if n_samples == 0:
            raise ValueError("Cannot fit CompositeKRR with zero samples.")

        X_train_blocks = self._prepare_blocks(X_blocks, n_samples=n_samples)

        K = self._composite_kernel(X_train_blocks, X_train_blocks)
        K[np.diag_indices_from(K)] += self.alpha

        self.X_train_blocks_ = X_train_blocks
        self.y_train_ = y
        self.dual_coef_ = solve(K, y, assume_a="pos")

        return self

    def predict(self, X_blocks: list[NDArray]) -> NDArray:
        if not hasattr(self, "dual_coef_"):
            raise ValueError("CompositeKRR has not been fitted.")

        X_eval_blocks = self._prepare_blocks(X_blocks)
        K_eval = self._composite_kernel(X_eval_blocks, self.X_train_blocks_)
        return K_eval @ self.dual_coef_

    def _prepare_blocks(
        self, X_blocks: list[NDArray], n_samples: int | None = None
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
            X = np.asarray(X, dtype=float)
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

    def _composite_kernel(
        self,
        X_left_blocks: list[NDArray],
        X_right_blocks: list[NDArray],
    ) -> NDArray:
        K_total = None

        for component, X_left, X_right in zip(
            self.components, X_left_blocks, X_right_blocks
        ):
            if X_left.shape[1] != X_right.shape[1]:
                raise ValueError(
                    f"Descriptor block {component.name} has incompatible feature "
                    f"counts: {X_left.shape[1]} and {X_right.shape[1]}."
                )

            K = pairwise_kernels(
                X_left,
                X_right,
                metric=component.kernel_type,
                filter_params=True,
                gamma=component.gamma,
            )
            weighted_K = component.kernel_weight * K

            if K_total is None:
                K_total = weighted_K
            else:
                K_total += weighted_K

        if K_total is None:
            raise ValueError("At least one kernel component is required.")

        return K_total
