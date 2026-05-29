import logging
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import ArrayLike, NDArray

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
        if self.weight < 0:
            raise ValueError(
                f"Kernel component {self.name} has negative weight {self.weight}."
            )


class CompositeKRR:
    alpha: float
    components: list[KernelComponent]

    def __init__(self, components: list[KernelComponent], alpha: float):
        if alpha < 0:
            raise ValueError(f"alpha must be non-negative, got {alpha}.")

        self.components = list(components) if components is not None else []
        self.alpha = alpha

    def fit():
        pass

class CompositeKRR_Estimator:
    alpha: float
    gammas: list[float]
    kernel_weights: list[float]
    names: list[str]
    kernel_types: list[str]
    random_state = None

    def __init__(self, alpha, gammas, weights):
        self.alpha = alpha
        self.gammas = gammas
        self.weights = weights

        if alpha <= 0:
            raise ValueError(f"Hyperparamter alpha must be positive, got {self.alpha}.")
        if len(gammas) != len(weights):
            raise ValueError(
                f"Number of gammas {(len(self.gammas))} must equal "
                f"the number of kernel weights ({len(self.kernel_weights)})."
            )

    def fit(self, X, y):

        kernel_components = []
        for i in range(len(self.gammas)):
            kernel_components.append(
                KernelComponent(
                    name=self.names[i],
                    gamma=self.gammas[i],
                    kernel_weight=self.kernel_weights[i],
                    kernel_type=self.kernel_types[i],
                )
            )

        self.model_ = CompositeKRR(components=kernel_components, alpha=self.alpha)

        return self
