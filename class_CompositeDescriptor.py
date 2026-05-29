from dataclasses import dataclass, field
from pathlib import Path
import logging

import numpy as np
from numpy.typing import NDArray

from config import VERBOSITY
from utilities import configure_logging

configure_logging(VERBOSITY)
logger = logging.getLogger("class_CompositeDescriptor")


@dataclass(frozen=True)
class DescriptorBlock:
    name: str
    path: Path
    normalization: str
    values: NDArray

    @property
    def n_samples(self) -> int:
        return self.values.shape[0]


@dataclass
class CompositeDescriptor:
    names: list[str]
    paths: list[str]
    normalizations: list[str]
    blocks: list[DescriptorBlock] = field(default_factory=list, init=False, repr=False)

    def __post_init__(self):
        if not (len(self.names) == len(self.paths) == len(self.normalizations)):
            raise ValueError(
                "names, paths, and normalizations must have the same length. "
                f"Got {len(self.names)}, {len(self.paths)}, {len(self.normalizations)}."
            )

    def load_descriptor_blocks_from_npy(self) -> list[DescriptorBlock]:
        blocks = []

        for name, path_str, normalization in zip(self.names, self.paths, self.normalizations):
            path = Path(path_str)

            if path.suffix != ".npy":
                raise ValueError(f"Descriptor path {path} is not .npy")

            values = np.load(path)
            blocks.append(
                DescriptorBlock(
                    name=name,
                    path=path,
                    normalization=normalization,
                    values=values,
                )
            )

            logger.debug(f"Descriptor {name} loaded with shape {values.shape}.")

        self._validate_sample_counts(blocks)
        self.blocks = blocks
        logger.info(f"{len(self.blocks)} descriptor blocks successfully loaded.")
        return self.blocks

    def load_descriptor_blocks_from_npz(self) -> list[DescriptorBlock]:
        if len(self.paths) != 1:
            raise ValueError(f"Only one path accepted for loading from .npz, got {len(self.paths)}")

        path = Path(self.paths[0])
        if path.suffix != ".npz":
            raise ValueError(f"Descriptor path {path} is not .npz")

        blocks = []
        npz = np.load(path)

        for name, normalization in zip(self.names, self.normalizations):
            values = npz[name]
            blocks.append(
                DescriptorBlock(
                    name=name,
                    path=path,
                    normalization=normalization,
                    values=values,
                )
            )

            logger.debug(f"Descriptor {name} loaded with shape {values.shape}.")

        self._validate_sample_counts(blocks)
        self.blocks = blocks
        logger.info(f"{len(self.blocks)} descriptor blocks successfully loaded.")
        return self.blocks

    @staticmethod
    def _validate_sample_counts(descriptors: list[DescriptorBlock]) -> None:
        if not descriptors:
            raise ValueError("No descriptors were loaded.")

        n_samples = descriptors[0].n_samples

        for descriptor in descriptors:
            if descriptor.n_samples != n_samples:
                raise ValueError(
                    f"Descriptor {descriptor.name} has {descriptor.n_samples} samples, "
                    f"but expected {n_samples}.")
