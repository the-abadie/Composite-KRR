from dataclasses import dataclass, field
from pathlib import Path
import logging

import numpy as np
from numpy.typing import NDArray
from descriptor_storage import load_descriptor_array


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
    archive_keys: list[str | None] | None = None
    blocks: list[DescriptorBlock] = field(default_factory=list, init=False, repr=False)

    def __post_init__(self):
        if not (len(self.names) == len(self.paths) == len(self.normalizations)):
            raise ValueError(
                "names, paths, and normalizations must have the same length. "
                f"Got {len(self.names)}, {len(self.paths)}, {len(self.normalizations)}."
            )
        if self.archive_keys is None:
            self.archive_keys = [None] * len(self.names)
        if len(self.archive_keys) != len(self.names):
            raise ValueError("archive_keys must have the same length as names.")

    def load_descriptor_blocks(self) -> list[DescriptorBlock]:
        blocks = []

        for name, path_str, normalization, archive_key in zip(self.names, self.paths, self.normalizations, self.archive_keys):
            path = Path(path_str)

            # Keep large descriptors on disk; only selected training or prediction
            # rows are materialized by the workflow.
            values = load_descriptor_array(path, archive_key)
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

    def load_descriptor_blocks_from_npy(self) -> list[DescriptorBlock]:
        """Compatibility entrypoint; the generic loader also accepts manifests."""
        return self.load_descriptor_blocks()

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
