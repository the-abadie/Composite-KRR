from dataclasses import dataclass, field
from pathlib import Path
import logging
import re

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
    description: str | None = None

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
        self.names = list(self.names)
        self.paths = list(self.paths)
        self.normalizations = list(self.normalizations)

        if not self.paths:
            raise ValueError("At least one descriptor path is required.")
        if not self.normalizations:
            raise ValueError("At least one descriptor normalization is required.")

    def load_descriptor_blocks(self) -> list[DescriptorBlock]:
        suffixes = [Path(path).suffix.lower() for path in self.paths]
        if all(suffix == ".npy" for suffix in suffixes):
            return self.load_descriptor_blocks_from_npy()
        if len(suffixes) == 1 and suffixes[0] == ".npz":
            return self.load_descriptor_blocks_from_npz()

        raise ValueError(
            "Descriptor inputs must be either one or more .npy files or exactly "
            "one .npz archive. Mixed descriptor formats are not supported."
        )

    def load_descriptor_blocks_from_npy(self) -> list[DescriptorBlock]:
        if not (
            len(self.names) == len(self.paths) == len(self.normalizations)
        ):
            raise ValueError(
                "For .npy descriptors, names, paths, and normalizations must "
                "have the same length. "
                f"Got {len(self.names)}, {len(self.paths)}, {len(self.normalizations)}."
            )
        blocks = []

        for name, path_str, normalization in zip(self.names, self.paths, self.normalizations):
            path = Path(path_str)

            if path.suffix.lower() != ".npy":
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
        if path.suffix.lower() != ".npz":
            raise ValueError(f"Descriptor path {path} is not .npz")

        configured_names = list(self.names)
        with np.load(path, allow_pickle=False) as npz:
            component_indices = self._npz_component_indices(npz, path)
            normalizations = self._resolve_npz_normalizations(
                len(component_indices), path
            )

            blocks = []
            for index, normalization in zip(
                component_indices, normalizations, strict=True
            ):
                prefix = f"component_{index}"
                name = self._load_npz_string(npz, f"{prefix}_name", path)
                description = self._load_npz_description(npz, prefix, path)
                data_key = f"{prefix}_data"
                if data_key not in npz.files:
                    raise ValueError(
                        f"Descriptor archive {path} is missing required key "
                        f"{data_key!r}."
                    )

                values = np.asarray(npz[data_key])
                self._validate_npz_values(values, name=name, path=path)
                blocks.append(
                    DescriptorBlock(
                        name=name,
                        path=path,
                        normalization=normalization,
                        values=values,
                        description=description,
                    )
                )

                logger.debug(
                    "Descriptor %s loaded from %s with shape %s: %s",
                    name,
                    path,
                    values.shape,
                    description,
                )

        self._validate_sample_counts(blocks)
        archive_names = [block.name for block in blocks]
        duplicate_names = sorted(
            name for name in set(archive_names) if archive_names.count(name) > 1
        )
        if duplicate_names:
            raise ValueError(
                f"Descriptor names in archive {path} must be unique; "
                f"duplicates: {duplicate_names}."
            )

        if configured_names:
            logger.warning(
                "Using descriptor names from %s and overriding configured names %s "
                "with %s.",
                path,
                configured_names,
                archive_names,
            )
        else:
            logger.info("Using descriptor names from %s: %s.", path, archive_names)

        self.names = archive_names
        self.normalizations = normalizations
        self.blocks = blocks
        logger.info(f"{len(self.blocks)} descriptor blocks successfully loaded.")
        return self.blocks

    def _resolve_npz_normalizations(
        self, n_descriptors: int, path: Path
    ) -> list[str]:
        if len(self.normalizations) == 1:
            resolved = self.normalizations * n_descriptors
            if n_descriptors > 1:
                logger.warning(
                    "Applying the single configured normalization %r to all %s "
                    "descriptors in %s.",
                    self.normalizations[0],
                    n_descriptors,
                    path,
                )
            return resolved

        if len(self.normalizations) != n_descriptors:
            raise ValueError(
                f"Descriptor archive {path} contains {n_descriptors} descriptors, "
                f"but {len(self.normalizations)} normalizations were configured. "
                "Provide either one normalization or one per descriptor."
            )

        logger.info(
            "Applying %s configured normalizations to the descriptors in archive order.",
            n_descriptors,
        )
        return list(self.normalizations)

    @staticmethod
    def _npz_component_indices(npz, path: Path) -> list[int]:
        pattern = re.compile(r"^component_(\d+)_data$")
        indices = sorted(
            int(match.group(1))
            for key in npz.files
            if (match := pattern.fullmatch(key)) is not None
        )
        if not indices:
            raise ValueError(
                f"Descriptor archive {path} contains no component_N_data arrays."
            )

        expected_indices = list(range(len(indices)))
        if indices != expected_indices:
            raise ValueError(
                f"Descriptor component indices in {path} must be contiguous from 0; "
                f"got {indices}."
            )

        if "component_count" in npz.files:
            raw_count = np.asarray(npz["component_count"])
            if raw_count.ndim != 0 or not np.issubdtype(
                raw_count.dtype, np.integer
            ):
                raise ValueError(
                    f"component_count in descriptor archive {path} must be a "
                    "scalar integer."
                )
            component_count = int(raw_count.item())
            if component_count != len(indices):
                raise ValueError(
                    f"Descriptor archive {path} declares {component_count} components "
                    f"but contains data arrays for {len(indices)} components."
                )

        return indices

    @staticmethod
    def _load_npz_string(npz, key: str, path: Path) -> str:
        if key not in npz.files:
            raise ValueError(
                f"Descriptor archive {path} is missing required key {key!r}."
            )

        value = np.asarray(npz[key])
        if value.ndim != 0 or not isinstance(value.item(), str):
            raise ValueError(
                f"Descriptor archive key {key!r} in {path} must be a scalar string."
            )

        resolved = value.item().strip()
        if not resolved:
            raise ValueError(
                f"Descriptor archive key {key!r} in {path} cannot be empty."
            )
        return resolved

    @classmethod
    def _load_npz_description(cls, npz, prefix: str, path: Path) -> str:
        description_key = f"{prefix}_description"
        short_key = f"{prefix}_desc"
        available_keys = [
            key for key in (description_key, short_key) if key in npz.files
        ]
        if not available_keys:
            raise ValueError(
                f"Descriptor archive {path} is missing required key "
                f"{short_key!r} (or {description_key!r})."
            )

        descriptions = [
            cls._load_npz_string(npz, key, path) for key in available_keys
        ]
        if len(set(descriptions)) != 1:
            raise ValueError(
                f"Descriptor archive {path} provides conflicting descriptions "
                f"for {prefix}."
            )
        return descriptions[0]

    @staticmethod
    def _validate_npz_values(values: NDArray, *, name: str, path: Path) -> None:
        if values.ndim == 0:
            raise ValueError(
                f"Descriptor {name!r} in {path} must have a sample axis."
            )
        if values.shape[0] == 0:
            raise ValueError(f"Descriptor {name!r} in {path} has no samples.")
        if not np.issubdtype(values.dtype, np.number) or np.issubdtype(
            values.dtype, np.complexfloating
        ):
            raise ValueError(
                f"Descriptor {name!r} in {path} must contain real numeric data; "
                f"got dtype {values.dtype}."
            )

    @staticmethod
    def _validate_sample_counts(descriptors: list[DescriptorBlock]) -> None:
        if not descriptors:
            raise ValueError("No descriptors were loaded.")

        for descriptor in descriptors:
            if descriptor.values.ndim == 0:
                raise ValueError(
                    f"Descriptor {descriptor.name} must have a sample axis."
                )

        n_samples = descriptors[0].n_samples

        for descriptor in descriptors:
            if descriptor.n_samples != n_samples:
                raise ValueError(
                    f"Descriptor {descriptor.name} has {descriptor.n_samples} samples, "
                    f"but expected {n_samples}.")
