from dataclasses import dataclass
import numpy as np
import logging
from config import VERBOSITY
from utilities import configure_logging
from pathlib import Path

np.random.standard_normal
configure_logging(VERBOSITY)
logger = logging.getLogger("class_CompositeDescriptor")

@dataclass(frozen=True)
class CompositeDescriptor:
    names: list[str]
    paths: list[str]
    normalizations: list[str]

    def load_composite_descriptor_from_npy(self):
        descriptors = []
        logger.debug(f"Descriptor paths: {self.paths}")
        logger.debug(f"Descriptor names: {self.names}")

        paths = [Path(path) for path in self.paths]
        for i in range(len(paths)):
            if paths[i].suffix != ".npy":
                raise ValueError(f"Target Path {paths[i]} is not .npy")

            descriptors.append(np.load(paths[i]))
            logger.debug(f"Descriptor {self.names[i]} appended.")

        composite_descriptor = np.stack(descriptors)
        logger.info(f"Composite Descriptor (order {len(self.names)}) successfully loaded.")
        return composite_descriptor

    def load_composite_descriptor_from_npz(self):
        if len(self.paths) != 1:
            raise ValueError(f"Only one path accepted for loading from .npz, got {len(self.paths)}")

        descriptors = []
        logger.debug(f"Descriptor path: {self.paths}")
        logger.debug(f"Descriptor names: {self.names}")

        npz = np.load(self.paths[0])
        for i in range(len(self.names)):
            descriptors.append(npz[self.names[i]])
            logger.debug(f"Descriptor {self.names[i]} appended.")

        composite_descriptor = np.stack(descriptors)
        logger.info(f"Composite Descriptor (order {len(self.names)}) successfully loaded.")
        return composite_descriptor
