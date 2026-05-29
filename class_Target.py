from dataclasses import dataclass, field
from pathlib import Path
import numpy as np
import logging
from numpy.typing import NDArray

from utilities import configure_logging
from config import VERBOSITY
configure_logging(VERBOSITY)

logger = logging.getLogger("class_Target")

@dataclass
class Target:
    name:str
    path:Path
    normalization:str = "standard"
    data: NDArray | None = field(default=None, init=False, repr=False)

    @property
    def n_samples(self) -> int:
        if self.data is None:
            raise ValueError("Target data has not been loaded.")
        return self.data.shape[0]

    def load_target_from_npy(self) -> NDArray:
        path = Path(self.path)
        if path.suffix != ".npy":
            raise ValueError(f"Target Path {path} is not .npy")
        y = np.load(path)
        logger.info(f"Target {self.name} successfully loaded.")
        self.data = y.reshape(-1)
        return self.data

    def load_target_from_npz(self) -> NDArray:
        path = Path(self.path)
        if path.suffix != ".npz":
            raise ValueError(f"Target Path {path} is not .npz")
        y = np.load(path)[self.name]
        logger.info(f"Target {path} successfully loaded.")
        self.data = y.reshape(-1)
        return self.data
