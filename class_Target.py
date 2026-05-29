from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import numpy as np
import logging

from utilities import configure_logging
from config import VERBOSITY
configure_logging(VERBOSITY)

logger = logging.getLogger("class_Target")

@dataclass(frozen=True)
class Target:
    name:str
    path:Path
    normalization:str = "standard"

    def load_target_from_npy(self) -> np.ndarray:
        if self.path.suffix != ".npy":
            raise ValueError(f"Target Path {self.path} is not .npy")
        y = np.load(self.path)
        logger.info(f"Target {self.path} successfully loaded.")
        y = y.reshape(-1)
        return y

    def load_target_from_npz(self) -> np.ndarray:
        if self.path.suffix != ".npz":
            raise ValueError(f"Target Path {self.path} is not .npz")
        y = np.load(self.path)[self.name]
        logger.info(f"Target {self.path} successfully loaded.")
        y = y.reshape(-1)
        return y
