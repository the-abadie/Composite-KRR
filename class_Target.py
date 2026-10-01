from dataclasses import dataclass, field
from pathlib import Path
import numpy as np
import logging
from numpy.typing import NDArray

from target_utils import as_target_array, as_target_matrix

logger = logging.getLogger("class_Target")

@dataclass
class Target:
    name:str | list[str]
    path:Path
    normalization:str
    data: NDArray | None = None
    column_names: list[str] = field(default_factory=list, init=False)

    @property
    def n_samples(self) -> int:
        if self.data is None:
            raise ValueError("Target data has not been loaded.")
        return self.data.shape[0]

    def load_target_from_npy(self) -> NDArray:
        path = Path(self.path)
        if path.suffix != ".npy":
            raise ValueError(f"Target Path {path} is not .npy")
        y = np.load(path, mmap_mode="r")
        logger.info(f"Target {self.name} successfully loaded.")
        self.data = as_target_array(y)
        self.column_names = target_names(self.name, self.data)
        return self.data

    def load_target_from_npz(self) -> NDArray:
        path = Path(self.path)
        if path.suffix != ".npz":
            raise ValueError(f"Target Path {path} is not .npz")
        with np.load(path) as npz:
            if isinstance(self.name, str):
                y = npz[self.name]
                self.column_names = target_names(self.name, y)
            else:
                if not self.name:
                    raise ValueError(
                        "At least one target name is required for .npz loading."
                    )
                target_blocks = [
                    as_target_matrix(npz[target_name])
                    for target_name in self.name
                ]
                self.column_names = [label for name, block in zip(self.name, target_blocks)
                                     for label in target_names(name, block)]
                y = np.concatenate(target_blocks, axis=1)
        logger.info(f"Target {path} successfully loaded.")
        self.data = as_target_array(y)
        return self.data


def target_names(name, y):
    """Label original-unit target columns in their exact matrix order."""
    count = as_target_matrix(y).shape[1]
    if isinstance(name, list):
        if len(name) != count:
            raise ValueError(f"Target name list has {len(name)} names for {count} columns.")
        return list(name)
    return [name] if count == 1 else [f"{name}[{index}]" for index in range(count)]
