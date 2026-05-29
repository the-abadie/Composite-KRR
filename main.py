import preprocess
import preparation
import config
import numpy as np
import random
import logging
from utilities import configure_logging
from class_CompositeDescriptor import CompositeDescriptor
from class_Target import Target

# 0) Initialize
configure_logging(config.VERBOSITY)
logger = logging.getLogger("CKRR")

random.seed(config.SEED)
np.random.seed(config.SEED)
rng = np.random.default_rng(config.SEED)

# 1) Load Descriptor
descriptor_data = CompositeDescriptor(names=config.X_NAMES, paths=config.X_PATHS, normalizations=config.X_NORMS)
X = descriptor_data.load_composite_descriptor_from_npy()
DESCRIPTOR_ORDER = len(X)

# 2) Load Target
target_data = Target(name=config.Y_NAME, path=config.Y_PATH, normalization=config.Y_NORM)
target = target_data.load_target_from_npy()

if len(X[0]) != len(target):
    raise ValueError(f"Length of descriptor ({len(X[0])}) does not match length of target ({len(target)}).")

# 3) Split/Validate
if config.N_SAMPLES > len(target):
    logger.warning(f"Provided N_SAMPLES ({config.N_SAMPLES}) is greater than the length of the target ({len(target)}). All indicies will be used.")
    valid_idx = np.arange(len(target), dtype=int)
elif config.N_SAMPLES == len(target):
    valid_idx = np.arange(len(target), dtype=int)
else:
    valid_idx = np.random.randint(0, len(target), size=config.N_SAMPLES)

N_SAMPLES_REMAIN:int = len(valid_idx)
N_TRAIN:int = int(np.floor(N_SAMPLES_REMAIN*config.TRAINING_SPLIT))

if config.STRATIFY:
    idx_train_local, idx_test_local = preparation.stratified_selection_with_remainder(
        target=target[valid_idx], n_strata=config.N_STRATA, n_total=N_TRAIN, rng=rng)
    idx_train_val = valid_idx[idx_train_local]
    idx_test      = valid_idx[idx_test_local]
else:
    idx_train_val, idx_test = preparation.randomized_selection_with_remainder(
        arr=valid_idx, N=N_TRAIN, rng=rng)
