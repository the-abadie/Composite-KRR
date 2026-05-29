import preparation
import config
import numpy as np
import random
import logging
from utilities import configure_logging
from class_CompositeDescriptor import CompositeDescriptor
from class_Target import Target
import kfold


# 0) Initialize
configure_logging(config.VERBOSITY)
logger = logging.getLogger("CKRR")

random.seed(config.SEED)
np.random.seed(config.SEED)
rng = np.random.default_rng(config.SEED)

# 1) Load Descriptor
descriptors = CompositeDescriptor(names=config.X_NAMES, paths=config.X_PATHS, normalizations=config.X_NORMS)
descriptors.load_descriptor_blocks_from_npy()
DESCRIPTOR_ORDER:int = len(descriptors.blocks)

# 2) Load Target
target = Target(name=config.Y_NAME, path=config.Y_PATH, normalization=config.Y_NORM)
target.load_target_from_npy()
TARGET_LENGTH:int = target.n_samples

preparation.validate_descriptor_target_lengths(descriptors.blocks, target.data)

# 3) Split/Validate
if config.N_SAMPLES > TARGET_LENGTH:
    logger.warning(f"Provided N_SAMPLES ({config.N_SAMPLES}) is greater than the length of the target ({TARGET_LENGTH}). All indicies will be used.")
    valid_idx = np.arange(TARGET_LENGTH, dtype=int)
elif config.N_SAMPLES == TARGET_LENGTH:
    valid_idx = np.arange(TARGET_LENGTH, dtype=int)
else:
    valid_idx = rng.choice(a=TARGET_LENGTH, size=config.N_SAMPLES, replace=False)

N_SAMPLES_REMAIN:int = len(valid_idx)
N_TRAIN_VAL:int = int(np.floor(N_SAMPLES_REMAIN*config.TRAIN_VAL_SPLIT))
N_TEST:int = N_SAMPLES_REMAIN - N_TRAIN_VAL

logger.info(f"{N_TRAIN_VAL} samples to be used for training/validation.")
logger.info(f"{N_TEST} samples to be held-out for testing.")

if config.STRATIFY:
    idx_train_local, idx_test_local = preparation.stratified_selection_with_remainder(
        target=target.data[valid_idx], n_strata=config.N_STRATA, n_total=N_TRAIN_VAL, rng=rng)
    idx_train_val = valid_idx[idx_train_local]
    idx_test      = valid_idx[idx_test_local]

else:
    idx_train_val, idx_test = preparation.randomized_selection_with_remainder(
        arr=valid_idx, N=N_TRAIN_VAL, rng=rng)

# 4) Prepare CV Splits
kf = kfold.initialize_kfold(n_splits=config.N_KFOLD, shuffle=False, seed=config.SEED)

folds = kfold.build_folds(
    idx_train_val=idx_train_val,
    composite_descriptor=descriptors,
    target=target,
    kf=kf,
)
