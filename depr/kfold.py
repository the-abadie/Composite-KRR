from class_CompositeDescriptor import CompositeDescriptor
from sklearn.model_selection import KFold, LeaveOneOut
import logging
from utilities import configure_logging
from config import VERBOSITY
from numpy.typing import NDArray
import preprocess
from dataclasses import dataclass
from class_Target import Target


configure_logging(VERBOSITY)
logger = logging.getLogger("kfold")

def initialize_kfold(n_splits:int, shuffle:bool, seed:int):
    if n_splits == 0:
        logger.info("LeaveOneOut CV selected")
        return LeaveOneOut()
    else:
        logger.info(f"{n_splits}-fold CV selected")
        return KFold(
            n_splits=n_splits,
            shuffle=shuffle,
            random_state=seed if shuffle else None,
        )

@dataclass
class FoldData:
    fold_id:int
    idx_train: NDArray
    idx_val: NDArray
    X_train_blocks: list[NDArray]
    X_val_blocks: list[NDArray]
    y_train: NDArray
    y_val: NDArray
    y_train_transform: NDArray
    y_preprocessor: object
    X_preprocessors: list[object]


def build_folds(idx_train_val:NDArray, composite_descriptor:CompositeDescriptor, target:Target, kf) -> list[FoldData]:
    if target.data is None:
        raise ValueError("Target data has not been loaded.")

    folds = []
    target_data = target.data

    for fold_id, (train_local, val_local) in enumerate(kf.split(idx_train_val)):
        idx_fold_train = idx_train_val[train_local]
        idx_fold_val = idx_train_val[val_local]

        X_train_blocks_t, X_val_blocks_t, X_preprocessors = preprocess.transform_descriptors_for_split(
            descriptors=composite_descriptor,
            train_idx=idx_fold_train,
            eval_idx=idx_fold_val,
        )

        y_preprocessor = preprocess.make_target_preprocessor(target.normalization)

        y_train_t = y_preprocessor.fit_transform(
            target_data[idx_fold_train].reshape(-1, 1)
        ).reshape(-1)

        folds.append(
            FoldData(
                fold_id=fold_id,
                idx_train=idx_fold_train,
                idx_val=idx_fold_val,
                X_train_blocks=X_train_blocks_t,
                X_val_blocks=X_val_blocks_t,
                y_train=target_data[idx_fold_train],
                y_val=target_data[idx_fold_val],
                y_train_transform=y_train_t,
                y_preprocessor=y_preprocessor,
                X_preprocessors=X_preprocessors,
            )
        )

    logger.info("CV folds built.")
    return folds
