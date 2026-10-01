"""Bound held-out descriptor memory independently of the number of test rows."""
from pathlib import Path

import numpy as np

from preprocess import descriptor_blocks_to_sample_matrix
from target_utils import as_target_matrix


class RegressionErrorAccumulator:
    """Merge float64 batch moments while retaining original-unit target errors."""

    def __init__(self):
        self.n_samples = 0
        self.absolute_sum = None
        self.squared_sum = None
        self.absolute_mean = 0.0
        self.absolute_m2 = 0.0

    def update(self, y_true, y_pred):
        true = as_target_matrix(y_true, dtype=np.float64)
        pred = as_target_matrix(y_pred, dtype=np.float64)
        if true.shape != pred.shape:
            raise ValueError(f"Prediction shape {pred.shape} does not match target shape {true.shape}.")
        if not np.all(np.isfinite(true)) or not np.all(np.isfinite(pred)):
            raise ValueError("Held-out targets and predictions must be finite.")
        error = pred - true
        absolute = np.abs(error)
        sums = np.sum(absolute, axis=0, dtype=np.float64)
        squares = np.sum(error * error, axis=0, dtype=np.float64)
        if self.absolute_sum is None:
            self.absolute_sum = np.zeros_like(sums)
            self.squared_sum = np.zeros_like(squares)
        elif self.absolute_sum.shape != sums.shape:
            raise ValueError("The target count changed between prediction batches.")
        self.absolute_sum += sums
        self.squared_sum += squares

        # Parallel variance formula avoids catastrophic cancellation from E[x^2]-E[x]^2.
        previous_count = self.n_samples * true.shape[1]
        batch_count = absolute.size
        batch_mean = float(np.mean(absolute))
        batch_m2 = float(np.sum((absolute - batch_mean) ** 2, dtype=np.float64))
        delta = batch_mean - self.absolute_mean
        count = previous_count + batch_count
        self.absolute_mean += delta * batch_count / count
        self.absolute_m2 += batch_m2 + delta * delta * previous_count * batch_count / count
        self.n_samples += true.shape[0]

    def summary(self):
        if self.n_samples == 0:
            raise ValueError("No held-out samples have been evaluated.")
        target_mae = self.absolute_sum / self.n_samples
        target_rmse = np.sqrt(self.squared_sum / self.n_samples)
        return {"mae": float(np.mean(target_mae)),
                "mae_std": float(np.sqrt(self.absolute_m2 / (self.n_samples * len(target_mae)))),
                "rmse": float(np.mean(target_rmse)),
                "target_mae": target_mae, "target_rmse": target_rmse}


def predict_held_out(estimator, descriptors, target, indices, output_directory, *, batch_size):
    """Save every held-out row in index order without constructing a full feature matrix."""
    if type(batch_size) is not int or batch_size <= 0:
        raise ValueError("prediction batch_size must be a positive integer.")
    indices = np.asarray(indices)
    if indices.ndim != 1 or not len(indices) or not np.issubdtype(indices.dtype, np.integer):
        raise ValueError("Held-out indices must be a nonempty integer vector.")
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    shape = (len(indices), *target.shape[1:])
    # Full prediction artifacts preserve float64 values regardless of kernel precision.
    predicted = np.lib.format.open_memmap(output / "y_predictions.npy", mode="w+", dtype=np.float64, shape=shape)
    truth = np.lib.format.open_memmap(output / "y_true.npy", mode="w+", dtype=np.float64, shape=shape)
    accumulator = RegressionErrorAccumulator()
    for start in range(0, len(indices), batch_size):
        stop = min(start + batch_size, len(indices))
        rows = indices[start:stop]
        X = descriptor_blocks_to_sample_matrix(descriptors, rows)
        y_true = target[rows]
        y_pred = estimator.predict(X)
        # Scalar vector/matrix conventions may differ between an estimator and its target.
        if np.shape(y_pred) != np.shape(y_true):
            if as_target_matrix(y_pred).shape != as_target_matrix(y_true).shape:
                raise ValueError("Prediction and target shapes do not match.")
            y_pred = np.asarray(y_pred).reshape(y_true.shape)
        accumulator.update(y_true, y_pred)
        predicted[start:stop] = y_pred
        truth[start:stop] = y_true
    predicted.flush()
    truth.flush()
    return predicted, truth, accumulator.summary()


def diagnostic_plot_sample(y_true, y_pred, *, max_samples):
    """Deterministic, evenly spaced sample for plots only; test metrics use every row."""
    count = min(len(y_true), max_samples)
    rows = np.linspace(0, len(y_true) - 1, count, dtype=np.int64)
    return y_true[rows], y_pred[rows]
