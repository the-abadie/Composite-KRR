"""Fold-fitted affine baseline for a designated descriptor block."""

import numpy as np


def fit_linear_baseline(X, y):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).reshape(len(X), -1)
    if X.ndim != 2 or not np.isfinite(X).all() or not np.isfinite(y).all():
        raise ValueError("Linear baseline inputs must be finite sample matrices")
    x_mean, y_mean = X.mean(axis=0), y.mean(axis=0)
    coefficients = np.linalg.lstsq(X - x_mean, y - y_mean, rcond=None)[0]
    intercept = y_mean - x_mean @ coefficients
    return coefficients, intercept


def predict_linear_baseline(X, fitted):
    coefficients, intercept = fitted
    return np.asarray(X, dtype=float) @ coefficients + intercept
