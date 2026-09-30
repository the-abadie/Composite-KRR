"""Affine extrapolation, fold leakage, and cached/direct backend equivalence."""

import importlib.util
import unittest

import numpy as np
from sklearn.base import clone
from sklearn.compose import TransformedTargetRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import StandardScaler

from cached_scoring import score_estimator_params_from_cache
from config_schema import RunConfig
from kernel_cache import build_distance_cache
from run_config import template
from search_random import CompositeKRREstimator, _maybe_build_distance_cache


class LinearBaselineTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(10)
        self.counts = rng.integers(0, 6, size=(24, 3)).astype(float)
        self.geometry = rng.normal(size=(24, 2))
        self.X = self.sample_matrix(self.counts, self.geometry)
        self.y = self.counts @ np.array([100., -17., 9.]) + np.sin(self.geometry[:, 0])
        self.cv = KFold(3, shuffle=True, random_state=42)

    @staticmethod
    def sample_matrix(*blocks):
        X = np.empty((len(blocks[0]), len(blocks)), dtype=object)
        for i, block in enumerate(blocks):
            X[:, i] = list(block)
        return X

    def estimator(self, backend="numpy"):
        return TransformedTargetRegressor(
            regressor=CompositeKRREstimator(names=["counts", "geometry"], kernel_types=["rbf", "laplacian"],
                         normalizations=["standard", "none"], gammas=[.1, .3], kernel_weights=[.2, .8],
                         alpha=.1, compute_dtype="float64", backend=backend, pytorch_device="cpu", linear_baseline="counts"),
            transformer=StandardScaler())

    def cache(self, y):
        return build_distance_cache(self.X, y, self.cv, names=["counts", "geometry"],
                  kernel_types=["rbf", "laplacian"], normalizations=["standard", "none"],
                  target_transformer=StandardScaler(), linear_baseline="counts", n_jobs=1)

    def test_affine_extrapolation_is_exact(self):
        y = self.counts @ np.array([100., -17., 9.]) + 4
        reg = self.estimator().fit(self.X, y)
        counts = np.array([[20, 2, 1], [0, 25, 2]])
        X = self.sample_matrix(counts, np.ones((2, 2)))
        np.testing.assert_allclose(reg.predict(X), counts @ [100., -17., 9.] + 4, atol=2e-11)

    def test_numpy_cached_and_direct_folds_match_scalar_and_multioutput(self):
        for y in (self.y, np.c_[self.y, self.y * 2 + self.geometry[:, 1]]):
            estimator = self.estimator()
            expected = cross_val_score(estimator, self.X, y, cv=self.cv, scoring="neg_mean_absolute_error")
            actual = score_estimator_params_from_cache(estimator, {}, self.cache(y), scoring="neg_mean_absolute_error", backend="numpy", n_jobs=1)
            np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-10)

    def test_baseline_matches_independent_least_squares(self):
        cache = self.cache(self.y)
        train, val = next(self.cv.split(self.X))
        scaler = StandardScaler().fit(self.y[train, None])
        model = LinearRegression().fit(self.counts[train], scaler.transform(self.y[train, None]))
        np.testing.assert_allclose(cache.folds[0].baseline_validation, model.predict(self.counts[val]), atol=1e-12)

    def test_validation_labels_do_not_affect_training_baseline(self):
        cache = self.cache(self.y)
        _, val = next(self.cv.split(self.X))
        changed = self.y.copy()
        changed[val] += 10000
        changed_cache = self.cache(changed)
        np.testing.assert_array_equal(cache.folds[0].y_train_transformed, changed_cache.folds[0].y_train_transformed)
        np.testing.assert_array_equal(cache.folds[0].baseline_validation, changed_cache.folds[0].baseline_validation)

    def test_search_cache_receives_baseline(self):
        cache = _maybe_build_distance_cache(self.estimator(), self.X, self.y, self.cv,
                   n_components=2, block_size=16, dtype="float64", n_jobs=1, memory_fraction=.8)
        self.assertIsNotNone(cache.folds[0].baseline_validation)

    @unittest.skipUnless(importlib.util.find_spec("torch"), "Torch unavailable")
    def test_torch_cached_and_direct_predictions_match_numpy(self):
        import torch
        y = np.c_[self.y, self.y * 2 + self.geometry[:, 1]]
        expected = cross_val_score(self.estimator(), self.X, y, cv=self.cv, scoring="neg_mean_absolute_error")
        for device in ("cpu", "cuda") if torch.cuda.is_available() else ("cpu",):
            estimator = self.estimator("pytorch").set_params(regressor__pytorch_device=device)
            actual = score_estimator_params_from_cache(estimator, {}, self.cache(y), scoring="neg_mean_absolute_error", backend="pytorch", pytorch_device=device, pytorch_dtype="float64", pytorch_candidate_batch_size=2)
            np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-10)
            a, b = self.estimator().fit(self.X, y), estimator.fit(self.X, y)
            np.testing.assert_allclose(a.predict(self.X), b.predict(self.X), rtol=1e-10, atol=1e-10)

    def test_configuration_rejects_missing_id_and_nystrom(self):
        config = template()
        config["model"]["linear_baseline"] = "unknown"
        with self.assertRaises(ValueError):
            RunConfig.model_validate(config)
        config["model"]["linear_baseline"] = config["descriptors"][0]["id"]
        config["model"]["backend"] = "nystrom"
        with self.assertRaises(ValueError):
            RunConfig.model_validate(config)


if __name__ == "__main__":
    unittest.main()
