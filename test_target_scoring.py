import importlib.util
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from sklearn.base import clone
from sklearn.compose import TransformedTargetRegressor
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import StandardScaler

from cached_scoring import score_estimator_params_from_cache
from estimator_factory import make_composite_krr_regressor
from postprocess import regression_error_summary, validation_error_from_score
from run_config import KRRConfig, run
from search_random import _maybe_build_distance_cache
from nystrom_cache import build_nystrom_distance_cache
from target_scoring import make_target_scorer


class TargetScoringTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(15)
        self.features = rng.normal(size=(24, 3))
        self.X = np.empty((24, 1), dtype=object)
        self.X[:, 0] = list(self.features)
        self.y = np.column_stack((np.sin(self.features[:, 0]), 10000 * self.features[:, 1] + 40000))
        self.cv = KFold(3, shuffle=True, random_state=13)
        self.scorer = make_target_scorer("neg_mean_absolute_error", "transformed")

    def estimator(self, kind):
        regressor = make_composite_krr_regressor(
            krr_backend=kind, names=["x"], kernel_types=["rbf"], normalizations=["standard"],
            pca_components=None, pca_whiten=False, normalize_kernel_weights=True,
            compute_dtype="float64", nystrom_n_landmarks=8, nystrom_landmark_selection="first", random_state=7)
        regressor.set_params(alpha=0.15, gammas=[0.2])
        return TransformedTargetRegressor(regressor=regressor, transformer=StandardScaler())

    def cache(self, estimator, y=None):
        if hasattr(estimator.regressor, "n_landmarks"):
            return build_nystrom_distance_cache(
                self.X, self.y if y is None else y, self.cv, names=["x"], kernel_types=["rbf"],
                normalizations=["standard"], target_transformer=StandardScaler(), n_landmarks=8,
                landmark_selection="first", batch_size=8, eigenvalue_floor=1e-12, n_jobs=1)
        return _maybe_build_distance_cache(
            estimator, self.X, self.y if y is None else y, self.cv, n_components=1,
            block_size=16, dtype="float64", n_jobs=1, memory_fraction=.8)

    def test_cached_and_uncached_match_independent_training_fold_scales(self):
        self.assertEqual(make_target_scorer("neg_mean_absolute_error"), "neg_mean_absolute_error")
        for kind in ("exact", "nystrom"):
            estimator = self.estimator(kind)
            manual = []
            for train, val in self.cv.split(self.X):
                fitted = clone(estimator).fit(self.X[train], self.y[train])
                error = fitted.predict(self.X[val]) - self.y[val]
                manual.append(-np.mean(np.abs(error) / np.std(self.y[train], axis=0)))
            direct = cross_val_score(estimator, self.X, self.y, cv=self.cv, scoring=self.scorer)
            cached = score_estimator_params_from_cache(estimator, {}, self.cache(estimator),
                         scoring=self.scorer, backend="numpy", n_jobs=1)
            np.testing.assert_allclose(direct, manual, atol=1e-11)
            np.testing.assert_allclose(cached, manual, atol=1e-11)
            original = cross_val_score(estimator, self.X, self.y, cv=self.cv, scoring="neg_mean_absolute_error")
            self.assertTrue(np.all(np.abs(original) > 1000 * np.abs(direct)))
            self.assertEqual(validation_error_from_score(-0.3, self.scorer), 0.3)

    def test_validation_values_never_fit_target_scaling(self):
        estimator = self.estimator("exact")
        cache = self.cache(estimator)
        _, val = next(self.cv.split(self.X))
        changed = self.y.copy()
        changed[val] += 1e8
        altered = self.cache(estimator, changed)
        np.testing.assert_array_equal(cache.folds[0].target_transformer.mean_, altered.folds[0].target_transformer.mean_)
        np.testing.assert_array_equal(cache.folds[0].target_transformer.scale_, altered.folds[0].target_transformer.scale_)

    @unittest.skipUnless(importlib.util.find_spec("torch"), "Torch unavailable")
    def test_torch_cached_scoring_matches_transformed_numpy_for_both_backends(self):
        for kind in ("exact", "nystrom"):
            estimator = self.estimator(kind)
            cache = self.cache(estimator)
            expected = score_estimator_params_from_cache(estimator, {}, cache, scoring=self.scorer, backend="numpy", n_jobs=1)
            actual = score_estimator_params_from_cache(estimator, {}, cache, scoring=self.scorer, backend="pytorch",
                                                      pytorch_device="cpu", pytorch_dtype="float64", pytorch_candidate_batch_size=2)
            np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-10)

    def test_run_records_transformed_objective_and_original_unit_named_targets(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            np.save(root / "x.npy", self.features)
            np.save(root / "y.npy", self.y)
            config = KRRConfig.from_mapping({
                "descriptors": [{"id": "x", "path": str(root / "x.npy")}],
                "target": {"path": str(root / "y.npy"), "name": ["first", "second"]},
                "split": {"mode": "random", "n_train": 16, "n_folds": 2},
                "search": {"target_space": "transformed", "random": {"stage1": 1, "stage2": 1},
                           "bayesian": {"trials": 1, "patience": 1}},
                "execution": {"n_jobs": 1, "prediction_batch_size": 3, "distance_cache": {"n_jobs": 1}},
                "output": {"directory": str(root / "output")}, "verbosity": 0})
            run(config)
            metrics = json.loads((root / "output/run_metrics.json").read_text())
            self.assertEqual(metrics["scoring_target_space"], "transformed")
            self.assertIsNone(metrics["cv_mae"])
            self.assertGreater(metrics["cv_transformed_mae"], 0)
            self.assertEqual(metrics["target_names"], ["first", "second"])
            self.assertEqual(metrics["n_targets"], 2)
            summary = regression_error_summary(np.load(root / "output/y_true.npy"), np.load(root / "output/y_predictions.npy"))
            np.testing.assert_allclose(metrics["target_mae"], summary["target_mae"])
            np.testing.assert_allclose(metrics["target_rmse"], summary["target_rmse"])
            self.assertGreater(metrics["target_mae"][1], 1000)


if __name__ == "__main__":
    unittest.main()
