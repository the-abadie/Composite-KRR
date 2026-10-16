"""Leakage, kernel-equivalence, cache, inference and artifact checks for DCT."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
from scipy.fft import dct
from scipy.spatial.distance import cdist
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

from config_schema import DescriptorConfig
from kernel_cache import build_distance_cache
from preprocess import SpatialDCT, descriptor_preprocessing, make_data_preprocessor
from run_config import KRRConfig
from saved_model import SavedExactModel
from search_random import CompositeKRREstimator


def samples(block):
    X = np.empty((len(block), 1), dtype=object)
    for i, row in enumerate(block):
        X[i, 0] = row
    return X


class SpatialDCTTests(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(27)
        self.x = self.rng.normal(size=(24, 12)) * np.geomspace(.01, 100, 12)
        self.x[:, 2] = 1
        self.recipe = {"normalization": "standard", "spatial_dct": {"block_size": 6, "n_components": 2}}

    def test_full_transform_preserves_training_and_cross_kernel_metrics(self):
        train, held = self.x[:16], self.x[16:]+1000
        scaler = StandardScaler().fit(train)
        reference_train, reference_held = scaler.transform(train), scaler.transform(held)
        pipeline = make_data_preprocessor({**self.recipe, "spatial_dct": {"block_size": 6, "n_components": 6}})
        actual_train, actual_held = pipeline.fit_transform(train), pipeline.transform(held)
        np.testing.assert_array_equal(pipeline.named_steps["scaler"].mean_, scaler.mean_)
        for a, b, left, right in ((actual_train, actual_train, reference_train, reference_train),
                                  (actual_held, actual_train, reference_held, reference_train)):
            da, db = cdist(a, b, "sqeuclidean"), cdist(left, right, "sqeuclidean")
            np.testing.assert_allclose(da, db, rtol=2e-14, atol=2e-10)
            np.testing.assert_allclose(np.exp(-.01*da), np.exp(-.01*db), atol=2e-14)

    def test_truncation_and_every_cached_fold_use_training_statistics_only(self):
        cv = KFold(3, shuffle=True, random_state=3)
        cache = build_distance_cache(samples(self.x), self.rng.normal(size=24), cv,
            names=["a"], kernel_types=["rbf"], normalizations=[self.recipe], n_jobs=1)
        for fold, (train, val) in zip(cache.folds, cv.split(self.x), strict=True):
            scaler = StandardScaler().fit(self.x[train])
            def transformed(rows):
                return dct(scaler.transform(self.x[rows]).reshape(len(rows), 2, 6),
                    type=2, norm="ortho", axis=-1)[..., :2].reshape(len(rows), 4)
            a, b = transformed(train), transformed(val)
            np.testing.assert_allclose(fold.train_distances[0], cdist(a, a, "sqeuclidean"), atol=1e-12)
            np.testing.assert_allclose(fold.validation_train_distances[0], cdist(b, a, "sqeuclidean"), atol=1e-12)

    def test_saved_inference_rebuilds_the_same_fold_fitted_transform(self):
        y = self.rng.normal(size=16)
        estimator = CompositeKRREstimator(names=["a"], kernel_types=["rbf"], normalizations=[self.recipe],
            compute_dtype="float64", gammas=[.2], kernel_weights=[1.], alpha=.1).fit(samples(self.x[:16]), y)
        spec = KRRConfig.from_mapping({"descriptors": [{"id": "a", "path": "a.npy", **self.recipe}],
            "target": {"path": "y.npy", "normalization": "none"}, "split": {"mode": "random", "n_train": 16},
            "output": {"directory": "unused"}}).spec
        saved = SavedExactModel(spec, [self.x[:16]], y, estimator.model_.dual_coef_, [.2], [1.], .1)
        np.testing.assert_allclose(saved.predict([self.x[16:]]), estimator.predict(samples(self.x[16:])), atol=2e-14)
        self.assertEqual(estimator.transformed_feature_counts_, [4])

    def test_post_scaling_cache_and_inference_use_only_training_coefficients(self):
        recipe = {**self.recipe, "spatial_dct": {**self.recipe["spatial_dct"], "post_normalization": "standard"}}
        cv = KFold(3, shuffle=True, random_state=3)
        cache = build_distance_cache(samples(self.x), self.rng.normal(size=24), cv,
            names=["a"], kernel_types=["rbf"], normalizations=[recipe], n_jobs=1)
        for fold, (train, val) in zip(cache.folds, cv.split(self.x), strict=True):
            first = StandardScaler().fit(self.x[train])
            def project(rows):
                return dct(first.transform(self.x[rows]).reshape(len(rows), 2, 6),
                    type=2, norm="ortho", axis=-1)[..., :2].reshape(len(rows), 4)
            last = StandardScaler().fit(project(train))
            a, b = last.transform(project(train)), last.transform(project(val))
            np.testing.assert_allclose(fold.train_distances[0], cdist(a, a, "sqeuclidean"), atol=1e-12)
            np.testing.assert_allclose(fold.validation_train_distances[0], cdist(b, a, "sqeuclidean"), atol=1e-12)
        y = self.rng.normal(size=16)
        estimator = CompositeKRREstimator(names=["a"], kernel_types=["rbf"], normalizations=[recipe],
            compute_dtype="float64", gammas=[.2], kernel_weights=[1.], alpha=.1).fit(samples(self.x[:16]), y)
        spec = KRRConfig.from_mapping({"descriptors": [{"id": "a", "path": "a.npy", **recipe}],
            "target": {"path": "y.npy", "normalization": "none"}, "split": {"mode": "random", "n_train": 16},
            "output": {"directory": "unused"}}).spec
        saved = SavedExactModel(spec, [self.x[:16]], y, estimator.model_.dual_coef_, [.2], [1.], .1)
        held = self.x[16:]+1000
        np.testing.assert_allclose(saved.predict([held]), estimator.predict(samples(held)), atol=2e-14)
        with self.assertRaises(ValueError):
            DescriptorConfig(id="a", path="a.npy", spatial_dct={**recipe["spatial_dct"], "post_normalization": "whiten"})

    def test_invalid_widths_and_legacy_loss_are_rejected(self):
        for values in ({"block_size": 6, "n_components": 7}, {"block_size": True, "n_components": 2}):
            with self.assertRaises(ValueError):
                DescriptorConfig(id="a", path="a.npy", spatial_dct=values)
        with self.assertRaises(ValueError):
            SpatialDCT(block_size=5, n_components=2).fit(self.x)
        config = KRRConfig.from_mapping({"descriptors": [{"id": "a", "path": "a.npy", **self.recipe}],
            "target": {"path": "y.npy"}, "split": {"mode": "random", "n_train": 16}, "output": {"directory": "unused"}})
        self.assertEqual(descriptor_preprocessing(config.spec.descriptors[0]), self.recipe)
        with self.assertRaisesRegex(ValueError, "structured"):
            _ = config.values

    def test_cli_records_input_and_effective_feature_counts(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            np.save(root/"a.npy", self.x)
            np.save(root/"y.npy", self.rng.normal(size=24))
            config = {"descriptors": [{"id": "a", "path": "a.npy", **self.recipe}],
                "target": {"path": "y.npy"}, "split": {"mode": "random", "n_train": 18, "n_folds": 3},
                "execution": {"n_jobs": 1, "distance_cache": {"n_jobs": 1}},
                "search": {"random": {"stage1": 3, "stage2": 2, "stage3": 1}, "bayesian": {"trials": 0}},
                "output": {"directory": "output"}}
            (root/"run.json").write_text(json.dumps(config))
            result = subprocess.run([sys.executable, str(Path(__file__).with_name("main.py")), "--config", str(root/"run.json")],
                capture_output=True, text=True, env={**os.environ, "MPLBACKEND": "Agg", "OPENBLAS_NUM_THREADS": "1"})
            self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
            metrics = json.loads((root/"output/run_metrics.json").read_text())
            self.assertEqual((metrics["features"], metrics["kernel_features"]), (12, 4))


if __name__ == "__main__":
    unittest.main()
