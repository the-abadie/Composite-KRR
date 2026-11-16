"""Asinh target mathematics, fold isolation, saved state and CUDA scoring parity."""
import importlib.util
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
from sklearn.base import clone
from sklearn.model_selection import cross_val_score

from cached_scoring import score_estimator_params_from_cache
from preprocess import AsinhIQRTransformer, make_target_preprocessor
from run_config import KRRConfig, run
import test_target_power as power_tests


class TargetAsinhTests(unittest.TestCase):
    def setUp(self):
        self.fixture = power_tests.TargetPowerTests()
        self.fixture.setUp()
        self.name = "asinh_iqr_standard"

    def test_explicit_formula_standardization_and_roundtrip(self):
        for y in (self.fixture.signed, self.fixture.positive, -self.fixture.positive):
            q25, q75 = np.quantile(y, [.25, .75], axis=0, method="linear")
            u = np.arcsinh(y / (q75 - q25))
            expected = (u - u.mean(axis=0)) / u.std(axis=0)
            transform = clone(make_target_preprocessor(self.name))
            actual = transform.fit_transform(y)
            np.testing.assert_allclose(actual, expected, atol=1e-12)
            np.testing.assert_allclose(transform.inverse_transform(actual), y, atol=1e-12)

    def test_zero_iqr_and_constant_columns(self):
        y = np.column_stack((np.r_[np.zeros(8), 9.], np.full(9, -3.), np.zeros(9)))
        transform = AsinhIQRTransformer().fit(y)
        np.testing.assert_array_equal(transform.scale_source_, ["std", "unit", "unit"])
        np.testing.assert_allclose(transform.scale_, [y[:, 0].std(), 1., 1.])
        pipeline = make_target_preprocessor(self.name)
        actual = pipeline.fit_transform(y)
        self.assertTrue(np.isfinite(actual).all())
        np.testing.assert_allclose(pipeline.inverse_transform(actual), y, atol=1e-12)
        with self.assertRaises(ValueError):
            transform.fit([[np.nan]])

    def test_validation_labels_cannot_fit_iqr_or_standardization(self):
        f = self.fixture
        original = f.cache(f.estimator(self.name), f.signed)
        altered_y = f.signed.copy()
        train, val = next(f.cv.split(f.X))
        altered_y[val] = altered_y[val] * 100 + 1000
        altered = f.cache(f.estimator(self.name), altered_y)
        left, right = original.folds[0].target_transformer, altered.folds[0].target_transformer
        for attribute in ("q25_", "q75_", "iqr_", "scale_"):
            np.testing.assert_array_equal(getattr(left.named_steps["asinh"], attribute),
                                          getattr(right.named_steps["asinh"], attribute))
        for attribute in ("mean_", "scale_"):
            np.testing.assert_array_equal(getattr(left.named_steps["scaler"], attribute),
                                          getattr(right.named_steps["scaler"], attribute))
        expected = np.quantile(f.signed[train], [.25, .75], axis=0)
        np.testing.assert_array_equal(left.named_steps["asinh"].scale_, expected[1] - expected[0])

    def test_numpy_cache_matches_sklearn_cv_in_original_units(self):
        f = self.fixture
        estimator = f.estimator(self.name)
        expected = cross_val_score(clone(estimator), f.X, f.signed, cv=f.cv,
            scoring="neg_mean_absolute_error", error_score="raise")
        actual = score_estimator_params_from_cache(estimator, {}, f.cache(estimator, f.signed),
            scoring="neg_mean_absolute_error", backend="numpy", n_jobs=1)
        np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-10)

    @unittest.skipUnless(importlib.util.find_spec("torch"), "Torch unavailable")
    def test_torch_inverse_and_cached_scores_match_numpy(self):
        import torch
        from pytorch_backend import _inverse_transform_tensor_pytorch
        f = self.fixture
        device = os.environ.get("KRR_TEST_TORCH_DEVICE", "cpu")
        for y in (f.signed, f.positive, -f.positive):
            transform = make_target_preprocessor(self.name).fit(y)
            values = np.linspace(-5, 5, 30).reshape(3, 5, 2)
            tensor = torch.tensor(values, dtype=torch.float64, device=device)
            actual = _inverse_transform_tensor_pytorch(transform, tensor).cpu().numpy()
            expected = np.stack([transform.inverse_transform(row) for row in values])
            np.testing.assert_allclose(actual, expected, rtol=1e-11, atol=1e-11)
            estimator = f.estimator(self.name)
            cache = f.cache(estimator, y)
            expected_scores = score_estimator_params_from_cache(estimator, {}, cache,
                scoring="neg_mean_absolute_error", backend="numpy", n_jobs=1)
            actual_scores = score_estimator_params_from_cache(estimator, {}, cache,
                scoring="neg_mean_absolute_error", backend="pytorch", pytorch_device=device,
                pytorch_dtype="float64", pytorch_candidate_batch_size=2)
            np.testing.assert_allclose(actual_scores, expected_scores, rtol=1e-10, atol=1e-10)

    def test_completed_run_saves_training_only_quartiles_and_scaling(self):
        f = self.fixture
        with TemporaryDirectory() as directory:
            root = Path(directory)
            np.save(root / "x.npy", f.features)
            np.save(root / "y.npy", f.signed[:, 0])
            config = KRRConfig.from_mapping({
                "descriptors": [{"id": "x", "path": str(root / "x.npy")}],
                "target": {"path": str(root / "y.npy"), "normalization": self.name},
                "split": {"mode": "random", "n_train": 30, "n_folds": 3},
                "search": {"random": {"stage1": 2, "stage2": 2, "stage3": 0},
                    "bayesian": {"trials": 0}, "alpha_bounds": [.1, 1.], "gamma_bounds": [.01, 1.]},
                "execution": {"n_jobs": 1, "distance_cache": {"n_jobs": 1}},
                "output": {"directory": str(root / "out")}, "verbosity": 0})
            run(config)
            metrics = json.loads((root / "out/run_metrics.json").read_text())
            self.assertNotIn("power_parameter_estimation", metrics["target_transform"])
            self.assertIn("training IQR", metrics["target_transform"]["scale_parameter_estimation"])
            train = np.load(root / "out/fold_val_idx.npy")
            fitted = make_target_preprocessor(self.name).fit(f.signed[train, :1])
            nonlinear, scaler = metrics["target_transform"]["steps"]
            for key in ("q25", "q75", "iqr", "scale", "scale_source"):
                np.testing.assert_array_equal(nonlinear[key], getattr(fitted.named_steps["asinh"], key + "_"))
            self.assertFalse(nonlinear["center"])
            np.testing.assert_array_equal(scaler["mean"], fitted.named_steps["scaler"].mean_)
            np.testing.assert_array_equal(scaler["scale"], fitted.named_steps["scaler"].scale_)


if __name__ == "__main__":
    unittest.main()
