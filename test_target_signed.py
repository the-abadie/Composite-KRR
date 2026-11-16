"""Signed target formulas, fold isolation, original-unit scoring and saved state."""
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
from preprocess import make_target_preprocessor
from run_config import KRRConfig, run
import test_target_power as power_tests


VARIANTS = ("signed_sqrt_standard", "cbrt_standard", "signed_log_iqr_standard", "asinh_median_iqr_standard")


def explicit_transform(name, y):
    if name == "signed_sqrt_standard":
        u = np.sign(y) * np.sqrt(np.abs(y))
    elif name == "cbrt_standard":
        u = np.cbrt(y)
    else:
        q25, q75 = np.quantile(y, [.25, .75], axis=0)
        sd = y.std(axis=0)
        scale = np.where(q75 > q25, q75 - q25, np.where(sd > 0, sd, 1.))
        if name == "signed_log_iqr_standard":
            u = np.sign(y) * np.log1p(np.abs(y) / scale)
        else:
            u = np.arcsinh((y - np.median(y, axis=0)) / scale)
    sd = np.where(np.ptp(u, axis=0) > 0, u.std(axis=0), 1.)
    return (u - u.mean(axis=0)) / sd


class SignedTargetTests(unittest.TestCase):
    def setUp(self):
        self.fixture = power_tests.TargetPowerTests()
        self.fixture.setUp()

    def test_formulas_cloning_and_roundtrip_for_every_sign_domain(self):
        f = self.fixture
        for name in VARIANTS:
            for y in (f.signed, f.positive, -f.positive):
                with self.subTest(name=name):
                    t = clone(make_target_preprocessor(name))
                    actual = t.fit_transform(y)
                    np.testing.assert_allclose(actual, explicit_transform(name, y), atol=1e-12)
                    np.testing.assert_allclose(t.inverse_transform(actual), y, atol=1e-11)

    def test_centering_precedes_nonlinearity(self):
        y = np.array([90., 95., 100., 110., 130.]).reshape(-1, 1)
        centered = make_target_preprocessor("asinh_median_iqr_standard")
        original = make_target_preprocessor("asinh_iqr_standard")
        z = centered.fit_transform(y)
        np.testing.assert_array_equal(centered.named_steps["asinh"].center_, [100.])
        self.assertFalse(np.allclose(z, original.fit_transform(y)))
        np.testing.assert_allclose(z, explicit_transform("asinh_median_iqr_standard", y), atol=1e-12)

    def test_degenerate_scales_and_zero_targets(self):
        y = np.column_stack((np.r_[np.zeros(8), -9.], np.full(9, 3.), np.zeros(9)))
        for name in VARIANTS:
            t = make_target_preprocessor(name)
            z = t.fit_transform(y)
            self.assertTrue(np.isfinite(z).all())
            np.testing.assert_allclose(z, explicit_transform(name, y), atol=1e-12)
            np.testing.assert_allclose(t.inverse_transform(z), y, atol=1e-12)
            if name in ("signed_log_iqr_standard", "asinh_median_iqr_standard"):
                np.testing.assert_array_equal(t.steps[0][1].scale_source_, ["std", "unit", "unit"])
                with self.assertRaises(ValueError):
                    t.fit([[np.nan]])

    def test_validation_targets_cannot_fit_scale_center_or_standardizer(self):
        f = self.fixture
        for name in VARIANTS:
            train, val = next(f.cv.split(f.X))
            altered_y = f.signed.copy()
            altered_y[val] = altered_y[val] * 100 + 1000
            left = f.cache(f.estimator(name), f.signed).folds[0].target_transformer
            right = f.cache(f.estimator(name), altered_y).folds[0].target_transformer
            for attr in ("mean_", "scale_"):
                np.testing.assert_array_equal(getattr(left.named_steps["scaler"], attr), getattr(right.named_steps["scaler"], attr))
            if name in ("signed_log_iqr_standard", "asinh_median_iqr_standard"):
                for attr in ("q25_", "q75_", "iqr_", "scale_", "center_"):
                    np.testing.assert_array_equal(getattr(left.steps[0][1], attr), getattr(right.steps[0][1], attr))
            if name == "asinh_median_iqr_standard":
                np.testing.assert_array_equal(left.steps[0][1].center_, np.median(f.signed[train], axis=0))

    def test_numpy_cache_matches_independent_sklearn_cv(self):
        f = self.fixture
        for name in VARIANTS:
            estimator = f.estimator(name)
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
        for name in VARIANTS:
            for y in (f.signed, f.positive, -f.positive):
                transformer = make_target_preprocessor(name).fit(y)
                values = np.linspace(-5, 5, 30).reshape(3, 5, 2)
                actual = _inverse_transform_tensor_pytorch(transformer, torch.tensor(values, dtype=torch.float64, device=device))
                expected = np.stack([transformer.inverse_transform(row) for row in values])
                np.testing.assert_allclose(actual.cpu().numpy(), expected, rtol=1e-11, atol=1e-11)
                estimator = f.estimator(name)
                cache = f.cache(estimator, y)
                expected_scores = score_estimator_params_from_cache(estimator, {}, cache,
                    scoring="neg_mean_absolute_error", backend="numpy", n_jobs=1)
                actual_scores = score_estimator_params_from_cache(estimator, {}, cache,
                    scoring="neg_mean_absolute_error", backend="pytorch", pytorch_device=device,
                    pytorch_dtype="float64", pytorch_candidate_batch_size=2)
                np.testing.assert_allclose(actual_scores, expected_scores, rtol=1e-10, atol=1e-10)

    def test_completed_runs_retain_exact_fitted_parameters(self):
        f = self.fixture
        with TemporaryDirectory() as directory:
            root = Path(directory)
            np.save(root / "x.npy", f.features)
            np.save(root / "y.npy", f.signed[:, 0])
            for name in VARIANTS:
                output = root / name
                cfg = KRRConfig.from_mapping({
                    "descriptors": [{"id": "x", "path": str(root / "x.npy")}],
                    "target": {"path": str(root / "y.npy"), "normalization": name},
                    "split": {"mode": "random", "n_train": 30, "n_folds": 3},
                    "search": {"random": {"stage1": 2, "stage2": 2, "stage3": 0},
                        "bayesian": {"trials": 0}, "alpha_bounds": [.1, 1.], "gamma_bounds": [.01, 1.]},
                    "execution": {"n_jobs": 1, "distance_cache": {"n_jobs": 1}},
                    "output": {"directory": str(output)}, "verbosity": 0})
                run(cfg)
                record = json.loads((output / "run_metrics.json").read_text())["target_transform"]
                self.assertNotIn("power_parameter_estimation", record)
                train = np.load(output / "fold_val_idx.npy")
                fitted = make_target_preprocessor(name).fit(f.signed[train, :1])
                nonlinear, scaler = record["steps"]
                np.testing.assert_array_equal(scaler["mean"], fitted.named_steps["scaler"].mean_)
                np.testing.assert_array_equal(scaler["scale"], fitted.named_steps["scaler"].scale_)
                if name in ("signed_log_iqr_standard", "asinh_median_iqr_standard"):
                    np.testing.assert_array_equal(nonlinear["scale"], fitted.steps[0][1].scale_)
                    np.testing.assert_array_equal(nonlinear["offset"], fitted.steps[0][1].center_)
                    self.assertEqual(nonlinear["center"], name == "asinh_median_iqr_standard")


if __name__ == "__main__":
    unittest.main()
