"""Independent sklearn/NumPy/Torch parity and leakage checks for power targets."""
import importlib.util
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
from sklearn.base import clone
from sklearn.compose import TransformedTargetRegressor
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import PowerTransformer

from estimator_factory import make_composite_krr_regressor
from cached_scoring import score_estimator_params_from_cache
from kernel_cache import score_predictions
from preprocess import make_target_preprocessor
from run_config import KRRConfig, run
from search_random import _maybe_build_distance_cache


class TargetPowerTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(319)
        self.features = rng.normal(size=(42, 3))
        self.X = np.empty((42, 1), dtype=object)
        self.X[:, 0] = list(self.features)
        self.positive = np.exp(.25 * self.features[:, :2])
        self.signed = np.column_stack((self.features[:, 0], self.features[:, 1] ** 3))
        self.cv = KFold(3, shuffle=True, random_state=12)

    def cases(self):
        return (("boxcox_standard", self.positive), ("yeojohnson_standard", self.signed),
                ("negate_boxcox_standard", -self.positive), ("negate_log_standard", -self.positive))

    def estimator(self, name):
        reg = make_composite_krr_regressor(krr_backend="exact", names=["x"],
            kernel_types=["rbf"], normalizations=["standard"], pca_components=None,
            pca_whiten=False, normalize_kernel_weights=True, compute_dtype="float64")
        reg.set_params(alpha=.2, gammas=[.1])
        return TransformedTargetRegressor(regressor=reg, transformer=make_target_preprocessor(name))

    def cache(self, estimator, y):
        return _maybe_build_distance_cache(estimator, self.X, y, self.cv, n_components=1,
            block_size=16, dtype="float64", n_jobs=1, memory_fraction=.8)

    def test_schema_standardization_roundtrip_and_domain(self):
        for name, y in self.cases():
            cfg = KRRConfig.from_mapping({"descriptors": [{"id": "x", "path": "x.npy"}],
                "target": {"path": "y.npy", "normalization": name},
                "split": {"mode": "random", "n_train": 20}, "output": {"directory": "out"}})
            self.assertEqual(cfg.spec.target.normalization, name)
            transformer = make_target_preprocessor(name)
            z = transformer.fit_transform(y)
            np.testing.assert_allclose(z.mean(0), 0, atol=1e-12)
            np.testing.assert_allclose(z.std(0), 1, atol=1e-12)
            np.testing.assert_allclose(transformer.inverse_transform(z), y, atol=1e-11)
        with self.assertRaises(ValueError):
            make_target_preprocessor("boxcox_standard").fit(self.signed)

    def test_validation_labels_cannot_fit_power_or_scaling(self):
        for name, y in self.cases():
            estimator = self.estimator(name)
            original = self.cache(estimator, y)
            altered_y = y.copy()
            _, val = next(self.cv.split(self.X))
            altered_y[val] *= 3
            altered = self.cache(estimator, altered_y)
            left, right = original.folds[0].target_transformer, altered.folds[0].target_transformer
            if name.startswith("negate_"):
                left, right = left.named_steps["transform"], right.named_steps["transform"]
            if name == "negate_log_standard":
                left, right = left.named_steps["scaler"], right.named_steps["scaler"]
                np.testing.assert_array_equal(left.mean_, right.mean_)
                np.testing.assert_array_equal(left.scale_, right.scale_)
                continue
            for a, b in ((left.lambdas_, right.lambdas_),
                         (left._scaler.mean_, right._scaler.mean_), (left._scaler.scale_, right._scaler.scale_)):
                np.testing.assert_array_equal(a, b)

    def test_numpy_cache_matches_independent_sklearn_cv(self):
        for name, y in self.cases():
            estimator = self.estimator(name)
            expected = cross_val_score(clone(estimator), self.X, y, cv=self.cv,
                scoring="neg_mean_absolute_error", error_score="raise")
            actual = score_estimator_params_from_cache(estimator, {}, self.cache(estimator, y),
                scoring="neg_mean_absolute_error", backend="numpy", n_jobs=1)
            np.testing.assert_allclose(actual, expected, atol=1e-10)

    @unittest.skipUnless(importlib.util.find_spec("torch"), "Torch unavailable")
    def test_torch_scores_match_numpy_original_units(self):
        for name, y in self.cases():
            estimator = self.estimator(name)
            cache = self.cache(estimator, y)
            expected = score_estimator_params_from_cache(estimator, {}, cache,
                scoring="neg_mean_absolute_error", backend="numpy", n_jobs=1)
            actual = score_estimator_params_from_cache(estimator, {}, cache,
                scoring="neg_mean_absolute_error", backend="pytorch",
                pytorch_device=os.environ.get("KRR_TEST_TORCH_DEVICE", "cpu"),
                pytorch_dtype="float64", pytorch_candidate_batch_size=2)
            np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-10)

    @unittest.skipUnless(importlib.util.find_spec("torch"), "Torch unavailable")
    def test_torch_inverse_matches_sklearn_special_powers_and_domains(self):
        import torch
        from pytorch_backend import _inverse_transform_tensor_pytorch
        for method in ("box-cox", "yeo-johnson"):
            for standardize in (False, True):
                for power in (-1., 0., .5, 1., 2., 3.):
                    transformer = PowerTransformer(method=method, standardize=standardize).fit(self.positive)
                    transformer.lambdas_[:] = power
                    values = np.linspace(-2, 2, 30).reshape(3, 5, 2)
                    with np.errstate(invalid="ignore", divide="ignore"):
                        expected = np.stack([transformer.inverse_transform(row) for row in values])
                    tensor = torch.tensor(values, device=os.environ.get("KRR_TEST_TORCH_DEVICE", "cpu"))
                    actual = _inverse_transform_tensor_pytorch(transformer, tensor).cpu().numpy()
                    np.testing.assert_allclose(actual, expected, rtol=1e-11, atol=1e-11, equal_nan=True)

    def test_invalid_inverse_prediction_is_ineligible(self):
        self.assertTrue(np.isnan(score_predictions([1., 2.], [1., np.nan], "neg_mean_absolute_error")))

    def test_completed_run_retains_exact_final_power_and_scaling(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            np.save(root / "x.npy", self.features)
            np.save(root / "y.npy", self.positive[:, 0])
            config = KRRConfig.from_mapping({
                "descriptors": [{"id": "x", "path": str(root / "x.npy")}],
                "target": {"path": str(root / "y.npy"), "normalization": "boxcox_standard"},
                "split": {"mode": "random", "n_train": 30, "n_folds": 3},
                "search": {"random": {"stage1": 2, "stage2": 2, "stage3": 0},
                           "bayesian": {"trials": 0}, "alpha_bounds": [.1, 1.], "gamma_bounds": [.01, 1.]},
                "execution": {"n_jobs": 1, "distance_cache": {"n_jobs": 1}},
                "output": {"directory": str(root / "out")}, "verbosity": 0})
            run(config)
            metrics = json.loads((root / "out/run_metrics.json").read_text())
            train = np.load(root / "out/fold_val_idx.npy")
            fitted = make_target_preprocessor("boxcox_standard").fit(self.positive[train, :1])
            record = metrics["target_transform"]
            np.testing.assert_array_equal(record["lambdas"], fitted.lambdas_)
            np.testing.assert_array_equal(record["mean"], fitted._scaler.mean_)
            np.testing.assert_array_equal(record["scale"], fitted._scaler.scale_)


if __name__ == "__main__":
    unittest.main()
