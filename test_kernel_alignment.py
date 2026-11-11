"""Scientific formula, fitted-preprocessing, and reporting checks for CKA."""
import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
from sklearn.compose import TransformedTargetRegressor
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.preprocessing import StandardScaler

from kernel_alignment import ARTIFACT_NAMES, analyze_kernel_alignment, centered_kernel_alignment
from nystrom_krr import CompositeNystromKRREstimator
from search_random import CompositeKRREstimator


def reference_alignment(left, right):
    h = np.eye(len(left)) - np.ones_like(left) / len(left)
    a, b = h @ left @ h, h @ right @ h
    return np.sum(a * b) / (np.linalg.norm(a) * np.linalg.norm(b))


class CenteredAlignmentTests(unittest.TestCase):
    def test_matches_explicit_centering_and_is_scale_and_offset_invariant(self):
        rng = np.random.default_rng(31)
        a, b = rng.normal(size=(9, 3)), rng.normal(size=(9, 5))
        left, right = a @ a.T, b @ b.T
        original = left.copy()
        expected = reference_alignment(left, right)
        self.assertAlmostEqual(centered_kernel_alignment(left, right), expected, places=14)
        self.assertAlmostEqual(centered_kernel_alignment(left * 3 + 7, right / 5 + 2), expected, places=14)
        self.assertAlmostEqual(centered_kernel_alignment(left * 1e200, right * 1e-200), expected, places=14)
        assert_array_equal(left, original)

    def test_duplicate_orthogonal_and_indefinite_kernels(self):
        a, b = np.array([1., -1, 0, 0]), np.array([0., 0, 1, -1])
        left, right = np.outer(a, a), np.outer(b, b)
        self.assertAlmostEqual(centered_kernel_alignment(left, left), 1)
        self.assertAlmostEqual(centered_kernel_alignment(left, right), 0)
        self.assertAlmostEqual(centered_kernel_alignment(left, -left), -1)

    def test_constant_or_zero_is_undefined(self):
        for constant in (np.ones((4, 4)), np.zeros((4, 4))):
            self.assertIsNone(centered_kernel_alignment(constant, np.eye(4)))

    def test_rejects_invalid_matrices(self):
        for bad in (np.ones((2, 3)), np.ones(4), np.ones((1, 1)),
                    np.array([[1., 2], [3, 4]]), np.eye(3) * np.nan,
                    np.eye(3, dtype=complex)):
            with self.subTest(shape=bad.shape), self.assertRaises(ValueError):
                centered_kernel_alignment(bad, np.eye(3))
        with self.assertRaisesRegex(ValueError, "same shape"):
            centered_kernel_alignment(np.eye(3), np.eye(4))


class FittedAlignmentTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(16)
        self.blocks = [rng.normal(size=(18, 4)), rng.normal(size=(18, 3))]
        self.X = np.empty((18, 2), dtype=object)
        for i, block in enumerate(self.blocks):
            self.X[:, i] = list(block)
        self.y = self.blocks[0][:, 0] + self.blocks[1][:, 0]

    def estimator(self, kind="exact", **kwargs):
        options = dict(names=["a", "b"], kernel_types=["rbf", "laplacian"],
                       normalizations=["standard", "standard"], gammas=[0.2, 0.4],
                       alpha=0.1, kernel_products=[[0, 1], [0, 0]],
                       kernel_weights=[0., 0.2, 0.3, 0.5], pca_components=[2, None],
                       compute_dtype="float64")
        options.update(kwargs)
        cls = CompositeNystromKRREstimator if kind == "nystrom" else CompositeKRREstimator
        if kind == "nystrom":
            options.update(n_landmarks=8, batch_size=5, random_state=7)
        return cls(**options).fit(self.X, self.y)

    def test_uses_fitted_transforms_gammas_and_unweighted_product_terms(self):
        for kind in ("exact", "nystrom"):
            for dtype in ("float32", "float64"):
                with self.subTest(kind=kind, dtype=dtype):
                    fitted = self.estimator(kind, compute_dtype=dtype)
                    transforms = fitted.X_preprocessors_
                    with patch.object(transforms[0], "fit", side_effect=AssertionError("must not refit")):
                        report = analyze_kernel_alignment(fitted, self.X, max_samples=7,
                                                         sample_indices=np.arange(18) * 3, random_state=11)
                    positions = report["sample_positions"]
                    bases = [pairwise_kernels(np.asarray(t.transform(block[positions].astype(dtype)), dtype=dtype),
                              metric=k, gamma=g, filter_params=True).astype(float)
                             for t, block, k, g in zip(transforms, self.blocks,
                                                     fitted.kernel_types_, fitted.gammas_)]
                    terms = [*bases, bases[0] * bases[1], bases[0] ** 2]
                    expected = [[reference_alignment(a, b) for b in terms] for a in terms]
                    assert_allclose(report["alignment"], expected, atol=2e-7 if dtype == "float32" else 1e-14)
                    self.assertEqual(report["terms"][0]["fitted_weight"], 0.)
                    self.assertEqual(report["terms"][0]["status"], "ok")
                    self.assertEqual(report["terms"][3]["descriptor_ids"], ["a", "a"])
                    self.assertEqual(report["sample_indices"], [3 * p for p in positions])
                    self.assertEqual(report["labels"], ["a", "b", "a * b", "a * a"])
                    self.assertFalse(report["weighted"])

    def test_sampling_is_reproducible_and_does_not_consume_global_rng(self):
        fitted = self.estimator()
        before = np.random.get_state()
        first = analyze_kernel_alignment(fitted, self.X, max_samples=7, random_state=12)
        second = analyze_kernel_alignment(fitted, self.X, max_samples=7, random_state=12)
        after = np.random.get_state()
        self.assertEqual(first, second)
        assert_array_equal(before[1], after[1])
        self.assertEqual(before[2:], after[2:])
        changed = analyze_kernel_alignment(fitted, self.X, max_samples=7, random_state=123)
        self.assertNotEqual(first["sample_positions"], changed["sample_positions"])
        base_only = analyze_kernel_alignment(fitted, self.X, include_products=False)
        self.assertEqual(base_only["labels"], ["a", "b"])
        self.assertEqual(base_only["n_samples"], 18)

    def test_transformed_target_wrapper_and_single_component(self):
        model = TransformedTargetRegressor(
            regressor=CompositeKRREstimator(names=["a"], gammas=[0.2], compute_dtype="float64"),
            transformer=StandardScaler(),
        ).fit(self.X[:, :1], self.y)
        report = analyze_kernel_alignment(model, self.X[:, :1])
        assert_allclose(report["alignment"], [[1.]])

    def test_linear_baseline_does_not_change_kernel_geometry(self):
        plain = analyze_kernel_alignment(self.estimator(), self.X)
        residual = analyze_kernel_alignment(self.estimator(linear_baseline="a"), self.X)
        assert_allclose(plain["alignment"], residual["alignment"], atol=1e-14)

    def test_exports_constant_and_insufficient_sample_entries_as_null(self):
        fitted = self.estimator(gammas=[0., 0.4])
        with tempfile.TemporaryDirectory() as temporary:
            report = analyze_kernel_alignment(fitted, self.X, output_dir=temporary)
            root = Path(temporary)
            for name in ARTIFACT_NAMES:
                self.assertGreater((root / name).stat().st_size, 0)
            saved = json.loads((root / "kernel_alignment.json").read_text())
            self.assertEqual(report, saved)
            self.assertNotIn("NaN", (root / "kernel_alignment.json").read_text())
            self.assertEqual(saved["terms"][0]["status"], "constant")
            self.assertIsNone(saved["alignment"][0][0])
            self.assertEqual(saved["alignment"][1][2], 1.)
            with (root / "kernel_alignment.csv").open() as handle:
                rows = list(csv.reader(handle))
            self.assertEqual(rows[0], ["kernel", *report["labels"]])
            self.assertEqual(rows[1][1], "")
            assert_array_equal(np.load(root / "kernel_alignment_sample_indices.npy"), np.arange(18))
        small = analyze_kernel_alignment(fitted, self.X[:2])
        self.assertEqual(small["status"], "partial")
        self.assertTrue(all(v is None for row in small["alignment"] for v in row))

    def test_reports_identity_limit_and_rejects_invalid_options(self):
        fitted = self.estimator(gammas=[1e6, 1e6])
        report = analyze_kernel_alignment(fitted, self.X)
        self.assertTrue(any("nearly identity" in warning for warning in report["warnings"]))
        for kwargs in ({"max_samples": True}, {"max_samples": 2},
                       {"include_products": 1}, {"sample_indices": np.ones(18, dtype=int)},
                       {"sample_indices": np.arange(18.)}, {"sample_indices": [-1] * 18}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                analyze_kernel_alignment(fitted, self.X, **kwargs)
        with self.assertRaises(ValueError):
            analyze_kernel_alignment(fitted, self.X[:, :1])

    @unittest.skipUnless(importlib.util.find_spec("torch"), "Torch is optional")
    def test_torch_cpu_and_numpy_models_have_matching_alignment(self):
        cpu = analyze_kernel_alignment(self.estimator(), self.X)
        gpu_api = analyze_kernel_alignment(self.estimator(backend="pytorch", pytorch_device="cpu"), self.X)
        assert_allclose(cpu["alignment"], gpu_api["alignment"], atol=1e-14)


class AlignmentRunTests(unittest.TestCase):
    def test_configured_runs_record_only_training_rows_and_checksum_all_artifacts(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            rng = np.random.default_rng(61)
            for name in ("a", "b"):
                np.save(root / f"{name}.npy", rng.normal(size=(24, 6)))
            np.save(root / "y.npy", rng.normal(size=(24, 2)))
            train = np.array([10, 4, 22, 6, 18, 0, 14, 2, 20, 8, 16, 12])
            held = np.setdiff1d(np.arange(24), train)
            np.save(root / "train.npy", np.array([], dtype=int))
            np.savez(root / "folds.npz", fold0=train[:6], fold1=train[6:])
            np.save(root / "test.npy", held)
            common = {
                "seed": 12,
                "descriptors": [
                    {"id": "a", "path": "a.npy", "pca_components": 2},
                    {"id": "b", "path": "b.npy", "kernel": "laplacian",
                     "spatial_dct": {"block_size": 3, "n_components": 2}},
                ],
                "kernel_products": [["a", "b"], ["a", "a"]],
                "target": {"path": "y.npy", "name": ["one", "two"]},
                "split": {"mode": "predefined", "train_indices": "train.npy",
                          "validation_folds": "folds.npz", "test_indices": "test.npy"},
                "search": {"random": {"stage1": 1, "stage2": 1}, "bayesian": {"trials": 0}},
                "execution": {"n_jobs": 1, "distance_cache": {"n_jobs": 1}},
            }
            for mode in ("exact", "disabled", "nystrom"):
                config = json.loads(json.dumps(common))
                config["output"] = {"directory": mode}
                if mode != "disabled":
                    config["reporting"] = {"kernel_alignment": {
                        "enabled": True, "max_samples": 7, "include_products": mode == "exact",
                    }}
                if mode == "nystrom":
                    config["model"] = {"backend": "nystrom", "nystrom": {"n_landmarks": 5, "batch_size": 4}}
                path = root / f"{mode}.json"
                path.write_text(json.dumps(config))
                result = subprocess.run(
                    [sys.executable, str(Path(__file__).with_name("main.py")), "--config", str(path)],
                    capture_output=True, text=True, timeout=60,
                    env={**os.environ, "MPLBACKEND": "Agg", "OPENBLAS_NUM_THREADS": "1"},
                )
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                output = root / mode
                metrics = json.loads((output / "run_metrics.json").read_text())
                self.assertTrue(metrics["complete"])
                if mode == "disabled":
                    self.assertNotIn("kernel_alignment", metrics)
                    self.assertFalse((output / "kernel_alignment.json").exists())
                    continue
                report = json.loads((output / "kernel_alignment.json").read_text())
                self.assertEqual(report["n_samples"], 7)
                self.assertEqual(report["n_available"], 12)
                self.assertEqual(len(report["terms"]), 4 if mode == "exact" else 2)
                self.assertEqual(report["sample_indices"], train[report["sample_positions"]].tolist())
                self.assertFalse(np.intersect1d(report["sample_indices"], held).size)
                self.assertEqual(metrics["kernel_alignment"]["n_samples"], 7)
                self.assertGreater(metrics["kernel_alignment"]["elapsed_seconds"], 0.)
                for name in ARTIFACT_NAMES:
                    expected = hashlib.sha256((output / name).read_bytes()).hexdigest()
                    self.assertEqual(metrics["artifacts"][name]["sha256"], expected)
            # Diagnostics must leave the fitted solution and all test predictions unchanged.
            for name in ("y_predictions.npy", "alpha.npy", "gammas.npy", "kernel_weights.npy"):
                assert_array_equal(np.load(root / "exact" / name), np.load(root / "disabled" / name))


if __name__ == "__main__":
    unittest.main()
