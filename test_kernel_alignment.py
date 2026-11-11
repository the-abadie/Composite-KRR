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
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from kernel_alignment import (ARTIFACT_NAMES, TARGET_ARTIFACT_NAMES,
                              analyze_kernel_alignment, centered_kernel_alignment)
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
            for mode in ("exact", "disabled", "pairwise_only", "nystrom"):
                config = json.loads(json.dumps(common))
                config["output"] = {"directory": mode}
                if mode != "disabled":
                    config["reporting"] = {"kernel_alignment": {
                        "enabled": True, "max_samples": 7, "include_products": mode == "exact",
                        "include_target": mode != "pairwise_only",
                        "target_space": "transformed" if mode == "nystrom" else "original",
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
                artifacts = ARTIFACT_NAMES
                if mode == "pairwise_only":
                    self.assertNotIn("target_alignment", report)
                    self.assertFalse((output / TARGET_ARTIFACT_NAMES[0]).exists())
                else:
                    target = report["target_alignment"]
                    self.assertEqual(target["target_names"], ["one", "two"])
                    self.assertEqual(target["target_space"], "transformed" if mode == "nystrom" else "original")
                    self.assertEqual(metrics["kernel_alignment"]["target_alignment"], target)
                    self.assertEqual(np.shape(target["alignment"]), (len(report["terms"]), 2))
                    # Reconstruct the first fitted descriptor kernel independently,
                    # and compare it with labels at the saved GLOBAL sample IDs.
                    from preprocess import make_data_preprocessor
                    transform = make_data_preprocessor("standard", pca_components=2)
                    block = transform.fit_transform(np.load(root / "a.npy")[train])
                    selected_block = block[report["sample_positions"]]
                    kernel = pairwise_kernels(selected_block, metric="rbf", gamma=report["gammas"][0])
                    labels = np.load(root / "y.npy")[report["sample_indices"]]
                    expected = [reference_alignment(kernel, np.outer(column, column)) for column in labels.T]
                    assert_allclose(target["alignment"][0], expected, atol=1e-14)
                    artifacts += TARGET_ARTIFACT_NAMES
                for name in artifacts:
                    expected = hashlib.sha256((output / name).read_bytes()).hexdigest()
                    self.assertEqual(metrics["artifacts"][name]["sha256"], expected)
            # Diagnostics must leave the fitted solution and all test predictions unchanged.
            for name in ("y_predictions.npy", "alpha.npy", "gammas.npy", "kernel_weights.npy"):
                assert_array_equal(np.load(root / "exact" / name), np.load(root / "disabled" / name))
                assert_array_equal(np.load(root / "exact" / name), np.load(root / "pairwise_only" / name))


class KernelTargetAlignmentTests(unittest.TestCase):
    def setUp(self):
        self.fixture = FittedAlignmentTests()
        self.fixture.setUp()

    def test_scalar_alignment_matches_explicit_outer_product_for_all_terms(self):
        case = self.fixture
        for kind in ("exact", "nystrom"):
            model = case.estimator(kind)
            report = analyze_kernel_alignment(model, case.X, y=case.y, include_target=True,
                                             max_samples=7, random_state=9)
            positions = report["sample_positions"]
            blocks = [t.transform(block[positions]) for t, block in zip(model.X_preprocessors_, case.blocks)]
            bases = [pairwise_kernels(block, metric=k, gamma=g, filter_params=True)
                     for block, k, g in zip(blocks, model.kernel_types_, model.gammas_)]
            terms = [*bases, bases[0] * bases[1], bases[0] ** 2]
            target = np.outer(case.y[positions], case.y[positions])
            expected = [[reference_alignment(kernel, target)] for kernel in terms]
            assert_allclose(report["target_alignment"]["alignment"], expected, atol=1e-14)
            self.assertEqual(report["target_alignment"]["target_names"], ["target"])
            self.assertFalse(report["target_alignment"]["baseline_residualized"])
            pairwise_only = analyze_kernel_alignment(model, case.X, max_samples=7, random_state=9)
            self.assertEqual(report["alignment"], pairwise_only["alignment"])

    def test_multiple_targets_are_separate_and_invariant_to_units_and_offsets(self):
        case = self.fixture
        y = np.column_stack((case.y, -5 * case.y + 20, np.ones(18)))
        report = analyze_kernel_alignment(case.estimator(), case.X, y=y, include_target=True,
                                         target_names=["energy", "rescaled", "constant"])
        scores = np.array(report["target_alignment"]["alignment"], dtype=float)
        assert_allclose(scores[:, 0], scores[:, 1], atol=1e-14)
        self.assertTrue(np.isnan(scores[:, 2]).all())
        self.assertEqual(report["target_alignment"]["targets"][2]["status"], "constant")
        self.assertEqual(report["status"], "partial")
        self.assertTrue(any("constant" in warning for warning in report["warnings"]))
        for scale in (1e200, 1e-200):
            rescaled = analyze_kernel_alignment(case.estimator(), case.X,
                                               y=case.y * scale, include_target=True)
            assert_allclose(np.asarray(rescaled["target_alignment"]["alignment"])[:, 0],
                            scores[:, 0], atol=1e-14)
        # The entire pool varies; the deterministic sampled rows happen to be constant.
        selected = analyze_kernel_alignment(case.estimator(), case.X, max_samples=3)["sample_positions"]
        y = case.y.copy()
        y[selected] = 1
        report = analyze_kernel_alignment(case.estimator(), case.X, y=y, include_target=True, max_samples=3)
        self.assertEqual(report["target_alignment"]["targets"][0]["status"], "constant")

    def test_transformed_target_alignment_reuses_fitted_transform_without_refitting(self):
        case = self.fixture
        positive = np.exp(case.y / 5 + 1)
        model = TransformedTargetRegressor(
            regressor=CompositeKRREstimator(compute_dtype="float64", gammas=[0.2, 0.4]),
            transformer=FunctionTransformer(np.log, np.exp),
        ).fit(case.X, positive)
        with patch.object(model.transformer_, "fit", side_effect=AssertionError("must not refit")):
            transformed = analyze_kernel_alignment(model, case.X, y=positive, include_target=True,
                                                   target_space="transformed")
        expected = analyze_kernel_alignment(model, case.X, y=np.log(positive), include_target=True)
        assert_allclose(transformed["target_alignment"]["alignment"], expected["target_alignment"]["alignment"], atol=1e-14)
        original = analyze_kernel_alignment(model, case.X, y=positive, include_target=True)
        self.assertGreater(np.max(np.abs(np.array(original["target_alignment"]["alignment"]) -
                                        np.array(transformed["target_alignment"]["alignment"]))), 1e-4)
        self.assertEqual(transformed["target_alignment"]["target_space"], "transformed")

    def test_degenerate_kernels_tiny_pools_and_target_artifact_exports(self):
        case = self.fixture
        model = case.estimator(gammas=[0., 0.4])
        with tempfile.TemporaryDirectory() as temporary:
            report = analyze_kernel_alignment(model, case.X, y=case.y, include_target=True,
                                             target_names="energy", output_dir=temporary)
            self.assertIsNone(report["target_alignment"]["alignment"][0][0])
            root = Path(temporary)
            for name in (*ARTIFACT_NAMES, *TARGET_ARTIFACT_NAMES):
                self.assertGreater((root / name).stat().st_size, 0)
            saved = json.loads((root / "kernel_alignment.json").read_text())
            self.assertEqual(saved, report)
            with (root / "kernel_target_alignment.csv").open() as handle:
                rows = list(csv.reader(handle))
            self.assertEqual(rows[0], ["kernel", "energy"])
            self.assertEqual(rows[1], ["a", ""])
            pairwise_only_dir = root / "pairwise_only"
            disabled = analyze_kernel_alignment(model, case.X, output_dir=pairwise_only_dir)
            self.assertNotIn("target_alignment", disabled)
            self.assertFalse((pairwise_only_dir / TARGET_ARTIFACT_NAMES[0]).exists())
        small = analyze_kernel_alignment(model, case.X[:2], y=case.y[:2], include_target=True)
        self.assertEqual(small["target_alignment"]["targets"][0]["status"], "insufficient_samples")
        self.assertTrue(all(row == [None] for row in small["target_alignment"]["alignment"]))

    def test_invalid_target_requests_fail_clearly(self):
        case = self.fixture
        model = case.estimator()
        for kwargs in ({}, {"y": np.ones(17)}, {"y": np.ones((18, 0))},
                       {"y": case.y * np.nan}, {"y": case.y.astype(complex)},
                       {"y": case.y, "target_names": ["one", "two"]},
                       {"y": case.y, "target_names": [""]},
                       {"y": case.y, "target_space": "transformed"},
                       {"y": case.y, "target_space": "invalid"},
                       {"y": case.y, "include_target": 1}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                analyze_kernel_alignment(model, case.X, **{"include_target": True, **kwargs})


if __name__ == "__main__":
    unittest.main()
