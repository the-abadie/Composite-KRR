"""Behavior tests against explicit kernel formulas; Torch is an optional extra."""
import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
from scipy.spatial.distance import cdist
from sklearn.base import clone
from sklearn.compose import TransformedTargetRegressor
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import StandardScaler

from cached_scoring import resolve_candidate_hyperparameter_arrays, score_candidates_from_cache
from class_CompositeKRR import CompositeKRR, KernelComponent
from kernel_cache import build_distance_cache, composite_kernel_from_distances
from kernel_mixing import resolve_kernel_products
from nystrom_cache import build_nystrom_distance_cache
from nystrom_krr import CompositeNystromKRREstimator
from run_config import KRRConfig, template
from search_random import CompositeKRREstimator, staged_random_search_cv

TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None


def sample_matrix(blocks):
    X = np.empty((len(blocks[0]), len(blocks)), dtype=object)
    for i, block in enumerate(blocks):
        X[:, i] = list(block)
    return X


def explicit_kernel(left, right, types, gammas, weights, products):
    bases = [pairwise_kernels(a, b, metric=k, gamma=g)
             for a, b, k, g in zip(left, right, types, gammas)]
    terms = bases + [np.prod([bases[i] for i in indices], axis=0) for indices in products]
    return sum(weight * term for weight, term in zip(weights, terms))


class ProductKernelTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(4)
        self.blocks = [rng.normal(size=(18, d)) for d in (3, 2, 4, 1)]
        self.X = sample_matrix(self.blocks)
        self.y = np.column_stack((np.sin(self.blocks[0][:, 0]), self.blocks[1][:, 0] ** 2))
        self.types = ["rbf", "laplacian", "rbf", "laplacian"]
        self.gammas = [0.15, 0.4, 0.2, 0.6]
        self.products = [[0, 1], [2, 3], [0, 0], [0, 1, 2]]
        self.weights = [0.0, 0.2, 0.3, 0.0, 0.7, 0.4, 0.1, 0.8]
        self.cv = KFold(3, shuffle=True, random_state=9)

    def estimator(self, kind="exact", **kwargs):
        options = dict(alpha=0.2, gammas=self.gammas, kernel_weights=self.weights,
                       kernel_types=self.types, kernel_products=self.products,
                       names=["a", "b", "c", "d"], normalizations="none", compute_dtype="float64")
        options.update(kwargs)
        if kind == "nystrom":
            return CompositeNystromKRREstimator(**options, n_landmarks=18,
                                               landmark_selection="first", batch_size=5)
        return CompositeKRREstimator(**options, pytorch_predict_batch_size=4)

    def cache(self, kind="exact", products=None, normalized=False):
        options = dict(names=["a", "b", "c", "d"], kernel_types=self.types,
                       kernel_products=self.products if products is None else products,
                       normalizations=["standard" if normalized else "none"] * 4,
                       target_transformer=StandardScaler() if normalized else None, n_jobs=1)
        if kind == "nystrom":
            return build_nystrom_distance_cache(self.X, self.y, self.cv, **options,
                    n_landmarks=18, landmark_selection="first", batch_size=5,
                    eigenvalue_floor=1e-12)
        return build_distance_cache(self.X, self.y, self.cv, **options)

    def test_exact_and_full_landmark_predictions_match_explicit_solve(self):
        for kind in ("exact", "nystrom"):
            for dtype in ("float32", "float64"):
                for multioutput in (False, True):
                    with self.subTest(kind=kind, dtype=dtype, multioutput=multioutput):
                        train = [b[:12] for b in self.blocks]
                        test = [b[12:] for b in self.blocks]
                        y = self.y[:12] if multioutput else self.y[:12, 0]
                        estimator = clone(self.estimator(kind, compute_dtype=dtype)).fit(self.X[:12], y)
                        K = explicit_kernel(train, train, self.types, self.gammas, self.weights, self.products)
                        C = explicit_kernel(test, train, self.types, self.gammas, self.weights, self.products)
                        expected = C @ np.linalg.solve(K + 0.2 * np.eye(12), y)
                        actual = estimator.predict(self.X[12:])
                        self.assertEqual(actual.shape, expected.shape)
                        assert_allclose(actual, expected, atol=3e-5 if dtype == "float32" else 1e-11)

    def test_cached_assembly_reuses_bases_without_mutation(self):
        for dtype in (np.float32, np.float64):
            distances = [cdist(b[:7], b[7:], metric="sqeuclidean" if k == "rbf" else "cityblock").astype(dtype)
                         for b, k in zip(self.blocks, self.types)]
            originals = [d.copy() for d in distances]
            out = np.empty_like(distances[0])
            with patch("kernel_cache.np.exp", wraps=np.exp) as exp:
                actual = composite_kernel_from_distances(distances, gammas=self.gammas,
                        weights=self.weights, kernel_types=self.types, kernel_products=self.products, out=out)
            self.assertEqual(exp.call_count, 4)
            self.assertIs(actual, out)
            self.assertEqual(actual.dtype, dtype)
            expected = explicit_kernel([b[:7] for b in self.blocks], [b[7:] for b in self.blocks],
                                       self.types, self.gammas, self.weights, self.products)
            assert_allclose(actual, expected, atol=2e-7 if dtype == np.float32 else 1e-14)
            for old, new in zip(originals, distances):
                assert_array_equal(old, new)

    def test_direct_assembly_computes_each_base_once(self):
        components = [KernelComponent(str(i), g, w, k)
                      for i, (g, w, k) in enumerate(zip(self.gammas, self.weights, self.types))]
        model = CompositeKRR(components, 0.2, kernel_products=self.products, product_weights=self.weights[4:])
        with patch("class_CompositeKRR.pairwise_kernels", wraps=pairwise_kernels) as pairwise:
            actual = model._composite_kernel(self.blocks, self.blocks)
        self.assertEqual(pairwise.call_count, 4)
        assert_allclose(actual, explicit_kernel(self.blocks, self.blocks, self.types, self.gammas,
                                                self.weights, self.products), atol=1e-14)

    def test_cached_cv_matches_fold_fitted_preprocessing(self):
        for kind in ("exact", "nystrom"):
            with self.subTest(kind=kind):
                estimator = TransformedTargetRegressor(
                    regressor=self.estimator(kind, normalizations="standard", normalize_kernel_weights=True),
                    transformer=StandardScaler())
                cache = self.cache(kind, normalized=True)
                plain_cache = self.cache(kind, products=[], normalized=True)
                self.assertEqual(cache.nbytes, plain_cache.nbytes)
                candidates = [{}, {"regressor__gammas": [0.2] * 4,
                                    "regressor__kernel_weights": [0.0] * 4 + [1.0] * 4}]
                a, g, w = resolve_candidate_hyperparameter_arrays(estimator, candidates, cache)
                self.assertEqual(g.shape, (2, 4))
                self.assertEqual(w.shape, (2, 8))
                assert_allclose(w.sum(axis=1), 1.0)
                expected = np.array([cross_val_score(clone(estimator).set_params(**p), self.X, self.y,
                        cv=self.cv, scoring="neg_mean_absolute_error") for p in candidates])
                for n_jobs in (1, 2):
                    actual = score_candidates_from_cache(alphas=a, gammas=g, weights=w, cache=cache,
                            scoring="neg_mean_absolute_error", n_jobs=n_jobs)
                    assert_allclose(actual, expected, atol=1e-11)
                for fold in cache.folds:
                    self.assertEqual(len(fold.train_distances), 4)

    def test_additive_default_is_unchanged(self):
        for kind in ("exact", "nystrom"):
            est = self.estimator(kind, kernel_products=None, kernel_weights=self.weights[:4])
            est.fit(self.X[:12], self.y[:12])
            K = explicit_kernel([b[:12] for b in self.blocks], [b[:12] for b in self.blocks],
                                self.types, self.gammas, self.weights[:4], [])
            C = explicit_kernel([b[12:] for b in self.blocks], [b[:12] for b in self.blocks],
                                self.types, self.gammas, self.weights[:4], [])
            assert_allclose(est.predict(self.X[12:]), C @ np.linalg.solve(K + 0.2 * np.eye(12), self.y[:12]), atol=1e-11)

    def test_rejects_invalid_products_weights_and_cache_layout(self):
        for products in ("0*1", [0, 1], [[0]], [[0, 4]], [[-1, 1]], [[True, 1]],
                         [[0.0, 1]], [[0, 1], [1, 0]]):
            with self.subTest(products=products), self.assertRaises(ValueError):
                resolve_kernel_products(products, 4)
        for kind in ("exact", "nystrom"):
            for weights in ([1.0] * 4, [np.nan] * 8, [-1.0] * 8, [np.inf] * 8):
                with self.subTest(kind=kind, weights=weights), self.assertRaises(ValueError):
                    self.estimator(kind, kernel_weights=weights).fit(self.X, self.y)
        with self.assertRaisesRegex(ValueError, "layout"):
            resolve_candidate_hyperparameter_arrays(self.estimator(), [{"kernel_products": [[0, 2]]}], self.cache())

    def test_staged_search_tunes_one_weight_per_term(self):
        for kind, cached, bayes_batch in (("exact", True, 1), ("exact", False, 1),
                                          ("nystrom", True, 2), ("nystrom", False, 1)):
            with self.subTest(kind=kind, cached=cached, bayes_batch=bayes_batch):
                estimator = TransformedTargetRegressor(regressor=self.estimator(kind), transformer=StandardScaler())
                result = staged_random_search_cv(estimator, self.X, self.y, n_components=4,
                    alpha_bounds=(0.01, 1), gamma_bounds=(0.02, 1), n_iter_stage1=2,
                    n_iter_stage2=2, n_iter_stage3=2, n_trials_bayesian=2,
                    scoring="neg_mean_absolute_error", cv=self.cv, random_state=2, n_jobs=1,
                    use_distance_cache=cached, distance_cache_n_jobs=1, prefix="regressor__",
                    bayesian_batch_size=bayes_batch)
                self.assertEqual(len(result.best_params_["regressor__gammas"]), 4)
                self.assertEqual(len(result.best_params_["regressor__kernel_weights"]), 8)
                self.assertTrue(np.all(np.isfinite(result.best_estimator_.predict(self.X))))
                for stage in result.stages:
                    for params in stage.cv_results_["params"]:
                        if "regressor__kernel_weights" in params:
                            self.assertEqual(len(params["regressor__kernel_weights"]), 8)
                self.assertIn("kernel_weight_logit_7", result.bayesian_stage.study.best_trial.params)
                self.assertNotIn("gamma_4", result.bayesian_stage.study.best_trial.params)

    def test_validated_config_round_trip(self):
        values = template()
        values.update(X_NAMES=["a", "b"], X_PATHS=["a.npy", "b.npy"],
                      X_NORMS=["none", "none"], KRR_KERNEL_PRODUCTS=[[0, 1]])
        config = KRRConfig.from_mapping(values)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "config.json"
            path.write_text(config.canonical_json())
            self.assertEqual(config.sha256(), KRRConfig.load(path).sha256())
        values["KRR_KERNEL_PRODUCTS"] = [[0, 2]]
        with self.assertRaises(ValueError):
            KRRConfig.from_mapping(values)

    @unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is optional and is not installed")
    def test_torch_predictions_and_candidate_batches_match_numpy(self):
        import torch
        from pytorch_backend import composite_kernel_from_distances_pytorch
        devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
        for device in devices:
            for kind in ("exact", "nystrom"):
                with self.subTest(device=device, kind=kind):
                    expected = self.estimator(kind).fit(self.X[:12], self.y[:12]).predict(self.X[12:])
                    actual = self.estimator(kind, backend="pytorch", pytorch_device=device).fit(
                        self.X[:12], self.y[:12]).predict(self.X[12:])
                    assert_allclose(actual, expected, atol=1e-10)
                    cache = self.cache(kind, normalized=True)
                    estimator = TransformedTargetRegressor(regressor=self.estimator(kind, normalizations="standard"),
                                                           transformer=StandardScaler())
                    a, g, w = resolve_candidate_hyperparameter_arrays(estimator, [{}, {"regressor__alpha": 0.4}], cache)
                    expected_scores = score_candidates_from_cache(alphas=a, gammas=g, weights=w, cache=cache,
                                                                 scoring="neg_mean_absolute_error")
                    actual_scores = score_candidates_from_cache(alphas=a, gammas=g, weights=w, cache=cache,
                            scoring="neg_mean_absolute_error", backend="pytorch", pytorch_device=device,
                            pytorch_candidate_batch_size=2)
                    assert_allclose(actual_scores, expected_scores, atol=1e-10)
            distances = [cdist(b[:5], b[5:], metric="sqeuclidean" if k == "rbf" else "cityblock")
                         for b, k in zip(self.blocks, self.types)]
            tensors = [torch.as_tensor(d, device=device) for d in distances]
            expected = composite_kernel_from_distances(distances, gammas=self.gammas, weights=self.weights,
                        kernel_types=self.types, kernel_products=self.products)
            actual = composite_kernel_from_distances_pytorch(tensors, gammas=[self.gammas] * 2,
                    weights=[self.weights] * 2, kernel_types=self.types, kernel_products=self.products)
            assert_allclose(actual.cpu().numpy(), np.stack([expected, expected]), atol=1e-12)
            for tensor, distance in zip(tensors, distances):
                assert_array_equal(tensor.cpu().numpy(), distance)


if __name__ == "__main__":
    unittest.main()
