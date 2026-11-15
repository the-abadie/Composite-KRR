"""Search provenance, failure recovery evidence and independent-RNG integration."""
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
from sklearn.compose import TransformedTargetRegressor
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

from run_config import KRRConfig, run, template
from search_random import CompositeKRREstimator, staged_random_search_cv
from search_bayes import fit_bayesian_search
from search_evidence import MeaningfulProgress, SearchJournal


def read_events(path):
    def reject(value):
        raise AssertionError(f"Nonstandard JSON value {value}")
    return [json.loads(line, parse_constant=reject) for line in Path(path).read_text().splitlines()]


class SearchEvidenceTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(17)
        self.features = rng.normal(size=(24, 3))
        self.X = np.empty((24, 2), dtype=object)
        for i, row in enumerate(self.features):
            self.X[i, 0], self.X[i, 1] = row, row[:2] ** 2
        self.y = np.sin(self.features[:, 0]) + .3 * self.features[:, 1] ** 2
        self.cv = KFold(3, shuffle=True, random_state=7)

    def estimator(self):
        return TransformedTargetRegressor(regressor=CompositeKRREstimator(
            names=['first', 'second'], kernel_types=['rbf', 'laplacian'],
            kernel_products=[[0, 1]], normalizations=['standard', 'standard'],
            normalize_kernel_weights=True, compute_dtype='float64'), transformer=StandardScaler())

    def options(self):
        return dict(n_components=2, alpha_bounds=(1e-6, .1), gamma_bounds=(1e-6, 100.),
                    n_iter_stage1=4, n_iter_stage2=3, n_iter_stage3=2,
                    scoring='neg_mean_absolute_error', cv=self.cv, random_state=99,
                    n_jobs=1, distance_cache_n_jobs=1, refit=False,
                    prefix='regressor__', n_trials_bayesian=4, bayesian_batch_size=2,
                    history_batch_size=2)

    def test_complete_history_matches_scores_and_logging_preserves_candidates(self):
        with TemporaryDirectory() as directory:
            result = staged_random_search_cv(self.estimator(), self.X, self.y,
                                           search_output_dir=directory, **self.options())
            unrecorded = staged_random_search_cv(self.estimator(), self.X, self.y, **self.options())
            self.assertEqual(result.best_params_, unrecorded.best_params_)
            self.assertAlmostEqual(result.best_score_, unrecorded.best_score_, places=13)
            events = read_events(Path(directory) / 'search_history.jsonl')
            trials = [e for e in events if e['event'] == 'trial']
            self.assertEqual(len(trials), 13)
            self.assertEqual([r['trial'] for r in trials], list(range(1, 14)))
            self.assertEqual([r['stage_trial'] for r in trials[-4:]], [1, 2, 3, 4])
            for record in trials:
                self.assertEqual(len(record['params']['gammas']), 2)
                self.assertEqual(len(record['params']['kernel_weights']), 3)
                self.assertAlmostEqual(sum(record['params']['kernel_weights']), 1.)
                self.assertAlmostEqual(record['mean_score'], np.mean(record['split_scores']))
                self.assertEqual(len(record['split_scores']), 3)
            for record in trials[:4]:
                np.testing.assert_allclose(record['params']['kernel_weights'], [1/3] * 3)
            for record in trials[-4:]:
                self.assertEqual(record['timing']['scope'], 'bayesian_batch')
                self.assertGreaterEqual(record['timing']['batch_wall_seconds'], 0.)
            summary = json.loads((Path(directory) / 'search_summary.json').read_text())
            self.assertEqual(summary['status'], 'complete')
            self.assertEqual(summary['n_trials'], 13)
            self.assertAlmostEqual(summary['selected']['score'], result.best_score_)
            self.assertEqual(summary['history_sha256'], hashlib.sha256((Path(directory) / 'search_history.jsonl').read_bytes()).hexdigest())
            self.assertEqual(len(summary['stages']), 4)
            self.assertEqual(summary['settings']['sampler']['multivariate'], True)
            with self.assertRaises(FileExistsError):
                staged_random_search_cv(self.estimator(), self.X, self.y,
                                       search_output_dir=directory, **self.options())

    def test_nondefault_controls_are_applied_and_saved(self):
        with TemporaryDirectory() as directory:
            options = self.options() | dict(gamma_prior_decades=1.2, refine_decades=.6,
                top_k_padding_decades=.2, stage3_weight_concentration=11.,
                bayesian_refine_decades=.7, bayesian_weight_logit_radius=2.3,
                sampler_options={'n_startup_trials': 2, 'n_ei_candidates': 7})
            result = staged_random_search_cv(self.estimator(), self.X, self.y,
                                           search_output_dir=directory, **options)
            summary = json.loads((Path(directory) / 'search_summary.json').read_text())
            self.assertEqual(summary['settings']['gamma_prior_decades'], 1.2)
            self.assertEqual(summary['settings']['refine_decades'], .6)
            self.assertEqual(summary['stages'][-1]['bounds']['weight_logit_radius'], 2.3)
            self.assertEqual(summary['stages'][-1]['bounds']['sampler']['n_ei_candidates'], 7)
            self.assertEqual(result.bayesian_stage.study.sampler._n_ei_candidates, 7)
            stage3 = summary['stages'][2]['bounds']['regressor__kernel_weights']
            self.assertAlmostEqual(sum(stage3['alpha']), 11.)
            for low, high in summary['stages'][0]['bounds']['regressor__gammas']['bounds']:
                self.assertAlmostEqual(np.log10(high / low), 2.4, places=11)

    def test_completed_batches_survive_failure_and_invalid_scores_are_json_safe(self):
        with TemporaryDirectory() as directory:
            calls = 0
            def score(**kwargs):
                nonlocal calls
                calls += 1
                if calls == 2:
                    raise RuntimeError('simulated interrupted batch')
                return np.array([[-.2, -.3, -.4], [np.nan, -.2, -.3]])
            with patch('search_random.score_candidates_from_cache', side_effect=score):
                with self.assertRaisesRegex(RuntimeError, 'interrupted'):
                    staged_random_search_cv(self.estimator(), self.X, self.y,
                                           search_output_dir=directory, **self.options())
            events = read_events(Path(directory) / 'search_history.jsonl')
            trials = [e for e in events if e['event'] == 'trial']
            self.assertEqual(len(trials), 4)
            self.assertEqual(trials[-1]["status"], "failed")
            self.assertEqual(trials[-1]["error"]["type"], "RuntimeError")
            self.assertEqual(trials[1]['failed_fold_indices'], [0])
            self.assertEqual(trials[1]['status'], 'invalid')
            self.assertIsNone(trials[1]['split_scores'][0])
            self.assertIsNone(trials[1]['mean_score'])
            summary = json.loads((Path(directory) / 'search_summary.json').read_text())
            self.assertEqual(summary['status'], 'failed')
            self.assertEqual(summary['n_trials'], 4)
            self.assertEqual(summary['stages'][0]['status'], 'failed')
            self.assertEqual(summary['error']['type'], 'RuntimeError')

    def test_bayesian_exceptions_preserve_attempted_parameters(self):
        for batch_size, target in ((1, 'search_bayes.score_estimator_params_from_cache'),
                                   (2, 'search_bayes._score_bayesian_param_batch')):
            with self.subTest(batch_size=batch_size), TemporaryDirectory() as directory:
                with patch(target, side_effect=RuntimeError('simulated Bayesian interruption')):
                    with self.assertRaisesRegex(RuntimeError, 'Bayesian interruption'):
                        staged_random_search_cv(self.estimator(), self.X, self.y,
                            search_output_dir=directory, **(self.options() | {'bayesian_batch_size': batch_size}))
                events = read_events(Path(directory) / 'search_history.jsonl')
                failures = [e for e in events if e['event'] == 'trial' and e['status'] == 'failed']
                self.assertEqual(len(failures), batch_size)
                self.assertTrue(all(len(e['params']['gammas']) == 2 for e in failures))
                summary = json.loads((Path(directory) / 'search_summary.json').read_text())
                self.assertEqual(summary['status'], 'failed')
                self.assertEqual(summary['stages'][-1]['status'], 'failed')

    def test_uncached_search_is_recorded(self):
        with TemporaryDirectory() as directory:
            options = self.options() | dict(use_distance_cache=False, n_trials_bayesian=2,
                                             bayesian_batch_size=1)
            staged_random_search_cv(self.estimator(), self.X, self.y,
                                   search_output_dir=directory, **options)
            events = read_events(Path(directory) / 'search_history.jsonl')
            trials = [e for e in events if e['event'] == 'trial']
            self.assertEqual(len(trials), 11)
            self.assertEqual(trials[-1]['timing']['scope'], 'trial')

    def test_patience_uses_meaningful_progress_but_retains_exact_best(self):
        for batch_size, expected in ((1, 3), (2, 4)):
            with self.subTest(batch_size=batch_size), TemporaryDirectory() as directory:
                count = 0
                def score(*args, **kwargs):
                    nonlocal count
                    value = -1. + count * .01
                    count += 1
                    return np.array([value, value, value])
                with SearchJournal(directory, settings={}) as journal, patch('search_bayes.cross_val_score', side_effect=score):
                    result = fit_bayesian_search(self.estimator(), self.X, self.y,
                        n_components=2, alpha_bounds=(1e-6, .1), gamma_bounds=[(1e-6, 100.)] * 2,
                        kernel_weight_bounds=(0., 1.), initial_params={'regressor__alpha': .01,
                        'regressor__gammas': [.1, .1], 'regressor__kernel_weights': [.4, .4, .2]},
                        scoring='neg_mean_absolute_error', cv=self.cv, random_state=7, n_jobs=1,
                        patience=2, n_trials=10, prefix='regressor__', refit=False,
                        bayesian_batch_size=batch_size, min_delta_absolute=.1, journal=journal)
                    journal.finish()
                self.assertEqual(len(result.study.trials), expected)
                self.assertAlmostEqual(result.best_score_, -1. + (expected - 1) * .01)
                summary = json.loads((Path(directory) / 'search_summary.json').read_text())
                self.assertEqual(summary['stages'][0]['stop_reason'], 'patience')
        progress = MeaningfulProgress(relative=.05)
        self.assertTrue(progress.observe(0, -10.))
        self.assertFalse(progress.observe(1, -9.8))
        self.assertTrue(progress.observe(2, -9.4))

    def test_config_roundtrip_and_validation(self):
        base = KRRConfig.from_mapping(template())
        updated = base.with_overrides({'search.seed': 123, 'search.gamma_prior_decades': 3.,
            'search.refine_decades': 1.5, 'search.top_k_padding_decades': .7,
            'search.stage3_weight_concentration': 12., 'search.history_batch_size': 4,
            'search.bayesian.refine_decades': .6, 'search.bayesian.weight_logit_radius': 2.,
            'search.bayesian.min_delta_absolute': .001, 'search.bayesian.min_delta_relative': .01,
            'search.bayesian.sampler.multivariate': True, 'search.bayesian.sampler.group': True})
        restored = KRRConfig.from_mapping(updated.values)
        self.assertEqual(restored.spec.search, updated.spec.search)
        self.assertNotEqual(base.sha256(), updated.sha256())
        for change in ({'search.seed': True}, {'search.seed': -1}, {'search.gamma_prior_decades': 0.},
            {'search.refine_decades': float('inf')}, {'search.bayesian.weight_logit_radius': -1.},
            {'search.bayesian.min_delta_relative': -.1}, {'search.history_batch_size': 0},
            {'search.bayesian.sampler.group': True, 'search.bayesian.sampler.multivariate': False}, {'search.bayesian.sampler.typo': True}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                base.with_overrides(change)

    def test_optimizer_seed_does_not_change_data_splits_and_native_artifacts_are_hashed(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            np.save(root / 'x.npy', self.features)
            np.save(root / 'y.npy', self.y)
            results = []
            for seed in (11, 12):
                output = root / str(seed)
                config = KRRConfig.from_mapping({'seed': 7,
                    'descriptors': [{'id': 'x', 'path': str(root / 'x.npy')}],
                    'target': {'path': str(root / 'y.npy')},
                    'split': {'mode': 'random', 'n_train': 18, 'n_folds': 3},
                    'search': {'seed': seed, 'history_batch_size': 2,
                        'random': {'stage1': 3, 'stage2': 2}, 'bayesian': {'trials': 2, 'rounds': 1}},
                    'execution': {'n_jobs': 1, 'distance_cache': {'n_jobs': 1}},
                    'output': {'directory': str(output)}, 'verbosity': 0})
                run(config)
                results.append(output)
                metrics = json.loads((output / 'run_metrics.json').read_text())
                self.assertEqual(metrics['search_config']['seed'], seed)
                for name in ('search_history.jsonl', 'search_summary.json'):
                    self.assertEqual(metrics['artifacts'][name]['sha256'], hashlib.sha256((output / name).read_bytes()).hexdigest())
            for name in ('fold_val_idx.npy', 'test_idx.npy'):
                np.testing.assert_array_equal(np.load(results[0] / name), np.load(results[1] / name))
            with np.load(results[0] / 'validation_folds.npz') as first, np.load(results[1] / 'validation_folds.npz') as second:
                for name in first.files:
                    np.testing.assert_array_equal(first[name], second[name])
            first = read_events(results[0] / 'search_history.jsonl')
            second = read_events(results[1] / 'search_history.jsonl')
            self.assertNotEqual(next(r['params'] for r in first if r['event'] == 'trial'),
                                next(r['params'] for r in second if r['event'] == 'trial'))


if __name__ == '__main__':
    unittest.main()
