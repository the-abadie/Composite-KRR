"""Earlier same-run scores inform TPE without becoming new CV evaluations."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
import optuna
import test_search_evidence as fixtures
from search_bayes import _import_same_run_observations
from search_random import staged_random_search_cv
from run_config import KRRConfig, template


class SameRunObservationTests(unittest.TestCase):
    def test_contribution_round_budget_is_validated_before_execution(self):
        cfg=KRRConfig.from_mapping(template()).with_overrides({'search.bayesian.rounds':2,
            'reporting.kernel_contributions':True,'reporting.contribution_bayesian_trials':2})
        with self.assertRaises(ValueError):
            cfg.with_overrides({'reporting.contribution_bayesian_trials':1})
        self.assertEqual(cfg.with_overrides({'reporting.contribution_bayesian_trials':0}).spec.search.bayesian.rounds,2)

    def test_only_valid_representable_in_box_points_are_imported(self):
        study=optuna.create_study(direction='maximize')
        weights=np.exp([1.,1.,-1.]);weights/=weights.sum()
        base=dict(params={'alpha':.01,'gammas':[.02],'kernel_weights':weights},
                  split_scores=[-.1,-.2,-.3],source_stage='Stage 2',source_trial=1)
        bad_score=base|{'split_scores':[float('nan'),-.2,-.3]}
        outside=base|{'params':base['params']|{'alpha':10.}}
        zero=base|{'params':base['params']|{'kernel_weights':[1.,0.,0.]}}
        report=_import_same_run_observations(study,[base,bad_score,outside,zero],
            n_components=1,n_terms=3,alpha_bounds=(1e-6,1.),gamma_bounds=[(1e-6,1.)],
            kernel_weight_center=[1/3]*3,kernel_weight_logit_radius=1.)
        self.assertEqual(report['accepted_count'],1)
        self.assertEqual(report['excluded_counts'],dict(invalid_scores=1,invalid_params=0,outside_bounds=1,nonrepresentable_weights=1))
        self.assertAlmostEqual(study.best_value,-.2)
        trial=study.best_trial
        self.assertTrue(trial.user_attrs['reused_observation'])
        logits=np.array([trial.params[f'kernel_weight_logit_{i}'] for i in range(3)])
        self.assertTrue(np.all(logits>=-1.) and np.all(logits<=1.))
        actual=np.exp(logits-logits.max());actual/=actual.sum()
        np.testing.assert_allclose(actual,weights,rtol=1e-13,atol=1e-13)
        np.testing.assert_array_equal(trial.user_attrs['observed_params']['kernel_weights'],weights)

    def test_new_trial_budgets_and_histories_exclude_imported_observations(self):
        fixture=fixtures.SearchEvidenceTests();fixture.setUp()
        for batch in (1,2):
            with self.subTest(batch=batch),TemporaryDirectory() as directory:
                result=staged_random_search_cv(fixture.estimator(),fixture.X,fixture.y,
                    search_output_dir=directory,**(fixture.options()|{
                        'bayesian_reuse_observations':True,'bayesian_batch_size':batch,
                        'bayesian_refine_decades':11.,'bayesian_weight_logit_radius':25.}))
                events=fixtures.read_events(Path(directory)/'search_history.jsonl')
                imports=[r for r in events if r['event']=='observations_imported']
                self.assertEqual(imports[0]['accepted_count'],9)
                trials=[r for r in events if r['event']=='trial']
                self.assertEqual(len(trials),13)
                self.assertEqual(len(result.bayesian_stage.study.trials),13)
                self.assertEqual(len(result.bayesian_search_history_),4)
                self.assertEqual([r['stage_trial'] for r in trials[-4:]],[1,2,3,4])
                self.assertEqual(result.best_score_,max(r['mean_score'] for r in trials))
                summary=json.loads((Path(directory)/'search_summary.json').read_text())
                self.assertEqual(summary['n_trials'],13)
                self.assertEqual(summary['stages'][-1]['recorded_trials'],4)
                self.assertEqual(summary['stages'][-1]['bounds']['sampler_seed'],99)
                self.assertEqual(summary['stages'][-1]['stop_reason'],'trial_limit')

    def test_patience_counts_only_new_evaluations_and_keeps_the_observed_best(self):
        fixture=fixtures.SearchEvidenceTests();fixture.setUp()
        with TemporaryDirectory() as directory,patch('search_bayes.score_estimator_params_from_cache',
                return_value=np.array([-1e6]*3)) as score:
            result=staged_random_search_cv(fixture.estimator(),fixture.X,fixture.y,
                search_output_dir=directory,**(fixture.options()|{'bayesian_reuse_observations':True,
                    'n_trials_bayesian':20,'bayesian_batch_size':1,'bayesian_patience':2,
                    'bayesian_refine_decades':11.,'bayesian_weight_logit_radius':25.}))
            self.assertEqual(score.call_count,3)
            self.assertEqual(len(result.bayesian_search_history_),3)
            self.assertTrue(result.bayesian_stage.study.best_trial.user_attrs['reused_observation'])
            self.assertEqual(result.best_score_,result.bayesian_stage.best_score_)
            summary=json.loads((Path(directory)/'search_summary.json').read_text())
            self.assertEqual(summary['n_trials'],12)
            self.assertEqual(summary['stages'][-1]['stop_reason'],'patience')

    def test_rounds_reuse_only_new_prior_round_evaluations(self):
        fixture=fixtures.SearchEvidenceTests();fixture.setUp()
        with TemporaryDirectory() as directory,patch('search_bayes.score_estimator_params_from_cache',
                return_value=np.array([-1e6]*3)):
            result=staged_random_search_cv(fixture.estimator(),fixture.X,fixture.y,
                search_output_dir=directory,**(fixture.options()|{'bayesian_reuse_observations':True,
                    'bayesian_rounds':2,'bayesian_batch_size':1,
                    'bayesian_refine_decades':11.,'bayesian_weight_logit_radius':25.}))
            events=fixtures.read_events(Path(directory)/'search_history.jsonl')
            imports=[r for r in events if r['event']=='observations_imported']
            self.assertEqual([r['accepted_count'] for r in imports],[9,11])
            self.assertEqual(len(result.bayesian_search_history_),4)
            self.assertEqual(len([r for r in events if r['event']=='trial']),13)
            summary=json.loads((Path(directory)/'search_summary.json').read_text())
            seeds=[r['bounds']['sampler_seed'] for r in summary['stages'][-2:]]
            self.assertEqual(seeds,[99,int(np.random.SeedSequence([99,1]).generate_state(1)[0])])


if __name__=='__main__':unittest.main()
