"""Independent-run recentering keeps caps, earlier incumbents and trial evidence."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np

from config_schema import BayesianConfig
from search_bayes import BayesianSearchResult
from search_random import _fit_bayesian_rounds, staged_random_search_cv
import test_search_evidence as fixture_module


class RecenteredSearchTests(unittest.TestCase):
    def test_budget_seeds_and_incumbent_survives_a_worse_round(self):
        start={'alpha':1.,'gammas':[1.], 'kernel_weights':[1.]}
        options=dict(initial_params=start,initial_score=-10.,n_components=1,
            alpha_bounds=(1e-9,100.),gamma_bounds=[(1e-9,100.)],rounds=3,
            refine_decades=1.,n_trials=10,prefix='',random_state=12,
            timeout=None,stage_name='Bayesian search')
        returned=[]
        def fake(*args,**kwargs):
            score=(-8.,-9.,-7.)[len(returned)]
            alpha=(2.,3.,4.)[len(returned)]
            params=start|{'alpha':alpha}
            result=BayesianSearchResult(object(),None,params,score,np.array([score]*3),
                [dict(iteration=i+1,mean_test_score=score) for i in range(kwargs['n_trials'])])
            returned.append((kwargs,result));return result
        with patch('search_random.fit_bayesian_search',side_effect=fake):
            result=_fit_bayesian_rounds(None,None,None,**options)
        self.assertEqual([c['n_trials'] for c,r in returned],[4,3,3])
        self.assertEqual([c['initial_params']['alpha'] for c,r in returned],[1.,2.,2.])
        self.assertEqual([c['alpha_bounds'] for c,r in returned],[(.1,10.),(.2,20.),(.2,20.)])
        self.assertEqual(returned[0][0]['random_state'],12)
        self.assertEqual(len({c['random_state'] for c,r in returned}),3)
        self.assertEqual(result.best_score_,-7.)
        self.assertEqual([r['iteration'] for r in result.search_history_],list(range(1,11)))
        self.assertEqual(len(result.round_results_),3)

    def test_timeout_is_shared_across_rounds(self):
        params={'alpha':1.,'gammas':[1.], 'kernel_weights':[1.]}
        result=BayesianSearchResult(None,None,params,-1.,np.array([-1.]),[])
        with patch('search_random.perf_counter',side_effect=[0.,.2,.4,1.2]), \
             patch('search_random.fit_bayesian_search',return_value=result) as fit:
            _fit_bayesian_rounds(None,None,None,initial_params=params,initial_score=-1.,
                n_components=1,alpha_bounds=(1e-9,100.),gamma_bounds=[(1e-9,100.)],
                rounds=3,refine_decades=1.,n_trials=9,prefix='',random_state=1,
                timeout=1.,stage_name='Bayesian search')
        self.assertEqual(fit.call_count,2)
        self.assertAlmostEqual(fit.call_args_list[0].kwargs['timeout'],.8)
        self.assertAlmostEqual(fit.call_args_list[1].kwargs['timeout'],.6)

    def test_cpu_round_evidence_and_repeatability(self):
        fixture=fixture_module.SearchEvidenceTests();fixture.setUp()
        with TemporaryDirectory() as one, TemporaryDirectory() as two:
            options=fixture.options()|{'n_trials_bayesian':10,'bayesian_rounds':3,
                                      'bayesian_batch_size':1}
            first=staged_random_search_cv(fixture.estimator(),fixture.X,fixture.y,
                                         search_output_dir=one,**options)
            second=staged_random_search_cv(fixture.estimator(),fixture.X,fixture.y,
                                          search_output_dir=two,**options)
            self.assertEqual(first.best_params_,second.best_params_)
            self.assertEqual(first.best_score_,second.best_score_)
            events=fixture_module.read_events(Path(one)/'search_history.jsonl')
            trials=[r for r in events if r['event']=='trial']
            summary=json.loads((Path(one)/'search_summary.json').read_text())
            self.assertEqual(summary['settings']['bayesian_rounds'],3)
            self.assertEqual([s['recorded_trials'] for s in summary['stages'][-3:]],[4,3,3])
            self.assertEqual(len(trials),19)
            self.assertEqual(len(first.bayesian_search_history_),10)
            self.assertEqual(first.best_score_,max(r['mean_score'] for r in trials))
            self.assertEqual([r['trial'] for r in trials],list(range(1,20)))
            for stage in summary['stages'][-3:]:
                self.assertEqual(stage['stop_reason'],'trial_limit')
                self.assertTrue(all(1e-6<=g[0]<g[1]<=100 for g in stage['bounds']['gammas']))

    def test_round_budget_validation(self):
        for invalid in (dict(rounds=0),dict(trials=2,rounds=3),dict(rounds=True)):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                BayesianConfig.model_validate(invalid)
        self.assertEqual(BayesianConfig(rounds=2,trials=0).rounds,2)


if __name__=='__main__':unittest.main()
