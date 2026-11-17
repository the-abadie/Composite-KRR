"""Constrained training MLE, endpoint inverses, leakage and cached-score parity."""
import importlib.util
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
from scipy.stats import yeojohnson_llf
from sklearn.base import clone
from sklearn.model_selection import cross_val_score

from cached_scoring import score_estimator_params_from_cache
from preprocess import BoundedYeoJohnsonTransformer, make_target_preprocessor
from run_artifacts import fitted_target_record
from run_config import KRRConfig, run
import test_target_power as power_tests


class TargetBoundedYJTests(unittest.TestCase):
    def setUp(self):
        self.fixture=power_tests.TargetPowerTests();self.fixture.setUp()
        self.name='yeojohnson_bounded_standard'

    def test_schema_clone_mle_bounds_and_standardization(self):
        rng=np.random.default_rng(193)
        skewed=rng.lognormal(1.,1.5,500)
        y=np.column_stack((skewed,-skewed,rng.normal(size=500)))
        config=KRRConfig.from_mapping(dict(descriptors=[dict(id='x',path='x.npy')],
            target=dict(path='y.npy',normalization=self.name),split=dict(mode="random",n_train=30),output=dict(directory='out')))
        self.assertEqual(config.spec.target.normalization,self.name)
        transform=clone(make_target_preprocessor(self.name));z=transform.fit_transform(y)
        self.assertIsInstance(transform,BoundedYeoJohnsonTransformer)
        np.testing.assert_allclose(z.mean(0),0,atol=1e-12)
        np.testing.assert_allclose(z.std(0),1,atol=1e-12)
        np.testing.assert_allclose(transform.inverse_transform(z),y,rtol=1e-11,atol=1e-11)
        np.testing.assert_array_equal(transform.lambdas_[:2],[0.,2.])
        for j,power in enumerate(transform.lambdas_):
            self.assertTrue(0<=power<=2)
            # Independent dense-grid likelihood oracle, not the fitting solver.
            best_grid=max(yeojohnson_llf(v,y[:,j]) for v in np.linspace(0,2,1001))
            self.assertGreaterEqual(yeojohnson_llf(power,y[:,j])+1e-7,best_grid)
        # The legacy recipe remains unconstrained and numerically distinct.
        old=make_target_preprocessor('yeojohnson_standard').fit(y)
        self.assertLess(old.lambdas_[0],0)
        self.assertGreater(old.lambdas_[1],2)

    def test_constant_column_and_arbitrary_latent_inverse(self):
        f=self.fixture
        constant=make_target_preprocessor(self.name).fit(np.full((20,1),7.))
        np.testing.assert_array_equal(constant.lambdas_,[1.])
        for power in (0.,.2,1.,1.8,2.):
            transform=BoundedYeoJohnsonTransformer(standardize=False).fit(f.signed)
            transform.lambdas_[:]=power
            latent=np.linspace(-15,15,30).reshape(15,2)
            inverse=transform.inverse_transform(latent)
            self.assertTrue(np.isfinite(inverse).all())
            np.testing.assert_allclose(transform.transform(inverse),latent,rtol=2e-10,atol=2e-10)

    def test_validation_labels_do_not_fit_power_or_scaling(self):
        f=self.fixture;est=f.estimator(self.name)
        original=f.cache(est,f.signed);altered=f.signed.copy()
        _,validation=next(f.cv.split(f.X));altered[validation]=altered[validation]*1000+500
        other=f.cache(est,altered)
        a,b=original.folds[0].target_transformer,other.folds[0].target_transformer
        np.testing.assert_array_equal(a.lambdas_,b.lambdas_)
        np.testing.assert_array_equal(a._scaler.mean_,b._scaler.mean_)
        np.testing.assert_array_equal(a._scaler.scale_,b._scaler.scale_)

    def test_numpy_cache_matches_independent_sklearn_cv(self):
        f=self.fixture;est=f.estimator(self.name)
        expected=cross_val_score(clone(est),f.X,f.signed,cv=f.cv,scoring='neg_mean_absolute_error',error_score='raise')
        actual=score_estimator_params_from_cache(est,{},f.cache(est,f.signed),scoring='neg_mean_absolute_error',backend='numpy',n_jobs=1)
        np.testing.assert_allclose(actual,expected,rtol=1e-10,atol=1e-10)

    @unittest.skipUnless(importlib.util.find_spec('torch'),'Torch unavailable')
    def test_torch_inverse_and_scores_match_numpy(self):
        import torch
        from pytorch_backend import _inverse_transform_tensor_pytorch
        f=self.fixture;device=os.environ.get('KRR_TEST_TORCH_DEVICE','cpu')
        for y in (f.positive,f.signed,-f.positive):
            transform=make_target_preprocessor(self.name).fit(y)
            values=np.linspace(-10,10,30).reshape(3,5,2)
            actual=_inverse_transform_tensor_pytorch(transform,torch.tensor(values,dtype=torch.float64,device=device)).cpu().numpy()
            expected=np.stack([transform.inverse_transform(v) for v in values])
            np.testing.assert_allclose(actual,expected,rtol=1e-10,atol=1e-10)
            est=f.estimator(self.name);cache=f.cache(est,y)
            expected=score_estimator_params_from_cache(est,{},cache,scoring='neg_mean_absolute_error',backend='numpy',n_jobs=1)
            actual=score_estimator_params_from_cache(est,{},cache,scoring='neg_mean_absolute_error',backend='pytorch',pytorch_device=device,pytorch_dtype='float64',pytorch_candidate_batch_size=2)
            np.testing.assert_allclose(actual,expected,rtol=1e-10,atol=1e-10)

    def test_native_metrics_save_training_only_bounded_state(self):
        f=self.fixture
        with TemporaryDirectory() as directory:
            p=Path(directory);np.save(p/'x.npy',f.features);np.save(p/'y.npy',f.signed[:,0])
            cfg=KRRConfig.from_mapping(dict(descriptors=[dict(id='x',path=str(p/'x.npy'))],
                target=dict(path=str(p/'y.npy'),normalization=self.name),split=dict(mode="random",n_train=30,n_folds=3),
                search=dict(random=dict(stage1=2,stage2=2,stage3=0),bayesian=dict(trials=0),alpha_bounds=[.1,1.],gamma_bounds=[.01,1.]),
                execution=dict(n_jobs=1,distance_cache=dict(n_jobs=1)),output=dict(directory=str(p/'out')),verbosity=0))
            run(cfg)
            metrics=json.loads((p/'out/run_metrics.json').read_text())
            indices=np.load(p/'out/fold_val_idx.npy')
            fitted=make_target_preprocessor(self.name).fit(f.signed[indices,:1])
            saved=metrics['target_transform']
            self.assertEqual(saved['lambda_bounds'],[0.,2.])
            self.assertIn('closed [0, 2]',saved['power_parameter_estimation'])
            for key in ('lambdas','mean','scale'):
                np.testing.assert_array_equal(saved[key],fitted_target_record(fitted)[key])


if __name__=='__main__':unittest.main()
