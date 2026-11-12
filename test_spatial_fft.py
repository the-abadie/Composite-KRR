"""Fourier packing, fold isolation, kernel equivalence and saved inference."""
import json
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from config_schema import DescriptorConfig
from kernel_cache import build_distance_cache
from preprocess import SpatialFFT, descriptor_preprocessing, make_data_preprocessor
from run_config import KRRConfig
from saved_model import SavedExactModel
from search_random import CompositeKRREstimator


def reference(x,n,k):
    channels=x.reshape(len(x),-1,n);columns=[]
    for channel in range(channels.shape[1]):
        c=np.fft.rfft(channels[:,channel],norm='ortho');columns.append(c[:,0].real)
        for frequency in range(1,k):
            if n%2==0 and frequency==n//2:columns.append(c[:,frequency].real)
            else:columns.extend((np.sqrt(2)*c[:,frequency].real,np.sqrt(2)*c[:,frequency].imag))
    return np.stack(columns,axis=1)


def samples(block):
    x=np.empty((len(block),1),dtype=object)
    for i,row in enumerate(block):x[i,0]=row
    return x


class SpatialFFTTests(unittest.TestCase):
    def setUp(self):
        self.rng=np.random.default_rng(42)
        self.x=self.rng.normal(size=(24,12))*np.geomspace(.01,100,12);self.x[:,2]=1.
        self.recipe=dict(normalization='standard',spatial_fft=dict(block_size=6,n_frequencies=2))

    def test_full_odd_even_and_single_bin_packing_preserve_cross_kernel_distances(self):
        for n in (1,2,5,6,121):
            x=self.rng.normal(size=(20,2*n));train=x[:13];held=x[13:]+1000
            recipe=dict(normalization='standard',spatial_fft=dict(block_size=n,n_frequencies=n//2+1))
            pipeline=make_data_preprocessor(recipe);a=pipeline.fit_transform(train);b=pipeline.transform(held)
            scaler=StandardScaler().fit(train)
            np.testing.assert_allclose(a,reference(scaler.transform(train),n,n//2+1),atol=3e-14)
            self.assertEqual(a.shape,train.shape)
            d1=cdist(b,a,'sqeuclidean');d2=cdist(scaler.transform(held),scaler.transform(train),'sqeuclidean')
            np.testing.assert_allclose(d1,d2,rtol=3e-14,atol=2e-8)
            np.testing.assert_allclose(np.exp(-1e-8*d1),np.exp(-1e-8*d2),atol=3e-14)

    def test_cached_folds_use_only_training_bin_and_optional_coefficient_statistics(self):
        for post in ('none','standard'):
            recipe={**self.recipe,'spatial_fft':{**self.recipe['spatial_fft'],'post_normalization':post}}
            cv=KFold(3,shuffle=True,random_state=3)
            cache=build_distance_cache(samples(self.x),self.rng.normal(size=24),cv,names=['q2'],kernel_types=['rbf'],normalizations=[recipe],n_jobs=1)
            for fold,(train,val) in zip(cache.folds,cv.split(self.x),strict=True):
                scaler=StandardScaler().fit(self.x[train]);a,b=reference(scaler.transform(self.x[train]),6,2),reference(scaler.transform(self.x[val]),6,2)
                if post=='standard':
                    second=StandardScaler().fit(a);a,b=second.transform(a),second.transform(b)
                np.testing.assert_allclose(fold.train_distances[0],cdist(a,a,'sqeuclidean'),atol=2e-12)
                np.testing.assert_allclose(fold.validation_train_distances[0],cdist(b,a,'sqeuclidean'),atol=2e-12)

    def test_saved_inference_reconstructs_training_fitted_fourier_pipeline(self):
        y=self.rng.normal(size=16)
        estimator=CompositeKRREstimator(names=['q2'],kernel_types=['rbf'],normalizations=[self.recipe],compute_dtype='float64',gammas=[.2],kernel_weights=[1.],alpha=.1).fit(samples(self.x[:16]),y)
        spec=KRRConfig.from_mapping(dict(descriptors=[dict(id='q2',path='q2.npy',**self.recipe)],target=dict(path='y.npy',normalization='none'),split=dict(mode='random',n_train=16),output=dict(directory='unused'))).spec
        saved=SavedExactModel(spec,[self.x[:16]],y,estimator.model_.dual_coef_,[.2],[1.],.1)
        np.testing.assert_allclose(saved.predict([self.x[16:]+10]),estimator.predict(samples(self.x[16:]+10)),atol=2e-13)
        self.assertEqual(estimator.transformed_feature_counts_,[6]);self.assertEqual(descriptor_preprocessing(spec.descriptors[0]),self.recipe)

    @unittest.skipUnless(importlib.util.find_spec('torch') is not None, 'PyTorch is an optional dependency')
    def test_torch_distance_cache_uses_the_same_fold_fitted_fourier_metric(self):
        cv=KFold(3,shuffle=True,random_state=3)
        cache=build_distance_cache(samples(self.x),self.rng.normal(size=24),cv,names=['q2'],kernel_types=['rbf'],normalizations=[self.recipe],n_jobs=1,distance_backend='pytorch',pytorch_device='cpu')
        for fold,(train,val) in zip(cache.folds,cv.split(self.x),strict=True):
            scaler=StandardScaler().fit(self.x[train]);a,b=reference(scaler.transform(self.x[train]),6,2),reference(scaler.transform(self.x[val]),6,2)
            np.testing.assert_allclose(fold.train_distances[0],cdist(a,a,'sqeuclidean'),atol=3e-12)
            np.testing.assert_allclose(fold.validation_train_distances[0],cdist(b,a,'sqeuclidean'),atol=3e-12)

    def test_invalid_schema_widths_and_legacy_loss_are_rejected(self):
        for settings in (dict(block_size=6,n_frequencies=5),dict(block_size=True,n_frequencies=1)):
            with self.assertRaises(ValueError):DescriptorConfig(id='q2',path='q2.npy',spatial_fft=settings)
        with self.assertRaises(ValueError):DescriptorConfig(id='q2',path='q2.npy',spatial_fft=dict(block_size=6,n_frequencies=2),spatial_dct=dict(block_size=6,n_components=2))
        with self.assertRaises(ValueError):SpatialFFT(block_size=5,n_frequencies=2).fit(self.x)
        config=KRRConfig.from_mapping(dict(descriptors=[dict(id='q2',path='q2.npy',**self.recipe)],target=dict(path='y.npy'),split=dict(mode='random',n_train=16),output=dict(directory='unused')))
        with self.assertRaisesRegex(ValueError,'structured'):_=config.values
        x=self.x.astype(np.float32);result=SpatialFFT(block_size=6,n_frequencies=4).fit_transform(x)
        self.assertEqual(result.dtype,np.float32);np.testing.assert_allclose(result,reference(x,6,4),rtol=2e-6,atol=2e-5)

    def test_cli_records_effective_width_and_training_only_target_alignment(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);np.save(root/'q2.npy',self.x);np.save(root/'y.npy',self.rng.normal(size=24))
            config=dict(descriptors=[dict(id='q2',path='q2.npy',**self.recipe)],target=dict(path='y.npy'),split=dict(mode='random',n_train=18,n_folds=3),execution=dict(n_jobs=1,distance_cache=dict(n_jobs=1)),search=dict(random=dict(stage1=3,stage2=2,stage3=1),bayesian=dict(trials=0)),reporting=dict(kernel_alignment=dict(enabled=True,include_target=True,max_samples=18)),output=dict(directory='output'))
            (root/'run.json').write_text(json.dumps(config))
            result=subprocess.run([sys.executable,str(Path(__file__).with_name('main.py')),'--config',str(root/'run.json')],capture_output=True,text=True,env={**os.environ,'MPLBACKEND':'Agg','OPENBLAS_NUM_THREADS':'1'})
            self.assertEqual(result.returncode,0,result.stdout+result.stderr)
            metrics=json.loads((root/'output/run_metrics.json').read_text());alignment=json.loads((root/'output/kernel_alignment.json').read_text())
            self.assertEqual((metrics['features'],metrics['kernel_features']),(12,6));self.assertTrue(alignment['include_target'])
            self.assertFalse(np.intersect1d(alignment['sample_indices'],np.load(root/'output/test_idx.npy')).size)

if __name__=='__main__':unittest.main()
