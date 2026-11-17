import pickle
from pathlib import Path
import subprocess
import sys
import textwrap
import unittest

import numpy as np
from sklearn.compose import TransformedTargetRegressor

from preprocess import make_target_preprocessor
from run_config import KRRConfig
from saved_model import SavedExactModel
from composite_krr.estimators import CompositeKRREstimator, CompositeNystromKRREstimator


class SavedModelTests(unittest.TestCase):
    def test_fixed_parameter_numpy_models_do_not_load_search_plotting_or_torch(self):
        script = textwrap.dedent("""
            import sys
            import numpy as np
            from composite_krr.estimators import CompositeKRREstimator, CompositeNystromKRREstimator

            x = np.empty((6, 1), dtype=object)
            x[:, 0] = list(np.arange(6.0).reshape(-1, 1))
            y = np.arange(6.0)
            for cls in (CompositeKRREstimator, CompositeNystromKRREstimator):
                prediction = cls(compute_dtype="float64").fit(x, y).predict(x)
                assert np.isfinite(prediction).all()
            assert {"optuna", "matplotlib.pyplot", "torch"}.isdisjoint(sys.modules)
        """)
        subprocess.run([sys.executable, "-c", script], cwd=Path(__file__).parent, check=True)

    def test_historical_estimator_names_in_pickles(self):
        for module, cls in [("search_random", CompositeKRREstimator),
                            ("nystrom_krr", CompositeNystromKRREstimator)]:
            with self.subTest(module=module):
                # Protocol 0 GLOBAL, as written when these modules owned the classes.
                reference = f"c{module}\n{cls.__name__}\n.".encode("ascii")
                self.assertIs(pickle.loads(reference), cls)

    def test_coefficients_reproduce_model_with_products_and_training_transforms(self):
        rng = np.random.default_rng(42)
        blocks = [rng.normal(size=(12, 4)), rng.normal(size=(12, 3))]
        y = rng.normal(size=12) * 100 - 900
        spec = KRRConfig.from_mapping({
            "descriptors": [{"id": "a", "path": "a.npy"}, {"id": "b", "path": "b.npy"}],
            "target": {"path": "y.npy"}, "split": {"mode": "random", "n_train": 10},
            "kernel_products": [["a", "b"]], "output": {"directory": "out"},
        }).spec
        x = np.empty((12, 2), dtype=object)
        for j, b in enumerate(blocks):
            for i, row in enumerate(b):
                x[i, j] = row
        params = dict(alpha=0.03, gammas=[0.1, 0.3], kernel_weights=[2, 3, 4])
        reg = CompositeKRREstimator(names=["a", "b"], normalizations=["standard"] * 2,
            kernel_products=[[0, 1]], normalize_kernel_weights=True, compute_dtype="float64", **params)
        model = TransformedTargetRegressor(regressor=reg, transformer=make_target_preprocessor("standard"))
        model.fit(x[:10], y[:10])
        coef = model.regressor_.model_.dual_coef_
        for backend in ["numpy", "pytorch"]:
            try:
                restored = SavedExactModel(spec, [b[:10] for b in blocks], y[:10], coef,
                    params["gammas"], params["kernel_weights"], params["alpha"], backend=backend)
            except ImportError:
                if backend == "pytorch":
                    continue
                raise
            np.testing.assert_allclose(restored.predict([b[10:] for b in blocks]), model.predict(x[10:]), atol=1e-9)
            with self.assertRaises(ValueError):
                restored.predict([blocks[0][10:, :2], blocks[1][10:]])


if __name__ == "__main__":
    unittest.main()
