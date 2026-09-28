import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

import numpy as np
from sklearn.model_selection import KFold

from run_config import KRRConfig
from run_artifacts import save_run_metrics


class RunArtifactsTests(unittest.TestCase):
    def test_completed_evidence_preserves_global_folds_and_refuses_replacement(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            np.save(root / "x.npy", np.ones((8, 3)))
            np.save(root / "y.npy", np.arange(8.0))
            train, test = np.array([6, 1, 4, 0, 7, 2]), np.array([3, 5])
            for name, value in {"fold_val_idx": train, "test_idx": test,
                                "y_predictions": [3., 5.], "y_true": [3., 5.],
                                "alpha": 0.1, "gammas": [0.2], "kernel_weights": [1.]}.items():
                np.save(root / f"{name}.npy", value)
            spec = KRRConfig.from_mapping({"descriptors": [{"id": "x", "path": str(root / "x.npy")}],
                    "target": {"path": str(root / "y.npy")}, "split": {"mode": "random", "n_train": 6},
                    "output": {"directory": str(root)}}).spec
            search = SimpleNamespace(best_split_scores_=np.array([-1., -2., -3.]),
                                     best_score_=-2., best_params_={"alpha": np.float64(0.1)})
            kwargs = dict(elapsed_seconds=2., training_seconds=1.)
            save_run_metrics(spec, search, KFold(3), train, test, np.arange(6.),
                             {"mae": 0., "rmse": 0.}, **kwargs)
            metrics = json.loads((root / "run_metrics.json").read_text())
            self.assertEqual(metrics["cv_mae"], 2.)
            self.assertEqual(metrics["features"], 3)
            self.assertEqual(len(metrics["descriptor_blocks"][0]["sha256"]), 64)
            with np.load(root / "validation_folds.npz") as folds:
                np.testing.assert_array_equal(folds["validation_0"], [6, 1])
                self.assertFalse(np.intersect1d(folds["validation_0"], test).size)
            with self.assertRaises(FileExistsError):
                save_run_metrics(spec, search, KFold(3), train, test, np.arange(6.),
                                 {"mae": 1., "rmse": 1.}, **kwargs)


if __name__ == "__main__":
    unittest.main()
