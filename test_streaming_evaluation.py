import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from class_CompositeDescriptor import CompositeDescriptor
from class_Target import Target
from postprocess import regression_error_summary
from streaming_evaluation import RegressionErrorAccumulator, diagnostic_plot_sample, predict_held_out


class StreamingEvaluationTests(unittest.TestCase):
    def test_full_evaluation_is_batched_and_preserves_index_order_and_precision(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            values = np.arange(111.0).reshape(37, 3) / 7
            targets = np.column_stack((values[:, 0] + 0.123456789, values[:, 0] * 100))
            np.save(root / "descriptor.npy", values)
            np.save(root / "target.npy", targets)
            descriptors = CompositeDescriptor(["x"], [str(root / "descriptor.npy")], ["standard"])
            descriptors.load_descriptor_blocks_from_npy()
            self.assertIsInstance(descriptors.blocks[0].values, np.memmap)
            target = Target(["a", "b"], root / "target.npy", "standard").load_target_from_npy()
            self.assertTrue(np.shares_memory(target, target.base))

            class Predictor:
                def __init__(self):
                    self.batch_counts = []

                def predict(self, X):
                    self.batch_counts.append(len(X))
                    first = np.array([sample[0] for sample in X[:, 0]])
                    return np.column_stack((first, first * 100 + 3.5))

            predictor = Predictor()
            rows = np.array([35, 2, 30, 1, 9, 16, 0, 11, 32, 36, 3])
            pred, true, summary = predict_held_out(predictor, descriptors, target, rows, root / "output", batch_size=4)
            self.assertEqual(predictor.batch_counts, [4, 4, 3])
            self.assertIsInstance(pred, np.memmap)
            np.testing.assert_array_equal(true, targets[rows])
            np.testing.assert_array_equal(np.load(root / "output/y_true.npy"), targets[rows])
            expected = regression_error_summary(targets[rows], pred)
            for key in expected:
                np.testing.assert_allclose(summary[key], expected[key], rtol=1e-14, atol=1e-14)
            sampled_true, sampled_pred = diagnostic_plot_sample(true, pred, max_samples=3)
            np.testing.assert_array_equal(sampled_true, true[[0, 5, 10]])
            np.testing.assert_array_equal(sampled_pred, pred[[0, 5, 10]])

    def test_absolute_error_variance_is_stable_with_large_offset(self):
        predicted = np.array([1e8 - 1, 1e8, 1e8 + 1, 1e8 + 2])
        accumulator = RegressionErrorAccumulator()
        for chunk in np.array_split(predicted, 2):
            accumulator.update(np.zeros_like(chunk), chunk)
        self.assertAlmostEqual(accumulator.summary()["mae_std"], np.std(predicted), places=10)

    def test_rejects_nonfinite_or_incompatible_batches(self):
        accumulator = RegressionErrorAccumulator()
        with self.assertRaisesRegex(ValueError, "No held-out"):
            accumulator.summary()
        with self.assertRaisesRegex(ValueError, "finite"):
            accumulator.update([1.0], [np.nan])
        with self.assertRaisesRegex(ValueError, "shape"):
            accumulator.update([1.0], [[1.0, 2.0]])


if __name__ == "__main__":
    unittest.main()
