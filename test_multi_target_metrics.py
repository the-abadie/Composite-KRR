import unittest

import numpy as np
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

from kernel_cache import score_predictions
from postprocess import regression_error_summary


class MultiTargetMetricTests(unittest.TestCase):
    def test_cached_mae_matches_sklearn_with_unequal_target_scales(self):
        rng = np.random.default_rng(5)
        true = rng.normal(size=(71, 3)) * [1e-8, 1, 1e8]
        pred = rng.normal(size=true.shape)
        for labels, predictions in ((true, pred), (true[:, 0], pred[:, :1])):
            self.assertEqual(score_predictions(labels, predictions, "neg_mean_absolute_error"),
                             -mean_absolute_error(labels, predictions))
        with self.assertRaises(ValueError):
            score_predictions([np.nan], [1.0], "neg_mean_absolute_error")

    def test_rmse_matches_search_aggregation_with_unequal_target_errors(self):
        true = np.zeros((2, 2))
        predicted = np.array([[1., 9.], [1., 9.]])
        summary = regression_error_summary(true, predicted)
        self.assertEqual(summary["rmse"], 5.)
        self.assertEqual(summary["rmse"], root_mean_squared_error(true, predicted))
        np.testing.assert_array_equal(summary["target_rmse"], [1., 9.])

    def test_single_target_rmse_is_unchanged_for_vector_and_matrix(self):
        for shape in ((2,), (2, 1)):
            true = np.zeros(shape)
            predicted = np.array([3., 4.]).reshape(shape)
            self.assertEqual(regression_error_summary(true, predicted)["rmse"],
                             root_mean_squared_error(true, predicted))


if __name__ == "__main__":
    unittest.main()
