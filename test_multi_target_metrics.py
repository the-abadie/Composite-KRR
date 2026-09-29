import unittest

import numpy as np
from sklearn.metrics import root_mean_squared_error

from postprocess import regression_error_summary


class MultiTargetMetricTests(unittest.TestCase):
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
