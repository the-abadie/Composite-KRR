"""A saved K-fold partition needs no samples permanently reserved for training."""
import unittest
import numpy as np
from sklearn.model_selection import KFold, PredefinedSplit
from preparation import validate_predefined_splits


class PredefinedFullCVTests(unittest.TestCase):
    def test_all_training_samples_can_participate_in_validation(self):
        pool = np.array([9, 2, 7, 0, 5, 3])
        expected = list(KFold(3, shuffle=True, random_state=12).split(pool))
        validation = [pool[val] for _, val in expected]
        test = np.array([1, 4, 6, 8])
        validate_predefined_splits(np.empty(0, dtype=int), validation, test, 10)
        ordered = np.concatenate(validation)
        labels = np.concatenate([np.full(len(val), i) for i, val in enumerate(validation)])
        for i, (train, val) in enumerate(PredefinedSplit(labels).split()):
            np.testing.assert_array_equal(np.sort(ordered[train]), np.sort(pool[expected[i][0]]))
            np.testing.assert_array_equal(np.sort(ordered[val]), np.sort(pool[expected[i][1]]))

    def test_single_fold_cannot_leave_no_training_samples(self):
        with self.assertRaisesRegex(ValueError, "leave training"):
            validate_predefined_splits(np.empty(0, dtype=int), [np.array([0, 1])], np.array([2]), 3)


if __name__ == "__main__":
    unittest.main()
