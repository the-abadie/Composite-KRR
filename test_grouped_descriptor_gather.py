"""Shared shard reads, rank-preserving gathers, and prediction equivalence."""
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np

from class_CompositeDescriptor import CompositeDescriptor
from descriptor_storage import ArchivePartCache
from preprocess import descriptor_blocks_to_sample_matrix
from search_random import CompositeKRREstimator


def bank(root, label, scale):
    arrays = {"rank3": np.arange(18 * 6.0).reshape(18, 2, 3) * scale,
              "vector": np.arange(18 * 4.0).reshape(18, 4) / 7 * scale,
              "scalar": np.arange(18.0).reshape(18, 1) / 11 * scale}
    parts = []
    for start in range(0, 18, 6):
        stop = start + 6
        path = root / f"{label}-{start}.npz"
        np.savez_compressed(path, **{key: values[start:stop] for key, values in arrays.items()})
        parts.append({"path": path.name, "start": start, "stop": stop,
                      "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    records = {key: {"shape": list(values.shape), "dtype": "float64"} for key, values in arrays.items()}
    records["combined"] = {"shape": [18, 11], "dtype": "float64", "source_keys": list(arrays)}
    path = root / f"{label}.json"
    path.write_text(json.dumps({"schema_version": 1, "arrays": records, "parts": parts}))
    arrays["combined"] = np.column_stack([values.reshape(18, -1) for values in arrays.values()])
    return path, arrays


class GroupedDescriptorGatherTests(unittest.TestCase):
    def setup_descriptors(self, root):
        first, a = bank(root, "first", 1)
        second, b = bank(root, "second", 2)
        plain = np.arange(18 * 5.0).reshape(18, 5) / 13
        np.save(root / "plain.npy", plain)
        descriptors = CompositeDescriptor(
            names=["tensor", "plain", "combined", "vector", "other", "tensor_again"],
            paths=[str(first), str(root / "plain.npy"), str(first), str(first), str(second), str(first)],
            normalizations=["standard"] * 6,
            archive_keys=["rank3", None, "combined", "vector", "combined", "rank3"])
        descriptors.load_descriptor_blocks()
        cache = ArchivePartCache()
        for descriptor in descriptors.blocks:
            if hasattr(descriptor.values, "cache"):
                descriptor.values.cache = cache
        reference = [a["rank3"], plain, a["combined"], a["vector"], b["combined"], a["rank3"]]
        return descriptors, reference, cache

    def test_each_touched_archive_opens_once_across_virtual_rank3_and_repeated_views(self):
        with TemporaryDirectory() as temporary:
            descriptors, reference, cache = self.setup_descriptors(Path(temporary))
            rows = np.array([17, 0, 8, 17, 6, 1, 12, 4, 10])
            with patch("descriptor_storage.np.load", wraps=np.load) as opened:
                previous = [descriptor.values[rows] for descriptor in descriptors.blocks]
            self.assertEqual(opened.call_count, 15)  # Previous independent-block traversal.
            with patch("descriptor_storage.np.load", wraps=np.load) as opened:
                result = descriptor_blocks_to_sample_matrix(descriptors, rows)
            self.assertEqual(opened.call_count, 6)  # 3 touched parts × 2 distinct manifests.
            archives = [Path(call.args[0]).name for call in opened.call_args_list]
            self.assertEqual(len(set(archives)), 6)
            for column, original in enumerate(reference):
                np.testing.assert_array_equal(np.stack(result[:, column]), original[rows])
                np.testing.assert_array_equal(np.stack(result[:, column]), previous[column])
            self.assertEqual(result[0, 0].shape, (2, 3))
            self.assertEqual(result[0, 2].shape, (11,))
            self.assertLessEqual(sum(values.nbytes for values in cache.arrays.values()), 6 * 11 * 8)

    def test_gathered_values_produce_identical_standardized_krr_predictions(self):
        with TemporaryDirectory() as temporary:
            descriptors, reference, _ = self.setup_descriptors(Path(temporary))
            rows = np.array([16, 0, 8, 17, 6, 1, 12, 4, 10, 3, 13, 9])
            actual = descriptor_blocks_to_sample_matrix(descriptors, rows)
            expected = np.empty(actual.shape, dtype=object)
            for column, values in enumerate(reference):
                for local, global_row in enumerate(rows):
                    expected[local, column] = values[global_row]
            y = np.sin(reference[0][rows, 0, 0]) + reference[1][rows, 0]

            def estimator():
                return CompositeKRREstimator(names=descriptors.names, kernel_types=["rbf"] * 6,
                          normalizations=["standard"] * 6, compute_dtype="float64", alpha=.2,
                          gammas=[.15] * 6, kernel_weights=[1 / 6] * 6)

            observed = estimator().fit(actual[:8], y[:8]).predict(actual[8:])
            wanted = estimator().fit(expected[:8], y[:8]).predict(expected[8:])
            np.testing.assert_array_equal(observed, wanted)


if __name__ == "__main__":
    unittest.main()
