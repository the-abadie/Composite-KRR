import hashlib
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from class_CompositeDescriptor import CompositeDescriptor
from descriptor_storage import ArchivePartCache, ShardedDescriptorArray, load_manifest
from run_artifacts import array_record
from run_config import KRRConfig, run


def shard_fixture(root):
    a = np.arange(30.0).reshape(10, 3)
    b = np.arange(20.0).reshape(10, 2) / 3
    parts = []
    for start in range(0, 10, 4):
        stop = min(start + 4, 10)
        path = root / f"part-{start}.npz"
        np.savez_compressed(path, component_0_data=a[start:stop], component_1_data=b[start:stop])
        parts.append({"path": path.name, "start": start, "stop": stop,
                      "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                      "generated_record": {"descriptor_hash": "scientific-provenance-preserved"},
                      "storage_record": {"format": "component-only-npz"}})
    manifest = {"schema_version": 1,
                "arrays": {"component_0_data": {"shape": [10, 3], "dtype": "float64"},
                           "component_1_data": {"shape": [10, 2], "dtype": "float64"},
                           "combined": {"shape": [10, 5], "dtype": "float64",
                                        "source_keys": ["component_0_data", "component_1_data"]}},
                "parts": parts}
    path = root / "manifest.json"
    path.write_text(json.dumps(manifest))
    return path, a, b


class DescriptorStorageTests(unittest.TestCase):
    def test_random_repeated_rows_virtual_concat_and_bounded_shared_cache(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            path, a, b = shard_fixture(root)
            combined = ShardedDescriptorArray(path, "combined")
            first = ShardedDescriptorArray(path, "component_0_data")
            cache = ArchivePartCache()
            combined.cache = first.cache = cache
            rows = np.array([9, 0, 4, 4, 8, 1])
            np.testing.assert_array_equal(combined[rows], np.column_stack((a, b))[rows])
            np.testing.assert_array_equal(first[rows], a[rows])
            np.testing.assert_array_equal(combined[::-2], np.column_stack((a, b))[::-2])
            np.testing.assert_array_equal(combined[-1], np.r_[a[-1], b[-1]])
            self.assertEqual(cache.path.name, "part-8.npz")
            self.assertLessEqual(sum(array.nbytes for array in cache.arrays.values()), 4 * 5 * 8)
            with self.assertRaisesRegex(TypeError, "Select descriptor rows"):
                np.asarray(combined)
            record = array_record(path, archive_key="combined")
            self.assertEqual(record["shape"], [10, 5])
            self.assertEqual(record["manifest"]["parts"][0]["generated_record"]["descriptor_hash"], "scientific-provenance-preserved")

    def test_corruption_and_incomplete_row_coverage_fail(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            path, a, _ = shard_fixture(root)
            descriptor = ShardedDescriptorArray(path, "component_0_data")
            part = root / "part-0.npz"
            part.write_bytes(part.read_bytes() + b"corruption")
            with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                descriptor[[0]]
            manifest = json.loads(path.read_text())
            manifest["parts"][1]["start"] += 1
            path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "contiguously"):
                load_manifest(path)

    def test_engine_runs_on_shards_and_records_complete_manifest_provenance(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            path, a, b = shard_fixture(root)
            np.save(root / "target.npy", a[:, 0] * 0.5 + b[:, 0])
            config = KRRConfig.from_mapping({
                "seed": 1,
                "descriptors": [{"id": "a", "path": str(path), "archive_key": "component_0_data"},
                                {"id": "b", "path": str(path), "archive_key": "component_1_data"}],
                "target": {"path": str(root / "target.npy")},
                "split": {"mode": "random", "n_train": 6, "n_folds": 2},
                "search": {"random": {"stage1": 1, "stage2": 1}, "bayesian": {"trials": 0}},
                "execution": {"n_jobs": 1, "prediction_batch_size": 2, "distance_cache": {"n_jobs": 1}},
                "output": {"directory": str(root / "output")}, "verbosity": 0})
            run(config)
            metrics = json.loads((root / "output/run_metrics.json").read_text())
            self.assertEqual(metrics["features"], 5)
            self.assertEqual(metrics["n_test"], 4)
            self.assertEqual(metrics["descriptor_blocks"][0]["archive_key"], "component_0_data")
            self.assertEqual(metrics["descriptor_blocks"][0]["manifest"]["parts"], json.loads(path.read_text())["parts"])


if __name__ == "__main__":
    unittest.main()
