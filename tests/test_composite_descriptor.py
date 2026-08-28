from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from class_CompositeDescriptor import CompositeDescriptor
from preprocess import descriptor_blocks_to_sample_matrix
from search_random import CompositeKRREstimator


def _write_archive(
    path: Path,
    *,
    first_samples: int = 4,
    second_samples: int = 4,
    description_suffix: str = "description",
) -> None:
    np.savez(
        path,
        component_count=np.asarray(2, dtype=np.int64),
        component_0_name=np.asarray("radial"),
        **{
            f"component_0_{description_suffix}": np.asarray(
                "Radial descriptor."
            )
        },
        component_0_data=np.arange(first_samples * 3, dtype=float).reshape(
            first_samples, 3
        ),
        component_1_name=np.asarray("angular"),
        **{
            f"component_1_{description_suffix}": np.asarray(
                "Angular descriptor."
            )
        },
        component_1_data=np.arange(second_samples * 4, dtype=float).reshape(
            second_samples, 2, 2
        ),
    )


class CompositeDescriptorNpzTests(unittest.TestCase):
    def test_npz_names_override_config_and_one_normalization_is_broadcast(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "descriptors.npz"
            _write_archive(path)

            descriptors = CompositeDescriptor(
                names=["configured-name"],
                paths=[str(path)],
                normalizations=["standard"],
            )
            with self.assertLogs("class_CompositeDescriptor", level="INFO") as logs:
                blocks = descriptors.load_descriptor_blocks()

            self.assertEqual([block.name for block in blocks], ["radial", "angular"])
            self.assertEqual(
                [block.normalization for block in blocks],
                ["standard", "standard"],
            )
            self.assertEqual(
                [block.description for block in blocks],
                ["Radial descriptor.", "Angular descriptor."],
            )
            self.assertEqual(blocks[1].values.shape, (4, 2, 2))
            self.assertEqual(descriptors.names, ["radial", "angular"])
            self.assertEqual(descriptors.normalizations, ["standard", "standard"])
            self.assertTrue(
                any("single configured normalization" in message for message in logs.output)
            )
            self.assertTrue(
                any("overriding configured names" in message for message in logs.output)
            )

    def test_npz_per_descriptor_normalizations_preserve_archive_order(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "descriptors.npz"
            _write_archive(path, description_suffix="desc")

            descriptors = CompositeDescriptor(
                names=[],
                paths=[str(path)],
                normalizations=["none", "log_standard"],
            )
            blocks = descriptors.load_descriptor_blocks()

            self.assertEqual(
                [block.normalization for block in blocks],
                ["none", "log_standard"],
            )

    def test_each_npz_descriptor_becomes_one_kernel_component(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "descriptors.npz"
            _write_archive(path)

            descriptors = CompositeDescriptor(
                names=["ignored"],
                paths=[str(path)],
                normalizations=["none"],
            )
            descriptors.load_descriptor_blocks()
            X = descriptor_blocks_to_sample_matrix(descriptors)

            estimator = CompositeKRREstimator(
                alpha=1.0,
                gammas=[0.1, 0.2],
                kernel_weights=[0.5, 0.5],
                names=descriptors.names,
                kernel_types=["rbf", "rbf"],
                normalizations=descriptors.normalizations,
                compute_dtype="float64",
            )
            estimator.fit(X, np.arange(4, dtype=float))

            self.assertEqual(estimator.names_, ["radial", "angular"])
            self.assertEqual(len(estimator.model_.components), 2)
            self.assertEqual(estimator.predict(X).shape, (4,))

    def test_npz_rejects_wrong_normalization_count(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "descriptors.npz"
            _write_archive(path)

            descriptors = CompositeDescriptor(
                names=[],
                paths=[str(path)],
                normalizations=["none", "standard", "log_standard"],
            )
            with self.assertRaisesRegex(
                ValueError, "Provide either one normalization or one per descriptor"
            ):
                descriptors.load_descriptor_blocks()

    def test_npz_rejects_mismatched_sample_counts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "descriptors.npz"
            _write_archive(path, first_samples=4, second_samples=3)

            descriptors = CompositeDescriptor(
                names=[],
                paths=[str(path)],
                normalizations=["standard"],
            )
            with self.assertRaisesRegex(ValueError, "has 3 samples, but expected 4"):
                descriptors.load_descriptor_blocks()

    def test_npy_loading_keeps_existing_one_file_per_descriptor_contract(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "descriptor.npy"
            values = np.arange(12, dtype=float).reshape(4, 3)
            np.save(path, values)

            descriptors = CompositeDescriptor(
                names=["configured-name"],
                paths=[str(path)],
                normalizations=["none"],
            )
            blocks = descriptors.load_descriptor_blocks()

            self.assertEqual(blocks[0].name, "configured-name")
            self.assertIsNone(blocks[0].description)
            np.testing.assert_array_equal(blocks[0].values, values)


if __name__ == "__main__":
    unittest.main()
