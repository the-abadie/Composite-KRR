from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import run_config
from run_config import KRRConfig, json_schema


class KRRConfigTests(unittest.TestCase):
    def sample(self, root: Path) -> dict:
        return {
            "SEED": 12,
            "X_PATHS": [str(root / "descriptor.npy")],
            "X_NAMES": ["descriptor"],
            "X_NORMS": ["standard"],
            "Y_PATH": str(root / "target.npy"),
            "Y_NAME": "target",
            "Y_NORM": "standard",
            "N_SAMPLES": 100,
            "TRAIN_VAL_SPLIT": 0.8,
            "KRR_KERNEL": "rbf",
            "OUTPUT_DIR": str(root / "output"),
        }

    def test_resolves_defaults_and_stable_hash(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = KRRConfig.from_mapping(self.sample(root))
            second = KRRConfig.from_mapping(json.loads(first.canonical_json()))
            self.assertEqual(first.sha256(), second.sha256())
            self.assertEqual(first.values["N_TRAIN"], 80)
            self.assertEqual(first.values["KRR_COMPUTE_DTYPE"], "float64")

    def test_rejects_unknown_fields(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            values = self.sample(Path(temporary))
            values["TYPO"] = True
            with self.assertRaisesRegex(ValueError, "Unknown"):
                KRRConfig.from_mapping(values)

    def test_schema_exposes_required_fields(self) -> None:
        schema = json_schema()
        self.assertIn("X_PATHS", schema["required"])
        self.assertFalse(schema["additionalProperties"])

    def test_run_imports_supplied_configuration(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "descriptor.npy").touch()
            (root / "target.npy").touch()
            # A source-tree config must never shadow the supplied run config.
            (root / "config.py").write_text('RUN_NAME = "source-default"\n')
            (root / "main.py").write_text(
                'import config\n'
                'assert config.RUN_NAME == "supplied-run", config.RUN_NAME\n'
                'assert config.KRR_KERNEL_PRODUCTS == [[0, 0]]\n'
            )
            values = self.sample(root)
            values.update(RUN_NAME="supplied-run", KRR_KERNEL_PRODUCTS=[[0, 0]])
            with patch.object(run_config, "__file__", str(root / "run_config.py")):
                execution = run_config.run(KRRConfig.from_mapping(values))
            self.assertEqual(execution.returncode, 0)


if __name__ == "__main__":
    unittest.main()
