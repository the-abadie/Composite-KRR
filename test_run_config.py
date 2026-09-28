from __future__ import annotations

import copy
import json
import logging
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from run_config import KRRConfig, json_schema, parse_assignment, run, template, write_immutable

ROOT = Path(__file__).resolve().parent


class KRRConfigTests(unittest.TestCase):
    def sample(self):
        return {"schema_version": 1, "seed": 12,
                "descriptors": [{"id": "a", "path": "a.npy", "kernel": "rbf"},
                                {"id": "b", "path": "b.npy", "kernel": "laplacian"}],
                "target": {"path": "target.npy"},
                "split": {"mode": "random", "n_samples": 20, "n_train": 12, "n_folds": 2},
                "kernel_products": [["a", "b"]], "output": {"directory": "output"},
                "search": {"random": {"stage1": 1, "stage2": 1}, "bayesian": {"trials": 0}},
                "execution": {"n_jobs": 1, "distance_cache": {"n_jobs": 1}}}

    def test_defaults_hash_schema_and_no_shared_state(self):
        first = KRRConfig.from_mapping(self.sample())
        second = KRRConfig.from_mapping(json.loads(first.canonical_json()))
        self.assertEqual(first.sha256(), second.sha256())
        self.assertEqual(first.values["N_TRAIN"], 12)
        first.values["X_NAMES"].append("mutated")
        self.assertEqual(first.values["X_NAMES"], ["a", "b"])
        schema = json_schema()
        self.assertIn("descriptors", schema["required"])
        self.assertFalse(schema["additionalProperties"])
        self.assertFalse(schema["$defs"]["RandomSplit"]["additionalProperties"])

    def test_rejects_invalid_structured_values(self):
        cases = [({"execution.n_jobs": 0}, "n_jobs"), ({"seed": True}, "seed"),
                 ({"split.n_train": "12"}, "n_train"), ({"split.n_train": 12.0}, "n_train"),
                 ({"split.train_fraction": 0.9, "split.n_train": 12}, "exactly one"),
                 ({"split.n_train": 20}, "held-out"), ({"split.n_train": 1}, "n_folds"),
                 ({"model.typo": 12}, "extra_forbidden"),
                 ({"schema_version": 2}, "schema_version"), ({"schema_version": True}, "schema_version"),
                 ({"execution.distance_cache.enabled": "false"}, "bool"),
                 ({"search.alpha_bounds": [1, 0.1]}, "low < high"),
                 ({"search.gamma_bounds": [float("nan"), 1]}, "finite"),
                 ({"kernel_products": [["a", "c"]]}, "Unknown"),
                 ({"kernel_products": [["a", "b"], ["b", "a"]]}, "Duplicate"),
                 ({"kernel_products": [["a"]]}, "at least 2")]
        base = KRRConfig.from_mapping(self.sample())
        for overrides, message in cases:
            with self.subTest(overrides=overrides), self.assertRaisesRegex(ValueError, message):
                base.with_overrides(overrides)
        data = self.sample()
        data["descriptors"][1]["id"] = "a"
        with self.assertRaisesRegex(ValueError, "unique"):
            KRRConfig.from_mapping(data)

    def test_legacy_migration_and_overrides(self):
        base = KRRConfig.from_mapping(template(legacy=True))
        self.assertEqual(base.values["N_TRAIN"], 80)
        count = base.with_overrides({"N_TRAIN": 12})
        self.assertEqual(count.spec.split.n_train, 12)
        self.assertIsNone(count.spec.split.train_fraction)
        fraction = count.with_overrides({"split.train_fraction": 0.6})
        self.assertIsNone(fraction.spec.split.n_train)
        self.assertEqual(fraction.values["N_TRAIN"], 60)
        legacy = base.values
        legacy["N_TRAIN"] = 50
        with self.assertRaisesRegex(ValueError, "conflicts"):
            KRRConfig.from_mapping(legacy)
        legacy = base.values
        legacy["TYPO"] = 1
        with self.assertRaisesRegex(ValueError, "Unknown"):
            KRRConfig.from_mapping(legacy)
        with self.assertRaisesRegex(ValueError, "list"):
            base.with_overrides({"KRR_KERNEL_PRODUCTS": False})

    def test_name_references_and_backend_validation(self):
        data = self.sample()
        data["descriptors"].reverse()
        self.assertEqual(KRRConfig.from_mapping(data).values["KRR_KERNEL_PRODUCTS"], [[1, 0]])
        data["descriptors"][0]["kernel"] = "linear"
        data["execution"]["backend"] = "pytorch"
        with self.assertRaisesRegex(ValueError, "rbf or laplacian"):
            KRRConfig.from_mapping(data)

    def test_paths_freezing_and_json_errors(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = KRRConfig.from_mapping(self.sample(), source_path=root / "input.json")
            self.assertEqual(config.resolved().spec.descriptors[0].path, str(root / "a.npy"))
            portable = config.portable(root)
            self.assertEqual(portable.spec.descriptors[0].path, "@workspace/a.npy")
            self.assertEqual(portable.resolved(workspace=root / "remote").spec.descriptors[0].path,
                             str(root / "remote/a.npy"))
            frozen = config.resolved()
            path = root / "elsewhere" / "frozen.json"
            write_immutable(path, frozen.canonical_json())
            write_immutable(path, frozen.canonical_json())
            self.assertEqual(KRRConfig.load(path).resolved().sha256(), frozen.sha256())
            with self.assertRaises(FileExistsError):
                write_immutable(path, "changed")
            for text in ('{"seed":1,"seed":2}', '{"seed":NaN}'):
                path.write_text(text)
                with self.assertRaises(ValueError):
                    KRRConfig.load(path)
        self.assertEqual(parse_assignment('split.n_train=12'), ('split.n_train', 12))

    def test_predefined_and_toml(self):
        data = self.sample()
        data["split"] = {"mode": "predefined", "train_indices": "train.npy",
                         "validation_folds": "folds.npz", "test_indices": "test.npy"}
        self.assertTrue(KRRConfig.from_mapping(data).values["USE_PREDEFINED_SPLITS"])
        data["split"]["n_train"] = 12
        with self.assertRaises(ValueError):
            KRRConfig.from_mapping(data)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'run.toml'
            path.write_text('[[descriptors]]\nid="a"\npath="a.npy"\n[target]\npath="y.npy"\n'
                            '[split]\nmode="random"\nn_train=10\n[output]\ndirectory="output"\n')
            self.assertEqual(KRRConfig.load(path).spec.split.n_train, 10)

    def test_import_main_does_not_execute_or_import_python_config(self):
        completed = subprocess.run([sys.executable, '-c',
            'import sys, logging; h=list(logging.getLogger().handlers); import main, runner; '
            'assert "config" not in sys.modules; assert h == logging.getLogger().handlers'],
            cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_main_config_runs_and_records_exact_counts(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            rng = np.random.default_rng(10)
            np.save(root / 'a.npy', rng.normal(size=(20, 3)))
            np.save(root / 'b.npy', rng.normal(size=(20, 2)))
            np.save(root / 'target.npy', rng.normal(size=20))
            config = root / 'run.json'
            config.write_text(json.dumps(self.sample()))
            completed = subprocess.run([sys.executable, str(ROOT / 'main.py'), '--config', str(config)],
                cwd='/tmp', capture_output=True, text=True,
                env={**os.environ, 'MPLBACKEND': 'Agg', 'OPENBLAS_NUM_THREADS': '1'})
            self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
            output = root / 'output'
            self.assertEqual(len(np.load(output / 'fold_val_idx.npy')), 12)
            self.assertEqual(len(np.load(output / 'test_idx.npy')), 8)
            saved = KRRConfig.load(output / 'resolved_config.json')
            self.assertEqual(saved.spec.kernel_products, [['a', 'b']])
            self.assertEqual((output / 'resolved_config.sha256').read_text().strip(), saved.sha256())
            self.assertFalse((output / '.run.lock').exists())
            with self.assertRaises(FileExistsError):
                run(saved)
            changed = saved.with_overrides({'output.overwrite': True, 'seed': 2})
            with self.assertRaises(FileExistsError):
                run(changed)

    def test_sequential_runs_restore_logging_and_seed_is_recorded(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for filename in ['a.npy', 'b.npy', 'target.npy']:
                (root / filename).touch()
            base = KRRConfig.from_mapping(self.sample(), source_path=root / 'run.json')
            before = list(logging.getLogger().handlers)
            with patch('runner._execute') as execute:
                for index in [1, 2]:
                    result = run(base.with_overrides({'seed': None, 'output.directory': f'out{index}'}))
                    self.assertIsInstance(execute.call_args.args[0].seed, int)
                    self.assertEqual(execute.call_args.args[0].seed, KRRConfig.load(result.output_dir / 'resolved_config.json').spec.seed)
                    self.assertEqual(before, logging.getLogger().handlers)
            self.assertEqual(execute.call_count, 2)


if __name__ == '__main__':
    unittest.main()
