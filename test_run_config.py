from __future__ import annotations

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

    def test_confirmed_search_defaults_match_template_and_explicit_old_recipe(self):
        data = self.sample()
        data.pop("search")
        cfg = KRRConfig.from_mapping(data)
        search = cfg.spec.search.model_dump(mode="json")
        self.assertEqual(search["random"], {"stage1": 200, "stage2": 200, "stage3": 50})
        bayes = search["bayesian"]
        self.assertEqual((bayes["trials"], bayes["rounds"], bayes["patience"]), (200, 4, 75))
        self.assertEqual((bayes["refine_decades"], bayes["weight_logit_radius"]), (.6, 3.))
        self.assertTrue(bayes["recenter"] and bayes["reuse_observations"] and bayes["sampler"]["multivariate"])
        self.assertEqual(template()["search"], search)
        bundled = KRRConfig.load(ROOT / "config.json")
        self.assertEqual(bundled.spec.search.model_dump(mode="json"), search)
        old = cfg.with_overrides({"search.random": {"stage1": 75, "stage2": 75, "stage3": 0},
            "search.bayesian.trials": 50, "search.bayesian.rounds": 1,
            "search.bayesian.refine_decades": .3, "search.bayesian.weight_logit_radius": 1.5,
            "search.bayesian.reuse_observations": False, "search.bayesian.sampler.multivariate": False,
            "search.bayesian.patience": 25})
        self.assertEqual(old.spec.search.bayesian.rounds, 1)
        self.assertFalse(old.spec.search.bayesian.reuse_observations)
        self.assertEqual(old.spec.search.random.stage1, 75)

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

    def test_workspace_binding_is_independent_of_engine_source_location(self):
        data = self.sample()
        data["descriptors"][0]["path"] = "@krr/sample/a.npy"
        data["target"]["path"] = "@workspace/custom-inputs/target.npy"
        data["output"]["directory"] = "@reports/output"
        with tempfile.TemporaryDirectory() as temporary:
            workspace = Path(temporary)
            with patch.dict(os.environ, {"KRR_WORKSPACE_ROOT": str(workspace)}):
                config = KRRConfig.from_mapping(data).resolved()
                self.assertEqual(config.spec.descriptors[0].path, str(workspace / "Composite-KRR/sample/a.npy"))
                self.assertEqual(config.spec.target.path, str(workspace / "custom-inputs/target.npy"))
                self.assertEqual(config.spec.output.directory, str(workspace / "reports/output"))
                explicit = KRRConfig.from_mapping(data).resolved(workspace=workspace / "explicit")
                self.assertEqual(explicit.spec.output.directory, str(workspace / "explicit/reports/output"))

    def test_kernel_alignment_schema_template_overrides_and_legacy_safety(self):
        base = KRRConfig.from_mapping(self.sample())
        self.assertFalse(base.spec.reporting.kernel_alignment.enabled)
        self.assertEqual(template()["reporting"]["kernel_alignment"],
                         {"enabled": False, "max_samples": 1000, "include_products": True,
                          "include_target": False, "target_space": "original"})
        definition = json_schema()["$defs"]["KernelAlignmentConfig"]
        self.assertEqual(definition["properties"]["max_samples"]["minimum"], 3)
        self.assertFalse(definition["additionalProperties"])
        enabled = base.with_overrides({"reporting.kernel_alignment.enabled": True,
                                       "reporting.kernel_alignment.max_samples": 6,
                                       "reporting.kernel_alignment.include_products": False,
                                       "reporting.kernel_alignment.include_target": True,
                                       "reporting.kernel_alignment.target_space": "transformed"})
        restored = KRRConfig.from_mapping(json.loads(enabled.canonical_json()))
        self.assertEqual(restored.spec.reporting.kernel_alignment.max_samples, 6)
        self.assertTrue(restored.spec.reporting.kernel_alignment.include_target)
        self.assertEqual(restored.spec.reporting.kernel_alignment.target_space, "transformed")
        self.assertEqual(restored.sha256(), enabled.sha256())
        self.assertNotEqual(base.sha256(), enabled.sha256())
        with self.assertRaisesRegex(ValueError, "structured"):
            _ = enabled.values
        for change in ({"reporting.kernel_alignment": True},
                       {"reporting.kernel_alignment.enabled": "true"},
                       {"reporting.kernel_alignment.max_samples": 2},
                       {"reporting.kernel_alignment.max_samples": True},
                       {"reporting.kernel_alignment.max_samples": 3.0},
                       {"reporting.kernel_alignment.include_products": 1},
                       {"reporting.kernel_alignment.include_target": "true"},
                       {"reporting.kernel_alignment.include_target": True},
                       {"reporting.kernel_alignment.target_space": "normalized"},
                       {"reporting.kernel_alignment.typo": True}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                base.with_overrides(change)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "run.toml"
            path.write_text('[[descriptors]]\nid="a"\npath="a.npy"\n[target]\npath="y.npy"\n'
                            '[split]\nmode="random"\nn_train=10\n[output]\ndirectory="out"\n'
                            '[reporting.kernel_alignment]\nenabled=true\nmax_samples=7\n'
                            'include_target=true\ntarget_space="transformed"\n')
            saved = KRRConfig.load(path).spec.reporting.kernel_alignment
            self.assertEqual(saved.max_samples, 7)
            self.assertTrue(saved.include_target)
            self.assertEqual(saved.target_space, "transformed")

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

    def test_import_main_preserves_the_callers_logging_handlers(self):
        completed = subprocess.run([sys.executable, '-c',
            'import logging; root=logging.getLogger(); handler=logging.NullHandler(); '
            'root.addHandler(handler); import main; assert root.handlers == [handler]'],
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

    def test_cli_resolve_and_migrate_freeze_portable_configs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            base = root / 'legacy.json'
            base.write_text(json.dumps({"krr": template(legacy=True)}))
            frozen = root / 'elsewhere/frozen.json'
            command = [sys.executable, str(ROOT / 'krr_cli.py'), 'migrate', str(base),
                       '--no-check-paths', '--set', 'split.n_train=12', '--output', str(frozen)]
            for _ in range(2):
                result = subprocess.run(command, capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stderr)
            saved = KRRConfig.load(frozen)
            self.assertEqual(saved.spec.split.n_train, 12)
            self.assertEqual(saved.spec.descriptors[0].path, str(root / 'descriptor.npy'))
            result = subprocess.run([*command, '--set', 'seed=7'], capture_output=True, text=True)
            self.assertEqual(result.returncode, 2)
            self.assertIn('Refusing to replace', result.stderr)

    def test_predefined_npz_run_preserves_indices_and_multiple_targets(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            rng = np.random.default_rng(21)
            for filename in ['a.npy', 'b.npy']:
                np.save(root / filename, rng.normal(size=(20, 3)))
            np.savez(root / 'target.npz', one=rng.normal(size=20), two=rng.normal(size=20))
            np.save(root / 'train.npy', np.arange(6))
            np.savez(root / 'folds.npz', fold0=np.arange(6, 8), fold1=np.arange(8, 10))
            np.save(root / 'test.npy', np.arange(10, 20))
            data = self.sample()
            data['split'] = {'mode': 'predefined', 'train_indices': 'train.npy',
                             'validation_folds': 'folds.npz', 'test_indices': 'test.npy'}
            data['target'] = {'path': 'target.npz', 'name': ['one', 'two']}
            result = run(KRRConfig.from_mapping(data, source_path=root / 'run.json'))
            np.testing.assert_array_equal(np.load(result.output_dir / 'fold_val_idx.npy'), np.arange(10))
            np.testing.assert_array_equal(np.load(result.output_dir / 'test_idx.npy'), np.arange(10, 20))
            self.assertEqual(np.load(result.output_dir / 'y_predictions.npy').shape, (10, 2))

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
