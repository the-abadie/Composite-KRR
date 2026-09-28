"""Load, validate, resolve, and execute independent JSON/TOML run configurations."""
from __future__ import annotations

import hashlib
import json
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from config_schema import RunConfig
from legacy_config import PREDEFINED_FIELDS, ALLOWED_FIELDS, from_flat, to_flat


def _object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _invalid_constant(value):
    raise ValueError(f"Non-finite JSON number: {value}")


def parse_json(text):
    return json.loads(text, object_pairs_hook=_object, parse_constant=_invalid_constant)


def parse_assignment(value: str) -> tuple[str, Any]:
    key, separator, raw = value.partition("=")
    if not separator or not all(part.isidentifier() for part in key.split(".")):
        raise ValueError(f"Expected field.path=JSON, got {value!r}")
    return key, parse_json(raw)


def write_immutable(path: Path, text: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x") as handle:
            handle.write(text)
    except FileExistsError:
        if path.read_text() != text:
            raise FileExistsError(f"Refusing to replace different configuration: {path}") from None


@dataclass(frozen=True)
class KRRConfig:
    spec: RunConfig
    source_path: Path | None = None

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any], *, source_path=None):
        if not isinstance(values, Mapping):
            raise ValueError("KRR configuration must be an object.")
        document = dict(values)
        if set(document) == {"krr"}:
            if not isinstance(document["krr"], dict):
                raise ValueError("krr must contain a configuration object.")
            document = document["krr"]
        if any(key in ALLOWED_FIELDS for key in document):
            document = from_flat(document)
        # Literal equality alone also accepts True and 1.0.
        if "schema_version" in document and type(document["schema_version"]) is not int:
            raise ValueError("schema_version must be integer 1.")
        return cls(RunConfig.model_validate(document), Path(source_path).resolve() if source_path else None)

    @classmethod
    def load(cls, path):
        path = Path(path).resolve()
        if path.suffix.lower() == ".json":
            values = parse_json(path.read_text())
        elif path.suffix.lower() == ".toml":
            values = tomllib.loads(path.read_text())
        else:
            raise ValueError("Configuration must be JSON or TOML; Python files are not executed.")
        return cls.from_mapping(values, source_path=path)

    @property
    def values(self):
        """Legacy flat view for callers migrating to .spec; never shared mutable state."""
        return to_flat(self.spec)

    def with_overrides(self, overrides: Mapping[str, Any]):
        if not overrides:
            return self
        if any(key in ALLOWED_FIELDS for key in overrides):
            if any(key not in ALLOWED_FIELDS for key in overrides):
                raise ValueError("Do not mix legacy uppercase and structured overrides.")
            values = self.values
            if self.spec.split.mode == "random" and self.spec.split.train_fraction is not None:
                values.pop("N_TRAIN", None)
            if "N_TRAIN" in overrides and "TRAIN_VAL_SPLIT" not in overrides:
                values.pop("TRAIN_VAL_SPLIT", None)
            if "TRAIN_VAL_SPLIT" in overrides and "N_TRAIN" not in overrides:
                values.pop("N_TRAIN", None)
            values.update(overrides)
            return self.from_mapping(values, source_path=self.source_path)
        document = self.spec.model_dump(mode="json")
        if "split.n_train" in overrides and "split.train_fraction" not in overrides:
            document["split"].pop("train_fraction", None)
        if "split.train_fraction" in overrides and "split.n_train" not in overrides:
            document["split"].pop("n_train", None)
        for path, value in overrides.items():
            target = document
            keys = path.split(".")
            for key in keys[:-1]:
                if key not in target or not isinstance(target[key], dict):
                    raise ValueError(f"Unknown configuration path: {path}")
                target = target[key]
            target[keys[-1]] = value
        return self.from_mapping(document, source_path=self.source_path)

    def resolved(self, *, workspace=None):
        base = self.source_path.parent if self.source_path else Path.cwd()
        workspace = Path(workspace).resolve() if workspace else Path(__file__).resolve().parent.parent
        roots = {"@workspace": workspace, "@krr": workspace / "Composite-KRR",
                 "@experiments": workspace / "experiments", "@reports": workspace / "reports"}
        def resolve(value):
            for token, root in roots.items():
                if value == token or value.startswith(token + "/"):
                    return str((root / value[len(token):].lstrip("/")).resolve())
            if value.startswith("@"):
                raise ValueError(f"Unknown workspace path token: {value}")
            path = Path(value).expanduser()
            return str((path if path.is_absolute() else base / path).resolve())
        document = self.spec.model_dump(mode="json")
        for descriptor in document["descriptors"]:
            descriptor["path"] = resolve(descriptor["path"])
        document["target"]["path"] = resolve(document["target"]["path"])
        document["output"]["directory"] = resolve(document["output"]["directory"])
        if self.spec.split.mode == "predefined":
            for field in PREDEFINED_FIELDS.values():
                document["split"][field] = resolve(document["split"][field])
        return self.from_mapping(document, source_path=self.source_path)

    def portable(self, workspace):
        """Resolve relative paths, retaining workspace-relative tokens for cluster transfer."""
        root = Path(workspace).resolve()
        document = self.resolved(workspace=root).spec.model_dump(mode="json")
        def convert(value):
            try:
                return "@workspace/" + str(Path(value).relative_to(root))
            except ValueError:
                return value
        for descriptor in document["descriptors"]:
            descriptor["path"] = convert(descriptor["path"])
        document["target"]["path"] = convert(document["target"]["path"])
        document["output"]["directory"] = convert(document["output"]["directory"])
        if self.spec.split.mode == "predefined":
            for field in PREDEFINED_FIELDS.values():
                document["split"][field] = convert(document["split"][field])
        return self.from_mapping(document)

    def validate(self, *, check_paths=True):
        spec = self.resolved().spec
        if check_paths:
            paths = [(d.path, {".npy"}) for d in spec.descriptors]
            paths.append((spec.target.path, {".npy", ".npz"}))
            if spec.split.mode == "predefined":
                paths.extend([(spec.split.train_indices, {".npy"}),
                              (spec.split.validation_folds, {".npy", ".npz"}),
                              (spec.split.test_indices, {".npy"})])
            for value, extensions in paths:
                path = Path(value)
                if path.suffix.lower() not in extensions:
                    raise ValueError(f"Expected {sorted(extensions)} input: {path}")
                if not path.is_file():
                    raise FileNotFoundError(f"Input file does not exist: {path}")
        from sklearn.metrics import get_scorer
        get_scorer("neg_root_mean_squared_error" if spec.search.scoring == "rmse" else spec.search.scoring)

    def canonical_json(self):
        return json.dumps(self.spec.model_dump(mode="json"), sort_keys=True, indent=2, allow_nan=False) + "\n"

    def sha256(self):
        return hashlib.sha256(self.canonical_json().encode()).hexdigest()


@dataclass(frozen=True)
class RunExecution:
    returncode: int
    output_dir: Path
    config_sha256: str

    @property
    def output_directory(self):
        return self.output_dir


def run(configuration: KRRConfig) -> RunExecution:
    """Execute with an explicit configuration; no generated Python or global config imports."""
    from runner import run as execute
    return execute(configuration)


def json_schema():
    return RunConfig.model_json_schema()


def template(*, legacy=False):
    config = KRRConfig.from_mapping({
        "schema_version": 1, "seed": 1, "run_name": "example",
        "descriptors": [{"id": "descriptor", "path": "descriptor.npy", "kernel": "rbf"}],
        "target": {"path": "target.npy", "name": "target"},
        "split": {"mode": "random", "n_samples": 100, "train_fraction": 0.8},
        "output": {"directory": "output/example"},
    })
    if legacy:
        values = config.values
        values.pop("N_TRAIN", None)
        values.update(KRR_KERNEL="rbf", X_PCA_COMPONENTS=None, X_PCA_WHITEN=False)
        return values
    return config.spec.model_dump(mode="json")
