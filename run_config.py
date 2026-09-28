"""Typed, serializable configuration and single-run process interface."""

from __future__ import annotations

import hashlib
import json
import os
import pprint
import subprocess
import sys
import tempfile
import tomllib
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping


DEFAULTS: dict[str, Any] = {
    "VERBOSITY": 2,
    "RUN_NAME": "krr-run",
    "USE_PREDEFINED_SPLITS": False,
    "N_KFOLD": 5,
    "STRATIFY": False,
    "N_STRATA": 5,
    "KRR_BACKEND": "exact",
    "KRR_ALPHA_BOUNDS": [1e-9, 1e2],
    "KRR_GAMMA_BOUNDS": [1e-9, 1e2],
    "KRR_RANDOM_SEARCH_STAGE1": 75,
    "KRR_RANDOM_SEARCH_STAGE2": 75,
    "KRR_RANDOM_SEARCH_STAGE3": 0,
    "KRR_TOP_K_FRACTION": 0.25,
    "KRR_TOP_K_MIN_CANDIDATES": 5,
    "KRR_BAYESIAN_SEARCH_TRIALS": 50,
    "KRR_BAYESIAN_SEARCH_TIMEOUT": None,
    "KRR_BAYESIAN_SEARCH_PATIENCE": 25,
    "KRR_BAYESIAN_BATCH_SIZE": 1,
    "KRR_EVALUATE_KERNEL_CONTRIBUTIONS": False,
    "KRR_KERNEL_CONTRIBUTION_BAYESIAN_SEARCH_TRIALS": None,
    "KRR_RANDOM_SEARCH_N_JOBS": -1,
    "KRR_RANDOM_SEARCH_BLAS_THREADS": 1,
    "KRR_COMPUTE_DTYPE": "float64",
    "KRR_USE_DISTANCE_CACHE": True,
    "KRR_DISTANCE_BLOCK_SIZE": 2048,
    "KRR_DISTANCE_CACHE_DTYPE": "float64",
    "KRR_DISTANCE_CACHE_N_JOBS": -1,
    "KRR_DISTANCE_CACHE_MEMORY_FRACTION": 0.8,
    "KRR_GAMMA_PRIOR_MAX_SAMPLES": 5000,
    "KRR_CACHED_SCORING_BACKEND": "numpy",
    "KRR_PYTORCH_DEVICE": "auto",
    "KRR_PYTORCH_DEVICES": None,
    "KRR_PYTORCH_CANDIDATE_BATCH_SIZE": 1,
    "KRR_PYTORCH_PREDICT_BATCH_SIZE": 2048,
    "KRR_NYSTROM_N_LANDMARKS": 2048,
    "KRR_NYSTROM_LANDMARK_SELECTION": "random",
    "KRR_NYSTROM_BATCH_SIZE": 2048,
    "KRR_NYSTROM_EIGENVALUE_FLOOR": 1e-12,
    "KRR_SCORE_METRIC": "neg_mean_absolute_error",
    "OVERWRITE_OK": False,
}

REQUIRED_FIELDS = {
    "SEED",
    "X_PATHS",
    "X_NAMES",
    "X_NORMS",
    "Y_PATH",
    "Y_NAME",
    "Y_NORM",
    "N_SAMPLES",
    "TRAIN_VAL_SPLIT",
    "KRR_KERNEL",
    "OUTPUT_DIR",
}

OPTIONAL_FIELDS = {
    "N_TRAIN",
    "X_PCA_COMPONENTS",
    "X_PCA_WHITEN",
    "PREDEF_TRAINING_IDX_PATH",
    "PREDEF_VAL_KFOLD_IDX_PATH",
    "PREDEF_TESTING_IDX_PATH",
}

ALLOWED_FIELDS = REQUIRED_FIELDS | OPTIONAL_FIELDS | set(DEFAULTS)
PATH_FIELDS = {
    "Y_PATH",
    "OUTPUT_DIR",
    "PREDEF_TRAINING_IDX_PATH",
    "PREDEF_VAL_KFOLD_IDX_PATH",
    "PREDEF_TESTING_IDX_PATH",
}


def _json_copy(value: Mapping[str, Any]) -> dict[str, Any]:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    decoded = json.loads(encoded)
    if not isinstance(decoded, dict):
        raise TypeError("KRR configuration must be an object")
    return decoded


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


@dataclass(frozen=True)
class KRRConfig:
    """A fully resolved, JSON-safe configuration for one KRR run."""

    values: dict[str, Any]
    source_path: Path | None = None

    @classmethod
    def from_mapping(
        cls,
        values: Mapping[str, Any],
        *,
        source_path: str | Path | None = None,
    ) -> "KRRConfig":
        unknown = set(values) - ALLOWED_FIELDS
        if unknown:
            raise ValueError(f"Unknown KRR configuration fields: {sorted(unknown)}")
        resolved = {**DEFAULTS, **_json_copy(values)}
        missing = sorted(REQUIRED_FIELDS - set(values))
        if missing:
            raise ValueError(f"Missing required KRR configuration fields: {missing}")
        if "N_TRAIN" not in resolved:
            resolved["N_TRAIN"] = int(
                float(resolved["N_SAMPLES"]) * float(resolved["TRAIN_VAL_SPLIT"])
            )
        source = None if source_path is None else Path(source_path).resolve()
        configuration = cls(_json_copy(resolved), source)
        configuration.validate(check_paths=False)
        return configuration

    @classmethod
    def load(cls, path: str | Path) -> "KRRConfig":
        source = Path(path).resolve()
        if source.suffix.lower() == ".toml":
            document = tomllib.loads(source.read_text())
        else:
            document = json.loads(source.read_text())
        if "krr" in document and isinstance(document["krr"], dict):
            document = document["krr"]
        if not isinstance(document, dict):
            raise TypeError("KRR configuration file must contain an object")
        return cls.from_mapping(document, source_path=source)

    def resolved(self) -> "KRRConfig":
        base = self.source_path.parent if self.source_path is not None else Path.cwd()
        values = _json_copy(self.values)
        values["X_PATHS"] = [str(_resolve_path(base, item)) for item in values["X_PATHS"]]
        for field in PATH_FIELDS:
            if values.get(field) is not None:
                values[field] = str(_resolve_path(base, values[field]))
        return KRRConfig(values, self.source_path)

    def validate(self, *, check_paths: bool = True) -> None:
        values = self.values
        if values["SEED"] is not None and not _is_int(values["SEED"]):
            raise ValueError("SEED must be an integer or null")
        if not _is_int(values["N_SAMPLES"]) or values["N_SAMPLES"] <= 0:
            raise ValueError("N_SAMPLES must be a positive integer")
        if not 0 < float(values["TRAIN_VAL_SPLIT"]) < 1:
            raise ValueError("TRAIN_VAL_SPLIT must be between zero and one")
        for field in ("X_PATHS", "X_NAMES", "X_NORMS"):
            if not isinstance(values[field], list) or not values[field]:
                raise ValueError(f"{field} must be a non-empty list")
        width = len(values["X_PATHS"])
        if len(values["X_NAMES"]) != width or len(values["X_NORMS"]) != width:
            raise ValueError("X_PATHS, X_NAMES, and X_NORMS must have equal lengths")
        kernels = values["KRR_KERNEL"]
        if isinstance(kernels, list) and len(kernels) != width:
            raise ValueError("KRR_KERNEL list length must match X_PATHS")
        if not isinstance(kernels, (str, list)):
            raise ValueError("KRR_KERNEL must be a string or list of strings")
        if values["USE_PREDEFINED_SPLITS"]:
            required = {
                "PREDEF_TRAINING_IDX_PATH",
                "PREDEF_VAL_KFOLD_IDX_PATH",
                "PREDEF_TESTING_IDX_PATH",
            }
            missing = sorted(name for name in required if not values.get(name))
            if missing:
                raise ValueError(f"Predefined split configuration is missing: {missing}")

        # Reuse the library's detailed cross-field validation without changing
        # its compatibility import of config.py.
        import config_validation

        original = config_validation.config
        config_validation.config = SimpleNamespace(**values)
        try:
            config_validation.validate_config()
        finally:
            config_validation.config = original

        if check_paths:
            resolved = self.resolved().values
            input_paths = [Path(item) for item in resolved["X_PATHS"]]
            input_paths.append(Path(resolved["Y_PATH"]))
            if resolved["USE_PREDEFINED_SPLITS"]:
                input_paths.extend(
                    Path(resolved[name])
                    for name in (
                        "PREDEF_TRAINING_IDX_PATH",
                        "PREDEF_VAL_KFOLD_IDX_PATH",
                        "PREDEF_TESTING_IDX_PATH",
                    )
                )
            missing_paths = [str(path) for path in input_paths if not path.is_file()]
            if missing_paths:
                raise FileNotFoundError(f"KRR input files do not exist: {missing_paths}")

    def canonical_json(self) -> str:
        return json.dumps(
            self.resolved().values,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )

    def sha256(self) -> str:
        return hashlib.sha256(self.canonical_json().encode()).hexdigest()

    def python_source(self) -> str:
        lines = ["# Generated from a validated resolved KRR configuration.\n"]
        for name in sorted(self.resolved().values):
            lines.append(f"{name} = {pprint.pformat(self.resolved().values[name])}\n")
        return "".join(lines)


@dataclass(frozen=True)
class RunExecution:
    returncode: int
    output_directory: Path
    config_sha256: str


def run(configuration: KRRConfig, *, check: bool = True) -> RunExecution:
    """Execute one validated configuration in an isolated compatibility process."""

    resolved = configuration.resolved()
    resolved.validate(check_paths=True)
    output = Path(resolved.values["OUTPUT_DIR"])
    output.mkdir(parents=True, exist_ok=True)
    resolved_path = output / "resolved_config.json"
    document = json.loads(resolved.canonical_json())
    if resolved_path.exists():
        if json.loads(resolved_path.read_text()) != document:
            raise FileExistsError(f"Changed resolved configuration at {resolved_path}")
    else:
        _write_json_atomic(resolved_path, document)

    root = Path(__file__).resolve().parent
    with tempfile.TemporaryDirectory(prefix="ckrr-run-") as temporary:
        work = Path(temporary)
        (work / "config.py").write_text(resolved.python_source())
        environment = dict(os.environ)
        prior_pythonpath = environment.get("PYTHONPATH")
        environment["PYTHONPATH"] = (
            str(root)
            if not prior_pythonpath
            else os.pathsep.join((str(root), prior_pythonpath))
        )
        completed = subprocess.run(
            [sys.executable, str(root / "main.py")],
            cwd=work,
            env=environment,
            check=False,
        )
    if check and completed.returncode:
        raise subprocess.CalledProcessError(completed.returncode, completed.args)
    return RunExecution(completed.returncode, output, resolved.sha256())


def json_schema() -> dict[str, Any]:
    """Return a machine-readable discovery schema for agents and editors."""

    properties: dict[str, Any] = {
        name: {"description": "Composite-KRR configuration field"}
        for name in sorted(ALLOWED_FIELDS)
    }
    typed = {
        "SEED": ["integer", "null"],
        "X_PATHS": "array",
        "X_NAMES": "array",
        "X_NORMS": "array",
        "Y_PATH": "string",
        "Y_NAME": "string",
        "Y_NORM": "string",
        "N_SAMPLES": "integer",
        "TRAIN_VAL_SPLIT": "number",
        "KRR_KERNEL": ["string", "array"],
        "OUTPUT_DIR": "string",
    }
    for name, field_type in typed.items():
        properties[name]["type"] = field_type
    for name, value in DEFAULTS.items():
        properties[name]["default"] = value
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "Composite-KRR resolved run configuration",
        "type": "object",
        "additionalProperties": False,
        "required": sorted(REQUIRED_FIELDS),
        "properties": properties,
    }


def template() -> dict[str, Any]:
    return {
        "SEED": 1,
        "X_PATHS": ["/absolute/path/descriptor.npy"],
        "X_NAMES": ["descriptor"],
        "X_NORMS": ["standard"],
        "Y_PATH": "/absolute/path/target.npy",
        "Y_NAME": "target",
        "Y_NORM": "standard",
        "N_SAMPLES": 1000,
        "TRAIN_VAL_SPLIT": 0.8,
        "KRR_KERNEL": "rbf",
        "OUTPUT_DIR": "/absolute/path/output",
    }


def _resolve_path(base: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def _write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    payload = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(payload)
    os.replace(temporary, path)

