"""Indexed access to immutable compressed descriptor shards and virtual concatenation."""
import hashlib
import json
from pathlib import Path

import numpy as np


class ArchivePartCache:
    """One decompressed archive part shared by all component descriptor views."""

    def __init__(self):
        self.path = None
        self.identity = None
        self.arrays = {}
        self.verified = {}

    def read(self, path, sha256, keys):
        stat = path.stat()
        identity = (stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
        if self.verified.get((path, sha256)) != identity:
            with path.open("rb") as handle:
                actual = hashlib.file_digest(handle, "sha256").hexdigest()
            if actual != sha256:
                raise ValueError(f"Descriptor shard checksum mismatch: {path}")
            self.verified[(path, sha256)] = identity
        if self.path != path or self.identity != identity:
            self.path, self.identity, self.arrays = path, identity, {}
        missing = [key for key in keys if key not in self.arrays]
        if missing:
            with np.load(path, allow_pickle=False) as archive:
                for key in missing:
                    self.arrays[key] = archive[key]
        return [self.arrays[key] for key in keys]


_PART_CACHE = ArchivePartCache()


def load_manifest(path):
    path = Path(path)
    manifest = json.loads(path.read_text())
    if type(manifest.get("schema_version")) is not int or manifest["schema_version"] != 1:
        raise ValueError("Descriptor manifest schema_version must be integer 1.")
    arrays, parts = manifest.get("arrays"), manifest.get("parts")
    if not isinstance(arrays, dict) or not arrays or not isinstance(parts, list) or not parts:
        raise ValueError("Descriptor manifest requires nonempty arrays and parts.")
    row_counts = set()
    for key, record in arrays.items():
        shape = record.get("shape")
        if not isinstance(key, str) or not key or not isinstance(shape, list) or len(shape) < 2:
            raise ValueError("Descriptor manifest array entries require a name and sample/feature shape.")
        if any(type(size) is not int or size <= 0 for size in shape):
            raise ValueError("Descriptor manifest shapes must contain positive integers.")
        dtype = np.dtype(record.get("dtype"))
        if not np.issubdtype(dtype, np.number) or np.issubdtype(dtype, np.complexfloating):
            raise ValueError("Descriptor manifest arrays must have real numeric dtypes.")
        row_counts.add(shape[0])
        source_keys = record.get("source_keys")
        if source_keys is not None:
            if (not isinstance(source_keys, list) or not source_keys
                    or any(source not in arrays or arrays[source].get("source_keys") is not None for source in source_keys)):
                raise ValueError("Virtual descriptor source_keys must name stored arrays.")
            width = sum(int(np.prod(arrays[source]["shape"][1:])) for source in source_keys)
            if shape != [shape[0], width]:
                raise ValueError("Virtual descriptor shape must match flattened source widths.")
    if len(row_counts) != 1:
        raise ValueError("Descriptor manifest arrays must have the same sample count.")
    next_start = 0
    for part in parts:
        if type(part.get("start")) is not int or type(part.get("stop")) is not int:
            raise ValueError("Descriptor part bounds must be integers.")
        if part["start"] != next_start or part["stop"] <= next_start:
            raise ValueError("Descriptor parts must cover rows contiguously without overlap.")
        next_start = part["stop"]
        shard = part.get("path")
        checksum = part.get("sha256")
        if not isinstance(shard, str) or not shard or Path(shard).suffix.lower() != ".npz":
            raise ValueError("Descriptor shard path must name an NPZ archive.")
        if not isinstance(checksum, str) or len(checksum) != 64 or any(c not in "0123456789abcdef" for c in checksum):
            raise ValueError("Descriptor shard requires its lowercase SHA-256.")
    if next_start != next(iter(row_counts)):
        raise ValueError("Descriptor parts must cover the complete declared sample axis.")
    return manifest


class ShardedDescriptorArray:
    """Array-like shape and indexed reads; full-array conversion is intentionally unavailable."""

    def __init__(self, path, archive_key=None):
        self.path = Path(path).resolve()
        self.manifest = load_manifest(self.path)
        arrays = self.manifest["arrays"]
        if archive_key is None:
            if len(arrays) != 1:
                raise ValueError("archive_key is required for a manifest with multiple arrays.")
            archive_key = next(iter(arrays))
        if archive_key not in arrays:
            raise ValueError(f"Unknown descriptor archive_key {archive_key!r}.")
        self.archive_key = archive_key
        record = arrays[archive_key]
        self.shape = tuple(record["shape"])
        self.dtype = np.dtype(record["dtype"])
        self.ndim = len(self.shape)
        self.source_keys = record.get("source_keys", [archive_key])
        self.starts = np.array([part["start"] for part in self.manifest["parts"]], dtype=np.int64)
        self.cache = _PART_CACHE

    def __len__(self):
        return self.shape[0]

    def __getitem__(self, selection):
        scalar = np.isscalar(selection)
        if isinstance(selection, slice):
            rows = np.arange(*selection.indices(len(self)), dtype=np.int64)
        else:
            rows = np.asarray(selection)
            if rows.dtype == np.bool_:
                if rows.shape != (len(self),):
                    raise IndexError("Boolean descriptor index has the wrong shape.")
                rows = np.flatnonzero(rows)
            if rows.ndim > 1 or not np.issubdtype(rows.dtype, np.integer):
                raise IndexError("Descriptor row selection must be an integer vector or slice.")
            rows = rows.reshape(-1).astype(np.int64, copy=True)
            rows[rows < 0] += len(self)
            if np.any(rows < 0) or np.any(rows >= len(self)):
                raise IndexError("Descriptor row index is out of range.")
        out = np.empty((len(rows), *self.shape[1:]), dtype=self.dtype)
        if len(rows):
            part_ids = np.searchsorted(self.starts, rows, side="right") - 1
            for part_id in np.unique(part_ids):
                part = self.manifest["parts"][part_id]
                path = Path(part["path"])
                path = path if path.is_absolute() else self.path.parent / path
                blocks = self.cache.read(path.resolve(), part["sha256"], self.source_keys)
                for key, block in zip(self.source_keys, blocks):
                    declared = self.manifest["arrays"][key]
                    expected = (part["stop"] - part["start"], *declared["shape"][1:])
                    if block.shape != expected or block.dtype != np.dtype(declared["dtype"]):
                        raise ValueError(f"Descriptor shard array {key} differs from declared shape/dtype: {path}")
                positions = np.flatnonzero(part_ids == part_id)
                local = rows[positions] - part["start"]
                if "source_keys" in self.manifest["arrays"][self.archive_key]:
                    out[positions] = np.concatenate([block[local].reshape(len(local), -1) for block in blocks], axis=1)
                else:
                    out[positions] = blocks[0][local]
        return out[0] if scalar else out

    def __array__(self, dtype=None, copy=None):
        raise TypeError("Select descriptor rows explicitly; a complete sharded descriptor cannot be materialized implicitly.")


def load_descriptor_array(path, archive_key=None):
    path = Path(path)
    if path.suffix.lower() == ".json":
        return ShardedDescriptorArray(path, archive_key)
    if path.suffix.lower() != ".npy" or archive_key is not None:
        raise ValueError("Descriptor inputs must be NPY arrays or JSON shard manifests; archive_key applies to manifests only.")
    return np.load(path, mmap_mode="r", allow_pickle=False)
