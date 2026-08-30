#!/usr/bin/env python3
"""Build physically separated V8R4 discovery and outer-inference inputs.

The legacy v3r1 cache and proposer archive are combined, row-aligned files.
They are valid historical inputs, but merely giving a discovery process access
to those paths gives it the capability to inspect outer-fold values.  This
builder is a CPU-only, fail-closed boundary compiler:

* ``discovery_cache`` contains only rows whose fold differs from ``outer_fold``;
* ``discovery_proposer_stack.npz`` contains only the corresponding target-free
  proposer anchors and uses a local, contiguous cache index;
* immutable maps/manifests bind each emitted pack to the legacy bytes and prove an
  ordered, disjoint, exact partition of the legacy cache index.

Preselection discovery emits two capability shards (outer 3 and outer 4),
each containing exactly three seed packs and never mounted together.  It does
not create, require, name, or bind an outer-prediction pack.  Promotion training
uses the same physical nonouter partition, but a distinct authorized path seals
the exact promotion capability into every cache, partition, and shard manifest.
The separate ``prediction`` scope emits an outer-specific three-seed shard index
whose packs contain only the ten target-free inference fields.  Neither promoted
path has a CLI bypass around an immutable, exact-bound V8R4 authorization.

Outer reference, reference-validity, identity, protocol, and quality fields are
never decoded.  The metadata file is necessarily hashed as an opaque legacy
input binding, but its protected outer fields are not parsed, converted, or
emitted.  Cache arrays are opened with ``O_NOFOLLOW`` and indexed through a
read-only mmap.  Object arrays and pickle are forbidden in every emitted file.

All outputs are deterministic, create-once, mode 0444 regular files.  A resume
is accepted only when every pre-existing byte is identical to the byte the
builder would publish.
"""

from __future__ import annotations

import argparse
from contextlib import ExitStack
import ctypes
from dataclasses import dataclass
import errno
import hashlib
import io
import json
import mmap
import os
from pathlib import Path
import stat
from typing import Any, Callable, Iterable, Iterator, Mapping, Sequence
import zipfile

import numpy as np


SCHEMA_VERSION = 1
CAMPAIGN_ID = "directed_harmonic_factor_expert_snn_v3r1_adaptive_retrospective"
PACK_REVISION = "V8R4"
N_FOLDS = 6
SEEDS = (20260828, 20260829, 20260830)
DEFAULT_INDEX_RELATIVE = Path(
    "artifacts/runs/harmonic_candidate_set_snn_v2/"
    "hcs_fixed_i3_pretest_v2/pretest_index.json"
)
DEFAULT_INDEX_SHA256 = (
    "aba5657a76764ca1a0a28a3f45aa85034194a1168ec6835c34729817d2a19958"
)
DEFAULT_INDEX_BYTES = 278_077
DEFAULT_OUTPUT_RELATIVE = Path(
    "artifacts/runs/directed_harmonic_factor_expert_snn_v3r1/v8r4_split_inputs"
)
DEFAULT_PROMOTION_OUTPUT_RELATIVE = Path(
    "artifacts/runs/directed_harmonic_factor_expert_snn_v3r1/"
    "v8r4_promotion_authorized_inputs"
)
NONOUTER_PACK_CLASSIFICATION = (
    "adaptive_v3r1_v8r4_nonouter_training_validation_pack"
)
NONOUTER_STACK_CLASSIFICATION = (
    "adaptive_v3r1_v8r4_nonouter_causal_proposer_stack"
)
NONOUTER_INDEX_CLASSIFICATION = "adaptive_v3r1_v8r4_nonouter_training_index"
PREDICTION_INDEX_CLASSIFICATION = (
    "adaptive_v3r1_v8r4_target_free_prediction_shard_index"
)
MODEL_BOUND_PREDICTION_INDEX_CLASSIFICATION = (
    "adaptive_v3r1_v8r4a_model_bound_target_free_prediction_shard_index"
)
MODEL_BOUND_PREDICTION_PACK_CLASSIFICATION = (
    "adaptive_v3r1_v8r4a_model_bound_target_free_prediction_pack"
)
MODEL_SOURCE_CAPABILITY_CLASSIFICATION = (
    "adaptive_v3r1_v8r4a_promotion_model_source_capability"
)
MODEL_SOURCE_SHARD_SEAL_CLASSIFICATION = (
    "adaptive_v3r1_v8r4a_model_source_shard_seal"
)
PROMOTION_TRAINING_AGGREGATOR_CLASSIFICATION = (
    "adaptive_v3r1_v8r4_promotion_training_shard_aggregator"
)
PROMOTION_TRAINING_SCOPE = "promotion_training_pack"
PREDICTION_SCOPE = "outer_prediction_pack"
PROMOTION_TRAINING_FOLDS = (0, 1, 2, 5)
PREDICTION_INDEX_FILENAME = "V8R4_TARGET_FREE_PREDICTION_INDEX.json"
MODEL_BOUND_PREDICTION_INDEX_FILENAME = (
    "V8R4A_MODEL_BOUND_TARGET_FREE_PREDICTION_INDEX.json"
)
MODEL_SOURCE_CAPABILITY_FILENAME = "MODEL_SOURCE_CAPABILITY.json"
MODEL_BOUND_PREDICTION_MANIFEST_FILENAME = (
    "MODEL_BOUND_OUTER_PREDICTION_PACK_MANIFEST.json"
)
MODEL_SOURCE_SHARD_SEAL_FILENAME = "MODEL_SOURCE_SHARD_SEAL.json"
PROMOTION_TRAINING_AGGREGATOR_FILENAME = (
    "V8R4_PROMOTION_TRAINING_SHARD_AGGREGATOR.json"
)

CACHE_FILES: dict[str, str] = {
    "feature_names": "feature_names.json",
    "metadata": "metadata.csv",
    "node_features": "node_features.npy",
    "candidate_bpm": "candidate_bpm.npy",
    "candidate_mask": "candidate_mask.npy",
    "joint_radar_mask": "joint_radar_mask.npy",
}
ARRAY_FILES = (
    "node_features.npy",
    "candidate_bpm.npy",
    "candidate_mask.npy",
    "joint_radar_mask.npy",
)
DISCOVERY_STACK_FIELDS = (
    "classification",
    "campaign_revision",
    "partition",
    "cache_index",
    "fold",
    "proposal_available",
    "nested_role",
    "prediction",
    "rr_std",
    "outer_fold",
    "seed",
    "outer_test_opened",
    "outer_rows_present",
)
OUTER_PREDICT_FIELDS = (
    "cache_index",
    "node_features",
    "candidate_rr_bpm",
    "candidate_mask",
    "joint_radar_mask",
    "proposer_anchor_bpm",
    "proposer_anchor_std_bpm",
    "proposer_anchor_available",
    "classical_rr_bpm",
    "session_reset",
)
OUTER_TOPOLOGY_COLUMNS = ("cache_index", "fold")
OUTER_ALLOWED_METADATA_COLUMNS = (
    "cache_index",
    "fold",
    "window_number",
    "classical_rr_bpm",
)
OUTER_FORBIDDEN_TOKENS = (
    "reference",
    "identity",
    "protocol",
    "quality",
    "fold",
)
OUTER_FORBIDDEN_EXACT_FIELDS = ("rr_bpm", "reference_valid")
REQUIRED_DISCOVERY_METADATA_COLUMNS = (
    "cache_index",
    "fold",
    "session_id",
    "identity",
    "window_number",
    "rr_bpm",
    "reference_valid",
    "classical_rr_bpm",
)
NONOUTER_METADATA_COLUMNS = REQUIRED_DISCOVERY_METADATA_COLUMNS
NPY_COPY_CHUNK_ROWS = 128
FIXED_ZIP_TIMESTAMP = (1980, 1, 1, 0, 0, 0)
# Test/audit hook.  Production leaves this ``None``; it never receives bytes,
# only the safe member name and the row index immediately before conversion.
NPZ_ROW_CONVERSION_HOOK: Callable[[str, int], None] | None = None


class PackError(RuntimeError):
    """A pack cannot be built without violating the V8R4 boundary."""


@dataclass(frozen=True)
class PromotionAuthorization:
    binding: FileBinding
    scopes: frozenset[str]


def validate_promotion_authorization(
    path: Path,
    *,
    expected_sha256: str,
    expected_bytes: int,
    required_scope: str,
) -> PromotionAuthorization:
    """Validate an immutable, exact-bound V8R4 promotion capability.

    There is intentionally no CLI bypass.  Synthetic tests call lower-level
    builders directly; any real promotion/prediction invocation must present a
    create-once 0444 authorization with an exact hash and byte count.
    """

    if not _is_sha256(expected_sha256) or type(expected_bytes) is not int or expected_bytes <= 0:
        raise PackError("promotion authorization requires exact SHA-256 and bytes")
    with StableFile(
        path,
        expected_sha256=expected_sha256,
        expected_bytes=expected_bytes,
    ) as source:
        info = os.fstat(source.fd)
        if stat.S_IMODE(info.st_mode) != 0o444:
            raise PackError("promotion authorization must be immutable mode 0444")
        document = _strict_json(source.read_bytes(), str(source.path))
        binding = source.binding
    if document.get("content_sha256") != canonical_content_sha256(document):
        raise PackError("promotion authorization content hash drifted")
    scopes = document.get("authorized_scopes")
    if not (
        document.get("classification")
        == "adaptive_v3r1_v8r4_promotion_authorization"
        and document.get("campaign_id") == CAMPAIGN_ID
        and document.get("campaign_revision") == PACK_REVISION
        and document.get("authorized_now") is True
        and isinstance(scopes, list)
        and all(isinstance(value, str) for value in scopes)
        and required_scope in scopes
    ):
        raise PackError(f"promotion authorization does not grant {required_scope}")
    return PromotionAuthorization(binding, frozenset(scopes))


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def semantic_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def canonical_content_sha256(value: Mapping[str, Any]) -> str:
    document = dict(value)
    document.pop("content_sha256", None)
    return semantic_sha256(document)


def _pretty_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _absolute_without_resolve(path: Path, *, base: Path | None = None) -> Path:
    candidate = path.expanduser()
    if not candidate.is_absolute():
        candidate = (base if base is not None else Path.cwd()) / candidate
    return Path(os.path.abspath(os.fspath(candidate)))


def _reject_symlink_components(path: Path, *, include_final: bool = True) -> None:
    """Reject symlinks in every existing component without dereferencing them."""

    absolute = _absolute_without_resolve(path)
    parts = absolute.parts
    current = Path(parts[0])
    limit = len(parts) if include_final else len(parts) - 1
    for part in parts[1:limit]:
        current /= part
        try:
            mode = os.lstat(current).st_mode
        except FileNotFoundError:
            continue
        except OSError as error:
            raise PackError(f"cannot inspect path component {current}: {error}") from error
        if stat.S_ISLNK(mode):
            raise PackError(f"symlinked path component is forbidden: {current}")


def _ensure_output_directory(path: Path) -> Path:
    absolute = _absolute_without_resolve(path)
    _reject_symlink_components(absolute, include_final=False)
    current = Path(absolute.parts[0])
    for part in absolute.parts[1:]:
        current /= part
        try:
            info = os.lstat(current)
        except FileNotFoundError:
            try:
                os.mkdir(current, 0o755)
            except FileExistsError:
                pass
            info = os.lstat(current)
        if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
            raise PackError(f"output directory component is unsafe: {current}")
    return absolute


def _strict_json(raw: bytes, label: str) -> dict[str, Any]:
    def unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise PackError(f"duplicate JSON key in {label}: {key}")
            result[key] = value
        return result

    try:
        value = json.loads(raw.decode("utf-8"), object_pairs_hook=unique)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise PackError(f"invalid JSON in {label}: {error}") from error
    if not isinstance(value, dict):
        raise PackError(f"{label} must be a JSON object")
    return value


@dataclass(frozen=True)
class FileBinding:
    path: str
    sha256: str
    bytes: int

    def as_dict(self) -> dict[str, Any]:
        return {"path": self.path, "sha256": self.sha256, "bytes": self.bytes}


class StableFile:
    """Single-link, no-follow descriptor with exact hash and inode stability."""

    def __init__(
        self,
        path: Path,
        *,
        expected_sha256: str | None = None,
        expected_bytes: int | None = None,
    ) -> None:
        self.path = _absolute_without_resolve(path)
        _reject_symlink_components(self.path)
        flags = os.O_RDONLY
        if hasattr(os, "O_CLOEXEC"):
            flags |= os.O_CLOEXEC
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        try:
            self.fd = os.open(self.path, flags)
        except OSError as error:
            raise PackError(f"cannot securely open {self.path}: {error}") from error
        try:
            info = os.fstat(self.fd)
            if not stat.S_ISREG(info.st_mode):
                raise PackError(f"input is not a regular file: {self.path}")
            if info.st_nlink != 1:
                raise PackError(f"input must have exactly one hard link: {self.path}")
            self._identity = (
                info.st_dev,
                info.st_ino,
                info.st_size,
                info.st_mtime_ns,
                info.st_ctime_ns,
            )
            digest = hashlib.sha256()
            offset = 0
            while offset < info.st_size:
                block = os.pread(self.fd, min(1024 * 1024, info.st_size - offset), offset)
                if not block:
                    raise PackError(f"short read while hashing {self.path}")
                digest.update(block)
                offset += len(block)
            actual_sha256 = digest.hexdigest()
            if expected_sha256 is not None and actual_sha256 != expected_sha256:
                raise PackError(f"SHA-256 drifted: {self.path}")
            if expected_bytes is not None and info.st_size != expected_bytes:
                raise PackError(f"byte count drifted: {self.path}")
            self.binding = FileBinding(str(self.path), actual_sha256, int(info.st_size))
            self.assert_stable()
        except Exception:
            os.close(self.fd)
            raise

    def assert_stable(self) -> None:
        try:
            info = os.fstat(self.fd)
        except OSError as error:
            raise PackError(f"input descriptor became invalid: {self.path}") from error
        current = (
            info.st_dev,
            info.st_ino,
            info.st_size,
            info.st_mtime_ns,
            info.st_ctime_ns,
        )
        if current != self._identity or info.st_nlink != 1 or not stat.S_ISREG(info.st_mode):
            raise PackError(f"input inode changed while building: {self.path}")

    def read_bytes(self) -> bytes:
        result = bytearray()
        offset = 0
        while offset < self.binding.bytes:
            block = os.pread(self.fd, min(1024 * 1024, self.binding.bytes - offset), offset)
            if not block:
                raise PackError(f"short read from {self.path}")
            result.extend(block)
            offset += len(block)
        self.assert_stable()
        return bytes(result)

    def duplicate_stream(self) -> io.BufferedReader:
        self.assert_stable()
        return os.fdopen(os.dup(self.fd), "rb", closefd=True)

    def close(self) -> None:
        if getattr(self, "fd", -1) >= 0:
            self.assert_stable()
            os.close(self.fd)
            self.fd = -1

    def __enter__(self) -> "StableFile":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()


def _revalidate_promotion_authorization(
    authorization: PromotionAuthorization,
    *,
    required_scope: str,
) -> PromotionAuthorization:
    """Re-pin an authorization immediately before capability-bearing output.

    ``PromotionAuthorization`` is deliberately a small, serializable value and
    does not retain an open descriptor.  A lower-level builder may be called
    directly after the path has been replaced, so each capability-bearing build
    reopens the exact bound bytes, rechecks mode 0444/single-link invariants, and
    revalidates the V8R4 scope before it creates an output directory.
    """

    if not isinstance(authorization, PromotionAuthorization):
        raise PackError("promotion build requires a validated authorization")
    if required_scope not in authorization.scopes:
        raise PackError(f"promotion authorization does not grant {required_scope}")
    rebound = validate_promotion_authorization(
        Path(authorization.binding.path),
        expected_sha256=authorization.binding.sha256,
        expected_bytes=authorization.binding.bytes,
        required_scope=required_scope,
    )
    if rebound != authorization:
        raise PackError("promotion authorization binding/scope drifted")
    return rebound


class IndexedNpy(StableFile):
    """A verified .npy mapped from the already-open no-follow descriptor."""

    def __init__(
        self,
        path: Path,
        *,
        expected_sha256: str,
        expected_bytes: int,
        label: str,
    ) -> None:
        self.label = label
        super().__init__(
            path,
            expected_sha256=expected_sha256,
            expected_bytes=expected_bytes,
        )
        stream = self.duplicate_stream()
        try:
            version = np.lib.format.read_magic(stream)
            if version == (1, 0):
                shape, fortran, dtype = np.lib.format.read_array_header_1_0(stream)
            elif version in {(2, 0), (3, 0)}:
                shape, fortran, dtype = np.lib.format._read_array_header(stream, version)
            else:
                raise PackError(f"unsupported npy format {version}: {self.path}")
            offset = stream.tell()
        finally:
            stream.close()
        self.dtype = np.dtype(dtype)
        self.shape = tuple(int(value) for value in shape)
        self.fortran_order = bool(fortran)
        if self.dtype.hasobject:
            raise PackError(f"object/pickle array is forbidden: {self.path}")
        if self.fortran_order:
            raise PackError(f"Fortran-order cache array is forbidden: {self.path}")
        expected_payload = int(np.prod(self.shape, dtype=np.int64)) * self.dtype.itemsize
        if offset + expected_payload != self.binding.bytes:
            raise PackError(f"npy payload size is inconsistent: {self.path}")
        try:
            self._mmap = mmap.mmap(self.fd, 0, access=mmap.ACCESS_READ)
        except OSError as error:
            raise PackError(f"cannot mmap {self.path}: {error}") from error
        self.array = np.ndarray(
            self.shape,
            dtype=self.dtype,
            buffer=self._mmap,
            offset=offset,
            order="C",
        )

    def take(self, positions: np.ndarray) -> np.ndarray:
        """Copy only explicitly supplied row positions in their given order."""

        index = np.asarray(positions)
        if index.dtype.kind not in "iu" or index.ndim != 1:
            raise PackError(f"indexed access positions are invalid for {self.label}")
        index = index.astype(np.int64, copy=False)
        if len(index) and (index[0] < 0 or index[-1] >= self.shape[0]):
            raise PackError(f"indexed access is out of range for {self.label}")
        if len(index) > 1 and np.any(index[1:] <= index[:-1]):
            raise PackError(f"indexed access must be strictly ordered for {self.label}")
        return np.ascontiguousarray(self.array[index])

    def close(self) -> None:
        if hasattr(self, "array"):
            del self.array
        if hasattr(self, "_mmap"):
            self._mmap.close()
        super().close()


class MetadataView(StableFile):
    """Byte-scanned CSV view that decodes only explicitly selected fields."""

    def __init__(
        self,
        path: Path,
        *,
        expected_sha256: str,
        expected_bytes: int,
    ) -> None:
        super().__init__(
            path,
            expected_sha256=expected_sha256,
            expected_bytes=expected_bytes,
        )
        self._mmap = mmap.mmap(self.fd, 0, access=mmap.ACCESS_READ)
        newline = self._mmap.find(b"\n")
        if newline < 0:
            raise PackError("cache metadata lacks a header line")
        self.header_start = 0
        self.header_stop = newline + 1
        header_values = self._decode_selected_row(0, newline, None)
        try:
            self.columns = tuple(value.decode("utf-8") for value in header_values)
        except UnicodeDecodeError as error:
            raise PackError("cache metadata header is not UTF-8") from error
        if len(set(self.columns)) != len(self.columns):
            raise PackError("cache metadata has duplicate columns")
        missing = sorted(set(REQUIRED_DISCOVERY_METADATA_COLUMNS) - set(self.columns))
        if missing:
            raise PackError(f"cache metadata lacks required columns: {missing}")
        if self.columns[:2] != OUTER_TOPOLOGY_COLUMNS:
            raise PackError("cache metadata must begin with cache_index,fold")
        self._column_index = {name: index for index, name in enumerate(self.columns)}
        self.row_spans: list[tuple[int, int]] = []
        cursor = self.header_stop
        while cursor < self.binding.bytes:
            line_end = self._mmap.find(b"\n", cursor)
            if line_end < 0:
                line_end = self.binding.bytes
                stop = line_end
            else:
                stop = line_end + 1
            if line_end > cursor and self._mmap[line_end - 1] == 13:
                logical_end = line_end - 1
            else:
                logical_end = line_end
            if logical_end > cursor:
                self.row_spans.append((cursor, stop))
            cursor = stop
        if not self.row_spans:
            raise PackError("cache metadata has no rows")

    def _decode_selected_row(
        self,
        start: int,
        logical_end: int,
        selected_indices: set[int] | None,
    ) -> list[bytes]:
        """Parse RFC4180 boundaries, copying bytes only for selected fields."""

        results: list[bytes] = []
        field = bytearray()
        field_index = 0
        quoted = False
        cursor = start
        while cursor <= logical_end:
            terminal = cursor == logical_end
            byte = None if terminal else self._mmap[cursor]
            wanted = selected_indices is None or field_index in selected_indices
            if terminal or (byte == 44 and not quoted):
                if wanted:
                    results.append(bytes(field))
                field.clear()
                field_index += 1
                cursor += 1
                continue
            if byte == 34:
                if quoted and cursor + 1 < logical_end and self._mmap[cursor + 1] == 34:
                    if wanted:
                        field.append(34)
                    cursor += 2
                    continue
                quoted = not quoted
                cursor += 1
                continue
            if wanted:
                field.append(int(byte))
            cursor += 1
        if quoted:
            raise PackError("unterminated quoted metadata field")
        return results

    def selected_values(self, row: int, columns: Sequence[str]) -> tuple[bytes, ...]:
        try:
            start, physical_stop = self.row_spans[row]
        except IndexError as error:
            raise PackError("metadata row is out of range") from error
        logical_end = physical_stop
        if logical_end > start and self._mmap[logical_end - 1] == 10:
            logical_end -= 1
        if logical_end > start and self._mmap[logical_end - 1] == 13:
            logical_end -= 1
        indices = [self._column_index[name] for name in columns]
        selected = self._decode_selected_row(start, logical_end, set(indices))
        # The scanner returns selected values in physical-column order.  Restore
        # caller order without decoding the skipped fields.
        physical = sorted(zip(indices, columns))
        by_name = {name: selected[offset] for offset, (_, name) in enumerate(physical)}
        return tuple(by_name[name] for name in columns)

    def topology(self) -> tuple[np.ndarray, np.ndarray]:
        index = np.empty(len(self.row_spans), dtype=np.int64)
        fold = np.empty(len(self.row_spans), dtype=np.int16)
        for row in range(len(self.row_spans)):
            raw_index, raw_fold = self.selected_values(row, OUTER_TOPOLOGY_COLUMNS)
            try:
                index[row] = int(raw_index.decode("ascii"))
                fold[row] = int(raw_fold.decode("ascii"))
            except (UnicodeDecodeError, ValueError, OverflowError) as error:
                raise PackError(f"invalid cache topology at row {row}") from error
        return index, fold

    def outer_context(self, positions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        window_number = np.empty(len(positions), dtype=np.int64)
        classical = np.empty(len(positions), dtype=np.float32)
        for local, row in enumerate(np.asarray(positions, dtype=np.int64)):
            raw_window, raw_classical = self.selected_values(
                int(row), ("window_number", "classical_rr_bpm")
            )
            try:
                window_number[local] = int(raw_window.decode("ascii"))
                classical[local] = float(raw_classical.decode("ascii"))
            except (UnicodeDecodeError, ValueError, OverflowError) as error:
                raise PackError(f"invalid target-free outer context at cache row {row}") from error
        if not np.isfinite(classical).all():
            raise PackError("outer classical_rr_bpm contains non-finite values")
        reset = window_number == 0
        if not len(reset) or not bool(reset[0]):
            raise PackError("outer filtered order does not begin at a session boundary")
        previous = -1
        for value in window_number:
            if value < 0 or (value != 0 and value != previous + 1):
                raise PackError("window_number cannot prove target-free session resets")
            previous = int(value)
        return classical, reset.astype(np.bool_, copy=False)

    def write_discovery_csv(self, stream: io.BufferedWriter, positions: np.ndarray) -> None:
        stream.write(b",".join(name.encode("ascii") for name in NONOUTER_METADATA_COLUMNS))
        stream.write(b"\n")

        def encode_csv_field(value: bytes) -> bytes:
            if any(token in value for token in (b",", b'"', b"\r", b"\n")):
                return b'"' + value.replace(b'"', b'""') + b'"'
            return value

        for row in np.asarray(positions, dtype=np.int64):
            # Only nonouter rows reach this method.  The eight consumer fields
            # are copied in fixed order; all other legacy columns disappear.
            values = self.selected_values(int(row), NONOUTER_METADATA_COLUMNS)
            stream.write(b",".join(encode_csv_field(value) for value in values))
            stream.write(b"\n")

    def close(self) -> None:
        if hasattr(self, "_mmap"):
            self._mmap.close()
        super().close()


class SelectiveNpz(StableFile):
    """Stream named NPY members by row; forbidden entries are never opened.

    ``numpy.load(...)[name]`` expands a complete compressed ZIP member before
    a caller can apply a row index.  That would materialize outer values even
    when the final slice is nonouter.  This class instead parses the embedded
    NPY header and walks its C-order first-axis records.  Bytes for unselected
    rows are read only as an opaque decompressor discard and are never handed
    to NumPy, decoded, converted, compared, or retained.
    """

    def __init__(
        self,
        path: Path,
        *,
        expected_sha256: str,
        expected_bytes: int,
    ) -> None:
        super().__init__(
            path,
            expected_sha256=expected_sha256,
            expected_bytes=expected_bytes,
        )
        self._stream = self.duplicate_stream()
        try:
            self._zip = zipfile.ZipFile(self._stream, "r")
        except (OSError, zipfile.BadZipFile) as error:
            self._stream.close()
            raise PackError(f"invalid proposer NPZ: {self.path}") from error
        names = self._zip.namelist()
        if len(names) != len(set(names)):
            raise PackError("proposer NPZ contains duplicate member names")
        self.files = frozenset(
            name[:-4] for name in names if name.endswith(".npy") and "/" not in name
        )

    @staticmethod
    def _read_exact(source: Any, size: int) -> bytes:
        chunks: list[bytes] = []
        remaining = size
        while remaining:
            block = source.read(remaining)
            if not block:
                raise PackError("truncated NPY member payload")
            chunks.append(block)
            remaining -= len(block)
        return b"".join(chunks)

    @staticmethod
    def _read_header(source: Any, name: str) -> tuple[tuple[int, ...], np.dtype[Any]]:
        try:
            version = np.lib.format.read_magic(source)
            if version == (1, 0):
                shape, fortran, dtype = np.lib.format.read_array_header_1_0(source)
            elif version in {(2, 0), (3, 0)}:
                shape, fortran, dtype = np.lib.format._read_array_header(source, version)
            else:
                raise PackError(f"unsupported proposer NPY version for {name}: {version}")
        except (EOFError, ValueError) as error:
            raise PackError(f"invalid proposer NPY header for {name}") from error
        dtype = np.dtype(dtype)
        shape = tuple(int(value) for value in shape)
        if dtype.hasobject:
            raise PackError(f"selected proposer field is object/pickle: {name}")
        if fortran:
            raise PackError(f"selected proposer field is not C-order: {name}")
        return shape, dtype

    def scalar(self, name: str) -> np.ndarray:
        member = f"{name}.npy"
        if name not in self.files:
            raise PackError(f"proposer NPZ lacks {name}")
        try:
            with self._zip.open(member, "r") as source:
                shape, dtype = self._read_header(source, name)
                if shape != ():
                    raise PackError(f"proposer field {name} is not scalar")
                payload = self._read_exact(source, dtype.itemsize)
                if source.read(1):
                    raise PackError(f"proposer field {name} has trailing payload")
                return np.frombuffer(payload, dtype=dtype, count=1).copy().reshape(())
        except (OSError, KeyError, EOFError) as error:
            raise PackError(f"cannot read safe proposer field {name}: {error}") from error

    def rows(
        self,
        name: str,
        positions: np.ndarray | None,
        *,
        expected_rows: int | None = None,
    ) -> np.ndarray:
        """Return selected C-order rows without converting any other row."""

        member = f"{name}.npy"
        if name not in self.files:
            raise PackError(f"proposer NPZ lacks {name}")
        try:
            with self._zip.open(member, "r") as source:
                shape, dtype = self._read_header(source, name)
                if not shape:
                    raise PackError(f"proposer field {name} is scalar, not row-aligned")
                rows = shape[0]
                if expected_rows is not None and rows != expected_rows:
                    raise PackError(f"proposer field {name} row count drifted")
                selected = (
                    np.arange(rows, dtype=np.int64)
                    if positions is None
                    else np.asarray(positions, dtype=np.int64)
                )
                if selected.ndim != 1:
                    raise PackError(f"proposer row selection is invalid: {name}")
                if len(selected) and (selected[0] < 0 or selected[-1] >= rows):
                    raise PackError(f"proposer row selection is out of range: {name}")
                if len(selected) > 1 and np.any(selected[1:] <= selected[:-1]):
                    raise PackError(f"proposer row selection is not strictly ordered: {name}")
                row_items = int(np.prod(shape[1:], dtype=np.int64))
                row_bytes = row_items * dtype.itemsize
                result = np.empty((len(selected), *shape[1:]), dtype=dtype)
                selected_offset = 0
                next_selected = int(selected[0]) if len(selected) else -1
                for row in range(rows):
                    raw = self._read_exact(source, row_bytes)
                    if row == next_selected:
                        # Conversion is deliberately inside the selected branch.
                        if NPZ_ROW_CONVERSION_HOOK is not None:
                            NPZ_ROW_CONVERSION_HOOK(name, row)
                        result[selected_offset] = np.frombuffer(
                            raw, dtype=dtype, count=row_items
                        ).reshape(shape[1:])
                        selected_offset += 1
                        next_selected = (
                            int(selected[selected_offset])
                            if selected_offset < len(selected)
                            else -1
                        )
                    # Unselected raw bytes fall out of scope here.
                if selected_offset != len(selected) or source.read(1):
                    raise PackError(f"proposer field {name} payload/topology drifted")
                return result
        except (OSError, KeyError, EOFError) as error:
            raise PackError(f"cannot stream safe proposer field {name}: {error}") from error

    def close(self) -> None:
        if hasattr(self, "_zip"):
            self._zip.close()
        if hasattr(self, "_stream"):
            self._stream.close()
        super().close()


def _array_sha256(values: np.ndarray, *, dtype: np.dtype[Any] = np.dtype("<i8")) -> str:
    array = np.ascontiguousarray(np.asarray(values, dtype=dtype))
    return hashlib.sha256(array.view(np.uint8)).hexdigest()


def _sha256_path_nofollow(path: Path) -> FileBinding:
    with StableFile(path) as source:
        return source.binding


_AT_EMPTY_PATH = 0x1000
_LIBC = ctypes.CDLL(None, use_errno=True)
_LINKAT = _LIBC.linkat
_LINKAT.argtypes = (ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_int)
_LINKAT.restype = ctypes.c_int


def _sha256_fd(descriptor: int) -> FileBinding:
    info = os.fstat(descriptor)
    if not stat.S_ISREG(info.st_mode):
        raise PackError("anonymous publication inode is not regular")
    digest = hashlib.sha256()
    offset = 0
    while True:
        payload = os.pread(descriptor, 1024 * 1024, offset)
        if not payload:
            break
        digest.update(payload)
        offset += len(payload)
    if offset != info.st_size:
        raise PackError("anonymous publication inode changed while hashing")
    return FileBinding(path="", sha256=digest.hexdigest(), bytes=offset)


def _verify_published_binding(
    destination: Path,
    *,
    expected_sha256: str,
    expected_bytes: int,
) -> FileBinding:
    try:
        with StableFile(
            destination,
            expected_sha256=expected_sha256,
            expected_bytes=expected_bytes,
        ) as published:
            info = os.fstat(published.fd)
            if not (
                stat.S_ISREG(info.st_mode)
                and stat.S_IMODE(info.st_mode) == 0o444
                and info.st_nlink == 1
            ):
                raise PackError(
                    f"published output is not regular mode 0444 nlink1: {destination}"
                )
            return published.binding
    except PackError:
        raise
    except OSError as error:
        raise PackError(f"cannot verify create-once output: {destination}") from error


def _publish_anonymous_create_once(descriptor: int, destination: Path) -> FileBinding:
    """Link one fully written anonymous inode exactly once.

    No named temporary pathname exists before the final ``linkat``.  Therefore
    SIGKILL can leave either no destination or a complete immutable destination,
    never a reusable partial file or an extra hard link.
    """

    destination = _absolute_without_resolve(destination)
    _reject_symlink_components(destination, include_final=False)
    parent = _ensure_output_directory(destination.parent)
    os.fsync(descriptor)
    os.fchmod(descriptor, 0o444)
    os.fsync(descriptor)
    before = os.fstat(descriptor)
    if not (
        stat.S_ISREG(before.st_mode)
        and stat.S_IMODE(before.st_mode) == 0o444
        and before.st_nlink == 0
    ):
        raise PackError("anonymous publication precondition is not 0444/nlink0")
    generated = _sha256_fd(descriptor)
    parent_flags = os.O_RDONLY | os.O_DIRECTORY
    if hasattr(os, "O_NOFOLLOW"):
        parent_flags |= os.O_NOFOLLOW
    parent_fd = os.open(parent, parent_flags)
    try:
        if _LINKAT(
            descriptor,
            b"",
            parent_fd,
            os.fsencode(destination.name),
            _AT_EMPTY_PATH,
        ) != 0:
            error_number = ctypes.get_errno()
            if error_number != errno.EEXIST:
                raise PackError(
                    f"anonymous create-once publication failed: {destination}: "
                    f"{os.strerror(error_number)}"
                )
            return _verify_published_binding(
                destination,
                expected_sha256=generated.sha256,
                expected_bytes=generated.bytes,
            )
        os.fsync(parent_fd)
    finally:
        os.close(parent_fd)
    return _verify_published_binding(
        destination,
        expected_sha256=generated.sha256,
        expected_bytes=generated.bytes,
    )


def _create_once_file(
    destination: Path,
    writer: Callable[[io.BufferedWriter], None],
) -> FileBinding:
    parent = _ensure_output_directory(destination.parent)
    if not hasattr(os, "O_TMPFILE"):
        raise PackError("O_TMPFILE is required for kill-safe publication")
    destination = _absolute_without_resolve(destination)
    destination_exists = False
    try:
        info = os.lstat(destination)
    except FileNotFoundError:
        pass
    except OSError as error:
        raise PackError(f"cannot inspect create-once output: {destination}") from error
    else:
        if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
            raise PackError(f"existing create-once output is unsafe: {destination}")
        destination_exists = True

    descriptor: int | None = None
    candidate = parent
    while descriptor is None:
        try:
            descriptor = os.open(candidate, os.O_RDWR | os.O_TMPFILE, 0o600)
        except OSError as error:
            if (
                destination_exists
                and error.errno in {errno.EACCES, errno.EPERM, errno.EROFS}
                and candidate.parent != candidate
            ):
                candidate = candidate.parent
                _reject_symlink_components(candidate)
                continue
            raise PackError(
                f"cannot create anonymous publication inode in {candidate}: {error}"
            ) from error
    try:
        with os.fdopen(os.dup(descriptor), "wb", closefd=True) as stream:
            writer(stream)
            stream.flush()
            os.fsync(stream.fileno())
        if destination_exists:
            os.fchmod(descriptor, 0o444)
            os.fsync(descriptor)
            generated = _sha256_fd(descriptor)
            return _verify_published_binding(
                destination,
                expected_sha256=generated.sha256,
                expected_bytes=generated.bytes,
            )
        return _publish_anonymous_create_once(descriptor, destination)
    finally:
        os.close(descriptor)


def _create_once_bytes(destination: Path, payload: bytes) -> FileBinding:
    return _create_once_file(destination, lambda stream: stream.write(payload))


def _freeze_directory_tree(root: Path) -> None:
    """Make a completed pack tree immutable without following any links.

    Files are already born 0444/nlink1.  Directories are frozen deepest first,
    leaving a SIGKILL either a writable resumable prefix or a fully 0555 tree.
    A retry recomputes and byte-validates existing files through anonymous
    inodes created in the nearest writable ancestor before completing the
    remaining directory transitions.
    """

    root = _absolute_without_resolve(root)
    _reject_symlink_components(root)
    directories: list[Path] = []
    pending = [root]
    while pending:
        directory = pending.pop()
        try:
            info = os.lstat(directory)
        except OSError as error:
            raise PackError(f"cannot inspect pack directory {directory}: {error}") from error
        if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
            raise PackError(f"pack directory is unsafe: {directory}")
        directories.append(directory)
        try:
            entries = list(os.scandir(directory))
        except OSError as error:
            raise PackError(f"cannot enumerate pack directory {directory}: {error}") from error
        for entry in entries:
            try:
                entry_info = entry.stat(follow_symlinks=False)
            except OSError as error:
                raise PackError(f"cannot inspect pack entry {entry.path}: {error}") from error
            entry_path = Path(entry.path)
            if stat.S_ISDIR(entry_info.st_mode):
                pending.append(entry_path)
            elif not (
                stat.S_ISREG(entry_info.st_mode)
                and stat.S_IMODE(entry_info.st_mode) == 0o444
                and entry_info.st_nlink == 1
            ):
                raise PackError(
                    f"pack file is not immutable regular 0444/nlink1: {entry_path}"
                )
    for directory in sorted(directories, key=lambda item: len(item.parts), reverse=True):
        flags = os.O_RDONLY | os.O_DIRECTORY
        if hasattr(os, "O_CLOEXEC"):
            flags |= os.O_CLOEXEC
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        try:
            descriptor = os.open(directory, flags)
        except OSError as error:
            raise PackError(f"cannot pin pack directory {directory}: {error}") from error
        try:
            before = os.fstat(descriptor)
            if not stat.S_ISDIR(before.st_mode):
                raise PackError(f"pack directory changed type: {directory}")
            if stat.S_IMODE(before.st_mode) != 0o555:
                os.fchmod(descriptor, 0o555)
            os.fsync(descriptor)
            after = os.fstat(descriptor)
            if (
                (after.st_dev, after.st_ino) != (before.st_dev, before.st_ino)
                or stat.S_IMODE(after.st_mode) != 0o555
            ):
                raise PackError(f"pack directory freeze drifted: {directory}")
        finally:
            os.close(descriptor)


def _copy_stable_file(source: StableFile, destination: Path) -> FileBinding:
    """Copy an already pinned source without reopening its pathname."""

    def write(stream: io.BufferedWriter) -> None:
        offset = 0
        while offset < source.binding.bytes:
            payload = os.pread(
                source.fd,
                min(1024 * 1024, source.binding.bytes - offset),
                offset,
            )
            if not payload:
                raise PackError(f"short read while copying {source.path}")
            stream.write(payload)
            offset += len(payload)
        source.assert_stable()

    binding = _create_once_file(destination, write)
    if (
        binding.sha256 != source.binding.sha256
        or binding.bytes != source.binding.bytes
    ):
        raise PackError(f"copied model artifact differs from source: {source.path}")
    return binding


def _npy_header(dtype: np.dtype[Any], shape: Sequence[int]) -> bytes:
    buffer = io.BytesIO()
    np.lib.format.write_array_header_2_0(
        buffer,
        {
            "descr": np.lib.format.dtype_to_descr(np.dtype(dtype)),
            "fortran_order": False,
            "shape": tuple(int(value) for value in shape),
        },
    )
    return buffer.getvalue()


def _write_subset_npy(
    destination: Path,
    source: IndexedNpy,
    positions: np.ndarray,
) -> FileBinding:
    positions = np.asarray(positions, dtype=np.int64)

    def writer(stream: io.BufferedWriter) -> None:
        stream.write(_npy_header(source.dtype, (len(positions), *source.shape[1:])))
        for start in range(0, len(positions), NPY_COPY_CHUNK_ROWS):
            chunk = source.take(positions[start : start + NPY_COPY_CHUNK_ROWS])
            stream.write(chunk.tobytes(order="C"))

    return _create_once_file(destination, writer)


def _write_array_npy(destination: Path, values: np.ndarray) -> FileBinding:
    array = np.ascontiguousarray(values)
    if array.dtype.hasobject:
        raise PackError(f"object array cannot be emitted: {destination}")

    def writer(stream: io.BufferedWriter) -> None:
        stream.write(_npy_header(array.dtype, array.shape))
        stream.write(array.tobytes(order="C"))

    return _create_once_file(destination, writer)


def _zip_info(name: str) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(f"{name}.npy", FIXED_ZIP_TIMESTAMP)
    info.compress_type = zipfile.ZIP_STORED
    info.create_system = 3
    info.external_attr = (stat.S_IFREG | 0o444) << 16
    info.flag_bits = 0
    return info


def _write_npz_entry(
    archive: zipfile.ZipFile,
    name: str,
    *,
    dtype: np.dtype[Any],
    shape: Sequence[int],
    chunks: Iterable[np.ndarray],
) -> None:
    dtype = np.dtype(dtype)
    if dtype.hasobject:
        raise PackError(f"object array cannot be emitted in NPZ: {name}")
    expected = int(np.prod(tuple(shape), dtype=np.int64)) * dtype.itemsize
    written = 0
    with archive.open(_zip_info(name), "w", force_zip64=True) as member:
        member.write(_npy_header(dtype, shape))
        for chunk in chunks:
            array = np.ascontiguousarray(chunk, dtype=dtype)
            payload = array.tobytes(order="C")
            member.write(payload)
            written += len(payload)
    if written != expected:
        raise PackError(f"NPZ field {name} payload length drifted")


def _single_chunk(values: np.ndarray) -> Iterator[np.ndarray]:
    yield np.ascontiguousarray(values)


def _indexed_chunks(source: IndexedNpy, positions: np.ndarray) -> Iterator[np.ndarray]:
    for start in range(0, len(positions), NPY_COPY_CHUNK_ROWS):
        yield source.take(positions[start : start + NPY_COPY_CHUNK_ROWS])


def _create_once_npz(
    destination: Path,
    entries: Sequence[
        tuple[str, np.dtype[Any], Sequence[int], Callable[[], Iterable[np.ndarray]]]
    ],
) -> FileBinding:
    if tuple(name for name, *_ in entries) != tuple(dict.fromkeys(name for name, *_ in entries)):
        raise PackError("duplicate deterministic NPZ field")

    def writer(stream: io.BufferedWriter) -> None:
        with zipfile.ZipFile(stream, "w", allowZip64=True) as archive:
            for name, dtype, shape, chunks in entries:
                _write_npz_entry(
                    archive,
                    name,
                    dtype=np.dtype(dtype),
                    shape=shape,
                    chunks=chunks(),
                )

    return _create_once_file(destination, writer)


def _binding_from_unit(unit: Mapping[str, Any], logical: str) -> Mapping[str, Any]:
    artifacts = unit.get("artifacts")
    aliases = {
        "cache_manifest": ("cache_manifest",),
        "proposer_stack": ("strict_stack", "proposer_stack"),
    }
    if isinstance(artifacts, Mapping):
        for alias in aliases[logical]:
            candidate = artifacts.get(alias)
            if isinstance(candidate, Mapping):
                return candidate
    candidate = unit.get(logical)
    if isinstance(candidate, Mapping):
        return candidate
    if logical == "cache_manifest":
        root = unit.get("cache_root")
        if isinstance(root, Mapping) and root.get("path") and root.get("manifest_sha256"):
            return {
                "path": str(Path(str(root["path"])) / "manifest.json"),
                "sha256": root["manifest_sha256"],
            }
    raise PackError(f"training unit lacks {logical} binding")


def _binding_path(
    binding: Mapping[str, Any], *, project_root: Path, owner: Path
) -> tuple[Path, str, int | None]:
    raw_path = binding.get("path")
    sha256 = binding.get("sha256", binding.get("file_sha256"))
    bytes_value = binding.get("bytes", binding.get("size_bytes"))
    if not isinstance(raw_path, str) or not _is_sha256(sha256):
        raise PackError("legacy input binding is incomplete")
    candidate = Path(raw_path)
    if not candidate.is_absolute():
        project_candidate = _absolute_without_resolve(candidate, base=project_root)
        owner_candidate = _absolute_without_resolve(candidate, base=owner.parent)
        if project_candidate.exists() and owner_candidate.exists() and project_candidate != owner_candidate:
            raise PackError(f"ambiguous relative legacy binding: {raw_path}")
        candidate = project_candidate if project_candidate.exists() else owner_candidate
    else:
        candidate = _absolute_without_resolve(candidate)
    if bytes_value is not None and (type(bytes_value) is not int or bytes_value < 0):
        raise PackError("legacy input byte binding is invalid")
    return candidate, str(sha256), bytes_value


def _manifest_output_binding(
    manifest: Mapping[str, Any], logical: str, expected_filename: str
) -> tuple[str, int]:
    outputs = manifest.get("outputs")
    if not isinstance(outputs, Mapping):
        raise PackError("legacy cache manifest has no outputs")
    record = outputs.get(logical)
    if not isinstance(record, Mapping) or set(record) != {"filename", "sha256", "bytes"}:
        raise PackError(f"legacy cache manifest binding schema drifted: {logical}")
    if record.get("filename") != expected_filename:
        raise PackError(f"legacy cache output filename drifted: {logical}")
    sha256 = record.get("sha256")
    byte_count = record.get("bytes")
    if not _is_sha256(sha256) or type(byte_count) is not int or byte_count < 0:
        raise PackError(f"legacy cache output binding invalid: {logical}")
    return str(sha256), int(byte_count)


@dataclass(frozen=True)
class UnitSource:
    outer_fold: int
    seed: int
    cache_manifest_path: Path
    cache_manifest_sha256: str
    cache_manifest_bytes: int | None
    proposer_path: Path
    proposer_sha256: str
    proposer_bytes: int


def load_training_index(
    project_root: Path,
    index_path: Path,
    *,
    expected_sha256: str | None,
    expected_bytes: int | None,
    require_exact_matrix: bool,
) -> tuple[list[UnitSource], FileBinding]:
    with StableFile(
        index_path,
        expected_sha256=expected_sha256,
        expected_bytes=expected_bytes,
    ) as source:
        document = _strict_json(source.read_bytes(), str(source.path))
        binding = source.binding
    if document.get("content_sha256") is not None and (
        document.get("content_sha256") != canonical_content_sha256(document)
    ):
        raise PackError("training index canonical content hash drifted")
    if not (
        document.get("status") == "complete"
        and document.get("outer_test_opened") is False
    ):
        raise PackError("training index is incomplete or test-opened")
    units = document.get("units")
    if not isinstance(units, list) or not units:
        raise PackError("training index has no units")
    owner = _absolute_without_resolve(index_path)
    result: list[UnitSource] = []
    observed: set[tuple[int, int]] = set()
    for unit in units:
        if not isinstance(unit, Mapping):
            raise PackError("training index unit is not an object")
        outer_fold = unit.get("outer_fold")
        seed = unit.get("seed")
        if type(outer_fold) is not int or outer_fold not in range(N_FOLDS):
            raise PackError("training index unit outer_fold is invalid")
        if type(seed) is not int:
            raise PackError("training index unit seed is invalid")
        key = (outer_fold, seed)
        if key in observed:
            raise PackError(f"duplicate training index unit: {key}")
        observed.add(key)
        cache = _binding_from_unit(unit, "cache_manifest")
        proposer = _binding_from_unit(unit, "proposer_stack")
        cache_path, cache_sha, cache_bytes = _binding_path(
            cache, project_root=project_root, owner=owner
        )
        proposer_path, proposer_sha, proposer_bytes = _binding_path(
            proposer, project_root=project_root, owner=owner
        )
        if proposer_bytes is None:
            raise PackError("proposer stack requires an exact byte binding")
        result.append(
            UnitSource(
                outer_fold,
                seed,
                cache_path,
                cache_sha,
                cache_bytes,
                proposer_path,
                proposer_sha,
                proposer_bytes,
            )
        )
    expected = {(fold, seed) for fold in range(N_FOLDS) for seed in SEEDS}
    if require_exact_matrix and observed != expected:
        raise PackError("training index is not the exact 6x3 outer-fold/seed cover")
    result.sort(key=lambda unit: (unit.outer_fold, unit.seed))
    return result, binding


def _pack_output_record(binding: FileBinding, filename: str) -> dict[str, Any]:
    return {"filename": filename, "sha256": binding.sha256, "bytes": binding.bytes}


def _relative_binding(binding: FileBinding, owner_root: Path) -> dict[str, Any]:
    try:
        relative = Path(binding.path).relative_to(owner_root)
    except ValueError as error:
        raise PackError(f"output binding escapes its owner root: {binding.path}") from error
    return {
        "path": relative.as_posix(),
        "sha256": binding.sha256,
        "bytes": binding.bytes,
    }


def _shape_record(binding: FileBinding, array: np.ndarray) -> dict[str, Any]:
    return {
        "filename": Path(binding.path).name,
        "sha256": binding.sha256,
        "bytes": binding.bytes,
        "dtype": np.dtype(array.dtype).str,
        "shape": list(array.shape),
    }


def build_unit_pack(
    source: UnitSource,
    *,
    index_binding: FileBinding,
    output_root: Path,
    promotion_authorization: PromotionAuthorization | None = None,
) -> dict[str, Any]:
    if promotion_authorization is not None:
        if source.outer_fold not in PROMOTION_TRAINING_FOLDS:
            raise PackError("promotion-training unit outer fold is not authorized")
        promotion_authorization = _revalidate_promotion_authorization(
            promotion_authorization,
            required_scope=PROMOTION_TRAINING_SCOPE,
        )
    unit_name = f"outer_{source.outer_fold}_seed_{source.seed}"
    unit_root = _ensure_output_directory(output_root / "units" / unit_name)
    discovery_cache = _ensure_output_directory(unit_root / "discovery_cache")

    with ExitStack() as stack:
        manifest_file = stack.enter_context(
            StableFile(
                source.cache_manifest_path,
                expected_sha256=source.cache_manifest_sha256,
                expected_bytes=source.cache_manifest_bytes,
            )
        )
        legacy_manifest = _strict_json(
            manifest_file.read_bytes(), str(source.cache_manifest_path)
        )
        if not (
            legacy_manifest.get("complete") is True
            and legacy_manifest.get("format_version") == 1
            and legacy_manifest.get("content_sha256")
            == canonical_content_sha256(legacy_manifest)
        ):
            raise PackError("legacy cache manifest is incomplete or content-hash drifted")
        cache_root = source.cache_manifest_path.parent
        cache_handles: dict[str, StableFile] = {}
        for logical, filename in CACHE_FILES.items():
            expected_sha, expected_size = _manifest_output_binding(
                legacy_manifest, logical, filename
            )
            if filename.endswith(".npy"):
                handle = IndexedNpy(
                    cache_root / filename,
                    expected_sha256=expected_sha,
                    expected_bytes=expected_size,
                    label=filename,
                )
            elif filename == "metadata.csv":
                handle = MetadataView(
                    cache_root / filename,
                    expected_sha256=expected_sha,
                    expected_bytes=expected_size,
                )
            else:
                handle = StableFile(
                    cache_root / filename,
                    expected_sha256=expected_sha,
                    expected_bytes=expected_size,
                )
            cache_handles[filename] = stack.enter_context(handle)
        metadata = cache_handles["metadata.csv"]
        assert isinstance(metadata, MetadataView)
        global_index, fold = metadata.topology()
        row_count = len(global_index)
        if not np.array_equal(global_index, np.arange(row_count, dtype=np.int64)):
            raise PackError("legacy cache_index must be a contiguous ordered exact cover")
        if np.any((fold < 0) | (fold >= N_FOLDS)):
            raise PackError("legacy fold topology is outside [0,5]")
        discovery_positions = np.flatnonzero(fold != source.outer_fold).astype(np.int64)
        outer_positions = np.flatnonzero(fold == source.outer_fold).astype(np.int64)
        if not len(discovery_positions) or not len(outer_positions):
            raise PackError("legacy cache has an empty V8R4 partition")
        if (
            len(discovery_positions) + len(outer_positions) != row_count
            or np.intersect1d(discovery_positions, outer_positions).size
            or not np.array_equal(
                np.sort(np.concatenate((discovery_positions, outer_positions))),
                np.arange(row_count, dtype=np.int64),
            )
        ):
            raise PackError("V8R4 partition is not an exact disjoint complement")

        arrays: dict[str, IndexedNpy] = {}
        for filename in ARRAY_FILES:
            handle = cache_handles[filename]
            assert isinstance(handle, IndexedNpy)
            if not handle.shape or handle.shape[0] != row_count:
                raise PackError(f"legacy cache row count drifted: {filename}")
            arrays[filename] = handle
        candidate_shape = arrays["candidate_bpm.npy"].shape
        if (
            arrays["candidate_bpm.npy"].dtype != np.dtype(np.float32)
            or len(candidate_shape) != 2
            or not 1 <= candidate_shape[1] <= 12
            or arrays["node_features.npy"].shape != (*candidate_shape, 571)
            or arrays["node_features.npy"].dtype != np.dtype(np.float32)
            or arrays["candidate_mask.npy"].shape != candidate_shape
            or arrays["candidate_mask.npy"].dtype != np.dtype(np.bool_)
            or arrays["joint_radar_mask.npy"].shape != (row_count, 3)
            or arrays["joint_radar_mask.npy"].dtype != np.dtype(np.bool_)
        ):
            raise PackError("legacy model-array shape/dtype contract drifted")

        proposer = stack.enter_context(
            SelectiveNpz(
                source.proposer_path,
                expected_sha256=source.proposer_sha256,
                expected_bytes=source.proposer_bytes,
            )
        )
        # Stage one: only target-free topology is decoded before any anchor.
        proposer_index = np.asarray(proposer.rows("cache_index", None), dtype=np.int64)
        proposer_fold = np.asarray(
            proposer.rows("fold", None, expected_rows=row_count), dtype=np.int16
        )
        if not (
            proposer_index.shape == proposer_fold.shape == (row_count,)
            and np.array_equal(proposer_index, global_index)
            and np.array_equal(proposer_fold, fold)
        ):
            raise PackError("legacy cache/proposer ordered row topology drifted")
        proposer_outer_fold = np.asarray(proposer.scalar("outer_fold"))
        proposer_seed = np.asarray(proposer.scalar("seed"))
        proposer_opened = np.asarray(proposer.scalar("outer_test_opened"))
        strict_nested = np.asarray(proposer.scalar("strict_nested"))
        if (
            proposer_outer_fold.shape != ()
            or int(proposer_outer_fold.item()) != source.outer_fold
            or proposer_seed.shape != ()
            or int(proposer_seed.item()) != source.seed
            or proposer_opened.shape != ()
            or bool(proposer_opened.item())
            or strict_nested.shape != ()
            or not bool(strict_nested.item())
        ):
            raise PackError("legacy proposer scalar identity drifted")
        # Stage two: stream only nonouter model-required proposer rows.  Outer
        # records in these compressed members are opaque decompressor discard;
        # they are never converted into NumPy values.  Legacy identity,
        # protocol, reference, quality, and nested-role members are unopened.
        available = np.asarray(
            proposer.rows(
                "proposal_available", discovery_positions, expected_rows=row_count
            )
        )
        prediction = np.asarray(
            proposer.rows("prediction", discovery_positions, expected_rows=row_count),
            dtype=np.float32,
        )
        rr_std = np.asarray(
            proposer.rows("rr_std", discovery_positions, expected_rows=row_count),
            dtype=np.float32,
        )
        if not (
            available.dtype == np.bool_
            and available.shape == prediction.shape == rr_std.shape
            == (len(discovery_positions),)
        ):
            raise PackError("legacy nonouter proposer anchor topology drifted")
        usable = available & np.isfinite(prediction) & np.isfinite(rr_std)
        usable &= (prediction >= 6.0) & (prediction <= 45.0) & (rr_std > 0.0)
        if np.any(available & ~usable):
            raise PackError("legacy available proposer anchor is invalid")
        if not available.all():
            raise PackError("legacy proposer is incomplete on discovery rows")

        local_global_path = discovery_cache / "local_to_global_cache_index.npy"
        local_global_binding = _write_array_npy(
            local_global_path, global_index[discovery_positions].astype(np.int64, copy=False)
        )
        feature_names = cache_handles["feature_names.json"]
        feature_binding = _create_once_bytes(
            discovery_cache / "feature_names.json", feature_names.read_bytes()
        )
        metadata_binding = _create_once_file(
            discovery_cache / "metadata.csv",
            lambda stream: metadata.write_discovery_csv(stream, discovery_positions),
        )
        discovery_array_bindings: dict[str, FileBinding] = {}
        for filename in ARRAY_FILES:
            discovery_array_bindings[filename] = _write_subset_npy(
                discovery_cache / filename, arrays[filename], discovery_positions
            )

        nonouter_fold = fold[discovery_positions].astype(np.int16, copy=False)
        nested_role = np.where(
            nonouter_fold == ((source.outer_fold + 1) % N_FOLDS),
            "validation",
            "training",
        )
        discovery_stack_values: dict[str, np.ndarray] = {
            "classification": np.asarray(NONOUTER_STACK_CLASSIFICATION),
            "campaign_revision": np.asarray(PACK_REVISION),
            "partition": np.asarray("outer_excluded_training_validation"),
            "cache_index": global_index[discovery_positions].astype(np.int64, copy=False),
            "fold": nonouter_fold,
            "proposal_available": available.astype(np.bool_, copy=False),
            "nested_role": nested_role,
            "prediction": prediction.astype(np.float32, copy=False),
            "rr_std": rr_std.astype(np.float32, copy=False),
            "outer_fold": np.asarray(source.outer_fold, dtype=np.int16),
            "seed": np.asarray(source.seed, dtype=np.int64),
            "outer_test_opened": np.asarray(False, dtype=np.bool_),
            "outer_rows_present": np.asarray(False, dtype=np.bool_),
        }
        proposer_entries = tuple(
            (
                name,
                values.dtype,
                values.shape,
                (lambda values=values: _single_chunk(values)),
            )
            for name in DISCOVERY_STACK_FIELDS
            for values in (discovery_stack_values[name],)
        )
        discovery_proposer_binding = _create_once_npz(
            unit_root / "discovery_proposer_stack.npz", proposer_entries
        )

        cache_outputs = {
            "feature_names": _pack_output_record(feature_binding, "feature_names.json"),
            "metadata": _pack_output_record(metadata_binding, "metadata.csv"),
            "node_features": _pack_output_record(
                discovery_array_bindings["node_features.npy"], "node_features.npy"
            ),
            "candidate_bpm": _pack_output_record(
                discovery_array_bindings["candidate_bpm.npy"], "candidate_bpm.npy"
            ),
            "candidate_mask": _pack_output_record(
                discovery_array_bindings["candidate_mask.npy"], "candidate_mask.npy"
            ),
            "joint_radar_mask": _pack_output_record(
                discovery_array_bindings["joint_radar_mask.npy"], "joint_radar_mask.npy"
            ),
            "local_to_global_cache_index": _pack_output_record(
                local_global_binding, "local_to_global_cache_index.npy"
            ),
        }
        cache_manifest: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "classification": NONOUTER_PACK_CLASSIFICATION,
            "campaign_id": CAMPAIGN_ID,
            "campaign_revision": PACK_REVISION,
            "format_version": 1,
            "complete": True,
            "outer_fold": source.outer_fold,
            "partition": "outer_excluded_training_validation",
            "source_combined_cache_open_authorized_by_consumer": False,
            "outer_test_rows_physically_present": False,
            "outer_prediction_pack_absent": True,
            "inputs": {
                "source_combined_cache": {
                    "sha256": manifest_file.binding.sha256,
                    "bytes": manifest_file.binding.bytes,
                },
                "proposer_stack": {
                    "sha256": discovery_proposer_binding.sha256,
                    "bytes": discovery_proposer_binding.bytes,
                },
            },
            "outputs": cache_outputs,
        }
        if promotion_authorization is not None:
            cache_manifest.update(
                {
                    "promotion_scope": PROMOTION_TRAINING_SCOPE,
                    "promotion_authorization": (
                        promotion_authorization.binding.as_dict()
                    ),
                }
            )
        cache_manifest["content_sha256"] = canonical_content_sha256(cache_manifest)
        discovery_manifest_binding = _create_once_bytes(
            discovery_cache / "manifest.json", _pretty_json_bytes(cache_manifest)
        )

        all_legacy_bindings = {
            "training_index": index_binding.as_dict(),
            "cache_manifest": manifest_file.binding.as_dict(),
            "proposer_stack": proposer.binding.as_dict(),
            "cache_outputs": {
                filename: cache_handles[filename].binding.as_dict()
                for filename in CACHE_FILES.values()
            },
        }
        partition_manifest: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "campaign_id": CAMPAIGN_ID,
            "campaign_revision": PACK_REVISION,
            "classification": "adaptive_v3r1_v8r4_sealed_nonouter_partition",
            "outer_fold": source.outer_fold,
            "seed": source.seed,
            "legacy_row_count": row_count,
            "partition": {
                "discovery_rows": len(discovery_positions),
                "discovery_outer_rows": 0,
                "legacy_outer_complement_rows": len(outer_positions),
                "outer_prediction_pack_rows": 0,
                "intersection_rows": 0,
                "union_rows": row_count,
                "exact_disjoint_complement": True,
                "global_index_sha256": _array_sha256(global_index),
                "discovery_global_index_sha256": _array_sha256(
                    global_index[discovery_positions]
                ),
                "outer_global_index_sha256": _array_sha256(
                    global_index[outer_positions]
                ),
                "fold_topology_sha256": _array_sha256(fold, dtype=np.dtype("<i2")),
            },
            "legacy_inputs": all_legacy_bindings,
            "outputs": {
                "discovery_cache_manifest": _relative_binding(
                    discovery_manifest_binding, unit_root
                ),
                "discovery_proposer_stack": _relative_binding(
                    discovery_proposer_binding, unit_root
                ),
                "discovery_local_to_global_map": _relative_binding(
                    local_global_binding, unit_root
                ),
            },
            "integration_interface": {
                "training_cache": "discovery_cache",
                "training_proposer_stack": "discovery_proposer_stack.npz",
                "trainer_outer_fold": source.outer_fold,
                "trainer_seed": source.seed,
                "cache_index_translation": (
                    "discovery_cache/local_to_global_cache_index.npy"
                ),
                "science_row_order_preserved": True,
                "model_feature_values_preserved": True,
                "discovery_reference_and_context_values_preserved": True,
            },
            "protected_outer_access": {
                "topology_columns_decoded_first": list(OUTER_TOPOLOGY_COLUMNS),
                "target_free_columns_decoded_after_partition": [],
                "emitted_fields": [],
                "exact_allowlist": True,
                "forbidden_fields_emitted": False,
                "outer_reference_decoded": False,
                "outer_reference_validity_decoded": False,
                "outer_identity_decoded": False,
                "outer_protocol_decoded": False,
                "outer_quality_decoded": False,
                "whole_legacy_metadata_hashed_as_opaque_binding": True,
            },
            "preselection_prediction_boundary": {
                "outer_prediction_pack_absent": True,
                "outer_prediction_path_bound": False,
                "outer_prediction_values_materialized": False,
                "promotion_authorization_required_before_prediction_pack": True,
            },
            "serialization": {
                "object_arrays": False,
                "pickle": False,
                "outputs_mode": "0444",
                "create_once_resume_requires_byte_equality": True,
                "zip_member_timestamp": list(FIXED_ZIP_TIMESTAMP),
            },
            "claim_boundary": {
                "adaptive_retrospective_only": True,
                "commercial_or_confirmatory_claim_allowed": False,
                "outer_targets_opened": False,
            },
        }
        if promotion_authorization is not None:
            partition_manifest.update(
                {
                    "promotion_scope": PROMOTION_TRAINING_SCOPE,
                    "promotion_authorization": (
                        promotion_authorization.binding.as_dict()
                    ),
                }
            )
        partition_manifest["content_sha256"] = canonical_content_sha256(
            partition_manifest
        )
        partition_binding = _create_once_bytes(
            unit_root / "PARTITION_MANIFEST.json",
            _pretty_json_bytes(partition_manifest),
        )
        for handle in cache_handles.values():
            handle.assert_stable()
        proposer.assert_stable()
        manifest_file.assert_stable()

    result = dict(partition_manifest)
    result["partition_manifest_binding"] = partition_binding.as_dict()
    return result


def build_authorized_prediction_pack(
    source: UnitSource,
    *,
    authorization: PromotionAuthorization,
    index_binding: FileBinding,
    output_root: Path,
) -> dict[str, Any]:
    """Build one target-free outer pack after an explicit promotion grant.

    One call owns one outer-fold/seed shard.  Callers must not place packs from
    different outer folds in a common runtime mount.  This function has no
    path to the discovery output and cannot create or amend a discovery index.
    """

    authorization = _revalidate_promotion_authorization(
        authorization,
        required_scope=PREDICTION_SCOPE,
    )
    output_root = _ensure_output_directory(output_root)
    unit_name = f"outer_{source.outer_fold}_seed_{source.seed}"
    unit_root = _ensure_output_directory(output_root / unit_name)
    with ExitStack() as stack:
        manifest_file = stack.enter_context(
            StableFile(
                source.cache_manifest_path,
                expected_sha256=source.cache_manifest_sha256,
                expected_bytes=source.cache_manifest_bytes,
            )
        )
        legacy_manifest = _strict_json(
            manifest_file.read_bytes(), str(source.cache_manifest_path)
        )
        if not (
            legacy_manifest.get("complete") is True
            and legacy_manifest.get("format_version") == 1
            and legacy_manifest.get("content_sha256")
            == canonical_content_sha256(legacy_manifest)
        ):
            raise PackError("legacy cache manifest is incomplete or content-hash drifted")
        cache_root = source.cache_manifest_path.parent
        expected_sha, expected_size = _manifest_output_binding(
            legacy_manifest, "metadata", "metadata.csv"
        )
        metadata = stack.enter_context(
            MetadataView(
                cache_root / "metadata.csv",
                expected_sha256=expected_sha,
                expected_bytes=expected_size,
            )
        )
        arrays: dict[str, IndexedNpy] = {}
        for logical, filename in (
            ("node_features", "node_features.npy"),
            ("candidate_bpm", "candidate_bpm.npy"),
            ("candidate_mask", "candidate_mask.npy"),
            ("joint_radar_mask", "joint_radar_mask.npy"),
        ):
            expected_sha, expected_size = _manifest_output_binding(
                legacy_manifest, logical, filename
            )
            arrays[filename] = stack.enter_context(
                IndexedNpy(
                    cache_root / filename,
                    expected_sha256=expected_sha,
                    expected_bytes=expected_size,
                    label=filename,
                )
            )
        global_index, fold = metadata.topology()
        row_count = len(global_index)
        if not np.array_equal(global_index, np.arange(row_count, dtype=np.int64)):
            raise PackError("legacy cache_index must be a contiguous ordered exact cover")
        outer_positions = np.flatnonzero(fold == source.outer_fold).astype(np.int64)
        nonouter_positions = np.flatnonzero(fold != source.outer_fold).astype(np.int64)
        if not len(outer_positions) or (
            len(outer_positions) + len(nonouter_positions) != row_count
            or np.intersect1d(outer_positions, nonouter_positions).size
        ):
            raise PackError("authorized prediction partition is not an exact complement")
        candidate_shape = arrays["candidate_bpm.npy"].shape
        if not (
            arrays["candidate_bpm.npy"].dtype == np.dtype(np.float32)
            and len(candidate_shape) == 2
            and arrays["node_features.npy"].shape == (*candidate_shape, 571)
            and arrays["node_features.npy"].dtype == np.dtype(np.float32)
            and arrays["candidate_mask.npy"].shape == candidate_shape
            and arrays["candidate_mask.npy"].dtype == np.dtype(np.bool_)
            and arrays["joint_radar_mask.npy"].shape == (row_count, 3)
            and arrays["joint_radar_mask.npy"].dtype == np.dtype(np.bool_)
        ):
            raise PackError("legacy prediction-array shape/dtype contract drifted")

        proposer = stack.enter_context(
            SelectiveNpz(
                source.proposer_path,
                expected_sha256=source.proposer_sha256,
                expected_bytes=source.proposer_bytes,
            )
        )
        proposer_index = np.asarray(proposer.rows("cache_index", None), dtype=np.int64)
        proposer_fold = np.asarray(
            proposer.rows("fold", None, expected_rows=row_count), dtype=np.int16
        )
        if not (
            np.array_equal(proposer_index, global_index)
            and np.array_equal(proposer_fold, fold)
            and int(proposer.scalar("outer_fold").item()) == source.outer_fold
            and int(proposer.scalar("seed").item()) == source.seed
            and not bool(proposer.scalar("outer_test_opened").item())
        ):
            raise PackError("legacy proposer prediction topology/identity drifted")
        # Only target-free anchor members and only authorized outer rows are
        # converted.  Nonouter bytes are opaque decompressor discard.
        outer_available = np.asarray(
            proposer.rows(
                "proposal_available", outer_positions, expected_rows=row_count
            )
        )
        outer_anchor = np.asarray(
            proposer.rows("prediction", outer_positions, expected_rows=row_count),
            dtype=np.float32,
        )
        outer_std = np.asarray(
            proposer.rows("rr_std", outer_positions, expected_rows=row_count),
            dtype=np.float32,
        )
        if not (
            outer_available.dtype == np.bool_
            and outer_available.shape == outer_anchor.shape == outer_std.shape
            == (len(outer_positions),)
        ):
            raise PackError("outer proposer anchor topology drifted")
        valid_anchor = (
            np.isfinite(outer_anchor)
            & np.isfinite(outer_std)
            & (outer_anchor >= 6.0)
            & (outer_anchor <= 45.0)
            & (outer_std > 0.0)
        )
        if np.any(outer_available & ~valid_anchor):
            raise PackError("available authorized outer proposer anchor is invalid")
        outer_anchor = outer_anchor.copy()
        outer_std = outer_std.copy()
        outer_anchor[~outer_available] = 0.0
        outer_std[~outer_available] = 1.0
        classical, session_reset = metadata.outer_context(outer_positions)

        entries = (
            (
                "cache_index",
                np.dtype(np.int64),
                (len(outer_positions),),
                lambda: _single_chunk(global_index[outer_positions]),
            ),
            (
                "node_features",
                arrays["node_features.npy"].dtype,
                (len(outer_positions), *arrays["node_features.npy"].shape[1:]),
                lambda: _indexed_chunks(arrays["node_features.npy"], outer_positions),
            ),
            (
                "candidate_rr_bpm",
                arrays["candidate_bpm.npy"].dtype,
                (len(outer_positions), *arrays["candidate_bpm.npy"].shape[1:]),
                lambda: _indexed_chunks(arrays["candidate_bpm.npy"], outer_positions),
            ),
            (
                "candidate_mask",
                arrays["candidate_mask.npy"].dtype,
                (len(outer_positions), *arrays["candidate_mask.npy"].shape[1:]),
                lambda: _indexed_chunks(arrays["candidate_mask.npy"], outer_positions),
            ),
            (
                "joint_radar_mask",
                arrays["joint_radar_mask.npy"].dtype,
                (len(outer_positions), *arrays["joint_radar_mask.npy"].shape[1:]),
                lambda: _indexed_chunks(arrays["joint_radar_mask.npy"], outer_positions),
            ),
            (
                "proposer_anchor_bpm",
                np.dtype(np.float32),
                outer_anchor.shape,
                lambda: _single_chunk(outer_anchor),
            ),
            (
                "proposer_anchor_std_bpm",
                np.dtype(np.float32),
                outer_std.shape,
                lambda: _single_chunk(outer_std),
            ),
            (
                "proposer_anchor_available",
                np.dtype(np.bool_),
                outer_available.shape,
                lambda: _single_chunk(outer_available),
            ),
            (
                "classical_rr_bpm",
                np.dtype(np.float32),
                classical.shape,
                lambda: _single_chunk(classical),
            ),
            (
                "session_reset",
                np.dtype(np.bool_),
                session_reset.shape,
                lambda: _single_chunk(session_reset),
            ),
        )
        if tuple(name for name, *_ in entries) != OUTER_PREDICT_FIELDS:
            raise PackError("outer prediction allowlist drifted")
        predict_binding = _create_once_npz(unit_root / "outer_predict_input.npz", entries)
        manifest: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "classification": "adaptive_v3r1_v8r4_authorized_outer_prediction_pack",
            "campaign_id": CAMPAIGN_ID,
            "campaign_revision": PACK_REVISION,
            "outer_fold": source.outer_fold,
            "seed": source.seed,
            "row_count": len(outer_positions),
            "fields": list(OUTER_PREDICT_FIELDS),
            "exact_allowlist": True,
            "forbidden_fields_emitted": False,
            "reference_identity_protocol_quality_decoded": False,
            "legacy_index": index_binding.as_dict(),
            "legacy_cache_manifest": manifest_file.binding.as_dict(),
            "legacy_proposer_stack": proposer.binding.as_dict(),
            "promotion_authorization": authorization.binding.as_dict(),
            "output": _relative_binding(predict_binding, unit_root),
            "global_cache_index_sha256": _array_sha256(
                global_index[outer_positions]
            ),
            "object_arrays": False,
            "pickle": False,
            "commercial_or_confirmatory_claim_allowed": False,
        }
        manifest["content_sha256"] = canonical_content_sha256(manifest)
        manifest_binding = _create_once_bytes(
            unit_root / "OUTER_PREDICTION_PACK_MANIFEST.json",
            _pretty_json_bytes(manifest),
        )
        metadata.assert_stable()
        proposer.assert_stable()
        for array in arrays.values():
            array.assert_stable()
    result = dict(manifest)
    result["manifest_binding"] = manifest_binding.as_dict()
    return result


def build_authorized_prediction_shard(
    sources: Sequence[UnitSource],
    *,
    authorization: PromotionAuthorization,
    index_binding: FileBinding,
    output_root: Path,
    outer_fold: int,
    selected_seed: int | None = None,
) -> dict[str, Any]:
    """Build one outer-specific target-free prediction shard.

    A seed filter is a kill-safe construction aid only.  It may create the one
    requested unit pack, but the final shard index is published only for the
    exact ordered three-seed cover.
    """

    authorization = _revalidate_promotion_authorization(
        authorization,
        required_scope=PREDICTION_SCOPE,
    )
    if type(outer_fold) is not int or outer_fold not in range(N_FOLDS):
        raise PackError("prediction shard outer fold is invalid")
    if selected_seed is not None and selected_seed not in SEEDS:
        raise PackError("prediction shard seed filter is invalid")
    observed: dict[int, UnitSource] = {}
    for source in sources:
        if source.outer_fold != outer_fold or source.seed not in SEEDS:
            raise PackError("prediction shard source identity drifted")
        if source.seed in observed:
            raise PackError("prediction shard contains a duplicate seed")
        observed[source.seed] = source
    requested_seeds = (
        (selected_seed,) if selected_seed is not None else SEEDS
    )
    if any(seed not in observed for seed in requested_seeds):
        raise PackError("prediction shard source is not available")

    shard_root = _ensure_output_directory(output_root)
    manifests = [
        build_authorized_prediction_pack(
            observed[seed],
            authorization=authorization,
            index_binding=index_binding,
            output_root=shard_root / "units",
        )
        for seed in requested_seeds
    ]
    complete_shard = (
        selected_seed is None
        and len(manifests) == len(SEEDS)
        and [int(item["seed"]) for item in manifests] == list(SEEDS)
    )
    shard_index: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "classification": PREDICTION_INDEX_CLASSIFICATION,
        "campaign_id": CAMPAIGN_ID,
        "campaign_revision": PACK_REVISION,
        "outer_fold": outer_fold,
        "seeds": list(SEEDS),
        "unit_count": len(manifests),
        "completed_units": len(manifests),
        "status": "complete" if complete_shard else "filtered_unit_complete",
        "outer_test_opened": False,
        "combined_target_bearing_cache_consumer_access_authorized": False,
        "physical_target_free_outer_prediction_packs": True,
        "outer_prediction_packs_absent": False,
        "cross_outer_shard_mounted": False,
        "promotion_authorization": authorization.binding.as_dict(),
        "units": [],
    }
    for item in manifests:
        seed = int(item["seed"])
        unit_name = f"outer_{outer_fold}_seed_{seed}"
        unit_root = shard_root / "units" / unit_name
        manifest_binding = FileBinding(**item["manifest_binding"])
        predict_binding = _sha256_path_nofollow(
            unit_root / "outer_predict_input.npz"
        )
        shard_index["units"].append(
            {
                "outer_fold": outer_fold,
                "seed": seed,
                "relative_path": f"units/{unit_name}",
                "artifacts": {
                    "prediction_pack_manifest": _relative_binding(
                        manifest_binding, shard_root
                    ),
                    "outer_predict_input": _relative_binding(
                        predict_binding, shard_root
                    ),
                },
            }
        )
    shard_index["content_sha256"] = canonical_content_sha256(shard_index)
    index_file_binding: FileBinding | None = None
    if complete_shard:
        # Re-pin after unit construction so the index cannot attest packs made
        # across an authorization replacement window.
        authorization = _revalidate_promotion_authorization(
            authorization,
            required_scope=PREDICTION_SCOPE,
        )
        if shard_index["promotion_authorization"] != authorization.binding.as_dict():
            raise PackError("prediction authorization changed before index publish")
        index_file_binding = _create_once_bytes(
            shard_root / PREDICTION_INDEX_FILENAME,
            _pretty_json_bytes(shard_index),
        )
        _freeze_directory_tree(shard_root)
    result = dict(shard_index)
    result["index_binding"] = (
        None if index_file_binding is None else index_file_binding.as_dict()
    )
    return result


def _validate_model_bound_selection(
    selection_lock_path: Path,
    *,
    selected_variant: str,
) -> tuple[dict[str, Any], FileBinding]:
    with StableFile(selection_lock_path) as source:
        info = os.fstat(source.fd)
        if stat.S_IMODE(info.st_mode) != 0o444:
            raise PackError("model-bound selection lock must be immutable mode 0444")
        document = _strict_json(source.read_bytes(), str(source.path))
        binding = source.binding
    if not (
        document.get("content_sha256") == canonical_content_sha256(document)
        and document.get("classification")
        == "adaptive_v3r1_v8r4_global_discovery_selection_lock"
        and document.get("campaign_id") == CAMPAIGN_ID
        and document.get("campaign_revision") == PACK_REVISION
        and document.get("selected_variant") == selected_variant
        and document.get("promotion_eligible") is True
        and document.get("promotion_authorized") is True
        and document.get("outer_test_features_or_targets_used") is False
        and document.get("commercial_claim_authorized") is False
    ):
        raise PackError("model-bound selection lock invariant drifted")
    return document, binding


def _require_source_artifact_binding(
    artifacts: Mapping[str, Any],
    *,
    name: str,
    source: StableFile,
) -> None:
    binding = artifacts.get(name)
    if not (
        isinstance(binding, Mapping)
        and binding.get("sha256") == source.binding.sha256
        and binding.get("bytes") == source.binding.bytes
        and isinstance(binding.get("path"), str)
    ):
        raise PackError(f"model source artifact binding drifted: {name}")


def build_model_bound_prediction_pack(
    source: UnitSource,
    *,
    model_source: Any,
    authorization: PromotionAuthorization,
    index_binding: FileBinding,
    selection_lock_path: Path,
    selected_variant: str,
    output_root: Path,
) -> dict[str, Any]:
    """Build one V8R4A successor pack containing its exact model capability.

    ``model_source`` is a host-side, deeply validated ``PromotionModelSource``.
    Only its immutable bytes and an opaque provenance projection are copied;
    the admitted prediction child receives this unit directory and never the
    source training/discovery tree.
    """

    authorization = _revalidate_promotion_authorization(
        authorization,
        required_scope=PREDICTION_SCOPE,
    )
    if not isinstance(selected_variant, str) or not selected_variant:
        raise PackError("model-bound prediction requires one selected variant")
    selection, selection_binding = _validate_model_bound_selection(
        selection_lock_path,
        selected_variant=selected_variant,
    )
    source_kind = getattr(model_source, "kind", None)
    source_receipt_path = getattr(model_source, "receipt_path", None)
    checkpoint_path = getattr(model_source, "checkpoint", None)
    scaler_path = getattr(model_source, "scaler", None)
    signature_sha256 = getattr(
        model_source, "scientific_signature_sha256", None
    )
    source_artifacts = getattr(model_source, "artifacts", None)
    source_receipt_value = getattr(model_source, "receipt", None)
    if not (
        source_kind in {"local_training", "discovery", "discovery_pointer"}
        and isinstance(source_receipt_path, Path)
        and isinstance(checkpoint_path, Path)
        and isinstance(scaler_path, Path)
        and _is_sha256(signature_sha256)
        and isinstance(source_artifacts, Mapping)
        and isinstance(source_receipt_value, Mapping)
    ):
        raise PackError("model-bound source capability object is invalid")

    with ExitStack() as stack:
        receipt_file = stack.enter_context(StableFile(source_receipt_path))
        checkpoint_file = stack.enter_context(StableFile(checkpoint_path))
        scaler_file = stack.enter_context(StableFile(scaler_path))
        for label, stable in (
            ("source receipt", receipt_file),
            ("source checkpoint", checkpoint_file),
            ("source scaler", scaler_file),
        ):
            if stat.S_IMODE(os.fstat(stable.fd).st_mode) != 0o444:
                raise PackError(f"model-bound {label} must be immutable mode 0444")
        receipt = _strict_json(
            receipt_file.read_bytes(), str(receipt_file.path)
        )
        if receipt != dict(source_receipt_value):
            raise PackError("model source receipt differs from its validated value")
        if not (
            receipt.get("content_sha256") == canonical_content_sha256(receipt)
            and receipt.get("campaign_id") == CAMPAIGN_ID
            and receipt.get("campaign_revision") == PACK_REVISION
            and receipt.get("outer_fold") == source.outer_fold
            and receipt.get("seed") == source.seed
            and receipt.get("variant") == selected_variant
            and receipt.get("outer_test_opened") is False
            and receipt.get("commercial_claim_authorized") is False
        ):
            raise PackError("model source receipt identity/leakage boundary drifted")
        receipt_signature = receipt.get("scientific_signature_sha256")
        if receipt_signature is None and isinstance(
            receipt.get("validated_output"), Mapping
        ):
            receipt_signature = receipt["validated_output"].get(
                "scientific_signature_sha256"
            )
        if receipt_signature != signature_sha256:
            raise PackError("model source scientific signature drifted")
        _require_source_artifact_binding(
            source_artifacts,
            name="best.pt",
            source=checkpoint_file,
        )
        _require_source_artifact_binding(
            source_artifacts,
            name="scaler.json",
            source=scaler_file,
        )

        # The target-free base pack is independently immutable.  Replaying it
        # after a kill verifies or completes exactly the same input bytes.
        base = build_authorized_prediction_pack(
            source,
            authorization=authorization,
            index_binding=index_binding,
            output_root=output_root,
        )
        unit_name = f"outer_{source.outer_fold}_seed_{source.seed}"
        unit_root = _ensure_output_directory(output_root / unit_name)
        checkpoint_binding = _copy_stable_file(
            checkpoint_file, unit_root / "model_checkpoint.pt"
        )
        scaler_binding = _copy_stable_file(
            scaler_file, unit_root / "model_scaler.json"
        )
        source_capability: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "classification": MODEL_SOURCE_CAPABILITY_CLASSIFICATION,
            "campaign_id": CAMPAIGN_ID,
            "campaign_revision": PACK_REVISION,
            "infrastructure_revision": "V8R4A",
            "outer_fold": source.outer_fold,
            "seed": source.seed,
            "selected_variant": selected_variant,
            "source_kind": source_kind,
            "scientific_signature_sha256": signature_sha256,
            "source_receipt": receipt_file.binding.as_dict(),
            "source_checkpoint": checkpoint_file.binding.as_dict(),
            "source_scaler": scaler_file.binding.as_dict(),
            "packed_checkpoint": _relative_binding(
                checkpoint_binding, unit_root
            ),
            "packed_scaler": _relative_binding(scaler_binding, unit_root),
            "selection_lock": selection_binding.as_dict(),
            "promotion_authorization": authorization.binding.as_dict(),
            "source_deep_validated_before_copy": True,
            "source_paths_or_peer_outputs_authorized_in_child": False,
            "target_reference_quality_identity_protocol_present": False,
            "model_bytes_changed": False,
            "commercial_or_confirmatory_claim_allowed": False,
        }
        source_capability["content_sha256"] = canonical_content_sha256(
            source_capability
        )
        capability_binding = _create_once_bytes(
            unit_root / MODEL_SOURCE_CAPABILITY_FILENAME,
            _pretty_json_bytes(source_capability),
        )
        base_manifest_binding = FileBinding(**base["manifest_binding"])
        input_binding = _sha256_path_nofollow(
            unit_root / "outer_predict_input.npz"
        )
        manifest: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "classification": MODEL_BOUND_PREDICTION_PACK_CLASSIFICATION,
            "campaign_id": CAMPAIGN_ID,
            "campaign_revision": PACK_REVISION,
            "infrastructure_revision": "V8R4A",
            "outer_fold": source.outer_fold,
            "seed": source.seed,
            "selected_variant": selected_variant,
            "row_count": base["row_count"],
            "global_cache_index_sha256": base[
                "global_cache_index_sha256"
            ],
            "fields": list(OUTER_PREDICT_FIELDS),
            "exact_target_free_allowlist": True,
            "selection_lock": selection_binding.as_dict(),
            "promotion_authorization": authorization.binding.as_dict(),
            "base_target_free_manifest": _relative_binding(
                base_manifest_binding, unit_root
            ),
            "artifacts": {
                "outer_predict_input": _relative_binding(
                    input_binding, unit_root
                ),
                "model_checkpoint": _relative_binding(
                    checkpoint_binding, unit_root
                ),
                "model_scaler": _relative_binding(scaler_binding, unit_root),
                "model_source_capability": _relative_binding(
                    capability_binding, unit_root
                ),
            },
            "exact_unit_file_inventory": sorted(
                {
                    "OUTER_PREDICTION_PACK_MANIFEST.json",
                    MODEL_BOUND_PREDICTION_MANIFEST_FILENAME,
                    MODEL_SOURCE_CAPABILITY_FILENAME,
                    "outer_predict_input.npz",
                    "model_checkpoint.pt",
                    "model_scaler.json",
                }
            ),
            "prediction_child_reads_model_only_from_this_pack": True,
            "source_paths_or_peer_outputs_authorized_in_child": False,
            "target_reference_quality_identity_protocol_present": False,
            "model_bytes_changed": False,
            "commercial_or_confirmatory_claim_allowed": False,
        }
        manifest["content_sha256"] = canonical_content_sha256(manifest)
        manifest_binding = _create_once_bytes(
            unit_root / MODEL_BOUND_PREDICTION_MANIFEST_FILENAME,
            _pretty_json_bytes(manifest),
        )
        receipt_file.assert_stable()
        checkpoint_file.assert_stable()
        scaler_file.assert_stable()
    result = dict(manifest)
    result["manifest_binding"] = manifest_binding.as_dict()
    result["model_source_capability"] = source_capability
    result["model_source_capability_binding"] = capability_binding.as_dict()
    result["selection"] = {
        "selected_variant": selection["selected_variant"],
        "binding": selection_binding.as_dict(),
    }
    return result


def build_model_bound_prediction_shard(
    sources: Sequence[UnitSource],
    *,
    model_sources: Mapping[int, Any],
    authorization: PromotionAuthorization,
    index_binding: FileBinding,
    selection_lock_path: Path,
    selected_variant: str,
    output_root: Path,
    outer_fold: int,
    selected_seed: int | None = None,
) -> dict[str, Any]:
    """Build one exact three-seed successor shard and its model-source seal."""

    authorization = _revalidate_promotion_authorization(
        authorization,
        required_scope=PREDICTION_SCOPE,
    )
    if type(outer_fold) is not int or outer_fold not in range(N_FOLDS):
        raise PackError("model-bound prediction shard outer fold is invalid")
    if selected_seed is not None and selected_seed not in SEEDS:
        raise PackError("model-bound prediction shard seed filter is invalid")
    observed: dict[int, UnitSource] = {}
    for item in sources:
        if item.outer_fold != outer_fold or item.seed not in SEEDS:
            raise PackError("model-bound prediction source identity drifted")
        if item.seed in observed:
            raise PackError("model-bound prediction shard contains duplicate seed")
        observed[item.seed] = item
    requested_seeds = (selected_seed,) if selected_seed is not None else SEEDS
    if set(requested_seeds) - set(observed) or set(requested_seeds) - set(model_sources):
        raise PackError("model-bound prediction shard lacks an exact source")

    shard_root = _ensure_output_directory(output_root)
    units = [
        build_model_bound_prediction_pack(
            observed[seed],
            model_source=model_sources[seed],
            authorization=authorization,
            index_binding=index_binding,
            selection_lock_path=selection_lock_path,
            selected_variant=selected_variant,
            output_root=shard_root / "units",
        )
        for seed in requested_seeds
    ]
    complete = selected_seed is None and [int(unit["seed"]) for unit in units] == list(SEEDS)
    rows: list[dict[str, Any]] = []
    for unit in units:
        seed = int(unit["seed"])
        unit_name = f"outer_{outer_fold}_seed_{seed}"
        unit_root = shard_root / "units" / unit_name
        rows.append(
            {
                "outer_fold": outer_fold,
                "seed": seed,
                "relative_path": f"units/{unit_name}",
                "scientific_signature_sha256": unit[
                    "model_source_capability"
                ]["scientific_signature_sha256"],
                "row_count": int(unit["row_count"]),
                "global_cache_index_sha256": unit[
                    "global_cache_index_sha256"
                ],
                "source_kind": unit["model_source_capability"]["source_kind"],
                "artifacts": {
                    "prediction_pack_manifest": _relative_binding(
                        _sha256_path_nofollow(
                            unit_root / "OUTER_PREDICTION_PACK_MANIFEST.json"
                        ),
                        shard_root,
                    ),
                    "model_bound_prediction_pack_manifest": _relative_binding(
                        _sha256_path_nofollow(
                            unit_root / MODEL_BOUND_PREDICTION_MANIFEST_FILENAME
                        ),
                        shard_root,
                    ),
                    "outer_predict_input": _relative_binding(
                        _sha256_path_nofollow(
                            unit_root / "outer_predict_input.npz"
                        ),
                        shard_root,
                    ),
                    "model_checkpoint": _relative_binding(
                        _sha256_path_nofollow(
                            unit_root / "model_checkpoint.pt"
                        ),
                        shard_root,
                    ),
                    "model_scaler": _relative_binding(
                        _sha256_path_nofollow(unit_root / "model_scaler.json"),
                        shard_root,
                    ),
                    "model_source_capability": _relative_binding(
                        _sha256_path_nofollow(
                            unit_root / MODEL_SOURCE_CAPABILITY_FILENAME
                        ),
                        shard_root,
                    ),
                },
            }
        )
    result: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "classification": MODEL_BOUND_PREDICTION_INDEX_CLASSIFICATION,
        "campaign_id": CAMPAIGN_ID,
        "campaign_revision": PACK_REVISION,
        "infrastructure_revision": "V8R4A",
        "outer_fold": outer_fold,
        "seeds": list(SEEDS),
        "unit_count": len(rows),
        "completed_units": len(rows),
        "status": "complete" if complete else "filtered_unit_complete",
        "selected_variant": selected_variant,
        "outer_test_opened": False,
        "combined_target_bearing_cache_consumer_access_authorized": False,
        "physical_target_free_input_and_model_packs": True,
        "source_paths_or_peer_outputs_authorized_in_child": False,
        "cross_outer_shard_mounted": False,
        "promotion_authorization": authorization.binding.as_dict(),
        "units": rows,
    }
    model_seal_binding: FileBinding | None = None
    index_file_binding: FileBinding | None = None
    if complete:
        selection, selection_binding = _validate_model_bound_selection(
            selection_lock_path,
            selected_variant=selected_variant,
        )
        del selection
        model_seal: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "classification": MODEL_SOURCE_SHARD_SEAL_CLASSIFICATION,
            "campaign_id": CAMPAIGN_ID,
            "campaign_revision": PACK_REVISION,
            "infrastructure_revision": "V8R4A",
            "outer_fold": outer_fold,
            "seeds": list(SEEDS),
            "selected_variant": selected_variant,
            "unit_count": len(SEEDS),
            "exact_three_seed_cover": True,
            "selection_lock": selection_binding.as_dict(),
            "promotion_authorization": authorization.binding.as_dict(),
            "units": [
                {
                    "outer_fold": row["outer_fold"],
                    "seed": row["seed"],
                    "source_kind": row["source_kind"],
                    "scientific_signature_sha256": row[
                        "scientific_signature_sha256"
                    ],
                    "row_count": row["row_count"],
                    "global_cache_index_sha256": row[
                        "global_cache_index_sha256"
                    ],
                    "model_bound_prediction_pack_manifest": row["artifacts"][
                        "model_bound_prediction_pack_manifest"
                    ],
                    "model_checkpoint": row["artifacts"]["model_checkpoint"],
                    "model_scaler": row["artifacts"]["model_scaler"],
                    "model_source_capability": row["artifacts"][
                        "model_source_capability"
                    ],
                }
                for row in rows
            ],
            "target_or_prediction_values_present": False,
            "source_paths_or_peer_outputs_authorized_in_child": False,
            "commercial_or_confirmatory_claim_allowed": False,
        }
        model_seal["content_sha256"] = canonical_content_sha256(model_seal)
        model_seal_binding = _create_once_bytes(
            shard_root / MODEL_SOURCE_SHARD_SEAL_FILENAME,
            _pretty_json_bytes(model_seal),
        )
        result["model_source_shard_seal"] = _relative_binding(
            model_seal_binding, shard_root
        )
        result["content_sha256"] = canonical_content_sha256(result)
        authorization = _revalidate_promotion_authorization(
            authorization,
            required_scope=PREDICTION_SCOPE,
        )
        if result["promotion_authorization"] != authorization.binding.as_dict():
            raise PackError("prediction authorization changed before successor index")
        index_file_binding = _create_once_bytes(
            shard_root / MODEL_BOUND_PREDICTION_INDEX_FILENAME,
            _pretty_json_bytes(result),
        )
        _freeze_directory_tree(shard_root)
    else:
        result["model_source_shard_seal"] = None
        result["content_sha256"] = canonical_content_sha256(result)
    result["index_binding"] = (
        None if index_file_binding is None else index_file_binding.as_dict()
    )
    result["model_source_shard_seal_binding"] = (
        None if model_seal_binding is None else model_seal_binding.as_dict()
    )
    return result


def _maybe_publish_promotion_training_aggregator(
    output_root: Path,
    *,
    authorization: PromotionAuthorization,
) -> FileBinding | None:
    """Publish a pack-free four-shard receipt once every isolated index exists."""

    authorization = _revalidate_promotion_authorization(
        authorization,
        required_scope=PROMOTION_TRAINING_SCOPE,
    )
    shard_rows: list[dict[str, Any]] = []
    expected_keys = {
        "schema_version",
        "classification",
        "campaign_id",
        "campaign_revision",
        "outer_fold",
        "seeds",
        "unit_count",
        "completed_units",
        "status",
        "outer_test_opened",
        "combined_target_bearing_cache_consumer_access_authorized",
        "physical_nonouter_training_packs",
        "outer_prediction_packs_absent",
        "cross_outer_shard_mounted",
        "promotion_scope",
        "promotion_authorization",
        "units",
        "content_sha256",
    }
    for outer_fold in PROMOTION_TRAINING_FOLDS:
        index_path = (
            output_root
            / f"promotion_training_shard_outer_{outer_fold}"
            / "V8R4_NONOUTER_TRAINING_INDEX.json"
        )
        if not index_path.exists():
            return None
        with StableFile(index_path) as source:
            if stat.S_IMODE(os.fstat(source.fd).st_mode) != 0o444:
                raise PackError("promotion-training shard index is not mode 0444")
            document = _strict_json(source.read_bytes(), str(index_path))
            binding = source.binding
        units = document.get("units")
        if not (
            set(document) == expected_keys
            and document.get("content_sha256")
            == canonical_content_sha256(document)
            and document.get("schema_version") == SCHEMA_VERSION
            and document.get("classification") == NONOUTER_INDEX_CLASSIFICATION
            and document.get("campaign_id") == CAMPAIGN_ID
            and document.get("campaign_revision") == PACK_REVISION
            and document.get("outer_fold") == outer_fold
            and document.get("seeds") == list(SEEDS)
            and document.get("unit_count") == len(SEEDS)
            and document.get("completed_units") == len(SEEDS)
            and document.get("status") == "complete"
            and document.get("outer_test_opened") is False
            and document.get(
                "combined_target_bearing_cache_consumer_access_authorized"
            )
            is False
            and document.get("physical_nonouter_training_packs") is True
            and document.get("outer_prediction_packs_absent") is True
            and document.get("cross_outer_shard_mounted") is False
            and document.get("promotion_scope") == PROMOTION_TRAINING_SCOPE
            and document.get("promotion_authorization")
            == authorization.binding.as_dict()
            and isinstance(units, list)
            and all(isinstance(row, Mapping) for row in units)
            and [(row.get("outer_fold"), row.get("seed")) for row in units]
            == [(outer_fold, seed) for seed in SEEDS]
        ):
            raise PackError("promotion-training shard index cover/binding drifted")
        shard_rows.append(
            {
                "outer_fold": outer_fold,
                "index_manifest": _relative_binding(binding, output_root),
                "runtime_must_mount_only_this_shard": True,
            }
        )
    aggregator: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "classification": PROMOTION_TRAINING_AGGREGATOR_CLASSIFICATION,
        "campaign_id": CAMPAIGN_ID,
        "campaign_revision": PACK_REVISION,
        "status": "complete",
        "authorized_outer_folds": list(PROMOTION_TRAINING_FOLDS),
        "seeds": list(SEEDS),
        "shard_count": len(PROMOTION_TRAINING_FOLDS),
        "exact_outer_fold_seed_cover": True,
        "runtime_mount_of_aggregator_authorized": False,
        "target_bearing_pack_directories_bound_by_aggregator": False,
        "promotion_scope": PROMOTION_TRAINING_SCOPE,
        "promotion_authorization": authorization.binding.as_dict(),
        "shards": shard_rows,
    }
    aggregator["content_sha256"] = canonical_content_sha256(aggregator)
    authorization = _revalidate_promotion_authorization(
        authorization,
        required_scope=PROMOTION_TRAINING_SCOPE,
    )
    if aggregator["promotion_authorization"] != authorization.binding.as_dict():
        raise PackError("promotion authorization changed before aggregator publish")
    return _create_once_bytes(
        output_root / PROMOTION_TRAINING_AGGREGATOR_FILENAME,
        _pretty_json_bytes(aggregator),
    )


def build_pack_matrix(
    *,
    project_root: Path,
    training_index: Path,
    output_root: Path,
    expected_index_sha256: str | None,
    expected_index_bytes: int | None,
    require_exact_matrix: bool = True,
    selected_outer_fold: int | None = None,
    selected_seed: int | None = None,
    promotion_authorization: PromotionAuthorization | None = None,
) -> dict[str, Any]:
    """Build discovery shards or one authorization-bound promotion shard.

    The legacy trust index is an 18-unit exact cover, but preselection is
    authorized for outer folds 3 and 4 only.  Even those six packs may not
    share a runtime mount: an outer-3 pack contains fold-4 validation labels,
    while an outer-4 pack contains fold-3 validation labels.  Accordingly each
    outer fold receives an independent three-seed shard and index.

    ``selected_*`` exists for synthetic unit tests and kill-safe internal
    construction.  A shard index is published only after its exact three-seed
    cover exists.  Discovery publishes its two-shard aggregator unchanged.
    Promotion training publishes a separate pack-free four-index aggregator
    only after isolated outer 0/1/2/5 shards all exist under one authorization.
    """

    project_root = _absolute_without_resolve(project_root)
    index_path = _absolute_without_resolve(training_index, base=project_root)
    promotion_mode = promotion_authorization is not None
    if promotion_authorization is not None:
        promotion_authorization = _revalidate_promotion_authorization(
            promotion_authorization,
            required_scope=PROMOTION_TRAINING_SCOPE,
        )
        if selected_outer_fold not in PROMOTION_TRAINING_FOLDS:
            raise PackError(
                "promotion-training requires one isolated outer fold in 0,1,2,5"
            )
    elif selected_outer_fold is not None and selected_outer_fold not in {3, 4}:
        raise PackError("preselection discovery permits only outer folds 3 and 4")
    output_root = _ensure_output_directory(
        _absolute_without_resolve(output_root, base=project_root)
    )
    units, index_binding = load_training_index(
        project_root,
        index_path,
        expected_sha256=expected_index_sha256,
        expected_bytes=expected_index_bytes,
        require_exact_matrix=require_exact_matrix,
    )
    permitted_folds = (
        (selected_outer_fold,) if selected_outer_fold is not None else (3, 4)
    )
    selected = [
        unit
        for unit in units
        if unit.outer_fold in permitted_folds
        and (selected_seed is None or unit.seed == selected_seed)
    ]
    if not selected:
        raise PackError("no training-index unit matches the requested filter")
    source_by_key = {(unit.outer_fold, unit.seed): unit for unit in selected}
    shard_results: list[dict[str, Any]] = []
    for outer_fold in permitted_folds:
        shard_directory = (
            f"promotion_training_shard_outer_{outer_fold}"
            if promotion_mode
            else f"discovery_shard_outer_{outer_fold}"
        )
        shard_root = _ensure_output_directory(
            output_root / shard_directory
        )
        manifests = [
            build_unit_pack(
                source_by_key[(outer_fold, seed)],
                index_binding=index_binding,
                output_root=shard_root,
                promotion_authorization=promotion_authorization,
            )
            for seed in SEEDS
            if (outer_fold, seed) in source_by_key
        ]
        complete_shard = len(manifests) == len(SEEDS) and {
            int(item["seed"]) for item in manifests
        } == set(SEEDS)
        shard_index: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "classification": NONOUTER_INDEX_CLASSIFICATION,
            "campaign_id": CAMPAIGN_ID,
            "campaign_revision": PACK_REVISION,
            "outer_fold": outer_fold,
            "seeds": list(SEEDS),
            "unit_count": len(manifests),
            "completed_units": len(manifests),
            "status": "complete" if complete_shard else "filtered_unit_complete",
            "outer_test_opened": False,
            "combined_target_bearing_cache_consumer_access_authorized": False,
            "physical_nonouter_training_packs": True,
            "outer_prediction_packs_absent": True,
            "cross_outer_shard_mounted": False,
            "units": [],
        }
        if promotion_authorization is not None:
            shard_index.update(
                {
                    "promotion_scope": PROMOTION_TRAINING_SCOPE,
                    "promotion_authorization": (
                        promotion_authorization.binding.as_dict()
                    ),
                }
            )
        for item in manifests:
            unit_name = f"outer_{outer_fold}_seed_{int(item['seed'])}"
            unit_root = shard_root / "units" / unit_name
            partition_binding = FileBinding(
                **item["partition_manifest_binding"]
            )
            cache_binding = _sha256_path_nofollow(
                unit_root / "discovery_cache" / "manifest.json"
            )
            proposer_binding = _sha256_path_nofollow(
                unit_root / "discovery_proposer_stack.npz"
            )
            shard_index["units"].append(
                {
                    "outer_fold": outer_fold,
                    "seed": int(item["seed"]),
                    "relative_path": f"units/{unit_name}",
                    "artifacts": {
                        "cache_manifest": _relative_binding(cache_binding, shard_root),
                        "proposer_stack": _relative_binding(proposer_binding, shard_root),
                        "partition_manifest": _relative_binding(
                            partition_binding, shard_root
                        ),
                    },
                }
            )
        shard_index["content_sha256"] = canonical_content_sha256(shard_index)
        index_file_binding: FileBinding | None = None
        if complete_shard:
            if promotion_authorization is not None:
                promotion_authorization = _revalidate_promotion_authorization(
                    promotion_authorization,
                    required_scope=PROMOTION_TRAINING_SCOPE,
                )
                if (
                    shard_index["promotion_authorization"]
                    != promotion_authorization.binding.as_dict()
                ):
                    raise PackError(
                        "promotion authorization changed before shard index publish"
                    )
            index_file_binding = _create_once_bytes(
                shard_root / "V8R4_NONOUTER_TRAINING_INDEX.json",
                _pretty_json_bytes(shard_index),
            )
            _freeze_directory_tree(shard_root)
        shard_results.append(
            {
                "outer_fold": outer_fold,
                "complete": complete_shard,
                "index": shard_index,
                "index_binding": (
                    None
                    if index_file_binding is None
                    else _relative_binding(index_file_binding, output_root)
                ),
            }
        )

    if promotion_authorization is not None:
        complete = len(shard_results) == 1 and shard_results[0]["complete"]
        aggregator_binding = (
            _maybe_publish_promotion_training_aggregator(
                output_root,
                authorization=promotion_authorization,
            )
            if complete
            else None
        )
        promotion_result: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "classification": "adaptive_v3r1_v8r4_promotion_training_shard_build",
            "campaign_id": CAMPAIGN_ID,
            "campaign_revision": PACK_REVISION,
            "status": "complete" if complete else "filtered_build_complete",
            "outer_fold": selected_outer_fold,
            "exact_three_seed_cover": complete,
            "promotion_scope": PROMOTION_TRAINING_SCOPE,
            "promotion_authorization": promotion_authorization.binding.as_dict(),
            "shard_index": shard_results[0]["index_binding"],
            "four_shard_aggregator": (
                None
                if aggregator_binding is None
                else _relative_binding(aggregator_binding, output_root)
            ),
        }
        promotion_result["content_sha256"] = canonical_content_sha256(
            promotion_result
        )
        return promotion_result

    both_complete = (
        {int(item["outer_fold"]) for item in shard_results} == {3, 4}
        and all(item["complete"] for item in shard_results)
    )
    result: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "classification": "adaptive_v3r1_v8r4_discovery_shard_aggregator",
        "campaign_id": CAMPAIGN_ID,
        "campaign_revision": PACK_REVISION,
        "status": "complete" if both_complete else "filtered_build_complete",
        "shard_count": len(shard_results),
        "exact_outer_fold_cover": both_complete,
        "runtime_mount_of_aggregator_authorized": False,
        "target_bearing_pack_directories_bound_by_aggregator": False,
        "shards": [
            {
                "outer_fold": item["outer_fold"],
                "index_manifest": item["index_binding"],
                "runtime_must_mount_only_this_shard": True,
            }
            for item in shard_results
        ],
    }
    result["content_sha256"] = canonical_content_sha256(result)
    if both_complete:
        aggregator_binding = _create_once_bytes(
            output_root / "V8R4_DISCOVERY_SHARD_AGGREGATOR.json",
            _pretty_json_bytes(result),
        )
        result["aggregator_binding"] = aggregator_binding.as_dict()
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--training-index", type=Path, default=DEFAULT_INDEX_RELATIVE)
    parser.add_argument(
        "--scope",
        choices=("discovery", "promotion-training", "prediction"),
        default="discovery",
    )
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--outer-fold", type=int, choices=range(N_FOLDS))
    parser.add_argument("--seed", type=int, choices=SEEDS)
    parser.add_argument("--promotion-authorization", type=Path)
    parser.add_argument("--promotion-authorization-sha256")
    parser.add_argument("--promotion-authorization-bytes", type=int)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        project_root = _absolute_without_resolve(args.project_root)
        if args.scope == "discovery":
            if any(
                value is not None
                for value in (
                    args.outer_fold,
                    args.seed,
                    args.promotion_authorization,
                    args.promotion_authorization_sha256,
                    args.promotion_authorization_bytes,
                )
            ):
                raise PackError("discovery scope refuses unit filters/promotion capability")
            result = build_pack_matrix(
                project_root=project_root,
                training_index=args.training_index,
                output_root=(
                    args.output_root
                    if args.output_root is not None
                    else DEFAULT_OUTPUT_RELATIVE
                ),
                expected_index_sha256=DEFAULT_INDEX_SHA256,
                expected_index_bytes=DEFAULT_INDEX_BYTES,
                require_exact_matrix=True,
                selected_outer_fold=None,
                selected_seed=None,
            )
        else:
            if (
                args.promotion_authorization is None
                or args.promotion_authorization_sha256 is None
                or args.promotion_authorization_bytes is None
            ):
                raise PackError("promotion scope requires immutable exact-bound authorization")
            required_scope = (
                PROMOTION_TRAINING_SCOPE
                if args.scope == "promotion-training"
                else PREDICTION_SCOPE
            )
            authorization = validate_promotion_authorization(
                _absolute_without_resolve(
                    args.promotion_authorization, base=project_root
                ),
                expected_sha256=args.promotion_authorization_sha256,
                expected_bytes=args.promotion_authorization_bytes,
                required_scope=required_scope,
            )
            promotion_root = _absolute_without_resolve(
                (
                    args.output_root
                    if args.output_root is not None
                    else DEFAULT_PROMOTION_OUTPUT_RELATIVE
                ),
                base=project_root,
            )
            discovery_root = _absolute_without_resolve(
                DEFAULT_OUTPUT_RELATIVE, base=project_root
            )
            if promotion_root == discovery_root or discovery_root in promotion_root.parents:
                raise PackError("promotion outputs cannot live under the discovery shard root")
            if args.scope == "promotion-training":
                if (
                    args.outer_fold not in PROMOTION_TRAINING_FOLDS
                    or args.seed is not None
                ):
                    raise PackError(
                        "promotion-training requires one of outer folds 0,1,2,5 and all seeds"
                    )
                result = build_pack_matrix(
                    project_root=project_root,
                    training_index=args.training_index,
                    output_root=promotion_root / "training_shards",
                    expected_index_sha256=DEFAULT_INDEX_SHA256,
                    expected_index_bytes=DEFAULT_INDEX_BYTES,
                    require_exact_matrix=True,
                    selected_outer_fold=args.outer_fold,
                    selected_seed=None,
                    promotion_authorization=authorization,
                )
            else:
                if args.outer_fold is None:
                    raise PackError("prediction scope requires exactly one outer fold")
                units, index_binding = load_training_index(
                    project_root,
                    _absolute_without_resolve(args.training_index, base=project_root),
                    expected_sha256=DEFAULT_INDEX_SHA256,
                    expected_bytes=DEFAULT_INDEX_BYTES,
                    require_exact_matrix=True,
                )
                matches = [
                    unit
                    for unit in units
                    if unit.outer_fold == args.outer_fold
                ]
                if len(matches) != len(SEEDS):
                    raise PackError("prediction shard is absent from the immutable index")
                result = build_authorized_prediction_shard(
                    matches,
                    authorization=authorization,
                    index_binding=index_binding,
                    outer_fold=args.outer_fold,
                    selected_seed=args.seed,
                    output_root=(
                        promotion_root
                        / "prediction_shards"
                        / f"prediction_shard_outer_{args.outer_fold}"
                    ),
                )
    except PackError as error:
        print(json.dumps({"status": "failed_closed", "error": str(error)}, sort_keys=True))
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
