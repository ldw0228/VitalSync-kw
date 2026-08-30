#!/usr/bin/env python3
"""Build a bounded-memory, label-free harmonic candidate-set feature cache."""

from __future__ import annotations

import argparse
import ctypes
from dataclasses import dataclass
from datetime import datetime, timezone
import errno
import fcntl
import hashlib
import json
import os
from pathlib import Path
import secrets
import stat
import sys
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from snn_rr.harmonic_set_data import (  # noqa: E402
    CANDIDATE_SOURCE_NAMES,
    FORBIDDEN_TARGET_QC_FIELDS,
    HARMONIC_RATIOS,
    RF_BRANCH_NAMES,
    SEMANTIC_ROW_FIELDS,
    VERIFIED_SVD_VARIANT_INDICES,
    VERIFIED_SVD_VARIANT_NAMES,
    CandidateSource,
    candidate_bank_from_metadata,
    iter_compact_node_feature_batches,
    resolve_joint_radar_mask,
    semantic_row_binding_sha256,
)
from snn_rr.harmonic_feature_layout_v3r1 import (  # noqa: E402
    EXPECTED_FEATURE_NAMES_SEMANTIC_SHA256,
    FEATURE_LAYOUT_SEMANTIC_SHA256,
    TOTAL_FEATURE_WIDTH,
    validate_ordered_feature_names,
)


DEFAULT_RF_CACHE = PROJECT_ROOT / "artifacts/cache/rf32s"
DEFAULT_SVD_CACHE = PROJECT_ROOT / "artifacts/cache/svd_components_all_v1"
DEFAULT_PROPOSER = (
    PROJECT_ROOT
    / "artifacts/runs/final_alias_gate_s12_deterministic/all_windows_cuda_v3"
    / "snn_all_windows.npz"
)
DEFAULT_FOLDS = (
    PROJECT_ROOT
    / "artifacts/runs/final_alias_gate_s12_deterministic/fold_assignments.json"
)
DEFAULT_OUTPUT = PROJECT_ROOT / "artifacts/cache/harmonic_set_v2"
# Version 2 adds a mandatory per-feature structural-availability tensor and
# exact output inventory.  Reusing version 1 here would let old consumers
# silently interpret two incompatible cache contracts as the same schema.
FORMAT_VERSION = 2
SCHEMA_ID = "snn_rr.harmonic_candidate_cache.v2"
MAX_CANDIDATES = 12
TOPK_PROPOSALS = 5
POSTERIOR_GRID_KEYS = ("posterior_rr_grid_bpm", "posterior_rr_bins_bpm")
BASE_PROPOSAL_CHOICES = ("none", "expected", "map", "expected-map")
ACQUISITION_V2_SCHEMA = "snn_rr.feature_cache_acquisition.v2"

RF_TIMING_MASK_CONTRACT = {
    "mask_required_for_gap_tolerant_consumers": True,
    "scientific_cache_requires_all_true": True,
    "diagnostic_cache_trainable": False,
    "invalid_cells_are_exact_zero_but_not_semantic_measurements": True,
}
SVD_TIMING_MASK_CONTRACT = {
    "mask_required_for_gap_tolerant_consumers": True,
    "scientific_source_requires_all_true": True,
    "diagnostic_output_trainable": False,
    "invalid_cells_are_exact_zero_but_not_semantic_measurements": True,
}

ARRAY_FILES = {
    "node_features": "node_features.npy",
    "candidate_bpm": "candidate_bpm.npy",
    "candidate_mask": "candidate_mask.npy",
    "candidate_confidence": "candidate_confidence.npy",
    "candidate_source_mask": "candidate_source_mask.npy",
    "candidate_primary_source": "candidate_primary_source.npy",
    "joint_radar_mask": "joint_radar_mask.npy",
    "rf_support_count": "rf_support_count.npy",
    "svd_support_count": "svd_support_count.npy",
}
NODE_FEATURE_AVAILABILITY_FILE = "node_feature_availability.npy"
_HARMONIC_RATIO_TOKENS = ("r1_4", "r1_3", "r1_2", "r1", "r2", "r3", "r4")
_FICLONE = 0x40049409
_AT_FDCWD = -100
_RENAME_NOREPLACE = 1


def _clear_private_directory_fd(descriptor: int) -> None:
    for name in os.listdir(descriptor):
        details = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
        if stat.S_ISDIR(details.st_mode):
            child = os.open(
                name,
                os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW,
                dir_fd=descriptor,
            )
            try:
                _clear_private_directory_fd(child)
            finally:
                os.close(child)
            os.rmdir(name, dir_fd=descriptor)
        else:
            os.unlink(name, dir_fd=descriptor)
    os.fsync(descriptor)


@dataclass(slots=True)
class _StablePrivateDirectory:
    """Private tree with fd/inode cleanup across parent path rebinding."""

    path: Path
    parent_path: Path
    name: str
    parent_descriptor: int
    root_descriptor: int
    root_device: int
    root_inode: int
    published: bool = False
    cleaned: bool = False

    @classmethod
    def create(cls, parent: Path, *, prefix: str) -> _StablePrivateDirectory:
        parent_path = parent.expanduser().resolve()
        parent_path.mkdir(parents=True, exist_ok=True)
        parent_descriptor = os.open(
            parent_path,
            os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC,
        )
        name = ""
        created_by_this_call = False
        root_descriptor = -1
        try:
            for _ in range(128):
                name = f"{prefix}{secrets.token_hex(16)}"
                try:
                    os.mkdir(name, 0o700, dir_fd=parent_descriptor)
                except FileExistsError:
                    continue
                created_by_this_call = True
                break
            else:
                raise RuntimeError("cannot allocate private harmonic directory")
            root_descriptor = os.open(
                name,
                os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW,
                dir_fd=parent_descriptor,
            )
            details = os.fstat(root_descriptor)
            return cls(
                path=parent_path / name,
                parent_path=parent_path,
                name=name,
                parent_descriptor=parent_descriptor,
                root_descriptor=root_descriptor,
                root_device=details.st_dev,
                root_inode=details.st_ino,
            )
        except BaseException:
            if root_descriptor >= 0:
                os.close(root_descriptor)
            if created_by_this_call:
                try:
                    os.rmdir(name, dir_fd=parent_descriptor)
                except OSError:
                    pass
            os.close(parent_descriptor)
            raise

    def assert_parent_path_current(self) -> None:
        captured = os.fstat(self.parent_descriptor)
        current = self.parent_path.stat()
        if not (
            stat.S_ISDIR(captured.st_mode)
            and stat.S_ISDIR(current.st_mode)
            and captured.st_dev == current.st_dev
            and captured.st_ino == current.st_ino
        ):
            raise RuntimeError("harmonic output parent changed during build")

    def mark_published(self) -> None:
        self.published = True
        os.close(self.root_descriptor)
        os.close(self.parent_descriptor)
        self.cleaned = True

    def cleanup(self) -> None:
        if self.cleaned:
            return
        cleanup_error: BaseException | None = None
        try:
            if self.published:
                return
            details = os.fstat(self.root_descriptor)
            if (
                not stat.S_ISDIR(details.st_mode)
                or details.st_dev != self.root_device
                or details.st_ino != self.root_inode
            ):
                raise RuntimeError("private harmonic directory identity changed")
            _clear_private_directory_fd(self.root_descriptor)
        except BaseException as error:
            cleanup_error = error
        removed = False
        try:
            candidates = [self.name]
            candidates.extend(
                name for name in os.listdir(self.parent_descriptor) if name != self.name
            )
            for name in candidates:
                try:
                    details = os.stat(
                        name,
                        dir_fd=self.parent_descriptor,
                        follow_symlinks=False,
                    )
                except FileNotFoundError:
                    continue
                if (
                    stat.S_ISDIR(details.st_mode)
                    and details.st_dev == self.root_device
                    and details.st_ino == self.root_inode
                ):
                    os.rmdir(name, dir_fd=self.parent_descriptor)
                    removed = True
                    break
            if not removed and cleanup_error is None:
                removed = True
            os.fsync(self.parent_descriptor)
        except BaseException as error:
            if cleanup_error is None:
                cleanup_error = error
        finally:
            try:
                os.close(self.root_descriptor)
            except OSError as error:
                if cleanup_error is None:
                    cleanup_error = error
            try:
                os.close(self.parent_descriptor)
            except OSError as error:
                if cleanup_error is None:
                    cleanup_error = error
        self.cleaned = cleanup_error is None and removed
        if cleanup_error is not None:
            raise RuntimeError("private harmonic directory cleanup failed") from cleanup_error


@dataclass(slots=True)
class _BoundInputSnapshot:
    """Private COW/stream copies of every file consumed by one build."""

    temporary: _StablePrivateDirectory
    rf_cache: Path
    svd_cache: Path
    proposer: Path
    folds: Path
    bindings: dict[str, dict[str, Any]]

    def assert_private_bytes_current(self) -> None:
        for relative, expected in self.bindings.items():
            path = self.temporary.path / relative
            observed = _file_binding(path)
            if (
                observed["sha256"] != expected["sha256"]
                or observed["bytes"] != expected["bytes"]
            ):
                raise RuntimeError(
                    f"private harmonic input snapshot changed during build: {relative}"
                )

    def cleanup(self) -> None:
        self.temporary.cleanup()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _file_binding(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return {
        "path": str(resolved),
        "sha256": sha256_file(resolved),
        "bytes": resolved.stat().st_size,
    }


def _capture_direct_entry_disk_binding(path: Path) -> dict[str, Any]:
    """Capture stable direct-entry source bytes during module initialization.

    This is not a claim about the bytes already compiled by the loader.
    Complete executed-code closure requires a fresh isolated child importing
    every dependency exclusively from one private source snapshot.
    """

    resolved = path.expanduser().resolve()
    digest = hashlib.sha256()
    try:
        with resolved.open("rb") as stream:
            before = os.fstat(stream.fileno())
            consumed = 0
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)
                consumed += len(chunk)
            after = os.fstat(stream.fileno())
    except OSError as error:
        raise RuntimeError(
            f"cannot capture direct builder import generation: {resolved}"
        ) from error
    if not (
        stat.S_ISREG(before.st_mode)
        and before.st_dev == after.st_dev
        and before.st_ino == after.st_ino
        and before.st_size == after.st_size == consumed
        and before.st_mtime_ns == after.st_mtime_ns
        and before.st_ctime_ns == after.st_ctime_ns
    ):
        raise RuntimeError(
            f"direct builder source changed while its generation was captured: {resolved}"
        )
    return {"path": str(resolved), "sha256": digest.hexdigest(), "bytes": consumed}


_DIRECT_ENTRY_DISK_BINDING = _capture_direct_entry_disk_binding(Path(__file__))


def _assert_direct_entry_disk_binding_current(
    expected: Mapping[str, Any] = _DIRECT_ENTRY_DISK_BINDING,
) -> dict[str, Any]:
    observed = _capture_direct_entry_disk_binding(
        Path(str(expected.get("path", "")))
    )
    if observed != dict(expected):
        raise RuntimeError(
            "direct harmonic builder source changed since its initialization-time "
            "disk binding; "
            "a fresh isolated private-source process is required"
        )
    return observed


def _copy_bound_regular_file(
    source_path: Path,
    destination: Path,
    expected: Mapping[str, Any],
) -> dict[str, Any]:
    """Create an exact private snapshot and prove it matches ``expected``.

    Linux reflink is attempted first so multi-gigabyte arrays remain practical;
    its copy-on-write inode is an instantaneous byte snapshot.  Filesystems
    without reflink use a bounded-memory streaming copy.  Both paths hash the
    destination and require stable source-inode metadata across the operation.
    """

    source = source_path.expanduser().resolve()
    if set(expected) != {"path", "sha256", "bytes"}:
        raise RuntimeError(f"input binding schema is invalid: {source}")
    if expected.get("path") != str(source):
        raise RuntimeError(f"input binding path mismatch: {source}")
    expected_hash = expected.get("sha256")
    expected_bytes = expected.get("bytes")
    if (
        not isinstance(expected_hash, str)
        or len(expected_hash) != 64
        or type(expected_bytes) is not int
        or expected_bytes < 0
    ):
        raise RuntimeError(f"input binding value is invalid: {source}")

    destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    try:
        with source.open("rb") as source_stream, destination.open("x+b") as target:
            before = os.fstat(source_stream.fileno())
            if not stat.S_ISREG(before.st_mode):
                raise RuntimeError(f"harmonic input is not a regular file: {source}")
            cloned = False
            try:
                fcntl.ioctl(target.fileno(), _FICLONE, source_stream.fileno())
                cloned = True
            except OSError:
                target.seek(0)
                target.truncate(0)
                source_stream.seek(0)
                digest = hashlib.sha256()
                copied = 0
                while chunk := source_stream.read(4 * 1024 * 1024):
                    target.write(chunk)
                    digest.update(chunk)
                    copied += len(chunk)
            target.flush()
            os.fsync(target.fileno())
            if cloned:
                target.seek(0)
                digest = hashlib.sha256()
                copied = 0
                while chunk := target.read(4 * 1024 * 1024):
                    digest.update(chunk)
                    copied += len(chunk)
            after = os.fstat(source_stream.fileno())
            if not (
                before.st_dev == after.st_dev
                and before.st_ino == after.st_ino
                and before.st_size == after.st_size
                and before.st_mtime_ns == after.st_mtime_ns
                and before.st_ctime_ns == after.st_ctime_ns
                and copied == after.st_size
            ):
                raise RuntimeError(
                    f"harmonic input changed while being snapshotted: {source}"
                )
            observed_hash = digest.hexdigest()
            if copied != expected_bytes or observed_hash != expected_hash:
                raise RuntimeError(
                    f"harmonic input snapshot differs from bound bytes: {source}"
                )
            os.fchmod(target.fileno(), 0o400)
    except BaseException:
        destination.unlink(missing_ok=True)
        raise
    return {"path": str(destination), "sha256": observed_hash, "bytes": copied}


def _materialize_bound_input_snapshot(
    *,
    rf_cache: Path,
    svd_cache: Path,
    proposer_path: Path,
    folds_path: Path,
    sessions: Sequence[str],
    input_bindings: Mapping[str, Any],
    parent: Path,
) -> _BoundInputSnapshot:
    """Copy every subsequently parsed/mapped data input into a private tree."""

    temporary = _StablePrivateDirectory.create(
        parent,
        prefix=".harmonic-input-snapshot.",
    )
    root = temporary.path
    snapshot_bindings: dict[str, dict[str, Any]] = {}

    def copy(
        source: Path, relative: Path, expected: Mapping[str, Any]
    ) -> Path:
        if (
            relative.is_absolute()
            or not relative.parts
            or any(part in {"", ".", ".."} for part in relative.parts)
        ):
            raise RuntimeError(f"unsafe harmonic snapshot path: {relative}")
        destination = (root / relative).resolve()
        try:
            destination.relative_to(root.resolve())
        except ValueError as error:
            raise RuntimeError(
                f"harmonic snapshot path escapes private root: {relative}"
            ) from error
        observed = _copy_bound_regular_file(source, destination, expected)
        snapshot_bindings[str(relative)] = observed
        return destination

    try:
        for session_id in sessions:
            _validate_safe_session_id(session_id, "harmonic input")
        copy(
            rf_cache / "manifest.json",
            Path("rf/manifest.json"),
            input_bindings["rf_root_manifest"],
        )
        copy(
            svd_cache / "manifest.json",
            Path("svd/manifest.json"),
            input_bindings["svd_root_manifest"],
        )
        snapshot_proposer = copy(
            proposer_path, Path("proposer/input.npz"), input_bindings["proposer"]
        )
        snapshot_folds = copy(
            folds_path, Path("folds/input.json"), input_bindings["fold_assignments"]
        )
        session_bindings = input_bindings.get("sessions")
        if not isinstance(session_bindings, Mapping) or set(session_bindings) != set(
            sessions
        ):
            raise RuntimeError("harmonic session input binding inventory is invalid")
        for session_id in sessions:
            bound = session_bindings[session_id]
            if not isinstance(bound, Mapping):
                raise RuntimeError(f"harmonic session binding is invalid: {session_id}")
            for side, cache_root in (("rf", rf_cache), ("svd", svd_cache)):
                session_root = (cache_root / session_id).resolve()
                try:
                    session_root.relative_to(cache_root.resolve())
                except ValueError as error:
                    raise RuntimeError(
                        f"harmonic {side} session path escapes cache root: {session_id}"
                    ) from error
                side_bindings = bound.get(side)
                if not isinstance(side_bindings, Mapping):
                    raise RuntimeError(
                        f"harmonic {side} session binding is invalid: {session_id}"
                    )
                for binding in side_bindings.values():
                    if not isinstance(binding, Mapping):
                        raise RuntimeError(
                            f"harmonic {side} file binding is invalid: {session_id}"
                        )
                    original = Path(str(binding.get("path", ""))).resolve()
                    try:
                        relative_name = original.relative_to(
                            session_root
                        )
                    except ValueError as error:
                        raise RuntimeError(
                            f"harmonic {side} binding escapes session root: {session_id}"
                        ) from error
                    if len(relative_name.parts) != 1:
                        raise RuntimeError(
                            f"harmonic {side} binding has nested payload path: {session_id}"
                        )
                    copy(
                        original,
                        Path(side) / session_id / relative_name,
                        binding,
                    )
        result = _BoundInputSnapshot(
            temporary=temporary,
            rf_cache=root / "rf",
            svd_cache=root / "svd",
            proposer=snapshot_proposer,
            folds=snapshot_folds,
            bindings=snapshot_bindings,
        )
        result.assert_private_bytes_current()
        return result
    except BaseException:
        temporary.cleanup()
        raise


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"invalid JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON root must be an object: {path}")
    return value


def _canonical_digest(value: Mapping[str, Any], *, exclude: str | None = None) -> str:
    payload = dict(value)
    if exclude is not None:
        payload.pop(exclude, None)
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _write_json(path: Path, value: Mapping[str, Any] | Sequence[Any]) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def _successful_sessions(manifest: Mapping[str, Any], label: str) -> list[str]:
    raw = manifest.get("sessions")
    if not isinstance(raw, list) or not raw:
        raise RuntimeError(f"{label} manifest has no sessions")
    result: list[str] = []
    for entry in raw:
        if not isinstance(entry, Mapping):
            raise RuntimeError(f"{label} manifest session entry is invalid")
        if entry.get("status", "ok") == "ok":
            session_id = _validate_safe_session_id(
                entry.get("session_id"), label
            )
            result.append(session_id)
    if not result or len(set(result)) != len(result):
        raise RuntimeError(f"{label} successful session order is invalid")
    return result


def _validate_safe_session_id(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise RuntimeError(f"{label} manifest has an invalid session_id")
    session_path = Path(value)
    if (
        session_path.is_absolute()
        or len(session_path.parts) != 1
        or session_path.name != value
        or value in {".", ".."}
        or "/" in value
        or "\\" in value
    ):
        raise RuntimeError(f"{label} manifest has an unsafe session_id: {value!r}")
    return value


def _fold_map(path: Path) -> dict[str, int]:
    document = _load_json(path)
    raw = document.get("identity_to_fold", document)
    if not isinstance(raw, Mapping) or not raw:
        raise RuntimeError("fold assignments contain no identity map")
    result: dict[str, int] = {}
    for identity, fold in raw.items():
        if not isinstance(identity, str) or not identity:
            raise RuntimeError("fold assignment identity is invalid")
        if isinstance(fold, bool) or not isinstance(fold, int) or fold < 0:
            raise RuntimeError(f"fold assignment for {identity!r} is invalid")
        result[identity] = int(fold)
    return result


def _normalized_semantic_frame(
    frame: pd.DataFrame, cache_index: np.ndarray, fold: np.ndarray
) -> pd.DataFrame:
    result = frame.copy()
    result["cache_index"] = np.asarray(cache_index, dtype=np.int64)
    result["fold"] = np.asarray(fold, dtype=np.int16)
    return result.loc[:, list(SEMANTIC_ROW_FIELDS)]


def _assert_common_rows(left: pd.DataFrame, right: pd.DataFrame, label: str) -> None:
    fields = (
        "session_id",
        "identity",
        "protocol",
        "window_number",
        "window_start_s",
        "window_end_s",
    )
    if len(left) != len(right):
        raise RuntimeError(f"{label} row count mismatch")
    for field in fields:
        if field not in left or field not in right:
            raise RuntimeError(f"{label} lacks semantic field {field}")
        if field in {"session_id", "identity", "protocol"}:
            equal = np.array_equal(
                left[field].astype(str).to_numpy(), right[field].astype(str).to_numpy()
            )
        elif field == "window_number":
            equal = np.array_equal(
                pd.to_numeric(left[field], errors="raise").to_numpy(np.int64),
                pd.to_numeric(right[field], errors="raise").to_numpy(np.int64),
            )
        else:
            equal = np.allclose(
                pd.to_numeric(left[field], errors="raise").to_numpy(float),
                pd.to_numeric(right[field], errors="raise").to_numpy(float),
                rtol=0.0,
                atol=5.0e-7,
            )
        if not equal:
            raise RuntimeError(f"{label} semantic field mismatch: {field}")


def _proposer_frame(data: Mapping[str, np.ndarray]) -> pd.DataFrame:
    required = {
        "cache_index",
        "fold",
        "session_id",
        "identity",
        "protocol",
        "window_number",
        "window_start_s",
        "window_end_s",
    }
    missing = sorted(required - set(data))
    if missing:
        raise RuntimeError(f"proposer lacks semantic fields: {missing}")
    return pd.DataFrame({field: np.asarray(data[field]) for field in required})


@dataclass(frozen=True, slots=True)
class ProposalBundle:
    """Validated label-free proposer content in fixed priority order."""

    bpm: np.ndarray
    confidence: np.ndarray
    mask: np.ndarray
    source: np.ndarray
    availability: np.ndarray
    direct_bpm: np.ndarray
    direct_confidence: np.ndarray
    direct_mask: np.ndarray
    posterior_probability: np.ndarray | None
    posterior_rr_grid_bpm: np.ndarray | None
    expected_bpm: np.ndarray | None
    map_bpm: np.ndarray | None
    entropy_normalized: np.ndarray | None
    rr_std_bpm: np.ndarray | None
    quality: np.ndarray | None
    alias_probability: np.ndarray | None
    spike_rate: np.ndarray | None
    radar_weights: np.ndarray | None
    posterior_grid_input_key: str | None


def _proposal_availability(data: Mapping[str, np.ndarray], rows: int) -> np.ndarray:
    if "proposal_available" not in data:
        return np.ones(rows, dtype=bool)
    raw = np.asarray(data["proposal_available"])
    if raw.dtype != np.bool_ or raw.shape != (rows,):
        raise RuntimeError("proposal_available must be a boolean [rows] array")
    return raw.astype(bool, copy=True)


def _topk_proposal_arrays(
    data: Mapping[str, np.ndarray], availability: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rr_key = "topk_rr_bpm" if "topk_rr_bpm" in data else "topk_rr"
    probability_key = "topk_probability"
    if rr_key not in data or probability_key not in data:
        raise RuntimeError("proposer lacks top-5 RR/probability arrays")
    rr = np.asarray(data[rr_key], dtype=np.float32)
    probability = np.asarray(data[probability_key], dtype=np.float32)
    if rr.ndim != 2 or probability.shape != rr.shape or rr.shape[1] < TOPK_PROPOSALS:
        raise RuntimeError("proposer top-k arrays are incompatible with top5 construction")
    rr = rr[:, :TOPK_PROPOSALS]
    probability = probability[:, :TOPK_PROPOSALS]
    if len(rr) != len(availability):
        raise RuntimeError("proposer top5/availability row count mismatch")
    if not np.isfinite(rr[availability]).all() or not np.isfinite(
        probability[availability]
    ).all():
        raise RuntimeError("available proposer top5 contains non-finite values")
    if np.any(probability[availability] < 0) or np.any(
        probability[availability] > 1.0 + 1.0e-6
    ):
        raise RuntimeError("available proposer top5 confidence is outside [0,1]")
    if np.any((rr[availability] < 6.0) | (rr[availability] > 45.0)):
        raise RuntimeError("available proposer top5 RR is outside [6,45] bpm")
    mask = np.broadcast_to(availability[:, None], rr.shape).copy()
    rr = np.where(mask, rr, 0.0).astype(np.float32, copy=False)
    probability = np.where(mask, probability, 0.0).astype(np.float32, copy=False)
    return rr, probability, mask


def _posterior_grid(
    data: Mapping[str, np.ndarray], rows: int, availability: np.ndarray
) -> tuple[np.ndarray, np.ndarray, str]:
    key = next((name for name in POSTERIOR_GRID_KEYS if name in data), None)
    if key is None or "posterior_probability" not in data:
        raise RuntimeError(
            "posterior selection/features require posterior_probability and "
            "posterior_rr_grid_bpm"
        )
    probability = np.asarray(data["posterior_probability"], dtype=np.float64)
    grid = np.asarray(data[key], dtype=np.float64)
    if probability.ndim != 2 or probability.shape[0] != rows:
        raise RuntimeError("posterior_probability must have shape [rows,rr_grid]")
    if grid.ndim != 1 or probability.shape[1] != len(grid) or len(grid) < 2:
        raise RuntimeError("posterior RR grid shape is inconsistent with probabilities")
    if not np.isfinite(grid).all() or np.any(np.diff(grid) <= 0):
        raise RuntimeError("posterior RR grid must be finite and strictly increasing")
    if grid[0] < 6.0 - 1.0e-6 or grid[-1] > 45.0 + 1.0e-6:
        raise RuntimeError("posterior RR grid is outside the declared [6,45] bpm range")
    active = probability[availability]
    if not np.isfinite(active).all() or np.any(active < 0):
        raise RuntimeError("available posterior probabilities are non-finite or negative")
    sums = active.sum(axis=1)
    if np.any(sums <= 0) or not np.allclose(sums, 1.0, rtol=0.0, atol=2.0e-3):
        raise RuntimeError("available posterior rows are not normalized probabilities")
    inactive = probability[~availability]
    if inactive.size and (
        not np.isfinite(inactive).all() or np.count_nonzero(inactive) != 0
    ):
        raise RuntimeError(
            "unavailable proposer rows must carry an exactly zero finite posterior"
        )
    normalized = np.zeros_like(probability, dtype=np.float64)
    normalized[availability] = active / sums[:, None]
    return normalized, grid, key


def posterior_nms_modes(
    posterior_probability: np.ndarray,
    posterior_rr_grid_bpm: np.ndarray,
    availability: np.ndarray,
    *,
    top_k: int = TOPK_PROPOSALS,
    suppression_bpm: float = 1.25,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Greedy deterministic posterior NMS with stable lower-grid tie priority."""

    probability = np.asarray(posterior_probability, dtype=np.float64)
    grid = np.asarray(posterior_rr_grid_bpm, dtype=np.float64)
    available = np.asarray(availability, dtype=bool)
    if probability.ndim != 2 or grid.shape != (probability.shape[1],):
        raise ValueError("posterior probability/grid shapes are incompatible")
    if available.shape != (probability.shape[0],):
        raise ValueError("posterior availability shape mismatch")
    if int(top_k) < 1 or not np.isfinite(suppression_bpm) or suppression_bpm < 0:
        raise ValueError("posterior NMS settings are invalid")
    rr = np.zeros((len(probability), int(top_k)), dtype=np.float32)
    confidence = np.zeros_like(rr)
    mask = np.zeros(rr.shape, dtype=bool)
    for row in np.flatnonzero(available):
        # mergesort/stable preserves increasing RR-grid order for exact ties.
        order = np.argsort(-probability[row], kind="stable")
        selected: list[int] = []
        for index in order:
            if probability[row, index] <= 0:
                break
            if any(
                abs(float(grid[index] - grid[other]))
                <= float(suppression_bpm) + 1.0e-12
                for other in selected
            ):
                continue
            selected.append(int(index))
            if len(selected) == int(top_k):
                break
        if selected:
            count = len(selected)
            rr[row, :count] = grid[selected]
            confidence[row, :count] = probability[row, selected]
            mask[row, :count] = True
    return rr, confidence, mask


def _required_scalar(
    data: Mapping[str, np.ndarray],
    keys: Sequence[str],
    availability: np.ndarray,
    *,
    label: str,
    minimum: float | None = None,
    maximum: float | None = None,
    strictly_positive: bool = False,
) -> np.ndarray:
    key = next((name for name in keys if name in data), None)
    if key is None:
        raise RuntimeError(f"proposer lacks required {label} array")
    values = np.asarray(data[key], dtype=np.float64)
    if values.shape != availability.shape:
        raise RuntimeError(f"proposer {label} must have shape [rows]")
    active = values[availability]
    if not np.isfinite(active).all():
        raise RuntimeError(f"available proposer {label} contains non-finite values")
    if strictly_positive and np.any(active <= 0):
        raise RuntimeError(f"available proposer {label} must be positive")
    if minimum is not None and np.any(active < minimum - 1.0e-7):
        raise RuntimeError(f"available proposer {label} is below {minimum}")
    if maximum is not None and np.any(active > maximum + 1.0e-7):
        raise RuntimeError(f"available proposer {label} is above {maximum}")
    return np.where(availability, values, 0.0).astype(np.float32)


def _proposal_bundle(
    data: Mapping[str, np.ndarray],
    *,
    selection: str,
    suppression_bpm: float,
    base_proposals: str,
    include_features: bool,
) -> ProposalBundle:
    rows = len(np.asarray(data.get("cache_index", ())))
    if rows < 1:
        raise RuntimeError("proposer contains no canonical rows")
    availability = _proposal_availability(data, rows)
    require_posterior = (
        selection == "posterior-nms"
        or base_proposals != "none"
        or include_features
    )
    posterior: np.ndarray | None = None
    grid: np.ndarray | None = None
    grid_key: str | None = None
    expected: np.ndarray | None = None
    map_bpm: np.ndarray | None = None
    entropy: np.ndarray | None = None
    rr_std: np.ndarray | None = None
    quality: np.ndarray | None = None
    alias: np.ndarray | None = None
    spike: np.ndarray | None = None
    radar_weights: np.ndarray | None = None
    if require_posterior:
        posterior, grid, grid_key = _posterior_grid(data, rows, availability)
        expected = (posterior @ grid).astype(np.float32)
        map_index = np.argmax(posterior, axis=1)
        map_bpm = np.where(availability, grid[map_index], 0.0).astype(np.float32)
        entropy = np.zeros(rows, dtype=np.float32)
        if availability.any():
            active = posterior[availability]
            entropy[availability] = (
                -(
                    active * np.log(np.maximum(active, np.finfo(np.float64).tiny))
                ).sum(axis=1)
                / np.log(float(active.shape[1]))
            ).astype(np.float32)
    if selection == "topk":
        direct_rr, direct_conf, direct_mask = _topk_proposal_arrays(
            data, availability
        )
    elif selection == "posterior-nms":
        assert posterior is not None and grid is not None
        direct_rr, direct_conf, direct_mask = posterior_nms_modes(
            posterior,
            grid,
            availability,
            top_k=TOPK_PROPOSALS,
            suppression_bpm=suppression_bpm,
        )
    else:
        raise ValueError("proposal selection must be topk or posterior-nms")

    proposal_parts: list[np.ndarray] = []
    confidence_parts: list[np.ndarray] = []
    mask_parts: list[np.ndarray] = []
    source_parts: list[np.ndarray] = []
    base_order = (
        ("expected", "map")
        if base_proposals == "expected-map"
        else () if base_proposals == "none" else (base_proposals,)
    )
    for kind in base_order:
        values = expected if kind == "expected" else map_bpm
        assert values is not None and posterior is not None and grid is not None
        if kind == "expected":
            base_confidence = np.asarray(
                [
                    posterior[row, np.abs(grid - values[row]) <= 1.0 + 1.0e-12].sum()
                    if availability[row]
                    else 0.0
                    for row in range(rows)
                ],
                dtype=np.float32,
            )
        else:
            base_confidence = posterior.max(axis=1).astype(np.float32)
        base_mask = availability & np.isfinite(values) & (values >= 6.0) & (values <= 45.0)
        proposal_parts.append(np.where(base_mask, values, 0.0)[:, None])
        confidence_parts.append(np.where(base_mask, base_confidence, 0.0)[:, None])
        mask_parts.append(base_mask[:, None])
        source_parts.append(
            np.full((rows, 1), int(CandidateSource.BASE), dtype=np.int16)
        )
    proposal_parts.append(direct_rr)
    confidence_parts.append(direct_conf)
    mask_parts.append(direct_mask)
    source_parts.append(
        np.full(direct_rr.shape, int(CandidateSource.DIRECT_MODE), dtype=np.int16)
    )

    if include_features:
        rr_std = _required_scalar(
            data,
            ("rr_std", "rr_std_bpm"),
            availability,
            label="rr_std",
            strictly_positive=True,
        )
        quality = _required_scalar(
            data, ("quality",), availability, label="quality", minimum=0.0, maximum=1.0
        )
        alias = _required_scalar(
            data,
            ("alias_probability",),
            availability,
            label="alias_probability",
            minimum=0.0,
            maximum=1.0,
        )
        spike = _required_scalar(
            data,
            ("spike_rate",),
            availability,
            label="spike_rate",
            minimum=0.0,
            maximum=1.0,
        )
        if "radar_weights" not in data:
            raise RuntimeError("proposer lacks required radar_weights array")
        raw_weights = np.asarray(data["radar_weights"], dtype=np.float64)
        if raw_weights.shape != (rows, 3):
            raise RuntimeError("proposer radar_weights must have shape [rows,3]")
        if (
            not np.isfinite(raw_weights[availability]).all()
            or np.any(raw_weights[availability] < -1.0e-7)
            or np.any(raw_weights[availability] > 1.0 + 1.0e-7)
        ):
            raise RuntimeError("available proposer radar_weights are invalid")
        radar_weights = np.where(availability[:, None], raw_weights, 0.0).astype(
            np.float32
        )

    return ProposalBundle(
        bpm=np.concatenate(proposal_parts, axis=1).astype(np.float32),
        confidence=np.concatenate(confidence_parts, axis=1).astype(np.float32),
        mask=np.concatenate(mask_parts, axis=1).astype(bool),
        source=np.concatenate(source_parts, axis=1).astype(np.int16),
        availability=availability,
        direct_bpm=direct_rr,
        direct_confidence=direct_conf,
        direct_mask=direct_mask,
        posterior_probability=posterior,
        posterior_rr_grid_bpm=grid,
        expected_bpm=expected,
        map_bpm=map_bpm,
        entropy_normalized=entropy,
        rr_std_bpm=rr_std,
        quality=quality,
        alias_probability=alias,
        spike_rate=spike,
        radar_weights=radar_weights,
        posterior_grid_input_key=grid_key,
    )


def _load_proposer(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {name: np.asarray(archive[name]) for name in archive.files}


PROPOSER_NODE_FEATURE_NAMES = (
    "proposer_available",
    "direct_mode_rank",
    "direct_mode_reciprocal_rank",
    "direct_mode_selected_probability",
    "posterior_nearest_bin_probability",
    "posterior_peak_probability",
    "posterior_local_mass_pm0p5_bpm",
    "posterior_local_mass_pm1p0_bpm",
    "posterior_entropy_normalized",
    "proposer_log_rr_std_bpm",
    "proposer_quality",
    "proposer_alias_probability",
    "proposer_spike_rate",
    "candidate_minus_expected_bpm",
    "candidate_abs_expected_distance_bpm",
    "candidate_minus_map_bpm",
    "candidate_abs_map_distance_bpm",
    "proposer_expected_anchor_match",
    "proposer_map_anchor_match",
    "proposer_radar1_weight",
    "proposer_radar2_weight",
    "proposer_radar3_weight",
)


def proposer_candidate_node_features(
    bundle: ProposalBundle,
    candidates: Any,
    row_selector: slice | np.ndarray,
    *,
    row_available: np.ndarray | None = None,
) -> tuple[np.ndarray, tuple[str, ...]]:
    """Bind full-posterior deployment descriptors to the final sorted anchors."""

    required = (
        bundle.posterior_probability,
        bundle.posterior_rr_grid_bpm,
        bundle.expected_bpm,
        bundle.map_bpm,
        bundle.entropy_normalized,
        bundle.rr_std_bpm,
        bundle.quality,
        bundle.alias_probability,
        bundle.spike_rate,
        bundle.radar_weights,
    )
    if any(value is None for value in required):
        raise RuntimeError("full proposer node features were not validated")
    posterior = np.asarray(bundle.posterior_probability)[row_selector]
    grid = np.asarray(bundle.posterior_rr_grid_bpm)
    expected = np.asarray(bundle.expected_bpm)[row_selector]
    map_bpm = np.asarray(bundle.map_bpm)[row_selector]
    entropy = np.asarray(bundle.entropy_normalized)[row_selector]
    rr_std = np.asarray(bundle.rr_std_bpm)[row_selector]
    quality = np.asarray(bundle.quality)[row_selector]
    alias = np.asarray(bundle.alias_probability)[row_selector]
    spike = np.asarray(bundle.spike_rate)[row_selector]
    radar_weights = np.asarray(bundle.radar_weights)[row_selector]
    upstream_availability = np.asarray(bundle.availability)[row_selector]
    if row_available is None:
        availability = upstream_availability
    else:
        availability = np.asarray(row_available)
        if (
            availability.dtype != np.bool_
            or availability.shape != upstream_availability.shape
            or np.any(availability & ~upstream_availability)
        ):
            raise RuntimeError("effective proposer row availability is invalid")
    direct_rr = np.asarray(bundle.direct_bpm)[row_selector]
    direct_confidence = np.asarray(bundle.direct_confidence)[row_selector]
    direct_mask = np.asarray(bundle.direct_mask)[row_selector]
    bpm = np.asarray(candidates.bpm, dtype=np.float64)
    candidate_mask = np.asarray(candidates.mask, dtype=bool)
    if bpm.shape[0] != len(availability):
        raise RuntimeError("proposer/candidate row subset mismatch")
    features = np.zeros(
        (*bpm.shape, len(PROPOSER_NODE_FEATURE_NAMES)), dtype=np.float32
    )
    name_to_index = {
        name: index for index, name in enumerate(PROPOSER_NODE_FEATURE_NAMES)
    }
    for row in range(len(bpm)):
        if not availability[row]:
            # This is the crucial outer-test behavior: even classical-only nodes
            # receive no proposer descriptor when their nested proposal is absent.
            continue
        active_candidates = np.flatnonzero(candidate_mask[row])
        for candidate in active_candidates:
            value = float(bpm[row, candidate])
            output = features[row, candidate]
            output[name_to_index["proposer_available"]] = 1.0
            eligible = np.flatnonzero(
                direct_mask[row]
                & (
                    np.abs(np.asarray(direct_rr[row], dtype=np.float64) - value)
                    <= float(candidates.merge_radius_bpm) + 1.0e-12
                )
            )
            if len(eligible):
                # Direct proposals are already in stable NMS/top-k rank order.
                rank_index = int(eligible[0])
                output[name_to_index["direct_mode_rank"]] = float(rank_index + 1)
                output[name_to_index["direct_mode_reciprocal_rank"]] = 1.0 / float(
                    rank_index + 1
                )
                output[name_to_index["direct_mode_selected_probability"]] = float(
                    direct_confidence[row, rank_index]
                )
            nearest = int(np.argmin(np.abs(grid - value)))
            output[name_to_index["posterior_nearest_bin_probability"]] = float(
                posterior[row, nearest]
            )
            local_peak = np.abs(grid - value) <= 0.5 + 1.0e-12
            output[name_to_index["posterior_peak_probability"]] = float(
                posterior[row, local_peak].max(initial=0.0)
            )
            for radius, feature_name in (
                (0.5, "posterior_local_mass_pm0p5_bpm"),
                (1.0, "posterior_local_mass_pm1p0_bpm"),
            ):
                output[name_to_index[feature_name]] = float(
                    posterior[row, np.abs(grid - value) <= radius + 1.0e-12].sum()
                )
            output[name_to_index["posterior_entropy_normalized"]] = float(
                entropy[row]
            )
            output[name_to_index["proposer_log_rr_std_bpm"]] = float(
                np.log(max(float(rr_std[row]), 1.0e-8))
            )
            output[name_to_index["proposer_quality"]] = float(quality[row])
            output[name_to_index["proposer_alias_probability"]] = float(alias[row])
            output[name_to_index["proposer_spike_rate"]] = float(spike[row])
            expected_delta = value - float(expected[row])
            map_delta = value - float(map_bpm[row])
            output[name_to_index["candidate_minus_expected_bpm"]] = expected_delta
            output[name_to_index["candidate_abs_expected_distance_bpm"]] = abs(
                expected_delta
            )
            output[name_to_index["candidate_minus_map_bpm"]] = map_delta
            output[name_to_index["candidate_abs_map_distance_bpm"]] = abs(map_delta)
            output[name_to_index["proposer_expected_anchor_match"]] = float(
                abs(expected_delta)
                <= float(candidates.merge_radius_bpm) + 1.0e-12
            )
            output[name_to_index["proposer_map_anchor_match"]] = float(
                abs(map_delta) <= float(candidates.merge_radius_bpm) + 1.0e-12
            )
            output[
                [
                    name_to_index["proposer_radar1_weight"],
                    name_to_index["proposer_radar2_weight"],
                    name_to_index["proposer_radar3_weight"],
                ]
            ] = radar_weights[row]
    if not np.isfinite(features[candidate_mask]).all():
        raise RuntimeError("proposer candidate feature construction is non-finite")
    features *= candidate_mask[..., None]
    return features, PROPOSER_NODE_FEATURE_NAMES


def _validate_root_manifests(
    rf_cache: Path, svd_cache: Path
) -> tuple[dict[str, Any], dict[str, Any], list[str]]:
    rf_path = rf_cache / "manifest.json"
    svd_path = svd_cache / "manifest.json"
    rf = _load_json(rf_path)
    svd = _load_json(svd_path)
    rf_sessions = _successful_sessions(rf, "RF")
    svd_sessions = _successful_sessions(svd, "SVD")
    if rf_sessions != svd_sessions:
        raise RuntimeError("RF/SVD successful session order differs")
    if bool(svd.get("valid_only", True)):
        raise RuntimeError("SVD cache is not all-window")
    if svd.get("label_inputs", []):
        raise RuntimeError("SVD cache declares label-derived inputs")
    if int(svd.get("components", -1)) < 6:
        raise RuntimeError("SVD cache has fewer than six components")
    names = tuple(svd.get("variant_names", ()))
    if tuple(names[index] for index in VERIFIED_SVD_VARIANT_INDICES) != tuple(
        VERIFIED_SVD_VARIANT_NAMES
    ):
        raise RuntimeError("SVD verified variant names/order mismatch")
    if str(svd.get("canonical_manifest_sha256", "")) != sha256_file(rf_path):
        raise RuntimeError("SVD cache is not bound to the requested RF manifest")
    return rf, svd, rf_sessions


def _acquisition_v2_root_contract(
    rf_manifest: Mapping[str, Any], svd_manifest: Mapping[str, Any]
) -> dict[str, Any] | None:
    """Return the common acquisition-v2 contract or fail closed on a mixed pair."""

    rf_contract = rf_manifest.get("acquisition_contract")
    svd_contract = svd_manifest.get("canonical_acquisition_contract")
    rf_v2 = (
        isinstance(rf_contract, Mapping)
        and rf_contract.get("schema_version") == ACQUISITION_V2_SCHEMA
    )
    svd_v2 = (
        isinstance(svd_contract, Mapping)
        and svd_contract.get("schema_version") == ACQUISITION_V2_SCHEMA
    )
    if not rf_v2 and not svd_v2:
        return None
    if not rf_v2 or not svd_v2:
        raise RuntimeError(
            "RF/SVD acquisition-v2 root contract presence differs"
        )
    if dict(rf_contract) != dict(svd_contract):
        raise RuntimeError("RF/SVD acquisition-v2 root contracts differ")
    mode = rf_contract.get("mode")
    if mode not in {"strict", "diagnostic"}:
        raise RuntimeError("acquisition-v2 root mode is invalid")
    if type(rf_contract.get("scientific_eligible")) is not bool:
        raise RuntimeError("acquisition-v2 scientific eligibility is invalid")
    if mode == "diagnostic" and rf_contract.get("scientific_eligible") is not False:
        raise RuntimeError("diagnostic acquisition-v2 input claims scientific eligibility")
    return dict(rf_contract)


def _load_bound_timing_mask(
    session_dir: Path,
    manifest: Mapping[str, Any],
    *,
    rows: int,
    label: str,
    expected_contract: Mapping[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    """Load one manifest-bound ``[row,view,interval]`` structural mask."""

    if manifest.get("content_sha256") != _canonical_digest(
        manifest, exclude="content_sha256"
    ):
        raise RuntimeError(f"{label} acquisition-v2 session manifest hash mismatch")
    inventory = manifest.get("file_inventory")
    if not isinstance(inventory, Mapping):
        raise RuntimeError(f"{label} acquisition-v2 file inventory is missing")
    if manifest.get("inventory_sha256") != _canonical_digest(inventory):
        raise RuntimeError(f"{label} acquisition-v2 file inventory hash mismatch")
    declared = inventory.get("radar_timing_valid_mask")
    if not isinstance(declared, Mapping):
        raise RuntimeError(f"{label} acquisition-v2 radar timing mask is missing")
    if declared.get("path") != "radar_timing_valid_mask.npy":
        raise RuntimeError(f"{label} acquisition-v2 radar timing mask path is invalid")
    path = session_dir / "radar_timing_valid_mask.npy"
    try:
        binding = _file_binding(path)
    except FileNotFoundError as exc:
        raise RuntimeError(
            f"{label} acquisition-v2 radar timing mask file is missing"
        ) from exc
    if (
        declared.get("sha256") != binding["sha256"]
        or declared.get("bytes") != binding["bytes"]
    ):
        raise RuntimeError(f"{label} acquisition-v2 radar timing mask binding mismatch")
    mask = np.load(path, mmap_mode="r", allow_pickle=False)
    if (
        mask.dtype != np.bool_
        or mask.ndim != 3
        or mask.shape[0] != int(rows)
        or mask.shape[1] != 3
        or mask.shape[2] <= 0
    ):
        raise RuntimeError(
            f"{label} acquisition-v2 radar timing mask shape/dtype is invalid"
        )
    if declared.get("shape") != list(mask.shape) or declared.get("dtype") != "bool":
        raise RuntimeError(
            f"{label} acquisition-v2 radar timing mask declaration mismatch"
        )
    if manifest.get("radar_timing_valid_mask_shape") != list(mask.shape):
        raise RuntimeError(
            f"{label} acquisition-v2 radar timing mask shape claim mismatch"
        )
    invalid = int(mask.size - np.count_nonzero(mask))
    if manifest.get("radar_timing_invalid_interval_count") != invalid:
        raise RuntimeError(
            f"{label} acquisition-v2 radar timing mask count claim mismatch"
        )
    if manifest.get("radar_timing_mask_contract") != dict(expected_contract):
        raise RuntimeError(f"{label} acquisition-v2 radar timing mask contract mismatch")
    return np.array(mask, dtype=np.bool_, copy=True), binding


def _load_acquisition_v2_timing_masks(
    rf_cache: Path,
    svd_cache: Path,
    sessions: Sequence[str],
    root_contract: Mapping[str, Any],
) -> tuple[dict[str, np.ndarray], dict[str, dict[str, Any]]]:
    """Validate equal RF/SVD interval masks and reduce them to view availability."""

    view_masks: dict[str, np.ndarray] = {}
    records: dict[str, dict[str, Any]] = {}
    for session_id in sessions:
        rf_dir = rf_cache / session_id
        svd_dir = svd_cache / session_id
        rf_manifest = _load_json(rf_dir / "manifest.json")
        svd_manifest = _load_json(svd_dir / "manifest.json")
        rf_binding = rf_manifest.get("acquisition_contract")
        svd_binding = svd_manifest.get("canonical_acquisition_binding")
        if (
            not isinstance(rf_binding, Mapping)
            or not isinstance(svd_binding, Mapping)
            or rf_binding.get("schema_version") != ACQUISITION_V2_SCHEMA
            or svd_binding.get("schema_version") != ACQUISITION_V2_SCHEMA
        ):
            raise RuntimeError(
                f"RF/SVD acquisition-v2 session binding is missing: {session_id}"
            )
        if dict(rf_binding) != dict(svd_binding):
            raise RuntimeError(
                f"RF/SVD acquisition-v2 session bindings differ: {session_id}"
            )
        rf_metadata = pd.read_csv(rf_dir / "metadata.csv")
        svd_metadata = pd.read_csv(svd_dir / "metadata.csv")
        _assert_common_rows(rf_metadata, svd_metadata, f"RF/SVD {session_id}")
        rf_mask, rf_file = _load_bound_timing_mask(
            rf_dir,
            rf_manifest,
            rows=len(rf_metadata),
            label=f"RF {session_id}",
            expected_contract=RF_TIMING_MASK_CONTRACT,
        )
        svd_mask, svd_file = _load_bound_timing_mask(
            svd_dir,
            svd_manifest,
            rows=len(svd_metadata),
            label=f"SVD {session_id}",
            expected_contract=SVD_TIMING_MASK_CONTRACT,
        )
        if rf_mask.shape != svd_mask.shape or not np.array_equal(rf_mask, svd_mask):
            raise RuntimeError(
                f"RF/SVD acquisition-v2 radar timing masks differ: {session_id}"
            )
        if root_contract.get("scientific_eligible") is True and not bool(rf_mask.all()):
            raise RuntimeError(
                f"scientific acquisition-v2 input has invalid radar timing: {session_id}"
            )
        # A view is structurally available only when every right-edge interval
        # in the exact half-open window support is valid.  Numeric evidence is
        # deliberately not consulted for this reduction.
        per_view = np.all(rf_mask, axis=2)
        view_masks[session_id] = per_view.astype(np.bool_, copy=False)
        records[session_id] = {
            "shape": list(rf_mask.shape),
            "interval_reduction": "all_exact_half_open_window_intervals",
            "invalid_interval_count": int(rf_mask.size - np.count_nonzero(rf_mask)),
            "unavailable_view_count": int(per_view.size - np.count_nonzero(per_view)),
            "rf_sha256": rf_file["sha256"],
            "svd_sha256": svd_file["sha256"],
            "rf_svd_semantically_equal": True,
        }
    if root_contract.get("mode") == "diagnostic" and root_contract.get(
        "scientific_eligible"
    ) is not False:
        raise RuntimeError("diagnostic acquisition-v2 timing masks became trainable")
    return view_masks, records


def _zero_unavailable_radar_features(
    values: np.ndarray,
    feature_names: Sequence[str],
    radar_mask: np.ndarray,
) -> None:
    """Force every radar-specific feature cell to exact zero when unavailable."""

    available = np.asarray(radar_mask, dtype=bool)
    if available.shape != (values.shape[0], 3):
        raise RuntimeError("batch structural radar mask shape mismatch")
    names = tuple(str(name) for name in feature_names)
    for radar in range(3):
        prefixes = (
            f"rf_radar{radar + 1}_",
            f"svd_radar{radar + 1}_",
            f"proposer_radar{radar + 1}_",
        )
        columns = np.asarray(
            [index for index, name in enumerate(names) if name.startswith(prefixes)],
            dtype=np.int64,
        )
        if columns.size == 0:
            raise RuntimeError(f"radar {radar + 1} evidence feature columns are missing")
        unavailable = ~available[:, radar]
        if unavailable.any():
            selected = np.ix_(unavailable, np.arange(values.shape[1]), columns)
            values[selected] = 0.0
            if np.count_nonzero(values[selected]):
                raise RuntimeError("structurally unavailable radar features are nonzero")


def _node_structural_availability(
    feature_names: Sequence[str],
    batch: Any,
    *,
    proposer_row_available: np.ndarray,
    classical_row_available: np.ndarray,
) -> np.ndarray:
    """Build the exact per-cell mask used to create this feature batch."""

    names = tuple(str(name) for name in feature_names)
    candidate = np.asarray(batch.candidates.mask)
    radar = np.asarray(batch.rf_support.radar_mask)
    if candidate.dtype != np.bool_ or radar.dtype != np.bool_:
        raise RuntimeError("candidate/radar structural masks must be boolean")
    if radar.shape != (candidate.shape[0], 3):
        raise RuntimeError("batch radar structural mask shape drifted")
    proposer = np.asarray(proposer_row_available)
    if proposer.dtype != np.bool_ or proposer.shape != (candidate.shape[0],):
        raise RuntimeError("proposer row availability mask shape drifted")
    classical = np.asarray(classical_row_available)
    if classical.dtype != np.bool_ or classical.shape != (candidate.shape[0],):
        raise RuntimeError("classical row availability mask shape drifted")
    rf_mask = np.asarray(batch.rf_support.mask)
    svd_mask = np.asarray(batch.svd_support.mask)
    if rf_mask.dtype != np.bool_ or svd_mask.dtype != np.bool_:
        raise RuntimeError("RF/SVD support masks must be boolean")
    availability = np.broadcast_to(
        candidate[..., None], (*candidate.shape, len(names))
    ).copy()
    previous_candidate = np.zeros_like(candidate)
    next_candidate = np.zeros_like(candidate)
    previous_candidate[:, 1:] = candidate[:, 1:] & candidate[:, :-1]
    next_candidate[:, :-1] = candidate[:, :-1] & candidate[:, 1:]
    source_mask = np.asarray(batch.candidates.source_mask, dtype=np.bool_)
    if source_mask.shape[:2] != candidate.shape:
        raise RuntimeError("candidate source structural mask shape drifted")
    confidence_available = (
        source_mask[
            ...,
            [int(CandidateSource.BASE), int(CandidateSource.DIRECT_MODE)],
        ].any(axis=2)
        & proposer[:, None]
    )
    confidence_available |= (
        source_mask[
            ...,
            int(CandidateSource.CLASSICAL_X1) : int(CandidateSource.CLASSICAL_X4)
            + 1,
        ].any(axis=2)
        & classical[:, None]
    )
    for radar_index in range(3):
        confidence_available |= (
            source_mask[..., int(CandidateSource.RADAR_PEAK_1) + radar_index]
            & radar[:, None, radar_index]
            & classical[:, None]
        )
    proposer_names = set(PROPOSER_NODE_FEATURE_NAMES)
    proposer_source_names = {
        "source_base",
        "source_direct_mode",
        "source_confidence_base",
        "source_confidence_direct_mode",
    }
    for feature_index, name in enumerate(names):
        if name == "previous_candidate_gap_bpm":
            availability[..., feature_index] = previous_candidate
            continue
        if name == "next_candidate_gap_bpm":
            availability[..., feature_index] = next_candidate
            continue
        if name == "candidate_confidence":
            availability[..., feature_index] = confidence_available
            continue
        matched = False
        for radar_index in range(3):
            for ratio_index, token in enumerate(_HARMONIC_RATIO_TOKENS):
                rf_prefix = f"rf_radar{radar_index + 1}_{token}_"
                svd_prefix = f"svd_radar{radar_index + 1}_{token}_"
                if name.startswith(rf_prefix):
                    cell = rf_mask[:, :, radar_index, ratio_index].copy()
                    if "_candidate_iq_phase_power_" in name:
                        cell.fill(False)
                    elif name.endswith("_cross_radar_consensus"):
                        peer_indices = [
                            index for index in range(3) if index != radar_index
                        ]
                        cell &= rf_mask[
                            :, :, peer_indices, ratio_index
                        ].any(axis=2)
                    availability[..., feature_index] = cell
                    matched = True
                    break
                if name.startswith(svd_prefix):
                    availability[..., feature_index] = svd_mask[
                        :, :, radar_index, ratio_index
                    ]
                    matched = True
                    break
            if matched:
                break
        if matched:
            continue
        if name.startswith(("rf_", "svd_")):
            raise RuntimeError(f"unrecognized radar feature schema: {name}")
        if name in proposer_names:
            availability[..., feature_index] &= proposer[:, None]
        if name in proposer_source_names:
            availability[..., feature_index] &= proposer[:, None]
        if name.startswith(("source_classical_", "source_confidence_classical_")):
            availability[..., feature_index] &= classical[:, None]
        for radar_index in range(3):
            radar_owned = {
                f"source_radar_peak_{radar_index + 1}",
                f"proposer_radar{radar_index + 1}_weight",
            }
            if name in radar_owned:
                availability[..., feature_index] &= radar[:, None, radar_index]
                break
            if name == f"source_confidence_radar_peak_{radar_index + 1}":
                availability[..., feature_index] &= (
                    radar[:, None, radar_index] & classical[:, None]
                )
                break
    return np.asarray(availability, dtype=np.bool_)


def _resolve_session_joint_radar_mask(
    rf_maps: np.ndarray,
    svd_spectra: np.ndarray,
    svd_attributes: np.ndarray,
    *,
    explicit_timing_mask: np.ndarray | None,
    svd_components: int,
    batch_size: int,
) -> np.ndarray:
    """Resolve one RF+SVD availability mask before candidate admission.

    This intentionally runs before the candidate bank so an unavailable SVD
    view can never survive as a radar-peak source.  Chunking preserves the
    cache builder's bounded-memory contract.
    """

    rows = int(np.asarray(rf_maps).shape[0])
    if not (
        np.asarray(svd_spectra).shape[0]
        == np.asarray(svd_attributes).shape[0]
        == rows
    ):
        raise RuntimeError("RF/SVD session evidence row counts differ")
    if explicit_timing_mask is not None:
        timing = np.asarray(explicit_timing_mask)
        if timing.dtype != np.bool_ or timing.shape != (rows, 3):
            raise RuntimeError("session timing-derived radar mask is invalid")
    else:
        timing = None
    resolved = np.zeros((rows, 3), dtype=np.bool_)
    for start in range(0, rows, int(batch_size)):
        stop = min(rows, start + int(batch_size))
        selected_svd = np.asarray(svd_spectra[start:stop])[
            :, :, VERIFIED_SVD_VARIANT_INDICES, : int(svd_components), :
        ]
        selected_attributes = np.asarray(svd_attributes[start:stop])[
            :, :, VERIFIED_SVD_VARIANT_INDICES, : int(svd_components), :
        ]
        # Only the verified raw-power half of the RF cache may establish
        # availability.  The reserved IQ hypothesis cannot rescue a view.
        raw_rf = np.asarray(rf_maps[start:stop])[..., :91]
        resolved[start:stop] = resolve_joint_radar_mask(
            raw_rf,
            selected_svd,
            svd_attributes=selected_attributes,
            explicit_mask=None if timing is None else timing[start:stop],
        )
    return resolved


def collect_input_bindings(
    rf_cache: Path,
    svd_cache: Path,
    proposer_path: Path,
    folds_path: Path,
    sessions: Sequence[str],
    *,
    acquisition_v2: bool = False,
) -> dict[str, Any]:
    direct_entry_disk_binding = _assert_direct_entry_disk_binding_current()
    result: dict[str, Any] = {
        "source": {
            "builder": direct_entry_disk_binding,
            "snn_rr_package": _file_binding(
                PROJECT_ROOT / "src/snn_rr/__init__.py"
            ),
            "harmonic_set_data": _file_binding(
                PROJECT_ROOT / "src/snn_rr/harmonic_set_data.py"
            ),
            "harmonic_feature_layout_v3r1": _file_binding(
                PROJECT_ROOT / "src/snn_rr/harmonic_feature_layout_v3r1.py"
            ),
        },
        "execution_source_generation": {
            "guard_scope": "initialization_time_direct_entry_disk_only",
            "direct_entry_disk_binding": direct_entry_disk_binding,
            "binds_actual_loader_compiled_bytes": False,
            "complete_private_import_closure": False,
            "scientific_authority_status": (
                "terminal_blocked_without_fresh_isolated_private_source_launcher"
            ),
        },
        "rf_root_manifest": _file_binding(rf_cache / "manifest.json"),
        "svd_root_manifest": _file_binding(svd_cache / "manifest.json"),
        "proposer": _file_binding(proposer_path),
        "fold_assignments": _file_binding(folds_path),
        "sessions": {},
    }
    for session_id in sessions:
        rf_dir = rf_cache / session_id
        svd_dir = svd_cache / session_id
        for label, cache_root, session_dir in (
            ("RF", rf_cache, rf_dir),
            ("SVD", svd_cache, svd_dir),
        ):
            try:
                session_dir.resolve().relative_to(cache_root.resolve())
            except ValueError as error:
                raise RuntimeError(
                    f"{label} session path escapes cache root: {session_id}"
                ) from error
        result["sessions"][session_id] = {
            "rf": {
                name: _file_binding(rf_dir / filename)
                for name, filename in (
                    ("maps", "maps.npy"),
                    ("metadata", "metadata.csv"),
                    ("frequencies", "frequencies_hz.npy"),
                    ("manifest", "manifest.json"),
                    *(
                        (("radar_timing_valid_mask", "radar_timing_valid_mask.npy"),)
                        if acquisition_v2
                        else ()
                    ),
                )
            },
            "svd": {
                name: _file_binding(svd_dir / filename)
                for name, filename in (
                    ("spectra", "spectra.npy"),
                    ("attributes", "attributes.npy"),
                    ("metadata", "metadata.csv"),
                    ("frequencies", "frequencies_hz.npy"),
                    ("manifest", "manifest.json"),
                    *(
                        (("radar_timing_valid_mask", "radar_timing_valid_mask.npy"),)
                        if acquisition_v2
                        else ()
                    ),
                )
            },
        }
    return result


def _verify_reuse(
    output_dir: Path, build_signature: str, input_bindings: Mapping[str, Any]
) -> dict[str, Any]:
    manifest_path = output_dir / "manifest.json"
    if not manifest_path.is_file():
        raise RuntimeError("output exists but has no complete manifest")
    manifest = _load_json(manifest_path)
    if (
        manifest.get("format_version") != FORMAT_VERSION
        or manifest.get("schema") != SCHEMA_ID
        or not manifest.get("complete")
    ):
        raise RuntimeError("output exists but is partial/incompatible")
    if manifest.get("content_sha256") != _canonical_digest(
        manifest, exclude="content_sha256"
    ):
        raise RuntimeError("existing output manifest content hash mismatch")
    if manifest.get("build_signature_sha256") != build_signature:
        raise RuntimeError("existing output was built from different settings or inputs")
    if manifest.get("inputs") != input_bindings:
        raise RuntimeError("existing output input binding mismatch")
    outputs = manifest.get("outputs")
    if not isinstance(outputs, Mapping):
        raise RuntimeError("existing output has no output bindings")
    expected_outputs = {
        **ARRAY_FILES,
        "node_feature_availability": NODE_FEATURE_AVAILABILITY_FILE,
        "metadata": "metadata.csv",
        "feature_names": "feature_names.json",
    }
    if set(outputs) != set(expected_outputs):
        raise RuntimeError("existing output binding inventory is incomplete or unknown")
    for name, filename in expected_outputs.items():
        binding = outputs[name]
        if (
            not isinstance(binding, Mapping)
            or set(binding) != {"filename", "sha256", "bytes"}
            or binding.get("filename") != filename
            or not isinstance(binding.get("sha256"), str)
            or len(str(binding["sha256"])) != 64
            or any(
                character not in "0123456789abcdef"
                for character in str(binding["sha256"])
            )
            or type(binding.get("bytes")) is not int
            or int(binding["bytes"]) <= 0
        ):
            raise RuntimeError(f"existing output binding is invalid: {name}")
        path = output_dir / filename
        if not path.is_file():
            raise RuntimeError(f"existing output file is missing: {name}")
        if path.stat().st_size != int(binding.get("bytes", -1)):
            raise RuntimeError(f"existing output size mismatch: {name}")
        if sha256_file(path) != binding.get("sha256"):
            raise RuntimeError(f"existing output SHA-256 mismatch: {name}")
    availability = np.load(
        output_dir / NODE_FEATURE_AVAILABILITY_FILE,
        mmap_mode="r",
        allow_pickle=False,
    )
    features = np.load(
        output_dir / ARRAY_FILES["node_features"],
        mmap_mode="r",
        allow_pickle=False,
    )
    if (
        features.dtype != np.float32
        or availability.dtype != np.bool_
        or availability.shape != features.shape
        or manifest.get("node_feature_shape") != list(features.shape)
        or manifest.get("node_feature_dtype") != "float32"
        or manifest.get("node_feature_availability_shape")
        != list(availability.shape)
        or manifest.get("node_feature_availability_dtype") != "bool"
    ):
        raise RuntimeError("existing node feature/availability schema drifted")
    for start in range(0, features.shape[0], 256):
        stop = min(start + 256, features.shape[0])
        feature_chunk = np.asarray(features[start:stop])
        availability_chunk = np.asarray(availability[start:stop])
        if np.count_nonzero(feature_chunk[~availability_chunk]):
            raise RuntimeError(
                "existing structurally unavailable node feature is nonzero"
            )
    return {"status": "reused", "output_dir": str(output_dir), "manifest": manifest}


def _open_output_arrays(stage: Path, rows: int) -> dict[str, np.memmap]:
    specifications = {
        "candidate_bpm": (np.float32, (rows, MAX_CANDIDATES)),
        "candidate_mask": (np.bool_, (rows, MAX_CANDIDATES)),
        "candidate_confidence": (np.float32, (rows, MAX_CANDIDATES)),
        "candidate_source_mask": (
            np.bool_,
            (rows, MAX_CANDIDATES, len(CANDIDATE_SOURCE_NAMES)),
        ),
        "candidate_primary_source": (np.int16, (rows, MAX_CANDIDATES)),
        "joint_radar_mask": (np.bool_, (rows, 3)),
        "rf_support_count": (np.uint8, (rows, MAX_CANDIDATES, len(HARMONIC_RATIOS))),
        "svd_support_count": (np.uint8, (rows, MAX_CANDIDATES, len(HARMONIC_RATIOS))),
    }
    return {
        name: np.lib.format.open_memmap(
            stage / ARRAY_FILES[name], mode="w+", dtype=dtype, shape=shape
        )
        for name, (dtype, shape) in specifications.items()
    }


def _flush(arrays: Mapping[str, np.memmap]) -> None:
    for array in arrays.values():
        array.flush()


def _fsync_regular_file(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_CLOEXEC)
    try:
        details = os.fstat(descriptor)
        if not stat.S_ISREG(details.st_mode):
            raise RuntimeError(f"publication payload is not regular: {path}")
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _rename_directory_noreplace(source: Path, destination: Path) -> None:
    """Atomically publish a directory without replacing any existing name."""

    try:
        renameat2 = ctypes.CDLL(None, use_errno=True).renameat2
    except (AttributeError, OSError) as error:
        raise RuntimeError(
            "atomic no-replace directory publication is unavailable"
        ) from error
    renameat2.argtypes = (
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    )
    renameat2.restype = ctypes.c_int
    result = renameat2(
        _AT_FDCWD,
        os.fsencode(source),
        _AT_FDCWD,
        os.fsencode(destination),
        _RENAME_NOREPLACE,
    )
    if result == 0:
        return
    error_number = ctypes.get_errno()
    if error_number in {errno.EEXIST, errno.ENOTEMPTY}:
        raise FileExistsError(
            error_number,
            "harmonic output appeared concurrently; refusing to overwrite",
            str(destination),
        )
    raise OSError(
        error_number,
        os.strerror(error_number),
        f"{source} -> {destination}",
    )


def _rename_directory_noreplace_at(
    parent_descriptor: int,
    source_name: str,
    destination_name: str,
) -> None:
    try:
        renameat2 = ctypes.CDLL(None, use_errno=True).renameat2
    except (AttributeError, OSError) as error:
        raise RuntimeError(
            "atomic no-replace directory publication is unavailable"
        ) from error
    renameat2.argtypes = (
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    )
    renameat2.restype = ctypes.c_int
    result = renameat2(
        parent_descriptor,
        os.fsencode(source_name),
        parent_descriptor,
        os.fsencode(destination_name),
        _RENAME_NOREPLACE,
    )
    if result == 0:
        return
    error_number = ctypes.get_errno()
    if error_number in {errno.EEXIST, errno.ENOTEMPTY}:
        raise FileExistsError(
            error_number,
            "harmonic output appeared concurrently; refusing to overwrite",
            destination_name,
        )
    raise OSError(
        error_number,
        os.strerror(error_number),
        f"{source_name} -> {destination_name}",
    )


def _durably_publish_stage(
    stage_owner: _StablePrivateDirectory,
    output_dir: Path,
) -> None:
    """Persist the exact stage inventory before/after atomic directory rename."""

    stage = stage_owner.path
    expected = {
        *ARRAY_FILES.values(),
        NODE_FEATURE_AVAILABILITY_FILE,
        "metadata.csv",
        "feature_names.json",
        "manifest.json",
    }
    observed = {path.name for path in stage.iterdir()}
    if observed != expected:
        raise RuntimeError(
            "harmonic publication stage inventory is incomplete or unknown"
        )
    for filename in sorted(expected):
        _fsync_regular_file(stage / filename)
    _fsync_directory(stage)
    stage_owner.assert_parent_path_current()
    renamed = False
    try:
        _rename_directory_noreplace_at(
            stage_owner.parent_descriptor,
            stage_owner.name,
            output_dir.name,
        )
        renamed = True
        stage_owner.assert_parent_path_current()
        published = os.stat(
            output_dir.name,
            dir_fd=stage_owner.parent_descriptor,
            follow_symlinks=False,
        )
        if (
            not stat.S_ISDIR(published.st_mode)
            or published.st_dev != stage_owner.root_device
            or published.st_ino != stage_owner.root_inode
        ):
            raise RuntimeError("published harmonic directory inode mismatch")
        _fsync_directory(output_dir.parent)
        stage_owner.assert_parent_path_current()
        os.fsync(stage_owner.parent_descriptor)
    except BaseException:
        if renamed:
            try:
                published = os.stat(
                    output_dir.name,
                    dir_fd=stage_owner.parent_descriptor,
                    follow_symlinks=False,
                )
                if (
                    not stat.S_ISDIR(published.st_mode)
                    or published.st_dev != stage_owner.root_device
                    or published.st_ino != stage_owner.root_inode
                ):
                    raise RuntimeError(
                        "cannot safely roll back mismatched harmonic publication"
                    )
                _rename_directory_noreplace_at(
                    stage_owner.parent_descriptor,
                    output_dir.name,
                    stage_owner.name,
                )
                os.fsync(stage_owner.parent_descriptor)
            except BaseException as rollback_error:
                raise RuntimeError(
                    "harmonic publication parent changed and rollback failed"
                ) from rollback_error
        raise
    stage_owner.mark_published()


def _output_bindings(stage: Path) -> dict[str, Any]:
    names = {
        **ARRAY_FILES,
        "node_feature_availability": NODE_FEATURE_AVAILABILITY_FILE,
        "metadata": "metadata.csv",
        "feature_names": "feature_names.json",
    }
    return {
        name: {
            "filename": filename,
            "sha256": sha256_file(stage / filename),
            "bytes": (stage / filename).stat().st_size,
        }
        for name, filename in names.items()
    }


def _build_with_snapshot_owner(
    args: argparse.Namespace,
    *,
    snapshot_owner: list[_BoundInputSnapshot],
) -> dict[str, Any]:
    rf_cache = args.rf_cache.expanduser().resolve()
    svd_cache = args.svd_cache.expanduser().resolve()
    proposer_path = args.proposer.expanduser().resolve()
    folds_path = args.fold_assignments.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    proposal_selection = str(getattr(args, "proposal_selection", "topk"))
    suppression_bpm = float(
        getattr(args, "posterior_nms_suppression_bpm", 1.25)
    )
    base_proposals = str(getattr(args, "base_proposals", "none"))
    include_proposer_features = bool(
        getattr(args, "proposer_features", False)
    )
    svd_components = int(getattr(args, "svd_components", 6))
    if args.batch_size < 1 or args.merge_radius_bpm < 0:
        raise ValueError("batch size must be positive and merge radius non-negative")
    if proposal_selection not in ("topk", "posterior-nms"):
        raise ValueError("proposal selection must be topk or posterior-nms")
    if base_proposals not in BASE_PROPOSAL_CHOICES:
        raise ValueError(f"base proposals must be one of {BASE_PROPOSAL_CHOICES}")
    if not np.isfinite(suppression_bpm) or suppression_bpm < 0:
        raise ValueError("posterior NMS suppression must be finite and non-negative")
    if svd_components not in (6, 12):
        raise ValueError("svd components must be 6 or 12")
    root_manifest_bindings = {
        "rf": _file_binding(rf_cache / "manifest.json"),
        "svd": _file_binding(svd_cache / "manifest.json"),
    }
    rf_root, svd_root, sessions = _validate_root_manifests(rf_cache, svd_cache)
    if root_manifest_bindings != {
        "rf": _file_binding(rf_cache / "manifest.json"),
        "svd": _file_binding(svd_cache / "manifest.json"),
    }:
        raise RuntimeError("root manifest changed while it was being parsed")
    acquisition_contract = _acquisition_v2_root_contract(rf_root, svd_root)
    # Bind every input before copying timing masks or loading the proposer.
    # Replaying this inventory immediately after those loads closes the window
    # in which a replacement could otherwise become the recorded provenance
    # while stale mask bytes remained in memory.
    try:
        input_bindings = collect_input_bindings(
            rf_cache,
            svd_cache,
            proposer_path,
            folds_path,
            sessions,
            acquisition_v2=acquisition_contract is not None,
        )
    except FileNotFoundError as error:
        missing = Path(str(error.args[0])) if error.args else Path("")
        if missing.name == "radar_timing_valid_mask.npy":
            raise RuntimeError(
                "acquisition-v2 radar timing mask file is missing"
            ) from error
        raise
    if (
        input_bindings.get("rf_root_manifest") != root_manifest_bindings["rf"]
        or input_bindings.get("svd_root_manifest") != root_manifest_bindings["svd"]
    ):
        raise RuntimeError("root manifest changed before full input binding")
    if collect_input_bindings(
        rf_cache,
        svd_cache,
        proposer_path,
        folds_path,
        sessions,
        acquisition_v2=acquisition_contract is not None,
    ) != input_bindings:
        raise RuntimeError("harmonic cache input changed while initial inputs loaded")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    input_snapshot = _materialize_bound_input_snapshot(
        rf_cache=rf_cache,
        svd_cache=svd_cache,
        proposer_path=proposer_path,
        folds_path=folds_path,
        sessions=sessions,
        input_bindings=input_bindings,
        parent=output_dir.parent,
    )
    snapshot_owner.append(input_snapshot)
    consumed_rf_cache = input_snapshot.rf_cache
    consumed_svd_cache = input_snapshot.svd_cache
    consumed_proposer_path = input_snapshot.proposer
    consumed_folds_path = input_snapshot.folds
    snapshot_rf_root, snapshot_svd_root, snapshot_sessions = (
        _validate_root_manifests(consumed_rf_cache, consumed_svd_cache)
    )
    if snapshot_sessions != sessions:
        raise RuntimeError("snapshotted root manifest session order drifted")
    if snapshot_rf_root != rf_root or snapshot_svd_root != svd_root:
        raise RuntimeError("snapshotted root manifest semantics drifted")
    acquisition_timing_masks: dict[str, np.ndarray] = {}
    acquisition_timing_records: dict[str, dict[str, Any]] = {}
    if acquisition_contract is not None:
        acquisition_timing_masks, acquisition_timing_records = (
            _load_acquisition_v2_timing_masks(
                consumed_rf_cache,
                consumed_svd_cache,
                sessions,
                acquisition_contract,
            )
        )
    if collect_input_bindings(
        rf_cache,
        svd_cache,
        proposer_path,
        folds_path,
        sessions,
        acquisition_v2=acquisition_contract is not None,
    ) != input_bindings:
        raise RuntimeError("harmonic cache input changed while initial inputs loaded")
    if int(svd_root.get("components", -1)) < svd_components:
        raise RuntimeError(
            f"SVD cache contains fewer than the requested {svd_components} components"
        )
    folds = _fold_map(consumed_folds_path)
    proposer = _load_proposer(consumed_proposer_path)
    input_snapshot.assert_private_bytes_current()
    proposer_frame = _proposer_frame(proposer)
    proposal_bundle = _proposal_bundle(
        proposer,
        selection=proposal_selection,
        suppression_bpm=suppression_bpm,
        base_proposals=base_proposals,
        include_features=include_proposer_features,
    )
    if len(proposer_frame) != len(proposal_bundle.bpm):
        raise RuntimeError("proposer semantic/proposal row count mismatch")
    cache_index = pd.to_numeric(
        proposer_frame["cache_index"], errors="raise"
    ).to_numpy(np.int64)
    if not np.array_equal(cache_index, np.arange(len(cache_index), dtype=np.int64)):
        raise RuntimeError("proposer cache_index must be canonical global order 0..N-1")
    if proposer_frame["cache_index"].duplicated().any():
        raise RuntimeError("proposer cache_index is not unique")
    expected_fold = np.asarray(
        [folds.get(str(identity), -1) for identity in proposer_frame["identity"]],
        dtype=np.int16,
    )
    if np.any(expected_fold < 0) or not np.array_equal(
        proposer_frame["fold"].to_numpy(np.int16), expected_fold
    ):
        raise RuntimeError("proposer fold ownership differs from fold assignments")
    if set(map(str, proposer_frame["identity"])) != set(folds):
        raise RuntimeError("fold assignments/proposer identity cover mismatch")

    settings = {
        "format_version": FORMAT_VERSION,
        "schema": SCHEMA_ID,
        "merge_radius_bpm": float(args.merge_radius_bpm),
        "batch_size": int(args.batch_size),
        "maximum_candidates": MAX_CANDIDATES,
        "proposer_topk": TOPK_PROPOSALS,
        "proposal_selection": proposal_selection,
        "posterior_nms_suppression_bpm": suppression_bpm,
        "posterior_grid_input_key": proposal_bundle.posterior_grid_input_key,
        "base_proposals": base_proposals,
        "proposal_priority": [
            *( ["base_expected"] if base_proposals in ("expected", "expected-map") else [] ),
            *( ["base_map"] if base_proposals in ("map", "expected-map") else [] ),
            f"{TOPK_PROPOSALS}_{proposal_selection}_direct_modes",
            "classical_x1_x2_x3_x4",
            "radar_peaks_1_2_3",
        ],
        "proposer_features": include_proposer_features,
        "proposer_feature_names": (
            list(PROPOSER_NODE_FEATURE_NAMES) if include_proposer_features else []
        ),
        "harmonic_ratios": list(HARMONIC_RATIOS),
        "rf_branch_policy": "raw_power_only_phase_columns_zeroed",
        "verified_svd_variant_indices": list(VERIFIED_SVD_VARIANT_INDICES),
        "svd_components": svd_components,
        "radar_availability_policy": (
            "acquisition_v2_explicit_rf_svd_equal_timing_mask_all_intervals"
            if acquisition_contract is not None
            else "legacy_numeric_evidence_fallback"
        ),
        "classical_availability_policy": (
            "explicit_all_three_joint_radar_views_true"
        ),
        "proposer_availability_policy": (
            "upstream_available_and_all_three_joint_radar_views_true"
            if acquisition_contract is not None
            else "legacy_upstream_proposer_availability"
        ),
        "input_consumption_policy": (
            "rf_svd_proposer_fold_payloads_private_cow_or_stream_snapshot_"
            "hash_equal_bound_bytes_with_original_namespace_publication_barrier"
        ),
    }
    build_signature = _canonical_digest({"settings": settings, "inputs": input_bindings})
    if output_dir.exists():
        result = _verify_reuse(output_dir, build_signature, input_bindings)
        input_snapshot.assert_private_bytes_current()
        if collect_input_bindings(
            rf_cache,
            svd_cache,
            proposer_path,
            folds_path,
            sessions,
            acquisition_v2=acquisition_contract is not None,
        ) != input_bindings:
            raise RuntimeError(
                "harmonic cache input changed while verified output was reused"
            )
        return result

    stage_owner = _StablePrivateDirectory.create(
        output_dir.parent,
        prefix=f".{output_dir.name}.building.",
    )
    stage = stage_owner.path
    arrays: dict[str, np.memmap] = {}
    node_features: np.memmap | None = None
    node_feature_availability: np.memmap | None = None
    feature_names: tuple[str, ...] | None = None
    metadata_parts: list[pd.DataFrame] = []
    session_records: list[dict[str, Any]] = []
    offset = 0
    try:
        arrays = _open_output_arrays(stage, len(proposer_frame))
        for session_id in sessions:
            rf_dir = consumed_rf_cache / session_id
            svd_dir = consumed_svd_cache / session_id
            rf_metadata = pd.read_csv(rf_dir / "metadata.csv")
            svd_metadata = pd.read_csv(svd_dir / "metadata.csv")
            _assert_common_rows(rf_metadata, svd_metadata, f"RF/SVD {session_id}")
            local_rows = len(svd_metadata)
            local_index = pd.to_numeric(
                svd_metadata["cache_index"], errors="raise"
            ).to_numpy(np.int64)
            if not np.array_equal(local_index, np.arange(offset, offset + local_rows)):
                raise RuntimeError(f"SVD cache_index is not contiguous for {session_id}")
            local_fold = np.asarray(
                [folds.get(str(identity), -1) for identity in svd_metadata["identity"]],
                dtype=np.int16,
            )
            if np.any(local_fold < 0):
                raise RuntimeError(f"SVD identity has no fold owner in {session_id}")
            rf_semantic = _normalized_semantic_frame(rf_metadata, local_index, local_fold)
            svd_semantic = _normalized_semantic_frame(svd_metadata, local_index, local_fold)
            if semantic_row_binding_sha256(rf_semantic) != semantic_row_binding_sha256(
                svd_semantic
            ):
                raise RuntimeError(f"RF/SVD semantic row binding mismatch: {session_id}")
            proposer_local = proposer_frame.iloc[offset : offset + local_rows]
            if semantic_row_binding_sha256(svd_semantic) != semantic_row_binding_sha256(
                proposer_local
            ):
                raise RuntimeError(f"SVD/proposer semantic row binding mismatch: {session_id}")

            rf_maps = np.load(rf_dir / "maps.npy", mmap_mode="r", allow_pickle=False)
            rf_frequency = np.load(rf_dir / "frequencies_hz.npy", allow_pickle=False)
            svd_spectra = np.load(
                svd_dir / "spectra.npy", mmap_mode="r", allow_pickle=False
            )
            svd_attributes = np.load(
                svd_dir / "attributes.npy", mmap_mode="r", allow_pickle=False
            )
            svd_frequency = np.load(
                svd_dir / "frequencies_hz.npy", allow_pickle=False
            )
            if not (
                rf_maps.shape[0]
                == svd_spectra.shape[0]
                == svd_attributes.shape[0]
                == local_rows
            ):
                raise RuntimeError(f"evidence row shape mismatch: {session_id}")
            if rf_maps.ndim != 4 or rf_maps.shape[-1] != 182:
                raise RuntimeError(f"RF branch/range layout is incompatible: {session_id}")
            if acquisition_contract is None:
                timing_radar_mask = None
            else:
                timing_radar_mask = acquisition_timing_masks[session_id]
                if timing_radar_mask.shape != (local_rows, 3):
                    raise RuntimeError(
                        f"acquisition-v2 view mask shape mismatch: {session_id}"
                    )
            joint_radar_mask = _resolve_session_joint_radar_mask(
                rf_maps,
                svd_spectra,
                svd_attributes,
                explicit_timing_mask=timing_radar_mask,
                svd_components=svd_components,
                batch_size=int(args.batch_size),
            )

            selector = slice(offset, offset + local_rows)
            joint_three_available = np.all(joint_radar_mask, axis=1)
            upstream_proposer_available = np.asarray(
                proposal_bundle.availability[selector], dtype=np.bool_
            )
            effective_proposer_available = (
                upstream_proposer_available & joint_three_available
                if acquisition_contract is not None
                else upstream_proposer_available
            )
            effective_proposal_mask = np.asarray(
                proposal_bundle.mask[selector], dtype=np.bool_
            ) & effective_proposer_available[:, None]
            bank = candidate_bank_from_metadata(
                rf_metadata,
                proposal_bpm=proposal_bundle.bpm[selector],
                proposal_confidence=proposal_bundle.confidence[selector],
                proposal_mask=effective_proposal_mask,
                proposal_source=proposal_bundle.source[selector],
                radar_mask=joint_radar_mask,
                # The fused classical estimate has no independent upstream
                # timing receipt.  Admit it only when all three contributing
                # radar views are structurally authorized for the row.
                classical_available=joint_three_available,
                merge_radius_bpm=float(args.merge_radius_bpm),
                max_candidates=MAX_CANDIDATES,
            )
            base_entered = np.asarray(bank.source_mask)[
                ..., int(CandidateSource.BASE)
            ].any()
            if base_proposals == "none" and base_entered:
                raise RuntimeError("implicit BASE source entered the direct-mode candidate bank")
            if base_proposals != "none":
                expected_base_rows = effective_proposer_available
                observed_base_rows = np.asarray(bank.source_mask)[
                    ..., int(CandidateSource.BASE)
                ].any(axis=1)
                if not np.array_equal(observed_base_rows, expected_base_rows):
                    raise RuntimeError("explicit BASE proposer candidate availability mismatch")
            arrays["candidate_bpm"][selector] = bank.bpm
            arrays["candidate_mask"][selector] = bank.mask
            arrays["candidate_confidence"][selector] = bank.confidence
            arrays["candidate_source_mask"][selector] = bank.source_mask
            arrays["candidate_primary_source"][selector] = bank.primary_source

            proposer_nodes: np.ndarray | None = None
            proposer_names: tuple[str, ...] | None = None
            if include_proposer_features:
                proposer_nodes, proposer_names = proposer_candidate_node_features(
                    proposal_bundle,
                    bank,
                    selector,
                    row_available=effective_proposer_available,
                )

            session_feature_cursor = 0
            for batch in iter_compact_node_feature_batches(
                rf_maps,
                rf_frequency,
                svd_spectra,
                svd_attributes,
                svd_frequency,
                bank,
                explicit_radar_mask=joint_radar_mask,
                ratios=HARMONIC_RATIOS,
                batch_size=int(args.batch_size),
                svd_components=svd_components,
                proposer_node_features=proposer_nodes,
                proposer_feature_names=proposer_names,
                include_source_confidence=include_proposer_features,
            ):
                if batch.row_slice.step not in (None, 1):
                    raise RuntimeError("feature batch row slice step is not contiguous")
                local_start = int(batch.row_slice.start or 0)
                local_stop = int(batch.row_slice.stop or local_rows)
                if not (
                    local_start == session_feature_cursor
                    and local_start < local_stop <= local_rows
                ):
                    raise RuntimeError(
                        f"feature batches do not exactly cover session rows: {session_id}"
                    )
                session_feature_cursor = local_stop
                start = offset + local_start
                stop = offset + local_stop
                output_slice = slice(start, stop)
                names = tuple(batch.nodes.feature_names)
                values = np.asarray(batch.nodes.features, dtype=np.float32).copy()
                # The second RF branch is a phase-power hypothesis, not verified
                # deployment evidence.  Preserve a stable schema but make every
                # such column exactly zero so no model can learn from it.
                phase_columns = np.asarray(
                    ["_candidate_iq_phase_power_" in name for name in names], dtype=bool
                )
                values[..., phase_columns] = 0.0
                joint = np.asarray(batch.rf_support.radar_mask, dtype=bool) & np.asarray(
                    batch.svd_support.radar_mask, dtype=bool
                )
                if not np.array_equal(
                    joint, joint_radar_mask[local_start:local_stop]
                ):
                    raise RuntimeError("feature batch joint radar mask drifted")
                structural_availability = _node_structural_availability(
                    names,
                    batch,
                    proposer_row_available=np.asarray(
                        effective_proposer_available[local_start:local_stop],
                        dtype=np.bool_,
                    ),
                    classical_row_available=np.asarray(
                        np.all(
                            joint_radar_mask[local_start:local_stop], axis=1
                        ),
                        dtype=np.bool_,
                    ),
                )
                values = np.where(
                    structural_availability, values, np.float32(0.0)
                )
                _zero_unavailable_radar_features(values, names, joint)
                if np.count_nonzero(values[~structural_availability]):
                    raise RuntimeError(
                        "structurally unavailable node feature is nonzero"
                    )
                if not np.isfinite(values).all():
                    raise RuntimeError("node feature construction produced non-finite values")
                if node_features is None:
                    feature_names = names
                    node_features = np.lib.format.open_memmap(
                        stage / ARRAY_FILES["node_features"],
                        mode="w+",
                        dtype=np.float32,
                        shape=(len(proposer_frame), MAX_CANDIDATES, len(names)),
                    )
                    node_feature_availability = np.lib.format.open_memmap(
                        stage / NODE_FEATURE_AVAILABILITY_FILE,
                        mode="w+",
                        dtype=np.bool_,
                        shape=(
                            len(proposer_frame),
                            MAX_CANDIDATES,
                            len(names),
                        ),
                    )
                elif names != feature_names:
                    raise RuntimeError("node feature schema changed between sessions")
                if node_feature_availability is None:
                    raise RuntimeError("node availability output was not initialized")
                node_features[output_slice] = values
                node_feature_availability[output_slice] = structural_availability
                arrays["joint_radar_mask"][output_slice] = joint
                arrays["rf_support_count"][output_slice] = np.asarray(
                    batch.rf_support.mask, dtype=np.uint8
                ).sum(axis=2)
                arrays["svd_support_count"][output_slice] = np.asarray(
                    batch.svd_support.mask, dtype=np.uint8
                ).sum(axis=2)

            if session_feature_cursor != local_rows:
                raise RuntimeError(
                    f"feature batches do not exactly cover session rows: {session_id}"
                )

            canonical_metadata = rf_metadata.copy()
            canonical_metadata.insert(0, "cache_index", local_index)
            canonical_metadata.insert(1, "fold", local_fold)
            metadata_parts.append(canonical_metadata)
            session_records.append(
                {
                    "session_id": session_id,
                    "row_start": offset,
                    "row_stop_exclusive": offset + local_rows,
                    "rows": local_rows,
                    "identity": str(rf_metadata["identity"].iloc[0]),
                    "fold": int(local_fold[0]),
                    "rf_frequency_grid": {
                        "count": len(rf_frequency),
                        "minimum_hz": float(rf_frequency[0]),
                        "maximum_hz": float(rf_frequency[-1]),
                        "sha256": sha256_file(rf_dir / "frequencies_hz.npy"),
                    },
                    "svd_frequency_grid": {
                        "count": len(svd_frequency),
                        "minimum_hz": float(svd_frequency[0]),
                        "maximum_hz": float(svd_frequency[-1]),
                        "sha256": sha256_file(svd_dir / "frequencies_hz.npy"),
                    },
                    **(
                        {
                            "acquisition_v2_radar_timing_mask": (
                                acquisition_timing_records[session_id]
                            )
                        }
                        if acquisition_contract is not None
                        else {}
                    ),
                }
            )
            offset += local_rows

        if (
            offset != len(proposer_frame)
            or node_features is None
            or node_feature_availability is None
            or feature_names is None
        ):
            raise RuntimeError("session construction did not exactly cover proposer rows")
        ordered_feature_names_semantic_sha256: str | None = None
        feature_layout_semantic_sha256: str | None = None
        if include_proposer_features:
            if len(feature_names) != TOTAL_FEATURE_WIDTH:
                raise RuntimeError(
                    "proposer-feature cache must use the canonical 571-wide layout"
                )
            ordered_feature_names_semantic_sha256 = validate_ordered_feature_names(
                feature_names
            )
            feature_layout_semantic_sha256 = FEATURE_LAYOUT_SEMANTIC_SHA256
        metadata = pd.concat(metadata_parts, ignore_index=True)
        if not np.array_equal(metadata["cache_index"].to_numpy(np.int64), cache_index):
            raise RuntimeError("constructed metadata cache_index exact cover failed")
        lineage = semantic_row_binding_sha256(metadata.loc[:, list(SEMANTIC_ROW_FIELDS)])
        if lineage != semantic_row_binding_sha256(proposer_frame):
            raise RuntimeError("final metadata/proposer row lineage mismatch")
        metadata.to_csv(stage / "metadata.csv", index=False)
        _write_json(
            stage / "feature_names.json",
            {
                "node_feature_names": list(feature_names),
                "candidate_source_names": list(CANDIDATE_SOURCE_NAMES),
                "forward_arrays": [
                    "node_features",
                    "node_feature_availability",
                    "candidate_bpm",
                    "candidate_mask",
                    "candidate_confidence",
                    "candidate_source_mask",
                    "joint_radar_mask",
                ],
                "forbidden_target_qc_forward_fields": [
                    *sorted(FORBIDDEN_TARGET_QC_FIELDS),
                ],
                "ordered_feature_names_semantic_sha256": (
                    ordered_feature_names_semantic_sha256
                ),
                "feature_layout_semantic_sha256": feature_layout_semantic_sha256,
                "axis_risk_router_v8r5_compatible": bool(
                    ordered_feature_names_semantic_sha256
                    == EXPECTED_FEATURE_NAMES_SEMANTIC_SHA256
                    and feature_layout_semantic_sha256
                    == FEATURE_LAYOUT_SEMANTIC_SHA256
                ),
            },
        )
        node_features.flush()
        node_feature_availability.flush()
        _flush(arrays)
        input_snapshot.assert_private_bytes_current()
        outputs = _output_bindings(stage)
        current_input_bindings = collect_input_bindings(
            rf_cache,
            svd_cache,
            proposer_path,
            folds_path,
            sessions,
            acquisition_v2=acquisition_contract is not None,
        )
        if current_input_bindings != input_bindings:
            raise RuntimeError(
                "harmonic cache input changed during build; refusing publication"
            )
        manifest: dict[str, Any] = {
            "format_version": FORMAT_VERSION,
            "schema": SCHEMA_ID,
            "complete": True,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "build_signature_sha256": build_signature,
            "row_count": len(metadata),
            "session_count": len(sessions),
            "identity_count": int(metadata["identity"].nunique()),
            "fold_count": int(metadata["fold"].nunique()),
            "node_feature_shape": list(node_features.shape),
            "node_feature_dtype": "float32",
            "node_feature_availability_shape": list(
                node_feature_availability.shape
            ),
            "node_feature_availability_dtype": "bool",
            "ordered_feature_names_semantic_sha256": (
                ordered_feature_names_semantic_sha256
            ),
            "feature_layout_semantic_sha256": feature_layout_semantic_sha256,
            "axis_risk_router_v8r5_compatible": bool(
                ordered_feature_names_semantic_sha256
                == EXPECTED_FEATURE_NAMES_SEMANTIC_SHA256
                and feature_layout_semantic_sha256
                == FEATURE_LAYOUT_SEMANTIC_SHA256
            ),
            "row_lineage_sha256": lineage,
            "settings": settings,
            "candidate_policy": {
                "maximum_candidates": MAX_CANDIDATES,
                "priority": settings["proposal_priority"],
                "merge_radius_bpm": float(args.merge_radius_bpm),
                "merge_anchor_policy": "first_source_anchor_never_moves_then_stable_bpm_sort",
                "proposal_selection": proposal_selection,
                "posterior_nms_suppression_bpm": suppression_bpm,
                "base_source_policy": (
                    "explicit_expected_then_map_before_direct_modes"
                    if base_proposals == "expected-map"
                    else f"explicit_{base_proposals}"
                    if base_proposals != "none"
                    else "none_direct_modes_never_implicit_base"
                ),
                "unavailable_proposer_policy": "no_base_or_direct_candidate_and_all_proposer_features_zero",
            },
            "evidence_policy": {
                "harmonic_ratios": list(HARMONIC_RATIOS),
                "native_frequency_grid_sampling": True,
                "out_of_band_policy": "exact_zero_and_false_mask_never_edge_clamp",
                "structural_availability_policy": (
                    "persist_exact_candidate_radar_native_grid_ratio_branch_"
                    "proposer_and_classical_joint3_mask_and_reapply_after_scaling"
                ),
                "rf_branch_policy": "raw_power_only_phase_feature_columns_exact_zero",
                "rf_range_policy": "preserve_91_raw_range_indices_before_compaction",
                "svd_variant_indices": list(VERIFIED_SVD_VARIANT_INDICES),
                "svd_variant_names": list(VERIFIED_SVD_VARIANT_NAMES),
                "svd_component_policy": (
                    f"first_{svd_components}_components_preserved_before_fixed_width_"
                    "reliability_compaction"
                ),
                "svd_components": svd_components,
                "radar_availability_policy": settings["radar_availability_policy"],
                "classical_availability_policy": settings[
                    "classical_availability_policy"
                ],
                "proposer_availability_policy": settings[
                    "proposer_availability_policy"
                ],
                "input_consumption_policy": settings[
                    "input_consumption_policy"
                ],
                "unavailable_radar_feature_policy": (
                    "radar_peak_sources_and_all_radar_specific_rf_svd_"
                    "proposer_columns_plus_joint3_classical_sources_masked_"
                    "and_exact_zero"
                ),
                "proposer_posterior_feature_policy": (
                    "full_posterior_candidate_local_summaries_plus_exact_row_diagnostics"
                    if include_proposer_features
                    else "disabled_backward_compatible_i1_schema"
                ),
            },
            "model_boundary": {
                "target_qc_excluded_from_candidate_and_feature_construction": True,
                "metadata_is_lineage_and_training_target_storage_not_a_forward_input": True,
                "identity_session_protocol_fold_excluded_from_forward_features": True,
                "proposal_reference_fields_ignored": True,
            },
            "inputs": input_bindings,
            "sessions": session_records,
            "outputs": outputs,
        }
        if acquisition_contract is not None:
            # RF/SVD root documents and an arbitrary proposer NPZ are not a
            # nested-proposer training authorization.  Until a separately
            # versioned verifier binds physical-identity folds, label-free
            # proposer checkpoints/predictions, scaler scope, and source
            # receipts, every acquisition-v2 harmonic output remains
            # diagnostic even when its upstream cache self-declares strict.
            manifest["classification"] = "acquisition_diagnostic"
            manifest["scientific_eligible"] = False
            manifest["trainable"] = False
            manifest["acquisition_v2"] = {
                "schema_version": ACQUISITION_V2_SCHEMA,
                "source_mode": acquisition_contract["mode"],
                "source_contract_sha256": _canonical_digest(acquisition_contract),
                "rf_svd_timing_mask_semantic_equality_required": True,
                "view_availability_reduction": "all_exact_half_open_window_intervals",
                "numeric_payload_cannot_enable_a_masked_view": True,
                "diagnostic_input_trainable": False,
                "scientific_promotion_authority": "absent",
                "scientific_promotion_blocker": (
                    "requires_versioned_nested_label_free_proposer_and_scaler_"
                    "authority_verifier"
                ),
            }
        else:
            # Historical source caches lack a machine-verifiable nested
            # proposer authority.  Preserve rebuild/inspection capability, but
            # do not let a newly generated cache imply that arbitrary NPZ
            # candidates were label-free or identity-disjoint.
            manifest["classification"] = (
                "retrospective_legacy_unverified_proposer"
            )
            manifest["scientific_eligible"] = False
            manifest["trainable"] = False
            manifest["training_blocker"] = (
                "requires_versioned_nested_label_free_proposer_authority"
            )
        manifest["content_sha256"] = _canonical_digest(manifest)
        _write_json(stage / "manifest.json", manifest)
        if output_dir.exists():
            raise RuntimeError("output appeared concurrently; refusing to overwrite")
        # The actual mmap/parse inputs were the immutable private copies; the
        # public namespace must still contain the exact bytes named in the
        # manifest immediately before publication.
        input_snapshot.assert_private_bytes_current()
        if collect_input_bindings(
            rf_cache,
            svd_cache,
            proposer_path,
            folds_path,
            sessions,
            acquisition_v2=acquisition_contract is not None,
        ) != input_bindings:
            raise RuntimeError(
                "harmonic cache input changed before publication"
            )
        _durably_publish_stage(stage_owner, output_dir)
        return {"status": "built", "output_dir": str(output_dir), "manifest": manifest}
    except BaseException:
        for array in arrays.values():
            try:
                array.flush()
            except Exception:
                pass
        if node_features is not None:
            try:
                node_features.flush()
            except Exception:
                pass
        if node_feature_availability is not None:
            try:
                node_feature_availability.flush()
            except Exception:
                pass
        stage_owner.cleanup()
        raise


def build(args: argparse.Namespace) -> dict[str, Any]:
    """Build with unconditional cleanup of every private input snapshot."""

    snapshots: list[_BoundInputSnapshot] = []
    try:
        return _build_with_snapshot_owner(args, snapshot_owner=snapshots)
    finally:
        for snapshot in reversed(snapshots):
            snapshot.cleanup()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rf-cache", type=Path, default=DEFAULT_RF_CACHE)
    parser.add_argument("--svd-cache", type=Path, default=DEFAULT_SVD_CACHE)
    parser.add_argument("--proposer", type=Path, default=DEFAULT_PROPOSER)
    parser.add_argument("--fold-assignments", type=Path, default=DEFAULT_FOLDS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--merge-radius-bpm", type=float, default=0.5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--proposal-selection",
        choices=("topk", "posterior-nms"),
        default="topk",
        help="Use stored top-k modes (i1 compatibility) or stable full-posterior NMS.",
    )
    parser.add_argument(
        "--posterior-nms-suppression-bpm",
        type=float,
        default=1.25,
        help="Inclusive RR separation suppressed after each stable posterior mode.",
    )
    parser.add_argument(
        "--base-proposals",
        choices=BASE_PROPOSAL_CHOICES,
        default="none",
        help="Optional full-posterior expected/MAP anchors, explicitly marked BASE.",
    )
    parser.add_argument(
        "--svd-components",
        type=int,
        choices=(6, 12),
        default=6,
        help="Retain 6 (i1 compatibility) or all 12 cached SVD components.",
    )
    parser.add_argument(
        "--proposer-features",
        action="store_true",
        help="Append validated full-posterior/node-local proposer diagnostics.",
    )
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    result = build(parse_args(argv))
    print(
        json.dumps(
            {
                "status": result["status"],
                "output_dir": result["output_dir"],
                "rows": result["manifest"]["row_count"],
                "build_signature_sha256": result["manifest"]["build_signature_sha256"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
