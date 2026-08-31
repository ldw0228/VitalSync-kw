#!/usr/bin/env python3
"""Build a versioned raw-window SVD component cache.

The cache is deliberately separate from ``artifacts/cache/rf32s`` so an
experimental representation can never silently replace the canonical input.
Feature values read radar samples and timing metadata only.  Reference columns
are copied for downstream evaluation.  Unless ``--all-windows`` is supplied,
the target-derived ``reference_valid`` column selects which rows are emitted;
that supervised row filter is disclosed in every manifest and never enters a
feature value.

Acquisition-cache V3 is a separate diagnostic-only generation.  It requires
the exact full 29-session/9,575-window cache and ``--all-windows``, reconstructs
windows solely from measured radar-relative support, carries explicit timing
reason and view-availability masks, and can never authorize training or a
scientific claim.  BIOPAC bytes are consumed only because the one-shot raw
graph receipt binds every acquisition file; no BIOPAC semantic value or time
mapping enters an SVD feature or row selection.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import ctypes
from datetime import datetime, timezone
import errno
import hashlib
from io import BytesIO
import json
import os
from pathlib import Path
import shutil
import stat
import sys
import tempfile
from typing import Any

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
for search_path in (PROJECT_ROOT, SOURCE_ROOT):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

from scripts.build_features import (  # noqa: E402
    _validate_v3_dataset_catalogue,
    replace_radar_outliers,
)
from snn_rr.acquisition_contract import (  # noqa: E402
    load_acquisition_reconstruction,
    validate_consumption_against_contract,
    validate_raw_input_bindings,
)
from snn_rr.cache import (  # noqa: E402
    ACQUISITION_CACHE_ROOT_SCHEMA_VERSION_V3,
    ACQUISITION_CACHE_SCHEMA_VERSION_V3,
    ACQUISITION_CACHE_SESSION_SCHEMA_VERSION_V3,
    load_feature_cache,
)
from snn_rr.data import build_dataset_manifest, load_xethru_recording  # noqa: E402
from snn_rr.preprocess import causal_block_mean  # noqa: E402
from snn_rr.radar_timing import (  # noqa: E402
    CAUSAL_UNIFORM_INVALID_REASON_SCHEMA_V1,
    CAUSAL_UNIFORM_INVALID_REASON_SEMANTICS_SHA256_V1,
    canonical_ndarray_sha256,
    causal_uniform_resample_radar_views_v1,
)
from snn_rr.raw_snapshot import RawSessionReader, graph_from_subject  # noqa: E402
from snn_rr.svd_features import (  # noqa: E402
    ATTRIBUTE_NAMES,
    DEFAULT_SVD_VARIANTS,
    svd_component_features,
)


PIPELINE_VERSION = 4
V3_SVD_PIPELINE_VERSION = 1
V3_SVD_ROOT_SCHEMA = "snn_rr.svd_component_cache_root.v3.diagnostic"
V3_SVD_SESSION_SCHEMA = "snn_rr.svd_component_cache_session.v3.diagnostic"
V3_SVD_RAW_BINDING_SCHEMA = "snn_rr.svd_component_raw_consumption.v1"
V3_SVD_TIMING_CONTRACT_SCHEMA = "snn_rr.svd_component_timing_support.v1"
V3_SVD_TARGET_FIREWALL_SCHEMA = "snn_rr.svd_component_target_firewall.v1"
V3_EXPECTED_SESSION_COUNT = 29
V3_EXPECTED_ROW_COUNT = 9575
_RENAME_NOREPLACE = 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root", type=Path, default=PROJECT_ROOT / "HAI_EXPERIMENT"
    )
    parser.add_argument(
        "--canonical-cache",
        type=Path,
        default=PROJECT_ROOT / "artifacts/cache/rf32s",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "artifacts/cache/svd_components_v1",
    )
    parser.add_argument("--subjects", nargs="*")
    parser.add_argument("--all-windows", action="store_true")
    parser.add_argument("--components", type=int, default=12)
    parser.add_argument("--nfft", type=int, default=4096)
    parser.add_argument("--n-iter", type=int, default=2)
    parser.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1))
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _atomic_save(path: Path, array: np.ndarray) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.save(handle, array, allow_pickle=False)
    temporary.replace(path)


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _canonical_content_sha256(value: dict[str, Any]) -> str:
    document = dict(value)
    document.pop("content_sha256", None)
    return _canonical_sha256(document)


def _strict_json_payload(payload: bytes, *, label: str) -> dict[str, Any]:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"{label} repeats JSON key {key!r}")
            result[key] = value
        return result

    def reject_nonfinite(value: str) -> None:
        raise ValueError(f"{label} contains non-finite JSON number {value}")

    try:
        document = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=reject_duplicates,
            parse_constant=reject_nonfinite,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot parse {label}: {error}") from error
    if not isinstance(document, dict):
        raise ValueError(f"{label} must be a JSON object")
    return document


def _stable_regular_file_payload(
    path: Path,
    *,
    label: str,
    require_single_link: bool = True,
) -> tuple[bytes, str]:
    required = ("O_NOFOLLOW", "O_CLOEXEC")
    if any(not hasattr(os, name) for name in required):
        raise RuntimeError("V3 SVD production requires secure Linux open flags")
    absolute = Path(os.path.abspath(os.fspath(path)))
    descriptor = os.open(absolute, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or (
            require_single_link and before.st_nlink != 1
        ):
            suffix = " with exactly one hard link" if require_single_link else ""
            raise ValueError(f"{label} must be a regular file{suffix}")
        fingerprint = (
            before.st_dev,
            before.st_ino,
            before.st_mode,
            before.st_nlink,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        )
        chunks: list[bytes] = []
        digest = hashlib.sha256()
        consumed = 0
        while True:
            chunk = os.read(descriptor, 4 * 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
            digest.update(chunk)
            consumed += len(chunk)
        after = os.fstat(descriptor)
        rebound = os.stat(absolute, follow_symlinks=False)
        after_fingerprint = (
            after.st_dev,
            after.st_ino,
            after.st_mode,
            after.st_nlink,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        )
        rebound_fingerprint = (
            rebound.st_dev,
            rebound.st_ino,
            rebound.st_mode,
            rebound.st_nlink,
            rebound.st_size,
            rebound.st_mtime_ns,
            rebound.st_ctime_ns,
        )
        if (
            fingerprint != after_fingerprint
            or fingerprint != rebound_fingerprint
            or consumed != before.st_size
        ):
            raise RuntimeError(f"{label} changed during exact-byte consumption")
        return b"".join(chunks), digest.hexdigest()
    finally:
        os.close(descriptor)


def _paths_overlap(first: Path, second: Path) -> bool:
    first_absolute = Path(os.path.abspath(os.fspath(first))).resolve()
    second_absolute = Path(os.path.abspath(os.fspath(second))).resolve()
    return bool(
        first_absolute == second_absolute
        or first_absolute in second_absolute.parents
        or second_absolute in first_absolute.parents
    )


def _private_atomic_bytes(path: Path, payload: bytes) -> None:
    if not path.parent.is_dir():
        raise FileNotFoundError(path.parent)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb", closefd=True) as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        os.chmod(path, 0o600, follow_symlinks=False)
        parent_fd = os.open(
            path.parent,
            os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
        )
        try:
            os.fsync(parent_fd)
        finally:
            os.close(parent_fd)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _private_atomic_json(path: Path, value: Any) -> None:
    _private_atomic_bytes(
        path,
        (
            json.dumps(
                value,
                indent=2,
                sort_keys=True,
                ensure_ascii=False,
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8"),
    )


def _private_atomic_array(path: Path, array: np.ndarray) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb", closefd=True) as stream:
            np.save(stream, array, allow_pickle=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        os.chmod(path, 0o600, follow_symlinks=False)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _private_atomic_metadata(path: Path, frame: pd.DataFrame) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        text=True,
    )
    temporary = Path(temporary_name)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="", closefd=True) as stream:
            frame.to_csv(stream, index=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        os.chmod(path, 0o600, follow_symlinks=False)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _rename_noreplace(source: Path, destination: Path) -> None:
    if os.path.lexists(destination):
        raise FileExistsError(destination)
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        raise RuntimeError("atomic V3 SVD publication requires renameat2")
    renameat2.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    renameat2.restype = ctypes.c_int
    result = renameat2(
        -100,
        os.fsencode(source),
        -100,
        os.fsencode(destination),
        _RENAME_NOREPLACE,
    )
    if result != 0:
        error_number = ctypes.get_errno()
        if error_number == errno.EEXIST:
            raise FileExistsError(destination)
        raise OSError(error_number, os.strerror(error_number), str(destination))


def _private_json_noreplace(path: Path, value: Any) -> None:
    """Durably publish one private JSON receipt without replacing evidence."""

    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        os.fchmod(descriptor, 0o600)
        payload = (
            json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
        ).encode("utf-8")
        with os.fdopen(descriptor, "wb", closefd=True) as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        _rename_noreplace(temporary, path)
        parent_fd = os.open(
            path.parent,
            os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
        )
        try:
            os.fsync(parent_fd)
        finally:
            os.close(parent_fd)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _secure_and_fsync_tree(root: Path) -> None:
    directories: list[Path] = []
    for current, child_directories, filenames in os.walk(root, topdown=False):
        current_path = Path(current)
        for name in filenames:
            path = current_path / name
            observed = path.lstat()
            if not stat.S_ISREG(observed.st_mode) or observed.st_nlink != 1:
                raise RuntimeError(f"V3 SVD output is not a private regular file: {path}")
            os.chmod(path, 0o600, follow_symlinks=False)
            descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        for name in child_directories:
            child = current_path / name
            observed = child.lstat()
            if not stat.S_ISDIR(observed.st_mode):
                raise RuntimeError(f"V3 SVD output contains a non-directory: {child}")
        directories.append(current_path)
    for directory in directories:
        os.chmod(directory, 0o700, follow_symlinks=False)
        descriptor = os.open(
            directory,
            os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
        )
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


def _private_receipt(value: dict[str, Any]) -> dict[str, Any]:
    document = dict(value)
    document["content_sha256"] = ""
    document["content_sha256"] = _canonical_content_sha256(document)
    return document


def _start_v3_attempt(
    output_root: Path,
    *,
    canonical_root: Path,
    dataset_root: Path,
) -> dict[str, Any]:
    final = Path(os.path.abspath(os.fspath(output_root)))
    parent = final.parent.resolve()
    final = parent / final.name
    parent.mkdir(parents=True, exist_ok=True)
    if os.path.lexists(final):
        raise FileExistsError(
            f"V3 SVD output already exists and cannot be overwritten: {final}"
        )
    failure_path = parent / f".{final.name}.FAILURE_RECEIPT.json"
    postcommit_path = parent / f".{final.name}.POSTCOMMIT_RECEIPT.json"
    claim_path = parent / f".{final.name}.attempt_claim.json"
    if (
        os.path.lexists(failure_path)
        or os.path.lexists(postcommit_path)
        or os.path.lexists(claim_path)
    ):
        raise FileExistsError(f"V3 SVD attempt evidence already exists for {final}")
    claim = _private_receipt(
        {
            "schema_version": "snn_rr.svd_component_cache_attempt.v1",
            "terminal_state": "running",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "requested_output_root": str(final),
            "canonical_cache": str(canonical_root),
            "dataset_root": str(dataset_root),
            "command": [str(item) for item in sys.argv],
            "completed_session_ids": [],
        }
    )
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC
    descriptor = os.open(claim_path, flags, 0o600)
    claim_stat = os.fstat(descriptor)
    attempt: dict[str, Any] = {
        **claim,
        "claim_path": str(claim_path),
        "claim_device": int(claim_stat.st_dev),
        "claim_inode": int(claim_stat.st_ino),
        "failure_path": str(failure_path),
        "postcommit_path": str(postcommit_path),
        "staging_root": None,
        "final_output_root": str(final),
        "publication_committed": False,
    }
    try:
        try:
            payload = (
                json.dumps(claim, indent=2, sort_keys=True, allow_nan=False) + "\n"
            ).encode("utf-8")
            view = memoryview(payload)
            while view:
                written = os.write(descriptor, view)
                if written <= 0:
                    raise RuntimeError("short write for V3 SVD attempt claim")
                view = view[written:]
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        staging = Path(
            tempfile.mkdtemp(dir=parent, prefix=f".{final.name}.staging.")
        )
        os.chmod(staging, 0o700, follow_symlinks=False)
        staging_stat = os.lstat(staging)
        attempt.update(
            {
                "staging_root": str(staging),
                "staging_device": int(staging_stat.st_dev),
                "staging_inode": int(staging_stat.st_ino),
            }
        )
        return attempt
    except BaseException as error:
        try:
            _record_v3_failure(attempt, error)
        finally:
            try:
                os.close(descriptor)
            except OSError:
                pass
        raise


def _unlink_bound_claim(attempt: dict[str, Any]) -> None:
    claim_path = Path(str(attempt["claim_path"]))
    try:
        observed = os.lstat(claim_path)
    except FileNotFoundError:
        return
    if (observed.st_dev, observed.st_ino) != (
        attempt.get("claim_device"),
        attempt.get("claim_inode"),
    ):
        raise RuntimeError("V3 SVD attempt claim generation changed")
    os.unlink(claim_path)


def _record_v3_failure(attempt: dict[str, Any], error: BaseException) -> None:
    if attempt.get("publication_committed") is True:
        raise RuntimeError(
            "refusing to record a pre-commit failure after V3 SVD publication"
        ) from error
    receipt = _private_receipt(
        {
            key: value
            for key, value in attempt.items()
            if key
            not in {
                "content_sha256",
                "claim_path",
                "failure_path",
                "postcommit_path",
            }
        }
        | {
            "schema_version": "snn_rr.svd_component_cache_failure.v1",
            "terminal_state": "failed",
            "failed_at_utc": datetime.now(timezone.utc).isoformat(),
            "error_type": type(error).__name__,
            "error": str(error),
        }
    )
    failure_path = Path(str(attempt["failure_path"]))
    _private_json_noreplace(failure_path, receipt)
    _unlink_bound_claim(attempt)
    parent_fd = os.open(
        failure_path.parent,
        os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
    )
    try:
        os.fsync(parent_fd)
    finally:
        os.close(parent_fd)


def _record_v3_postcommit_issue(
    attempt: dict[str, Any], error: BaseException
) -> None:
    """Record a committed publication whose durability/cleanup was incomplete."""

    receipt = _private_receipt(
        {
            key: value
            for key, value in attempt.items()
            if key
            not in {
                "content_sha256",
                "claim_path",
                "failure_path",
                "postcommit_path",
            }
        }
        | {
            "schema_version": "snn_rr.svd_component_cache_postcommit.v1",
            "terminal_state": "publication_committed_cleanup_incomplete",
            "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
            "error_type": type(error).__name__,
            "error": str(error),
        }
    )
    _private_json_noreplace(Path(str(attempt["postcommit_path"])), receipt)


def _publish_v3_attempt(attempt: dict[str, Any]) -> Path:
    staging = Path(str(attempt["staging_root"]))
    final = Path(str(attempt["final_output_root"]))
    _secure_and_fsync_tree(staging)
    _rename_noreplace(staging, final)
    # The namespace commit is irreversible.  Mark it before the first
    # fallible durability/cleanup operation so the caller can never emit a
    # contradictory terminal_state=failed receipt for a visible final tree.
    attempt["publication_committed"] = True
    parent_fd = os.open(
        final.parent,
        os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
    )
    try:
        os.fsync(parent_fd)
    finally:
        os.close(parent_fd)
    _unlink_bound_claim(attempt)
    parent_fd = os.open(
        final.parent,
        os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
    )
    try:
        os.fsync(parent_fd)
    finally:
        os.close(parent_fd)
    return final


def _pipeline_paths() -> list[Path]:
    """Historical V1/V2 digest surface; keep byte-for-byte dependency scope."""

    return [
        Path(__file__),
        SOURCE_ROOT / "snn_rr" / "svd_features.py",
        PROJECT_ROOT / "scripts" / "build_features.py",
        SOURCE_ROOT / "snn_rr" / "acquisition_contract.py",
        SOURCE_ROOT / "snn_rr" / "cache.py",
        SOURCE_ROOT / "snn_rr" / "data.py",
        SOURCE_ROOT / "snn_rr" / "preprocess.py",
        SOURCE_ROOT / "snn_rr" / "synchronization.py",
        SOURCE_ROOT / "snn_rr" / "radar_timing.py",
    ]


def _v3_pipeline_paths() -> list[Path]:
    return [*_pipeline_paths(), SOURCE_ROOT / "snn_rr" / "raw_snapshot.py"]


def _v3_execution_source_generation(pipeline_sha256: str) -> dict[str, Any]:
    return {
        "schema": "snn_rr.svd_execution_source_generation.v1",
        "guard_scope": "post_import_path_snapshot_only",
        "pipeline_path_snapshot_sha256": pipeline_sha256,
        "binds_actual_loader_compiled_bytes": False,
        "complete_private_import_closure": False,
        "diagnostic_only": True,
        "scientific_authority": False,
    }


def _pipeline_sha256(paths: list[Path]) -> str:
    return hashlib.sha256(
        "".join(_sha256(path) for path in paths).encode()
    ).hexdigest()


def _assert_pipeline_unchanged(paths: list[Path], expected_sha256: str) -> None:
    if _pipeline_sha256(paths) != expected_sha256:
        raise RuntimeError("SVD pipeline source changed while the cache was being built")


def _array_inventory(path: Path, array: np.ndarray) -> dict[str, Any]:
    return {
        "path": path.name,
        "sha256": _sha256(path),
        "bytes": path.stat().st_size,
        "shape": list(array.shape),
        "dtype": str(array.dtype),
    }


def _metadata_inventory(path: Path, metadata: pd.DataFrame) -> dict[str, Any]:
    return {
        "path": path.name,
        "sha256": _sha256(path),
        "bytes": path.stat().st_size,
        "shape": [len(metadata), len(metadata.columns)],
        "dtype": "csv",
    }


def _v3_file_inventory(
    path: Path,
    *,
    shape: list[int] | None,
    dtype: str,
) -> dict[str, Any]:
    payload, digest = _stable_regular_file_payload(
        path,
        label=f"V3 SVD output {path.name}",
    )
    return {
        "path": path.name,
        "sha256": digest,
        "bytes": len(payload),
        "shape": shape,
        "dtype": dtype,
    }


def _v3_array_inventory(path: Path, expected: np.ndarray) -> dict[str, Any]:
    payload, digest = _stable_regular_file_payload(
        path,
        label=f"V3 SVD output {path.name}",
    )
    try:
        observed = np.load(BytesIO(payload), allow_pickle=False)
    except (OSError, ValueError) as error:
        raise RuntimeError(f"cannot parse written V3 SVD array {path.name}") from error
    if (
        type(observed) is not np.ndarray
        or observed.shape != expected.shape
        or observed.dtype != expected.dtype
        or not np.array_equal(observed, expected, equal_nan=True)
    ):
        raise RuntimeError(f"written V3 SVD array differs from memory: {path.name}")
    return {
        "path": path.name,
        "sha256": digest,
        "bytes": len(payload),
        "shape": list(expected.shape),
        "dtype": str(expected.dtype),
    }


def _v3_metadata_inventory(path: Path, expected: pd.DataFrame) -> dict[str, Any]:
    payload, digest = _stable_regular_file_payload(
        path,
        label="V3 SVD output metadata.csv",
    )
    expected_payload = expected.to_csv(index=False).encode("utf-8")
    if payload != expected_payload:
        raise RuntimeError("written V3 SVD metadata differs from memory")
    return {
        "path": path.name,
        "sha256": digest,
        "bytes": len(payload),
        "shape": list(expected.shape),
        "dtype": "csv",
    }


def _v3_json_inventory(
    path: Path,
    expected: dict[str, Any],
) -> dict[str, Any]:
    payload, digest = _stable_regular_file_payload(
        path,
        label=f"V3 SVD output {path.name}",
    )
    if _strict_json_payload(payload, label=f"V3 SVD output {path.name}") != expected:
        raise RuntimeError(f"written V3 SVD JSON differs from memory: {path.name}")
    return {
        "path": path.name,
        "sha256": digest,
        "bytes": len(payload),
        "shape": None,
        "dtype": "json",
    }


def _inventory_files_match(root: Path, inventory: Any) -> bool:
    if not isinstance(inventory, dict) or not inventory:
        return False
    for entry in inventory.values():
        if not isinstance(entry, dict) or not isinstance(entry.get("path"), str):
            return False
        path = (root / entry["path"]).resolve()
        try:
            path.relative_to(root.resolve())
        except ValueError:
            return False
        if not path.is_file():
            return False
        if entry.get("bytes") != path.stat().st_size:
            return False
        if entry.get("sha256") != _sha256(path):
            return False
    return True


def _detect_v3_canonical_cache(
    canonical_root: Path,
) -> tuple[bool, dict[str, Any], bytes, str]:
    payload, file_sha256 = _stable_regular_file_payload(
        canonical_root / "manifest.json",
        label="canonical feature-cache root manifest",
        require_single_link=False,
    )
    manifest = _strict_json_payload(payload, label="canonical feature-cache root manifest")
    contract = manifest.get("acquisition_contract")
    root_schema = manifest.get("schema_version")
    contract_schema = contract.get("schema_version") if isinstance(contract, dict) else None
    session_schemas = {
        item.get("schema_version")
        for item in manifest.get("sessions", [])
        if isinstance(item, dict)
    }
    exact_root = root_schema == ACQUISITION_CACHE_ROOT_SCHEMA_VERSION_V3
    exact_contract = contract_schema == ACQUISITION_CACHE_SCHEMA_VERSION_V3
    exact_session = ACQUISITION_CACHE_SESSION_SCHEMA_VERSION_V3 in session_schemas
    v3_marked = bool(
        exact_root
        or exact_contract
        or exact_session
        or (isinstance(root_schema, str) and ".v3" in root_schema)
        or (isinstance(contract_schema, str) and ".v3" in contract_schema)
        or any(isinstance(value, str) and ".v3" in value for value in session_schemas)
    )
    if v3_marked and not (
        exact_root
        and exact_contract
        and session_schemas == {ACQUISITION_CACHE_SESSION_SCHEMA_VERSION_V3}
    ):
        raise ValueError(
            "mixed or lookalike version-3 canonical cache schemas are forbidden"
        )
    if exact_root:
        if (canonical_root / "manifest.json").lstat().st_nlink != 1:
            raise ValueError(
                "canonical version-3 root manifest must have exactly one hard link"
            )
        content_sha256 = manifest.get("content_sha256")
        if (
            not isinstance(content_sha256, str)
            or content_sha256 != _canonical_content_sha256(manifest)
        ):
            raise ValueError("canonical version-3 root content hash mismatch")
    return exact_root, manifest, payload, file_sha256


def _v3_output_manifest_is_current(
    root: Path,
    manifest: dict[str, Any],
) -> bool:
    pipeline_sha256 = manifest.get("pipeline_sha256")
    if (
        manifest.get("schema_version") != V3_SVD_SESSION_SCHEMA
        or manifest.get("diagnostic_only") is not True
        or manifest.get("scientific_eligible") is not False
        or manifest.get("training_authorized") is not False
        or manifest.get("content_sha256") != _canonical_content_sha256(manifest)
        or not isinstance(pipeline_sha256, str)
        or manifest.get("execution_source_generation")
        != _v3_execution_source_generation(pipeline_sha256)
    ):
        return False
    inventory = manifest.get("file_inventory")
    required = {
        "spectra",
        "component_signals",
        "attributes",
        "frequencies_hz",
        "metadata",
        "radar_timing_valid_mask",
        "radar_timing_invalid_reason_mask",
        "radar_view_availability_mask",
        "raw_consumption_receipt",
    }
    if (
        not isinstance(inventory, dict)
        or set(inventory) != required
        or manifest.get("inventory_sha256") != _canonical_sha256(inventory)
    ):
        return False
    for logical_name, entry in inventory.items():
        if not isinstance(entry, dict) or set(entry) != {
            "path",
            "sha256",
            "bytes",
            "shape",
            "dtype",
        }:
            return False
        filename = entry.get("path")
        if not isinstance(filename, str) or filename != Path(filename).name:
            return False
        try:
            payload, digest = _stable_regular_file_payload(
                root / filename,
                label=f"published V3 SVD {logical_name}",
            )
        except (OSError, ValueError, RuntimeError):
            return False
        if len(payload) != entry.get("bytes") or digest != entry.get("sha256"):
            return False
    return True


def _cached_svd_manifest_is_current(
    root: Path,
    manifest: dict[str, Any],
    *,
    acquisition_v2: bool,
) -> bool:
    declared_content = manifest.get("content_sha256")
    if not isinstance(declared_content, str):
        return False
    content_payload = dict(manifest)
    content_payload.pop("content_sha256", None)
    if declared_content != _canonical_sha256(content_payload):
        return False
    inventory = manifest.get("file_inventory")
    required = {
        "spectra",
        "component_signals",
        "attributes",
        "frequencies_hz",
        "metadata",
    }
    if acquisition_v2:
        required.add("radar_timing_valid_mask")
    if not isinstance(inventory, dict) or set(inventory) != required:
        return False
    if manifest.get("inventory_sha256") != _canonical_sha256(inventory):
        return False
    return _inventory_files_match(root, inventory)


def _verify_bound_canonical_file(
    canonical_dir: Path,
    canonical_manifest: dict[str, Any],
    key: str,
    expected_name: str,
) -> Path:
    inventory = canonical_manifest.get("file_inventory")
    if not isinstance(inventory, dict) or not isinstance(inventory.get(key), dict):
        raise RuntimeError(f"canonical cache lacks bound {key} inventory")
    entry = inventory[key]
    if entry.get("path") != expected_name:
        raise RuntimeError(f"canonical {key} inventory redirects its loader target")
    path = (canonical_dir / expected_name).resolve()
    try:
        path.relative_to(canonical_dir.resolve())
    except ValueError as error:
        raise RuntimeError(f"canonical {key} path escapes the session root") from error
    if not path.is_file():
        raise RuntimeError(f"canonical {key} input is missing")
    if entry.get("bytes") != path.stat().st_size or entry.get("sha256") != _sha256(path):
        raise RuntimeError(f"canonical {key} input differs from its inventory")
    return path


def _canonical_acquisition_binding(manifest: dict[str, Any]) -> dict[str, Any] | None:
    """Return the exact acquisition binding carried by a canonical cache manifest.

    The per-session acquisition contract is already content-addressed and
    contains the reconstruction, synchronization, mapping, approval, and range
    artifact hashes.  Binding the entire mapping avoids silently dropping a
    newly added provenance hash from the SVD session signature.
    """

    value = manifest.get("acquisition_contract")
    if value is None:
        return None
    if not isinstance(value, dict) or not value:
        raise ValueError("canonical cache acquisition_contract is malformed")
    # Round-trip through canonical JSON to reject non-JSON values and detach
    # the task payload from the mutable manifest object.
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return json.loads(encoded)


def _unique_session_ids(value: Any, *, location: str) -> list[str]:
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError(f"{location} must be a non-empty session-ID list")
    session_ids: list[str] = []
    for item in value:
        if not isinstance(item, str) or not item.strip() or item != item.strip():
            raise ValueError(f"{location} contains an invalid session ID")
        session_ids.append(item)
    if len(set(session_ids)) != len(session_ids):
        raise ValueError(f"{location} contains duplicate session IDs")
    return session_ids


def _canonical_expected_session_ids(
    canonical_available_session_ids: list[str],
    canonical_acquisition_contract: Any,
) -> list[str]:
    """Resolve the authoritative cohort independently of an SVD selection."""

    available = _unique_session_ids(
        canonical_available_session_ids,
        location="canonical cache usable-session catalogue",
    )
    if not (
        isinstance(canonical_acquisition_contract, dict)
        and canonical_acquisition_contract.get("schema_version")
        == "snn_rr.feature_cache_acquisition.v2"
    ):
        return available

    expected = _unique_session_ids(
        canonical_acquisition_contract.get("expected_usable_session_ids"),
        location="canonical acquisition expected usable-session IDs",
    )
    cache_usable = _unique_session_ids(
        canonical_acquisition_contract.get("cache_usable_session_ids"),
        location="canonical acquisition cache usable-session IDs",
    )
    if canonical_acquisition_contract.get(
        "expected_usable_session_ids_sha256"
    ) != _canonical_sha256(expected):
        raise ValueError("canonical acquisition expected-session hash mismatch")
    if canonical_acquisition_contract.get(
        "cache_usable_session_ids_sha256"
    ) != _canonical_sha256(cache_usable):
        raise ValueError("canonical acquisition cache-session hash mismatch")
    if cache_usable != available:
        raise ValueError(
            "canonical acquisition usable-session IDs differ from its root catalogue"
        )
    return expected


def _derive_output_selection_contract(
    *,
    expected_session_ids: list[str],
    selected_session_ids: list[str],
    results: list[dict[str, Any]],
    subjects_filter_applied: bool,
    canonical_acquisition_contract: Any,
) -> dict[str, Any]:
    """Derive SVD cohort scope and scientific eligibility from bound evidence."""

    expected = _unique_session_ids(
        expected_session_ids,
        location="SVD expected session IDs",
    )
    selected = _unique_session_ids(
        selected_session_ids,
        location="SVD selected session IDs",
    )
    if type(subjects_filter_applied) is not bool:
        raise ValueError("subjects_filter_applied must be an explicit boolean")
    if not isinstance(results, list) or any(not isinstance(item, dict) for item in results):
        raise ValueError("SVD results must be a list of mappings")
    result_ids = _unique_session_ids(
        [item.get("session_id") for item in results],
        location="SVD result session IDs",
    )
    if result_ids != selected:
        raise RuntimeError("SVD result catalogue differs from the selected session IDs")

    selection_is_full = bool(
        not subjects_filter_applied and selected == expected
    )
    selection_scope = "full_cohort" if selection_is_full else "diagnostic_subset"
    all_results_ok = bool(results) and all(
        item.get("status") == "ok" for item in results
    )

    acquisition_v2 = bool(
        isinstance(canonical_acquisition_contract, dict)
        and canonical_acquisition_contract.get("schema_version")
        == "snn_rr.feature_cache_acquisition.v2"
    )
    canonical_full = True
    canonical_scientific = False
    session_bindings_scientific = False
    if acquisition_v2:
        assert isinstance(canonical_acquisition_contract, dict)
        canonical_expected = _unique_session_ids(
            canonical_acquisition_contract.get("expected_usable_session_ids"),
            location="canonical acquisition expected usable-session IDs",
        )
        canonical_cache = _unique_session_ids(
            canonical_acquisition_contract.get("cache_usable_session_ids"),
            location="canonical acquisition cache usable-session IDs",
        )
        if canonical_expected != expected:
            raise ValueError("SVD/canonical acquisition expected-session IDs mismatch")
        if canonical_acquisition_contract.get(
            "expected_usable_session_ids_sha256"
        ) != _canonical_sha256(canonical_expected):
            raise ValueError("canonical acquisition expected-session hash mismatch")
        if canonical_acquisition_contract.get(
            "cache_usable_session_ids_sha256"
        ) != _canonical_sha256(canonical_cache):
            raise ValueError("canonical acquisition cache-session hash mismatch")
        canonical_full = bool(
            canonical_acquisition_contract.get("selection_scope") == "full_cohort"
            and canonical_acquisition_contract.get("full_cohort_complete") is True
            and canonical_acquisition_contract.get(
                "reconstruction_full_cohort_complete"
            )
            is True
            and canonical_cache == canonical_expected
        )
        canonical_scientific = bool(
            canonical_full
            and canonical_acquisition_contract.get("mode") == "strict"
            and canonical_acquisition_contract.get("scientific_eligible") is True
        )
        session_bindings_scientific = bool(
            results
            and all(
                isinstance(item.get("canonical_acquisition_binding"), dict)
                and item["canonical_acquisition_binding"].get("schema_version")
                == "snn_rr.feature_cache_acquisition.v2"
                and item["canonical_acquisition_binding"].get(
                    "scientific_eligible"
                )
                is True
                for item in results
            )
        )

    full_cohort_complete = bool(
        selection_is_full and all_results_ok and canonical_full
    )
    scientific_eligible = bool(
        acquisition_v2
        and full_cohort_complete
        and canonical_scientific
        and session_bindings_scientific
    )
    return {
        "expected_session_ids": expected,
        "expected_session_ids_sha256": _canonical_sha256(expected),
        "selected_session_ids": selected,
        "selected_session_ids_sha256": _canonical_sha256(selected),
        "subjects_filter_applied": subjects_filter_applied,
        "selection_scope": selection_scope,
        "full_cohort_complete": full_cohort_complete,
        "scientific_eligible": scientific_eligible,
    }


def _acquisition_session_manifest_sha256(
    binding: dict[str, Any] | None,
) -> str | None:
    if binding is None:
        return None
    value = binding.get("acquisition_session_manifest_sha256")
    if not isinstance(value, str) or len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ValueError(
            "canonical acquisition contract lacks acquisition_session_manifest_sha256"
        )
    return value


def _session_signature(task: dict[str, Any]) -> str:
    keys: tuple[str, ...] = (
        "pipeline_sha256",
        "canonical_root_manifest_sha256",
        "canonical_source_fingerprint",
        "canonical_session_manifest_sha256",
        "canonical_acquisition_session_manifest_sha256",
        "canonical_acquisition_binding",
        "acquisition_reconstruction_content_sha256",
        "dataset_root",
        "selected_rows_sha256",
        "valid_only",
        "components",
        "nfft",
        "n_iter",
        "variant_names",
    )
    if task.get("acquisition_v3") is True:
        keys += (
            "canonical_root_content_sha256",
            "canonical_reconstruction_file_sha256",
            "canonical_reconstruction_content_sha256",
            "canonical_session_content_sha256",
            "canonical_upstream_session_content_sha256",
            "canonical_raw_portable_content_sha256",
            "raw_graph_sha256",
            "dataset_catalogue_sha256",
            "reference_mapping_available",
            "declared_window_count",
            "cache_offset",
        )
    payload = {key: task[key] for key in keys}
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _build_v3_session(task: dict[str, Any]) -> dict[str, Any]:
    session_id = str(task["session_id"])
    canonical_root = Path(task["canonical_root"])
    canonical_dir = canonical_root / session_id
    output_dir = Path(task["output_dir"])
    if task.get("acquisition_v3") is not True or task.get("valid_only") is not False:
        raise RuntimeError(f"{session_id}: V3 SVD requires exact all-window mode")
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(f"{session_id}: V3 SVD session output already exists")

    root_payload, root_file_sha256 = _stable_regular_file_payload(
        canonical_root / "manifest.json",
        label=f"{session_id} canonical V3 root manifest",
    )
    root_manifest = _strict_json_payload(
        root_payload,
        label=f"{session_id} canonical V3 root manifest",
    )
    if (
        root_file_sha256 != task["canonical_root_manifest_sha256"]
        or root_manifest.get("content_sha256")
        != task["canonical_root_content_sha256"]
        or _canonical_content_sha256(root_manifest)
        != task["canonical_root_content_sha256"]
        or root_manifest.get("schema_version")
        != ACQUISITION_CACHE_ROOT_SCHEMA_VERSION_V3
        or not isinstance(root_manifest.get("acquisition_contract"), dict)
        or root_manifest["acquisition_contract"].get("schema_version")
        != ACQUISITION_CACHE_SCHEMA_VERSION_V3
    ):
        raise RuntimeError(f"{session_id}: canonical V3 root generation changed")

    session_payload, session_file_sha256 = _stable_regular_file_payload(
        canonical_dir / "manifest.json",
        label=f"{session_id} canonical V3 session manifest",
    )
    canonical_manifest = _strict_json_payload(
        session_payload,
        label=f"{session_id} canonical V3 session manifest",
    )
    if (
        session_file_sha256 != task["canonical_session_manifest_sha256"]
        or canonical_manifest.get("schema_version")
        != ACQUISITION_CACHE_SESSION_SCHEMA_VERSION_V3
        or canonical_manifest.get("session_id") != session_id
        or canonical_manifest.get("content_sha256")
        != task["canonical_session_content_sha256"]
        or _canonical_content_sha256(canonical_manifest)
        != task["canonical_session_content_sha256"]
        or canonical_manifest.get("upstream_session_content_sha256")
        != task["canonical_upstream_session_content_sha256"]
    ):
        raise RuntimeError(f"{session_id}: canonical V3 session generation changed")

    canonical_cache = load_feature_cache(
        canonical_root,
        sessions=[session_id],
        mmap=False,
        require_acquisition_contract=True,
    )
    provenance = canonical_cache.provenance
    timing_valid = canonical_cache.radar_timing_valid_mask
    timing_reasons = canonical_cache.radar_timing_invalid_reason_mask
    if (
        provenance is None
        or provenance.classification != "acquisition_diagnostic"
        or provenance.scientific_eligible
        or provenance.root_manifest_sha256
        != task["canonical_root_manifest_sha256"]
        or provenance.root_manifest_content_sha256
        != task["canonical_root_content_sha256"]
        or provenance.reconstruction_content_sha256
        != task["canonical_reconstruction_content_sha256"]
        or timing_valid is None
        or timing_reasons is None
    ):
        raise RuntimeError(f"{session_id}: canonical V3 cache authority mismatch")
    metadata = canonical_cache.metadata.copy(deep=True)
    selected_timing_masks = np.array(timing_valid, dtype=np.bool_, copy=True, order="C")
    selected_timing_reasons = np.array(
        timing_reasons, dtype=np.uint8, copy=True, order="C"
    )
    del canonical_cache
    declared_window_count = int(task["declared_window_count"])
    if (
        len(metadata) != declared_window_count
        or selected_timing_masks.shape != (declared_window_count, 3, 320)
        or selected_timing_reasons.shape != selected_timing_masks.shape
        or not np.array_equal(selected_timing_reasons != 0, ~selected_timing_masks)
    ):
        raise RuntimeError(f"{session_id}: canonical V3 row/timing support mismatch")
    reference_mapping_available = canonical_manifest.get(
        "reference_mapping_available"
    )
    if type(reference_mapping_available) is not bool:
        raise RuntimeError(f"{session_id}: mapping availability is not explicit")
    row_mapping_values = metadata.get("reference_mapping_available")
    if row_mapping_values is None or any(
        type(value) not in {bool, np.bool_}
        or bool(value) != reference_mapping_available
        for value in row_mapping_values
    ):
        raise RuntimeError(f"{session_id}: row/session mapping availability differs")
    if any(
        type(value) not in {bool, np.bool_} or bool(value)
        for value in metadata["reference_valid"].to_numpy()
    ):
        raise RuntimeError(f"{session_id}: diagnostic V3 labels are unexpectedly valid")
    selected_local = np.arange(declared_window_count, dtype=np.int64)
    if hashlib.sha256(selected_local.tobytes()).hexdigest() != task[
        "selected_rows_sha256"
    ]:
        raise RuntimeError(f"{session_id}: V3 all-window selection digest mismatch")
    selected_metadata = metadata.reset_index(drop=True)
    selected_metadata.insert(
        0,
        "cache_index",
        int(task["cache_offset"]) + selected_local,
    )

    upstream_session = canonical_manifest.get("upstream_session_contract")
    if not isinstance(upstream_session, dict):
        raise RuntimeError(f"{session_id}: upstream acquisition session is absent")
    raw_consumption = upstream_session.get("raw_consumption")
    if (
        upstream_session.get("content_sha256")
        != task["canonical_upstream_session_content_sha256"]
        or _canonical_content_sha256(upstream_session)
        != task["canonical_upstream_session_content_sha256"]
        or not isinstance(raw_consumption, dict)
        or raw_consumption.get("portable_content_sha256")
        != task["canonical_raw_portable_content_sha256"]
    ):
        raise RuntimeError(f"{session_id}: upstream raw authority binding mismatch")

    loaded = RawSessionReader(
        Path(task["dataset_root"]),
        task["raw_graph"],
        timezone_name="Asia/Seoul",
        fallback_rate_hz=40.0,
        biopac_strict=False,
        require_valid_records=True,
    ).consume()
    live_raw_binding = validate_consumption_against_contract(
        loaded.receipt,
        upstream_session,
    )
    if (
        live_raw_binding.get("portable_content_sha256")
        != task["canonical_raw_portable_content_sha256"]
        or live_raw_binding.get("diagnostic_only") is not True
        or live_raw_binding.get("scientific_authority") is not False
    ):
        raise RuntimeError(f"{session_id}: live raw consumption authority mismatch")
    raw_receipt_document = loaded.receipt.to_dict()

    radar_arrays: list[np.ndarray] = []
    outlier_counts: list[int] = []
    relative_times: list[np.ndarray] = []
    frame_sequences: list[np.ndarray] = []
    radar_start_epochs_s: list[float] = []
    timestamp_sources: list[str] = []
    for radar_id in (1, 2, 3):
        recording = loaded.radars[radar_id]
        values, outlier_count = replace_radar_outliers(
            np.asarray(recording.bins, dtype=np.float32)
        )
        if recording.meta.start_epoch_ms is None:
            raise RuntimeError(f"{session_id}: radar{radar_id} lacks an absolute anchor")
        radar_arrays.append(values)
        outlier_counts.append(int(outlier_count))
        relative_times.append(
            np.array(recording.timestamps_ms, dtype=np.float64, copy=True)
            / 1000.0
        )
        frame_sequences.append(
            np.array(recording.frame_sequence, dtype=np.uint32, copy=True)
        )
        radar_start_epochs_s.append(float(recording.meta.start_epoch_ms) / 1000.0)
        timestamp_sources.append(str(recording.meta.timestamp_source))
    del loaded

    sensor_summary = upstream_session.get("sensor_summary")
    radar_summary = sensor_summary.get("radar") if isinstance(sensor_summary, dict) else None
    bound_resampling = (
        radar_summary.get("feature_resampling")
        if isinstance(radar_summary, dict)
        else None
    )
    if not isinstance(bound_resampling, dict):
        raise RuntimeError(f"{session_id}: upstream measured resampling is absent")
    measured_resampling = causal_uniform_resample_radar_views_v1(
        radar_arrays,
        relative_times,
        radar_start_epochs_s,
        frame_sequences,
        output_hz=10.0,
        max_gap_s=0.050,
        gap_policy=str(bound_resampling.get("gap_policy")),
        timestamp_sources=timestamp_sources,
        require_measured_timestamps=True,
    )
    if (
        _canonical_sha256(measured_resampling.summary)
        != _canonical_sha256(bound_resampling)
        or radar_summary.get("past_only_outlier_replacements") != outlier_counts
        or measured_resampling.invalid_reason_mask is None
        or not np.array_equal(
            measured_resampling.invalid_reason_mask != 0,
            ~measured_resampling.valid_mask,
        )
        or canonical_manifest.get("radar_timing_invalid_reason_schema_version")
        != CAUSAL_UNIFORM_INVALID_REASON_SCHEMA_V1
        or canonical_manifest.get(
            "radar_timing_invalid_reason_semantics_sha256"
        )
        != CAUSAL_UNIFORM_INVALID_REASON_SEMANTICS_SHA256_V1
    ):
        raise RuntimeError(f"{session_id}: measured timing/reason authority differs")
    radar_arrays = [np.asarray(values, dtype=np.float32) for values in measured_resampling.values]
    common = min(map(len, radar_arrays))
    radar_arrays = [values[:common] for values in radar_arrays]

    variants = tuple(task["variant_names"])
    components = int(task["components"])
    nfft = int(task["nfft"])
    sample = svd_component_features(
        radar_arrays[0][:320],
        components=components,
        nfft=nfft,
        n_iter=int(task["n_iter"]),
        variants=variants,
    )
    spectra = np.empty(
        (
            declared_window_count,
            3,
            len(variants),
            components,
            sample.spectra.shape[-1],
        ),
        dtype=np.float16,
    )
    component_signals = np.empty(
        (declared_window_count, 3, len(variants), components, 320),
        dtype=np.float16,
    )
    attributes = np.empty(
        (
            declared_window_count,
            3,
            len(variants),
            components,
            len(ATTRIBUTE_NAMES),
        ),
        dtype=np.float32,
    )
    first_left = float(
        measured_resampling.times_s[0] - measured_resampling.interval_s
    )
    radar_support_records: list[dict[str, Any]] = []
    with threadpool_limits(limits=1):
        for output_row, row in selected_metadata.iterrows():
            requested_start = float(row["radar_window_start_relative_s"])
            requested_end = float(row["radar_window_end_relative_s"])
            start_coordinate = (
                requested_start - first_left
            ) / measured_resampling.interval_s
            start = int(round(start_coordinate))
            reconstructed_start = first_left + start * measured_resampling.interval_s
            if (
                not np.isfinite([requested_start, requested_end]).all()
                or abs(reconstructed_start - requested_start) > 1e-8
                or abs(requested_end - requested_start - 32.0) > 1e-8
                or start < 0
                or start + 320 > common
                or abs(
                    float(measured_resampling.times_s[start + 319]) - requested_end
                )
                > 1e-8
            ):
                raise RuntimeError(
                    f"{session_id} row {output_row}: measured radar support mismatch"
                )
            reconstructed_valid = np.asarray(
                measured_resampling.valid_mask[:, start : start + 320],
                dtype=np.bool_,
            )
            reconstructed_reasons = np.asarray(
                measured_resampling.invalid_reason_mask[:, start : start + 320],
                dtype=np.uint8,
            )
            if (
                not np.array_equal(
                    reconstructed_valid, selected_timing_masks[output_row]
                )
                or not np.array_equal(
                    reconstructed_reasons, selected_timing_reasons[output_row]
                )
            ):
                raise RuntimeError(
                    f"{session_id} row {output_row}: timing mask/reason join differs"
                )
            radar_support_records.append(
                {
                    "session_id": session_id,
                    "window_number": int(row["window_number"]),
                    "cache_index": int(row["cache_index"]),
                    "radar_window_start_relative_s": requested_start,
                    "radar_window_end_relative_s": requested_end,
                }
            )
            for radar_index, values in enumerate(radar_arrays):
                feature = svd_component_features(
                    values[start : start + 320],
                    components=components,
                    nfft=nfft,
                    n_iter=int(task["n_iter"]),
                    variants=variants,
                )
                if not np.array_equal(feature.frequencies_hz, sample.frequencies_hz):
                    raise RuntimeError("SVD frequency grid changed within a session")
                spectra[output_row, radar_index] = feature.spectra.astype(np.float16)
                component_signals[output_row, radar_index] = (
                    feature.component_signals.astype(np.float16)
                )
                attributes[output_row, radar_index] = feature.attributes
    radar_view_availability = np.all(selected_timing_masks, axis=2)
    spectra[~radar_view_availability] = np.float16(0.0)
    component_signals[~radar_view_availability] = np.float16(0.0)
    attributes[~radar_view_availability] = np.float32(0.0)
    if not (
        np.isfinite(spectra).all()
        and np.isfinite(component_signals).all()
        and np.isfinite(attributes).all()
        and np.isfinite(sample.frequencies_hz).all()
        and np.all(spectra[~radar_view_availability] == 0)
        and not np.any(np.signbit(spectra[~radar_view_availability]))
        and np.all(component_signals[~radar_view_availability] == 0)
        and not np.any(np.signbit(component_signals[~radar_view_availability]))
        and np.all(attributes[~radar_view_availability] == 0)
        and not np.any(np.signbit(attributes[~radar_view_availability]))
    ):
        raise RuntimeError(f"{session_id}: V3 SVD output contains non-finite features")

    os.mkdir(output_dir, mode=0o700)
    paths = {
        "spectra": output_dir / "spectra.npy",
        "component_signals": output_dir / "component_signals.npy",
        "attributes": output_dir / "attributes.npy",
        "frequencies_hz": output_dir / "frequencies_hz.npy",
        "metadata": output_dir / "metadata.csv",
        "radar_timing_valid_mask": output_dir / "radar_timing_valid_mask.npy",
        "radar_timing_invalid_reason_mask": (
            output_dir / "radar_timing_invalid_reason_mask.npy"
        ),
        "radar_view_availability_mask": (
            output_dir / "radar_view_availability_mask.npy"
        ),
        "raw_consumption_receipt": output_dir / "raw_consumption_receipt.json",
    }
    _private_atomic_array(paths["spectra"], spectra)
    _private_atomic_array(paths["component_signals"], component_signals)
    _private_atomic_array(paths["attributes"], attributes)
    _private_atomic_array(paths["frequencies_hz"], sample.frequencies_hz)
    _private_atomic_array(paths["radar_timing_valid_mask"], selected_timing_masks)
    _private_atomic_array(
        paths["radar_timing_invalid_reason_mask"], selected_timing_reasons
    )
    _private_atomic_array(
        paths["radar_view_availability_mask"], radar_view_availability
    )
    _private_atomic_metadata(paths["metadata"], selected_metadata)
    _private_atomic_json(paths["raw_consumption_receipt"], raw_receipt_document)
    file_inventory = {
        "spectra": _v3_array_inventory(paths["spectra"], spectra),
        "component_signals": _v3_array_inventory(
            paths["component_signals"], component_signals
        ),
        "attributes": _v3_array_inventory(paths["attributes"], attributes),
        "frequencies_hz": _v3_array_inventory(
            paths["frequencies_hz"], sample.frequencies_hz
        ),
        "metadata": _v3_metadata_inventory(paths["metadata"], selected_metadata),
        "radar_timing_valid_mask": _v3_array_inventory(
            paths["radar_timing_valid_mask"], selected_timing_masks
        ),
        "radar_timing_invalid_reason_mask": _v3_array_inventory(
            paths["radar_timing_invalid_reason_mask"], selected_timing_reasons
        ),
        "radar_view_availability_mask": _v3_array_inventory(
            paths["radar_view_availability_mask"], radar_view_availability
        ),
        "raw_consumption_receipt": _v3_json_inventory(
            paths["raw_consumption_receipt"], raw_receipt_document
        ),
    }
    raw_binding = {
        "schema_version": V3_SVD_RAW_BINDING_SCHEMA,
        "canonical_upstream_session_content_sha256": task[
            "canonical_upstream_session_content_sha256"
        ],
        "raw_consumption_receipt_content_sha256": raw_receipt_document[
            "content_sha256"
        ],
        "raw_consumption_portable_content_sha256": live_raw_binding[
            "portable_content_sha256"
        ],
        "validated_against_acquisition_contract": True,
        "biopac_bytes_consumed_for_full_graph_provenance_only": True,
        "biopac_semantic_values_used": False,
        "scientific_authority": False,
    }
    timing_contract = {
        "schema_version": V3_SVD_TIMING_CONTRACT_SCHEMA,
        "invalid_reason_schema_version": CAUSAL_UNIFORM_INVALID_REASON_SCHEMA_V1,
        "invalid_reason_semantics_sha256": (
            CAUSAL_UNIFORM_INVALID_REASON_SEMANTICS_SHA256_V1
        ),
        "axes": ["window", "radar_view", "interval"],
        "shape": list(selected_timing_masks.shape),
        "valid_mask_sha256": canonical_ndarray_sha256(selected_timing_masks),
        "invalid_reason_mask_sha256": canonical_ndarray_sha256(
            selected_timing_reasons
        ),
        "radar_view_availability_rule": "all_320_intervals_valid",
        "radar_view_availability_sha256": canonical_ndarray_sha256(
            radar_view_availability
        ),
        "unavailable_radar_view_count": int(
            radar_view_availability.size - radar_view_availability.sum()
        ),
        "unavailable_view_feature_values_exact_positive_zero": True,
        "invalid_reason_union_equals_not_valid": True,
        "recomputed_from_exact_consumed_raw_bytes": True,
        "canonical_cache_exact_join": True,
        "invalid_interval_count": int(
            np.size(selected_timing_masks) - selected_timing_masks.sum()
        ),
        "diagnostic_output_trainable": False,
    }
    target_firewall = {
        "schema_version": V3_SVD_TARGET_FIREWALL_SCHEMA,
        "inference_payloads": [
            "spectra",
            "component_signals",
            "attributes",
            "frequencies_hz",
            "radar_timing_valid_mask",
            "radar_timing_invalid_reason_mask",
            "radar_view_availability_mask",
        ],
        "metadata_is_annotation_only": True,
        "all_windows_required": True,
        "target_dependent_row_selection": False,
        "row_selection_label_inputs": [],
        "feature_value_label_inputs": [],
        "mapping_used_for_window_support": False,
        "biopac_semantic_values_used": False,
        "reference_values_used": False,
        "reference_mapping_available_recorded_not_inferred": True,
        "training_authorized": False,
    }
    support_contract = {
        "axes": ["window"],
        "key_columns": [
            "session_id",
            "window_number",
            "cache_index",
            "radar_window_start_relative_s",
            "radar_window_end_relative_s",
        ],
        "duration_s": 32.0,
        "model_hz": 10.0,
        "interval_count": 320,
        "mapping_required": False,
        "reference_time_columns_used": False,
        "rows_sha256": _canonical_sha256(radar_support_records),
    }
    manifest: dict[str, Any] = {
        "schema_version": V3_SVD_SESSION_SCHEMA,
        "content_sha256": "",
        "pipeline_version": V3_SVD_PIPELINE_VERSION,
        "session_id": session_id,
        "session_signature": _session_signature(task),
        "diagnostic_only": True,
        "scientific_eligible": False,
        "training_authorized": False,
        "canonical_cache_root_manifest_sha256": task[
            "canonical_root_manifest_sha256"
        ],
        "canonical_cache_root_content_sha256": task[
            "canonical_root_content_sha256"
        ],
        "canonical_cache_session_manifest_sha256": task[
            "canonical_session_manifest_sha256"
        ],
        "canonical_cache_session_content_sha256": task[
            "canonical_session_content_sha256"
        ],
        "canonical_reconstruction_file_sha256": task[
            "canonical_reconstruction_file_sha256"
        ],
        "canonical_reconstruction_content_sha256": task[
            "canonical_reconstruction_content_sha256"
        ],
        "canonical_upstream_session_content_sha256": task[
            "canonical_upstream_session_content_sha256"
        ],
        "canonical_source_fingerprint": task["canonical_source_fingerprint"],
        "pipeline_sha256": task["pipeline_sha256"],
        "execution_source_generation": _v3_execution_source_generation(
            str(task["pipeline_sha256"])
        ),
        "reference_mapping_available": reference_mapping_available,
        "valid_only": False,
        "row_count": declared_window_count,
        "cache_index_min": int(selected_metadata["cache_index"].min()),
        "cache_index_max": int(selected_metadata["cache_index"].max()),
        "selected_rows_sha256": task["selected_rows_sha256"],
        "variant_names": list(variants),
        "attribute_names": list(ATTRIBUTE_NAMES),
        "components": components,
        "nfft": nfft,
        "n_iter": int(task["n_iter"]),
        "spectra_shape": list(spectra.shape),
        "component_signals_shape": list(component_signals.shape),
        "attributes_shape": list(attributes.shape),
        "radar_view_availability_mask_shape": list(
            radar_view_availability.shape
        ),
        "frequency_grid_sha256": canonical_ndarray_sha256(sample.frequencies_hz),
        "radar_outlier_replacements": outlier_counts,
        "radar_resampling_summary": measured_resampling.summary,
        "raw_consumption_contract": raw_binding,
        "raw_consumption_contract_sha256": _canonical_sha256(raw_binding),
        "timing_support_contract": timing_contract,
        "timing_support_contract_sha256": _canonical_sha256(timing_contract),
        "radar_relative_support_contract": support_contract,
        "radar_relative_support_contract_sha256": _canonical_sha256(
            support_contract
        ),
        "target_firewall": target_firewall,
        "target_firewall_sha256": _canonical_sha256(target_firewall),
        "file_inventory": file_inventory,
        "inventory_sha256": _canonical_sha256(file_inventory),
    }
    manifest["content_sha256"] = _canonical_content_sha256(manifest)
    _private_atomic_json(output_dir / "manifest.json", manifest)
    if not _v3_output_manifest_is_current(output_dir, manifest):
        raise RuntimeError(f"{session_id}: V3 SVD output failed its own byte audit")

    final_root_payload, final_root_sha256 = _stable_regular_file_payload(
        canonical_root / "manifest.json",
        label=f"{session_id} final canonical V3 root barrier",
    )
    final_session_payload, final_session_sha256 = _stable_regular_file_payload(
        canonical_dir / "manifest.json",
        label=f"{session_id} final canonical V3 session barrier",
    )
    if (
        final_root_payload != root_payload
        or final_root_sha256 != root_file_sha256
        or final_session_payload != session_payload
        or final_session_sha256 != session_file_sha256
    ):
        raise RuntimeError(f"{session_id}: canonical V3 inputs changed during SVD")
    manifest_payload, manifest_file_sha256 = _stable_regular_file_payload(
        output_dir / "manifest.json",
        label=f"{session_id} V3 SVD session manifest",
    )
    if _strict_json_payload(
        manifest_payload, label=f"{session_id} V3 SVD session manifest"
    ) != manifest:
        raise RuntimeError(f"{session_id}: V3 SVD manifest publication differs")
    return {
        "session_id": session_id,
        "status": "ok",
        "schema_version": V3_SVD_SESSION_SCHEMA,
        "manifest_path": f"{session_id}/manifest.json",
        "manifest_sha256": manifest_file_sha256,
        "manifest_content_sha256": manifest["content_sha256"],
        "inventory_sha256": manifest["inventory_sha256"],
        "row_count": declared_window_count,
        "reference_mapping_available": reference_mapping_available,
        "scientific_eligible": False,
        "training_authorized": False,
    }


def _build_session(task: dict[str, Any]) -> dict[str, Any]:
    session_id = str(task["session_id"])
    output_dir = Path(task["output_dir"])
    canonical_dir = Path(task["canonical_dir"])
    canonical_root_manifest_path = canonical_dir.parent / "manifest.json"
    if _sha256(canonical_root_manifest_path) != task[
        "canonical_root_manifest_sha256"
    ]:
        raise RuntimeError(
            f"{session_id}: canonical root manifest changed after task creation"
        )
    canonical_manifest_path = canonical_dir / "manifest.json"
    observed_manifest_sha256 = _sha256(canonical_manifest_path)
    if observed_manifest_sha256 != task["canonical_session_manifest_sha256"]:
        raise RuntimeError(f"{session_id}: canonical session manifest changed after task creation")
    canonical_manifest = json.loads(canonical_manifest_path.read_text(encoding="utf-8"))
    observed_acquisition_binding = _canonical_acquisition_binding(canonical_manifest)
    if observed_acquisition_binding != task["canonical_acquisition_binding"]:
        raise RuntimeError(f"{session_id}: canonical acquisition binding changed after task creation")
    if _acquisition_session_manifest_sha256(observed_acquisition_binding) != task[
        "canonical_acquisition_session_manifest_sha256"
    ]:
        raise RuntimeError(f"{session_id}: canonical acquisition content hash changed")
    acquisition_v2 = bool(
        isinstance(observed_acquisition_binding, dict)
        and observed_acquisition_binding.get("schema_version")
        == "snn_rr.feature_cache_acquisition.v2"
    )
    if acquisition_v2:
        reconstruction_path_value = task.get("acquisition_reconstruction_manifest")
        if not isinstance(reconstruction_path_value, str):
            raise RuntimeError(
                f"{session_id}: acquisition reconstruction path is not bound"
            )
        reconstruction = load_acquisition_reconstruction(
            Path(reconstruction_path_value)
        )
        if reconstruction.content_sha256 != task.get(
            "acquisition_reconstruction_content_sha256"
        ):
            raise RuntimeError(
                f"{session_id}: acquisition reconstruction content hash changed"
            )
        session_contract = reconstruction.sessions.get(session_id)
        if session_contract is None or session_contract.content_sha256 != task[
            "canonical_acquisition_session_manifest_sha256"
        ]:
            raise RuntimeError(
                f"{session_id}: acquisition session binding changed"
            )
        # This occurs before the output-cache fast path.  A same-size,
        # same-mtime raw mutation therefore cannot reuse stale SVD features.
        validate_raw_input_bindings(session_contract, Path(task["dataset_root"]))

    metadata_path = (
        _verify_bound_canonical_file(
            canonical_dir,
            canonical_manifest,
            "metadata",
            "metadata.csv",
        )
        if acquisition_v2
        else canonical_dir / "metadata.csv"
    )
    timing_mask_path = (
        _verify_bound_canonical_file(
            canonical_dir,
            canonical_manifest,
            "radar_timing_valid_mask",
            "radar_timing_valid_mask.npy",
        )
        if acquisition_v2
        else None
    )
    signature = _session_signature(task)
    complete = [
        output_dir / "spectra.npy",
        output_dir / "component_signals.npy",
        output_dir / "attributes.npy",
        output_dir / "frequencies_hz.npy",
        output_dir / "metadata.csv",
        output_dir / "manifest.json",
    ]
    if acquisition_v2:
        complete.append(output_dir / "radar_timing_valid_mask.npy")
    if not task["force"] and all(path.is_file() for path in complete):
        previous = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
        if previous.get("session_signature") == signature and (
            _cached_svd_manifest_is_current(
                output_dir,
                previous,
                acquisition_v2=acquisition_v2,
            )
        ):
            return {"session_id": session_id, "status": "ok", "cached": True, **previous}

    metadata = pd.read_csv(metadata_path)
    if task["valid_only"]:
        selected_local = np.flatnonzero(metadata["reference_valid"].to_numpy(dtype=bool))
    else:
        selected_local = np.arange(len(metadata), dtype=np.int64)
    if not len(selected_local):
        return {"session_id": session_id, "status": "skipped", "reason": "no selected rows"}
    selected_metadata = metadata.iloc[selected_local].copy().reset_index(drop=True)
    selected_metadata.insert(0, "cache_index", int(task["cache_offset"]) + selected_local)
    selected_timing_masks: np.ndarray | None = None
    if acquisition_v2:
        assert timing_mask_path is not None
        canonical_timing_masks = np.load(timing_mask_path, allow_pickle=False)
        if (
            canonical_timing_masks.dtype != np.bool_
            or canonical_timing_masks.shape != (len(metadata), 3, 320)
        ):
            raise RuntimeError(
                f"{session_id}: canonical timing mask shape/dtype is invalid"
            )
        selected_timing_masks = np.asarray(
            canonical_timing_masks[selected_local], dtype=np.bool_
        )

    radar_arrays: list[np.ndarray] = []
    outlier_counts: list[int] = []
    relative_times: list[np.ndarray] = []
    frame_sequences: list[np.ndarray] = []
    timestamp_sources: list[str] = []
    for recording_dir in task["recording_dirs"]:
        recording = load_xethru_recording(recording_dir, strict=True)
        values, outlier_count = replace_radar_outliers(
            np.asarray(recording.records["bins"], dtype=np.float32)
        )
        radar_arrays.append(values)
        outlier_counts.append(int(outlier_count))
        relative_times.append(
            np.asarray(recording.timestamps_ms, dtype=np.float64) / 1000.0
        )
        frame_sequences.append(
            np.asarray(recording.records["frame_sequence"], dtype=np.uint32)
        )
        timestamp_sources.append(
            recording.meta.timestamp_source
            if recording.meta is not None
            else "fallback_40hz"
        )
    measured_resampling = None
    if acquisition_v2:
        bound_summary = canonical_manifest.get("radar_timing_summary")
        if not isinstance(bound_summary, dict):
            raise RuntimeError(
                f"{session_id}: canonical cache lacks measured timing summary"
            )
        measured_resampling = causal_uniform_resample_radar_views_v1(
            radar_arrays,
            relative_times,
            task["radar_start_epochs_s"],
            frame_sequences,
            output_hz=10.0,
            max_gap_s=0.050,
            gap_policy=str(bound_summary.get("gap_policy")),
            timestamp_sources=timestamp_sources,
            require_measured_timestamps=bool(
                observed_acquisition_binding.get("measured_timing_eligible")
            ),
        )
        if json.dumps(
            measured_resampling.summary,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ) != json.dumps(
            bound_summary,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ):
            raise RuntimeError(
                f"{session_id}: measured SVD timing differs from canonical cache"
            )
        if canonical_manifest.get("radar_outlier_replacements") != outlier_counts:
            raise RuntimeError(
                f"{session_id}: SVD outlier preprocessing differs from canonical cache"
            )
        radar_arrays = [values for values in measured_resampling.values]
    else:
        radar_arrays = [causal_block_mean(values, 4) for values in radar_arrays]
    common = min(map(len, radar_arrays))
    radar_arrays = [values[:common] for values in radar_arrays]

    variants = tuple(task["variant_names"])
    components = int(task["components"])
    nfft = int(task["nfft"])
    sample = svd_component_features(
        radar_arrays[0][:320],
        components=components,
        nfft=nfft,
        n_iter=int(task["n_iter"]),
        variants=variants,
    )
    spectra = np.empty(
        (
            len(selected_metadata),
            3,
            len(variants),
            components,
            sample.spectra.shape[-1],
        ),
        dtype=np.float16,
    )
    component_signals = np.empty(
        (
            len(selected_metadata),
            3,
            len(variants),
            components,
            320,
        ),
        dtype=np.float16,
    )
    attributes = np.empty(
        (len(selected_metadata), 3, len(variants), components, len(ATTRIBUTE_NAMES)),
        dtype=np.float32,
    )
    radar_start = float(canonical_manifest["radar_start_epoch"])
    biopac_start = float(canonical_manifest["biopac_start_epoch"])

    with threadpool_limits(limits=1):
        for output_row, row in selected_metadata.iterrows():
            if acquisition_v2:
                assert measured_resampling is not None
                first_left = float(
                    measured_resampling.times_s[0]
                    - measured_resampling.interval_s
                )
                requested_start = float(row["radar_window_start_relative_s"])
                start = int(
                    round(
                        (requested_start - first_left)
                        / measured_resampling.interval_s
                    )
                )
                reconstructed_start = (
                    first_left + start * measured_resampling.interval_s
                )
                if abs(reconstructed_start - requested_start) > 1e-8:
                    raise RuntimeError(
                        f"{session_id} row {output_row}: measured start is off grid"
                    )
                requested_end = float(row["radar_window_end_relative_s"])
                if start + 320 > common or abs(
                    float(measured_resampling.times_s[start + 319]) - requested_end
                ) > 1e-8:
                    raise RuntimeError(
                        f"{session_id} row {output_row}: measured 32 s support mismatch"
                    )
            else:
                window_start_epoch = biopac_start + float(row["window_start_s"])
                start = int(round((window_start_epoch - radar_start) * 10.0))
                reconstructed = radar_start + start / 10.0 - biopac_start
                if abs(reconstructed - float(row["window_start_s"])) > 0.051:
                    raise RuntimeError(
                        f"{session_id} row {output_row}: timestamp cannot map to 10 Hz grid"
                    )
            if start < 0 or start + 320 > common:
                raise RuntimeError(
                    f"{session_id} row {output_row}: raw window [{start},{start + 320}) "
                    f"escapes [0,{common})"
                )
            if acquisition_v2:
                assert measured_resampling is not None
                assert selected_timing_masks is not None
                reconstructed_mask = np.asarray(
                    measured_resampling.valid_mask[:, start : start + 320],
                    dtype=np.bool_,
                )
                if not np.array_equal(
                    reconstructed_mask, selected_timing_masks[output_row]
                ):
                    raise RuntimeError(
                        f"{session_id} row {output_row}: timing mask differs from "
                        "canonical support"
                    )
            for radar_index, values in enumerate(radar_arrays):
                feature = svd_component_features(
                    values[start : start + 320],
                    components=components,
                    nfft=nfft,
                    n_iter=int(task["n_iter"]),
                    variants=variants,
                )
                if not np.array_equal(feature.frequencies_hz, sample.frequencies_hz):
                    raise RuntimeError("SVD frequency grid changed within a session")
                spectra[output_row, radar_index] = feature.spectra.astype(np.float16)
                component_signals[output_row, radar_index] = (
                    feature.component_signals.astype(np.float16)
                )
                attributes[output_row, radar_index] = feature.attributes

    output_dir.mkdir(parents=True, exist_ok=True)
    _atomic_save(output_dir / "spectra.npy", spectra)
    _atomic_save(output_dir / "component_signals.npy", component_signals)
    _atomic_save(output_dir / "attributes.npy", attributes)
    _atomic_save(output_dir / "frequencies_hz.npy", sample.frequencies_hz)
    if selected_timing_masks is not None:
        _atomic_save(
            output_dir / "radar_timing_valid_mask.npy",
            selected_timing_masks,
        )
    metadata_temporary = output_dir / "metadata.csv.tmp"
    selected_metadata.to_csv(metadata_temporary, index=False)
    metadata_temporary.replace(output_dir / "metadata.csv")
    file_inventory = {
        "spectra": _array_inventory(output_dir / "spectra.npy", spectra),
        "component_signals": _array_inventory(
            output_dir / "component_signals.npy", component_signals
        ),
        "attributes": _array_inventory(
            output_dir / "attributes.npy", attributes
        ),
        "frequencies_hz": _array_inventory(
            output_dir / "frequencies_hz.npy", sample.frequencies_hz
        ),
        "metadata": _metadata_inventory(
            output_dir / "metadata.csv", selected_metadata
        ),
    }
    if selected_timing_masks is not None:
        file_inventory["radar_timing_valid_mask"] = _array_inventory(
            output_dir / "radar_timing_valid_mask.npy",
            selected_timing_masks,
        )
    manifest = {
        "pipeline_version": PIPELINE_VERSION,
        "session_signature": signature,
        "canonical_source_fingerprint": task["canonical_source_fingerprint"],
        "canonical_session_manifest_sha256": task["canonical_session_manifest_sha256"],
        "canonical_acquisition_session_manifest_sha256": task[
            "canonical_acquisition_session_manifest_sha256"
        ],
        "canonical_acquisition_binding": task["canonical_acquisition_binding"],
        "selected_rows_sha256": task["selected_rows_sha256"],
        "valid_only": bool(task["valid_only"]),
        "row_count": int(len(selected_metadata)),
        "cache_index_min": int(selected_metadata["cache_index"].min()),
        "cache_index_max": int(selected_metadata["cache_index"].max()),
        "spectra_shape": list(spectra.shape),
        "component_signals_shape": list(component_signals.shape),
        "attributes_shape": list(attributes.shape),
        "spectra_dtype": str(spectra.dtype),
        "component_signals_dtype": str(component_signals.dtype),
        "attributes_dtype": str(attributes.dtype),
        "radar_timing_valid_mask_shape": (
            None
            if selected_timing_masks is None
            else list(selected_timing_masks.shape)
        ),
        "radar_timing_invalid_interval_count": (
            None
            if selected_timing_masks is None
            else int(np.size(selected_timing_masks) - selected_timing_masks.sum())
        ),
        "radar_timing_mask_contract": (
            None
            if selected_timing_masks is None
            else {
                "mask_required_for_gap_tolerant_consumers": True,
                "scientific_source_requires_all_true": True,
                "diagnostic_output_trainable": False,
                "invalid_cells_are_exact_zero_but_not_semantic_measurements": True,
            }
        ),
        "variant_names": list(variants),
        "attribute_names": list(ATTRIBUTE_NAMES),
        "components": components,
        "nfft": nfft,
        "n_iter": int(task["n_iter"]),
        "frequency_min_hz": float(sample.frequencies_hz[0]),
        "frequency_max_hz": float(sample.frequencies_hz[-1]),
        "radar_outlier_replacements": outlier_counts,
        "feature_inputs": [
            "raw_radar_window",
            "past_only_outlier_state_from_session_start",
            (
                "measured_causal_10hz_radar_grid"
                if acquisition_v2
                else "legacy_nominal_10hz_radar_grid"
            ),
            "canonical_radar_and_biopac_timestamps_for_alignment",
        ],
        "causality_scope": {
            "window_end_prediction_uses_only_past_32s": True,
            "within_window_prefix_causal_representation": False,
            "reason": (
                "detrending, standardization, and randomized SVD are fitted over "
                "the complete 32 s window; an early component sample may depend "
                "on later samples inside that already-observed window"
            ),
            "streaming_prefix_causality_claim_allowed": False,
        },
        "radar_timing_summary": (
            None if measured_resampling is None else measured_resampling.summary
        ),
        "file_inventory": file_inventory,
        "inventory_sha256": _canonical_sha256(file_inventory),
        "label_inputs": (
            ["canonical_metadata.reference_valid (row_selection_only)"]
            if task["valid_only"]
            else []
        ),
        "feature_value_label_inputs": [],
        "target_dependent_row_selection": bool(task["valid_only"]),
        "split_half_layout_status": "unverified_hypothesis",
    }
    manifest["content_sha256"] = _canonical_sha256(manifest)
    _write_json(output_dir / "manifest.json", manifest)
    return {"session_id": session_id, "status": "ok", "cached": False, **manifest}


def _main_v3(
    args: argparse.Namespace,
    *,
    dataset_root: Path,
    canonical_root: Path,
    output_root: Path,
    canonical_root_manifest: dict[str, Any],
    canonical_root_payload: bytes,
    canonical_root_manifest_sha256: str,
) -> int:
    if not args.all_windows:
        raise ValueError("version-3 diagnostic SVD requires --all-windows")
    if args.subjects is not None:
        raise ValueError("version-3 diagnostic SVD forbids subject subsets")
    if args.force:
        raise ValueError("version-3 diagnostic SVD forbids --force and all overwrite")
    if any(
        _paths_overlap(output_root, protected)
        for protected in (dataset_root, canonical_root)
    ):
        raise ValueError("version-3 SVD output must be disjoint from raw/cache inputs")
    if os.path.lexists(output_root):
        raise FileExistsError(
            f"version-3 SVD output already exists and cannot be overwritten: {output_root}"
        )

    root_contract = canonical_root_manifest.get("acquisition_contract")
    if not isinstance(root_contract, dict):
        raise ValueError("version-3 canonical acquisition contract is absent")
    expected_ids = _unique_session_ids(
        root_contract.get("expected_usable_session_ids"),
        location="version-3 expected usable session IDs",
    )
    cache_ids = _unique_session_ids(
        root_contract.get("cache_usable_session_ids"),
        location="version-3 cache usable session IDs",
    )
    raw_root_items = canonical_root_manifest.get("sessions")
    if not isinstance(raw_root_items, list):
        raise ValueError("version-3 canonical root sessions are absent")
    root_items: list[dict[str, Any]] = []
    for item in raw_root_items:
        if not isinstance(item, dict):
            raise ValueError("version-3 canonical root session item is malformed")
        root_items.append(item)
    root_ids = _unique_session_ids(
        [item.get("session_id") for item in root_items],
        location="version-3 canonical root session IDs",
    )
    if (
        root_contract.get("schema_version") != ACQUISITION_CACHE_SCHEMA_VERSION_V3
        or root_contract.get("mode") != "diagnostic"
        or root_contract.get("scientific_eligible") is not False
        or root_contract.get("subjects_filter_applied") is not False
        or root_contract.get("selection_scope") != "full_cohort"
        or root_contract.get("full_cohort_complete") is not True
        or root_contract.get("expected_usable_session_ids_sha256")
        != _canonical_sha256(expected_ids)
        or root_contract.get("cache_usable_session_ids_sha256")
        != _canonical_sha256(cache_ids)
        or expected_ids != cache_ids
        or cache_ids != root_ids
        or len(root_ids) != V3_EXPECTED_SESSION_COUNT
        or any(
            item.get("schema_version")
            != ACQUISITION_CACHE_SESSION_SCHEMA_VERSION_V3
            or item.get("status") != "ok"
            or item.get("scientific_eligible") is not False
            for item in root_items
        )
    ):
        raise ValueError("version-3 canonical cache is not the exact diagnostic cohort")

    preflight = load_feature_cache(
        canonical_root,
        sessions=[root_ids[0]],
        mmap=False,
        require_acquisition_contract=True,
    )
    if (
        preflight.provenance is None
        or preflight.provenance.classification != "acquisition_diagnostic"
        or preflight.provenance.scientific_eligible
        or preflight.provenance.root_manifest_sha256
        != canonical_root_manifest_sha256
        or preflight.provenance.root_manifest_content_sha256
        != canonical_root_manifest["content_sha256"]
    ):
        raise ValueError("version-3 canonical cache preflight authority failed")
    del preflight

    reconstruction_name = root_contract.get("reconstruction_manifest")
    if (
        not isinstance(reconstruction_name, str)
        or reconstruction_name != Path(reconstruction_name).name
    ):
        raise ValueError("version-3 reconstruction path is not a contained leaf")
    reconstruction_path = canonical_root / reconstruction_name
    reconstruction_payload, reconstruction_file_sha256 = (
        _stable_regular_file_payload(
            reconstruction_path,
            label="canonical version-3 reconstruction",
        )
    )
    reconstruction = _strict_json_payload(
        reconstruction_payload,
        label="canonical version-3 reconstruction",
    )
    reconstruction_content_sha256 = reconstruction.get("content_sha256")
    if (
        reconstruction_file_sha256
        != root_contract.get("reconstruction_manifest_sha256")
        or len(reconstruction_payload)
        != root_contract.get("reconstruction_manifest_bytes")
        or reconstruction_content_sha256
        != root_contract.get("reconstruction_content_sha256")
        or not isinstance(reconstruction_content_sha256, str)
        or reconstruction_content_sha256
        != _canonical_content_sha256(reconstruction)
    ):
        raise ValueError("version-3 reconstruction content binding mismatch")

    dataset_manifest = build_dataset_manifest(dataset_root)
    dataset_catalogue = _validate_v3_dataset_catalogue(
        dataset_root, manifest=dataset_manifest
    )
    dataset_catalogue_sha256 = _canonical_sha256(dataset_catalogue)
    by_session = {
        subject.subject_id: subject for subject in dataset_manifest.usable_subjects
    }
    if list(by_session) != root_ids:
        raise ValueError("raw dataset usable catalogue differs from V3 cache")
    dataset_manifest_sha256 = _canonical_sha256(dataset_manifest.to_dict())
    pipeline_paths = _v3_pipeline_paths()
    pipeline_digest = _pipeline_sha256(pipeline_paths)

    attempt = _start_v3_attempt(
        output_root,
        canonical_root=canonical_root,
        dataset_root=dataset_root,
    )
    attempt.update(
        {
            "canonical_root_manifest_sha256": canonical_root_manifest_sha256,
            "canonical_root_content_sha256": canonical_root_manifest[
                "content_sha256"
            ],
            "canonical_reconstruction_file_sha256": (
                reconstruction_file_sha256
            ),
            "canonical_reconstruction_content_sha256": (
                reconstruction_content_sha256
            ),
            "dataset_manifest_sha256": dataset_manifest_sha256,
            "dataset_catalogue_sha256": dataset_catalogue_sha256,
            "pipeline_sha256": pipeline_digest,
        }
    )
    staging_root = Path(str(attempt["staging_root"]))
    try:
        offsets: dict[str, int] = {}
        task_inputs: list[tuple[dict[str, Any], dict[str, Any]]] = []
        offset = 0
        for root_item in root_items:
            session_id = str(root_item["session_id"])
            session_manifest_path = canonical_root / session_id / "manifest.json"
            session_payload, session_file_sha256 = _stable_regular_file_payload(
                session_manifest_path,
                label=f"canonical V3 session manifest {session_id}",
            )
            session_manifest = _strict_json_payload(
                session_payload,
                label=f"canonical V3 session manifest {session_id}",
            )
            session_content_sha256 = session_manifest.get("content_sha256")
            upstream_session = session_manifest.get("upstream_session_contract")
            raw_consumption = (
                upstream_session.get("raw_consumption")
                if isinstance(upstream_session, dict)
                else None
            )
            declared_window_count = session_manifest.get("window_count")
            reference_mapping_available = session_manifest.get(
                "reference_mapping_available"
            )
            if (
                session_manifest.get("schema_version")
                != ACQUISITION_CACHE_SESSION_SCHEMA_VERSION_V3
                or session_manifest.get("session_id") != session_id
                or session_file_sha256 != root_item.get("manifest_sha256")
                or session_content_sha256
                != root_item.get("manifest_content_sha256")
                or not isinstance(session_content_sha256, str)
                or session_content_sha256
                != _canonical_content_sha256(session_manifest)
                or session_manifest.get("scientific_eligible") is not False
                or session_manifest.get("sync_authorized") is not False
                or type(reference_mapping_available) is not bool
                or not isinstance(upstream_session, dict)
                or upstream_session.get("content_sha256")
                != root_item.get("upstream_session_content_sha256")
                or session_manifest.get("upstream_session_content_sha256")
                != root_item.get("upstream_session_content_sha256")
                or not isinstance(raw_consumption, dict)
                or not isinstance(
                    raw_consumption.get("portable_content_sha256"), str
                )
                or type(declared_window_count) is not int
                or declared_window_count <= 0
            ):
                raise ValueError(
                    f"{session_id}: canonical V3 session input is malformed"
                )
            offsets[session_id] = offset
            selected = np.arange(declared_window_count, dtype=np.int64)
            raw_graph = graph_from_subject(dataset_root, by_session[session_id])
            task_inputs.append(
                (
                    root_item,
                    {
                        "session_id": session_id,
                        "acquisition_v3": True,
                        "raw_graph": raw_graph,
                        "raw_graph_sha256": _canonical_sha256(raw_graph.to_dict()),
                        "canonical_root": str(canonical_root),
                        "canonical_dir": str(canonical_root / session_id),
                        "canonical_root_manifest_sha256": (
                            canonical_root_manifest_sha256
                        ),
                        "canonical_root_content_sha256": (
                            canonical_root_manifest["content_sha256"]
                        ),
                        "canonical_reconstruction_file_sha256": (
                            reconstruction_file_sha256
                        ),
                        "canonical_reconstruction_content_sha256": (
                            reconstruction_content_sha256
                        ),
                        "canonical_session_manifest_sha256": (
                            session_file_sha256
                        ),
                        "canonical_session_content_sha256": (
                            session_content_sha256
                        ),
                        "canonical_upstream_session_content_sha256": (
                            root_item["upstream_session_content_sha256"]
                        ),
                        "canonical_raw_portable_content_sha256": (
                            raw_consumption["portable_content_sha256"]
                        ),
                        "canonical_source_fingerprint": session_manifest[
                            "source_fingerprint"
                        ],
                        # Base signature fields retained for exact continuity.
                        "canonical_acquisition_session_manifest_sha256": (
                            root_item["upstream_session_content_sha256"]
                        ),
                        "canonical_acquisition_binding": root_contract,
                        "acquisition_reconstruction_content_sha256": (
                            reconstruction_content_sha256
                        ),
                        "dataset_root": str(dataset_root),
                        "dataset_manifest_sha256": dataset_manifest_sha256,
                        "dataset_catalogue_sha256": dataset_catalogue_sha256,
                        "cache_offset": offset,
                        "declared_window_count": declared_window_count,
                        "selected_rows_sha256": hashlib.sha256(
                            selected.tobytes()
                        ).hexdigest(),
                        "valid_only": False,
                        "reference_mapping_available": (
                            reference_mapping_available
                        ),
                        "components": int(args.components),
                        "nfft": int(args.nfft),
                        "n_iter": int(args.n_iter),
                        "variant_names": list(DEFAULT_SVD_VARIANTS),
                        "pipeline_sha256": pipeline_digest,
                        "force": False,
                        "output_dir": str(staging_root / session_id),
                    },
                )
            )
            offset += declared_window_count
        if offset != V3_EXPECTED_ROW_COUNT:
            raise ValueError(
                f"version-3 canonical cache must contain exact {V3_EXPECTED_ROW_COUNT} rows"
            )

        tasks = [task for _, task in task_inputs]
        results: list[dict[str, Any]] = []
        if int(args.workers) == 1:
            for task in tasks:
                print(f"[{task['session_id']}] building V3 diagnostic SVD", flush=True)
                result = _build_v3_session(task)
                results.append(result)
                attempt["completed_session_ids"].append(result["session_id"])
        else:
            with ProcessPoolExecutor(max_workers=max(1, int(args.workers))) as executor:
                futures = {
                    executor.submit(_build_v3_session, task): task for task in tasks
                }
                for future in as_completed(futures):
                    task = futures[future]
                    result = future.result()
                    results.append(result)
                    attempt["completed_session_ids"].append(result["session_id"])
                    print(
                        f"[{task['session_id']}] ok rows={result['row_count']}",
                        flush=True,
                    )
        order = {session_id: index for index, session_id in enumerate(root_ids)}
        results.sort(key=lambda value: order[str(value["session_id"])])
        if (
            [str(item["session_id"]) for item in results] != root_ids
            or any(item.get("status") != "ok" for item in results)
            or sum(int(item.get("row_count", 0)) for item in results)
            != V3_EXPECTED_ROW_COUNT
            or any(item.get("scientific_eligible") is not False for item in results)
            or any(item.get("training_authorized") is not False for item in results)
        ):
            raise RuntimeError("version-3 SVD result cohort is incomplete")

        _assert_pipeline_unchanged(pipeline_paths, pipeline_digest)
        final_root_payload, final_root_sha256 = _stable_regular_file_payload(
            canonical_root / "manifest.json",
            label="final canonical V3 root publication barrier",
        )
        final_reconstruction_payload, final_reconstruction_sha256 = (
            _stable_regular_file_payload(
                reconstruction_path,
                label="final canonical V3 reconstruction publication barrier",
            )
        )
        if (
            final_root_payload != canonical_root_payload
            or final_root_sha256 != canonical_root_manifest_sha256
            or final_reconstruction_payload != reconstruction_payload
            or final_reconstruction_sha256 != reconstruction_file_sha256
            or _canonical_sha256(build_dataset_manifest(dataset_root).to_dict())
            != dataset_manifest_sha256
            or _canonical_sha256(
                _validate_v3_dataset_catalogue(
                    dataset_root,
                    manifest=build_dataset_manifest(dataset_root),
                )
            )
            != dataset_catalogue_sha256
        ):
            raise RuntimeError("version-3 SVD input generation changed before publication")
        for task, result in zip(tasks, results, strict=True):
            session_id = str(task["session_id"])
            canonical_payload, canonical_sha256 = _stable_regular_file_payload(
                canonical_root / session_id / "manifest.json",
                label=f"final canonical V3 session barrier {session_id}",
            )
            if (
                canonical_sha256 != task["canonical_session_manifest_sha256"]
                or _strict_json_payload(
                    canonical_payload,
                    label=f"final canonical V3 session {session_id}",
                ).get("content_sha256")
                != task["canonical_session_content_sha256"]
            ):
                raise RuntimeError(
                    f"{session_id}: canonical session changed before publication"
                )
            output_manifest_payload, output_manifest_file_sha256 = (
                _stable_regular_file_payload(
                    staging_root / session_id / "manifest.json",
                    label=f"final V3 SVD session manifest {session_id}",
                )
            )
            output_manifest = _strict_json_payload(
                output_manifest_payload,
                label=f"final V3 SVD session manifest {session_id}",
            )
            if (
                output_manifest_file_sha256 != result["manifest_sha256"]
                or output_manifest.get("content_sha256")
                != result["manifest_content_sha256"]
                or output_manifest.get("session_signature")
                != _session_signature(task)
                or not _v3_output_manifest_is_current(
                    staging_root / session_id, output_manifest
                )
            ):
                raise RuntimeError(
                    f"{session_id}: V3 SVD output changed before publication"
                )

        root_target_firewall = {
            "schema_version": V3_SVD_TARGET_FIREWALL_SCHEMA,
            "all_windows_required": True,
            "target_dependent_row_selection": False,
            "row_selection_label_inputs": [],
            "feature_value_label_inputs": [],
            "mapping_used_for_window_support": False,
            "biopac_semantic_values_used": False,
            "reference_values_used": False,
            "training_authorized": False,
        }
        root_manifest: dict[str, Any] = {
            "schema_version": V3_SVD_ROOT_SCHEMA,
            "content_sha256": "",
            "pipeline_version": V3_SVD_PIPELINE_VERSION,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "diagnostic_only": True,
            "scientific_eligible": False,
            "training_authorized": False,
            "dataset_root": str(dataset_root),
            "dataset_manifest_sha256": dataset_manifest_sha256,
            "dataset_catalogue_sha256": dataset_catalogue_sha256,
            "canonical_cache": str(canonical_root),
            "canonical_cache_root_manifest_sha256": (
                canonical_root_manifest_sha256
            ),
            "canonical_cache_root_content_sha256": canonical_root_manifest[
                "content_sha256"
            ],
            "canonical_cache_acquisition_contract": root_contract,
            "canonical_reconstruction_file_sha256": reconstruction_file_sha256,
            "canonical_reconstruction_content_sha256": (
                reconstruction_content_sha256
            ),
            "pipeline_sha256": pipeline_digest,
            "execution_source_generation": _v3_execution_source_generation(
                pipeline_digest
            ),
            "all_windows": True,
            "valid_only": False,
            "target_dependent_row_selection": False,
            "subjects_filter_applied": False,
            "selection_scope": "full_cohort_diagnostic",
            "full_cohort_complete": True,
            "expected_session_ids": root_ids,
            "expected_session_ids_sha256": _canonical_sha256(root_ids),
            "selected_session_ids": root_ids,
            "selected_session_ids_sha256": _canonical_sha256(root_ids),
            "session_count": V3_EXPECTED_SESSION_COUNT,
            "row_count": V3_EXPECTED_ROW_COUNT,
            "mapped_session_count": int(
                sum(bool(item["reference_mapping_available"]) for item in results)
            ),
            "unmapped_session_count": int(
                sum(not bool(item["reference_mapping_available"]) for item in results)
            ),
            "variant_names": list(DEFAULT_SVD_VARIANTS),
            "attribute_names": list(ATTRIBUTE_NAMES),
            "components": int(args.components),
            "nfft": int(args.nfft),
            "n_iter": int(args.n_iter),
            "target_firewall": root_target_firewall,
            "target_firewall_sha256": _canonical_sha256(root_target_firewall),
            "session_inventory_aggregate_sha256": _canonical_sha256(
                {
                    str(item["session_id"]): item["inventory_sha256"]
                    for item in results
                }
            ),
            "sessions": results,
        }
        if (
            root_manifest["mapped_session_count"] != 18
            or root_manifest["unmapped_session_count"] != 11
        ):
            raise RuntimeError("version-3 mapping-availability cohort count differs")
        root_manifest["content_sha256"] = _canonical_content_sha256(root_manifest)
        _private_atomic_json(staging_root / "manifest.json", root_manifest)
        published_root_payload, _ = _stable_regular_file_payload(
            staging_root / "manifest.json",
            label="staged V3 SVD root manifest",
        )
        if _strict_json_payload(
            published_root_payload, label="staged V3 SVD root manifest"
        ) != root_manifest:
            raise RuntimeError("staged V3 SVD root manifest differs from memory")
        final_root = _publish_v3_attempt(attempt)
        print(
            f"Built exact {V3_EXPECTED_SESSION_COUNT}-session/"
            f"{V3_EXPECTED_ROW_COUNT}-row diagnostic V3 SVD cache in {final_root}",
            flush=True,
        )
        return 0
    except BaseException as error:
        if attempt.get("publication_committed") is True:
            _record_v3_postcommit_issue(attempt, error)
        else:
            _record_v3_failure(attempt, error)
        raise


def main() -> int:
    args = parse_args()
    dataset_input = Path(os.path.abspath(os.fspath(args.dataset_root)))
    canonical_input = Path(os.path.abspath(os.fspath(args.canonical_cache)))
    output_input = Path(os.path.abspath(os.fspath(args.output_dir)))
    (
        acquisition_v3,
        detected_root_manifest,
        detected_root_payload,
        detected_root_sha256,
    ) = _detect_v3_canonical_cache(canonical_input)
    if acquisition_v3:
        return _main_v3(
            args,
            dataset_root=dataset_input,
            canonical_root=canonical_input,
            output_root=output_input,
            canonical_root_manifest=detected_root_manifest,
            canonical_root_payload=detected_root_payload,
            canonical_root_manifest_sha256=detected_root_sha256,
        )

    dataset_root = args.dataset_root.resolve()
    canonical_root = args.canonical_cache.resolve()
    output_root = args.output_dir.resolve()
    if output_root == canonical_root:
        raise ValueError("experimental SVD cache must not overwrite the canonical cache")
    output_root.mkdir(parents=True, exist_ok=True)

    canonical_root_manifest = json.loads(
        (canonical_root / "manifest.json").read_text(encoding="utf-8")
    )
    canonical_root_manifest_sha256 = _sha256(canonical_root / "manifest.json")
    dataset_manifest = build_dataset_manifest(dataset_root)
    by_session = {subject.subject_id: subject for subject in dataset_manifest.usable_subjects}
    canonical_available = [
        item for item in canonical_root_manifest["sessions"] if item["status"] == "ok"
    ]
    canonical_available_session_ids = _unique_session_ids(
        [item.get("session_id") for item in canonical_available],
        location="canonical cache usable-session catalogue",
    )
    root_acquisition_binding = canonical_root_manifest.get("acquisition_contract")
    expected_session_ids = _canonical_expected_session_ids(
        canonical_available_session_ids,
        root_acquisition_binding,
    )
    subjects_filter_applied = args.subjects is not None
    available = list(canonical_available)
    if subjects_filter_applied:
        wanted = set(args.subjects)
        available = [item for item in available if item["session_id"] in wanted]
        missing = wanted - {item["session_id"] for item in available}
        if missing:
            raise KeyError(f"unknown or unusable sessions: {sorted(missing)}")
    if not available:
        raise ValueError("canonical cache contains no selected usable sessions")
    selected_session_ids = _unique_session_ids(
        [item.get("session_id") for item in available],
        location="selected canonical cache sessions",
    )

    acquisition_v2 = bool(
        isinstance(root_acquisition_binding, dict)
        and root_acquisition_binding.get("schema_version")
        == "snn_rr.feature_cache_acquisition.v2"
    )
    # One call validates the entire v2 reconstruction/cache graph and every
    # content-addressed session inventory.  Loading one selected session as a
    # memory map avoids concatenating the full cache while retaining the
    # fail-closed graph check.
    load_feature_cache(
        canonical_root,
        sessions=[str(available[0]["session_id"])],
        mmap=True,
        require_acquisition_contract=acquisition_v2,
    )
    acquisition_reconstruction_manifest: Path | None = None
    acquisition_reconstruction_content_sha256: str | None = None
    verified_reconstruction = None
    if acquisition_v2:
        assert isinstance(root_acquisition_binding, dict)
        reconstruction_value = root_acquisition_binding.get(
            "reconstruction_manifest"
        )
        if not isinstance(reconstruction_value, str) or not reconstruction_value:
            raise ValueError(
                "canonical acquisition cache lacks a reconstruction manifest path"
            )
        acquisition_reconstruction_manifest = Path(reconstruction_value)
        if not acquisition_reconstruction_manifest.is_absolute():
            acquisition_reconstruction_manifest = (
                canonical_root / acquisition_reconstruction_manifest
            )
        acquisition_reconstruction_manifest = (
            acquisition_reconstruction_manifest.resolve()
        )
        verified_reconstruction = load_acquisition_reconstruction(
            acquisition_reconstruction_manifest
        )
        acquisition_reconstruction_content_sha256 = (
            verified_reconstruction.content_sha256
        )
        if acquisition_reconstruction_content_sha256 != root_acquisition_binding.get(
            "reconstruction_content_sha256"
        ):
            raise ValueError(
                "canonical cache/reconstruction content hash mismatch"
            )

    pipeline_paths = _pipeline_paths()
    pipeline_digest = _pipeline_sha256(pipeline_paths)
    offsets: dict[str, int] = {}
    offset = 0
    for item in [
        value for value in canonical_root_manifest["sessions"] if value["status"] == "ok"
    ]:
        offsets[item["session_id"]] = offset
        offset += int(item["window_count"])

    tasks: list[dict[str, Any]] = []
    for item in available:
        session_id = item["session_id"]
        subject = by_session[session_id]
        if subject.selected_session is None:
            raise RuntimeError(f"{session_id} has no selected radar session")
        canonical_dir = canonical_root / session_id
        canonical_session_manifest_path = canonical_dir / "manifest.json"
        canonical_session_manifest = json.loads(
            canonical_session_manifest_path.read_text(encoding="utf-8")
        )
        canonical_acquisition_binding = _canonical_acquisition_binding(
            canonical_session_manifest
        )
        metadata = pd.read_csv(canonical_dir / "metadata.csv")
        selected = (
            np.flatnonzero(metadata["reference_valid"].to_numpy(dtype=bool))
            if not args.all_windows
            else np.arange(len(metadata), dtype=np.int64)
        )
        selected_digest = hashlib.sha256(selected.tobytes()).hexdigest()
        tasks.append(
            {
                "session_id": session_id,
                "recording_dirs": [
                    str(subject.selected_session.radars[radar].recording_dir)
                    for radar in (1, 2, 3)
                ],
                "radar_start_epochs_s": [
                    float(subject.selected_session.radars[radar].start_epoch_ms)
                    / 1000.0
                    for radar in (1, 2, 3)
                ],
                "canonical_dir": str(canonical_dir),
                "canonical_root_manifest_sha256": canonical_root_manifest_sha256,
                "canonical_source_fingerprint": item["source_fingerprint"],
                "canonical_session_manifest_sha256": _sha256(
                    canonical_session_manifest_path
                ),
                "canonical_acquisition_session_manifest_sha256": (
                    _acquisition_session_manifest_sha256(
                        canonical_acquisition_binding
                    )
                ),
                "canonical_acquisition_binding": canonical_acquisition_binding,
                "acquisition_reconstruction_manifest": (
                    None
                    if acquisition_reconstruction_manifest is None
                    else str(acquisition_reconstruction_manifest)
                ),
                "acquisition_reconstruction_content_sha256": (
                    acquisition_reconstruction_content_sha256
                ),
                "dataset_root": str(dataset_root),
                "cache_offset": offsets[session_id],
                "output_dir": str(output_root / session_id),
                "selected_rows_sha256": selected_digest,
                "valid_only": not args.all_windows,
                "components": int(args.components),
                "nfft": int(args.nfft),
                "n_iter": int(args.n_iter),
                "variant_names": list(DEFAULT_SVD_VARIANTS),
                "pipeline_sha256": pipeline_digest,
                "force": bool(args.force),
            }
        )

    results: list[dict[str, Any]] = []
    if args.workers == 1:
        for task in tasks:
            print(f"[{task['session_id']}] building", flush=True)
            results.append(_build_session(task))
    else:
        with ProcessPoolExecutor(max_workers=max(1, int(args.workers))) as executor:
            futures = {executor.submit(_build_session, task): task for task in tasks}
            for future in as_completed(futures):
                task = futures[future]
                result = future.result()
                results.append(result)
                print(
                    f"[{task['session_id']}] {result['status']} "
                    f"rows={result.get('row_count', 0)} cached={result.get('cached', False)}",
                    flush=True,
                )
    order = {task["session_id"]: index for index, task in enumerate(tasks)}
    results.sort(key=lambda value: order[value["session_id"]])
    _assert_pipeline_unchanged(pipeline_paths, pipeline_digest)
    if _sha256(canonical_root / "manifest.json") != canonical_root_manifest_sha256:
        raise RuntimeError(
            "canonical cache root changed while the SVD cache was being built"
        )
    final_reconstruction = None
    if acquisition_v2:
        assert acquisition_reconstruction_manifest is not None
        final_reconstruction = load_acquisition_reconstruction(
            acquisition_reconstruction_manifest
        )
        if final_reconstruction.content_sha256 != (
            acquisition_reconstruction_content_sha256
        ):
            raise RuntimeError(
                "acquisition reconstruction changed while SVD was being built"
            )
    result_by_session = {
        str(item["session_id"]): item for item in results
    }
    for task in tasks:
        session_id = str(task["session_id"])
        canonical_dir = Path(task["canonical_dir"])
        session_manifest_path = canonical_dir / "manifest.json"
        if _sha256(session_manifest_path) != task[
            "canonical_session_manifest_sha256"
        ]:
            raise RuntimeError(
                f"{session_id}: canonical session changed while SVD was being built"
            )
        if acquisition_v2:
            current_session_manifest = json.loads(
                session_manifest_path.read_text(encoding="utf-8")
            )
            _verify_bound_canonical_file(
                canonical_dir,
                current_session_manifest,
                "metadata",
                "metadata.csv",
            )
            _verify_bound_canonical_file(
                canonical_dir,
                current_session_manifest,
                "radar_timing_valid_mask",
                "radar_timing_valid_mask.npy",
            )
            assert final_reconstruction is not None
            session_contract = final_reconstruction.sessions.get(session_id)
            if session_contract is None:
                raise RuntimeError(
                    f"{session_id}: acquisition session disappeared during SVD build"
                )
            validate_raw_input_bindings(session_contract, dataset_root)
        result = result_by_session.get(session_id)
        if result is None:
            raise RuntimeError(f"{session_id}: SVD task result is missing")
        if result.get("status") == "ok":
            output_dir = Path(task["output_dir"])
            output_manifest_path = output_dir / "manifest.json"
            if not output_manifest_path.is_file():
                raise RuntimeError(
                    f"{session_id}: SVD output manifest is missing at publication"
                )
            output_manifest = json.loads(
                output_manifest_path.read_text(encoding="utf-8")
            )
            if output_manifest.get("session_signature") != _session_signature(task):
                raise RuntimeError(
                    f"{session_id}: SVD output signature changed before publication"
                )
            if not _cached_svd_manifest_is_current(
                output_dir,
                output_manifest,
                acquisition_v2=acquisition_v2,
            ):
                raise RuntimeError(
                    f"{session_id}: SVD output inventory changed before publication"
                )
            if result.get("content_sha256") != output_manifest.get(
                "content_sha256"
            ):
                raise RuntimeError(
                    f"{session_id}: in-memory/output SVD manifest mismatch"
                )
    selection_contract = _derive_output_selection_contract(
        expected_session_ids=expected_session_ids,
        selected_session_ids=selected_session_ids,
        results=results,
        subjects_filter_applied=subjects_filter_applied,
        canonical_acquisition_contract=root_acquisition_binding,
    )
    root_manifest = {
        "pipeline_version": PIPELINE_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_root": str(dataset_root),
        "canonical_cache": str(canonical_root),
        "canonical_manifest_sha256": canonical_root_manifest_sha256,
        "canonical_acquisition_contract": canonical_root_manifest.get(
            "acquisition_contract"
        ),
        "canonical_acquisition_reconstruction_content_sha256": (
            canonical_root_manifest.get("acquisition_contract", {}).get(
                "reconstruction_content_sha256"
            )
            if isinstance(canonical_root_manifest.get("acquisition_contract"), dict)
            else None
        ),
        "pipeline_sha256": pipeline_digest,
        "valid_only": not args.all_windows,
        "target_dependent_row_selection": bool(not args.all_windows),
        "row_selection_label_input": (
            "canonical_metadata.reference_valid"
            if not args.all_windows
            else None
        ),
        "variant_names": list(DEFAULT_SVD_VARIANTS),
        "attribute_names": list(ATTRIBUTE_NAMES),
        "components": int(args.components),
        "nfft": int(args.nfft),
        "n_iter": int(args.n_iter),
        "row_count": int(sum(item.get("row_count", 0) for item in results)),
        "sessions": results,
        **selection_contract,
    }
    _assert_pipeline_unchanged(pipeline_paths, pipeline_digest)
    root_manifest["content_sha256"] = _canonical_sha256(root_manifest)
    _write_json(output_root / "manifest.json", root_manifest)
    print(
        f"Built {sum(item['status'] == 'ok' for item in results)} sessions and "
        f"{root_manifest['row_count']} rows in {output_root}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
