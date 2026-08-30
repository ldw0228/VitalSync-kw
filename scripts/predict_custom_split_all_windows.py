#!/usr/bin/env python3
"""Label-free inference for one immutable custom identity-split checkpoint.

Unlike the six-fold deployment predictor, this entry point owns exactly one
``prediction`` identity partition.  It deliberately predicts every cache row
for those identities (including rows without a valid reference) and treats
reference values only as masked output metadata.  The split manifest, its
referenced cache/fold artifacts, the checkpoint split, and the training-fitted
auxiliary scaler all fail closed before model construction.
"""

from __future__ import annotations

import argparse
import ctypes
from dataclasses import dataclass
from datetime import datetime, timezone
import errno
import fcntl
import hashlib
import io
import json
import os
from pathlib import Path
import secrets
import stat
import sys
import tempfile
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from scripts.predict_all_windows import (  # noqa: E402
    _cache_inventory_sha256,
    _assert_publication_sources_current,
    _as_numpy_scaler,
    _sha256_file,
    predict_label_free,
    validate_cache_sources,
    validate_label_free_forward_interface,
    validate_model_kwargs,
)
from scripts.train import (  # noqa: E402
    FeatureCache,
    append_mask_aware_causal_history_features,
    build_model,
    fit_aux_scaler,
    infer_auxiliary_layout,
    load_feature_cache,
    make_loader,
    transform_aux,
)
from snn_rr.acquisition_contract import (  # noqa: E402
    load_acquisition_reconstruction,
)
from snn_rr.split_authority import (  # noqa: E402
    IdentitySplitAuthority,
    load_identity_split_authority,
)


FORMAT_VERSION = 1
OUTPUT_FIELDS = (
    "cache_index",
    "session_id",
    "identity",
    "protocol",
    "window_number",
    "reference_valid",
    "reference_rr_bpm",
    "prediction",
    "map_prediction",
    "rr_std",
    "uncertainty",
    "quality",
    "alias_probability",
    "posterior_entropy",
    "topk_rr",
    "topk_probability",
    "posterior_probability",
    "spike_rate",
    "radar_weights",
)
_FICLONE = 0x40049409
_AT_FDCWD = -100
_AT_SYMLINK_FOLLOW = 0x400
_RENAME_NOREPLACE = 1
_RENAME_EXCHANGE = 2

RUNTIME_SOURCE_PATHS = (
    Path(__file__),
    PROJECT_ROOT / "scripts/__init__.py",
    PROJECT_ROOT / "scripts/predict_all_windows.py",
    PROJECT_ROOT / "scripts/train.py",
    PROJECT_ROOT / "scripts/build_features.py",
    PROJECT_ROOT / "src/snn_rr/cache.py",
    PROJECT_ROOT / "src/snn_rr/__init__.py",
    PROJECT_ROOT / "src/snn_rr/models.py",
    PROJECT_ROOT / "src/snn_rr/metrics.py",
    PROJECT_ROOT / "src/snn_rr/data.py",
    PROJECT_ROOT / "src/snn_rr/preprocess.py",
    PROJECT_ROOT / "src/snn_rr/acquisition_contract.py",
    PROJECT_ROOT / "src/snn_rr/acquisition_protocol.py",
    PROJECT_ROOT / "src/snn_rr/synchronization.py",
    PROJECT_ROOT / "src/snn_rr/radar_timing.py",
    PROJECT_ROOT / "src/snn_rr/range_tracking.py",
    PROJECT_ROOT / "src/snn_rr/split_authority.py",
)


def _load_json(path: Path) -> dict[str, Any]:
    try:
        result = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"invalid JSON file {path}: {exc}") from exc
    if not isinstance(result, dict):
        raise RuntimeError(f"JSON root must be an object: {path}")
    return result


def _read_regular_snapshot(path: Path) -> tuple[bytes, str]:
    """Read one inode-backed immutable execution snapshot and its exact hash."""

    resolved = path.expanduser().resolve()
    try:
        with resolved.open("rb") as handle:
            before = os.fstat(handle.fileno())
            payload = handle.read()
            after = os.fstat(handle.fileno())
    except OSError as error:
        raise RuntimeError(f"cannot snapshot execution input {resolved}: {error}") from error
    if not (
        stat.S_ISREG(before.st_mode)
        and before.st_dev == after.st_dev
        and before.st_ino == after.st_ino
        and before.st_size == after.st_size == len(payload)
        and before.st_mtime_ns == after.st_mtime_ns
        and before.st_ctime_ns == after.st_ctime_ns
        and before.st_nlink == after.st_nlink == 1
    ):
        raise RuntimeError(f"execution input changed while snapshotted: {resolved}")
    return payload, hashlib.sha256(payload).hexdigest()


def _stable_regular_file_observation(
    path: Path,
    *,
    label: str,
    retain_payload: bool = False,
) -> tuple[bytes | None, dict[str, Any]]:
    """Hash one stable inode and retain change-sensitive namespace state.

    Content hashes alone cannot detect an in-place rewrite followed by a byte-
    exact restoration while inference is running.  The inode timestamps and
    link count are therefore part of the in-process publication barrier.  The
    portable receipt still contains the exact byte count and SHA-256 as well.
    """

    resolved = path.expanduser().resolve()
    chunks: list[bytes] | None = [] if retain_payload else None
    digest = hashlib.sha256()
    total = 0
    try:
        with resolved.open("rb") as handle:
            before = os.fstat(handle.fileno())
            while chunk := handle.read(4 * 1024 * 1024):
                digest.update(chunk)
                total += len(chunk)
                if chunks is not None:
                    chunks.append(chunk)
            after = os.fstat(handle.fileno())
    except OSError as error:
        raise RuntimeError(f"cannot bind {label} {resolved}: {error}") from error
    if not (
        stat.S_ISREG(before.st_mode)
        and before.st_dev == after.st_dev
        and before.st_ino == after.st_ino
        and before.st_size == after.st_size == total
        and before.st_mtime_ns == after.st_mtime_ns
        and before.st_ctime_ns == after.st_ctime_ns
        and before.st_nlink == after.st_nlink
        and before.st_nlink >= 1
    ):
        raise RuntimeError(f"{label} changed while it was bound: {resolved}")
    return (
        None if chunks is None else b"".join(chunks),
        {
            "sha256": digest.hexdigest(),
            "bytes": total,
            "device": before.st_dev,
            "inode": before.st_ino,
            "mtime_ns": before.st_mtime_ns,
            "ctime_ns": before.st_ctime_ns,
            "links": before.st_nlink,
        },
    )


def _capture_direct_entry_disk_binding(path: Path) -> dict[str, Any]:
    """Capture stable direct-entry source bytes during module initialization.

    This point-in-time disk binding is intentionally not described as the
    bytes actually compiled by the loader.  Complete executed-code closure
    requires launching a fresh isolated child from a private snapshot of every
    importable source file.
    """

    resolved = path.expanduser().resolve()
    payload, digest = _read_regular_snapshot(resolved)
    return {"path": str(resolved), "sha256": digest, "bytes": len(payload)}


_DIRECT_ENTRY_DISK_BINDING = _capture_direct_entry_disk_binding(Path(__file__))


def _assert_direct_entry_disk_binding_current(
    expected: Mapping[str, Any] = _DIRECT_ENTRY_DISK_BINDING,
) -> None:
    observed = _capture_direct_entry_disk_binding(
        Path(str(expected.get("path", "")))
    )
    if observed != dict(expected):
        raise RuntimeError(
            "direct inference entry source changed since its initialization-time "
            "disk binding; "
            "a fresh isolated private-source process is required"
        )


def _clear_private_directory_fd(descriptor: int) -> None:
    """Remove a private tree without depending on its current absolute path."""

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
    """Private directory whose cleanup survives parent-path rename/rebinding."""

    path: Path
    name: str
    parent_descriptor: int
    root_descriptor: int
    root_device: int
    root_inode: int
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
                raise RuntimeError("cannot allocate private snapshot directory")
            root_descriptor = os.open(
                name,
                os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW,
                dir_fd=parent_descriptor,
            )
            details = os.fstat(root_descriptor)
            if not stat.S_ISDIR(details.st_mode):
                raise RuntimeError("private snapshot root is not a directory")
            return cls(
                path=parent_path / name,
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

    def cleanup(self) -> None:
        if self.cleaned:
            return
        cleanup_error: BaseException | None = None
        try:
            details = os.fstat(self.root_descriptor)
            if (
                details.st_dev != self.root_device
                or details.st_ino != self.root_inode
                or not stat.S_ISDIR(details.st_mode)
            ):
                raise RuntimeError("private snapshot directory identity changed")
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
                # The directory may already have been unlinked after becoming
                # empty.  Absence of its exact inode is a completed cleanup.
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
            raise RuntimeError("private feature-cache snapshot cleanup failed") from cleanup_error


@dataclass(slots=True)
class _FeatureCacheSnapshot:
    """Private exact bytes used for model-forward cache tensors/metadata."""

    temporary: _StablePrivateDirectory
    source_cache_dir: Path
    cache_dir: Path
    session_ids: tuple[str, ...]
    root_is_v2: bool
    source_bindings: dict[str, dict[str, Any]]

    def _current_relative_files(self) -> tuple[str, ...]:
        result = ["manifest.json"]
        for session_id in self.session_ids:
            session = self.source_cache_dir / session_id
            result.extend(
                f"{session_id}/{name}"
                for name in (
                    "manifest.json",
                    "maps.npy",
                    "aux.npy",
                    "metadata.csv",
                    "frequencies_hz.npy",
                )
            )
            for optional in ("radar_timing_valid_mask.npy", "range_aux.npy"):
                if (session / optional).is_file():
                    result.append(f"{session_id}/{optional}")
        return tuple(sorted(result))

    def assert_source_current(self) -> None:
        if self._current_relative_files() != tuple(sorted(self.source_bindings)):
            raise RuntimeError("feature-cache input topology changed during inference")
        for relative, expected in self.source_bindings.items():
            path = self.source_cache_dir / relative
            try:
                _, observed = _stable_regular_file_observation(
                    path,
                    label=f"feature-cache input {relative}",
                )
            except (OSError, RuntimeError) as error:
                raise RuntimeError(
                    f"feature-cache input became unreadable: {relative}"
                ) from error
            if observed != expected:
                raise RuntimeError(
                    f"feature-cache input changed during inference: {relative}"
                )

    def assert_private_current(self) -> None:
        for relative, expected in self.source_bindings.items():
            path = self.cache_dir / relative
            try:
                observed = {
                    "sha256": _sha256_file(path),
                    "bytes": path.stat().st_size,
                }
            except OSError as error:
                raise RuntimeError(
                    f"private feature-cache snapshot became unreadable: {relative}"
                ) from error
            if observed != {
                "sha256": expected["sha256"],
                "bytes": expected["bytes"],
            }:
                raise RuntimeError(
                    f"private feature-cache snapshot changed: {relative}"
                )

    def cleanup(self) -> None:
        self.temporary.cleanup()


@dataclass(frozen=True, slots=True)
class _OutputIsolationGuard:
    """Read-only inode/tree closure that output publication may not touch."""

    protected_files: tuple[Path, ...]
    protected_trees: tuple[Path, ...]
    protected_inodes: frozenset[tuple[int, int]]

    def assert_disjoint(self, output_path: Path) -> None:
        output = output_path.expanduser().resolve()
        for tree in self.protected_trees:
            if output == tree or output.is_relative_to(tree):
                raise RuntimeError(
                    f"output path intrudes protected inference input tree: {tree}"
                )
            if tree.is_relative_to(output):
                raise RuntimeError(
                    f"output path would replace an ancestor of protected inputs: {output}"
                )
        if output in self.protected_files:
            raise RuntimeError(f"output aliases protected inference input: {output}")
        # Lexical/resolved containment catches ordinary paths and symlinks.
        # Inode checks on every existing output ancestor additionally reject a
        # bind-mounted alias of a protected cache/source directory.
        for ancestor in output.parents:
            try:
                ancestor_details = ancestor.stat()
            except FileNotFoundError:
                continue
            except OSError as error:
                raise RuntimeError(
                    f"cannot inspect output parent path: {ancestor}"
                ) from error
            if (ancestor_details.st_dev, ancestor_details.st_ino) in self.protected_inodes:
                raise RuntimeError(
                    f"output parent aliases protected inference input tree: {ancestor}"
                )
        try:
            details = output.stat()
        except FileNotFoundError:
            details = None
        except OSError as error:
            raise RuntimeError(f"cannot inspect output path: {output}") from error
        if details is not None:
            if stat.S_ISDIR(details.st_mode):
                raise RuntimeError("output path is an existing directory")
            if (details.st_dev, details.st_ino) in self.protected_inodes:
                raise RuntimeError(
                    f"output inode aliases protected inference input: {output}"
                )


def _json_snapshot(path: Path, label: str) -> dict[str, Any]:
    payload, _ = _read_regular_snapshot(path)
    try:
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"invalid {label} snapshot: {path}") from error
    if not isinstance(value, dict):
        raise RuntimeError(f"{label} snapshot root must be an object: {path}")
    return value


def _document_path(
    value: Any,
    *,
    relative_to: Path,
) -> Path | None:
    if not isinstance(value, str) or not value.strip():
        return None
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = relative_to / path
    return path.resolve()


def _authority_path_inside(root: Path, value: Any, label: str) -> Path:
    """Mirror the acquisition loader's contained-path authority rule."""

    if not isinstance(value, str) or not value:
        raise RuntimeError(f"{label} must be a non-empty path")
    candidate = (root / Path(value)).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError as error:
        raise RuntimeError(f"{label} escapes its authority root") from error
    if not candidate.is_file():
        raise RuntimeError(f"{label} is missing: {candidate}")
    return candidate


def _authority_external_path(root: Path, value: Any, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise RuntimeError(f"{label} must be a non-empty path")
    path = Path(value)
    if not path.is_absolute():
        path = root / path
    path = path.resolve()
    if not path.is_file():
        raise RuntimeError(f"{label} is missing: {path}")
    return path


def _capture_acquisition_authority_bindings(
    cache_dir: Path,
    *,
    cache_manifest: Mapping[str, Any] | None = None,
) -> dict[str, dict[str, Any]]:
    """Bind every non-raw file consumed through the acquisition graph.

    Cache tensors/manifests are bound by ``_FeatureCacheSnapshot`` and raw
    payloads by ``validate_cache_sources``.  This inventory closes the
    remaining reconstruction leaves: session manifests, synchronization
    receipts/approvals, range artifacts, and external acquisition authority
    inputs.  JSON path discovery always uses the same bytes that were hashed.
    """

    cache_root = cache_dir.expanduser().resolve()
    root_document = (
        dict(cache_manifest)
        if cache_manifest is not None
        else _json_snapshot(cache_root / "manifest.json", "cache manifest")
    )
    contract = root_document.get("acquisition_contract")
    if not isinstance(contract, Mapping):
        return {}
    reconstruction_value = contract.get("reconstruction_manifest")
    if not isinstance(reconstruction_value, str) or not reconstruction_value:
        raise RuntimeError("acquisition reconstruction path is missing")
    reconstruction_path = Path(reconstruction_value)
    if not reconstruction_path.is_absolute():
        reconstruction_path = cache_root / reconstruction_path
    reconstruction_path = reconstruction_path.resolve()

    bindings: dict[str, dict[str, Any]] = {}

    def bind(
        path: Path,
        role: str,
        *,
        parse_json: bool = False,
    ) -> dict[str, Any] | None:
        payload, observation = _stable_regular_file_observation(
            path,
            label=role,
            retain_payload=parse_json,
        )
        key = str(path.resolve())
        existing = bindings.get(key)
        if existing is None:
            record = dict(observation)
            record["roles"] = [role]
            bindings[key] = record
        else:
            existing_observation = dict(existing)
            roles = list(existing_observation.pop("roles", ()))
            if existing_observation != observation:
                raise RuntimeError(f"authority file changed between roles: {path}")
            if role not in roles:
                roles.append(role)
                roles.sort()
                existing["roles"] = roles
        if not parse_json:
            return None
        assert payload is not None
        try:
            document = json.loads(payload.decode("utf-8"))
        except (UnicodeError, json.JSONDecodeError) as error:
            raise RuntimeError(f"invalid {role}: {path}") from error
        if not isinstance(document, dict):
            raise RuntimeError(f"{role} root must be an object: {path}")
        return document

    reconstruction = bind(
        reconstruction_path,
        "acquisition reconstruction manifest",
        parse_json=True,
    )
    assert reconstruction is not None
    reconstruction_root = reconstruction_path.parent.resolve()

    for key in ("cohort_authority", "sync_config", "protocol_config", "spreadsheet"):
        value = reconstruction.get(key)
        if value is None:
            continue
        bind(
            _authority_external_path(
                reconstruction_root,
                value,
                f"acquisition {key}",
            ),
            f"acquisition {key}",
        )

    sessions = reconstruction.get("sessions")
    if not isinstance(sessions, list):
        raise RuntimeError("acquisition reconstruction sessions are missing")
    for entry in sessions:
        if not isinstance(entry, Mapping):
            raise RuntimeError("acquisition reconstruction session entry is malformed")
        session_id = entry.get("session_id")
        if not isinstance(session_id, str) or not session_id:
            raise RuntimeError("acquisition reconstruction session ID is malformed")
        session_path = _authority_path_inside(
            reconstruction_root,
            entry.get("manifest"),
            f"acquisition session manifest {session_id}",
        )
        session = bind(
            session_path,
            f"acquisition session manifest {session_id}",
            parse_json=True,
        )
        assert session is not None
        # The canonical loader consumes dependent artifacts only for usable
        # sessions, but consumes every session manifest before that decision.
        if session.get("usable") is not True:
            continue
        synchronization = session.get("synchronization")
        if not isinstance(synchronization, Mapping):
            raise RuntimeError(
                f"acquisition synchronization record is missing: {session_id}"
            )
        receipt_path = _authority_path_inside(
            reconstruction_root,
            synchronization.get("receipt"),
            f"synchronization receipt {session_id}",
        )
        bind(receipt_path, f"synchronization receipt {session_id}")
        approval_value = synchronization.get("manual_approval")
        if approval_value is not None:
            approval_path = _authority_path_inside(
                reconstruction_root,
                approval_value,
                f"manual synchronization approval {session_id}",
            )
            bind(approval_path, f"manual synchronization approval {session_id}")
        range_tracking = session.get("range_tracking")
        if (
            isinstance(range_tracking, Mapping)
            and range_tracking.get("status") == "built"
        ):
            range_path = _authority_path_inside(
                session_path.parent,
                range_tracking.get("artifact"),
                f"range tracking artifact {session_id}",
            )
            bind(range_path, f"range tracking artifact {session_id}")
    return dict(sorted(bindings.items()))


def _assert_acquisition_authority_bindings_current(
    cache_dir: Path,
    expected: Mapping[str, Mapping[str, Any]],
    *,
    cache_manifest: Mapping[str, Any] | None = None,
) -> None:
    observed = _capture_acquisition_authority_bindings(
        cache_dir,
        cache_manifest=cache_manifest,
    )
    if observed != {str(path): dict(binding) for path, binding in expected.items()}:
        raise RuntimeError(
            "acquisition authority graph changed during custom-split inference"
        )


def _revalidate_full_acquisition_authority(
    cache_dir: Path,
    *,
    cache_manifest: Mapping[str, Any],
    expected_bindings: Mapping[str, Mapping[str, Any]],
) -> None:
    """Replay the canonical full-cohort authority consumer before publication.

    This is deliberately independent of ``--verify-raw-sources``.  The v2
    feature-cache loader itself derives the complete 30-session dataset
    authority (including unusable sessions) and hashes the selected usable raw
    graph, so publication must re-run that same contract rather than silently
    weakening the loader's authority when optional diagnostic verification is
    disabled.
    """

    contract = cache_manifest.get("acquisition_contract")
    if not isinstance(contract, Mapping):
        if expected_bindings:
            raise RuntimeError("legacy cache unexpectedly has acquisition bindings")
        return
    reconstruction_value = contract.get("reconstruction_manifest")
    if not isinstance(reconstruction_value, str) or not reconstruction_value:
        raise RuntimeError("acquisition reconstruction path is missing")
    reconstruction_path = Path(reconstruction_value)
    if not reconstruction_path.is_absolute():
        reconstruction_path = cache_dir.expanduser().resolve() / reconstruction_path
    reconstruction_path = reconstruction_path.resolve()
    _assert_acquisition_authority_bindings_current(
        cache_dir,
        expected_bindings,
        cache_manifest=cache_manifest,
    )
    try:
        verified = load_acquisition_reconstruction(reconstruction_path)
    except (OSError, ValueError) as error:
        raise RuntimeError(
            "full acquisition authority changed during custom-split inference"
        ) from error
    if (
        verified.manifest_path.resolve() != reconstruction_path
        or verified.manifest.get("content_sha256")
        != contract.get("reconstruction_content_sha256")
    ):
        raise RuntimeError("full acquisition authority binding mismatch")
    _assert_acquisition_authority_bindings_current(
        cache_dir,
        expected_bindings,
        cache_manifest=cache_manifest,
    )


def _inode(path: Path) -> tuple[int, int]:
    try:
        details = path.stat()
    except OSError as error:
        raise RuntimeError(f"protected inference input became unreadable: {path}") from error
    return details.st_dev, details.st_ino


def _tree_inodes(tree: Path) -> set[tuple[int, int]]:
    result = {_inode(tree)}
    try:
        for directory, names, files in os.walk(tree, followlinks=False):
            base = Path(directory)
            for name in (*names, *files):
                result.add(_inode(base / name))
    except OSError as error:
        raise RuntimeError(f"cannot enumerate protected input tree: {tree}") from error
    return result


def _build_output_isolation_guard(
    *,
    output_path: Path,
    cache_dir: Path,
    checkpoint_path: Path,
    run_config_path: Path,
    split_manifest_path: Path,
    extra_files: Sequence[Path] = (),
) -> _OutputIsolationGuard:
    """Resolve every protected file/tree before any output-side mutation."""

    cache_root = cache_dir.expanduser().resolve()
    split_path = split_manifest_path.expanduser().resolve()
    protected_files: set[Path] = {
        checkpoint_path.expanduser().resolve(),
        run_config_path.expanduser().resolve(),
        split_path,
        *(path.expanduser().resolve() for path in RUNTIME_SOURCE_PATHS),
        *(path.expanduser().resolve() for path in extra_files),
    }
    protected_trees: set[Path] = {
        cache_root,
        (PROJECT_ROOT / "src").resolve(),
        (PROJECT_ROOT / "scripts").resolve(),
    }

    cache_manifest_path = cache_root / "manifest.json"
    cache_manifest = _json_snapshot(cache_manifest_path, "cache manifest")
    protected_files.add(cache_manifest_path.resolve())
    config_path = _document_path(
        cache_manifest.get("config"), relative_to=PROJECT_ROOT
    )
    dataset_root = _document_path(
        cache_manifest.get("dataset_root"), relative_to=PROJECT_ROOT
    )
    if config_path is not None and config_path.exists():
        protected_files.add(config_path)
    if dataset_root is not None and dataset_root.is_dir():
        protected_trees.add(dataset_root)
    contract = cache_manifest.get("acquisition_contract")
    if isinstance(contract, Mapping):
        reconstruction_path = _document_path(
            contract.get("reconstruction_manifest"), relative_to=cache_root
        )
        if reconstruction_path is not None:
            if not reconstruction_path.is_file():
                raise RuntimeError(
                    "bound acquisition reconstruction manifest is missing"
                )
            protected_files.add(reconstruction_path)
            # V2 reconstruction consumers resolve session manifests, sync
            # receipts, manual approvals, and range artifacts inside this
            # authority root.  Protecting the full tree closes those
            # transitively consumed paths without trusting an incomplete
            # hand-maintained list of session entries.
            protected_trees.add(reconstruction_path.parent.resolve())
            reconstruction = _json_snapshot(
                reconstruction_path, "acquisition reconstruction manifest"
            )
            reconstruction_sessions = reconstruction.get("sessions")
            if isinstance(reconstruction_sessions, list):
                for entry in reconstruction_sessions:
                    if not isinstance(entry, Mapping):
                        continue
                    session_manifest = _document_path(
                        entry.get("manifest"),
                        relative_to=reconstruction_path.parent,
                    )
                    if session_manifest is None:
                        continue
                    if not session_manifest.is_file():
                        raise RuntimeError(
                            "bound acquisition session manifest is missing"
                        )
                    protected_files.add(session_manifest)
            for key in (
                "cohort_authority",
                "sync_config",
                "protocol_config",
                "spreadsheet",
            ):
                authority_path = _document_path(
                    reconstruction.get(key), relative_to=reconstruction_path.parent
                )
                if authority_path is None:
                    continue
                if not authority_path.is_file():
                    raise RuntimeError(
                        f"bound acquisition authority file is missing: {key}"
                    )
                protected_files.add(authority_path)
            reconstruction_dataset = _document_path(
                reconstruction.get("dataset_root"),
                relative_to=reconstruction_path.parent,
            )
            if reconstruction_dataset is not None:
                if not reconstruction_dataset.is_dir():
                    raise RuntimeError(
                        "bound reconstruction dataset root is missing"
                    )
                protected_trees.add(reconstruction_dataset)

    split = _json_snapshot(split_path, "identity split manifest")
    fold_binding = split.get("fold_assignments")
    if isinstance(fold_binding, Mapping):
        fold_path = _document_path(
            fold_binding.get("path"), relative_to=split_path.parent
        )
        if fold_path is not None and fold_path.exists():
            protected_files.add(fold_path)
    cache_binding = split.get("cache")
    if isinstance(cache_binding, Mapping):
        bound_cache_manifest = _document_path(
            cache_binding.get("manifest_path"), relative_to=split_path.parent
        )
        if bound_cache_manifest is not None and bound_cache_manifest.exists():
            protected_files.add(bound_cache_manifest)

    file_inodes = {_inode(path) for path in protected_files}
    tree_inodes: set[tuple[int, int]] = set()
    for tree in protected_trees:
        if not tree.is_dir():
            raise RuntimeError(f"protected inference input tree is missing: {tree}")
        tree_inodes.update(_tree_inodes(tree))
    guard = _OutputIsolationGuard(
        protected_files=tuple(sorted(protected_files, key=str)),
        protected_trees=tuple(sorted(protected_trees, key=str)),
        protected_inodes=frozenset(file_inodes | tree_inodes),
    )
    guard.assert_disjoint(output_path)
    return guard


def _copy_regular_file_snapshot(source_path: Path, destination: Path) -> dict[str, Any]:
    """Copy/reflink one stable regular inode and hash the exact private bytes."""

    source = source_path.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    try:
        with source.open("rb") as source_stream, destination.open("x+b") as target:
            before = os.fstat(source_stream.fileno())
            if not stat.S_ISREG(before.st_mode):
                raise RuntimeError(f"feature-cache input is not regular: {source}")
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
                and before.st_size == after.st_size == copied
                and before.st_mtime_ns == after.st_mtime_ns
                and before.st_ctime_ns == after.st_ctime_ns
            ):
                raise RuntimeError(
                    f"feature-cache input changed while snapshotted: {source}"
                )
            os.fchmod(target.fileno(), 0o400)
    except BaseException:
        destination.unlink(missing_ok=True)
        raise
    return {
        "sha256": digest.hexdigest(),
        "bytes": copied,
        "device": before.st_dev,
        "inode": before.st_ino,
        "mtime_ns": before.st_mtime_ns,
        "ctime_ns": before.st_ctime_ns,
        "links": before.st_nlink,
    }


def _materialize_feature_cache_snapshot(
    cache_dir: Path, *, parent: Path | None = None
) -> _FeatureCacheSnapshot:
    """Freeze all cache files consumed by custom inference before loading."""

    source_root = cache_dir.expanduser().resolve()
    if not source_root.is_dir():
        raise FileNotFoundError(source_root)
    snapshot_parent = (
        Path(tempfile.gettempdir()).resolve() if parent is None else parent.resolve()
    )
    temporary = _StablePrivateDirectory.create(
        snapshot_parent,
        prefix=".custom-feature-cache-snapshot.",
    )
    private_root = temporary.path
    bindings: dict[str, dict[str, Any]] = {}

    def copy(relative: str) -> None:
        relative_path = Path(relative)
        if (
            relative_path.is_absolute()
            or len(relative_path.parts) < 1
            or any(part in {"", ".", ".."} for part in relative_path.parts)
        ):
            raise RuntimeError(f"unsafe feature-cache snapshot path: {relative}")
        source_path = (source_root / relative_path).resolve()
        destination_path = (private_root / relative_path).resolve()
        try:
            source_path.relative_to(source_root)
            destination_path.relative_to(private_root)
        except ValueError as error:
            raise RuntimeError(
                f"feature-cache snapshot path escapes its root: {relative}"
            ) from error
        bindings[relative] = _copy_regular_file_snapshot(
            source_path, destination_path
        )

    try:
        copy("manifest.json")
        try:
            root = json.loads(
                (private_root / "manifest.json").read_text(encoding="utf-8")
            )
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise RuntimeError("snapshotted feature-cache root manifest is invalid") from error
        if not isinstance(root, dict):
            raise RuntimeError("snapshotted feature-cache root must be an object")
        sessions = root.get("sessions")
        if not isinstance(sessions, list) or any(
            not isinstance(item, Mapping) for item in sessions
        ):
            raise RuntimeError("snapshotted feature-cache session catalogue is invalid")
        session_ids = tuple(
            str(item.get("session_id", ""))
            for item in sessions
            if item.get("status") == "ok"
        )
        if (
            not session_ids
            or any(not session_id for session_id in session_ids)
            or len(set(session_ids)) != len(session_ids)
        ):
            raise RuntimeError("snapshotted feature-cache usable sessions are invalid")
        for session_id in session_ids:
            session_path = Path(session_id)
            if (
                session_path.is_absolute()
                or len(session_path.parts) != 1
                or session_path.name != session_id
                or session_id in {".", ".."}
                or "/" in session_id
                or "\\" in session_id
            ):
                raise RuntimeError(
                    f"unsafe feature-cache session_id: {session_id!r}"
                )
        for session_id in session_ids:
            session = source_root / session_id
            for name in (
                "manifest.json",
                "maps.npy",
                "aux.npy",
                "metadata.csv",
                "frequencies_hz.npy",
            ):
                copy(f"{session_id}/{name}")
            for optional in ("radar_timing_valid_mask.npy", "range_aux.npy"):
                if (session / optional).is_file():
                    copy(f"{session_id}/{optional}")
        contract = root.get("acquisition_contract")
        result = _FeatureCacheSnapshot(
            temporary=temporary,
            source_cache_dir=source_root,
            cache_dir=private_root,
            session_ids=session_ids,
            root_is_v2=bool(
                isinstance(contract, Mapping)
                and contract.get("schema_version")
                == "snn_rr.feature_cache_acquisition.v2"
            ),
            source_bindings=bindings,
        )
        result.assert_private_current()
        result.assert_source_current()
        return result
    except BaseException:
        temporary.cleanup()
        raise


def _load_feature_cache_snapshot_payload(
    snapshot: _FeatureCacheSnapshot,
    *,
    validated_provenance: Any,
) -> FeatureCache:
    """Load only private bytes after the canonical live cache was validated."""

    selected = tuple(getattr(validated_provenance, "selected_sessions", ()))
    if selected != snapshot.session_ids:
        raise RuntimeError("validated cache sessions differ from private snapshot")
    inventory_hash, inventory_count = _cache_inventory_sha256(
        snapshot.cache_dir, snapshot.session_ids
    )
    if (
        inventory_hash != getattr(validated_provenance, "inventory_sha256", None)
        or inventory_count
        != getattr(validated_provenance, "inventory_file_count", None)
    ):
        raise RuntimeError("private feature-cache snapshot/provenance mismatch")

    maps: list[np.ndarray] = []
    aux: list[np.ndarray] = []
    metadata: list[Any] = []
    timing: list[np.ndarray] = []
    frequency_grid: np.ndarray | None = None
    for session_id in snapshot.session_ids:
        session = snapshot.cache_dir / session_id
        local_maps = np.load(session / "maps.npy", allow_pickle=False)
        local_aux = np.load(session / "aux.npy", allow_pickle=False)
        local_metadata = pd.read_csv(session / "metadata.csv")
        local_frequency = np.load(
            session / "frequencies_hz.npy", allow_pickle=False
        )
        local_timing = (
            np.load(
                session / "radar_timing_valid_mask.npy", allow_pickle=False
            )
            if snapshot.root_is_v2
            else None
        )
        if not (len(local_maps) == len(local_aux) == len(local_metadata)):
            raise RuntimeError(f"private feature-cache length mismatch: {session_id}")
        if local_timing is not None and len(local_timing) != len(local_maps):
            raise RuntimeError(
                f"private feature-cache timing length mismatch: {session_id}"
            )
        if frequency_grid is None:
            frequency_grid = np.asarray(local_frequency)
        elif not np.allclose(frequency_grid, local_frequency):
            raise RuntimeError(
                f"private feature-cache frequency grid mismatch: {session_id}"
            )
        maps.append(local_maps)
        aux.append(local_aux)
        metadata.append(local_metadata)
        if local_timing is not None:
            timing.append(local_timing)
    if frequency_grid is None:
        raise RuntimeError("private feature-cache contains no usable sessions")
    snapshot.assert_private_current()
    return FeatureCache(
        maps=maps[0] if len(maps) == 1 else np.concatenate(maps, axis=0),
        aux=aux[0] if len(aux) == 1 else np.concatenate(aux, axis=0),
        metadata=pd.concat(metadata, ignore_index=True),
        frequencies_hz=frequency_grid,
        provenance=validated_provenance,
        radar_timing_valid_mask=(
            None
            if not timing
            else timing[0]
            if len(timing) == 1
            else np.concatenate(timing, axis=0)
        ),
    )


def _same_regular_inode(details: os.stat_result, expected: os.stat_result) -> bool:
    return bool(
        stat.S_ISREG(details.st_mode)
        and details.st_dev == expected.st_dev
        and details.st_ino == expected.st_ino
    )


def _link_open_fd_noreplace(
    descriptor: int,
    parent_descriptor: int,
    destination_name: str,
) -> None:
    """Atomically link the exact open inode, never a mutable source pathname."""

    try:
        linkat = ctypes.CDLL(None, use_errno=True).linkat
    except (AttributeError, OSError, RuntimeError) as error:
        raise RuntimeError("fd-coupled atomic publication is unavailable") from error
    linkat.argtypes = (
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
    )
    linkat.restype = ctypes.c_int
    source = os.fsencode(f"/proc/self/fd/{descriptor}")
    result = linkat(
        _AT_FDCWD,
        source,
        parent_descriptor,
        os.fsencode(destination_name),
        _AT_SYMLINK_FOLLOW,
    )
    if result == 0:
        return
    error_number = ctypes.get_errno()
    if error_number == errno.EEXIST:
        raise FileExistsError(
            error_number,
            "output appeared concurrently; refusing to overwrite",
            destination_name,
        )
    raise OSError(error_number, os.strerror(error_number), destination_name)


def _rename_at2(
    left_descriptor: int,
    left_name: str,
    right_descriptor: int,
    right_name: str,
    flags: int,
) -> None:
    try:
        renameat2 = ctypes.CDLL(None, use_errno=True).renameat2
    except (AttributeError, OSError, RuntimeError) as error:
        raise RuntimeError("atomic exchange publication is unavailable") from error
    renameat2.argtypes = (
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    )
    renameat2.restype = ctypes.c_int
    result = renameat2(
        left_descriptor,
        os.fsencode(left_name),
        right_descriptor,
        os.fsencode(right_name),
        flags,
    )
    if result != 0:
        error_number = ctypes.get_errno()
        if flags == _RENAME_NOREPLACE and error_number == errno.EEXIST:
            raise FileExistsError(
                error_number,
                "output appeared concurrently; refusing to overwrite",
                right_name,
            )
        raise OSError(
            error_number,
            os.strerror(error_number),
            f"{left_name} <-> {right_name}",
        )


def _stat_name(parent_descriptor: int, name: str) -> os.stat_result:
    return os.stat(name, dir_fd=parent_descriptor, follow_symlinks=False)


def _same_inode(left: os.stat_result, right: os.stat_result) -> bool:
    return left.st_dev == right.st_dev and left.st_ino == right.st_ino


def _unlink_name(parent_descriptor: int, name: str) -> None:
    """Unlink one private random stage through the already-open directory."""

    try:
        unlinkat = ctypes.CDLL(None, use_errno=True).unlinkat
    except (AttributeError, OSError, RuntimeError) as error:
        raise RuntimeError("atomic stage cleanup is unavailable") from error
    unlinkat.argtypes = (ctypes.c_int, ctypes.c_char_p, ctypes.c_int)
    unlinkat.restype = ctypes.c_int
    if unlinkat(parent_descriptor, os.fsencode(name), 0) != 0:
        error_number = ctypes.get_errno()
        raise OSError(error_number, os.strerror(error_number), name)


def _exchange_stage_with_output(
    stage_descriptor: int,
    stage_name: str,
    parent_descriptor: int,
    output_name: str,
    expected_new: os.stat_result,
) -> None:
    """Publish from a private directory, preserving rollback on exchange."""

    for _ in range(8):
        stage_before = _stat_name(stage_descriptor, stage_name)
        if not _same_regular_inode(stage_before, expected_new):
            raise RuntimeError("fd-coupled publication stage changed before exchange")
        try:
            output_before = _stat_name(parent_descriptor, output_name)
        except FileNotFoundError:
            try:
                _rename_at2(
                    stage_descriptor,
                    stage_name,
                    parent_descriptor,
                    output_name,
                    _RENAME_NOREPLACE,
                )
            except FileExistsError:
                continue
            output_after = _stat_name(parent_descriptor, output_name)
            if not _same_regular_inode(output_after, expected_new):
                raise RuntimeError("no-replace force publication inode mismatch")
            return
        try:
            _rename_at2(
                stage_descriptor,
                stage_name,
                parent_descriptor,
                output_name,
                _RENAME_EXCHANGE,
            )
        except FileNotFoundError:
            continue
        output_after = _stat_name(parent_descriptor, output_name)
        if not _same_regular_inode(output_after, expected_new):
            # If the old output is still parked in the private directory,
            # exchange back before failing.
            try:
                stage_after = _stat_name(stage_descriptor, stage_name)
                if not _same_inode(stage_after, output_before):
                    raise RuntimeError("old output is no longer recoverable from stage")
                _rename_at2(
                    stage_descriptor,
                    stage_name,
                    parent_descriptor,
                    output_name,
                    _RENAME_EXCHANGE,
                )
                restored = _stat_name(parent_descriptor, output_name)
                if not _same_inode(restored, output_before):
                    raise RuntimeError("old output rollback verification failed")
            except BaseException as rollback_error:
                raise RuntimeError(
                    "atomic exchange published a foreign inode and rollback failed"
                ) from rollback_error
            raise RuntimeError(
                "atomic exchange rejected a foreign stage and restored the old output"
            )
        stage_after = _stat_name(stage_descriptor, stage_name)
        if not _same_inode(stage_after, output_before):
            raise RuntimeError(
                "private replaced-output stage changed; published output retained"
            )
        _unlink_name(stage_descriptor, stage_name)
        return
    raise RuntimeError("force publication could not stabilize the output namespace")


def _assert_output_parent_fd_current(
    output_parent: Path,
    parent_descriptor: int,
) -> None:
    captured = os.fstat(parent_descriptor)
    current = output_parent.stat()
    if not (
        stat.S_ISDIR(captured.st_mode)
        and stat.S_ISDIR(current.st_mode)
        and _same_inode(captured, current)
    ):
        raise RuntimeError("output parent changed during atomic publication")


def _make_private_stage_directory(
    parent_descriptor: int,
    output_name: str,
) -> tuple[str, int]:
    for _ in range(128):
        name = f".{output_name}.{secrets.token_hex(16)}.stage"
        try:
            os.mkdir(name, 0o700, dir_fd=parent_descriptor)
        except FileExistsError:
            continue
        descriptor = os.open(
            name,
            os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW,
            dir_fd=parent_descriptor,
        )
        details = os.fstat(descriptor)
        if not stat.S_ISDIR(details.st_mode):
            os.close(descriptor)
            raise RuntimeError("private publication stage is not a directory")
        return name, descriptor
    raise RuntimeError("cannot allocate a private publication directory")


def _atomic_npz(
    path: Path,
    arrays: Mapping[str, Any],
    *,
    before_replace: Callable[[], None] | None = None,
    replace_existing: bool = True,
) -> str:
    """Durably publish exact serialized bytes from an unnamed open inode."""

    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        parent_descriptor = os.open(
            path.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC
        )
        _assert_output_parent_fd_current(path.parent, parent_descriptor)
    except (AttributeError, OSError, RuntimeError) as error:
        try:
            os.close(parent_descriptor)
        except (NameError, OSError):
            pass
        raise RuntimeError("output parent cannot be bound for publication") from error
    private_name = ""
    private_descriptor = -1
    descriptor = -1
    try:
        private_name, private_descriptor = _make_private_stage_directory(
            parent_descriptor,
            path.name,
        )
        descriptor = os.open(
            ".",
            os.O_RDWR | os.O_CLOEXEC | os.O_TMPFILE,
            0o600,
            dir_fd=private_descriptor,
        )
    except (AttributeError, OSError, RuntimeError) as error:
        if private_descriptor >= 0:
            os.close(private_descriptor)
        if private_name:
            try:
                os.rmdir(private_name, dir_fd=parent_descriptor)
            except OSError:
                pass
        os.close(parent_descriptor)
        raise RuntimeError(
            "unnamed same-filesystem atomic output creation is unavailable"
        ) from error
    created = os.fstat(descriptor)
    private_payload = "payload.npz"
    stage_linked = False
    completed = False
    try:
        if not stat.S_ISREG(created.st_mode) or created.st_nlink != 0:
            raise RuntimeError("atomic output temporary is not an unnamed regular file")
        with os.fdopen(descriptor, "wb", closefd=False) as stream:
            np.savez_compressed(stream, **arrays)
            stream.flush()
            os.fsync(descriptor)
        # Revalidate every authority only after the complete compressed payload
        # is durable and immediately before its inode can enter the output
        # namespace.
        if before_replace is not None:
            before_replace()
        _assert_output_parent_fd_current(path.parent, parent_descriptor)
        stable = os.fstat(descriptor)
        if not (
            _same_regular_inode(stable, created)
            and stable.st_nlink == 0
            and created.st_size <= stable.st_size
        ):
            raise RuntimeError("atomic output temporary inode changed before publication")
        try:
            _link_open_fd_noreplace(descriptor, parent_descriptor, path.name)
        except FileExistsError:
            if not replace_existing:
                raise
            _link_open_fd_noreplace(
                descriptor,
                private_descriptor,
                private_payload,
            )
            stage_linked = True
            _exchange_stage_with_output(
                private_descriptor,
                private_payload,
                parent_descriptor,
                path.name,
                created,
            )
            stage_linked = False

        _assert_output_parent_fd_current(path.parent, parent_descriptor)
        published_entry = os.stat(path, follow_symlinks=False)
        if not _same_regular_inode(published_entry, created):
            raise RuntimeError(
                "published output changed concurrently; foreign entry preserved"
            )
        if published_entry.st_nlink != 1:
            raise RuntimeError("published output has an unexpected hard-link alias")
        os.fsync(parent_descriptor)
        published_descriptor = os.open(
            path.name,
            os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW,
            dir_fd=parent_descriptor,
        )
        with os.fdopen(published_descriptor, "rb") as published_stream:
            before = os.fstat(published_stream.fileno())
            if not _same_regular_inode(before, created):
                raise RuntimeError("published output inode differs from durable temporary")
            digest = hashlib.sha256()
            while chunk := published_stream.read(4 * 1024 * 1024):
                digest.update(chunk)
            after = os.fstat(published_stream.fileno())
        _assert_output_parent_fd_current(path.parent, parent_descriptor)
        current_output = os.stat(path, follow_symlinks=False)
        if not (
            before.st_dev == after.st_dev == current_output.st_dev
            and before.st_ino == after.st_ino == current_output.st_ino
            and before.st_size == after.st_size == current_output.st_size
            and before.st_mtime_ns == after.st_mtime_ns
            and before.st_ctime_ns == after.st_ctime_ns
            and current_output.st_nlink == 1
        ):
            raise RuntimeError("published output changed while its hash was verified")
        completed = True
        return digest.hexdigest()
    finally:
        cleanup_error: BaseException | None = None
        if stage_linked:
            try:
                stage_details = _stat_name(private_descriptor, private_payload)
                if _same_regular_inode(stage_details, created):
                    _unlink_name(private_descriptor, private_payload)
                    stage_linked = False
            except FileNotFoundError:
                stage_linked = False
            except BaseException as error:
                cleanup_error = error
        if descriptor >= 0:
            os.close(descriptor)
        if private_descriptor >= 0:
            os.close(private_descriptor)
        if private_name:
            try:
                os.rmdir(private_name, dir_fd=parent_descriptor)
            except OSError as error:
                if cleanup_error is None:
                    cleanup_error = error
        os.close(parent_descriptor)
        if cleanup_error is not None and completed:
            raise RuntimeError("private atomic publication cleanup failed") from cleanup_error


def _canonical_hash(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _verified_cache_provenance(cache: FeatureCache) -> dict[str, Any]:
    """Return the loader-issued cache authority, failing closed on drift."""

    provenance = cache.provenance
    if provenance is None:
        raise RuntimeError("feature cache loader returned no verified provenance")
    to_dict = getattr(provenance, "to_dict", None)
    content_sha256 = getattr(provenance, "content_sha256", None)
    if not callable(to_dict) or not isinstance(content_sha256, str):
        raise RuntimeError("feature cache loader returned malformed provenance")
    document = dict(to_dict())
    document["content_sha256"] = content_sha256
    payload = dict(document)
    payload.pop("content_sha256", None)
    if _canonical_hash(payload) != content_sha256:
        raise RuntimeError("feature cache provenance canonical hash mismatch")
    return document


def _runtime_source_hashes() -> dict[str, str]:
    return {
        str(path.resolve().relative_to(PROJECT_ROOT)): _sha256_file(path.resolve())
        for path in RUNTIME_SOURCE_PATHS
    }


def _assert_execution_inputs_current(
    *,
    checkpoint_path: Path,
    checkpoint_sha256: str,
    run_config_path: Path,
    run_config_sha256: str,
    source_hashes: Mapping[str, str],
) -> None:
    _assert_direct_entry_disk_binding_current()
    if _sha256_file(checkpoint_path) != checkpoint_sha256:
        raise RuntimeError("checkpoint changed during custom-split inference")
    if _sha256_file(run_config_path) != run_config_sha256:
        raise RuntimeError("run_config changed during custom-split inference")
    if _runtime_source_hashes() != dict(source_hashes):
        raise RuntimeError("runtime source changed during custom-split inference")


def _assert_split_authority_current(authority: IdentitySplitAuthority) -> None:
    """Revalidate every file that granted identity-split authority.

    The loader validates these files before returning ``authority``.  Inference
    can be long-running, so publication must also prove that the same bytes are
    still present rather than attaching stale in-memory authorization to a new
    on-disk split.
    """

    bindings = (
        (
            authority.manifest_path,
            authority.manifest_file_sha256,
            "identity split manifest",
        ),
        (
            authority.fold_assignments_path,
            authority.fold_assignments_sha256,
            "fold assignments",
        ),
        (
            authority.cache_manifest_path,
            authority.cache_manifest_sha256,
            "cache manifest",
        ),
    )
    for path, expected_sha256, label in bindings:
        try:
            observed_sha256 = _sha256_file(path)
        except OSError as error:
            raise RuntimeError(f"{label} became unreadable during inference") from error
        if observed_sha256 != expected_sha256:
            raise RuntimeError(f"{label} changed during custom-split inference")


def _resolve_recorded_path(value: Any) -> Path:
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def prepare_custom_cache(
    cache_dir: Path,
    run_config: Mapping[str, Any],
    *,
    raw_cache: FeatureCache | None = None,
) -> tuple[FeatureCache, int]:
    """Recreate branch selection and causal auxiliary topology from training."""

    arguments = run_config.get("arguments")
    if not isinstance(arguments, Mapping):
        raise RuntimeError("run_config arguments are missing")
    if _resolve_recorded_path(arguments.get("cache_dir")) != cache_dir.resolve():
        raise RuntimeError("--cache-dir differs from the cache bound in run_config")

    raw = (
        load_feature_cache(cache_dir, mmap=False)
        if raw_cache is None
        else raw_cache
    )
    stored_bins = int(raw.maps.shape[-1])
    if stored_bins % 2:
        raise RuntimeError("cache range dimension is not raw/phase separable")
    branch = str(arguments.get("map_branch", "both"))
    if branch == "both":
        maps = raw.maps
        expected_branches = 2
    elif branch == "raw":
        maps = raw.maps[..., : stored_bins // 2]
        expected_branches = 1
    elif branch == "phase":
        maps = raw.maps[..., stored_bins // 2 :]
        expected_branches = 1
    else:
        raise RuntimeError(f"unsupported checkpoint map_branch: {branch}")
    if int(arguments.get("input_branches", -1)) != expected_branches:
        raise RuntimeError("run_config map_branch/input_branches binding is inconsistent")

    cache = FeatureCache(
        maps=maps,
        aux=raw.aux,
        metadata=raw.metadata,
        frequencies_hz=raw.frequencies_hz,
        provenance=raw.provenance,
        radar_timing_valid_mask=raw.radar_timing_valid_mask,
    )
    base_aux_dim = int(cache.aux.shape[1])
    history_names: list[str] = []
    if bool(arguments.get("use_aux", False)) and bool(
        arguments.get("causal_history", False)
    ):
        augmented, history_names = append_mask_aware_causal_history_features(cache)
        cache = FeatureCache(
            maps=cache.maps,
            aux=augmented,
            metadata=cache.metadata,
            frequencies_hz=cache.frequencies_hz,
            provenance=cache.provenance,
            radar_timing_valid_mask=cache.radar_timing_valid_mask,
        )
    recorded_shape = run_config.get("cache_shape")
    observed_shape = {"maps": list(cache.maps.shape), "aux": list(cache.aux.shape)}
    if recorded_shape != observed_shape:
        raise RuntimeError(
            f"cache topology differs from run_config: {observed_shape} != {recorded_shape}"
        )
    if list(run_config.get("causal_history_feature_names", [])) != history_names:
        raise RuntimeError("causal-history feature schema differs from run_config")
    return cache, base_aux_dim


def validate_prediction_ownership(
    metadata: Any, prediction_identities: Sequence[str]
) -> np.ndarray:
    identities = metadata["identity"].astype(str).to_numpy()
    expected = np.flatnonzero(np.isin(identities, tuple(prediction_identities)))
    if len(expected) == 0:
        raise RuntimeError("custom prediction identities own no cache rows")
    observed = set(identities[expected].tolist())
    if observed != set(prediction_identities):
        raise RuntimeError("not every custom prediction identity owns a cache row")
    return expected.astype(np.int64, copy=False)


def validate_custom_checkpoint(
    checkpoint_path: Path,
    *,
    authority: IdentitySplitAuthority,
    cache: FeatureCache,
    base_aux_dim: int,
    run_config: Mapping[str, Any],
    checkpoint_bytes: bytes | None = None,
) -> dict[str, Any]:
    """Validate custom split, fold, scaler and model/cache provenance."""

    payload = (
        _read_regular_snapshot(checkpoint_path)[0]
        if checkpoint_bytes is None
        else bytes(checkpoint_bytes)
    )
    checkpoint = torch.load(
        io.BytesIO(payload), map_location="cpu", weights_only=True
    )
    if checkpoint.get("format_version") != 2:
        raise RuntimeError("custom proposer checkpoint must use format_version=2")
    if checkpoint.get("model_type") != "snn":
        raise RuntimeError("custom proposer checkpoint must be an SNN")
    if int(checkpoint.get("fold", -1)) != authority.fold_id:
        raise RuntimeError("checkpoint fold does not match split manifest fold_id")
    if checkpoint.get("run_signature") != run_config.get("run_signature"):
        raise RuntimeError("checkpoint/run_config signature mismatch")
    expected_provenance = authority.checkpoint_provenance()
    if checkpoint.get("split_authority_provenance") != expected_provenance:
        raise RuntimeError("checkpoint split-authority provenance mismatch")

    split = checkpoint.get("split")
    if not isinstance(split, Mapping):
        raise RuntimeError("checkpoint custom split is missing")
    expected_split = {
        "train_identities": list(authority.train_identities),
        "validation_identities": list(authority.validation_identities),
        "prediction_identities": list(authority.prediction_identities),
        "excluded_identities": list(authority.excluded_identities),
        "scaler_identities": list(authority.scaler_identities),
    }
    normalized = {
        key: sorted(map(str, split.get(key, ()))) for key in expected_split
    }
    if normalized != {key: sorted(value) for key, value in expected_split.items()}:
        raise RuntimeError("checkpoint identities differ from the custom split manifest")

    arguments = run_config.get("arguments")
    if not isinstance(arguments, Mapping):
        raise RuntimeError("run_config arguments are missing")
    if str(arguments.get("identity_split_manifest_sha256", "")) != authority.content_sha256:
        raise RuntimeError("run_config split manifest content hash mismatch")
    loaded_cache_provenance = _verified_cache_provenance(cache)
    if run_config.get("cache_provenance") != loaded_cache_provenance:
        raise RuntimeError("run_config/loaded cache provenance mismatch")
    if checkpoint.get("cache_provenance") != loaded_cache_provenance:
        raise RuntimeError("checkpoint/loaded cache provenance mismatch")
    validate_model_kwargs(
        checkpoint, cache, base_aux_dim=base_aux_dim, run_config=run_config
    )

    identity = cache.metadata["identity"].astype(str).to_numpy()
    reference_valid = cache.metadata["reference_valid"].to_numpy(dtype=bool)
    train_mask = np.isin(identity, authority.train_identities)
    if not bool(arguments.get("include_invalid", False)):
        train_mask &= reference_valid
    train_index = np.flatnonzero(train_mask)
    authority.validate_scaler_indices(cache.metadata, train_index)
    expected_center, expected_scale = fit_aux_scaler(cache.aux, train_index)
    center = _as_numpy_scaler(checkpoint, "aux_center")
    scale = _as_numpy_scaler(checkpoint, "aux_scale")
    use_aux = bool(arguments.get("use_aux", False))
    if use_aux:
        if center.shape != (cache.aux.shape[1],) or scale.shape != center.shape:
            raise RuntimeError("checkpoint auxiliary scaler dimension mismatch")
        if not np.isfinite(scale).all() or np.any(scale <= 0):
            raise RuntimeError("checkpoint auxiliary scaler is invalid")
        if not np.allclose(center, expected_center, rtol=1e-6, atol=1e-7):
            raise RuntimeError("checkpoint auxiliary center is not train-only fitted")
        if not np.allclose(scale, expected_scale, rtol=1e-6, atol=1e-7):
            raise RuntimeError("checkpoint auxiliary scale is not train-only fitted")
    elif center.size or scale.size:
        raise RuntimeError("aux-disabled checkpoint unexpectedly contains a scaler")
    return checkpoint


def _posterior_grid(checkpoint: Mapping[str, Any]) -> np.ndarray:
    state = checkpoint.get("model_state")
    if not isinstance(state, Mapping) or "rr_bins" not in state:
        raise RuntimeError("checkpoint lacks posterior RR grid")
    grid = np.asarray(state["rr_bins"].detach().cpu(), dtype=np.float32)
    if grid.ndim != 1 or len(grid) < 2 or not np.isfinite(grid).all():
        raise RuntimeError("checkpoint posterior RR grid is invalid")
    return grid


def build_output_arrays(
    bundle: Any,
    metadata: Any,
    expected_index: np.ndarray,
    *,
    checkpoint: Mapping[str, Any],
    checkpoint_path: Path,
    authority: IdentitySplitAuthority,
    run_config_path: Path,
    cache_source: Mapping[str, Any],
    acquisition_authority_bindings: Mapping[str, Mapping[str, Any]],
    source_hashes: Mapping[str, str],
    direct_entry_disk_binding: Mapping[str, Any],
    checkpoint_sha256: str,
    run_config_sha256: str,
) -> dict[str, Any]:
    index = np.asarray(bundle.index, dtype=np.int64)
    if not np.array_equal(index, expected_index):
        raise RuntimeError("inference did not exactly cover prediction-identity rows")
    if len(np.unique(index)) != len(index):
        raise RuntimeError("inference returned duplicate cache rows")
    rows = metadata.iloc[index]
    observed_identities = set(rows["identity"].astype(str))
    if observed_identities != set(authority.prediction_identities):
        raise RuntimeError("inference result violates prediction identity ownership")

    reference_valid = rows["reference_valid"].to_numpy(dtype=bool)
    if not np.array_equal(np.asarray(bundle.reference_valid, dtype=bool), reference_valid):
        raise RuntimeError("prediction bundle reference mask differs from cache metadata")
    reference_rr = rows["rr_bpm"].to_numpy(dtype=np.float32)
    reference_rr = np.where(reference_valid, reference_rr, np.nan).astype(np.float32)

    canonical_cache_provenance = cache_source.get("canonical_cache_provenance")
    if not isinstance(canonical_cache_provenance, Mapping):
        raise RuntimeError("validated canonical cache provenance is missing")
    authority_bindings = {
        str(path): dict(binding)
        for path, binding in sorted(acquisition_authority_bindings.items())
    }
    provenance: dict[str, Any] = {
        "format_version": FORMAT_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "mode": "custom_identity_split_label_free_all_windows",
        "strict_nested_role": "prediction",
        "strict_retrospective": True,
        "labels_forwarded_to_model": False,
        "reference_invalid_rows_included": True,
        "commercial_performance_claim_eligible": False,
        "fold_id": authority.fold_id,
        "prediction_identities": list(authority.prediction_identities),
        "checkpoint_path": str(checkpoint_path.resolve()),
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_run_signature": str(checkpoint.get("run_signature", "")),
        "run_config_path": str(run_config_path.resolve()),
        "run_config_sha256": run_config_sha256,
        "split_manifest_path": str(authority.manifest_path),
        "split_manifest_file_sha256": authority.manifest_file_sha256,
        "split_manifest_content_sha256": authority.content_sha256,
        "fold_assignments_sha256": authority.fold_assignments_sha256,
        "cache_manifest_sha256": authority.cache_manifest_sha256,
        "source_hashes": dict(source_hashes),
        "execution_source_generation": {
            "guard_scope": "initialization_time_direct_entry_disk_only",
            "direct_entry_disk_binding": dict(direct_entry_disk_binding),
            "binds_actual_loader_compiled_bytes": False,
            "complete_private_import_closure": False,
            "scientific_authority_status": (
                "terminal_blocked_without_fresh_isolated_private_source_launcher"
            ),
        },
        "canonical_cache_provenance": dict(canonical_cache_provenance),
        "cache_content_signature_sha256": str(
            cache_source.get("cache_content_signature_sha256", "")
        ),
        "cache_consumption_policy": (
            "private_cow_or_stream_snapshot_inventory_equal_loader_provenance_"
            "with_original_namespace_publication_barrier"
        ),
        "acquisition_authority_binding_policy": (
            "stable_pre_loader_hash_stat_inventory_with_post_loader_and_"
            "prepublication_revalidation_plus_full_canonical_reconstruction_replay"
        ),
        "full_acquisition_authority_replay_independent_of_verify_raw_sources": bool(
            authority_bindings
        ),
        "acquisition_authority_file_bindings": authority_bindings,
        "acquisition_authority_file_count": len(authority_bindings),
        "acquisition_authority_bindings_sha256": _canonical_hash(
            {"files": authority_bindings}
        ),
        "raw_source_fingerprints_verified": bool(
            cache_source.get("raw_source_fingerprints_verified", False)
        ),
        "raw_input_sha256_bindings_verified": bool(
            cache_source.get("raw_input_sha256_bindings_verified", False)
        ),
        "row_count": len(index),
        "reference_valid_count": int(reference_valid.sum()),
        "output_allowlist": list(OUTPUT_FIELDS),
        "excluded_model_inputs": ["reference_rr", "reference_valid", "identity", "protocol"],
    }
    provenance["inference_signature_sha256"] = _canonical_hash(provenance)
    arrays: dict[str, Any] = {
        "cache_index": index,
        "session_id": rows["session_id"].astype(str).to_numpy(dtype=np.str_),
        "identity": rows["identity"].astype(str).to_numpy(dtype=np.str_),
        "protocol": rows["protocol"].astype(str).to_numpy(dtype=np.str_),
        "window_number": rows["window_number"].to_numpy(dtype=np.int32),
        "reference_valid": reference_valid,
        "reference_rr_bpm": reference_rr,
        "prediction": np.asarray(bundle.prediction, dtype=np.float32),
        "map_prediction": np.asarray(bundle.map_prediction, dtype=np.float32),
        "rr_std": np.asarray(bundle.rr_std, dtype=np.float32),
        "uncertainty": np.asarray(bundle.uncertainty, dtype=np.float32),
        "quality": np.asarray(bundle.quality, dtype=np.float32),
        "alias_probability": np.asarray(bundle.alias_probability, dtype=np.float32),
        "posterior_entropy": np.asarray(bundle.posterior_entropy, dtype=np.float32),
        "topk_rr": np.asarray(bundle.topk_rr, dtype=np.float32),
        "topk_probability": np.asarray(bundle.topk_probability, dtype=np.float32),
        "posterior_probability": np.asarray(bundle.posterior_probability, dtype=np.float16),
        "spike_rate": np.asarray(bundle.spike_rate, dtype=np.float32),
        "radar_weights": np.asarray(bundle.radar_weights, dtype=np.float32),
        "posterior_rr_grid_bpm": _posterior_grid(checkpoint),
        "fold_id": np.asarray(authority.fold_id, dtype=np.int16),
        "checkpoint_sha256": np.asarray(provenance["checkpoint_sha256"]),
        "split_manifest_file_sha256": np.asarray(authority.manifest_file_sha256),
        "split_manifest_content_sha256": np.asarray(authority.content_sha256),
        "fold_assignments_sha256": np.asarray(authority.fold_assignments_sha256),
        "cache_manifest_sha256": np.asarray(authority.cache_manifest_sha256),
        "run_config_sha256": np.asarray(provenance["run_config_sha256"]),
        "inference_signature_sha256": np.asarray(provenance["inference_signature_sha256"]),
        "strict_retrospective": np.asarray(True),
        "strict_nested_prediction_role": np.asarray(True),
        "provenance_json": np.asarray(
            json.dumps(provenance, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        ),
    }
    for name in OUTPUT_FIELDS:
        if name not in arrays:
            raise RuntimeError(f"internal output allowlist omission: {name}")
    return arrays


def _run_with_snapshot_owner(
    args: argparse.Namespace,
    *,
    snapshot_owner: list[_FeatureCacheSnapshot],
) -> dict[str, Any]:
    cache_dir = args.cache_dir.expanduser().resolve()
    checkpoint_path = args.checkpoint.expanduser().resolve()
    manifest_path = args.identity_split_manifest.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    run_config_path = checkpoint_path.parent.parent / "run_config.json"
    output_guard = _build_output_isolation_guard(
        output_path=output_path,
        cache_dir=cache_dir,
        checkpoint_path=checkpoint_path,
        run_config_path=run_config_path,
        split_manifest_path=manifest_path,
    )
    if output_path.exists() and not args.force:
        raise FileExistsError(f"output exists; pass --force to replace: {output_path}")
    checkpoint_bytes, checkpoint_sha256 = _read_regular_snapshot(checkpoint_path)
    run_config_bytes, run_config_sha256 = _read_regular_snapshot(run_config_path)
    _assert_direct_entry_disk_binding_current()
    direct_entry_disk_binding = dict(_DIRECT_ENTRY_DISK_BINDING)
    source_hashes = _runtime_source_hashes()
    try:
        run_config = json.loads(run_config_bytes.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"invalid run_config snapshot: {error}") from error
    if not isinstance(run_config, dict):
        raise RuntimeError("run_config snapshot root must be an object")
    _assert_execution_inputs_current(
        checkpoint_path=checkpoint_path,
        checkpoint_sha256=checkpoint_sha256,
        run_config_path=run_config_path,
        run_config_sha256=run_config_sha256,
        source_hashes=source_hashes,
    )
    cache_snapshot = _materialize_feature_cache_snapshot(cache_dir)
    snapshot_owner.append(cache_snapshot)
    snapshot_cache_manifest = _json_snapshot(
        cache_snapshot.cache_dir / "manifest.json",
        "private feature-cache manifest",
    )
    acquisition_authority_bindings = _capture_acquisition_authority_bindings(
        cache_dir,
        cache_manifest=snapshot_cache_manifest,
    )
    # The canonical loader still validates the full acquisition/provenance
    # graph at its original paths.  Its arrays are deliberately discarded;
    # model-forward data comes only from the exact private copies whose
    # inventory digest must equal the loader-issued provenance.
    validated_live_cache = load_feature_cache(cache_dir, mmap=False)
    validated_provenance = validated_live_cache.provenance
    del validated_live_cache
    cache_snapshot.assert_source_current()
    _assert_acquisition_authority_bindings_current(
        cache_dir,
        acquisition_authority_bindings,
        cache_manifest=snapshot_cache_manifest,
    )
    snapshot_raw_cache = _load_feature_cache_snapshot_payload(
        cache_snapshot,
        validated_provenance=validated_provenance,
    )
    cache, base_aux_dim = prepare_custom_cache(
        cache_dir, run_config, raw_cache=snapshot_raw_cache
    )
    cache_source = validate_cache_sources(
        cache_dir,
        cache.metadata,
        verify_raw_sources=args.verify_raw_sources,
        verified_cache_provenance=cache.provenance,
    )
    _assert_acquisition_authority_bindings_current(
        cache_dir,
        acquisition_authority_bindings,
        cache_manifest=snapshot_cache_manifest,
    )
    authority = load_identity_split_authority(
        manifest_path, metadata=cache.metadata, cache_dir=cache_dir
    )
    _assert_split_authority_current(authority)
    output_guard = _build_output_isolation_guard(
        output_path=output_path,
        cache_dir=cache_dir,
        checkpoint_path=checkpoint_path,
        run_config_path=run_config_path,
        split_manifest_path=manifest_path,
        extra_files=(
            authority.fold_assignments_path,
            authority.cache_manifest_path,
        ),
    )
    checkpoint = validate_custom_checkpoint(
        checkpoint_path,
        authority=authority,
        cache=cache,
        base_aux_dim=base_aux_dim,
        run_config=run_config,
        checkpoint_bytes=checkpoint_bytes,
    )
    _assert_execution_inputs_current(
        checkpoint_path=checkpoint_path,
        checkpoint_sha256=checkpoint_sha256,
        run_config_path=run_config_path,
        run_config_sha256=run_config_sha256,
        source_hashes=source_hashes,
    )
    expected_index = validate_prediction_ownership(
        cache.metadata, authority.prediction_identities
    )

    arguments = run_config["arguments"]
    if bool(arguments.get("use_aux", False)):
        center = _as_numpy_scaler(checkpoint, "aux_center")
        scale = _as_numpy_scaler(checkpoint, "aux_scale")
        aux = transform_aux(cache.aux, center, scale)
    else:
        aux = np.empty((len(cache.metadata), 0), dtype=np.float32)
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    amp = bool(args.amp and device.type == "cuda")
    loader = make_loader(
        cache,
        aux,
        expected_index,
        batch_size=args.batch_size,
        workers=args.workers,
        device=device,
        seed=int(arguments.get("seed", 0)) + 99001,
        train=False,
        auxiliary_layout=(
            infer_auxiliary_layout(base_aux_dim)
            if bool(arguments.get("use_aux", False))
            else None
        ),
    )
    model = build_model(str(checkpoint["model_type"]), checkpoint["model_kwargs"])
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model = model.to(device)
    validate_label_free_forward_interface(model)
    bundle = predict_label_free(model, loader, device, amp=amp)
    _assert_split_authority_current(authority)
    _assert_execution_inputs_current(
        checkpoint_path=checkpoint_path,
        checkpoint_sha256=checkpoint_sha256,
        run_config_path=run_config_path,
        run_config_sha256=run_config_sha256,
        source_hashes=source_hashes,
    )
    arrays = build_output_arrays(
        bundle,
        cache.metadata,
        expected_index,
        checkpoint=checkpoint,
        checkpoint_path=checkpoint_path,
        authority=authority,
        run_config_path=run_config_path,
        cache_source=cache_source,
        acquisition_authority_bindings=acquisition_authority_bindings,
        source_hashes=source_hashes,
        direct_entry_disk_binding=direct_entry_disk_binding,
        checkpoint_sha256=checkpoint_sha256,
        run_config_sha256=run_config_sha256,
    )
    def final_publication_barrier() -> None:
        output_guard.assert_disjoint(output_path)
        cache_snapshot.assert_private_current()
        cache_snapshot.assert_source_current()
        _assert_acquisition_authority_bindings_current(
            cache_dir,
            acquisition_authority_bindings,
            cache_manifest=snapshot_cache_manifest,
        )
        _revalidate_full_acquisition_authority(
            cache_dir,
            cache_manifest=snapshot_cache_manifest,
            expected_bindings=acquisition_authority_bindings,
        )
        _assert_publication_sources_current(cache_source, source_hashes)
        _assert_split_authority_current(authority)
        _assert_execution_inputs_current(
            checkpoint_path=checkpoint_path,
            checkpoint_sha256=checkpoint_sha256,
            run_config_path=run_config_path,
            run_config_sha256=run_config_sha256,
            source_hashes=source_hashes,
        )

    # Check once before potentially expensive serialization and again from
    # inside `_atomic_npz` after the temp file is complete but before replace.
    final_publication_barrier()
    output_sha256 = _atomic_npz(
        output_path,
        arrays,
        before_replace=final_publication_barrier,
        replace_existing=bool(args.force),
    )
    return {
        "output": str(output_path),
        "output_sha256": output_sha256,
        "rows": len(expected_index),
        "invalid_reference_rows": int((~arrays["reference_valid"]).sum()),
        "fold_id": authority.fold_id,
        "prediction_identities": list(authority.prediction_identities),
        "inference_signature_sha256": str(arrays["inference_signature_sha256"]),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Run inference with unconditional cleanup of private cache snapshots."""

    snapshots: list[_FeatureCacheSnapshot] = []
    try:
        return _run_with_snapshot_owner(args, snapshot_owner=snapshots)
    finally:
        for snapshot in reversed(snapshots):
            snapshot.cleanup()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--identity-split-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--verify-raw-sources",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="verify raw source fingerprints and acquisition-v2 SHA bindings",
    )
    parser.add_argument("--force", action="store_true")
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.batch_size <= 0 or args.workers < 0:
        parser.error("--batch-size must be positive and --workers non-negative")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    result = run(parse_args(argv))
    print(json.dumps(result, sort_keys=True, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
