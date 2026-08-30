#!/usr/bin/env python3
"""Create the V8R4A dedicated GPU-state directories without resetting history.

The migration is deliberately narrower than a campaign launcher.  It opens no
target data and starts no GPU process.  It verifies the frozen V8R4A authority,
locks the five legacy state files, replays both ledgers, publishes three
dedicated directories with renameat2(RENAME_NOREPLACE), retires the legacy
files read-only without changing their bytes or inodes, and emits one immutable
path/inode-lineage receipt.
"""

from __future__ import annotations

import argparse
from contextlib import ExitStack, contextmanager
import ctypes
from dataclasses import dataclass
from datetime import datetime, timezone
import errno
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path, PurePosixPath
import secrets
import stat
import sys
from typing import Any, Iterator, Mapping, Sequence


SCHEMA_VERSION = 1
CAMPAIGN_ID = "directed_harmonic_factor_expert_snn_v3r1_adaptive_retrospective"
SCIENTIFIC_CAMPAIGN_REVISION = "V8R4"
INFRASTRUCTURE_REVISION = "V8R4A"
AUTHORIZATION_CLASSIFICATION = (
    "pretrain_adaptive_v3r1_v8r4a_dedicated_gpu_state_directory_"
    "migration_correction_authorization"
)
MIGRATION_CLASSIFICATION = "adaptive_v3r1_v8r4a_gpu_state_migration_receipt"
AUTHORIZATION_FILE_SHA256 = (
    "5427853d4c530d5996a4bc63f12c7cd5fef5ca505a6d754c392546a17d2b6b30"
)
AUTHORIZATION_CONTENT_SHA256 = (
    "ea3cd60df547184faebacf6ac76bcb84a10888a6a0224a29986f27bb156e5a0b"
)
SOURCE_SUCCESSION_AUTHORIZATION_CLASSIFICATION = (
    "pretrain_adaptive_v3r1_v8r4a_migrated_state_source_succession_"
    "correction_addendum"
)
SOURCE_SUCCESSION_DIAGNOSTIC_CLASSIFICATION = (
    "pretrain_adaptive_v3r1_v8r4a_migrated_state_source_succession_"
    "failure_diagnostic"
)
SOURCE_SUCCESSION_AUTHORIZATION_FILE_SHA256 = (
    "4a3673a406f49287b5abe16cc9ddde5d90d55f3a18d82a346ed390b55ccd91d9"
)
SOURCE_SUCCESSION_AUTHORIZATION_CONTENT_SHA256 = (
    "75ef405e7c8e6c4a18f2c676f610ef9574ccfd5353a294f16acda5154aa890b3"
)
SOURCE_SUCCESSION_AUTHORIZATION_BYTES = 6310
SOURCE_SUCCESSION_DIAGNOSTIC_FILE_SHA256 = (
    "265eb0cb62f6412d26bc7491ad959c8b3d6e49ffc47241573ed0fecf5111ac1e"
)
SOURCE_SUCCESSION_DIAGNOSTIC_CONTENT_SHA256 = (
    "84480a9a35275bc23ed656541de9bc0c4857f705ca297481d9e54e0f7af58e57"
)
SOURCE_SUCCESSION_DIAGNOSTIC_BYTES = 3677
EMPTY_SHA256 = hashlib.sha256(b"").hexdigest()
GPU_BUDGET_NS = 36_000_000_000_000
LEGACY_GENESIS_SHA256 = (
    "c7b463e4db2e8d475f428dc61dfc8fa0d27910f62fb2d811e2845fcf4932e035"
)

CAMPAIGN_GOVERNANCE_ROOT = PurePosixPath(
    "artifacts/campaigns/directed_harmonic_factor_expert_snn_v3r1"
)
CAMPAIGN_RUN_ROOT = PurePosixPath(
    "artifacts/runs/directed_harmonic_factor_expert_snn_v3r1"
)
AUTHORIZATION_RELATIVE = CAMPAIGN_GOVERNANCE_ROOT / (
    "IMPLEMENTATION_CORRECTION_AUTHORIZATION_V8R4A.json"
)
SOURCE_SUCCESSION_AUTHORIZATION_RELATIVE = CAMPAIGN_GOVERNANCE_ROOT / (
    "IMPLEMENTATION_CORRECTION_AUTHORIZATION_V8R4A_MIGRATION_SOURCE_SUCCESSION.json"
)
SOURCE_SUCCESSION_DIAGNOSTIC_RELATIVE = CAMPAIGN_GOVERNANCE_ROOT / (
    "diagnostics/v3r1_v8r4a_migrated_state_source_succession_failure.json"
)
RECEIPT_RELATIVE = CAMPAIGN_GOVERNANCE_ROOT / (
    "GPU_STATE_MIGRATION_RECEIPT_V8R4A.json"
)
TARGET_ROOT_RELATIVE = CAMPAIGN_RUN_ROOT / "gpu_state_v8r4a"

LEGACY_RELATIVE: dict[str, PurePosixPath] = {
    "admission_lock": CAMPAIGN_RUN_ROOT / "gpu_admission_v7.lock",
    "execution_ledger": CAMPAIGN_RUN_ROOT / "gpu_execution_ledger_v7.jsonl",
    "execution_ledger_lock": CAMPAIGN_RUN_ROOT
    / "gpu_execution_ledger_v7.jsonl.lock",
    "usage_ledger": CAMPAIGN_RUN_ROOT / "campaign_gpu_usage_chain_v6.jsonl",
    "usage_ledger_lock": CAMPAIGN_RUN_ROOT
    / "campaign_gpu_usage_chain_v6.jsonl.lock",
}
TARGET_RELATIVE: dict[str, PurePosixPath] = {
    "admission_lock": TARGET_ROOT_RELATIVE / "admission/gpu_admission_v7.lock",
    "execution_ledger": TARGET_ROOT_RELATIVE
    / "execution/gpu_execution_ledger_v7.jsonl",
    "execution_ledger_lock": TARGET_ROOT_RELATIVE
    / "execution/gpu_execution_ledger_v7.jsonl.lock",
    "usage_ledger": TARGET_ROOT_RELATIVE
    / "usage/campaign_gpu_usage_chain_v6.jsonl",
    "usage_ledger_lock": TARGET_ROOT_RELATIVE
    / "usage/campaign_gpu_usage_chain_v6.jsonl.lock",
}
ROLE_DIRECTORIES: dict[str, PurePosixPath] = {
    "admission": TARGET_ROOT_RELATIVE / "admission",
    "execution": TARGET_ROOT_RELATIVE / "execution",
    "usage": TARGET_ROOT_RELATIVE / "usage",
}
ROLE_ENTRIES: dict[str, tuple[str, ...]] = {
    "admission": ("gpu_admission_v7.lock",),
    "execution": (
        "gpu_execution_ledger_v7.jsonl",
        "gpu_execution_ledger_v7.jsonl.lock",
    ),
    "usage": (
        "campaign_gpu_usage_chain_v6.jsonl",
        "campaign_gpu_usage_chain_v6.jsonl.lock",
    ),
}
FILE_ROLES = tuple(sorted(LEGACY_RELATIVE))
LOCK_ROLES = (
    "admission_lock",
    "usage_ledger_lock",
    "execution_ledger_lock",
)
LEDGER_ROLES = ("usage_ledger", "execution_ledger")

AUTHORIZATION_KEYS = frozenset(
    {
        "authority_basis",
        "authorized_modifications",
        "campaign_id",
        "canonical_gpu_state_layout",
        "claim_boundary",
        "classification",
        "content_sha256",
        "created_utc",
        "diagnostic",
        "forbidden_changes",
        "frozen_implementation_bindings",
        "historical_evidence_acceptance",
        "infrastructure_revision",
        "mandatory_invariants",
        "migration_protocol",
        "required_reauthorization",
        "schema_version",
        "scientific_campaign_revision",
    }
)
RECEIPT_KEYS = frozenset(
    {
        "authority",
        "campaign_id",
        "classification",
        "content_sha256",
        "created_utc",
        "directory_inventory",
        "historical_evidence",
        "infrastructure_revision",
        "lifecycle_state",
        "migrated_state",
        "original_state",
        "path_inode_lineage",
        "prefix_replay",
        "production_runtime_authorized",
        "schema_version",
        "scientific_campaign_revision",
    }
)
FILE_BINDING_KEYS = frozenset(
    {"bytes", "mode", "nlink", "path", "sha256", "st_dev", "st_ino"}
)
DIRECTORY_BINDING_KEYS = frozenset(
    {"exact_entries", "mode", "path", "st_dev", "st_ino"}
)
GOVERNANCE_BINDING_KEYS = frozenset(
    {
        "bytes",
        "content_sha256",
        "mode",
        "nlink",
        "path",
        "sha256",
        "st_dev",
        "st_ino",
    }
)
RENAME_NOREPLACE = 1
SUCCESSOR_SOURCE_BINDINGS = {
    "gpu_budget_ledger": {
        "bytes": 99764,
        "path": "src/snn_rr/gpu_budget_ledger.py",
        "sha256": "a23d8ee38e296ff863cfb5b72d730b91d104ccd56008575ee6e43decd1c20dbb",
    },
    "run_gpu_admitted": {
        "bytes": 142530,
        "path": "scripts/run_gpu_admitted.py",
        "sha256": "97aa8d9095f919a78ccc639c5eaa9d83d1354e945c1b64b3a5ccb53e2850ae1a",
    },
}


class MigrationError(RuntimeError):
    """The authorized migration cannot be completed without ambiguity."""


@dataclass(frozen=True)
class MigrationResult:
    receipt_path: Path
    receipt: dict[str, Any]
    resumed: bool


@dataclass(frozen=True)
class MigratedStateValidation:
    """Read-only live state returned to the target-sealed runtime."""

    receipt_path: Path
    receipt: dict[str, Any]
    receipt_binding: dict[str, Any]
    canonical_paths: dict[str, Path]
    directory_bindings: dict[str, dict[str, Any]]
    current_file_bindings: dict[str, dict[str, Any]]
    usage_state: dict[str, Any]
    execution_state: dict[str, Any]


@dataclass(frozen=True)
class AuthorityContext:
    document: dict[str, Any]
    binding: dict[str, Any]
    historical_evidence: dict[str, dict[str, Any]]
    diagnostic: dict[str, Any]
    source_bindings: dict[str, dict[str, Any]]


@dataclass(frozen=True)
class TreeSnapshot:
    root: dict[str, Any]
    roles: dict[str, dict[str, Any]]
    files: dict[str, dict[str, Any]]
    raw_files: dict[str, bytes]


def _unique_object(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise MigrationError(f"duplicate JSON key: {key}")
        value[key] = item
    return value


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def pretty_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False) + "\n"
    ).encode("utf-8")


def semantic_sha256(document: Mapping[str, Any]) -> str:
    value = dict(document)
    value.pop("content_sha256", None)
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _decode_json(raw: bytes, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(
            raw,
            object_pairs_hook=_unique_object,
            parse_constant=lambda token: (_ for _ in ()).throw(
                MigrationError(f"non-finite JSON constant in {label}: {token}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise MigrationError(f"invalid JSON in {label}: {error}") from error
    if not isinstance(value, dict):
        raise MigrationError(f"{label} is not a JSON object")
    return value


def _validate_self_hash(document: Mapping[str, Any], *, label: str) -> None:
    claimed = document.get("content_sha256")
    if not isinstance(claimed, str) or claimed != semantic_sha256(document):
        raise MigrationError(f"{label} content_sha256 mismatch")


def _format_mode(mode: int) -> str:
    return f"{stat.S_IMODE(mode):04o}"


def _same_inode(left: os.stat_result, right: os.stat_result) -> bool:
    return (left.st_dev, left.st_ino) == (right.st_dev, right.st_ino)


def _canonical_project_root(path: Path) -> Path:
    lexical = path.absolute()
    try:
        resolved = path.resolve(strict=True)
    except OSError as error:
        raise MigrationError(f"project root does not exist: {path}") from error
    if lexical != resolved or not resolved.is_dir():
        raise MigrationError("project root must be an existing canonical directory")
    if resolved.is_symlink():
        raise MigrationError("project root may not be a symlink")
    return resolved


def _safe_path(root: Path, relative: PurePosixPath | str) -> Path:
    item = PurePosixPath(relative)
    if item.is_absolute() or not item.parts or any(part in {"", ".", ".."} for part in item.parts):
        raise MigrationError(f"unsafe project-relative path: {relative}")
    path = root.joinpath(*item.parts)
    try:
        resolved_parent = path.parent.resolve(strict=True)
    except OSError as error:
        raise MigrationError(f"missing path parent: {item}") from error
    if resolved_parent != path.parent.absolute():
        raise MigrationError(f"symlinked path parent refused: {item}")
    try:
        resolved_parent.relative_to(root)
    except ValueError as error:
        raise MigrationError(f"path escapes project root: {item}") from error
    return path


def _relative_path(root: Path, path: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError as error:
        raise MigrationError(f"path is outside project root: {path}") from error


class PinnedRegular:
    """A nofollow regular-file descriptor plus its pinned named parent."""

    def __init__(self, root: Path, relative: PurePosixPath | str, *, label: str):
        self.root = root
        self.path = _safe_path(root, relative)
        self.label = label
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
        if hasattr(os, "O_DIRECTORY"):
            flags |= os.O_DIRECTORY
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        try:
            self.parent_fd = os.open(self.path.parent, flags)
        except OSError as error:
            raise MigrationError(f"cannot pin {label} parent: {error}") from error
        self.fd = -1
        try:
            self.parent_status = os.fstat(self.parent_fd)
            named_parent = os.stat(self.path.parent, follow_symlinks=False)
            if not stat.S_ISDIR(self.parent_status.st_mode) or not _same_inode(
                self.parent_status, named_parent
            ):
                raise MigrationError(f"{label} parent identity drifted")
            file_flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
            if hasattr(os, "O_NOFOLLOW"):
                file_flags |= os.O_NOFOLLOW
            try:
                self.fd = os.open(
                    self.path.name, file_flags, dir_fd=self.parent_fd
                )
            except OSError as error:
                raise MigrationError(
                    f"cannot open {label} as a nofollow regular file: {error}"
                ) from error
            status = os.fstat(self.fd)
            named = os.stat(
                self.path.name, dir_fd=self.parent_fd, follow_symlinks=False
            )
            if (
                not stat.S_ISREG(status.st_mode)
                or status.st_nlink != 1
                or not _same_inode(status, named)
            ):
                raise MigrationError(f"{label} is aliased or not a regular file")
            self.identity = (status.st_dev, status.st_ino)
        except BaseException:
            self.close()
            raise

    def __enter__(self) -> "PinnedRegular":
        return self

    def __exit__(self, *_args: Any) -> None:
        self.close()

    def close(self) -> None:
        if self.fd >= 0:
            os.close(self.fd)
            self.fd = -1
        if getattr(self, "parent_fd", -1) >= 0:
            os.close(self.parent_fd)
            self.parent_fd = -1

    def read(self) -> bytes:
        before = os.fstat(self.fd)
        os.lseek(self.fd, 0, os.SEEK_SET)
        chunks: list[bytes] = []
        while True:
            block = os.read(self.fd, 1024 * 1024)
            if not block:
                break
            chunks.append(block)
        after = os.fstat(self.fd)
        if not _same_inode(before, after) or before.st_size != after.st_size:
            raise MigrationError(f"{self.label} changed while read")
        raw = b"".join(chunks)
        if len(raw) != after.st_size:
            raise MigrationError(f"short read from {self.label}")
        return raw

    def binding(self, *, raw: bytes | None = None) -> dict[str, Any]:
        payload = self.read() if raw is None else raw
        status = self.revalidate()
        return {
            "bytes": len(payload),
            "mode": _format_mode(status.st_mode),
            "nlink": status.st_nlink,
            "path": _relative_path(self.root, self.path),
            "sha256": hashlib.sha256(payload).hexdigest(),
            "st_dev": status.st_dev,
            "st_ino": status.st_ino,
        }

    def revalidate(self) -> os.stat_result:
        parent = os.fstat(self.parent_fd)
        named_parent = os.stat(self.path.parent, follow_symlinks=False)
        descriptor = os.fstat(self.fd)
        named = os.stat(
            self.path.name, dir_fd=self.parent_fd, follow_symlinks=False
        )
        if (
            not _same_inode(parent, self.parent_status)
            or not _same_inode(parent, named_parent)
            or not _same_inode(descriptor, named)
            or (descriptor.st_dev, descriptor.st_ino) != self.identity
            or not stat.S_ISREG(descriptor.st_mode)
            or descriptor.st_nlink != 1
        ):
            raise MigrationError(f"{self.label} inode drifted")
        return descriptor

    def chmod(self, mode: int) -> None:
        self.revalidate()
        os.fchmod(self.fd, mode)
        self.revalidate()


@contextmanager
def _exclusive_lock(
    root: Path, relative: PurePosixPath, *, label: str
) -> Iterator[PinnedRegular]:
    pinned = PinnedRegular(root, relative, label=label)
    try:
        raw = pinned.read()
        if raw:
            raise MigrationError(f"{label} is not empty")
        try:
            fcntl.flock(pinned.fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise MigrationError(f"{label} is already locked") from error
        pinned.revalidate()
        yield pinned
    finally:
        if pinned.fd >= 0:
            try:
                fcntl.flock(pinned.fd, fcntl.LOCK_UN)
            except OSError:
                pass
        pinned.close()


def _binding_matches(
    observed: Mapping[str, Any], expected: Mapping[str, Any], *, label: str
) -> None:
    if set(observed) != FILE_BINDING_KEYS or set(expected) != FILE_BINDING_KEYS:
        raise MigrationError(f"{label} file-binding schema drifted")
    if dict(observed) != dict(expected):
        raise MigrationError(f"{label} file binding drifted")


def _read_json_binding(
    root: Path,
    relative: PurePosixPath | str,
    *,
    label: str,
    require_immutable: bool,
    expected_sha256: str | None = None,
    expected_bytes: int | None = None,
    expected_content_sha256: str | None = None,
    require_sorted_pretty: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    with PinnedRegular(root, relative, label=label) as pinned:
        raw = pinned.read()
        binding = pinned.binding(raw=raw)
    if require_immutable and binding["mode"] != "0444":
        raise MigrationError(f"{label} is not immutable mode 0444")
    if expected_sha256 is not None and binding["sha256"] != expected_sha256:
        raise MigrationError(f"{label} file sha256 mismatch")
    if expected_bytes is not None and binding["bytes"] != expected_bytes:
        raise MigrationError(f"{label} byte count mismatch")
    document = _decode_json(raw, label=label)
    _validate_self_hash(document, label=label)
    if (
        expected_content_sha256 is not None
        and document.get("content_sha256") != expected_content_sha256
    ):
        raise MigrationError(f"{label} semantic hash mismatch")
    if require_sorted_pretty and raw != pretty_json_bytes(document):
        raise MigrationError(f"{label} encoding is not deterministic pretty JSON")
    governance = {
        **binding,
        "content_sha256": document["content_sha256"],
    }
    if set(governance) != GOVERNANCE_BINDING_KEYS:
        raise AssertionError("internal governance binding schema drifted")
    return document, governance


def _validate_source_succession(
    root: Path,
    *,
    original_authority_binding: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    """Validate the sole additive source succession accepted by migration.

    The historical migration receipt remains governed by its original
    authority.  This addendum only identifies the exact later-authorized
    parser/wrapper bytes that may replay the same immutable receipt and live
    append-only state.
    """

    diagnostic, diagnostic_binding = _read_json_binding(
        root,
        SOURCE_SUCCESSION_DIAGNOSTIC_RELATIVE,
        label="migration source-succession diagnostic",
        require_immutable=True,
        expected_sha256=SOURCE_SUCCESSION_DIAGNOSTIC_FILE_SHA256,
        expected_bytes=SOURCE_SUCCESSION_DIAGNOSTIC_BYTES,
        expected_content_sha256=SOURCE_SUCCESSION_DIAGNOSTIC_CONTENT_SHA256,
    )
    if not (
        diagnostic.get("classification")
        == SOURCE_SUCCESSION_DIAGNOSTIC_CLASSIFICATION
        and diagnostic.get("campaign_id") == CAMPAIGN_ID
        and diagnostic.get("scientific_campaign_revision")
        == SCIENTIFIC_CAMPAIGN_REVISION
        and diagnostic.get("status")
        == "diagnosed_not_authorized_by_diagnostic"
        and diagnostic.get("claim_boundary", {}).get(
            "gpu_execution_authorized"
        )
        is False
        and diagnostic.get("claim_boundary", {}).get(
            "outer_reference_accessed"
        )
        is False
    ):
        raise MigrationError("migration source-succession diagnostic drifted")
    authority, authority_binding = _read_json_binding(
        root,
        SOURCE_SUCCESSION_AUTHORIZATION_RELATIVE,
        label="migration source-succession authority",
        require_immutable=True,
        expected_sha256=SOURCE_SUCCESSION_AUTHORIZATION_FILE_SHA256,
        expected_bytes=SOURCE_SUCCESSION_AUTHORIZATION_BYTES,
        expected_content_sha256=SOURCE_SUCCESSION_AUTHORIZATION_CONTENT_SHA256,
    )
    expected_keys = {
        "authority_basis",
        "authorized_modifications",
        "campaign_id",
        "claim_boundary",
        "classification",
        "content_sha256",
        "created_utc",
        "forbidden_changes",
        "infrastructure_revision",
        "mandatory_invariants",
        "required_reauthorization",
        "schema_version",
        "scientific_campaign_revision",
    }
    basis = authority.get("authority_basis")
    claim = authority.get("claim_boundary")
    mandatory = authority.get("mandatory_invariants")
    if not (
        set(authority) == expected_keys
        and authority.get("schema_version") == SCHEMA_VERSION
        and authority.get("classification")
        == SOURCE_SUCCESSION_AUTHORIZATION_CLASSIFICATION
        and authority.get("campaign_id") == CAMPAIGN_ID
        and authority.get("scientific_campaign_revision")
        == SCIENTIFIC_CAMPAIGN_REVISION
        and authority.get("infrastructure_revision")
        == INFRASTRUCTURE_REVISION
        and isinstance(basis, Mapping)
        and set(basis)
        == {
            "diagnostic",
            "execution_closure_authority",
            "immutable_migration_receipt",
            "original_migration_authority",
            "user_goal_scope",
        }
        and basis.get("diagnostic", {}).get("file_sha256")
        == diagnostic_binding["sha256"]
        and basis.get("diagnostic", {}).get("bytes")
        == diagnostic_binding["bytes"]
        and basis.get("diagnostic", {}).get("content_sha256")
        == diagnostic_binding["content_sha256"]
        and basis.get("original_migration_authority", {}).get("file_sha256")
        == original_authority_binding["sha256"]
        and basis.get("original_migration_authority", {}).get("bytes")
        == original_authority_binding["bytes"]
        and basis.get("original_migration_authority", {}).get(
            "content_sha256"
        )
        == original_authority_binding["content_sha256"]
        and isinstance(claim, Mapping)
        and claim.get("commercial_claim_authorized") is False
        and claim.get("confirmatory") is False
        and claim.get("gpu_execution_authorized_by_this_document") is False
        and claim.get("outer_reference_access_authorized") is False
        and claim.get("scientific_change_authorized") is False
        and isinstance(mandatory, Mapping)
        and mandatory.get("gpu_budget_ns") == GPU_BUDGET_NS
        and mandatory.get("strict_closed_live_state_required") is True
        and mandatory.get("immutable_original_migration_receipt") is True
        and mandatory.get("successor_sources") == SUCCESSOR_SOURCE_BINDINGS
    ):
        raise MigrationError("migration source-succession authority drifted")

    source_bindings: dict[str, dict[str, Any]] = {}
    for name, row in SUCCESSOR_SOURCE_BINDINGS.items():
        with PinnedRegular(
            root, str(row["path"]), label=f"successor source {name}"
        ) as pinned:
            raw = pinned.read()
            observed = pinned.binding(raw=raw)
        if not (
            observed["sha256"] == row["sha256"]
            and observed["bytes"] == row["bytes"]
            and observed["mode"] == "0444"
            and observed["nlink"] == 1
        ):
            raise MigrationError(f"successor source {name} drifted")
        source_bindings[name] = observed
    return source_bindings


def _validate_authority(root: Path) -> AuthorityContext:
    document, binding = _read_json_binding(
        root,
        AUTHORIZATION_RELATIVE,
        label="V8R4A correction authorization",
        require_immutable=True,
        expected_sha256=AUTHORIZATION_FILE_SHA256,
        expected_content_sha256=AUTHORIZATION_CONTENT_SHA256,
        require_sorted_pretty=True,
    )
    if set(document) != AUTHORIZATION_KEYS:
        raise MigrationError("V8R4A authorization schema drifted")
    if (
        document.get("schema_version") != SCHEMA_VERSION
        or document.get("classification") != AUTHORIZATION_CLASSIFICATION
        or document.get("campaign_id") != CAMPAIGN_ID
        or document.get("scientific_campaign_revision")
        != SCIENTIFIC_CAMPAIGN_REVISION
        or document.get("infrastructure_revision") != INFRASTRUCTURE_REVISION
    ):
        raise MigrationError("V8R4A authorization identity drifted")
    claim = document.get("claim_boundary")
    if not isinstance(claim, dict) or any(
        claim.get(key) is not False
        for key in (
            "commercial_claim_authorized",
            "confirmatory",
            "gpu_execution_authorized_by_this_document",
            "selection_or_promotion_authorized",
        )
    ):
        raise MigrationError("V8R4A authority claim boundary drifted")

    receipt_policy = document["migration_protocol"]["migration_receipt"]
    if (
        receipt_policy.get("classification") != MIGRATION_CLASSIFICATION
        or receipt_policy.get("path") != RECEIPT_RELATIVE.as_posix()
        or receipt_policy.get("production_runtime_authorized") is not False
        or frozenset(receipt_policy.get("exact_top_level_keys", []))
        != RECEIPT_KEYS
    ):
        raise MigrationError("V8R4A migration receipt policy drifted")
    layout = document.get("canonical_gpu_state_layout")
    if (
        not isinstance(layout, dict)
        or layout.get("root") != TARGET_ROOT_RELATIVE.as_posix()
        or layout.get("directory_mode") != "0700"
        or layout.get("root_exact_entries") != ["admission", "execution", "usage"]
        or layout.get("symlink_hardlink_or_extra_entry_allowed") is not False
    ):
        raise MigrationError("V8R4A canonical state layout drifted")
    for role, relative in ROLE_DIRECTORIES.items():
        row = layout.get("roles", {}).get(role)
        if (
            not isinstance(row, dict)
            or row.get("path") != relative.as_posix()
            or row.get("exact_entries") != list(ROLE_ENTRIES[role])
        ):
            raise MigrationError(f"V8R4A {role} layout drifted")

    evidence_rows: dict[str, Mapping[str, Any]] = {
        "parent_v8r4_authorization": document["authority_basis"][
            "parent_correction_authorization"
        ],
        "v8r4a_diagnostic": document["diagnostic"],
        "quarantine_owner_receipt": document["historical_evidence_acceptance"][
            "quarantine_owner_receipt"
        ],
        "quarantined_output_seal": document["historical_evidence_acceptance"][
            "quarantined_output_seal"
        ],
        "v8r4_failure_diagnostic": document["historical_evidence_acceptance"][
            "v8r4_failure_diagnostic"
        ],
    }
    historical: dict[str, dict[str, Any]] = {}
    diagnostic_document: dict[str, Any] | None = None
    for name, row in evidence_rows.items():
        if not isinstance(row, Mapping) or not {
            "path",
            "file_sha256",
            "bytes",
            "content_sha256",
        }.issubset(row):
            raise MigrationError(f"{name} authority binding schema drifted")
        evidence, observed = _read_json_binding(
            root,
            str(row["path"]),
            label=name,
            require_immutable=True,
            expected_sha256=str(row["file_sha256"]),
            expected_bytes=int(row["bytes"]),
            expected_content_sha256=str(row["content_sha256"]),
            require_sorted_pretty=False,
        )
        historical[name] = observed
        if name == "v8r4a_diagnostic":
            diagnostic_document = evidence
    assert diagnostic_document is not None

    original_sources = document.get("frozen_implementation_bindings")
    expected_original_sources = {
        "gpu_budget_ledger": {
            "bytes": 87716,
            "path": "src/snn_rr/gpu_budget_ledger.py",
            "sha256": "b71e792b860764ef78a3d9f77806dc2af77cc8f421856a0666d75650634f4c41",
        },
        "run_gpu_admitted": {
            "bytes": 138667,
            "path": "scripts/run_gpu_admitted.py",
            "sha256": "dfbf47cda09cd2e74f60ca298922ef1eb96f39f2d74b5c2ed2f06e73637be161",
        },
        "semantic_requirement": (
            "Both sources remain byte-identical. V8R4A changes only caller-supplied "
            "canonical paths, capability mounts, migration validation, and governance "
            "propagation."
        ),
    }
    if original_sources != expected_original_sources:
        raise MigrationError("original frozen implementation declaration drifted")
    source_bindings = _validate_source_succession(
        root,
        original_authority_binding=binding,
    )
    return AuthorityContext(
        document=document,
        binding=binding,
        historical_evidence=historical,
        diagnostic=diagnostic_document,
        source_bindings=source_bindings,
    )


def _load_budget_module(root: Path, authority: AuthorityContext) -> Any:
    source = authority.document["frozen_implementation_bindings"][
        "gpu_budget_ledger"
    ]["path"]
    path = _safe_path(root, str(source))
    name = "_snn_rr_v8r4a_frozen_gpu_budget_ledger"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise MigrationError("cannot load frozen GPU budget validator")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    prior_dont_write_bytecode = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        spec.loader.exec_module(module)
    except BaseException as error:
        sys.modules.pop(name, None)
        raise MigrationError(f"cannot execute frozen GPU budget validator: {error}") from error
    finally:
        sys.dont_write_bytecode = prior_dont_write_bytecode
    return module


def _verify_usage_ledger(
    raw: bytes, *, budget_module: Any, require_closed: bool
) -> dict[str, Any]:
    try:
        state = budget_module.verify_ledger_bytes(
            raw,
            budget_ns=GPU_BUDGET_NS,
            expected_legacy_genesis_sha256=LEGACY_GENESIS_SHA256,
        )
    except Exception as error:
        raise MigrationError(f"GPU usage ledger replay failed: {error}") from error
    if require_closed and state.open_reservations:
        raise MigrationError("GPU usage ledger has open reservations")
    return {
        "budget_ns": state.budget_ns,
        "open_reservation_count": len(state.open_reservations),
        "open_reservation_ns": state.open_reservation_ns,
        "record_count": len(state.records),
        "remaining_ns": state.remaining_ns,
        "settled_usage_ns": state.settled_usage_ns,
        "tail_record_sha256": state.tail_sha256,
    }


def _decode_execution_ledger(raw: bytes) -> list[dict[str, Any]]:
    if not raw:
        return []
    if not raw.endswith(b"\n"):
        raise MigrationError("GPU execution ledger has a torn tail")
    rows: list[dict[str, Any]] = []
    starts: set[str] = set()
    terminals: set[str] = set()
    for number, raw_line in enumerate(raw.splitlines(keepends=True), 1):
        if not raw_line.endswith(b"\n") or raw_line.endswith(b"\r\n"):
            raise MigrationError(
                f"GPU execution ledger line {number} has a noncanonical newline"
            )
        row = _decode_json(raw_line[:-1], label=f"execution ledger line {number}")
        if raw_line != canonical_json_bytes(row) + b"\n":
            raise MigrationError(
                f"GPU execution ledger line {number} is not canonical JSON"
            )
        event = row.get("event")
        if event not in {"start", "end", "wrapper_exception"}:
            raise MigrationError(
                f"GPU execution ledger line {number} has an invalid event"
            )
        lifecycle = row.get("lifecycle_id", row.get("job_id"))
        if not isinstance(lifecycle, str) or not lifecycle:
            raise MigrationError(
                f"GPU execution ledger line {number} lacks a lifecycle identity"
            )
        if event == "start":
            if lifecycle in starts:
                raise MigrationError("duplicate GPU execution start")
            starts.add(lifecycle)
        else:
            if lifecycle not in starts or lifecycle in terminals:
                raise MigrationError("orphan or duplicate GPU execution terminal")
            terminals.add(lifecycle)
        rows.append(row)
    return rows


def _verify_execution_ledger(raw: bytes, *, require_closed: bool) -> dict[str, Any]:
    rows = _decode_execution_ledger(raw)
    starts: dict[str, dict[str, Any]] = {}
    closed: set[str] = set()
    for row in rows:
        lifecycle = str(row.get("lifecycle_id", row.get("job_id", "")))
        if row.get("event") == "start":
            starts[lifecycle] = row
        else:
            closed.add(lifecycle)
    open_starts = sorted(set(starts) - closed)
    if require_closed and open_starts:
        raise MigrationError("GPU execution ledger has open starts")
    last_terminal = None
    if rows:
        value = rows[-1].get("terminal_record_sha256")
        if isinstance(value, str):
            last_terminal = value
    return {
        "last_line_sha256": (
            hashlib.sha256(raw.splitlines()[-1]).hexdigest() if raw else None
        ),
        "last_terminal_record_sha256": last_terminal,
        "open_start_count": len(open_starts),
        "record_count": len(rows),
    }


def _diagnostic_identities(authority: AuthorityContext) -> dict[str, tuple[int, int]]:
    state = authority.diagnostic.get("original_gpu_state")
    if not isinstance(state, dict):
        raise MigrationError("V8R4A diagnostic original state is missing")
    mapping: dict[str, tuple[int, int]] = {}
    for role in FILE_ROLES:
        row = state.get(role)
        if not isinstance(row, dict):
            raise MigrationError(f"V8R4A diagnostic lacks {role}")
        dev, ino = row.get("st_dev"), row.get("st_ino")
        if type(dev) is not int or type(ino) is not int or dev < 0 or ino <= 0:
            raise MigrationError(f"V8R4A diagnostic {role} identity is invalid")
        mapping[role] = (dev, ino)
    return mapping


def _validate_identity_override(
    value: Mapping[str, tuple[int, int]],
) -> dict[str, tuple[int, int]]:
    if set(value) != set(FILE_ROLES):
        raise MigrationError("test identity override role set drifted")
    result: dict[str, tuple[int, int]] = {}
    for role, identity in value.items():
        if (
            not isinstance(identity, tuple)
            or len(identity) != 2
            or any(type(item) is not int for item in identity)
            or identity[0] < 0
            or identity[1] <= 0
        ):
            raise MigrationError(f"test identity override for {role} is invalid")
        result[role] = identity
    return result


def _expected_prefix(authority: AuthorityContext) -> dict[str, Any]:
    policy = authority.document["migration_protocol"]["prefix_and_replay"]
    required = {
        "execution_initial_prefix_bytes": 22822,
        "execution_initial_prefix_sha256": (
            "079dcca8066a976e6a8746ac33479360b7fd39bab82efa7eb7991a3c95514cf4"
        ),
        "execution_open_start_count_after_copy": 0,
        "usage_initial_prefix_bytes": 109121,
        "usage_initial_prefix_sha256": (
            "9ce990030f51b40c5ccffc5146d20a0c754bf763e37e6cc6f76b8854edfaacba"
        ),
        "usage_open_reservation_count_after_copy": 0,
        "usage_tail_record_sha256_after_copy": (
            "8dbc0493125f22c130444e1344533d1f3d9c4ac445df6adfe8b597972e9691c5"
        ),
    }
    for key, expected in required.items():
        if policy.get(key) != expected:
            raise MigrationError(f"V8R4A prefix policy {key} drifted")
    return policy


def _capture_legacy(
    root: Path,
    authority: AuthorityContext,
    *,
    expected_identities: Mapping[str, tuple[int, int]],
) -> tuple[
    ExitStack,
    dict[str, PinnedRegular],
    dict[str, bytes],
    dict[str, dict[str, Any]],
    dict[str, Any],
    dict[str, Any],
]:
    stack = ExitStack()
    pinned: dict[str, PinnedRegular] = {}
    try:
        for role in LOCK_ROLES:
            pinned[role] = stack.enter_context(
                _exclusive_lock(root, LEGACY_RELATIVE[role], label=f"legacy {role}")
            )
        for role in LEDGER_ROLES:
            pinned[role] = stack.enter_context(
                PinnedRegular(root, LEGACY_RELATIVE[role], label=f"legacy {role}")
            )
        raw = {role: pinned[role].read() for role in FILE_ROLES}
        bindings = {
            role: pinned[role].binding(raw=raw[role]) for role in FILE_ROLES
        }
        state = authority.diagnostic["original_gpu_state"]
        for role in FILE_ROLES:
            binding = bindings[role]
            diagnostic = state[role]
            if (
                (binding["st_dev"], binding["st_ino"])
                != expected_identities[role]
                or binding["path"] != LEGACY_RELATIVE[role].as_posix()
                or binding["bytes"] != diagnostic["bytes"]
                or binding["sha256"] != diagnostic["sha256"]
                or binding["mode"] != "0644"
                or binding["nlink"] != 1
            ):
                raise MigrationError(f"legacy {role} does not match diagnostic state")
        budget_module = _load_budget_module(root, authority)
        usage_state = _verify_usage_ledger(
            raw["usage_ledger"], budget_module=budget_module, require_closed=True
        )
        execution_state = _verify_execution_ledger(
            raw["execution_ledger"], require_closed=True
        )
        prefix = _expected_prefix(authority)
        if (
            len(raw["usage_ledger"]) != prefix["usage_initial_prefix_bytes"]
            or hashlib.sha256(raw["usage_ledger"]).hexdigest()
            != prefix["usage_initial_prefix_sha256"]
            or usage_state["tail_record_sha256"]
            != prefix["usage_tail_record_sha256_after_copy"]
            or usage_state["open_reservation_count"] != 0
            or len(raw["execution_ledger"])
            != prefix["execution_initial_prefix_bytes"]
            or hashlib.sha256(raw["execution_ledger"]).hexdigest()
            != prefix["execution_initial_prefix_sha256"]
            or execution_state["open_start_count"] != 0
        ):
            raise MigrationError("legacy ledger prefix or closed state drifted")
        return stack, pinned, raw, bindings, usage_state, execution_state
    except BaseException:
        stack.close()
        raise


def _directory_binding(
    root: Path, relative: PurePosixPath, *, expected_entries: Sequence[str]
) -> dict[str, Any]:
    path = _safe_path(root, relative / "placeholder").parent
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(path, flags)
    except OSError as error:
        raise MigrationError(f"cannot open state directory {relative}: {error}") from error
    try:
        status = os.fstat(fd)
        named = os.stat(path, follow_symlinks=False)
        if not stat.S_ISDIR(status.st_mode) or not _same_inode(status, named):
            raise MigrationError(f"state directory {relative} identity drifted")
        entries = sorted(os.listdir(fd))
        if entries != sorted(expected_entries):
            raise MigrationError(f"state directory {relative} inventory drifted")
        if _format_mode(status.st_mode) != "0700":
            raise MigrationError(f"state directory {relative} mode drifted")
        return {
            "exact_entries": entries,
            "mode": _format_mode(status.st_mode),
            "path": relative.as_posix(),
            "st_dev": status.st_dev,
            "st_ino": status.st_ino,
        }
    finally:
        os.close(fd)


def _snapshot_target_tree(root: Path) -> TreeSnapshot:
    root_binding = _directory_binding(
        root, TARGET_ROOT_RELATIVE, expected_entries=tuple(ROLE_DIRECTORIES)
    )
    roles = {
        role: _directory_binding(root, relative, expected_entries=ROLE_ENTRIES[role])
        for role, relative in ROLE_DIRECTORIES.items()
    }
    identities = {(row["st_dev"], row["st_ino"]) for row in roles.values()}
    if len(identities) != len(roles):
        raise MigrationError("V8R4A role directories alias each other")
    files: dict[str, dict[str, Any]] = {}
    raw: dict[str, bytes] = {}
    for role in FILE_ROLES:
        with PinnedRegular(root, TARGET_RELATIVE[role], label=f"migrated {role}") as pin:
            payload = pin.read()
            binding = pin.binding(raw=payload)
        expected_mode = "0644"
        if binding["mode"] != expected_mode:
            raise MigrationError(f"migrated {role} mode drifted")
        if role in LOCK_ROLES and payload:
            raise MigrationError(f"migrated {role} is not empty")
        files[role] = binding
        raw[role] = payload
    return TreeSnapshot(root=root_binding, roles=roles, files=files, raw_files=raw)


def _write_all(fd: int, payload: bytes, *, label: str) -> None:
    offset = 0
    while offset < len(payload):
        written = os.write(fd, payload[offset:])
        if written <= 0:
            raise MigrationError(f"short write while creating {label}")
        offset += written


def _rename_noreplace(parent_fd: int, source: str, destination: str) -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        raise MigrationError("renameat2(RENAME_NOREPLACE) is unavailable")
    renameat2.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
    renameat2.restype = ctypes.c_int
    result = renameat2(
        parent_fd,
        os.fsencode(source),
        parent_fd,
        os.fsencode(destination),
        RENAME_NOREPLACE,
    )
    if result != 0:
        number = ctypes.get_errno()
        if number == errno.EEXIST:
            raise FileExistsError(number, os.strerror(number), destination)
        raise OSError(number, os.strerror(number), destination)


def _cleanup_unpublished_tree(parent_fd: int, name: str) -> None:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        root_fd = os.open(name, flags, dir_fd=parent_fd)
    except FileNotFoundError:
        return
    try:
        entries = sorted(os.listdir(root_fd))
        if entries != sorted(ROLE_DIRECTORIES):
            return
        for role in sorted(ROLE_DIRECTORIES):
            try:
                role_fd = os.open(role, flags, dir_fd=root_fd)
            except OSError:
                return
            try:
                if sorted(os.listdir(role_fd)) != sorted(ROLE_ENTRIES[role]):
                    return
                for filename in ROLE_ENTRIES[role]:
                    os.unlink(filename, dir_fd=role_fd)
            finally:
                os.close(role_fd)
            os.rmdir(role, dir_fd=root_fd)
    finally:
        os.close(root_fd)
    os.rmdir(name, dir_fd=parent_fd)


def _publish_target_tree(
    root: Path,
    legacy_raw: Mapping[str, bytes],
    *,
    temporary_token: str | None,
) -> TreeSnapshot:
    target_parent = _safe_path(
        root, TARGET_ROOT_RELATIVE.parent / "placeholder"
    ).parent
    target_path = target_parent / TARGET_ROOT_RELATIVE.name
    if os.path.lexists(target_path):
        raise MigrationError("V8R4A state root already exists without a receipt")
    parent = target_parent
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    parent_fd = os.open(parent, flags)
    token = temporary_token or secrets.token_hex(16)
    if not token or any(character not in "0123456789abcdefghijklmnopqrstuvwxyz" for character in token):
        os.close(parent_fd)
        raise MigrationError("temporary publication token is invalid")
    temporary = f".gpu_state_v8r4a.migrate.{token}"
    created_root = False
    published = False
    try:
        os.mkdir(temporary, 0o700, dir_fd=parent_fd)
        created_root = True
        root_fd = os.open(temporary, flags, dir_fd=parent_fd)
        try:
            os.fchmod(root_fd, 0o700)
            for role in sorted(ROLE_DIRECTORIES):
                os.mkdir(role, 0o700, dir_fd=root_fd)
                role_fd = os.open(role, flags, dir_fd=root_fd)
                try:
                    os.fchmod(role_fd, 0o700)
                    for filename in ROLE_ENTRIES[role]:
                        file_role = next(
                            key
                            for key, relative in TARGET_RELATIVE.items()
                            if relative.name == filename
                        )
                        payload = (
                            legacy_raw[file_role]
                            if file_role in LEDGER_ROLES
                            else b""
                        )
                        create_flags = (
                            os.O_WRONLY
                            | os.O_CREAT
                            | os.O_EXCL
                            | getattr(os, "O_CLOEXEC", 0)
                            | getattr(os, "O_NOFOLLOW", 0)
                        )
                        fd = os.open(filename, create_flags, 0o600, dir_fd=role_fd)
                        try:
                            _write_all(fd, payload, label=file_role)
                            os.fchmod(fd, 0o644)
                            os.fsync(fd)
                            status = os.fstat(fd)
                            if not stat.S_ISREG(status.st_mode) or status.st_nlink != 1:
                                raise MigrationError(f"created {file_role} is aliased")
                        finally:
                            os.close(fd)
                    os.fsync(role_fd)
                finally:
                    os.close(role_fd)
            os.fsync(root_fd)
        finally:
            os.close(root_fd)
        _rename_noreplace(parent_fd, temporary, target_path.name)
        published = True
        os.fsync(parent_fd)
    except FileExistsError as error:
        raise MigrationError("V8R4A state root publication raced") from error
    except OSError as error:
        raise MigrationError(f"V8R4A state root publication failed: {error}") from error
    finally:
        if created_root and not published:
            _cleanup_unpublished_tree(parent_fd, temporary)
        os.close(parent_fd)
    return _snapshot_target_tree(root)


def _create_once_receipt(root: Path, document: Mapping[str, Any], *, token: str) -> Path:
    path = _safe_path(root, RECEIPT_RELATIVE)
    parent_flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    parent_flags |= getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    parent_fd = os.open(path.parent, parent_flags)
    temporary = f".{path.name}.tmp.{token}"
    created_temp = False
    published = False
    try:
        flags = (
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0)
        )
        fd = os.open(temporary, flags, 0o400, dir_fd=parent_fd)
        created_temp = True
        try:
            payload = pretty_json_bytes(document)
            _write_all(fd, payload, label="migration receipt")
            os.fchmod(fd, 0o444)
            os.fsync(fd)
        finally:
            os.close(fd)
        _rename_noreplace(parent_fd, temporary, path.name)
        published = True
        os.fsync(parent_fd)
    except FileExistsError as error:
        raise MigrationError("migration receipt already exists or publication raced") from error
    except OSError as error:
        raise MigrationError(f"migration receipt publication failed: {error}") from error
    finally:
        if created_temp and not published:
            try:
                os.unlink(temporary, dir_fd=parent_fd)
            except FileNotFoundError:
                pass
        os.close(parent_fd)
    return path


def _validate_timestamp(value: str) -> str:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise MigrationError("created_utc must be a UTC Z timestamp")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as error:
        raise MigrationError("created_utc is invalid") from error
    if parsed.tzinfo != timezone.utc:
        raise MigrationError("created_utc is not UTC")
    return value


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace(
        "+00:00", "Z"
    )


def _build_receipt(
    *,
    authority: AuthorityContext,
    created_utc: str,
    original_bindings: Mapping[str, Mapping[str, Any]],
    retired_bindings: Mapping[str, Mapping[str, Any]],
    target: TreeSnapshot,
    usage_state: Mapping[str, Any],
    execution_state: Mapping[str, Any],
) -> dict[str, Any]:
    lineage = {
        role: {
            "legacy_retired": dict(retired_bindings[role]),
            "migrated_initial": dict(target.files[role]),
            "original": dict(original_bindings[role]),
        }
        for role in FILE_ROLES
    }
    receipt: dict[str, Any] = {
        "authority": dict(authority.binding),
        "campaign_id": CAMPAIGN_ID,
        "classification": MIGRATION_CLASSIFICATION,
        "content_sha256": "",
        "created_utc": _validate_timestamp(created_utc),
        "directory_inventory": {
            "roles": target.roles,
            "root": target.root,
        },
        "historical_evidence": authority.historical_evidence,
        "infrastructure_revision": INFRASTRUCTURE_REVISION,
        "lifecycle_state": {
            "all_lifecycles_closed": True,
            "execution_open_start_count": execution_state["open_start_count"],
            "usage_open_reservation_count": usage_state["open_reservation_count"],
        },
        "migrated_state": {
            "files": target.files,
            "initial_ledger_bytes_equal_legacy": True,
        },
        "original_state": {
            "captured_before_publication": True,
            "files": dict(original_bindings),
        },
        "path_inode_lineage": lineage,
        "prefix_replay": {
            "execution": {
                "future_state_must_begin_with_exact_prefix": True,
                "initial_prefix_bytes": len(target.raw_files["execution_ledger"]),
                "initial_prefix_sha256": hashlib.sha256(
                    target.raw_files["execution_ledger"]
                ).hexdigest(),
                **dict(execution_state),
            },
            "old_path_values_accepted_only_within_exact_prefix": True,
            "usage": {
                "future_state_must_begin_with_exact_prefix": True,
                "initial_prefix_bytes": len(target.raw_files["usage_ledger"]),
                "initial_prefix_sha256": hashlib.sha256(
                    target.raw_files["usage_ledger"]
                ).hexdigest(),
                **dict(usage_state),
            },
        },
        "production_runtime_authorized": False,
        "schema_version": SCHEMA_VERSION,
        "scientific_campaign_revision": SCIENTIFIC_CAMPAIGN_REVISION,
    }
    receipt["content_sha256"] = semantic_sha256(receipt)
    if set(receipt) != RECEIPT_KEYS:
        raise AssertionError("internal migration receipt schema drifted")
    return receipt


def _contains_exact_string(value: Any, forbidden: set[str]) -> bool:
    if isinstance(value, str):
        return value in forbidden
    if isinstance(value, list):
        return any(_contains_exact_string(item, forbidden) for item in value)
    if isinstance(value, dict):
        return any(_contains_exact_string(item, forbidden) for item in value.values())
    return False


def _verify_live_prefixes(
    root: Path,
    authority: AuthorityContext,
    *,
    legacy_raw: Mapping[str, bytes],
    target: TreeSnapshot,
    require_closed: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    prefix = _expected_prefix(authority)
    usage = target.raw_files["usage_ledger"]
    execution = target.raw_files["execution_ledger"]
    usage_bytes = prefix["usage_initial_prefix_bytes"]
    execution_bytes = prefix["execution_initial_prefix_bytes"]
    if (
        len(usage) < usage_bytes
        or usage[:usage_bytes] != legacy_raw["usage_ledger"]
        or hashlib.sha256(usage[:usage_bytes]).hexdigest()
        != prefix["usage_initial_prefix_sha256"]
        or len(execution) < execution_bytes
        or execution[:execution_bytes] != legacy_raw["execution_ledger"]
        or hashlib.sha256(execution[:execution_bytes]).hexdigest()
        != prefix["execution_initial_prefix_sha256"]
    ):
        raise MigrationError("migrated ledger exact prefix drifted")
    budget_module = _load_budget_module(root, authority)
    usage_state = _verify_usage_ledger(
        usage, budget_module=budget_module, require_closed=require_closed
    )
    execution_state = _verify_execution_ledger(
        execution, require_closed=require_closed
    )
    if (
        usage_state["record_count"] < 75
        or usage_state["settled_usage_ns"] < 1_408_703_699_500
        or usage_state["budget_ns"] != GPU_BUDGET_NS
        or execution_state["record_count"] < 8
    ):
        raise MigrationError("migrated ledger accounting regressed")
    legacy_paths = {str(_safe_path(root, relative)) for relative in LEGACY_RELATIVE.values()}
    for row in _decode_execution_ledger(execution[execution_bytes:]):
        if _contains_exact_string(row, legacy_paths):
            raise MigrationError("legacy state path appears outside execution prefix")
    usage_suffix = usage[usage_bytes:]
    if usage_suffix:
        try:
            suffix_rows = [
                _decode_json(line, label="usage suffix record")
                for line in usage_suffix.splitlines()
            ]
        except MigrationError:
            raise
        for row in suffix_rows:
            if _contains_exact_string(row, legacy_paths):
                raise MigrationError("legacy state path appears outside usage prefix")
    return usage_state, execution_state


def _load_budget_module_from_frozen_source(root: Path) -> Any:
    """Load the already capability-bound budget parser without global authority.

    Target-sealed children intentionally cannot see the historical governance
    tree used by :func:`_validate_authority`.  Their runtime capability binds
    the active source snapshot and the exact ``gpu_budget_ledger.py`` bytes,
    so lock-free replay must not reopen the unavailable historical chain.
    """

    path = _safe_path(root, PurePosixPath("src/snn_rr/gpu_budget_ledger.py"))
    specification = importlib.util.spec_from_file_location(
        "_hfr_v3r1_v8r4a_lock_free_gpu_budget", path
    )
    if specification is None or specification.loader is None:
        raise MigrationError("cannot load lock-free GPU budget validator")
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    prior_dont_write_bytecode = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        specification.loader.exec_module(module)
    except BaseException as error:
        raise MigrationError(
            f"cannot load lock-free GPU budget validator: {error}"
        ) from error
    finally:
        sys.dont_write_bytecode = prior_dont_write_bytecode
        sys.modules.pop(specification.name, None)
    return module


def _validate_trusted_state_snapshot(value: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the exact prelaunch state embedded in a runtime capability."""

    expected_top = {
        "migration_receipt",
        "directories",
        "files",
        "usage_state",
        "execution_state",
    }
    if not isinstance(value, Mapping) or set(value) != expected_top:
        raise MigrationError("trusted prelaunch state schema drifted")
    result = {key: value[key] for key in expected_top}
    receipt = result["migration_receipt"]
    directories = result["directories"]
    files = result["files"]
    usage = result["usage_state"]
    execution = result["execution_state"]
    if not (
        isinstance(receipt, Mapping)
        and set(receipt) == GOVERNANCE_BINDING_KEYS
        and receipt.get("path") == RECEIPT_RELATIVE.as_posix()
        and receipt.get("mode") == "0444"
        and receipt.get("nlink") == 1
        and isinstance(directories, Mapping)
        and set(directories) == {"root", *ROLE_DIRECTORIES}
        and isinstance(files, Mapping)
        and set(files) == set(FILE_ROLES)
        and isinstance(usage, Mapping)
        and type(usage.get("record_count")) is int
        and type(usage.get("open_reservation_count")) is int
        and isinstance(execution, Mapping)
        and type(execution.get("record_count")) is int
        and type(execution.get("open_start_count")) is int
    ):
        raise MigrationError("trusted prelaunch state identity drifted")
    for role, row in directories.items():
        if not isinstance(row, Mapping) or set(row) != DIRECTORY_BINDING_KEYS:
            raise MigrationError(f"trusted {role} directory binding drifted")
    for role, row in files.items():
        if not isinstance(row, Mapping) or set(row) != FILE_BINDING_KEYS:
            raise MigrationError(f"trusted {role} file binding drifted")
    return result


def _verify_lock_free_live_prefixes(
    root: Path,
    *,
    receipt: Mapping[str, Any],
    target: TreeSnapshot,
    trusted_state: Mapping[str, Any],
    require_closed: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Replay live ledgers while the admitted wrapper owns the lock hierarchy.

    The writer uses atomic replacement, therefore a live ledger's inode is
    deliberately *not* compared with its migration-time or prelaunch inode.
    Stable lock files and all three directory capabilities remain exact.
    """

    prefix = receipt.get("prefix_replay")
    if not isinstance(prefix, Mapping):
        raise MigrationError("migration receipt prefix replay is absent")
    initial_files = receipt.get("migrated_state", {}).get("files")
    if not isinstance(initial_files, Mapping) or set(initial_files) != set(FILE_ROLES):
        raise MigrationError("migration receipt initial files are absent")
    trusted_files = trusted_state["files"]
    for role in LOCK_ROLES:
        if target.files[role] != trusted_files[role]:
            raise MigrationError(f"lock-free {role} stable inode drifted")
        if target.files[role] != initial_files[role]:
            raise MigrationError(f"lock-free {role} migration inode drifted")

    for role, prefix_role in (
        ("usage_ledger", "usage"),
        ("execution_ledger", "execution"),
    ):
        current = target.files[role]
        trusted = trusted_files[role]
        policy = prefix.get(prefix_role)
        if not isinstance(policy, Mapping):
            raise MigrationError(f"lock-free {role} prefix policy drifted")
        raw = target.raw_files[role]
        trusted_bytes = trusted.get("bytes")
        trusted_sha = trusted.get("sha256")
        initial_bytes = policy.get("initial_prefix_bytes")
        initial_sha = policy.get("initial_prefix_sha256")
        if not (
            current.get("path") == TARGET_RELATIVE[role].as_posix()
            and current.get("mode") == "0644"
            and current.get("nlink") == 1
            and type(trusted_bytes) is int
            and trusted_bytes >= 0
            and isinstance(trusted_sha, str)
            and len(raw) >= trusted_bytes
            and hashlib.sha256(raw[:trusted_bytes]).hexdigest() == trusted_sha
            and type(initial_bytes) is int
            and initial_bytes >= 0
            and initial_bytes <= trusted_bytes
            and hashlib.sha256(raw[:initial_bytes]).hexdigest() == initial_sha
        ):
            raise MigrationError(f"lock-free {role} trusted prefix drifted")

    usage_raw = target.raw_files["usage_ledger"]
    execution_raw = target.raw_files["execution_ledger"]
    budget_module = _load_budget_module_from_frozen_source(root)
    usage_state = _verify_usage_ledger(
        usage_raw,
        budget_module=budget_module,
        require_closed=require_closed,
    )
    execution_state = _verify_execution_ledger(
        execution_raw, require_closed=require_closed
    )
    trusted_usage = trusted_state["usage_state"]
    trusted_execution = trusted_state["execution_state"]
    if (
        usage_state["record_count"] < trusted_usage["record_count"]
        or usage_state["settled_usage_ns"] < trusted_usage["settled_usage_ns"]
        or usage_state["budget_ns"] != GPU_BUDGET_NS
        or execution_state["record_count"] < trusted_execution["record_count"]
    ):
        raise MigrationError("lock-free migrated ledger accounting regressed")

    usage_initial_bytes = int(prefix["usage"]["initial_prefix_bytes"])
    execution_initial_bytes = int(prefix["execution"]["initial_prefix_bytes"])
    legacy_paths = {
        str(root.joinpath(*relative.parts)) for relative in LEGACY_RELATIVE.values()
    }
    for row in _decode_execution_ledger(execution_raw[execution_initial_bytes:]):
        if _contains_exact_string(row, legacy_paths):
            raise MigrationError("legacy state path appears outside execution prefix")
    for line in usage_raw[usage_initial_bytes:].splitlines():
        row = _decode_json(line, label="usage suffix record")
        if _contains_exact_string(row, legacy_paths):
            raise MigrationError("legacy state path appears outside usage prefix")
    return usage_state, execution_state


def _validate_receipt_document(
    root: Path,
    authority: AuthorityContext,
    document: Mapping[str, Any],
    *,
    legacy_raw: Mapping[str, bytes],
    retired_bindings: Mapping[str, Mapping[str, Any]],
    target: TreeSnapshot,
    require_closed: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if set(document) != RECEIPT_KEYS:
        raise MigrationError("migration receipt top-level schema drifted")
    _validate_self_hash(document, label="migration receipt")
    if (
        document.get("schema_version") != SCHEMA_VERSION
        or document.get("classification") != MIGRATION_CLASSIFICATION
        or document.get("campaign_id") != CAMPAIGN_ID
        or document.get("scientific_campaign_revision")
        != SCIENTIFIC_CAMPAIGN_REVISION
        or document.get("infrastructure_revision") != INFRASTRUCTURE_REVISION
        or document.get("production_runtime_authorized") is not False
    ):
        raise MigrationError("migration receipt identity drifted")
    _validate_timestamp(str(document.get("created_utc")))
    if document.get("authority") != authority.binding:
        raise MigrationError("migration receipt authority binding drifted")
    if document.get("historical_evidence") != authority.historical_evidence:
        raise MigrationError("migration receipt historical evidence drifted")
    inventory = document.get("directory_inventory")
    if not isinstance(inventory, dict) or set(inventory) != {"root", "roles"}:
        raise MigrationError("migration receipt directory inventory schema drifted")
    if inventory["root"] != target.root or inventory["roles"] != target.roles:
        raise MigrationError("migration receipt directory inode inventory drifted")
    for row in [inventory["root"], *inventory["roles"].values()]:
        if set(row) != DIRECTORY_BINDING_KEYS:
            raise MigrationError("migration receipt directory binding schema drifted")

    original = document.get("original_state")
    migrated = document.get("migrated_state")
    lineage = document.get("path_inode_lineage")
    if (
        not isinstance(original, dict)
        or set(original) != {"captured_before_publication", "files"}
        or original.get("captured_before_publication") is not True
        or not isinstance(migrated, dict)
        or set(migrated) != {"files", "initial_ledger_bytes_equal_legacy"}
        or migrated.get("initial_ledger_bytes_equal_legacy") is not True
        or not isinstance(lineage, dict)
        or set(lineage) != set(FILE_ROLES)
    ):
        raise MigrationError("migration receipt state lineage schema drifted")
    if set(original["files"]) != set(FILE_ROLES) or set(migrated["files"]) != set(FILE_ROLES):
        raise MigrationError("migration receipt file role set drifted")
    for role in FILE_ROLES:
        edge = lineage[role]
        if not isinstance(edge, dict) or set(edge) != {
            "legacy_retired",
            "migrated_initial",
            "original",
        }:
            raise MigrationError(f"migration receipt {role} lineage schema drifted")
        for binding in (edge["legacy_retired"], edge["migrated_initial"], edge["original"]):
            if not isinstance(binding, dict) or set(binding) != FILE_BINDING_KEYS:
                raise MigrationError(f"migration receipt {role} binding schema drifted")
        if (
            edge["original"] != original["files"][role]
            or edge["migrated_initial"] != migrated["files"][role]
            or edge["legacy_retired"] != retired_bindings[role]
        ):
            raise MigrationError(f"migration receipt {role} lineage drifted")
        if role in LOCK_ROLES and migrated["files"][role] != target.files[role]:
            raise MigrationError(f"migrated {role} stable inode drifted")
        initial = migrated["files"][role]
        if initial["path"] != TARGET_RELATIVE[role].as_posix():
            raise MigrationError(f"migrated {role} path drifted")
        if initial["nlink"] != 1 or initial["mode"] != "0644":
            raise MigrationError(f"migrated {role} initial metadata drifted")
        if edge["original"]["path"] != LEGACY_RELATIVE[role].as_posix():
            raise MigrationError(f"legacy {role} original path drifted")
        if edge["original"]["mode"] != "0644":
            raise MigrationError(f"legacy {role} original mode drifted")
        if edge["legacy_retired"]["mode"] != "0444":
            raise MigrationError(f"legacy {role} retirement mode drifted")
        if (
            edge["original"]["st_dev"],
            edge["original"]["st_ino"],
        ) == (initial["st_dev"], initial["st_ino"]):
            raise MigrationError(f"legacy and migrated {role} inodes alias")
        for key in ("bytes", "sha256", "st_dev", "st_ino", "nlink", "path"):
            if edge["original"][key] != edge["legacy_retired"][key]:
                raise MigrationError(f"legacy {role} bytes or inode changed on retirement")
        if role in LEDGER_ROLES:
            expected_policy = _expected_prefix(authority)
            prefix_name = "usage" if role == "usage_ledger" else "execution"
            if (
                initial["bytes"]
                != expected_policy[f"{prefix_name}_initial_prefix_bytes"]
                or initial["sha256"]
                != expected_policy[f"{prefix_name}_initial_prefix_sha256"]
            ):
                raise MigrationError(f"migrated {role} initial prefix binding drifted")
        elif initial["bytes"] != 0 or initial["sha256"] != EMPTY_SHA256:
            raise MigrationError(f"migrated {role} initial lock binding drifted")

    usage_state, execution_state = _verify_live_prefixes(
        root,
        authority,
        legacy_raw=legacy_raw,
        target=target,
        require_closed=require_closed,
    )
    lifecycle = document.get("lifecycle_state")
    if (
        not isinstance(lifecycle, dict)
        or set(lifecycle)
        != {
            "all_lifecycles_closed",
            "execution_open_start_count",
            "usage_open_reservation_count",
        }
        or lifecycle.get("all_lifecycles_closed") is not True
        or lifecycle.get("execution_open_start_count") != 0
        or lifecycle.get("usage_open_reservation_count") != 0
    ):
        raise MigrationError("migration receipt lifecycle state drifted")
    prefix = document.get("prefix_replay")
    if not isinstance(prefix, dict) or set(prefix) != {"execution", "old_path_values_accepted_only_within_exact_prefix", "usage"}:
        raise MigrationError("migration receipt prefix schema drifted")
    if prefix.get("old_path_values_accepted_only_within_exact_prefix") is not True:
        raise MigrationError("migration receipt historical path scope drifted")
    expected_policy = _expected_prefix(authority)
    if (
        prefix["usage"].get("initial_prefix_bytes")
        != expected_policy["usage_initial_prefix_bytes"]
        or prefix["usage"].get("initial_prefix_sha256")
        != expected_policy["usage_initial_prefix_sha256"]
        or prefix["execution"].get("initial_prefix_bytes")
        != expected_policy["execution_initial_prefix_bytes"]
        or prefix["execution"].get("initial_prefix_sha256")
        != expected_policy["execution_initial_prefix_sha256"]
    ):
        raise MigrationError("migration receipt exact prefix binding drifted")
    if require_closed and (
        usage_state["open_reservation_count"] != 0
        or execution_state["open_start_count"] != 0
    ):
        raise MigrationError("live migrated ledgers are not closed")
    return usage_state, execution_state


def _read_receipt(root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    return _read_json_binding(
        root,
        RECEIPT_RELATIVE,
        label="V8R4A GPU-state migration receipt",
        require_immutable=True,
        require_sorted_pretty=True,
    )


def _path_exists(root: Path, relative: PurePosixPath) -> bool:
    path = root.joinpath(*relative.parts)
    return os.path.lexists(path)


def _acquire_target_locks(root: Path, stack: ExitStack) -> dict[str, PinnedRegular]:
    pinned: dict[str, PinnedRegular] = {}
    for role in LOCK_ROLES:
        pinned[role] = stack.enter_context(
            _exclusive_lock(root, TARGET_RELATIVE[role], label=f"migrated {role}")
        )
    return pinned


def _retired_bindings_from_pins(
    pinned: Mapping[str, PinnedRegular], raw: Mapping[str, bytes]
) -> dict[str, dict[str, Any]]:
    return {role: pinned[role].binding(raw=raw[role]) for role in FILE_ROLES}


def _verify_retired_against_receipt(
    receipt: Mapping[str, Any], retired: Mapping[str, Mapping[str, Any]]
) -> None:
    lineage = receipt.get("path_inode_lineage")
    if not isinstance(lineage, dict) or set(lineage) != set(FILE_ROLES):
        raise MigrationError("migration receipt lineage is missing")
    for role in FILE_ROLES:
        edge = lineage.get(role)
        if not isinstance(edge, dict) or edge.get("legacy_retired") != retired[role]:
            raise MigrationError(f"retired legacy {role} drifted")


def _live_validation(
    root: Path,
    authority: AuthorityContext,
    *,
    receipt_path: Path,
    expected_identities: Mapping[str, tuple[int, int]],
    require_closed: bool,
) -> MigratedStateValidation:
    expected_receipt = _safe_path(root, RECEIPT_RELATIVE)
    if receipt_path.absolute() != expected_receipt:
        raise MigrationError("migration receipt path is not canonical V8R4A")
    receipt, receipt_binding = _read_receipt(root)
    with ExitStack() as stack:
        legacy_pins: dict[str, PinnedRegular] = {}
        for role in LOCK_ROLES:
            legacy_pins[role] = stack.enter_context(
                _exclusive_lock(root, LEGACY_RELATIVE[role], label=f"legacy {role}")
            )
        for role in LEDGER_ROLES:
            legacy_pins[role] = stack.enter_context(
                PinnedRegular(root, LEGACY_RELATIVE[role], label=f"legacy {role}")
            )
        legacy_raw = {role: legacy_pins[role].read() for role in FILE_ROLES}
        retired = _retired_bindings_from_pins(legacy_pins, legacy_raw)
        for role in FILE_ROLES:
            if (
                (retired[role]["st_dev"], retired[role]["st_ino"])
                != expected_identities[role]
                or retired[role]["mode"] != "0444"
            ):
                raise MigrationError(f"retired legacy {role} identity or mode drifted")
        _verify_retired_against_receipt(receipt, retired)
        _acquire_target_locks(root, stack)
        target = _snapshot_target_tree(root)
        usage_state, execution_state = _validate_receipt_document(
            root,
            authority,
            receipt,
            legacy_raw=legacy_raw,
            retired_bindings=retired,
            target=target,
            require_closed=require_closed,
        )
        reloaded, reloaded_binding = _read_receipt(root)
        if reloaded != receipt or reloaded_binding != receipt_binding:
            raise MigrationError("migration receipt changed during live validation")
        refreshed_authority = _validate_authority(root)
        if (
            refreshed_authority.binding != authority.binding
            or refreshed_authority.historical_evidence
            != authority.historical_evidence
            or refreshed_authority.source_bindings != authority.source_bindings
        ):
            raise MigrationError("V8R4A authority inputs changed during validation")
        if receipt_binding["mode"] != "0444" or receipt_binding["nlink"] != 1:
            raise MigrationError("migration receipt publication is mutable or aliased")
    canonical_paths = {
        **{role: _safe_path(root, relative) for role, relative in TARGET_RELATIVE.items()},
        **{
            f"{role}_directory": _safe_path(root, relative / "placeholder").parent
            for role, relative in ROLE_DIRECTORIES.items()
        },
    }
    directories = {"root": target.root, **target.roles}
    return MigratedStateValidation(
        receipt_path=expected_receipt,
        receipt=dict(receipt),
        receipt_binding=dict(receipt_binding),
        canonical_paths=canonical_paths,
        directory_bindings=directories,
        current_file_bindings=target.files,
        usage_state=usage_state,
        execution_state=execution_state,
    )


def _resume_validation(
    root: Path,
    authority: AuthorityContext,
    *,
    expected_identities: Mapping[str, tuple[int, int]],
    require_closed: bool,
) -> MigrationResult:
    validated = _live_validation(
        root,
        authority,
        receipt_path=_safe_path(root, RECEIPT_RELATIVE),
        expected_identities=expected_identities,
        require_closed=require_closed,
    )
    return MigrationResult(
        receipt_path=validated.receipt_path,
        receipt=validated.receipt,
        resumed=True,
    )


def _migrate_gpu_state(
    project_root: Path,
    *,
    created_utc: str | None = None,
    temporary_token: str | None = None,
    expected_original_identities: Mapping[str, tuple[int, int]] | None = None,
) -> MigrationResult:
    """Internal engine; identity injection exists only for relocated CPU fixtures."""

    root = _canonical_project_root(project_root)
    authority = _validate_authority(root)
    expected = (
        _diagnostic_identities(authority)
        if expected_original_identities is None
        else _validate_identity_override(expected_original_identities)
    )
    receipt_exists = _path_exists(root, RECEIPT_RELATIVE)
    target_exists = _path_exists(root, TARGET_ROOT_RELATIVE)
    if receipt_exists != target_exists:
        raise MigrationError("partial migration state: receipt/tree presence differs")
    if receipt_exists:
        return _resume_validation(
            root, authority, expected_identities=expected, require_closed=True
        )

    stack, pinned, legacy_raw, original, usage_state, execution_state = _capture_legacy(
        root, authority, expected_identities=expected
    )
    token = temporary_token or secrets.token_hex(16)
    try:
        target = _publish_target_tree(
            root, legacy_raw, temporary_token=token
        )
        for role in FILE_ROLES:
            pinned[role].revalidate()
            if pinned[role].read() != legacy_raw[role]:
                raise MigrationError(f"legacy {role} changed during publication")
        for name, before in authority.source_bindings.items():
            with PinnedRegular(
                root, before["path"], label=f"frozen source {name}"
            ) as source:
                after = source.binding()
            if after != before:
                raise MigrationError(f"frozen source {name} changed during migration")
        _verify_live_prefixes(
            root,
            authority,
            legacy_raw=legacy_raw,
            target=target,
            require_closed=True,
        )
        for role in FILE_ROLES:
            pinned[role].chmod(0o444)
        retired = _retired_bindings_from_pins(pinned, legacy_raw)
        receipt = _build_receipt(
            authority=authority,
            created_utc=created_utc or utc_now(),
            original_bindings=original,
            retired_bindings=retired,
            target=target,
            usage_state=usage_state,
            execution_state=execution_state,
        )
        receipt_path = _create_once_receipt(root, receipt, token=token)
        published, _ = _read_receipt(root)
        if published != receipt:
            raise MigrationError("published migration receipt bytes drifted")
        _validate_receipt_document(
            root,
            authority,
            published,
            legacy_raw=legacy_raw,
            retired_bindings=retired,
            target=_snapshot_target_tree(root),
            require_closed=True,
        )
        return MigrationResult(
            receipt_path=receipt_path,
            receipt=published,
            resumed=False,
        )
    finally:
        stack.close()


def migrate_gpu_state(project_root: Path) -> MigrationResult:
    """Run the production migration against diagnostic-bound legacy inodes."""

    return _migrate_gpu_state(project_root)


def _validate_migration_receipt(
    project_root: Path,
    *,
    expected_original_identities: Mapping[str, tuple[int, int]] | None = None,
    require_closed: bool = True,
) -> MigrationResult:
    root = _canonical_project_root(project_root)
    authority = _validate_authority(root)
    expected = (
        _diagnostic_identities(authority)
        if expected_original_identities is None
        else _validate_identity_override(expected_original_identities)
    )
    if not _path_exists(root, RECEIPT_RELATIVE) or not _path_exists(
        root, TARGET_ROOT_RELATIVE
    ):
        raise MigrationError("migration receipt or state tree is absent")
    return _resume_validation(
        root,
        authority,
        expected_identities=expected,
        require_closed=require_closed,
    )


def validate_migration_receipt(project_root: Path) -> MigrationResult:
    """Validate the production receipt, exact prefix, and current closed state."""

    return _validate_migration_receipt(project_root)


def validate_migrated_state(
    project_root: Path,
    receipt_path: Path,
    *,
    require_closed: bool = True,
) -> MigratedStateValidation:
    """Read-only runtime validation of the exact V8R4A capability state.

    The call performs no publication, chmod, repair, reconciliation, or ledger
    append.  It briefly takes nonblocking flocks only to obtain a coherent
    snapshot, then returns the three exact directory capabilities, all five
    current file bindings, and the reduced usage/execution states.
    """

    root = _canonical_project_root(project_root)
    authority = _validate_authority(root)
    expected = _diagnostic_identities(authority)
    if not _path_exists(root, RECEIPT_RELATIVE) or not _path_exists(
        root, TARGET_ROOT_RELATIVE
    ):
        raise MigrationError("migration receipt or state tree is absent")
    return _live_validation(
        root,
        authority,
        receipt_path=receipt_path,
        expected_identities=expected,
        require_closed=require_closed,
    )


def validate_migrated_state_lock_free(
    project_root: Path,
    receipt_path: Path,
    *,
    trusted_prelaunch_state: Mapping[str, Any],
    require_closed: bool = True,
) -> MigratedStateValidation:
    """Replay a capability-bound state without attempting to acquire locks.

    This entry point is only for a target-sealed child.  Its trusted input is
    the exact ``prelaunch_gpu_state`` object from the independently validated
    runtime capability receipt.  It performs no mutation and never acquires a
    flock, which avoids self-deadlock while the admitted wrapper owns the
    admission, usage, and execution locks.
    """

    root = _canonical_project_root(project_root)
    expected_receipt = _safe_path(root, RECEIPT_RELATIVE)
    if receipt_path.absolute() != expected_receipt:
        raise MigrationError("migration receipt path is not canonical V8R4A")
    trusted = _validate_trusted_state_snapshot(trusted_prelaunch_state)
    receipt, receipt_binding = _read_receipt(root)
    if receipt_binding != trusted["migration_receipt"]:
        raise MigrationError("migration receipt differs from runtime capability")
    if not (
        set(receipt) == RECEIPT_KEYS
        and receipt.get("schema_version") == SCHEMA_VERSION
        and receipt.get("classification") == MIGRATION_CLASSIFICATION
        and receipt.get("campaign_id") == CAMPAIGN_ID
        and receipt.get("scientific_campaign_revision")
        == SCIENTIFIC_CAMPAIGN_REVISION
        and receipt.get("infrastructure_revision") == INFRASTRUCTURE_REVISION
        and receipt.get("production_runtime_authorized") is False
    ):
        raise MigrationError("lock-free migration receipt identity drifted")
    _validate_self_hash(receipt, label="migration receipt")

    target = _snapshot_target_tree(root)
    inventory = receipt.get("directory_inventory")
    if not (
        isinstance(inventory, Mapping)
        and set(inventory) == {"root", "roles"}
        and inventory.get("root") == target.root
        and inventory.get("roles") == target.roles
        and target.root == trusted["directories"]["root"]
        and all(
            target.roles[role] == trusted["directories"][role]
            for role in ROLE_DIRECTORIES
        )
    ):
        raise MigrationError("lock-free directory capability drifted")
    usage_state, execution_state = _verify_lock_free_live_prefixes(
        root,
        receipt=receipt,
        target=target,
        trusted_state=trusted,
        require_closed=require_closed,
    )
    reloaded, reloaded_binding = _read_receipt(root)
    if reloaded != receipt or reloaded_binding != receipt_binding:
        raise MigrationError("migration receipt changed during lock-free replay")
    canonical_paths = {
        **{role: _safe_path(root, relative) for role, relative in TARGET_RELATIVE.items()},
        **{
            f"{role}_directory": _safe_path(root, relative / "placeholder").parent
            for role, relative in ROLE_DIRECTORIES.items()
        },
    }
    return MigratedStateValidation(
        receipt_path=expected_receipt,
        receipt=dict(receipt),
        receipt_binding=dict(receipt_binding),
        canonical_paths=canonical_paths,
        directory_bindings={"root": target.root, **target.roles},
        current_file_bindings=target.files,
        usage_state=usage_state,
        execution_state=execution_state,
    )


def validate_migrated_state_target_scoped(
    project_root: Path,
    receipt_path: Path,
    *,
    trusted_prelaunch_state: Mapping[str, Any],
    require_closed: bool = True,
) -> MigratedStateValidation:
    """Take the canonical target locks around capability-scoped replay."""

    root = _canonical_project_root(project_root)
    with ExitStack() as stack:
        _acquire_target_locks(root, stack)
        return validate_migrated_state_lock_free(
            root,
            receipt_path,
            trusted_prelaunch_state=trusted_prelaunch_state,
            require_closed=require_closed,
        )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create or validate the authorized V8R4A GPU-state migration"
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Canonical project root (defaults to the installed script parent)",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Require and fully replay an existing immutable migration receipt",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        result = (
            validate_migration_receipt(args.project_root)
            if args.validate_only
            else migrate_gpu_state(args.project_root)
        )
    except MigrationError as error:
        print(f"V8R4A GPU-state migration refused: {error}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "content_sha256": result.receipt["content_sha256"],
                "receipt": str(result.receipt_path),
                "resumed": result.resumed,
                "status": "validated" if args.validate_only else "complete",
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
