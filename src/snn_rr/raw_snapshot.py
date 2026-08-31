"""One-shot, descriptor-pinned consumption of one raw acquisition session.

This module closes the gap between hashing a pathname and later reopening it
for parsing.  A :class:`RawSessionReader` opens the exact graph below one
pinned dataset-root descriptor, holds every raw descriptor for the complete
operation, and derives both owned arrays and their hashes from the same bytes.

The returned receipt is diagnostic provenance only.  It deliberately contains
``scientific_authority=False`` and cannot authorize synchronization, training,
evaluation, or release by itself.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import stat
import threading
from types import MappingProxyType
from typing import Any, Callable, Mapping

import numpy as np

from .data import (
    RADAR_BINS,
    RADAR_SAMPLE_RATE_HZ,
    XETHRU_META_CLOSE_TIMESTAMP_POLICY,
    XETHRU_META_EVIDENCE_SCHEMA,
    XETHRU_RECORD_BYTES,
    XETHRU_RECORD_DTYPE,
    BiopacParserEvidence,
    BiopacRecording,
    XeThruMeta,
    XeThruMetaEvidence,
    load_biopac_mat_bytes,
    parse_xethru_meta_bytes,
)


RAW_CONSUMPTION_SCHEMA = "snn_rr.raw_consumption_receipt.diagnostic.v1"
RAW_CONSUMPTION_PORTABLE_SCHEMA = "snn_rr.raw_consumption_portable.v1"
RAW_GRAPH_SCHEMA = "snn_rr.selected_session_raw_input_graph.v1"
XETHRU_RECORD_CONTRACT_SCHEMA = "snn_rr.xethru_record_contract.v1"
XETHRU_METADATA_CONTRACT_SCHEMA = "snn_rr.xethru_metadata_contract.v1"


class RawSnapshotError(ValueError):
    """Raised when exact raw-byte consumption cannot be proven."""


# Tests replace this hook to force namespace and byte races at deterministic
# boundaries.  Production callers have no hook argument and cannot weaken the
# reader contract through its public API.
_TEST_EVENT_HOOK: Callable[[str, str], None] | None = None


def _test_event(event: str, key: str) -> None:
    hook = _TEST_EVENT_HOOK
    if hook is not None:
        hook(event, key)


def _canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise RawSnapshotError("raw-consumption receipt is not canonical JSON") from exc


@dataclass(frozen=True, slots=True)
class FileIdentity:
    """Stable descriptor/path signature observed during one consumption."""

    device: int
    inode: int
    mode: int
    link_count: int
    byte_count: int
    mtime_ns: int
    ctime_ns: int

    @classmethod
    def from_stat(cls, value: os.stat_result) -> FileIdentity:
        return cls(
            device=int(value.st_dev),
            inode=int(value.st_ino),
            mode=int(value.st_mode),
            link_count=int(value.st_nlink),
            byte_count=int(value.st_size),
            mtime_ns=int(value.st_mtime_ns),
            ctime_ns=int(value.st_ctime_ns),
        )

    def to_dict(self) -> dict[str, int]:
        return {
            "device": self.device,
            "inode": self.inode,
            "mode": self.mode,
            "link_count": self.link_count,
            "bytes": self.byte_count,
            "mtime_ns": self.mtime_ns,
            "ctime_ns": self.ctime_ns,
        }


def _relative_path(value: str, label: str) -> str:
    if type(value) is not str or not value:
        raise RawSnapshotError(f"{label} must be a non-empty relative path")
    candidate = PurePosixPath(value)
    if (
        candidate.is_absolute()
        or value != candidate.as_posix()
        or any(part in {"", ".", ".."} for part in candidate.parts)
    ):
        raise RawSnapshotError(f"{label} must be a canonical contained relative path")
    return value


@dataclass(frozen=True, slots=True)
class RawRadarGraph:
    """Exact ordered metadata/chunk graph for one physical radar."""

    radar_id: int
    metadata_path: str
    data_paths: tuple[str, ...]

    def __post_init__(self) -> None:
        if type(self.radar_id) is not int or self.radar_id not in {1, 2, 3}:
            raise RawSnapshotError("radar_id must be one of exact integers 1, 2, 3")
        _relative_path(self.metadata_path, f"radar{self.radar_id}.metadata_path")
        if type(self.data_paths) is not tuple or not self.data_paths:
            raise RawSnapshotError(f"radar{self.radar_id}.data_paths must be non-empty")
        for index, value in enumerate(self.data_paths):
            _relative_path(value, f"radar{self.radar_id}.data_paths[{index}]")
        if len(set(self.data_paths)) != len(self.data_paths):
            raise RawSnapshotError(f"radar{self.radar_id} repeats a chunk path")
        if self.metadata_path in self.data_paths:
            raise RawSnapshotError(f"radar{self.radar_id} metadata aliases a chunk")

    def to_dict(self) -> dict[str, Any]:
        return {
            "radar_id": self.radar_id,
            "metadata_path": self.metadata_path,
            "data_paths": list(self.data_paths),
        }


@dataclass(frozen=True, slots=True)
class RawSessionGraph:
    """Exact relative raw graph selected independently of any sync receipt."""

    session_id: str
    selected_logical_session_id: str
    biopac_path: str
    radars: tuple[RawRadarGraph, ...]

    def __post_init__(self) -> None:
        if (
            type(self.session_id) is not str
            or not self.session_id
            or self.session_id != self.session_id.strip()
        ):
            raise RawSnapshotError("session_id must be a non-empty trimmed string")
        if (
            type(self.selected_logical_session_id) is not str
            or not self.selected_logical_session_id
            or self.selected_logical_session_id
            != self.selected_logical_session_id.strip()
        ):
            raise RawSnapshotError(
                "selected_logical_session_id must be a non-empty trimmed string"
            )
        _relative_path(self.biopac_path, "biopac_path")
        if (
            type(self.radars) is not tuple
            or any(type(item) is not RawRadarGraph for item in self.radars)
            or tuple(item.radar_id for item in self.radars) != (1, 2, 3)
        ):
            raise RawSnapshotError("raw session graph must contain ordered radars 1, 2, 3")
        all_paths = [self.biopac_path]
        for radar in self.radars:
            all_paths.append(radar.metadata_path)
            all_paths.extend(radar.data_paths)
        if len(set(all_paths)) != len(all_paths):
            raise RawSnapshotError("raw session graph contains aliased path entries")
        if any(PurePosixPath(path).parts[0] != self.session_id for path in all_paths):
            raise RawSnapshotError(
                "every raw graph path must remain inside its exact session directory"
            )

    def to_dict(self) -> dict[str, Any]:
        bindings: list[str] = ["biopac"]
        radar_documents: list[dict[str, Any]] = []
        for radar in self.radars:
            meta_key = f"radar{radar.radar_id}_meta"
            data_keys = [
                f"radar{radar.radar_id}_data_{index:02d}"
                for index in range(len(radar.data_paths))
            ]
            bindings.append(meta_key)
            bindings.extend(data_keys)
            radar_documents.append(
                {
                    "radar_id": radar.radar_id,
                    "metadata_binding": meta_key,
                    "data_bindings": data_keys,
                }
            )
        return {
            "schema": RAW_GRAPH_SCHEMA,
            "session_id": self.session_id,
            "selected_logical_session_id": self.selected_logical_session_id,
            "binding_keys": bindings,
            "biopac_binding": "biopac",
            "radars": radar_documents,
        }


@dataclass(frozen=True, slots=True)
class ConsumedFileBinding:
    """Content and descriptor identity for the bytes actually consumed."""

    key: str
    role: str
    relative_path: str
    byte_count: int
    sha256: str
    identity: FileIdentity

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "role": self.role,
            "path": self.relative_path,
            "filename": PurePosixPath(self.relative_path).name,
            "bytes": self.byte_count,
            "sha256": self.sha256,
            "descriptor_identity": self.identity.to_dict(),
        }

    def to_compatibility_binding(self) -> dict[str, Any]:
        return {
            "path": self.relative_path,
            "bytes": self.byte_count,
            "sha256": self.sha256,
        }


@dataclass(frozen=True, slots=True)
class RadarChunkEvidence:
    radar_id: int
    chunk_index: int
    binding_key: str
    filename: str
    byte_count: int
    frame_count: int
    zero_header_nonzero: int
    bin_count_invalid: int

    @property
    def eligible(self) -> bool:
        return self.zero_header_nonzero == 0 and self.bin_count_invalid == 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk_index": self.chunk_index,
            "filename": self.filename,
            "bytes": self.byte_count,
            "frame_count": self.frame_count,
            "record_bytes": XETHRU_RECORD_BYTES,
            "payload_bin_count": RADAR_BINS,
            "record_size_remainder_bytes": 0,
            "zero_header_nonzero": self.zero_header_nonzero,
            "bin_count_invalid": self.bin_count_invalid,
        }


@dataclass(frozen=True, slots=True)
class RadarRecordEvidence:
    radar_id: int
    chunks: tuple[RadarChunkEvidence, ...]

    @property
    def eligible(self) -> bool:
        return bool(self.chunks) and all(item.eligible for item in self.chunks)

    def to_dict(self) -> dict[str, Any]:
        return {
            "radar_id": self.radar_id,
            "chunks": [item.to_dict() for item in self.chunks],
            "eligible": self.eligible,
        }


@dataclass(frozen=True, slots=True)
class RadarMetadataEvidence:
    radar_id: int
    evidence: XeThruMetaEvidence

    def to_dict(self) -> dict[str, Any]:
        if type(self.radar_id) is not int or self.radar_id not in {1, 2, 3}:
            raise RawSnapshotError("metadata radar_id must be one of 1, 2, 3")
        if type(self.evidence) is not XeThruMetaEvidence:
            raise RawSnapshotError("metadata evidence must be exact XeThruMetaEvidence")
        document = self.evidence.to_dict()
        return {
            "radar_id": self.radar_id,
            "metadata_evidence": document,
            "eligible": document["consumption_eligible"],
        }


@dataclass(frozen=True, slots=True)
class RawConsumptionReceipt:
    """Diagnostic exact-consumption evidence; never an authorization token."""

    session_id: str
    dataset_root: str
    root_identity: FileIdentity
    graph: RawSessionGraph
    timezone_name: str
    fallback_rate_hz: float
    biopac_strict: bool
    require_valid_records: bool
    file_bindings: tuple[ConsumedFileBinding, ...]
    radar_record_evidence: tuple[RadarRecordEvidence, ...]
    radar_metadata_evidence: tuple[RadarMetadataEvidence, ...]
    biopac_parser_evidence: BiopacParserEvidence
    content_sha256: str

    @classmethod
    def build(
        cls,
        *,
        session_id: str,
        dataset_root: str,
        root_identity: FileIdentity,
        graph: RawSessionGraph,
        timezone_name: str,
        fallback_rate_hz: float,
        biopac_strict: bool,
        require_valid_records: bool,
        file_bindings: tuple[ConsumedFileBinding, ...],
        radar_record_evidence: tuple[RadarRecordEvidence, ...],
        radar_metadata_evidence: tuple[RadarMetadataEvidence, ...],
        biopac_parser_evidence: BiopacParserEvidence,
    ) -> RawConsumptionReceipt:
        if type(biopac_parser_evidence) is not BiopacParserEvidence:
            raise RawSnapshotError(
                "biopac_parser_evidence must be an exact BiopacParserEvidence"
            )
        # Validate every cross-field interval/channel invariant before hashing
        # the evidence into the diagnostic receipt.
        biopac_parser_evidence.to_dict()
        if (
            type(radar_metadata_evidence) is not tuple
            or tuple(item.radar_id for item in radar_metadata_evidence) != (1, 2, 3)
        ):
            raise RawSnapshotError(
                "radar_metadata_evidence must contain ordered radars 1, 2, 3"
            )
        for item in radar_metadata_evidence:
            item.to_dict()
        provisional = cls(
            session_id=session_id,
            dataset_root=dataset_root,
            root_identity=root_identity,
            graph=graph,
            timezone_name=timezone_name,
            fallback_rate_hz=fallback_rate_hz,
            biopac_strict=biopac_strict,
            require_valid_records=require_valid_records,
            file_bindings=file_bindings,
            radar_record_evidence=radar_record_evidence,
            radar_metadata_evidence=radar_metadata_evidence,
            biopac_parser_evidence=biopac_parser_evidence,
            content_sha256="",
        )
        digest = hashlib.sha256(
            _canonical_json_bytes(provisional._document(include_hash=False))
        ).hexdigest()
        return cls(
            session_id=session_id,
            dataset_root=dataset_root,
            root_identity=root_identity,
            graph=graph,
            timezone_name=timezone_name,
            fallback_rate_hz=fallback_rate_hz,
            biopac_strict=biopac_strict,
            require_valid_records=require_valid_records,
            file_bindings=file_bindings,
            radar_record_evidence=radar_record_evidence,
            radar_metadata_evidence=radar_metadata_evidence,
            biopac_parser_evidence=biopac_parser_evidence,
            content_sha256=digest,
        )

    @property
    def input_bindings(self) -> dict[str, dict[str, Any]]:
        return {
            item.key: item.to_compatibility_binding() for item in self.file_bindings
        }

    @property
    def raw_input_graph(self) -> dict[str, Any]:
        return self.graph.to_dict()

    @property
    def xethru_record_contract(self) -> dict[str, Any]:
        views = [item.to_dict() for item in self.radar_record_evidence]
        return {
            "schema": XETHRU_RECORD_CONTRACT_SCHEMA,
            "record_bytes": XETHRU_RECORD_BYTES,
            "payload_bin_count": RADAR_BINS,
            "views": views,
            "eligible": all(item["eligible"] for item in views),
        }

    @property
    def xethru_metadata_contract(self) -> dict[str, Any]:
        views = [item.to_dict() for item in self.radar_metadata_evidence]
        return {
            "schema": XETHRU_METADATA_CONTRACT_SCHEMA,
            "metadata_evidence_schema": XETHRU_META_EVIDENCE_SCHEMA,
            "record_bytes": XETHRU_RECORD_BYTES,
            "close_timestamp_policy": XETHRU_META_CLOSE_TIMESTAMP_POLICY,
            "views": views,
            "eligible": all(item["eligible"] for item in views),
        }

    @property
    def portable_projection(self) -> dict[str, Any]:
        """Return the restoration-portable exact-consumption projection.

        Descriptor identity and the absolute dataset root are deliberately
        excluded.  They remain in :meth:`to_dict` as local race-detection
        evidence, while this projection binds only scientific content and
        semantic parser policy that must survive a faithful restore.
        """

        return {
            "schema": RAW_CONSUMPTION_PORTABLE_SCHEMA,
            "diagnostic_only": True,
            "scientific_authority": False,
            "session_id": self.session_id,
            "reader_contract": {
                "one_shot": True,
                "root_descriptor_pinned": True,
                "all_graph_files_opened_before_consumption": True,
                "no_follow": True,
                "regular_file_required": True,
                "single_link_required": True,
                "owned_arrays": True,
                "live_raw_memmap_returned": False,
            },
            "raw_input_graph": self.raw_input_graph,
            "input_bindings": [
                {
                    "key": item.key,
                    "path": item.relative_path,
                    "bytes": item.byte_count,
                    "sha256": item.sha256,
                }
                for item in self.file_bindings
            ],
            "parser_policy": {
                "timezone_name": self.timezone_name,
                "fallback_rate_hz": self.fallback_rate_hz,
                "biopac_strict": self.biopac_strict,
                "xethru_meta_strict": False,
                "require_valid_records": self.require_valid_records,
            },
            "biopac_parser_evidence": self.biopac_parser_evidence.to_dict(),
            "xethru_metadata_contract": self.xethru_metadata_contract,
            "xethru_record_contract": self.xethru_record_contract,
        }

    @property
    def portable_content_sha256(self) -> str:
        """SHA-256 of :attr:`portable_projection` canonical JSON bytes."""

        return hashlib.sha256(
            _canonical_json_bytes(self.portable_projection)
        ).hexdigest()

    def _document(self, *, include_hash: bool) -> dict[str, Any]:
        document: dict[str, Any] = {
            "schema": RAW_CONSUMPTION_SCHEMA,
            "diagnostic_only": True,
            "scientific_authority": False,
            "session_id": self.session_id,
            "dataset_root": self.dataset_root,
            "root_identity": self.root_identity.to_dict(),
            "reader_contract": {
                "one_shot": True,
                "root_descriptor_pinned": True,
                "all_graph_files_opened_before_consumption": True,
                "no_follow": True,
                "regular_file_required": True,
                "single_link_required": True,
                "owned_arrays": True,
                "live_raw_memmap_returned": False,
                "timezone_name": self.timezone_name,
                "fallback_rate_hz": self.fallback_rate_hz,
                "biopac_strict": self.biopac_strict,
                "xethru_meta_strict": False,
                "require_valid_records": self.require_valid_records,
            },
            "raw_input_graph": self.raw_input_graph,
            "input_bindings": self.input_bindings,
            "consumed_files": [item.to_dict() for item in self.file_bindings],
            "biopac_parser_evidence": self.biopac_parser_evidence.to_dict(),
            "xethru_metadata_contract": self.xethru_metadata_contract,
            "xethru_record_contract": self.xethru_record_contract,
            "portable_projection": self.portable_projection,
            "portable_content_sha256": self.portable_content_sha256,
        }
        if include_hash:
            document["content_sha256"] = self.content_sha256
        return document

    def to_dict(self) -> dict[str, Any]:
        document = self._document(include_hash=True)
        expected = hashlib.sha256(
            _canonical_json_bytes(self._document(include_hash=False))
        ).hexdigest()
        if self.content_sha256 != expected:
            raise RawSnapshotError("raw-consumption receipt content hash mismatch")
        return document


@dataclass(frozen=True, slots=True)
class OwnedRadarRecording:
    """Owned, immutable arrays decoded from the exact hashed radar bytes."""

    radar_id: int
    zero: np.ndarray
    frame_sequence: np.ndarray
    bin_count: np.ndarray
    bins: np.ndarray
    meta: XeThruMeta
    chunk_lengths: tuple[int, ...]
    evidence: RadarRecordEvidence

    @property
    def timestamps_ms(self) -> np.ndarray:
        return self.meta.relative_timestamps_ms

    @property
    def frame_count(self) -> int:
        return int(self.frame_sequence.size)


@dataclass(frozen=True, slots=True)
class LoadedRawSession:
    """Exact session payload returned once, with no live raw file mapping."""

    session_id: str
    biopac: BiopacRecording
    radars: Mapping[int, OwnedRadarRecording]
    receipt: RawConsumptionReceipt


@dataclass(frozen=True, slots=True)
class _FileSpec:
    key: str
    role: str
    relative_path: str


@dataclass(frozen=True, slots=True)
class _DirectoryStep:
    parent_fd: int
    name: str
    fd: int
    identity: FileIdentity


@dataclass(slots=True)
class _PinnedFile:
    spec: _FileSpec
    fd: int
    identity: FileIdentity
    parent_fd: int
    filename: str
    directory_fds: tuple[int, ...]
    directory_steps: tuple[_DirectoryStep, ...]

    def verify_stable(self) -> None:
        try:
            after_fd = FileIdentity.from_stat(os.fstat(self.fd))
            after_path = FileIdentity.from_stat(
                os.stat(self.filename, dir_fd=self.parent_fd, follow_symlinks=False)
            )
        except OSError as exc:
            raise RawSnapshotError(
                f"raw file disappeared or rebound: {self.spec.relative_path}"
            ) from exc
        if after_fd != self.identity or after_path != self.identity:
            raise RawSnapshotError(
                f"raw file changed or rebound during consumption: {self.spec.relative_path}"
            )
        for step in self.directory_steps:
            try:
                child_fd = FileIdentity.from_stat(os.fstat(step.fd))
                child_path = FileIdentity.from_stat(
                    os.stat(step.name, dir_fd=step.parent_fd, follow_symlinks=False)
                )
            except OSError as exc:
                raise RawSnapshotError(
                    f"raw directory changed while consuming {self.spec.relative_path}"
                ) from exc
            if child_fd != step.identity or child_path != step.identity:
                raise RawSnapshotError(
                    f"raw directory changed or rebound while consuming "
                    f"{self.spec.relative_path}"
                )

    def close(self) -> None:
        try:
            os.close(self.fd)
        finally:
            for descriptor in reversed(self.directory_fds):
                try:
                    os.close(descriptor)
                except OSError:
                    pass


_FILE_FLAGS = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
_DIRECTORY_FLAGS = (
    os.O_RDONLY
    | getattr(os, "O_CLOEXEC", 0)
    | getattr(os, "O_NOFOLLOW", 0)
    | getattr(os, "O_DIRECTORY", 0)
)
_REQUIRED_OPEN_FLAG_NAMES = ("O_CLOEXEC", "O_NOFOLLOW", "O_DIRECTORY")


def _open_root_without_symlink_components(path: Path) -> tuple[int, FileIdentity]:
    """Open an absolute dataset root without following any path component."""

    if not path.is_absolute():
        raise RawSnapshotError("dataset root must be absolute before descriptor pinning")
    current_fd = -1
    try:
        current_fd = os.open(os.sep, _DIRECTORY_FLAGS)
        current_identity = FileIdentity.from_stat(os.fstat(current_fd))
        if not stat.S_ISDIR(current_identity.mode):
            raise RawSnapshotError("filesystem root is not a directory")
        for component in path.parts[1:]:
            before = FileIdentity.from_stat(
                os.stat(component, dir_fd=current_fd, follow_symlinks=False)
            )
            child_fd = os.open(component, _DIRECTORY_FLAGS, dir_fd=current_fd)
            opened = FileIdentity.from_stat(os.fstat(child_fd))
            if not stat.S_ISDIR(before.mode) or before != opened:
                os.close(child_fd)
                raise RawSnapshotError(
                    "dataset root contains a symlinked or unstable directory component"
                )
            os.close(current_fd)
            current_fd = child_fd
            current_identity = opened
        return current_fd, current_identity
    except (OSError, RawSnapshotError) as exc:
        if current_fd >= 0:
            os.close(current_fd)
        if isinstance(exc, RawSnapshotError):
            raise
        raise RawSnapshotError(f"cannot pin dataset root {path}: {exc}") from exc


def _open_pinned_file(root_fd: int, spec: _FileSpec) -> _PinnedFile:
    parts = PurePosixPath(spec.relative_path).parts
    directory_fds: list[int] = []
    directory_steps: list[_DirectoryStep] = []
    file_fd = -1
    try:
        current_fd = os.dup(root_fd)
        directory_fds.append(current_fd)
        for component in parts[:-1]:
            before = FileIdentity.from_stat(
                os.stat(component, dir_fd=current_fd, follow_symlinks=False)
            )
            child_fd = os.open(component, _DIRECTORY_FLAGS, dir_fd=current_fd)
            opened = FileIdentity.from_stat(os.fstat(child_fd))
            if not stat.S_ISDIR(before.mode) or before != opened:
                os.close(child_fd)
                raise RawSnapshotError(
                    f"raw directory is not a stable real directory: {spec.relative_path}"
                )
            directory_fds.append(child_fd)
            directory_steps.append(
                _DirectoryStep(
                    parent_fd=current_fd,
                    name=component,
                    fd=child_fd,
                    identity=opened,
                )
            )
            current_fd = child_fd

        filename = parts[-1]
        before_file = FileIdentity.from_stat(
            os.stat(filename, dir_fd=current_fd, follow_symlinks=False)
        )
        _test_event("after_file_path_stat", spec.key)
        file_fd = os.open(filename, _FILE_FLAGS, dir_fd=current_fd)
        opened_file = FileIdentity.from_stat(os.fstat(file_fd))
        if (
            not stat.S_ISREG(before_file.mode)
            or not stat.S_ISREG(opened_file.mode)
            or before_file.link_count != 1
            or opened_file.link_count != 1
            or before_file != opened_file
        ):
            raise RawSnapshotError(
                f"raw file must be a stable, unaliased regular file: {spec.relative_path}"
            )
        return _PinnedFile(
            spec=spec,
            fd=file_fd,
            identity=opened_file,
            parent_fd=current_fd,
            filename=filename,
            directory_fds=tuple(directory_fds),
            directory_steps=tuple(directory_steps),
        )
    except (OSError, RawSnapshotError) as exc:
        if file_fd >= 0:
            os.close(file_fd)
        for descriptor in reversed(directory_fds):
            try:
                os.close(descriptor)
            except OSError:
                pass
        if isinstance(exc, RawSnapshotError):
            raise
        raise RawSnapshotError(f"cannot pin raw file {spec.relative_path}: {exc}") from exc


def _read_exact(fd: int, count: int, *, key: str) -> bytes:
    chunks: list[bytes] = []
    remaining = count
    while remaining:
        try:
            chunk = os.read(fd, remaining)
        except OSError as exc:
            raise RawSnapshotError(f"cannot consume raw file {key}: {exc}") from exc
        if not chunk:
            raise RawSnapshotError(f"raw file was truncated during consumption: {key}")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _consume_file_bytes(
    pinned: _PinnedFile, *, block_bytes: int = 1024 * 1024
) -> tuple[bytes, str]:
    digest = hashlib.sha256()
    payload_chunks: list[bytes] = []
    remaining = pinned.identity.byte_count
    while remaining:
        amount = min(block_bytes, remaining)
        payload = _read_exact(pinned.fd, amount, key=pinned.spec.key)
        digest.update(payload)
        payload_chunks.append(payload)
        remaining -= len(payload)
        _test_event("after_read_block", pinned.spec.key)
    payload = b"".join(payload_chunks)
    if len(payload) != pinned.identity.byte_count:
        raise RawSnapshotError(f"raw byte count changed for {pinned.spec.key}")
    return payload, digest.hexdigest()


def _consume_radar_chunk(
    pinned: _PinnedFile,
    *,
    radar_id: int,
    chunk_index: int,
    records_per_block: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str, RadarChunkEvidence]:
    byte_count = pinned.identity.byte_count
    remainder = byte_count % XETHRU_RECORD_BYTES
    frame_count = byte_count // XETHRU_RECORD_BYTES
    if remainder or frame_count <= 0:
        raise RawSnapshotError(
            f"radar{radar_id} chunk {chunk_index} has invalid record geometry"
        )
    zero = np.empty(frame_count, dtype=np.uint32)
    sequence = np.empty(frame_count, dtype=np.uint32)
    bin_count = np.empty(frame_count, dtype=np.uint32)
    bins = np.empty((frame_count, RADAR_BINS), dtype=np.float32)
    digest = hashlib.sha256()
    block_bytes = records_per_block * XETHRU_RECORD_BYTES
    remaining = byte_count
    offset = 0
    zero_nonzero = 0
    bin_invalid = 0
    while remaining:
        amount = min(block_bytes, remaining)
        payload = _read_exact(pinned.fd, amount, key=pinned.spec.key)
        if len(payload) % XETHRU_RECORD_BYTES:
            raise RawSnapshotError(
                f"radar{radar_id} chunk {chunk_index} yielded a partial record"
            )
        digest.update(payload)
        records = np.frombuffer(payload, dtype=XETHRU_RECORD_DTYPE)
        stop = offset + len(records)
        zero[offset:stop] = records["zero"]
        sequence[offset:stop] = records["frame_sequence"]
        bin_count[offset:stop] = records["bin_count"]
        bins[offset:stop] = records["bins"]
        zero_nonzero += int(np.count_nonzero(records["zero"]))
        bin_invalid += int(np.count_nonzero(records["bin_count"] != RADAR_BINS))
        offset = stop
        remaining -= len(payload)
        _test_event("after_read_block", pinned.spec.key)
    if offset != frame_count:
        raise RawSnapshotError(
            f"radar{radar_id} chunk {chunk_index} frame count changed while reading"
        )
    evidence = RadarChunkEvidence(
        radar_id=radar_id,
        chunk_index=chunk_index,
        binding_key=pinned.spec.key,
        filename=pinned.filename,
        byte_count=byte_count,
        frame_count=frame_count,
        zero_header_nonzero=zero_nonzero,
        bin_count_invalid=bin_invalid,
    )
    return zero, sequence, bin_count, bins, digest.hexdigest(), evidence


def _binding(pinned: _PinnedFile, digest: str) -> ConsumedFileBinding:
    return ConsumedFileBinding(
        key=pinned.spec.key,
        role=pinned.spec.role,
        relative_path=pinned.spec.relative_path,
        byte_count=pinned.identity.byte_count,
        sha256=digest,
        identity=pinned.identity,
    )


def _lexical_relative(root: Path, value: Path, label: str) -> str:
    root_absolute = Path(os.path.abspath(os.fspath(root)))
    value_absolute = Path(os.path.abspath(os.fspath(value)))
    try:
        relative = value_absolute.relative_to(root_absolute).as_posix()
    except ValueError as exc:
        raise RawSnapshotError(f"{label} escapes the dataset root") from exc
    return _relative_path(relative, label)


def graph_from_subject(dataset_root: str | Path, subject: Any) -> RawSessionGraph:
    """Convert a discovered subject into an explicit, lexical relative graph."""

    root = Path(dataset_root)
    session_id = getattr(subject, "subject_id", None)
    selected = getattr(subject, "selected_session", None)
    biopac_path = getattr(subject, "biopac_path", None)
    if type(session_id) is not str or selected is None or biopac_path is None:
        raise RawSnapshotError("subject lacks a selected three-radar/BIOPAC graph")
    radars: list[RawRadarGraph] = []
    for radar_id in (1, 2, 3):
        stream = selected.radars.get(radar_id)
        if stream is None or stream.meta_path is None or not stream.data_paths:
            raise RawSnapshotError(f"{session_id} lacks exact radar{radar_id} paths")
        radars.append(
            RawRadarGraph(
                radar_id=radar_id,
                metadata_path=_lexical_relative(
                    root, Path(stream.meta_path), f"radar{radar_id}.metadata_path"
                ),
                data_paths=tuple(
                    _lexical_relative(
                        root, Path(path), f"radar{radar_id}.data_paths[{index}]"
                    )
                    for index, path in enumerate(stream.data_paths)
                ),
            )
        )
    return RawSessionGraph(
        session_id=session_id,
        selected_logical_session_id=str(selected.session_id),
        biopac_path=_lexical_relative(root, Path(biopac_path), "biopac_path"),
        radars=tuple(radars),
    )


class RawSessionReader:
    """Consume one explicit raw session graph exactly once.

    The reader creates no files and returns no mapping into live raw storage.
    All graph files are opened before the first byte is consumed.  File and
    directory identities are checked before open, against the opened
    descriptor, and again after all parsing is complete.
    """

    def __init__(
        self,
        dataset_root: str | Path,
        graph: RawSessionGraph,
        *,
        timezone_name: str = "Asia/Seoul",
        fallback_rate_hz: float = RADAR_SAMPLE_RATE_HZ,
        biopac_strict: bool = False,
        require_valid_records: bool = True,
        records_per_block: int = 4096,
    ) -> None:
        missing_flags = [
            name for name in _REQUIRED_OPEN_FLAG_NAMES if not hasattr(os, name)
        ]
        if missing_flags:
            raise RawSnapshotError(
                "platform lacks required descriptor flags: " + ", ".join(missing_flags)
            )
        if type(graph) is not RawSessionGraph:
            raise RawSnapshotError("graph must be an exact RawSessionGraph")
        if type(timezone_name) is not str or not timezone_name:
            raise RawSnapshotError("timezone_name must be a non-empty string")
        if type(fallback_rate_hz) not in {int, float} or isinstance(
            fallback_rate_hz, bool
        ):
            raise RawSnapshotError("fallback_rate_hz must be a finite real number")
        rate = float(fallback_rate_hz)
        if not math.isfinite(rate) or rate <= 0.0:
            raise RawSnapshotError("fallback_rate_hz must be positive and finite")
        if type(biopac_strict) is not bool or type(require_valid_records) is not bool:
            raise RawSnapshotError("reader policy flags must be exact booleans")
        if type(records_per_block) is not int or records_per_block <= 0:
            raise RawSnapshotError("records_per_block must be a positive exact integer")
        self.dataset_root = Path(os.path.abspath(os.fspath(dataset_root)))
        self.graph = graph
        self.timezone_name = timezone_name
        self.fallback_rate_hz = rate
        self.biopac_strict = biopac_strict
        self.require_valid_records = require_valid_records
        self.records_per_block = records_per_block
        self._consumed = False
        self._lock = threading.Lock()

    @classmethod
    def from_subject(
        cls,
        dataset_root: str | Path,
        subject: Any,
        **kwargs: Any,
    ) -> RawSessionReader:
        return cls(dataset_root, graph_from_subject(dataset_root, subject), **kwargs)

    def _file_specs(self) -> tuple[_FileSpec, ...]:
        specs: list[_FileSpec] = [
            _FileSpec("biopac", "biopac_mat", self.graph.biopac_path)
        ]
        for radar in self.graph.radars:
            specs.append(
                _FileSpec(
                    f"radar{radar.radar_id}_meta",
                    "xethru_metadata",
                    radar.metadata_path,
                )
            )
            specs.extend(
                _FileSpec(
                    f"radar{radar.radar_id}_data_{index:02d}",
                    "xethru_records",
                    path,
                )
                for index, path in enumerate(radar.data_paths)
            )
        return tuple(specs)

    def consume(self) -> LoadedRawSession:
        with self._lock:
            if self._consumed:
                raise RawSnapshotError("RawSessionReader is one-shot and was already consumed")
            self._consumed = True
        return self._consume_once()

    def _consume_once(self) -> LoadedRawSession:
        root_fd = -1
        pinned: list[_PinnedFile] = []
        root_identity: FileIdentity | None = None
        try:
            root_fd, root_identity = _open_root_without_symlink_components(
                self.dataset_root
            )
            _test_event("after_root_open", "dataset_root")

            specs = self._file_specs()
            for spec in specs:
                pinned.append(_open_pinned_file(root_fd, spec))
            inode_keys = [(item.identity.device, item.identity.inode) for item in pinned]
            if len(set(inode_keys)) != len(inode_keys):
                raise RawSnapshotError("raw graph aliases the same inode more than once")
            _test_event("after_all_files_opened", self.graph.session_id)
            by_key = {item.spec.key: item for item in pinned}
            bindings: dict[str, ConsumedFileBinding] = {}

            biopac_payload, biopac_digest = _consume_file_bytes(by_key["biopac"])
            bindings["biopac"] = _binding(by_key["biopac"], biopac_digest)
            biopac = load_biopac_mat_bytes(
                biopac_payload,
                source_path=self.dataset_root / self.graph.biopac_path,
                timezone_name=self.timezone_name,
                strict=self.biopac_strict,
            )
            biopac.data.setflags(write=False)

            loaded_radars: dict[int, OwnedRadarRecording] = {}
            record_evidence: list[RadarRecordEvidence] = []
            metadata_evidence: list[RadarMetadataEvidence] = []
            for radar_graph in self.graph.radars:
                radar_id = radar_graph.radar_id
                meta_key = f"radar{radar_id}_meta"
                chunk_keys = [
                    f"radar{radar_id}_data_{index:02d}"
                    for index in range(len(radar_graph.data_paths))
                ]
                expected_frames = sum(
                    by_key[key].identity.byte_count // XETHRU_RECORD_BYTES
                    for key in chunk_keys
                )
                if any(
                    by_key[key].identity.byte_count % XETHRU_RECORD_BYTES
                    for key in chunk_keys
                ):
                    raise RawSnapshotError(f"radar{radar_id} contains a partial record")
                meta_payload, meta_digest = _consume_file_bytes(by_key[meta_key])
                bindings[meta_key] = _binding(by_key[meta_key], meta_digest)
                meta = parse_xethru_meta_bytes(
                    meta_payload,
                    source_path=self.dataset_root / radar_graph.metadata_path,
                    expected_frames=expected_frames,
                    expected_chunk_filenames=tuple(
                        PurePosixPath(path).name for path in radar_graph.data_paths
                    ),
                    expected_chunk_byte_counts=tuple(
                        by_key[key].identity.byte_count for key in chunk_keys
                    ),
                    fallback_rate_hz=self.fallback_rate_hz,
                    strict=False,
                )
                if (
                    type(meta.metadata_evidence) is not XeThruMetaEvidence
                    or not meta.metadata_evidence.consumption_eligible
                ):
                    raise RawSnapshotError(
                        f"radar{radar_id} explicit chunk order/metadata geometry "
                        "does not exactly join the consumed chunk graph"
                    )
                metadata_evidence.append(
                    RadarMetadataEvidence(radar_id, meta.metadata_evidence)
                )
                expected_names = tuple(
                    PurePosixPath(path).name for path in radar_graph.data_paths
                )
                if meta.chunk_filenames and meta.chunk_filenames != expected_names:
                    raise RawSnapshotError(
                        f"radar{radar_id} explicit chunk order differs from metadata footer"
                    )
                if (
                    meta.declared_chunk_count is not None
                    and meta.declared_chunk_count != len(expected_names)
                ):
                    raise RawSnapshotError(
                        f"radar{radar_id} metadata chunk count differs from the graph"
                    )

                zero_parts: list[np.ndarray] = []
                sequence_parts: list[np.ndarray] = []
                bin_count_parts: list[np.ndarray] = []
                bin_parts: list[np.ndarray] = []
                chunk_evidence: list[RadarChunkEvidence] = []
                for chunk_index, key in enumerate(chunk_keys):
                    (
                        zero,
                        sequence,
                        bin_count,
                        bins,
                        chunk_digest,
                        evidence,
                    ) = _consume_radar_chunk(
                        by_key[key],
                        radar_id=radar_id,
                        chunk_index=chunk_index,
                        records_per_block=self.records_per_block,
                    )
                    bindings[key] = _binding(by_key[key], chunk_digest)
                    zero_parts.append(zero)
                    sequence_parts.append(sequence)
                    bin_count_parts.append(bin_count)
                    bin_parts.append(bins)
                    chunk_evidence.append(evidence)
                evidence = RadarRecordEvidence(radar_id, tuple(chunk_evidence))
                if self.require_valid_records and not evidence.eligible:
                    raise RawSnapshotError(
                        f"radar{radar_id} violates zero-header/bin-count record contract"
                    )
                zero_all = np.concatenate(zero_parts)
                sequence_all = np.concatenate(sequence_parts)
                bin_count_all = np.concatenate(bin_count_parts)
                bins_all = np.concatenate(bin_parts, axis=0)
                for array in (zero_all, sequence_all, bin_count_all, bins_all):
                    array.setflags(write=False)
                meta.relative_timestamps_ms.setflags(write=False)
                loaded_radars[radar_id] = OwnedRadarRecording(
                    radar_id=radar_id,
                    zero=zero_all,
                    frame_sequence=sequence_all,
                    bin_count=bin_count_all,
                    bins=bins_all,
                    meta=meta,
                    chunk_lengths=tuple(item.frame_count for item in chunk_evidence),
                    evidence=evidence,
                )
                record_evidence.append(evidence)

            _test_event("before_final_verify", self.graph.session_id)
            for item in pinned:
                item.verify_stable()
            try:
                after_root_fd = FileIdentity.from_stat(os.fstat(root_fd))
                after_root_path = FileIdentity.from_stat(
                    os.stat(self.dataset_root, follow_symlinks=False)
                )
            except OSError as exc:
                raise RawSnapshotError("dataset root disappeared or rebound") from exc
            if after_root_fd != root_identity or after_root_path != root_identity:
                raise RawSnapshotError("dataset root changed or rebound during consumption")

            ordered_bindings = tuple(bindings[spec.key] for spec in specs)
            receipt = RawConsumptionReceipt.build(
                session_id=self.graph.session_id,
                dataset_root=str(self.dataset_root),
                root_identity=root_identity,
                graph=self.graph,
                timezone_name=self.timezone_name,
                fallback_rate_hz=self.fallback_rate_hz,
                biopac_strict=self.biopac_strict,
                require_valid_records=self.require_valid_records,
                file_bindings=ordered_bindings,
                radar_record_evidence=tuple(record_evidence),
                radar_metadata_evidence=tuple(metadata_evidence),
                biopac_parser_evidence=biopac.parser_evidence,
            )
            return LoadedRawSession(
                session_id=self.graph.session_id,
                biopac=biopac,
                radars=MappingProxyType(dict(loaded_radars)),
                receipt=receipt,
            )
        finally:
            for item in reversed(pinned):
                item.close()
            if root_fd >= 0:
                os.close(root_fd)


__all__ = [
    "RAW_CONSUMPTION_SCHEMA",
    "RAW_CONSUMPTION_PORTABLE_SCHEMA",
    "RAW_GRAPH_SCHEMA",
    "XETHRU_METADATA_CONTRACT_SCHEMA",
    "RawSnapshotError",
    "FileIdentity",
    "RawRadarGraph",
    "RawSessionGraph",
    "ConsumedFileBinding",
    "RadarChunkEvidence",
    "RadarRecordEvidence",
    "RadarMetadataEvidence",
    "RawConsumptionReceipt",
    "OwnedRadarRecording",
    "LoadedRawSession",
    "graph_from_subject",
    "RawSessionReader",
]
