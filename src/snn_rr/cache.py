"""Feature-cache loading for grouped training and evaluation."""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
import math
import os
from pathlib import Path
import stat
import tempfile
from typing import Any, BinaryIO

import numpy as np
import pandas as pd

from .acquisition_contract import (
    ACQUISITION_COHORT_V1_CONTENT_SHA256,
    AcquisitionSessionContract,
    assign_stage_window,
    load_acquisition_reconstruction,
)
from .preprocess import SESSION_IDENTITY, identity_for_session
from .radar_timing import (
    CAUSAL_UNIFORM_INVALID_REASON_SCHEMA_V1,
    CAUSAL_UNIFORM_INVALID_REASON_SEMANTICS_SHA256_V1,
    canonical_ndarray_sha256,
)
from .synchronization import TimeMapping


DEFAULT_HISTORY_LAGS = (1, 2, 4, 8)
DEFAULT_HISTORY_ROLLING_WINDOWS = (4, 8)
ACQUISITION_CACHE_SCHEMA_VERSION = "snn_rr.feature_cache_acquisition.v1"
ACQUISITION_CACHE_SCHEMA_VERSION_V2 = "snn_rr.feature_cache_acquisition.v2"
ACQUISITION_CACHE_SCHEMA_VERSION_V3 = "snn_rr.feature_cache_acquisition.v3"
ACQUISITION_CACHE_SESSION_SCHEMA_VERSION_V3 = "snn_rr.feature_cache_session.v3"
ACQUISITION_CACHE_ROOT_SCHEMA_VERSION_V3 = "snn_rr.feature_cache_root.v3"
ACQUISITION_RECONSTRUCTION_SCHEMA_VERSION_V2 = "snn_rr.acquisition_reconstruction.v2"
ACQUISITION_RECONSTRUCTION_SCHEMA_VERSION_V3 = "snn_rr.acquisition_reconstruction.v3"
V3_INFERENCE_FEATURE_SCHEMA_VERSION = "snn_rr.feature_cache_inference_features.v1"
V3_FEATURE_AVAILABILITY_SCHEMA_VERSION = (
    "snn_rr.feature_cache_inference_availability.v1"
)
V3_TARGET_FIREWALL_SCHEMA_VERSION = "snn_rr.feature_cache_target_firewall.v1"
V3_METADATA_JOIN_SCHEMA_VERSION = "snn_rr.feature_cache_metadata_join.v1"
V3_REFERENCE_SUPPORT_SCHEMA_VERSION = "snn_rr.feature_cache_reference_support.v1"
V3_MAP_RANGE_FEATURE_NAMES = tuple(
    [f"raw_power_pooled_range_{index:03d}" for index in range(91)]
    + [f"candidate_iq_phase_power_range_{index:03d}" for index in range(91)]
)
V3_MAP_SOURCE_LINEAGE = (
    "radar_only_causal_frequency_power_features_raw_and_candidate_iq_phase_"
    "concatenated_on_range_axis"
)
SUPPORTED_ACQUISITION_CACHE_SCHEMA_VERSIONS = frozenset(
    {
        ACQUISITION_CACHE_SCHEMA_VERSION,
        ACQUISITION_CACHE_SCHEMA_VERSION_V2,
        ACQUISITION_CACHE_SCHEMA_VERSION_V3,
    }
)

# These fields are deliberately metadata-only.  In particular, the acquisition
# phase and phase-7 assignment may be informed by offline BIOPAC annotations
# and must never be added to an inference feature allowlist.
REQUIRED_ACQUISITION_ANNOTATION_COLUMNS = frozenset(
    {
        "reference_start_sample",
        "reference_end_sample",
        "reference_window_start_biopac_s",
        "reference_window_end_biopac_s",
        "radar_window_start_relative_s",
        "radar_window_end_relative_s",
        "sync_authorized",
        "sync_confidence",
        "alignment_scientific_eligible",
        "acquisition_phase",
        "acquisition_phase_name",
        "acquisition_phase_status",
        "acquisition_phase_confidence",
        "phase_overlap_fraction",
        "transition_window",
        "eligible_for_stage_metrics",
        "phase7_assignment",
        "acquisition_batch",
    }
)
V3_REQUIRED_ACQUISITION_ANNOTATION_COLUMNS = frozenset(
    {*REQUIRED_ACQUISITION_ANNOTATION_COLUMNS, "reference_mapping_available"}
)

_ACQUISITION_INDICATOR_KEYS = frozenset(
    {
        "acquisition_contract",
        "acquisition_contract_sha256",
        "acquisition_manifest_sha256",
        "acquisition_reconstruction_manifest_sha256",
        "acquisition_session_manifest_sha256",
        "scientific_eligible",
    }
)


@dataclass(frozen=True, slots=True)
class CacheProvenance:
    """Content binding and claim boundary for one loaded feature cache.

    Version-1/2 acquisition caches remain readable for historical diagnostics,
    but only a fully verified version-3 cache can be classified as
    ``acquisition_scientific``.  ``inventory_sha256`` binds the exact files
    consumed for ``selected_sessions``.  Version-3 derives that identity from
    the descriptor-pinned byte snapshots actually parsed, never a later path
    rehash.  Older versions retain diagnostic compatibility only.
    """

    classification: str
    root_manifest_path: str
    root_manifest_sha256: str
    root_manifest_content_sha256: str
    acquisition_schema_version: str | None
    acquisition_mode: str | None
    scientific_eligible: bool
    config_sha256: str | None
    pipeline_sha256: str | None
    reconstruction_content_sha256: str | None
    inventory_sha256: str
    inventory_file_count: int
    selected_sessions: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "classification": self.classification,
            "root_manifest_path": self.root_manifest_path,
            "root_manifest_sha256": self.root_manifest_sha256,
            "root_manifest_content_sha256": self.root_manifest_content_sha256,
            "acquisition_schema_version": self.acquisition_schema_version,
            "acquisition_mode": self.acquisition_mode,
            "scientific_eligible": self.scientific_eligible,
            "config_sha256": self.config_sha256,
            "pipeline_sha256": self.pipeline_sha256,
            "reconstruction_content_sha256": self.reconstruction_content_sha256,
            "inventory_sha256": self.inventory_sha256,
            "inventory_file_count": self.inventory_file_count,
            "selected_sessions": list(self.selected_sessions),
        }

    @property
    def content_sha256(self) -> str:
        return _canonical_sha256(self.to_dict())


@dataclass(slots=True)
class FeatureCache:
    maps: np.ndarray
    aux: np.ndarray
    metadata: pd.DataFrame
    frequencies_hz: np.ndarray
    provenance: CacheProvenance | None = None
    radar_timing_valid_mask: np.ndarray | None = None
    radar_timing_invalid_reason_mask: np.ndarray | None = None
    feature_availability_mask: np.ndarray | None = None
    feature_availability_names: tuple[str, ...] | None = None
    map_view_availability_mask: np.ndarray | None = None
    aux_feature_availability_mask: np.ndarray | None = None
    aux_feature_names: tuple[str, ...] | None = None

    def subset(self, indices: np.ndarray) -> "FeatureCache":
        indices = np.asarray(indices)
        metadata = self.metadata.iloc[indices].reset_index(drop=True)
        provenance = self.provenance
        if provenance is not None:
            selected_sessions = tuple(
                dict.fromkeys(metadata["session_id"].dropna().astype(str))
            ) if "session_id" in metadata else ()
            provenance = replace(
                provenance,
                classification=(
                    "acquisition_diagnostic"
                    if provenance.acquisition_schema_version is not None
                    else provenance.classification
                ),
                scientific_eligible=False,
                selected_sessions=selected_sessions,
            )
        return FeatureCache(
            maps=self.maps[indices],
            aux=self.aux[indices],
            metadata=metadata,
            frequencies_hz=self.frequencies_hz,
            provenance=provenance,
            radar_timing_valid_mask=(
                None
                if self.radar_timing_valid_mask is None
                else self.radar_timing_valid_mask[indices]
            ),
            radar_timing_invalid_reason_mask=(
                None
                if self.radar_timing_invalid_reason_mask is None
                else self.radar_timing_invalid_reason_mask[indices]
            ),
            feature_availability_mask=(
                None
                if self.feature_availability_mask is None
                else self.feature_availability_mask[indices]
            ),
            feature_availability_names=self.feature_availability_names,
            map_view_availability_mask=(
                None
                if self.map_view_availability_mask is None
                else self.map_view_availability_mask[indices]
            ),
            aux_feature_availability_mask=(
                None
                if self.aux_feature_availability_mask is None
                else self.aux_feature_availability_mask[indices]
            ),
            aux_feature_names=self.aux_feature_names,
        )


def load_feature_cache(
    cache_dir: str | Path,
    *,
    sessions: list[str] | None = None,
    mmap: bool = False,
    require_acquisition_contract: bool = False,
    require_scientific_eligible: bool = False,
) -> FeatureCache:
    """Load and concatenate per-session feature files.

    ``mmap=True`` is only useful when loading one session; NumPy must allocate
    when multiple memory maps are concatenated.

    Legacy caches retain their historical behavior unless either strict flag
    is enabled.  An acquisition indicator anywhere in the root catalogue or a
    session manifest enables contract validation automatically, so omitting a
    flag cannot downgrade an acquisition-aware cache to legacy behavior.
    ``require_acquisition_contract`` fail-closes on a partially annotated
    cache: the root reconstruction contract and every usable session's
    content-addressed binding must be mutually consistent, and every metadata
    file must carry the complete offline-annotation schema.
    ``require_scientific_eligible`` implies the acquisition contract check and
    additionally requires the cache root and every usable session to be
    explicitly eligible.  This prevents selecting only the convenient rows or
    sessions from a mixed/unauthorized cache.
    """

    root = Path(cache_dir)
    root_manifest_path = root / "manifest.json"
    if not root_manifest_path.is_file():
        raise FileNotFoundError(f"feature cache manifest missing: {root_manifest_path}")
    root_manifest = _read_strict_json(root_manifest_path, "feature cache root manifest")
    indicated_contract = root_manifest.get("acquisition_contract")
    root_schema_is_v3 = bool(
        root_manifest.get("schema_version")
        == ACQUISITION_CACHE_ROOT_SCHEMA_VERSION_V3
    )
    contract_schema_is_v3 = bool(
        isinstance(indicated_contract, dict)
        and indicated_contract.get("schema_version")
        == ACQUISITION_CACHE_SCHEMA_VERSION_V3
    )
    if root_schema_is_v3 != contract_schema_is_v3:
        raise ValueError(
            "mixed version-3 cache schemas are forbidden; the root and "
            "acquisition contract schemas must agree exactly"
        )
    if root_schema_is_v3 and contract_schema_is_v3:
        return _load_feature_cache_v3(
            root,
            sessions=sessions,
            mmap=mmap,
            require_scientific_eligible=require_scientific_eligible,
        )
    declared_root_content = root_manifest.get("content_sha256")
    root_manifest_content_sha256 = _canonical_content_sha256(root_manifest)
    indicated_root_contract = root_manifest.get("acquisition_contract")
    root_is_v2 = bool(
        isinstance(indicated_root_contract, dict)
        and indicated_root_contract.get("schema_version")
        == ACQUISITION_CACHE_SCHEMA_VERSION_V2
    )
    if root_is_v2:
        _validate_sha256(
            declared_root_content,
            location="version-2 feature cache root content_sha256",
        )
    if declared_root_content is not None and declared_root_content != root_manifest_content_sha256:
        raise ValueError("feature cache root manifest content_sha256 mismatch")
    session_items = root_manifest.get("sessions")
    if not isinstance(session_items, list):
        raise ValueError("feature cache root manifest must contain a sessions list")
    if any(not isinstance(item, dict) for item in session_items):
        raise ValueError("feature cache root sessions must contain manifest objects")
    available_items = [item for item in session_items if item.get("status") == "ok"]
    available = [item["session_id"] for item in available_items]
    if len(set(available)) != len(available):
        raise ValueError("feature cache root manifest contains duplicate usable session IDs")
    selected = available if sessions is None else sessions
    if sessions is not None and require_scientific_eligible:
        raise ValueError(
            "scientific cache loading forbids any sessions filter; load the "
            "verified full-cohort catalogue"
        )
    if len(set(selected)) != len(selected):
        raise ValueError("requested sessions must not contain duplicates")
    missing = sorted(set(selected) - set(available))
    if missing:
        raise KeyError(f"sessions not present in cache: {missing}")

    # Acquisition-aware caches must never fall back to the permissive legacy
    # path merely because a caller forgot an opt-in flag.  Inspect catalogue
    # items and session manifests as well as the root so a partially stripped
    # root contract also fails closed.
    acquisition_indicated = _has_acquisition_indicator(root_manifest) or any(
        _has_acquisition_indicator(item) for item in available_items
    )
    if not acquisition_indicated:
        for session_id in available:
            session_manifest_path = root / session_id / "manifest.json"
            if not session_manifest_path.is_file():
                continue
            session_manifest = _read_strict_json(
                session_manifest_path,
                f"feature cache session manifest {session_id}",
            )
            if _has_acquisition_indicator(session_manifest):
                acquisition_indicated = True
                break

    strict_acquisition = bool(
        acquisition_indicated
        or require_acquisition_contract
        or require_scientific_eligible
    )
    acquisition_contract: dict[str, Any] | None = None
    if strict_acquisition:
        acquisition_contract = _validate_acquisition_cache_contract(
            root,
            root_manifest,
            available_items,
            # V1/V2 are validated fully for historical reproduction, but no
            # legacy claim may satisfy the current scientific gate.
            require_scientific_eligible=False,
        )
        if require_scientific_eligible:
            raise ValueError(
                "scientific cache loading requires acquisition cache version 3; "
                "versions 1 and 2 are historical/diagnostic only"
            )

    inventory_files: list[Path] = []
    for session_id in selected:
        session_dir = root / session_id
        session_manifest_path = session_dir / "manifest.json"
        if session_manifest_path.is_file():
            inventory_files.append(session_manifest_path)
        inventory_files.extend(
            session_dir / name
            for name in (
                "maps.npy",
                "aux.npy",
                "metadata.csv",
                "frequencies_hz.npy",
            )
        )
        timing_mask_path = session_dir / "radar_timing_valid_mask.npy"
        if timing_mask_path.is_file():
            inventory_files.append(timing_mask_path)
        range_aux_path = session_dir / "range_aux.npy"
        if range_aux_path.is_file():
            inventory_files.append(range_aux_path)
    pre_load_inventory_sha256: str | None = None
    pre_load_inventory_file_count: int | None = None
    if root_is_v2:
        (
            pre_load_inventory_sha256,
            pre_load_inventory_file_count,
        ) = _inventory_sha256(root, inventory_files)

    map_arrays: list[np.ndarray] = []
    aux_arrays: list[np.ndarray] = []
    frames: list[pd.DataFrame] = []
    timing_mask_arrays: list[np.ndarray] = []
    frequency_grid: np.ndarray | None = None
    for session_id in selected:
        session_dir = root / session_id
        mode = "r" if mmap else None
        map_array = np.load(session_dir / "maps.npy", mmap_mode=mode)
        aux_array = np.load(session_dir / "aux.npy", mmap_mode=mode)
        frame = pd.read_csv(session_dir / "metadata.csv")
        frequencies = np.load(session_dir / "frequencies_hz.npy")
        timing_mask = (
            np.load(
                session_dir / "radar_timing_valid_mask.npy",
                mmap_mode=mode,
                allow_pickle=False,
            )
            if root_is_v2
            else None
        )
        if not (len(map_array) == len(aux_array) == len(frame)):
            raise ValueError(f"cache length mismatch in {session_id}")
        if timing_mask is not None and len(timing_mask) != len(map_array):
            raise ValueError(f"cache timing-mask length mismatch in {session_id}")
        if frequency_grid is None:
            frequency_grid = frequencies
        elif not np.allclose(frequency_grid, frequencies):
            raise ValueError(f"frequency grid mismatch in {session_id}")
        map_arrays.append(map_array)
        aux_arrays.append(aux_array)
        frames.append(frame)
        if timing_mask is not None:
            timing_mask_arrays.append(timing_mask)

    if not map_arrays:
        raise ValueError("feature cache contains no selected sessions")
    maps = map_arrays[0] if len(map_arrays) == 1 else np.concatenate(map_arrays, axis=0)
    aux = aux_arrays[0] if len(aux_arrays) == 1 else np.concatenate(aux_arrays, axis=0)
    metadata = pd.concat(frames, ignore_index=True)
    radar_timing_valid_mask = (
        None
        if not timing_mask_arrays
        else (
            timing_mask_arrays[0]
            if len(timing_mask_arrays) == 1
            else np.concatenate(timing_mask_arrays, axis=0)
        )
    )
    inventory_sha256, inventory_file_count = _inventory_sha256(root, inventory_files)
    if root_is_v2 and (
        inventory_sha256 != pre_load_inventory_sha256
        or inventory_file_count != pre_load_inventory_file_count
    ):
        raise ValueError(
            "version-2 feature cache inventory changed during load"
        )

    schema_version = (
        None
        if acquisition_contract is None
        else str(acquisition_contract["schema_version"])
    )
    declared_eligible = bool(
        acquisition_contract is not None
        and acquisition_contract.get("scientific_eligible") is True
    )
    # V1/V2 remain historical/diagnostic compatibility surfaces.  Only the
    # descriptor-pinned V3 dispatch above may issue scientific provenance.
    effective_eligible = False
    if acquisition_contract is None:
        classification = "legacy"
    elif schema_version == ACQUISITION_CACHE_SCHEMA_VERSION:
        classification = "acquisition_historical_v1"
    elif (
        schema_version == ACQUISITION_CACHE_SCHEMA_VERSION_V2
        and declared_eligible
    ):
        classification = "acquisition_historical_v2"
    else:
        classification = "acquisition_diagnostic"
    provenance = CacheProvenance(
        classification=classification,
        root_manifest_path=str(root_manifest_path.resolve()),
        root_manifest_sha256=_sha256_file(root_manifest_path),
        root_manifest_content_sha256=root_manifest_content_sha256,
        acquisition_schema_version=schema_version,
        acquisition_mode=(
            None
            if acquisition_contract is None
            else str(acquisition_contract.get("mode"))
        ),
        scientific_eligible=effective_eligible,
        config_sha256=_optional_sha256(
            root_manifest.get("config_sha256"),
            strict=acquisition_contract is not None,
        ),
        pipeline_sha256=_optional_sha256(
            root_manifest.get("pipeline_sha256"),
            strict=acquisition_contract is not None,
        ),
        reconstruction_content_sha256=(
            None
            if acquisition_contract is None
            else str(acquisition_contract.get("reconstruction_content_sha256"))
        ),
        inventory_sha256=inventory_sha256,
        inventory_file_count=inventory_file_count,
        selected_sessions=tuple(map(str, selected)),
    )
    return FeatureCache(
        maps,
        aux,
        metadata,
        np.asarray(frequency_grid),
        provenance=provenance,
        radar_timing_valid_mask=radar_timing_valid_mask,
    )


def _strict_json_from_bytes(payload: bytes, label: str) -> dict[str, Any]:
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
        raise ValueError(f"cannot parse {label} ({error})") from error
    if not isinstance(document, dict):
        raise ValueError(f"{label} must be a JSON object")
    return document


def _read_strict_json(path: Path, label: str) -> dict[str, Any]:
    try:
        payload = path.read_bytes()
    except OSError as error:
        raise ValueError(f"cannot read {label}: {path} ({error})") from error
    return _strict_json_from_bytes(payload, label)


@dataclass(slots=True)
class _ConsumedFileSnapshot:
    """Unlinked private snapshot of one exact descriptor-consumed generation."""

    stream: BinaryIO
    byte_count: int
    sha256: str
    source_stat: tuple[int, ...]

    def close(self) -> None:
        self.stream.close()


def _safe_leaf_name(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value or value in {".", ".."}:
        raise ValueError(f"{label} must be a non-empty path leaf")
    if "\x00" in value or "/" in value or "\\" in value:
        raise ValueError(f"{label} contains path traversal")
    return value


def _directory_open_flags() -> int:
    required = ("O_DIRECTORY", "O_NOFOLLOW", "O_CLOEXEC")
    if any(not hasattr(os, name) for name in required):
        raise RuntimeError("version-3 cache loading requires secure Linux open flags")
    return os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC


def _open_directory_path_nofollow(path: Path, *, label: str) -> tuple[int, str]:
    """Open every absolute path component without following symlinks."""

    absolute = os.path.abspath(os.fspath(path))
    components = Path(absolute).parts
    flags = _directory_open_flags()
    descriptor = os.open(os.path.sep, flags)
    try:
        for component in components[1:]:
            leaf = _safe_leaf_name(component, label=f"{label} component")
            next_descriptor = os.open(leaf, flags, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = next_descriptor
        observed = os.fstat(descriptor)
        if not stat.S_ISDIR(observed.st_mode):
            raise ValueError(f"{label} is not a directory")
        return descriptor, absolute
    except BaseException:
        os.close(descriptor)
        raise


def _open_child_directory_nofollow(
    parent_fd: int,
    name: Any,
    *,
    label: str,
) -> tuple[int, tuple[int, ...]]:
    leaf = _safe_leaf_name(name, label=label)
    try:
        descriptor = os.open(leaf, _directory_open_flags(), dir_fd=parent_fd)
    except OSError as error:
        raise ValueError(f"cannot securely open {label}: {leaf} ({error})") from error
    observed = os.fstat(descriptor)
    if not stat.S_ISDIR(observed.st_mode):
        os.close(descriptor)
        raise ValueError(f"{label} is not a directory")
    return descriptor, _stable_stat_fingerprint(observed)


def _stable_stat_fingerprint(observed: os.stat_result) -> tuple[int, ...]:
    return (
        int(observed.st_dev),
        int(observed.st_ino),
        int(observed.st_mode),
        int(observed.st_nlink),
        int(observed.st_uid),
        int(observed.st_gid),
        int(observed.st_size),
        int(observed.st_mtime_ns),
        int(observed.st_ctime_ns),
    )


def _assert_child_directory_still_bound(
    parent_fd: int,
    name: str,
    expected: tuple[int, ...],
    *,
    label: str,
) -> None:
    try:
        observed = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    except OSError as error:
        raise ValueError(f"{label} changed during cache load ({error})") from error
    if not stat.S_ISDIR(observed.st_mode) or _stable_stat_fingerprint(observed) != expected:
        raise ValueError(f"{label} changed during cache load")


def _assert_regular_entry_still_bound(
    directory_fd: int,
    name: str,
    expected: tuple[int, ...],
    *,
    label: str,
) -> None:
    try:
        observed = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
    except OSError as error:
        raise ValueError(f"{label} changed during cache load ({error})") from error
    if (
        not stat.S_ISREG(observed.st_mode)
        or observed.st_nlink != 1
        or _stable_stat_fingerprint(observed) != expected
    ):
        raise ValueError(f"{label} changed during cache load")


def _snapshot_regular_file_at(
    directory_fd: int,
    name: Any,
    *,
    label: str,
) -> _ConsumedFileSnapshot:
    """Copy/hash one pinned regular nlink-1 file in the same single pass."""

    leaf = _safe_leaf_name(name, label=label)
    required = ("O_NOFOLLOW", "O_CLOEXEC")
    if any(not hasattr(os, item) for item in required):
        raise RuntimeError("version-3 cache loading requires secure Linux open flags")
    try:
        source_fd = os.open(
            leaf,
            os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC,
            dir_fd=directory_fd,
        )
    except OSError as error:
        raise ValueError(f"cannot securely open {label}: {leaf} ({error})") from error
    snapshot: BinaryIO | None = None
    try:
        before = os.fstat(source_fd)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError(f"{label} must be a regular file")
        if before.st_nlink != 1:
            raise ValueError(f"{label} must have exactly one hard link")
        before_fingerprint = _stable_stat_fingerprint(before)
        snapshot = tempfile.TemporaryFile(mode="w+b")
        digest = hashlib.sha256()
        consumed = 0
        while True:
            chunk = os.read(source_fd, 4 * 1024 * 1024)
            if not chunk:
                break
            written = snapshot.write(chunk)
            if written != len(chunk):
                raise ValueError(f"private snapshot short write for {label}")
            digest.update(chunk)
            consumed += len(chunk)
        after = os.fstat(source_fd)
        if _stable_stat_fingerprint(after) != before_fingerprint:
            raise ValueError(f"{label} changed during descriptor consumption")
        if consumed != before.st_size:
            raise ValueError(f"{label} byte count changed during consumption")
        try:
            rebound = os.stat(leaf, dir_fd=directory_fd, follow_symlinks=False)
        except OSError as error:
            raise ValueError(f"{label} directory entry changed during load") from error
        if _stable_stat_fingerprint(rebound) != before_fingerprint:
            raise ValueError(f"{label} directory entry changed during load")
        snapshot.flush()
        os.fsync(snapshot.fileno())
        snapshot.seek(0)
        result = _ConsumedFileSnapshot(
            stream=snapshot,
            byte_count=consumed,
            sha256=digest.hexdigest(),
            source_stat=before_fingerprint,
        )
        snapshot = None
        return result
    finally:
        os.close(source_fd)
        if snapshot is not None:
            snapshot.close()


def _snapshot_json_at(
    directory_fd: int,
    name: str,
    *,
    label: str,
) -> tuple[dict[str, Any], _ConsumedFileSnapshot]:
    snapshot = _snapshot_regular_file_at(directory_fd, name, label=label)
    try:
        payload = snapshot.stream.read()
        snapshot.stream.seek(0)
        document = _strict_json_from_bytes(payload, label)
    except BaseException:
        snapshot.close()
        raise
    return document, snapshot


def _consumed_inventory_record(
    relative_path: str,
    snapshot: _ConsumedFileSnapshot,
) -> dict[str, Any]:
    return {
        "path": relative_path,
        "bytes": snapshot.byte_count,
        "sha256": snapshot.sha256,
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(4 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _inventory_sha256(root: Path, paths: list[Path]) -> tuple[str, int]:
    inventory: list[dict[str, Any]] = []
    for path in sorted(set(paths), key=lambda item: str(item.relative_to(root))):
        if not path.is_file():
            raise FileNotFoundError(f"feature cache inventory file missing: {path}")
        inventory.append(
            {
                "path": str(path.relative_to(root)),
                "bytes": path.stat().st_size,
                "sha256": _sha256_file(path),
            }
        )
    return _canonical_sha256(inventory), len(inventory)


def _optional_sha256(value: Any, *, strict: bool) -> str | None:
    if value is None:
        return None
    try:
        return _validate_sha256(value, location="feature cache provenance SHA-256")
    except ValueError:
        if strict:
            raise
        # Some synthetic/early legacy manifests used descriptive placeholders
        # rather than digests.  Explicit legacy mode remains readable, but the
        # unverified value is not promoted into provenance.
        return None


def _canonical_json(value: Any) -> str:
    """Return a deterministic representation for exact contract comparison."""

    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    except (TypeError, ValueError) as error:
        raise ValueError("acquisition contract must be canonical JSON data") from error


def _canonical_content_sha256(document: dict[str, Any]) -> str:
    payload = dict(document)
    payload.pop("content_sha256", None)
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _validate_sha256(value: Any, *, location: str, nullable: bool = False) -> str | None:
    if value is None and nullable:
        return None
    if not isinstance(value, str):
        raise ValueError(f"{location} must be a lowercase SHA-256 digest")
    digest = value
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise ValueError(f"{location} must be a lowercase SHA-256 digest")
    return digest


def _annotation_columns(
    value: Any,
    *,
    location: str,
    expected_columns: frozenset[str] = REQUIRED_ACQUISITION_ANNOTATION_COLUMNS,
) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{location} must be a non-empty list")
    columns = tuple(str(column) for column in value)
    if len(set(columns)) != len(columns):
        raise ValueError(f"{location} must not contain duplicates")
    if set(columns) != set(expected_columns):
        missing = sorted(expected_columns - set(columns))
        extra = sorted(set(columns) - expected_columns)
        raise ValueError(
            f"{location} does not match the acquisition metadata schema; "
            f"missing={missing}, extra={extra}"
        )
    return columns


def _has_acquisition_indicator(document: dict[str, Any]) -> bool:
    return bool(_ACQUISITION_INDICATOR_KEYS.intersection(document))


def _explicit_boolean(value: Any, *, location: str) -> bool:
    """Read a JSON/CSV boolean without accepting truthy strings or numbers."""

    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    raise ValueError(f"{location} must be an explicit boolean")


def _session_ids_sha256(session_ids: list[str]) -> str:
    return _canonical_sha256(list(map(str, session_ids)))


def _unique_string_list(value: Any, *, location: str) -> list[str]:
    if not isinstance(value, list) or any(
        not isinstance(item, str) or not item for item in value
    ):
        raise ValueError(f"{location} must be a list of non-empty strings")
    if len(set(value)) != len(value):
        raise ValueError(f"{location} must not contain duplicates")
    return list(value)


_V3_ROOT_MANIFEST_KEYS = frozenset(
    {
        "schema_version",
        "content_sha256",
        "config_sha256",
        "pipeline_sha256",
        "acquisition_contract",
        "sessions",
    }
)
_V3_ROOT_CONTRACT_KEYS = frozenset(
    {
        "schema_version",
        "upstream_reconstruction_schema_version",
        "reconstruction_manifest",
        "reconstruction_manifest_sha256",
        "reconstruction_manifest_bytes",
        "reconstruction_content_sha256",
        "cohort_authority_sha256",
        "cohort_authority_content_sha256",
        "mode",
        "scientific_eligible",
        "subjects_filter_applied",
        "selection_scope",
        "full_cohort_complete",
        "expected_usable_session_ids",
        "expected_usable_session_ids_sha256",
        "cache_usable_session_ids",
        "cache_usable_session_ids_sha256",
        "cache_inventory_aggregate_sha256",
        "annotation_only_columns",
    }
)
_V3_ROOT_SESSION_ITEM_KEYS = frozenset(
    {
        "session_id",
        "status",
        "schema_version",
        "manifest_path",
        "manifest_sha256",
        "manifest_content_sha256",
        "inventory_sha256",
        "upstream_session_content_sha256",
        "scientific_eligible",
    }
)
_V3_SESSION_MANIFEST_KEYS = frozenset(
    {
        "schema_version",
        "session_id",
        "content_sha256",
        "config_sha256",
        "pipeline_sha256",
        "window_count",
        "scientific_eligible",
        "raw_consumed_bytes_verified",
        "timing_adjudicated",
        "sync_raw_replay_verified",
        "protocol_raw_replay_verified",
        "sync_authorized",
        "reference_mapping_available",
        "upstream_session_content_sha256",
        "upstream_session_contract",
        "upstream_sync_receipt",
        "radar_timing_invalid_reason_schema_version",
        "radar_timing_invalid_reason_semantics_sha256",
        "source_fingerprint",
        "feature_schema",
        "feature_schema_sha256",
        "target_firewall",
        "target_firewall_sha256",
        "metadata_join_contract",
        "metadata_join_sha256",
        "reference_support_contract",
        "reference_support_sha256",
        "file_inventory",
        "inventory_sha256",
    }
)
_V3_FILE_BINDING_KEYS = frozenset(
    {"path", "sha256", "bytes", "shape", "dtype"}
)
_V3_REQUIRED_FILES = {
    "maps": ("maps.npy", "float16"),
    "aux": ("aux.npy", "float32"),
    "metadata": ("metadata.csv", "csv"),
    "frequencies_hz": ("frequencies_hz.npy", "float64"),
    "radar_timing_valid_mask": ("radar_timing_valid_mask.npy", "bool"),
    "radar_timing_invalid_reason_mask": (
        "radar_timing_invalid_reason_mask.npy",
        "uint8",
    ),
    "feature_availability_mask": ("feature_availability_mask.npy", "bool"),
}

_V3_FEATURE_SCHEMA_KEYS = frozenset(
    {"schema_version", "maps", "aux", "availability"}
)
_V3_MAP_SCHEMA_KEYS = frozenset(
    {
        "axes",
        "radar_names",
        "range_feature_names",
        "shape",
        "dtype",
        "frequency_grid_sha256",
        "source_lineage",
        "target_derived_inputs",
    }
)
_V3_AUX_SCHEMA_KEYS = frozenset(
    {
        "axes",
        "feature_names",
        "feature_names_sha256",
        "shape",
        "dtype",
        "source_lineage",
        "target_derived_inputs",
    }
)
_V3_AVAILABILITY_SCHEMA_KEYS = frozenset(
    {
        "schema_version",
        "axes",
        "feature_names",
        "feature_names_sha256",
        "shape",
        "dtype",
        "semantics",
    }
)
_V3_TARGET_FIREWALL_KEYS = frozenset(
    {
        "schema_version",
        "inference_payloads",
        "target_derived_metadata_columns",
        "annotation_only_columns",
        "forbidden_inference_feature_names",
        "radar_observable_role",
        "target_values_used_in_inference_features",
    }
)
_V3_METADATA_JOIN_KEYS = frozenset(
    {
        "schema_version",
        "reference_mapping_available",
        "mapping",
        "mapping_sha256",
        "protocol",
        "protocol_sha256",
        "biopac_sample_rate_hz",
        "model_hz",
        "window_duration_s",
        "window_interval_count",
        "window_minimum_overlap_fraction",
        "transition_guard_s",
        "sync_authorized",
        "stage_metric_eligible",
        "joined_columns",
        "joined_rows_sha256",
    }
)
_V3_TARGET_DERIVED_METADATA_COLUMNS = frozenset(
    {
        "rr_bpm",
        "rr_spectral_bpm",
        "rr_phase_bpm",
        "rr_events_bpm",
        "reference_valid",
        "reference_quality",
        "reference_sigma_bpm",
        "spectral_concentration",
        "periodicity",
        "interval_cv",
        "estimator_disagreement_bpm",
        "phase_residual_rad",
        "clip_fraction",
        "guard_clip_fraction",
        "plateau_fraction",
        "breath_count",
        "classical_error_bpm",
        "radar_observable",
        "classical_acceptable_within_2bpm",
    }
)
_V3_METADATA_JOIN_COLUMNS = (
    "session_id",
    "identity",
    "window_number",
    "window_start_s",
    "window_end_s",
    "reference_start_sample",
    "reference_end_sample",
    "reference_window_start_biopac_s",
    "reference_window_end_biopac_s",
    "radar_window_start_relative_s",
    "radar_window_end_relative_s",
    "reference_mapping_available",
    "sync_authorized",
    "alignment_scientific_eligible",
    "acquisition_phase",
    "acquisition_phase_name",
    "acquisition_phase_status",
    "acquisition_phase_confidence",
    "phase_overlap_fraction",
    "transition_window",
    "eligible_for_stage_metrics",
    "phase7_assignment",
    "acquisition_batch",
)
_V3_REFERENCE_FLOAT_NAN_COLUMNS = tuple(
    sorted(
        (_V3_TARGET_DERIVED_METADATA_COLUMNS - {
            "reference_valid",
            "radar_observable",
            "classical_acceptable_within_2bpm",
        })
        | {
            "window_start_s",
            "window_end_s",
            "reference_window_start_biopac_s",
            "reference_window_end_biopac_s",
            "sync_confidence",
            "acquisition_phase_confidence",
            "phase_overlap_fraction",
        }
    )
)
_V3_REFERENCE_INTEGER_MINUS_ONE_COLUMNS = (
    "reference_start_sample",
    "reference_end_sample",
)
_V3_REFERENCE_NULL_STRING_COLUMNS = (
    "acquisition_phase",
    "acquisition_phase_name",
    "acquisition_phase_status",
    "phase7_assignment",
)
_V3_REFERENCE_FALSE_BOOLEAN_COLUMNS = (
    "reference_mapping_available",
    "reference_valid",
    "radar_observable",
    "classical_acceptable_within_2bpm",
    "sync_authorized",
    "alignment_scientific_eligible",
    "transition_window",
    "eligible_for_stage_metrics",
)


@dataclass(slots=True)
class _LoadedV3Session:
    maps: np.ndarray
    aux: np.ndarray
    metadata: pd.DataFrame
    frequencies_hz: np.ndarray
    timing_valid_mask: np.ndarray
    timing_reason_mask: np.ndarray
    map_view_availability_mask: np.ndarray
    aux_feature_availability_mask: np.ndarray
    aux_feature_names: tuple[str, ...]
    consumed_inventory: list[dict[str, Any]]
    inventory_sha256: str
    upstream_session_content_sha256: str
    scientific_eligible: bool
    authority_claims_complete: bool


def _require_exact_keys(
    document: Any,
    expected: frozenset[str],
    *,
    label: str,
) -> dict[str, Any]:
    if not isinstance(document, dict):
        raise ValueError(f"{label} must be a JSON object")
    observed = set(document)
    if observed != set(expected):
        raise ValueError(
            f"{label} keys do not match its schema; "
            f"missing={sorted(set(expected) - observed)}, "
            f"extra={sorted(observed - set(expected))}"
        )
    return document


def _strict_nonnegative_int(value: Any, *, location: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise ValueError(f"{location} must be an exact non-negative integer")
    result = int(value)
    if result < 0:
        raise ValueError(f"{location} must be an exact non-negative integer")
    return result


def _validate_v3_file_binding(
    raw_binding: Any,
    snapshot: _ConsumedFileSnapshot,
    *,
    logical_name: str,
    expected_filename: str,
    expected_dtype: str,
    session_id: str,
) -> tuple[tuple[int, ...], str]:
    binding = _require_exact_keys(
        raw_binding,
        _V3_FILE_BINDING_KEYS,
        label=f"version-3 file binding {session_id}/{logical_name}",
    )
    if binding.get("path") != expected_filename:
        raise ValueError(
            f"version-3 file binding path does not bind loader target: "
            f"{session_id}/{logical_name}"
        )
    declared_sha = _validate_sha256(
        binding.get("sha256"),
        location=f"version-3 file binding {session_id}/{logical_name} SHA-256",
    )
    declared_bytes = _strict_nonnegative_int(
        binding.get("bytes"),
        location=f"version-3 file binding {session_id}/{logical_name} bytes",
    )
    if declared_sha != snapshot.sha256 or declared_bytes != snapshot.byte_count:
        raise ValueError(
            f"version-3 consumed-byte binding mismatch: {session_id}/{logical_name}"
        )
    shape_value = binding.get("shape")
    if not isinstance(shape_value, list):
        raise ValueError(f"version-3 file shape is invalid: {session_id}/{logical_name}")
    shape = tuple(
        _strict_nonnegative_int(
            dimension,
            location=f"version-3 {session_id}/{logical_name} shape dimension",
        )
        for dimension in shape_value
    )
    dtype = binding.get("dtype")
    if dtype != expected_dtype:
        raise ValueError(f"version-3 file dtype is invalid: {session_id}/{logical_name}")
    return shape, str(dtype)


def _load_owned_npy_snapshot(
    snapshot: _ConsumedFileSnapshot,
    *,
    label: str,
) -> np.ndarray:
    snapshot.stream.seek(0)
    try:
        loaded = np.load(snapshot.stream, allow_pickle=False)
    except (OSError, ValueError, EOFError) as error:
        raise ValueError(f"cannot parse {label} from consumed bytes") from error
    if not isinstance(loaded, np.ndarray) or loaded.dtype.hasobject:
        raise ValueError(f"{label} must contain one non-object NPY array")
    return np.array(loaded, copy=True, order="C", subok=False)


def _readonly_owned(array: np.ndarray) -> np.ndarray:
    owned = np.array(array, copy=True, order="C", subok=False)
    owned.setflags(write=False)
    return owned


_V3_AUX_SCALAR_NAMES = (
    "log_low_power",
    "log_delta_power",
    "log_delta_to_low_ratio",
    "log_median_range_variance",
    "log_q90_range_variance",
    "range_entropy",
    "q90_peak_probability",
    "q98_peak_probability",
)
_V3_AUX_CONSENSUS_NAMES = (
    "radar_peak_std_bpm",
    "radar_peak_range_bpm",
    "q90_correlation_radar_1_radar_2",
    "q90_correlation_radar_1_radar_3",
    "q90_correlation_radar_2_radar_3",
)


def _v3_aux_feature_names(frequency_bins: int) -> tuple[str, ...]:
    if isinstance(frequency_bins, bool) or not isinstance(
        frequency_bins, (int, np.integer)
    ) or int(frequency_bins) < 1:
        raise ValueError("version-3 auxiliary frequency count must be positive")
    count = int(frequency_bins)
    names: list[str] = []
    for radar_id in (1, 2, 3):
        for quantile in ("q90", "q98"):
            names.extend(
                f"radar_{radar_id}_{quantile}_frequency_{index:04d}"
                for index in range(count)
            )
    for radar_id in (1, 2, 3):
        names.extend(
            f"radar_{radar_id}_{name}" for name in _V3_AUX_SCALAR_NAMES
        )
    names.extend(f"fused_median_frequency_{index:04d}" for index in range(count))
    names.extend(f"fused_max_frequency_{index:04d}" for index in range(count))
    names.extend(_V3_AUX_CONSENSUS_NAMES)
    if len(names) != 8 * count + 29 or len(set(names)) != len(names):
        raise RuntimeError("version-3 auxiliary feature-name derivation failed")
    return tuple(names)


def _v3_expected_feature_availability(
    timing_valid: np.ndarray,
    *,
    auxiliary_frequency_bins: int,
) -> tuple[np.ndarray, np.ndarray, tuple[str, ...], tuple[str, ...]]:
    timing = np.asarray(timing_valid)
    if timing.dtype != np.bool_ or timing.ndim != 3 or timing.shape[1:] != (3, 320):
        raise ValueError("version-3 timing mask must have exact [window,3,320] support")
    map_available = np.all(timing, axis=2)
    frequency_bins = int(auxiliary_frequency_bins)
    aux_names = _v3_aux_feature_names(frequency_bins)
    aux_available = np.zeros((len(timing), len(aux_names)), dtype=np.bool_)
    cursor = 0
    for radar_index in range(3):
        width = 2 * frequency_bins
        aux_available[:, cursor : cursor + width] = map_available[
            :, radar_index, None
        ]
        cursor += width
    for radar_index in range(3):
        width = len(_V3_AUX_SCALAR_NAMES)
        aux_available[:, cursor : cursor + width] = map_available[
            :, radar_index, None
        ]
        cursor += width
    joint = np.all(map_available, axis=1)
    joint_width = 2 * frequency_bins + len(_V3_AUX_CONSENSUS_NAMES)
    aux_available[:, cursor : cursor + joint_width] = joint[:, None]
    cursor += joint_width
    if cursor != len(aux_names):
        raise RuntimeError("version-3 auxiliary availability layout drifted")
    availability_names = tuple(
        [f"map_radar_{radar_id}" for radar_id in (1, 2, 3)]
        + [f"aux:{name}" for name in aux_names]
    )
    return map_available, aux_available, aux_names, availability_names


def _v3_metadata_scalar(value: Any, *, label: str) -> Any:
    if value is None or pd.isna(value):
        return None
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        result = float(value)
        if not np.isfinite(result):
            raise ValueError(f"{label} contains a non-finite value")
        return result
    if isinstance(value, str):
        return value
    raise ValueError(f"{label} contains a non-canonical scalar")


def _v3_metadata_join_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    missing = sorted(set(_V3_METADATA_JOIN_COLUMNS) - set(frame.columns))
    if missing:
        raise ValueError(f"version-3 metadata join columns are missing: {missing}")
    records: list[dict[str, Any]] = []
    for row_index, row in frame.loc[:, _V3_METADATA_JOIN_COLUMNS].iterrows():
        records.append(
            {
                column: _v3_metadata_scalar(
                    row[column], label=f"version-3 metadata row {row_index}/{column}"
                )
                for column in _V3_METADATA_JOIN_COLUMNS
            }
        )
    return records


def _v3_reference_support_contract(reference_mapping_available: bool) -> dict[str, Any]:
    if type(reference_mapping_available) is not bool:
        raise ValueError("version-3 reference mapping availability must be boolean")
    return {
        "schema_version": V3_REFERENCE_SUPPORT_SCHEMA_VERSION,
        "reference_mapping_available": reference_mapping_available,
        "radar_support_columns": [
            "radar_window_start_relative_s",
            "radar_window_end_relative_s",
        ],
        "unavailable_float_sentinel": "nan",
        "unavailable_float_columns": list(_V3_REFERENCE_FLOAT_NAN_COLUMNS),
        "unavailable_integer_sentinel": -1,
        "unavailable_integer_columns": list(
            _V3_REFERENCE_INTEGER_MINUS_ONE_COLUMNS
        ),
        "unavailable_string_sentinel": None,
        "unavailable_string_columns": list(_V3_REFERENCE_NULL_STRING_COLUMNS),
        "unavailable_boolean_sentinel": False,
        "unavailable_boolean_columns": list(_V3_REFERENCE_FALSE_BOOLEAN_COLUMNS),
        "numeric_zero_never_implies_reference_availability": True,
    }


def _validate_v3_unmapped_reference_rows(frame: pd.DataFrame, *, session_id: str) -> None:
    missing = sorted(
        set(
            _V3_REFERENCE_FLOAT_NAN_COLUMNS
            + _V3_REFERENCE_INTEGER_MINUS_ONE_COLUMNS
            + _V3_REFERENCE_NULL_STRING_COLUMNS
            + _V3_REFERENCE_FALSE_BOOLEAN_COLUMNS
            + (
                "radar_window_start_relative_s",
                "radar_window_end_relative_s",
            )
        )
        - set(frame.columns)
    )
    if missing:
        raise ValueError(
            f"{session_id} unmapped reference sentinel columns are missing: {missing}"
        )
    for column in _V3_REFERENCE_FLOAT_NAN_COLUMNS:
        values = frame[column]
        if not pd.api.types.is_float_dtype(values.dtype) or not values.isna().all():
            raise ValueError(
                f"{session_id} unmapped reference column {column} must be all NaN"
            )
    for column in _V3_REFERENCE_INTEGER_MINUS_ONE_COLUMNS:
        values = frame[column]
        if (
            not pd.api.types.is_integer_dtype(values.dtype)
            or not np.array_equal(values.to_numpy(), np.full(len(frame), -1))
        ):
            raise ValueError(
                f"{session_id} unmapped reference column {column} must be exact -1"
            )
    for column in _V3_REFERENCE_NULL_STRING_COLUMNS:
        values = frame[column]
        if not values.isna().all():
            raise ValueError(
                f"{session_id} unmapped reference column {column} must be null"
            )
    for column in _V3_REFERENCE_FALSE_BOOLEAN_COLUMNS:
        values = {
            _explicit_boolean(
                value,
                location=f"{session_id} unmapped reference column {column}",
            )
            for value in frame[column].unique()
        }
        if values != {False}:
            raise ValueError(
                f"{session_id} unmapped reference column {column} must be false"
            )
    try:
        radar_start = pd.to_numeric(
            frame["radar_window_start_relative_s"], errors="raise"
        ).to_numpy(dtype=float)
        radar_end = pd.to_numeric(
            frame["radar_window_end_relative_s"], errors="raise"
        ).to_numpy(dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"{session_id} unmapped measured radar support is invalid"
        ) from error
    if not (
        np.isfinite(radar_start).all()
        and np.isfinite(radar_end).all()
        and np.allclose(radar_end - radar_start, 32.0, rtol=0.0, atol=1e-12)
        and (len(frame) < 2 or np.all(np.diff(radar_start) > 0))
        and (len(frame) < 2 or np.all(np.diff(radar_end) > 0))
        and len(np.unique(np.column_stack((radar_start, radar_end)), axis=0))
        == len(frame)
    ):
        raise ValueError(f"{session_id} unmapped measured radar support is invalid")


def _validate_v3_upstream_reference_evidence(
    session_manifest: dict[str, Any],
    *,
    session_id: str,
) -> tuple[bool, dict[str, Any] | None, dict[str, Any]]:
    upstream_session = session_manifest.get("upstream_session_contract")
    upstream_receipt = session_manifest.get("upstream_sync_receipt")
    if not isinstance(upstream_session, dict) or not isinstance(upstream_receipt, dict):
        raise ValueError(f"version-3 upstream reference evidence is absent: {session_id}")
    upstream_content = str(
        _validate_sha256(
            upstream_session.get("content_sha256"),
            location=f"version-3 upstream session content SHA-256 {session_id}",
        )
    )
    declared_upstream_content = str(
        _validate_sha256(
            session_manifest.get("upstream_session_content_sha256"),
            location=f"version-3 session upstream content SHA-256 {session_id}",
        )
    )
    synchronization = upstream_session.get("synchronization")
    receipt_result = upstream_receipt.get("result")
    if not isinstance(synchronization, dict) or not isinstance(receipt_result, dict):
        raise ValueError(f"version-3 upstream synchronization evidence is malformed: {session_id}")
    receipt_content = str(
        _validate_sha256(
            upstream_receipt.get("content_sha256"),
            location=f"version-3 upstream sync receipt content SHA-256 {session_id}",
        )
    )
    mapping = synchronization.get("mapping")
    if mapping is not None and not isinstance(mapping, dict):
        raise ValueError(f"version-3 upstream mapping is malformed: {session_id}")
    if (
        upstream_session.get("session_id") != session_id
        or upstream_receipt.get("session_id") != session_id
        or upstream_content != declared_upstream_content
        or _canonical_content_sha256(upstream_session) != upstream_content
        or _canonical_content_sha256(upstream_receipt) != receipt_content
        or synchronization.get("receipt_content_sha256") != receipt_content
        or receipt_result.get("mapping") != mapping
    ):
        raise ValueError(f"version-3 upstream reference evidence mismatch: {session_id}")
    available = isinstance(mapping, dict)
    if _explicit_boolean(
        session_manifest.get("reference_mapping_available"),
        location=f"version-3 reference mapping availability {session_id}",
    ) != available:
        raise ValueError(f"version-3 cached/upstream mapping availability mismatch: {session_id}")
    protocol = upstream_session.get("protocol")
    if not isinstance(protocol, dict):
        raise ValueError(f"version-3 upstream protocol is malformed: {session_id}")
    return available, mapping, protocol


@dataclass(frozen=True, slots=True)
class _V3StageJoinAuthority:
    session_id: str
    protocol: dict[str, Any]
    stage_metric_eligible: bool
    window_minimum_overlap_fraction: float
    transition_guard_s: float


def _optional_metadata_string(value: Any) -> str | None:
    if value is None or pd.isna(value) or str(value).strip() == "":
        return None
    return str(value)


def _validate_v3_metadata_join_contract(
    frame: pd.DataFrame,
    *,
    session_id: str,
    raw_contract: Any,
    declared_sha256: Any,
    expected_reference_mapping_available: bool,
    expected_mapping: dict[str, Any] | None,
    expected_protocol: dict[str, Any],
) -> None:
    contract = _require_exact_keys(
        raw_contract,
        _V3_METADATA_JOIN_KEYS,
        label=f"version-3 metadata join contract {session_id}",
    )
    declared_hash = _validate_sha256(
        declared_sha256,
        location=f"version-3 metadata join SHA-256 {session_id}",
    )
    if _canonical_sha256(contract) != declared_hash:
        raise ValueError(f"version-3 metadata join content mismatch: {session_id}")
    if contract.get("schema_version") != V3_METADATA_JOIN_SCHEMA_VERSION:
        raise ValueError(f"version-3 metadata join schema mismatch: {session_id}")
    mapping_document = contract.get("mapping")
    protocol = contract.get("protocol")
    reference_mapping_available = _explicit_boolean(
        contract.get("reference_mapping_available"),
        location=f"version-3 metadata mapping availability {session_id}",
    )
    if (
        reference_mapping_available != expected_reference_mapping_available
        or mapping_document != expected_mapping
        or protocol != expected_protocol
        or (mapping_document is not None and not isinstance(mapping_document, dict))
        or not isinstance(protocol, dict)
    ):
        raise ValueError(f"version-3 metadata join authority is malformed: {session_id}")
    if _canonical_sha256(mapping_document) != _validate_sha256(
        contract.get("mapping_sha256"),
        location=f"version-3 metadata mapping SHA-256 {session_id}",
    ) or _canonical_sha256(protocol) != _validate_sha256(
        contract.get("protocol_sha256"),
        location=f"version-3 metadata protocol SHA-256 {session_id}",
    ):
        raise ValueError(f"version-3 metadata upstream authority hash mismatch: {session_id}")
    mapping: TimeMapping | None = None
    if reference_mapping_available:
        assert isinstance(mapping_document, dict)
        try:
            mapping = TimeMapping.from_dict(mapping_document)
        except (TypeError, ValueError) as error:
            raise ValueError(f"version-3 metadata mapping is invalid: {session_id}") from error
    if (
        contract.get("biopac_sample_rate_hz")
        != (250.0 if reference_mapping_available else None)
        or contract.get("model_hz") != 10.0
        or contract.get("window_duration_s") != 32.0
        or contract.get("window_interval_count") != 320
        or contract.get("sync_authorized") is not False
        or contract.get("stage_metric_eligible") is not False
        or contract.get("joined_columns") != list(_V3_METADATA_JOIN_COLUMNS)
    ):
        raise ValueError(f"version-3 metadata support contract mismatch: {session_id}")
    minimum_overlap = contract.get("window_minimum_overlap_fraction")
    transition_guard = contract.get("transition_guard_s")
    if (
        isinstance(minimum_overlap, bool)
        or not isinstance(minimum_overlap, (int, float))
        or not np.isfinite(float(minimum_overlap))
        or not 0.0 <= float(minimum_overlap) <= 1.0
        or isinstance(transition_guard, bool)
        or not isinstance(transition_guard, (int, float))
        or not np.isfinite(float(transition_guard))
        or float(transition_guard) < 0.0
    ):
        raise ValueError(f"version-3 metadata stage policy is invalid: {session_id}")
    records = _v3_metadata_join_records(frame)
    if _canonical_sha256(records) != _validate_sha256(
        contract.get("joined_rows_sha256"),
        location=f"version-3 metadata joined rows SHA-256 {session_id}",
    ):
        raise ValueError(f"version-3 metadata joined-row hash mismatch: {session_id}")
    row_mapping_values = {
        _explicit_boolean(
            value,
            location=f"version-3 row mapping availability {session_id}",
        )
        for value in frame["reference_mapping_available"].unique()
    }
    if row_mapping_values != {reference_mapping_available}:
        raise ValueError(f"version-3 row/contract mapping availability mismatch: {session_id}")
    if not reference_mapping_available:
        _validate_v3_unmapped_reference_rows(frame, session_id=session_id)
        return

    authority = _V3StageJoinAuthority(
        session_id=session_id,
        protocol=protocol,
        stage_metric_eligible=False,
        window_minimum_overlap_fraction=float(minimum_overlap),
        transition_guard_s=float(transition_guard),
    )
    for row_index, row in frame.iterrows():
        assert mapping is not None
        radar_start = float(row["radar_window_start_relative_s"])
        radar_end = float(row["radar_window_end_relative_s"])
        reference_start = float(row["reference_window_start_biopac_s"])
        reference_end = float(row["reference_window_end_biopac_s"])
        if not (
            np.isclose(radar_end - radar_start, 32.0, rtol=0.0, atol=1e-12)
            and np.isclose(
                float(mapping.radar_to_rsp(radar_start)),
                reference_start,
                rtol=0.0,
                atol=1e-9,
            )
            and np.isclose(
                float(mapping.radar_to_rsp(radar_end)),
                reference_end,
                rtol=0.0,
                atol=1e-9,
            )
            and np.isclose(
                float(row["window_start_s"]),
                reference_start,
                rtol=0.0,
                atol=1e-9,
            )
            and np.isclose(
                float(row["window_end_s"]),
                reference_end,
                rtol=0.0,
                atol=1e-9,
            )
            and int(row["reference_start_sample"])
            == _half_open_sample_index(reference_start, 250.0)
            and int(row["reference_end_sample"])
            == _half_open_sample_index(reference_end, 250.0)
        ):
            raise ValueError(
                f"version-3 metadata mapping/sample support mismatch: {session_id}/{row_index}"
            )
        assignment = assign_stage_window(authority, reference_start, reference_end)  # type: ignore[arg-type]
        if (
            _optional_metadata_string(row["acquisition_phase"]) != assignment.stage_id
            or _optional_metadata_string(row["acquisition_phase_name"])
            != assignment.stage_name
            or _optional_metadata_string(row["acquisition_phase_status"])
            != assignment.stage_status
            or _optional_metadata_string(row["phase7_assignment"])
            != assignment.phase7_assignment
            or _explicit_boolean(
                row["transition_window"],
                location=f"version-3 transition_window {session_id}/{row_index}",
            )
            != assignment.transition_window
            or _explicit_boolean(
                row["eligible_for_stage_metrics"],
                location=f"version-3 stage eligibility {session_id}/{row_index}",
            )
            is not False
            or not np.isclose(
                float(row["phase_overlap_fraction"]),
                assignment.overlap_fraction,
                rtol=0.0,
                atol=1e-12,
            )
        ):
            raise ValueError(
                f"version-3 metadata stage annotation mismatch: {session_id}/{row_index}"
            )
        observed_confidence = row["acquisition_phase_confidence"]
        if assignment.stage_confidence is None:
            confidence_matches = pd.isna(observed_confidence)
        else:
            confidence_matches = bool(
                np.isclose(
                    float(observed_confidence),
                    assignment.stage_confidence,
                    rtol=0.0,
                    atol=1e-12,
                )
            )
        if not confidence_matches:
            raise ValueError(
                f"version-3 metadata stage confidence mismatch: {session_id}/{row_index}"
            )


def _validate_v3_feature_and_firewall_contracts(
    *,
    session_id: str,
    session_manifest: dict[str, Any],
    maps: np.ndarray,
    aux: np.ndarray,
    frequencies: np.ndarray,
    timing_valid: np.ndarray,
    feature_availability: np.ndarray,
    metadata: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
    feature_schema = _require_exact_keys(
        session_manifest.get("feature_schema"),
        _V3_FEATURE_SCHEMA_KEYS,
        label=f"version-3 feature schema {session_id}",
    )
    if feature_schema.get("schema_version") != V3_INFERENCE_FEATURE_SCHEMA_VERSION:
        raise ValueError(f"version-3 inference feature schema mismatch: {session_id}")
    if _canonical_sha256(feature_schema) != _validate_sha256(
        session_manifest.get("feature_schema_sha256"),
        location=f"version-3 feature schema SHA-256 {session_id}",
    ):
        raise ValueError(f"version-3 inference feature schema hash mismatch: {session_id}")
    maps_schema = _require_exact_keys(
        feature_schema.get("maps"),
        _V3_MAP_SCHEMA_KEYS,
        label=f"version-3 maps schema {session_id}",
    )
    aux_schema = _require_exact_keys(
        feature_schema.get("aux"),
        _V3_AUX_SCHEMA_KEYS,
        label=f"version-3 aux schema {session_id}",
    )
    availability_schema = _require_exact_keys(
        feature_schema.get("availability"),
        _V3_AVAILABILITY_SCHEMA_KEYS,
        label=f"version-3 availability schema {session_id}",
    )
    if (
        maps.ndim != 4
        or maps.shape[1] != 3
        or maps.shape[3] != 182
        or maps_schema.get("axes")
        != ["window", "radar_view", "frequency", "range_feature"]
        or maps_schema.get("radar_names") != ["radar_1", "radar_2", "radar_3"]
        or maps_schema.get("range_feature_names")
        != list(V3_MAP_RANGE_FEATURE_NAMES)
        or maps_schema.get("shape") != list(maps.shape)
        or maps_schema.get("dtype") != "float16"
        or maps_schema.get("source_lineage")
        != V3_MAP_SOURCE_LINEAGE
        or maps_schema.get("target_derived_inputs") is not False
        or maps_schema.get("frequency_grid_sha256")
        != canonical_ndarray_sha256(frequencies)
    ):
        raise ValueError(f"version-3 maps feature schema mismatch: {session_id}")
    if aux.ndim != 2 or aux.shape[1] < 37 or (aux.shape[1] - 29) % 8:
        raise ValueError(f"version-3 auxiliary layout is invalid: {session_id}")
    auxiliary_frequency_bins = (aux.shape[1] - 29) // 8
    if auxiliary_frequency_bins not in {
        2 * maps.shape[2],
        2 * maps.shape[2] + 1,
    }:
        raise ValueError(f"version-3 map/aux frequency geometry mismatch: {session_id}")
    (
        expected_map_available,
        expected_aux_available,
        expected_aux_names,
        expected_availability_names,
    ) = _v3_expected_feature_availability(
        timing_valid,
        auxiliary_frequency_bins=auxiliary_frequency_bins,
    )
    if (
        aux_schema.get("axes") != ["window", "feature"]
        or aux_schema.get("feature_names") != list(expected_aux_names)
        or aux_schema.get("feature_names_sha256")
        != _canonical_sha256(list(expected_aux_names))
        or aux_schema.get("shape") != list(aux.shape)
        or aux_schema.get("dtype") != "float32"
        or aux_schema.get("source_lineage")
        != "radar_only_causal_auxiliary_spectra_statistics"
        or aux_schema.get("target_derived_inputs") is not False
    ):
        raise ValueError(f"version-3 auxiliary feature schema mismatch: {session_id}")
    expected_availability = np.concatenate(
        (expected_map_available, expected_aux_available), axis=1
    )
    if (
        feature_availability.dtype != np.bool_
        or not np.array_equal(feature_availability, expected_availability)
        or availability_schema.get("schema_version")
        != V3_FEATURE_AVAILABILITY_SCHEMA_VERSION
        or availability_schema.get("axes") != ["window", "feature"]
        or availability_schema.get("feature_names")
        != list(expected_availability_names)
        or availability_schema.get("feature_names_sha256")
        != _canonical_sha256(list(expected_availability_names))
        or availability_schema.get("shape") != list(feature_availability.shape)
        or availability_schema.get("dtype") != "bool"
        or availability_schema.get("semantics")
        != (
            "first three cells authorize complete map radar views; remaining "
            "cells explicitly authorize aux columns; numeric zero never implies availability"
        )
    ):
        raise ValueError(f"version-3 feature availability contract mismatch: {session_id}")

    unavailable_maps = maps[~expected_map_available]
    unavailable_aux = aux[~expected_aux_available]
    if (
        np.any(unavailable_maps != 0)
        or np.any(np.signbit(unavailable_maps))
        or np.any(unavailable_aux != 0)
        or np.any(np.signbit(unavailable_aux))
    ):
        raise ValueError(f"version-3 unavailable inference cells are not exact +0.0: {session_id}")
    if not np.isfinite(maps).all() or not np.isfinite(aux).all():
        raise ValueError(f"version-3 inference features contain non-finite values: {session_id}")

    firewall = _require_exact_keys(
        session_manifest.get("target_firewall"),
        _V3_TARGET_FIREWALL_KEYS,
        label=f"version-3 target firewall {session_id}",
    )
    if _canonical_sha256(firewall) != _validate_sha256(
        session_manifest.get("target_firewall_sha256"),
        location=f"version-3 target firewall SHA-256 {session_id}",
    ):
        raise ValueError(f"version-3 target firewall hash mismatch: {session_id}")
    forbidden = sorted(
        _V3_TARGET_DERIVED_METADATA_COLUMNS
        | V3_REQUIRED_ACQUISITION_ANNOTATION_COLUMNS
    )
    inference_names = set(expected_aux_names) | set(expected_availability_names)
    if (
        firewall.get("schema_version") != V3_TARGET_FIREWALL_SCHEMA_VERSION
        or firewall.get("inference_payloads")
        != ["maps", "aux", "feature_availability_mask", "frequencies_hz"]
        or firewall.get("target_derived_metadata_columns")
        != sorted(_V3_TARGET_DERIVED_METADATA_COLUMNS)
        or firewall.get("annotation_only_columns")
        != sorted(V3_REQUIRED_ACQUISITION_ANNOTATION_COLUMNS)
        or firewall.get("forbidden_inference_feature_names") != forbidden
        or firewall.get("radar_observable_role")
        != "target_derived_metadata_only_forbidden_at_inference"
        or firewall.get("target_values_used_in_inference_features") is not False
        or inference_names.intersection(forbidden)
        or not _V3_TARGET_DERIVED_METADATA_COLUMNS <= set(metadata.columns)
    ):
        raise ValueError(f"version-3 target firewall contract mismatch: {session_id}")
    reference_valid = np.asarray(
        [
            _explicit_boolean(
                value,
                location=f"version-3 reference_valid {session_id}",
            )
            for value in metadata["reference_valid"].to_numpy()
        ],
        dtype=np.bool_,
    )
    observable = np.asarray(
        [
            _explicit_boolean(
                value,
                location=f"version-3 radar_observable {session_id}",
            )
            for value in metadata["radar_observable"].to_numpy()
        ],
        dtype=np.bool_,
    )
    classical_acceptable = np.asarray(
        [
            _explicit_boolean(
                value,
                location=f"version-3 classical acceptable {session_id}",
            )
            for value in metadata["classical_acceptable_within_2bpm"].to_numpy()
        ],
        dtype=np.bool_,
    )
    rr = pd.to_numeric(metadata["rr_bpm"], errors="coerce").to_numpy(dtype=float)
    classical = pd.to_numeric(
        metadata["classical_rr_bpm"], errors="coerce"
    ).to_numpy(dtype=float)
    expected_observable = reference_valid & np.isfinite(rr) & np.isfinite(classical) & (
        np.abs(classical - rr) <= 2.0
    )
    if (
        reference_valid.any()
        or not np.array_equal(observable, expected_observable)
        or not np.array_equal(classical_acceptable, expected_observable)
    ):
        raise ValueError(f"version-3 diagnostic target firewall row mismatch: {session_id}")
    return expected_map_available, expected_aux_available, expected_aux_names


def _validate_v3_metadata_frame(
    frame: pd.DataFrame,
    *,
    session_id: str,
    session_manifest: dict[str, Any],
    session_scientific_eligible: bool,
    sync_authorized: bool,
) -> None:
    reference_mapping_available = _explicit_boolean(
        session_manifest.get("reference_mapping_available"),
        location=f"version-3 reference mapping availability {session_id}",
    )
    if reference_mapping_available:
        _validate_acquisition_metadata(
            frame,
            session_id=session_id,
            session_scientific_eligible=session_scientific_eligible,
            synchronization_authorized=sync_authorized,
        )
    else:
        missing_annotations = sorted(
            V3_REQUIRED_ACQUISITION_ANNOTATION_COLUMNS - set(frame.columns)
        )
        if missing_annotations:
            raise ValueError(
                f"acquisition metadata columns missing in {session_id}: "
                f"{missing_annotations}"
            )
        if frame.empty or set(frame["session_id"].dropna().astype(str)) != {session_id}:
            raise ValueError(f"version-3 unmapped metadata/session mismatch: {session_id}")
        if session_scientific_eligible or sync_authorized:
            raise ValueError(
                f"version-3 unmapped session cannot claim scientific/sync authority: {session_id}"
            )
        _validate_v3_unmapped_reference_rows(frame, session_id=session_id)
        batch = frame["acquisition_batch"]
        if batch.isna().any() or not batch.astype(str).str.strip().ne("").all():
            raise ValueError(
                f"version-3 unmapped acquisition_batch is blank: {session_id}"
            )
    required = {"identity", "window_number"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"version-3 metadata bound columns missing in {session_id}: {missing}")
    try:
        expected_identity = identity_for_session(session_id)
    except ValueError as error:
        raise ValueError(f"unknown canonical session identity: {session_id}") from error
    identities = frame["identity"]
    if identities.isna().any() or set(identities.astype(str)) != {expected_identity}:
        raise ValueError(f"canonical physical identity mismatch in {session_id}")
    raw_windows = pd.to_numeric(frame["window_number"], errors="coerce").to_numpy(
        dtype=float
    )
    if not (
        np.isfinite(raw_windows).all()
        and np.equal(raw_windows, np.rint(raw_windows)).all()
        and np.array_equal(
            raw_windows.astype(np.int64), np.arange(len(frame), dtype=np.int64)
        )
    ):
        raise ValueError(
            f"window_number must be unique, ordered, and consecutive in {session_id}"
        )
    window_count = _strict_nonnegative_int(
        session_manifest.get("window_count"),
        location=f"version-3 session {session_id} window_count",
    )
    if window_count != len(frame):
        raise ValueError(f"version-3 session window_count mismatch: {session_id}")


def _v3_upstream_reconstruction_is_independently_verified(
    *,
    reconstruction: dict[str, Any],
    reconstruction_manifest_sha256: str,
    reconstruction_content_sha256: str,
    session_content_bindings: dict[str, str],
) -> bool:
    """Terminal integration hook for a separately governed V3 verifier.

    Local flags and self-hashes cannot establish raw replay authority.  A
    future externally governed source generation must replace this terminal
    false hook with signature/trust-root verification over these exact
    consumed bindings.  Tests may monkeypatch it only to exercise containment
    after all structural and byte-level gates pass.
    """

    del (
        reconstruction,
        reconstruction_manifest_sha256,
        reconstruction_content_sha256,
        session_content_bindings,
    )
    return False


def _load_v3_session_from_pinned_directory(
    root_fd: int,
    root_item: dict[str, Any],
    *,
    root_config_sha256: str,
    root_pipeline_sha256: str,
) -> _LoadedV3Session:
    session_id = _safe_leaf_name(
        root_item.get("session_id"), label="version-3 session_id"
    )
    session_fd, directory_fingerprint = _open_child_directory_nofollow(
        root_fd,
        session_id,
        label=f"version-3 session root {session_id}",
    )
    entry_fingerprints: dict[str, tuple[int, ...]] = {}
    consumed_inventory: list[dict[str, Any]] = []
    try:
        session_manifest, manifest_snapshot = _snapshot_json_at(
            session_fd,
            "manifest.json",
            label=f"version-3 session manifest {session_id}",
        )
        try:
            entry_fingerprints["manifest.json"] = manifest_snapshot.source_stat
            consumed_inventory.append(
                _consumed_inventory_record(
                    f"{session_id}/manifest.json", manifest_snapshot
                )
            )
            declared_manifest_sha = _validate_sha256(
                root_item.get("manifest_sha256"),
                location=f"version-3 root item {session_id} manifest_sha256",
            )
            if declared_manifest_sha != manifest_snapshot.sha256:
                raise ValueError(
                    f"version-3 consumed session manifest SHA-256 mismatch: {session_id}"
                )
        finally:
            manifest_snapshot.close()

        _require_exact_keys(
            session_manifest,
            _V3_SESSION_MANIFEST_KEYS,
            label=f"version-3 session manifest {session_id}",
        )
        if session_manifest.get("schema_version") != (
            ACQUISITION_CACHE_SESSION_SCHEMA_VERSION_V3
        ) or root_item.get("schema_version") != (
            ACQUISITION_CACHE_SESSION_SCHEMA_VERSION_V3
        ):
            raise ValueError(f"version-3 session schema mismatch: {session_id}")
        if session_manifest.get("session_id") != session_id:
            raise ValueError(f"version-3 session ID mismatch: {session_id}")
        if root_item.get("manifest_path") != f"{session_id}/manifest.json":
            raise ValueError(f"version-3 session manifest path is not contained: {session_id}")
        declared_content = _validate_sha256(
            session_manifest.get("content_sha256"),
            location=f"version-3 session {session_id} content_sha256",
        )
        root_content = _validate_sha256(
            root_item.get("manifest_content_sha256"),
            location=f"version-3 root item {session_id} manifest content SHA-256",
        )
        if (
            declared_content != root_content
            or _canonical_content_sha256(session_manifest) != declared_content
        ):
            raise ValueError(f"version-3 session canonical content mismatch: {session_id}")
        if session_manifest.get("config_sha256") != root_config_sha256:
            raise ValueError(f"version-3 root/session config SHA-256 mismatch: {session_id}")
        if session_manifest.get("pipeline_sha256") != root_pipeline_sha256:
            raise ValueError(f"version-3 root/session pipeline SHA-256 mismatch: {session_id}")

        session_eligible = _explicit_boolean(
            session_manifest.get("scientific_eligible"),
            location=f"version-3 session {session_id} scientific_eligible",
        )
        if _explicit_boolean(
            root_item.get("scientific_eligible"),
            location=f"version-3 root item {session_id} scientific_eligible",
        ) != session_eligible:
            raise ValueError(f"version-3 root/session eligibility mismatch: {session_id}")
        claim_names = (
            "raw_consumed_bytes_verified",
            "timing_adjudicated",
            "sync_raw_replay_verified",
            "protocol_raw_replay_verified",
            "sync_authorized",
        )
        authority_claims = {
            name: _explicit_boolean(
                session_manifest.get(name),
                location=f"version-3 session {session_id} {name}",
            )
            for name in claim_names
        }
        authority_claims_complete = all(authority_claims.values())
        if session_eligible and not authority_claims_complete:
            raise ValueError(
                f"version-3 eligible session lacks upstream authority: {session_id}"
            )
        upstream_session_hash = str(
            _validate_sha256(
                session_manifest.get("upstream_session_content_sha256"),
                location=f"version-3 session {session_id} upstream content SHA-256",
            )
        )
        if root_item.get("upstream_session_content_sha256") != upstream_session_hash:
            raise ValueError(
                f"version-3 root/session upstream hash mismatch: {session_id}"
            )
        (
            reference_mapping_available,
            upstream_mapping,
            upstream_protocol,
        ) = _validate_v3_upstream_reference_evidence(
            session_manifest,
            session_id=session_id,
        )
        reference_support_contract = session_manifest.get(
            "reference_support_contract"
        )
        expected_reference_support = _v3_reference_support_contract(
            reference_mapping_available
        )
        if (
            reference_support_contract != expected_reference_support
            or _canonical_sha256(reference_support_contract)
            != _validate_sha256(
                session_manifest.get("reference_support_sha256"),
                location=(
                    f"version-3 reference support SHA-256 {session_id}"
                ),
            )
        ):
            raise ValueError(
                f"version-3 reference support contract mismatch: {session_id}"
            )
        if session_manifest.get("radar_timing_invalid_reason_schema_version") != (
            CAUSAL_UNIFORM_INVALID_REASON_SCHEMA_V1
        ):
            raise ValueError(f"version-3 timing reason schema mismatch: {session_id}")
        if session_manifest.get(
            "radar_timing_invalid_reason_semantics_sha256"
        ) != CAUSAL_UNIFORM_INVALID_REASON_SEMANTICS_SHA256_V1:
            raise ValueError(f"version-3 timing reason semantics mismatch: {session_id}")

        inventory = session_manifest.get("file_inventory")
        if not isinstance(inventory, dict) or set(inventory) != set(_V3_REQUIRED_FILES):
            raise ValueError(f"version-3 file inventory is incomplete: {session_id}")
        inventory_sha = str(
            _validate_sha256(
                session_manifest.get("inventory_sha256"),
                location=f"version-3 session {session_id} inventory_sha256",
            )
        )
        if _canonical_sha256(inventory) != inventory_sha:
            raise ValueError(f"version-3 inventory canonical hash mismatch: {session_id}")
        if root_item.get("inventory_sha256") != inventory_sha:
            raise ValueError(f"version-3 root/session inventory hash mismatch: {session_id}")

        arrays: dict[str, np.ndarray] = {}
        frame: pd.DataFrame | None = None
        for logical_name, (filename, expected_dtype) in _V3_REQUIRED_FILES.items():
            snapshot = _snapshot_regular_file_at(
                session_fd,
                filename,
                label=f"version-3 payload {session_id}/{logical_name}",
            )
            try:
                entry_fingerprints[filename] = snapshot.source_stat
                consumed_inventory.append(
                    _consumed_inventory_record(f"{session_id}/{filename}", snapshot)
                )
                declared_shape, _ = _validate_v3_file_binding(
                    inventory.get(logical_name),
                    snapshot,
                    logical_name=logical_name,
                    expected_filename=filename,
                    expected_dtype=expected_dtype,
                    session_id=session_id,
                )
                if logical_name == "metadata":
                    snapshot.stream.seek(0)
                    try:
                        observed_frame = pd.read_csv(
                            snapshot.stream,
                            keep_default_na=False,
                            na_values=[""],
                        )
                    except (OSError, ValueError, UnicodeDecodeError) as error:
                        raise ValueError(
                            f"cannot parse version-3 metadata from consumed bytes: {session_id}"
                        ) from error
                    if tuple(observed_frame.shape) != declared_shape:
                        raise ValueError(
                            f"version-3 consumed metadata shape mismatch: {session_id}"
                        )
                    frame = observed_frame.copy(deep=True)
                else:
                    observed_array = _load_owned_npy_snapshot(
                        snapshot,
                        label=f"version-3 payload {session_id}/{logical_name}",
                    )
                    if tuple(observed_array.shape) != declared_shape:
                        raise ValueError(
                            f"version-3 consumed array shape mismatch: "
                            f"{session_id}/{logical_name}"
                        )
                    if str(observed_array.dtype) != expected_dtype:
                        raise ValueError(
                            f"version-3 consumed array dtype mismatch: "
                            f"{session_id}/{logical_name}"
                        )
                    arrays[logical_name] = observed_array
            finally:
                snapshot.close()

        if frame is None or set(arrays) != set(_V3_REQUIRED_FILES) - {"metadata"}:
            raise ValueError(f"version-3 session payload set is incomplete: {session_id}")
        maps = arrays["maps"]
        aux = arrays["aux"]
        frequencies = arrays["frequencies_hz"]
        timing_valid = arrays["radar_timing_valid_mask"]
        timing_reasons = arrays["radar_timing_invalid_reason_mask"]
        feature_availability = arrays["feature_availability_mask"]
        if (
            maps.ndim != 4
            or maps.shape[1] != 3
            or aux.ndim != 2
            or frequencies.ndim != 1
            or timing_valid.ndim != 3
            or timing_valid.shape[1:] != (3, 320)
            or timing_reasons.shape != timing_valid.shape
            or feature_availability.ndim != 2
            or len(maps) != len(aux)
            or len(maps) != len(frame)
            or len(maps) != len(timing_valid)
            or len(maps) != len(feature_availability)
            or maps.shape[2] != len(frequencies)
        ):
            raise ValueError(f"version-3 cache payload shape relation failed: {session_id}")
        if not (
            np.isfinite(maps).all()
            and np.isfinite(aux).all()
            and np.isfinite(frequencies).all()
            and (np.diff(frequencies) > 0).all()
        ):
            raise ValueError(f"version-3 cache payload contains invalid numeric data: {session_id}")
        unknown_reason_bits = np.bitwise_and(timing_reasons, np.uint8(0xE0))
        if np.any(unknown_reason_bits != 0) or not np.array_equal(
            timing_reasons != 0, ~timing_valid
        ):
            raise ValueError(f"version-3 timing reason union mismatch: {session_id}")
        _validate_v3_metadata_frame(
            frame,
            session_id=session_id,
            session_manifest=session_manifest,
            session_scientific_eligible=session_eligible,
            sync_authorized=authority_claims["sync_authorized"],
        )
        _validate_sha256(
            session_manifest.get("source_fingerprint"),
            location=f"version-3 source fingerprint {session_id}",
        )
        _validate_v3_metadata_join_contract(
            frame,
            session_id=session_id,
            raw_contract=session_manifest.get("metadata_join_contract"),
            declared_sha256=session_manifest.get("metadata_join_sha256"),
            expected_reference_mapping_available=reference_mapping_available,
            expected_mapping=upstream_mapping,
            expected_protocol=upstream_protocol,
        )
        (
            map_view_available,
            aux_feature_available,
            aux_feature_names,
        ) = _validate_v3_feature_and_firewall_contracts(
            session_id=session_id,
            session_manifest=session_manifest,
            maps=maps,
            aux=aux,
            frequencies=frequencies,
            timing_valid=timing_valid,
            feature_availability=feature_availability,
            metadata=frame,
        )

        for filename, fingerprint in entry_fingerprints.items():
            _assert_regular_entry_still_bound(
                session_fd,
                filename,
                fingerprint,
                label=f"version-3 session entry {session_id}/{filename}",
            )
        _assert_child_directory_still_bound(
            root_fd,
            session_id,
            directory_fingerprint,
            label=f"version-3 session root {session_id}",
        )
        return _LoadedV3Session(
            maps=_readonly_owned(maps),
            aux=_readonly_owned(aux),
            metadata=frame,
            frequencies_hz=_readonly_owned(frequencies),
            timing_valid_mask=_readonly_owned(timing_valid),
            timing_reason_mask=_readonly_owned(timing_reasons),
            map_view_availability_mask=_readonly_owned(map_view_available),
            aux_feature_availability_mask=_readonly_owned(aux_feature_available),
            aux_feature_names=aux_feature_names,
            consumed_inventory=consumed_inventory,
            inventory_sha256=inventory_sha,
            upstream_session_content_sha256=upstream_session_hash,
            scientific_eligible=session_eligible,
            authority_claims_complete=authority_claims_complete,
        )
    finally:
        os.close(session_fd)


def _load_feature_cache_v3(
    root: Path,
    *,
    sessions: list[str] | None,
    mmap: bool,
    require_scientific_eligible: bool,
) -> FeatureCache:
    """Load V3 only from descriptor-pinned, exact consumed-byte snapshots."""

    # Reject before opening any session/payload descriptor.  V3 scientific
    # authority requires owned arrays detached from mutable cache inodes.
    if mmap:
        raise ValueError("version-3 acquisition caches forbid mmap=True")
    if sessions is not None and require_scientific_eligible:
        raise ValueError(
            "scientific cache loading forbids any sessions filter; load the "
            "verified full-cohort catalogue"
        )

    root_fd, root_absolute = _open_directory_path_nofollow(
        root, label="version-3 cache root"
    )
    root_directory_fingerprint = _stable_stat_fingerprint(os.fstat(root_fd))
    try:
        root_manifest, root_snapshot = _snapshot_json_at(
            root_fd,
            "manifest.json",
            label="version-3 feature cache root manifest",
        )
        try:
            root_manifest_sha256 = root_snapshot.sha256
            root_manifest_fingerprint = root_snapshot.source_stat
        finally:
            root_snapshot.close()
        _require_exact_keys(
            root_manifest,
            _V3_ROOT_MANIFEST_KEYS,
            label="version-3 feature cache root manifest",
        )
        if root_manifest.get("schema_version") != (
            ACQUISITION_CACHE_ROOT_SCHEMA_VERSION_V3
        ):
            raise ValueError("version-3 feature cache root schema mismatch")
        root_content_sha256 = str(
            _validate_sha256(
                root_manifest.get("content_sha256"),
                location="version-3 feature cache root content_sha256",
            )
        )
        if _canonical_content_sha256(root_manifest) != root_content_sha256:
            raise ValueError("version-3 feature cache root canonical content mismatch")
        root_config_sha256 = str(
            _validate_sha256(
                root_manifest.get("config_sha256"),
                location="version-3 feature cache config_sha256",
            )
        )
        root_pipeline_sha256 = str(
            _validate_sha256(
                root_manifest.get("pipeline_sha256"),
                location="version-3 feature cache pipeline_sha256",
            )
        )
        root_contract = _require_exact_keys(
            root_manifest.get("acquisition_contract"),
            _V3_ROOT_CONTRACT_KEYS,
            label="version-3 root acquisition contract",
        )
        if root_contract.get("schema_version") != ACQUISITION_CACHE_SCHEMA_VERSION_V3:
            raise ValueError("version-3 root acquisition schema mismatch")
        if root_contract.get("upstream_reconstruction_schema_version") != (
            ACQUISITION_RECONSTRUCTION_SCHEMA_VERSION_V3
        ):
            raise ValueError("version-3 upstream reconstruction schema binding mismatch")
        for key in (
            "reconstruction_content_sha256",
            "cohort_authority_sha256",
            "cohort_authority_content_sha256",
            "expected_usable_session_ids_sha256",
            "cache_usable_session_ids_sha256",
        ):
            _validate_sha256(
                root_contract.get(key),
                location=f"version-3 root acquisition contract {key}",
            )
        _annotation_columns(
            root_contract.get("annotation_only_columns"),
            location="version-3 root annotation_only_columns",
            expected_columns=V3_REQUIRED_ACQUISITION_ANNOTATION_COLUMNS,
        )
        mode = root_contract.get("mode")
        if mode not in {"strict", "diagnostic"}:
            raise ValueError("version-3 cache mode must be strict or diagnostic")
        root_eligible = _explicit_boolean(
            root_contract.get("scientific_eligible"),
            location="version-3 root scientific_eligible",
        )
        subjects_filter_applied = _explicit_boolean(
            root_contract.get("subjects_filter_applied"),
            location="version-3 root subjects_filter_applied",
        )
        full_cohort_complete = _explicit_boolean(
            root_contract.get("full_cohort_complete"),
            location="version-3 root full_cohort_complete",
        )
        selection_scope = root_contract.get("selection_scope")
        if selection_scope not in {"full_cohort", "diagnostic_subset"}:
            raise ValueError("version-3 cache selection_scope is invalid")
        if root_eligible and (
            mode != "strict"
            or subjects_filter_applied
            or not full_cohort_complete
            or selection_scope != "full_cohort"
        ):
            raise ValueError("version-3 scientific root has incomplete cohort claims")

        raw_items = root_manifest.get("sessions")
        if not isinstance(raw_items, list) or not raw_items:
            raise ValueError("version-3 root sessions must be a non-empty list")
        root_items: list[dict[str, Any]] = []
        for raw_item in raw_items:
            item = _require_exact_keys(
                raw_item,
                _V3_ROOT_SESSION_ITEM_KEYS,
                label="version-3 root session item",
            )
            session_id = _safe_leaf_name(
                item.get("session_id"), label="version-3 root session_id"
            )
            if item.get("status") != "ok":
                raise ValueError(f"version-3 cache session is not usable: {session_id}")
            if item.get("schema_version") != ACQUISITION_CACHE_SESSION_SCHEMA_VERSION_V3:
                raise ValueError(f"version-3 root session schema mismatch: {session_id}")
            for key in (
                "manifest_sha256",
                "manifest_content_sha256",
                "inventory_sha256",
                "upstream_session_content_sha256",
            ):
                _validate_sha256(
                    item.get(key),
                    location=f"version-3 root item {session_id} {key}",
                )
            if item.get("manifest_path") != f"{session_id}/manifest.json":
                raise ValueError(
                    f"version-3 root session manifest path is not contained: {session_id}"
                )
            item_eligible = _explicit_boolean(
                item.get("scientific_eligible"),
                location=f"version-3 root item {session_id} scientific_eligible",
            )
            if item_eligible != root_eligible:
                raise ValueError(f"version-3 root/session claim mismatch: {session_id}")
            root_items.append(item)
        available = [str(item["session_id"]) for item in root_items]
        if len(set(available)) != len(available):
            raise ValueError("version-3 root contains duplicate session IDs")

        expected_ids = _unique_string_list(
            root_contract.get("expected_usable_session_ids"),
            location="version-3 expected usable session IDs",
        )
        cache_ids = _unique_string_list(
            root_contract.get("cache_usable_session_ids"),
            location="version-3 cache usable session IDs",
        )
        for session_id in (*expected_ids, *cache_ids):
            _safe_leaf_name(session_id, label="version-3 declared session ID")
        if _session_ids_sha256(expected_ids) != root_contract.get(
            "expected_usable_session_ids_sha256"
        ):
            raise ValueError("version-3 expected usable session ID hash mismatch")
        if _session_ids_sha256(cache_ids) != root_contract.get(
            "cache_usable_session_ids_sha256"
        ):
            raise ValueError("version-3 cache usable session ID hash mismatch")
        if cache_ids != available:
            raise ValueError("version-3 cache catalogue/session ID mismatch")
        inventory_bindings = {
            str(item["session_id"]): str(item["inventory_sha256"])
            for item in root_items
        }
        declared_inventory_aggregate = _validate_sha256(
            root_contract.get("cache_inventory_aggregate_sha256"),
            location="version-3 cache inventory aggregate SHA-256",
        )
        if _canonical_sha256(inventory_bindings) != declared_inventory_aggregate:
            raise ValueError("version-3 cache inventory aggregate mismatch")

        selected = available if sessions is None else list(sessions)
        if not selected:
            raise ValueError("version-3 cache contains no selected sessions")
        if any(not isinstance(item, str) for item in selected):
            raise ValueError("requested sessions must be strings")
        if len(set(selected)) != len(selected):
            raise ValueError("requested sessions must not contain duplicates")
        missing = sorted(set(selected) - set(available))
        if missing:
            raise KeyError(f"sessions not present in cache: {missing}")

        reconstruction_leaf = _safe_leaf_name(
            root_contract.get("reconstruction_manifest"),
            label="version-3 reconstruction manifest",
        )
        reconstruction, reconstruction_snapshot = _snapshot_json_at(
            root_fd,
            reconstruction_leaf,
            label="version-3 upstream reconstruction manifest",
        )
        consumed_inventory = [
            _consumed_inventory_record(reconstruction_leaf, reconstruction_snapshot)
        ]
        reconstruction_fingerprint = reconstruction_snapshot.source_stat
        try:
            declared_reconstruction_file_sha = _validate_sha256(
                root_contract.get("reconstruction_manifest_sha256"),
                location="version-3 reconstruction manifest SHA-256",
            )
            declared_reconstruction_bytes = _strict_nonnegative_int(
                root_contract.get("reconstruction_manifest_bytes"),
                location="version-3 reconstruction manifest bytes",
            )
            if (
                reconstruction_snapshot.sha256 != declared_reconstruction_file_sha
                or reconstruction_snapshot.byte_count != declared_reconstruction_bytes
            ):
                raise ValueError("version-3 reconstruction consumed-byte mismatch")
        finally:
            reconstruction_snapshot.close()
        if reconstruction.get("schema_version") != (
            ACQUISITION_RECONSTRUCTION_SCHEMA_VERSION_V3
        ):
            raise ValueError("version-3 reconstruction schema mismatch")
        reconstruction_content_sha256 = str(
            _validate_sha256(
                reconstruction.get("content_sha256"),
                location="version-3 reconstruction content_sha256",
            )
        )
        if (
            reconstruction_content_sha256
            != root_contract.get("reconstruction_content_sha256")
            or _canonical_content_sha256(reconstruction)
            != reconstruction_content_sha256
        ):
            raise ValueError("version-3 reconstruction canonical content mismatch")

        reconstruction_expected_usable = _unique_string_list(
            reconstruction.get("expected_usable_session_ids"),
            location="version-3 reconstruction expected usable session IDs",
        )
        reconstruction_expected_all = _unique_string_list(
            reconstruction.get("expected_session_ids"),
            location="version-3 reconstruction expected session IDs",
        )
        reconstruction_selected = _unique_string_list(
            reconstruction.get("selected_session_ids"),
            location="version-3 reconstruction selected session IDs",
        )
        canonical_all_session_ids = list(SESSION_IDENTITY)
        canonical_usable_session_ids = [
            session_id
            for session_id in canonical_all_session_ids
            if session_id != "S24_KHJ"
        ]
        if (
            reconstruction_expected_usable != expected_ids
            or reconstruction_expected_all != canonical_all_session_ids
            or reconstruction_expected_usable != canonical_usable_session_ids
            or reconstruction_selected != reconstruction_expected_all
            or root_contract.get("cohort_authority_content_sha256")
            != ACQUISITION_COHORT_V1_CONTENT_SHA256
            or reconstruction.get("expected_usable_session_ids_sha256")
            != _session_ids_sha256(reconstruction_expected_usable)
            or reconstruction.get("expected_session_ids_sha256")
            != _session_ids_sha256(reconstruction_expected_all)
            or reconstruction.get("selected_session_ids_sha256")
            != _session_ids_sha256(reconstruction_selected)
        ):
            raise ValueError(
                "version-3 reconstruction must bind the exact full 30-session cohort"
            )
        reconstruction_claim_names = (
            "execution_complete",
            "complete",
            "full_cohort_complete",
            "scientific_eligible",
            "raw_consumed_bytes_verified",
            "timing_adjudicated",
            "sync_raw_replay_verified",
            "protocol_raw_replay_verified",
        )
        reconstruction_claims = {
            name: _explicit_boolean(
                reconstruction.get(name),
                location=f"version-3 reconstruction {name}",
            )
            for name in reconstruction_claim_names
        }
        usable_count = _strict_nonnegative_int(
            reconstruction.get("dataset_usable_session_count"),
            location="version-3 reconstruction dataset usable session count",
        )
        identity_count = _strict_nonnegative_int(
            reconstruction.get("dataset_physical_identity_count"),
            location="version-3 reconstruction physical identity count",
        )
        dataset_session_count = _strict_nonnegative_int(
            reconstruction.get("dataset_session_count"),
            location="version-3 reconstruction dataset session count",
        )
        selected_session_count = _strict_nonnegative_int(
            reconstruction.get("selected_session_count"),
            location="version-3 reconstruction selected session count",
        )
        session_count = _strict_nonnegative_int(
            reconstruction.get("session_count"),
            location="version-3 reconstruction session count",
        )
        raw_reconstruction_sessions = reconstruction.get("sessions")
        if not isinstance(raw_reconstruction_sessions, list):
            raise ValueError("version-3 reconstruction sessions must be a list")
        reconstruction_sessions: dict[str, dict[str, Any]] = {}
        upstream_required = {
            "session_id",
            "usable",
            "content_sha256",
            "scientific_eligible",
            "raw_consumed_bytes_verified",
            "timing_adjudicated",
            "sync_raw_replay_verified",
            "protocol_raw_replay_verified",
            "sync_authorized",
        }
        for raw_entry in raw_reconstruction_sessions:
            if not isinstance(raw_entry, dict) or not upstream_required <= set(raw_entry):
                raise ValueError("version-3 reconstruction session claim is incomplete")
            session_id = _safe_leaf_name(
                raw_entry.get("session_id"),
                label="version-3 reconstruction session_id",
            )
            if session_id in reconstruction_sessions:
                raise ValueError("version-3 reconstruction repeats a session ID")
            _validate_sha256(
                raw_entry.get("content_sha256"),
                location=f"version-3 reconstruction session {session_id} content SHA-256",
            )
            for name in upstream_required - {"session_id", "content_sha256"}:
                _explicit_boolean(
                    raw_entry.get(name),
                    location=f"version-3 reconstruction session {session_id} {name}",
                )
            reconstruction_sessions[session_id] = raw_entry
        if (
            list(reconstruction_sessions) != reconstruction_expected_all
            or dataset_session_count != 30
            or selected_session_count != 30
            or session_count != 30
        ):
            raise ValueError("version-3 reconstruction session catalogue mismatch")
        reconstruction_usable_ids = [
            session_id
            for session_id, entry in reconstruction_sessions.items()
            if _explicit_boolean(
                entry["usable"],
                location=f"version-3 reconstruction session {session_id} usable",
            )
        ]
        excluded_ids = [
            session_id
            for session_id in reconstruction_expected_all
            if session_id not in reconstruction_usable_ids
        ]
        if (
            reconstruction_usable_ids != reconstruction_expected_usable
            or reconstruction_usable_ids != cache_ids
            or excluded_ids != ["S24_KHJ"]
            or usable_count != 29
        ):
            raise ValueError(
                "version-3 cache must be the exact 29-session usable projection "
                "of the full reconstruction"
            )
        session_content_bindings = {
            session_id: str(reconstruction_sessions[session_id]["content_sha256"])
            for session_id in reconstruction_usable_ids
        }
        for item in root_items:
            session_id = str(item["session_id"])
            if item.get("upstream_session_content_sha256") != session_content_bindings.get(
                session_id
            ):
                raise ValueError(
                    f"version-3 cache/reconstruction session hash mismatch: {session_id}"
                )

        root_items_by_id = {str(item["session_id"]): item for item in root_items}
        loaded_sessions: list[_LoadedV3Session] = []
        for session_id in selected:
            loaded_session = _load_v3_session_from_pinned_directory(
                root_fd,
                root_items_by_id[session_id],
                root_config_sha256=root_config_sha256,
                root_pipeline_sha256=root_pipeline_sha256,
            )
            if loaded_session.upstream_session_content_sha256 != session_content_bindings[
                session_id
            ]:
                raise ValueError(
                    f"version-3 consumed cache/upstream session mismatch: {session_id}"
                )
            upstream_entry = reconstruction_sessions[session_id]
            upstream_claims_complete = all(
                _explicit_boolean(
                    upstream_entry[name],
                    location=f"version-3 reconstruction session {session_id} {name}",
                )
                for name in upstream_required
                - {"session_id", "content_sha256", "usable"}
            ) and _explicit_boolean(
                upstream_entry["usable"],
                location=f"version-3 reconstruction session {session_id} usable",
            )
            if loaded_session.scientific_eligible and not upstream_claims_complete:
                raise ValueError(
                    f"version-3 eligible cache session lacks upstream claims: {session_id}"
                )
            consumed_inventory.extend(loaded_session.consumed_inventory)
            loaded_sessions.append(loaded_session)

        if not loaded_sessions:
            raise ValueError("version-3 cache contains no selected sessions")
        frequency_grid = loaded_sessions[0].frequencies_hz
        aux_feature_names = loaded_sessions[0].aux_feature_names
        for loaded_session in loaded_sessions[1:]:
            if not np.array_equal(frequency_grid, loaded_session.frequencies_hz):
                raise ValueError("version-3 frequency grid must be exactly equal")
            if loaded_session.aux_feature_names != aux_feature_names:
                raise ValueError("version-3 auxiliary feature names must be exactly equal")

        scientific_cohort_shape = bool(
            len(expected_ids) == 29
            and len(cache_ids) == 29
            and len(reconstruction_expected_all) == 30
            and reconstruction_selected == reconstruction_expected_all
            and usable_count == 29
            and dataset_session_count == 30
            and identity_count == 18
            and len({identity_for_session(item) for item in cache_ids}) == 18
        )
        root_claims_complete = bool(
            root_eligible
            and mode == "strict"
            and not subjects_filter_applied
            and full_cohort_complete
            and selection_scope == "full_cohort"
            and expected_ids == cache_ids
            and reconstruction_claims["execution_complete"]
            and reconstruction_claims["complete"]
            and reconstruction_claims["full_cohort_complete"]
            and reconstruction_claims["scientific_eligible"]
            and reconstruction_claims["raw_consumed_bytes_verified"]
            and reconstruction_claims["timing_adjudicated"]
            and reconstruction_claims["sync_raw_replay_verified"]
            and reconstruction_claims["protocol_raw_replay_verified"]
            and scientific_cohort_shape
            and all(item.scientific_eligible for item in loaded_sessions)
            and all(item.authority_claims_complete for item in loaded_sessions)
            and sessions is None
        )
        independent_upstream_verified = bool(
            root_claims_complete
            and (
                _v3_upstream_reconstruction_is_independently_verified(
                    reconstruction=reconstruction,
                    reconstruction_manifest_sha256=str(
                        root_contract["reconstruction_manifest_sha256"]
                    ),
                    reconstruction_content_sha256=reconstruction_content_sha256,
                    session_content_bindings=session_content_bindings,
                )
                is True
            )
        )
        effective_eligible = bool(root_claims_complete and independent_upstream_verified)
        if require_scientific_eligible and not effective_eligible:
            raise ValueError(
                "version-3 cache lacks independently verified upstream scientific authority"
            )

        paths = [str(item["path"]) for item in consumed_inventory]
        if len(set(paths)) != len(paths):
            raise ValueError("version-3 consumed inventory repeats a path")
        canonical_consumed_inventory = sorted(
            consumed_inventory, key=lambda item: str(item["path"])
        )
        inventory_sha256 = _canonical_sha256(canonical_consumed_inventory)

        def concatenate_readonly(
            values: list[np.ndarray],
        ) -> np.ndarray:
            combined = (
                np.array(values[0], copy=True, order="C")
                if len(values) == 1
                else np.concatenate(values, axis=0)
            )
            combined.setflags(write=False)
            return combined

        maps = concatenate_readonly([item.maps for item in loaded_sessions])
        aux = concatenate_readonly([item.aux for item in loaded_sessions])
        timing_valid = concatenate_readonly(
            [item.timing_valid_mask for item in loaded_sessions]
        )
        timing_reasons = concatenate_readonly(
            [item.timing_reason_mask for item in loaded_sessions]
        )
        map_view_available = concatenate_readonly(
            [item.map_view_availability_mask for item in loaded_sessions]
        )
        aux_feature_available = concatenate_readonly(
            [item.aux_feature_availability_mask for item in loaded_sessions]
        )
        feature_available = concatenate_readonly(
            [
                np.concatenate(
                    (
                        item.map_view_availability_mask,
                        item.aux_feature_availability_mask,
                    ),
                    axis=1,
                )
                for item in loaded_sessions
            ]
        )
        feature_availability_names = tuple(
            [f"map_radar_{radar_id}" for radar_id in (1, 2, 3)]
            + [f"aux:{name}" for name in aux_feature_names]
        )
        frequencies = _readonly_owned(frequency_grid)
        metadata = pd.concat(
            [item.metadata for item in loaded_sessions], ignore_index=True
        ).copy(deep=True)

        _assert_regular_entry_still_bound(
            root_fd,
            "manifest.json",
            root_manifest_fingerprint,
            label="version-3 root manifest",
        )
        _assert_regular_entry_still_bound(
            root_fd,
            reconstruction_leaf,
            reconstruction_fingerprint,
            label="version-3 reconstruction manifest",
        )
        observed_root = os.stat(root_absolute, follow_symlinks=False)
        if (
            not stat.S_ISDIR(observed_root.st_mode)
            or _stable_stat_fingerprint(observed_root)
            != root_directory_fingerprint
        ):
            raise ValueError("version-3 cache root changed during load")

        provenance = CacheProvenance(
            classification=(
                "acquisition_scientific"
                if effective_eligible
                else "acquisition_diagnostic"
            ),
            root_manifest_path=os.path.join(root_absolute, "manifest.json"),
            root_manifest_sha256=root_manifest_sha256,
            root_manifest_content_sha256=root_content_sha256,
            acquisition_schema_version=ACQUISITION_CACHE_SCHEMA_VERSION_V3,
            acquisition_mode=str(mode),
            scientific_eligible=effective_eligible,
            config_sha256=root_config_sha256,
            pipeline_sha256=root_pipeline_sha256,
            reconstruction_content_sha256=reconstruction_content_sha256,
            inventory_sha256=inventory_sha256,
            inventory_file_count=len(canonical_consumed_inventory),
            selected_sessions=tuple(selected),
        )
        return FeatureCache(
            maps=maps,
            aux=aux,
            metadata=metadata,
            frequencies_hz=frequencies,
            provenance=provenance,
            radar_timing_valid_mask=timing_valid,
            radar_timing_invalid_reason_mask=timing_reasons,
            feature_availability_mask=feature_available,
            feature_availability_names=feature_availability_names,
            map_view_availability_mask=map_view_available,
            aux_feature_availability_mask=aux_feature_available,
            aux_feature_names=aux_feature_names,
        )
    finally:
        os.close(root_fd)


def _validate_v2_session_inventory(
    session_dir: Path,
    session_manifest: dict[str, Any],
    *,
    session_id: str,
    require_all_timing_valid: bool,
) -> None:
    inventory = session_manifest.get("file_inventory")
    if not isinstance(inventory, dict):
        raise ValueError(f"version-2 file_inventory is missing: {session_id}")
    required = {
        "maps",
        "aux",
        "metadata",
        "frequencies_hz",
        "radar_timing_valid_mask",
    }
    allowed = required | {"range_aux"}
    if set(inventory) < required or set(inventory) - allowed:
        raise ValueError(
            f"version-2 file_inventory keys are invalid in {session_id}: "
            f"missing={sorted(required - set(inventory))}, "
            f"extra={sorted(set(inventory) - allowed)}"
        )
    expected_inventory_hash = _validate_sha256(
        session_manifest.get("inventory_sha256"),
        location=f"session {session_id} inventory_sha256",
    )
    if _canonical_sha256(inventory) != expected_inventory_hash:
        raise ValueError(f"file_inventory canonical hash mismatch: {session_id}")

    expected_paths = {
        "maps": "maps.npy",
        "aux": "aux.npy",
        "metadata": "metadata.csv",
        "frequencies_hz": "frequencies_hz.npy",
        "radar_timing_valid_mask": "radar_timing_valid_mask.npy",
        "range_aux": "range_aux.npy",
    }
    observed_shapes: dict[str, list[int]] = {}
    observed_dtypes: dict[str, str] = {}
    for name, raw_binding in inventory.items():
        if not isinstance(raw_binding, dict):
            raise ValueError(f"file_inventory entry {name} is invalid: {session_id}")
        relative = raw_binding.get("path")
        if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
            raise ValueError(
                f"file_inventory path {name} must be relative: {session_id}"
            )
        if relative != expected_paths[name]:
            raise ValueError(
                f"file_inventory path {name} does not bind the loader target: "
                f"{session_id}"
            )
        path = (session_dir / relative).resolve()
        try:
            path.relative_to(session_dir.resolve())
        except ValueError as error:
            raise ValueError(
                f"file_inventory path {name} escapes session root: {session_id}"
            ) from error
        if not path.is_file():
            raise FileNotFoundError(
                f"file_inventory file {name} is missing in {session_id}: {path}"
            )
        declared_bytes = raw_binding.get("bytes")
        if (
            isinstance(declared_bytes, bool)
            or not isinstance(declared_bytes, int)
            or declared_bytes < 0
            or path.stat().st_size != declared_bytes
        ):
            raise ValueError(f"file_inventory byte count mismatch: {session_id}/{name}")
        declared_sha = _validate_sha256(
            raw_binding.get("sha256"),
            location=f"session {session_id} file_inventory {name} sha256",
        )
        if _sha256_file(path) != declared_sha:
            raise ValueError(f"file_inventory SHA-256 mismatch: {session_id}/{name}")
        shape = raw_binding.get("shape")
        if (
            not isinstance(shape, list)
            or any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in shape)
        ):
            raise ValueError(f"file_inventory shape is invalid: {session_id}/{name}")
        dtype = raw_binding.get("dtype")
        if not isinstance(dtype, str) or not dtype:
            raise ValueError(f"file_inventory dtype is invalid: {session_id}/{name}")
        if name == "metadata":
            if dtype != "csv":
                raise ValueError(f"metadata inventory dtype must be csv: {session_id}")
            observed = pd.read_csv(path)
            observed_shape = list(observed.shape)
            if observed_shape != shape:
                raise ValueError(f"metadata inventory shape mismatch: {session_id}")
        else:
            observed_array = np.load(path, mmap_mode="r", allow_pickle=False)
            if list(observed_array.shape) != shape:
                raise ValueError(
                    f"file_inventory array shape mismatch: {session_id}/{name}"
                )
            if str(observed_array.dtype) != dtype:
                raise ValueError(
                    f"file_inventory array dtype mismatch: {session_id}/{name}"
                )
        observed_shapes[name] = list(shape)
        observed_dtypes[name] = str(dtype)

    row_counts = {
        observed_shapes[name][0]
        for name in ("maps", "aux", "metadata", "radar_timing_valid_mask")
        if observed_shapes[name]
    }
    if len(row_counts) != 1:
        raise ValueError(f"version-2 inventory row-axis mismatch: {session_id}")
    maps_shape = observed_shapes["maps"]
    timing_shape = observed_shapes["radar_timing_valid_mask"]
    if len(maps_shape) < 2 or maps_shape[1] != 3:
        raise ValueError(f"version-2 maps radar axis must contain three views: {session_id}")
    if len(timing_shape) != 3 or timing_shape[1] != 3 or timing_shape[2] <= 0:
        raise ValueError(
            f"radar_timing_valid_mask shape must be [N,3,window_samples]: {session_id}"
        )
    if observed_dtypes["radar_timing_valid_mask"] != "bool":
        raise ValueError(f"radar_timing_valid_mask dtype must be bool: {session_id}")
    measured_support = session_manifest.get("measured_window_support")
    if not isinstance(measured_support, dict):
        raise ValueError(
            f"version-2 measured_window_support is missing: {session_id}"
        )
    interval_count = measured_support.get("window_interval_count")
    duration = measured_support.get("window_duration_s")
    expected_reference_indexing = {
        "sample_timestamp_semantics": "i / sample_rate_hz",
        "support_membership": "start_s <= i / sample_rate_hz < end_s",
        "slice_boundary_rule": "ceil_both_boundaries",
        "near_integer_canonicalization": (
            "abs(coordinate-rint(coordinate)) <= "
            "max(1e-9,8*spacing(max(abs(coordinate),1)))"
        ),
    }
    if (
        type(interval_count) is not int
        or interval_count <= 0
        or isinstance(duration, bool)
        or not isinstance(duration, (int, float))
        or not np.isfinite(float(duration))
        or float(duration) != 32.0
        or measured_support.get("timestamp_semantics") != "right_edge_exclusive"
        or measured_support.get("reference_sample_indexing")
        != expected_reference_indexing
        or timing_shape[2] != interval_count
    ):
        raise ValueError(
            f"version-2 measured_window_support must bind exact 32s support: {session_id}"
        )
    timing_mask = np.load(
        session_dir / "radar_timing_valid_mask.npy",
        mmap_mode="r",
        allow_pickle=False,
    )
    if session_manifest.get("radar_timing_valid_mask_shape") != timing_shape:
        raise ValueError(
            f"radar_timing_valid_mask_shape declaration mismatch: {session_id}"
        )
    declared_invalid_count = session_manifest.get(
        "radar_timing_invalid_interval_count"
    )
    observed_invalid_count = int(
        np.size(timing_mask) - np.count_nonzero(timing_mask)
    )
    if (
        type(declared_invalid_count) is not int
        or declared_invalid_count != observed_invalid_count
    ):
        raise ValueError(
            f"radar_timing_invalid_interval_count mismatch: {session_id}"
        )
    if require_all_timing_valid and not bool(np.asarray(timing_mask).all()):
        raise ValueError(
            f"scientific acquisition cache contains invalid radar timing: {session_id}"
        )


def _half_open_sample_index(time_s: float, sample_rate_hz: float) -> int:
    """Mirror the producer's stable half-open BIOPAC boundary conversion."""

    coordinate = float(time_s) * float(sample_rate_hz)
    if not np.isfinite(coordinate):
        raise ValueError("BIOPAC time/sample-rate coordinate must be finite")
    nearest = float(np.rint(coordinate))
    tolerance_samples = max(
        1.0e-9,
        8.0 * abs(float(np.spacing(max(abs(coordinate), 1.0)))),
    )
    canonical_coordinate = (
        nearest
        if abs(coordinate - nearest) <= tolerance_samples
        else coordinate
    )
    index = int(math.ceil(canonical_coordinate))
    if not index - 1 < canonical_coordinate <= index:
        raise ValueError("BIOPAC half-open sample-index proof failed")
    return index


def _validate_acquisition_metadata(
    frame: pd.DataFrame,
    *,
    session_id: str,
    session_scientific_eligible: bool,
    synchronization_authorized: bool,
    source_contract: AcquisitionSessionContract | None = None,
    cache_session_manifest: dict[str, Any] | None = None,
) -> None:
    missing = sorted(REQUIRED_ACQUISITION_ANNOTATION_COLUMNS - set(frame.columns))
    if missing:
        raise ValueError(
            f"acquisition metadata columns missing in {session_id}: {missing}"
        )
    if "session_id" not in frame:
        raise ValueError(f"acquisition metadata in {session_id} is missing session_id")
    observed_sessions = set(frame["session_id"].dropna().astype(str))
    if observed_sessions != {session_id}:
        raise ValueError(
            f"acquisition metadata/session manifest mismatch in {session_id}: "
            f"observed {sorted(observed_sessions)}"
        )

    if frame.empty:
        raise ValueError(f"acquisition metadata contains no rows in {session_id}")

    if source_contract is not None:
        if cache_session_manifest is None:
            raise ValueError(
                f"version-2 cache session manifest is required in {session_id}"
            )
        required_bound_columns = {
            "identity",
            "window_number",
            "window_start_s",
            "window_end_s",
        }
        missing_bound = sorted(required_bound_columns - set(frame.columns))
        if missing_bound:
            raise ValueError(
                f"version-2 metadata bound columns missing in {session_id}: {missing_bound}"
            )
        try:
            canonical_identity = identity_for_session(session_id)
        except ValueError as error:
            raise ValueError(
                f"canonical physical-identity authority has no session {session_id}"
            ) from error
        identities = frame["identity"]
        if identities.isna().any() or set(identities.astype(str)) != {
            canonical_identity
        }:
            raise ValueError(f"canonical physical identity mismatch in {session_id}")

        raw_window_numbers = pd.to_numeric(
            frame["window_number"], errors="coerce"
        ).to_numpy(dtype=float)
        if (
            not np.isfinite(raw_window_numbers).all()
            or not np.equal(raw_window_numbers, np.rint(raw_window_numbers)).all()
            or not np.array_equal(
                raw_window_numbers.astype(np.int64),
                np.arange(len(frame), dtype=np.int64),
            )
        ):
            raise ValueError(
                f"window_number must be unique, ordered, and consecutive in {session_id}"
            )
        declared_window_count = cache_session_manifest.get("window_count")
        if type(declared_window_count) is not int or declared_window_count != len(frame):
            raise ValueError(f"session manifest window_count mismatch in {session_id}")

    for column in (
        "sync_authorized",
        "alignment_scientific_eligible",
        "transition_window",
        "eligible_for_stage_metrics",
    ):
        if frame[column].isna().any():
            raise ValueError(f"{session_id} metadata column {column} contains nulls")
        values = frame[column].unique()
        for value in values:
            _explicit_boolean(value, location=f"{session_id} metadata column {column}")

    alignment_values = {
        _explicit_boolean(
            value,
            location=f"{session_id} metadata column alignment_scientific_eligible",
        )
        for value in frame["alignment_scientific_eligible"].unique()
    }
    if alignment_values != {session_scientific_eligible}:
        raise ValueError(f"alignment scientific eligibility mismatch in {session_id}")
    sync_values = {
        _explicit_boolean(
            value, location=f"{session_id} metadata column sync_authorized"
        )
        for value in frame["sync_authorized"].unique()
    }
    if sync_values != {synchronization_authorized}:
        raise ValueError(f"synchronization authorization mismatch in {session_id}")
    if not session_scientific_eligible and any(
        _explicit_boolean(
            value,
            location=f"{session_id} metadata column eligible_for_stage_metrics",
        )
        for value in frame["eligible_for_stage_metrics"].unique()
    ):
        raise ValueError(
            f"ineligible session exposes rows as eligible for stage metrics: {session_id}"
        )

    start = pd.to_numeric(frame["reference_start_sample"], errors="coerce").to_numpy(
        dtype=float
    )
    end = pd.to_numeric(frame["reference_end_sample"], errors="coerce").to_numpy(
        dtype=float
    )
    if not (
        np.isfinite(start).all()
        and np.isfinite(end).all()
        and np.equal(start, np.rint(start)).all()
        and np.equal(end, np.rint(end)).all()
        and (start >= 0).all()
        and (end > start).all()
    ):
        raise ValueError(
            f"acquisition reference sample bounds are invalid in {session_id}"
        )

    overlap = pd.to_numeric(frame["phase_overlap_fraction"], errors="coerce").to_numpy(
        dtype=float
    )
    if not (np.isfinite(overlap).all() and (overlap >= 0).all() and (overlap <= 1).all()):
        raise ValueError(f"phase_overlap_fraction is invalid in {session_id}")

    confidence = pd.to_numeric(
        frame["acquisition_phase_confidence"], errors="coerce"
    ).to_numpy(dtype=float)
    sync_confidence = pd.to_numeric(frame["sync_confidence"], errors="coerce").to_numpy(
        dtype=float
    )
    if not (
        np.isfinite(sync_confidence).all()
        and (sync_confidence >= 0).all()
        and (sync_confidence <= 1).all()
    ):
        raise ValueError(f"acquisition confidence is invalid in {session_id}")

    phase = frame["acquisition_phase"]
    phase_missing = phase.isna() | phase.astype(str).str.strip().eq("")
    for column in ("acquisition_phase_name", "acquisition_phase_status"):
        values = frame[column]
        missing_values = values.isna() | values.astype(str).str.strip().eq("")
        if not np.array_equal(missing_values.to_numpy(), phase_missing.to_numpy()):
            raise ValueError(f"partial acquisition phase annotation in {session_id}")
    confidence_missing = ~np.isfinite(confidence)
    if not np.array_equal(confidence_missing, phase_missing.to_numpy()):
        raise ValueError(f"partial acquisition phase confidence in {session_id}")
    assigned_confidence = confidence[~confidence_missing]
    if not (
        (assigned_confidence >= 0).all() and (assigned_confidence <= 1).all()
    ):
        raise ValueError(f"acquisition phase confidence is invalid in {session_id}")
    if frame.loc[phase_missing, "eligible_for_stage_metrics"].astype(bool).any():
        raise ValueError(f"unassigned acquisition phase is metric-eligible in {session_id}")

    reference_start_s = pd.to_numeric(
        frame["reference_window_start_biopac_s"], errors="coerce"
    ).to_numpy(dtype=float)
    reference_end_s = pd.to_numeric(
        frame["reference_window_end_biopac_s"], errors="coerce"
    ).to_numpy(dtype=float)
    radar_start_s = pd.to_numeric(
        frame["radar_window_start_relative_s"], errors="coerce"
    ).to_numpy(dtype=float)
    radar_end_s = pd.to_numeric(
        frame["radar_window_end_relative_s"], errors="coerce"
    ).to_numpy(dtype=float)
    if not (
        np.isfinite(reference_start_s).all()
        and np.isfinite(reference_end_s).all()
        and np.isfinite(radar_start_s).all()
        and np.isfinite(radar_end_s).all()
        and (reference_end_s > reference_start_s).all()
        and (radar_end_s > radar_start_s).all()
    ):
        raise ValueError(f"acquisition window timing is invalid in {session_id}")

    if source_contract is None:
        batch = frame["acquisition_batch"]
        if batch.isna().any() or not batch.astype(str).str.strip().ne("").all():
            raise ValueError(
                f"acquisition metadata column acquisition_batch is blank in {session_id}"
            )
        return

    assert cache_session_manifest is not None

    measured_support = cache_session_manifest.get("measured_window_support")
    if not isinstance(measured_support, dict):
        raise ValueError(f"measured_window_support is missing in {session_id}")
    duration = measured_support.get("window_duration_s")
    if isinstance(duration, bool) or not isinstance(duration, (int, float)):
        raise ValueError(f"measured_window_support.window_duration_s is invalid in {session_id}")
    window_duration_s = float(duration)
    if not np.isfinite(window_duration_s) or window_duration_s != 32.0:
        raise ValueError(f"scientific metadata requires exact 32s windows in {session_id}")
    canonical_start_s = pd.to_numeric(
        frame["window_start_s"], errors="coerce"
    ).to_numpy(dtype=float)
    canonical_end_s = pd.to_numeric(
        frame["window_end_s"], errors="coerce"
    ).to_numpy(dtype=float)
    if source_contract.mapping is None:
        raise ValueError(f"bound synchronization mapping is absent in {session_id}")
    mapping_scale = float(source_contract.mapping.scale)
    if not (
        np.isfinite(canonical_start_s).all()
        and np.isfinite(canonical_end_s).all()
        and np.allclose(canonical_start_s, reference_start_s, rtol=0.0, atol=1e-9)
        and np.allclose(canonical_end_s, reference_end_s, rtol=0.0, atol=1e-9)
        and np.allclose(
            reference_end_s - reference_start_s,
            mapping_scale * window_duration_s,
            rtol=0.0,
            atol=1e-9,
        )
        and np.allclose(
            radar_end_s - radar_start_s,
            window_duration_s,
            rtol=0.0,
            atol=1e-9,
        )
    ):
        raise ValueError(f"metadata does not preserve exact 32s support in {session_id}")
    if len(frame) > 1 and not (
        np.all(np.diff(reference_start_s) > 0)
        and np.all(np.diff(reference_end_s) > 0)
        and np.all(np.diff(radar_start_s) > 0)
        and np.all(np.diff(radar_end_s) > 0)
    ):
        raise ValueError(f"metadata windows are not in chronological order in {session_id}")
    support_rows = np.column_stack(
        (reference_start_s, reference_end_s, radar_start_s, radar_end_s)
    )
    if len(np.unique(support_rows, axis=0)) != len(frame):
        raise ValueError(f"metadata contains duplicate window support in {session_id}")

    source_sensor = source_contract.manifest.get("sensor_summary")
    biopac = source_sensor.get("biopac") if isinstance(source_sensor, dict) else None
    if not isinstance(biopac, dict) or "sample_rate_hz" not in biopac:
        raise ValueError(
            f"producer field sensor_summary.biopac.sample_rate_hz is required: {session_id}"
        )
    sample_rate_value = biopac.get("sample_rate_hz")
    if isinstance(sample_rate_value, bool):
        raise ValueError(f"BIOPAC sample_rate_hz is invalid in {session_id}")
    try:
        sample_rate_hz = float(sample_rate_value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"BIOPAC sample_rate_hz is invalid in {session_id}") from error
    if not np.isfinite(sample_rate_hz) or sample_rate_hz != 250.0:
        raise ValueError(f"BIOPAC sample_rate_hz must be 250 Hz in {session_id}")
    expected_start_samples = np.asarray(
        [
            _half_open_sample_index(value, sample_rate_hz)
            for value in reference_start_s
        ],
        dtype=np.int64,
    )
    expected_end_samples = np.asarray(
        [
            _half_open_sample_index(value, sample_rate_hz)
            for value in reference_end_s
        ],
        dtype=np.int64,
    )
    if not (
        np.array_equal(start.astype(np.int64), expected_start_samples)
        and np.array_equal(end.astype(np.int64), expected_end_samples)
    ):
        raise ValueError(f"reference sample/time/sample-rate relation mismatch in {session_id}")

    mapped_start = np.asarray(
        source_contract.mapping.radar_to_rsp(radar_start_s), dtype=np.float64
    )
    mapped_end = np.asarray(
        source_contract.mapping.radar_to_rsp(radar_end_s), dtype=np.float64
    )
    if not (
        np.allclose(mapped_start, reference_start_s, rtol=0.0, atol=1e-9)
        and np.allclose(mapped_end, reference_end_s, rtol=0.0, atol=1e-9)
    ):
        raise ValueError(f"radar/reference synchronization mapping mismatch in {session_id}")

    receipt_result = source_contract.receipt.get("result")
    receipt_confidence = (
        receipt_result.get("confidence") if isinstance(receipt_result, dict) else None
    )
    try:
        expected_sync_confidence = float(receipt_confidence)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"bound synchronization confidence is invalid in {session_id}") from error
    if not np.allclose(
        sync_confidence,
        expected_sync_confidence,
        rtol=0.0,
        atol=1e-12,
    ):
        raise ValueError(f"synchronization confidence mismatch in {session_id}")

    def optional_text(value: Any) -> str | None:
        if pd.isna(value) or str(value).strip() == "":
            return None
        return str(value)

    transition_values = [
        _explicit_boolean(
            value,
            location=f"{session_id} metadata column transition_window",
        )
        for value in frame["transition_window"]
    ]
    metric_values = [
        _explicit_boolean(
            value,
            location=f"{session_id} metadata column eligible_for_stage_metrics",
        )
        for value in frame["eligible_for_stage_metrics"]
    ]
    for row_index in range(len(frame)):
        assignment = assign_stage_window(
            source_contract,
            float(reference_start_s[row_index]),
            float(reference_end_s[row_index]),
        )
        observed_confidence = confidence[row_index]
        expected_confidence = assignment.stage_confidence
        confidence_matches = (
            (expected_confidence is None and not np.isfinite(observed_confidence))
            or (
                expected_confidence is not None
                and np.isclose(
                    observed_confidence,
                    expected_confidence,
                    rtol=0.0,
                    atol=1e-12,
                )
            )
        )
        expected_metric_eligible = bool(
            session_scientific_eligible
            and source_contract.stage_metric_eligible
            and assignment.eligible_for_stage_metrics
        )
        if not (
            optional_text(frame.iloc[row_index]["acquisition_phase"])
            == assignment.stage_id
            and optional_text(frame.iloc[row_index]["acquisition_phase_name"])
            == assignment.stage_name
            and optional_text(frame.iloc[row_index]["acquisition_phase_status"])
            == assignment.stage_status
            and confidence_matches
            and np.isclose(
                overlap[row_index],
                assignment.overlap_fraction,
                rtol=0.0,
                atol=1e-12,
            )
            and transition_values[row_index] == assignment.transition_window
            and metric_values[row_index] == expected_metric_eligible
            and optional_text(frame.iloc[row_index]["phase7_assignment"])
            == assignment.phase7_assignment
        ):
            raise ValueError(
                f"metadata phase assignment does not match reconstructed protocol "
                f"in {session_id} row {row_index}"
            )

    batch = frame["acquisition_batch"]
    if batch.isna().any() or not batch.astype(str).str.strip().ne("").all():
        raise ValueError(f"acquisition metadata column acquisition_batch is blank in {session_id}")


def _validate_acquisition_cache_contract(
    root: Path,
    root_manifest: dict[str, Any],
    available_items: list[dict[str, Any]],
    *,
    require_scientific_eligible: bool,
) -> dict[str, Any]:
    """Fail closed on mixed, inconsistent, or unauthorized acquisition caches."""

    root_contract = root_manifest.get("acquisition_contract")
    if not isinstance(root_contract, dict) or not root_contract:
        raise ValueError("feature cache root acquisition_contract is missing or empty")
    schema_version = root_contract.get("schema_version")
    if schema_version not in SUPPORTED_ACQUISITION_CACHE_SCHEMA_VERSIONS:
        raise ValueError("feature cache root acquisition schema_version is unsupported")
    if schema_version == ACQUISITION_CACHE_SCHEMA_VERSION_V3:
        raise ValueError("version-3 acquisition caches require the pinned-byte loader")
    is_v2 = schema_version == ACQUISITION_CACHE_SCHEMA_VERSION_V2
    mode = root_contract.get("mode")
    if mode not in {"strict", "diagnostic"}:
        raise ValueError("feature cache acquisition mode must be strict or diagnostic")
    root_annotation_columns = _annotation_columns(
        root_contract.get("annotation_only_columns"),
        location="feature cache root annotation_only_columns",
    )
    root_eligible = _explicit_boolean(
        root_contract.get("scientific_eligible"),
        location="feature cache root acquisition scientific_eligible",
    )
    if root_eligible and mode != "strict":
        raise ValueError("a scientifically eligible acquisition cache must use strict mode")
    if require_scientific_eligible:
        raise ValueError(
            "scientific cache loading requires acquisition cache version 3; "
            "versions 1 and 2 are historical/diagnostic only"
        )

    root_config_sha256: str | None = None
    root_pipeline_sha256: str | None = None
    root_cohort_authority_sha256: str | None = None
    root_cohort_authority_content_sha256: str | None = None
    if is_v2:
        subjects_filter_applied = _explicit_boolean(
            root_contract.get("subjects_filter_applied"),
            location="feature cache root subjects_filter_applied",
        )
        root_filter_claim = _explicit_boolean(
            root_manifest.get("subjects_filter_applied"),
            location="feature cache manifest subjects_filter_applied",
        )
        if root_filter_claim != subjects_filter_applied:
            raise ValueError(
                "feature cache subjects_filter_applied root/contract mismatch"
            )
        selection_scope = root_contract.get("selection_scope")
        if selection_scope not in {"full_cohort", "diagnostic_subset"}:
            raise ValueError("version-2 acquisition selection_scope is invalid")
        full_cohort_complete = _explicit_boolean(
            root_contract.get("full_cohort_complete"),
            location="feature cache root full_cohort_complete",
        )
        reconstruction_full_claim = _explicit_boolean(
            root_contract.get("reconstruction_full_cohort_complete"),
            location="feature cache root reconstruction_full_cohort_complete",
        )
        expected_ids = _unique_string_list(
            root_contract.get("expected_usable_session_ids"),
            location="feature cache expected_usable_session_ids",
        )
        cache_ids = _unique_string_list(
            root_contract.get("cache_usable_session_ids"),
            location="feature cache cache_usable_session_ids",
        )
        expected_ids_hash = _validate_sha256(
            root_contract.get("expected_usable_session_ids_sha256"),
            location="feature cache expected_usable_session_ids_sha256",
        )
        cache_ids_hash = _validate_sha256(
            root_contract.get("cache_usable_session_ids_sha256"),
            location="feature cache cache_usable_session_ids_sha256",
        )
        if expected_ids_hash != _session_ids_sha256(expected_ids):
            raise ValueError("feature cache expected usable-session ID hash mismatch")
        if cache_ids_hash != _session_ids_sha256(cache_ids):
            raise ValueError("feature cache usable-session ID hash mismatch")
        observed_ids = [str(item.get("session_id", "")) for item in available_items]
        if cache_ids != observed_ids:
            raise ValueError("feature cache usable-session catalogue order mismatch")
        expected_scope = (
            "full_cohort"
            if not subjects_filter_applied and cache_ids == expected_ids
            else "diagnostic_subset"
        )
        if selection_scope != expected_scope:
            raise ValueError("feature cache selection_scope does not match its IDs")
        expected_full_cohort_complete = bool(
            not subjects_filter_applied
            and reconstruction_full_claim
            and cache_ids == expected_ids
        )
        if full_cohort_complete != expected_full_cohort_complete:
            raise ValueError(
                "feature cache full_cohort_complete does not match its coverage"
            )
        inventory_bindings: dict[str, str] = {}
        for item in available_items:
            session_id = str(item.get("session_id", ""))
            inventory_bindings[session_id] = str(
                _validate_sha256(
                    item.get("inventory_sha256"),
                    location=f"root catalogue {session_id} inventory_sha256",
                )
            )
        declared_inventory_aggregate = _validate_sha256(
            root_contract.get("cache_inventory_aggregate_sha256"),
            location="feature cache cache_inventory_aggregate_sha256",
        )
        if _canonical_sha256(inventory_bindings) != declared_inventory_aggregate:
            raise ValueError("feature cache inventory aggregate SHA-256 mismatch")
        if root_eligible and (
            selection_scope != "full_cohort"
            or subjects_filter_applied
            or not full_cohort_complete
            or not reconstruction_full_claim
            or expected_ids_hash != cache_ids_hash
        ):
            raise ValueError(
                "scientifically eligible version-2 cache lacks exact full-cohort coverage"
            )
        root_config_sha256 = _validate_sha256(
            root_manifest.get("config_sha256"),
            location="version-2 feature cache config_sha256",
        )
        root_pipeline_sha256 = _validate_sha256(
            root_manifest.get("pipeline_sha256"),
            location="version-2 feature cache pipeline_sha256",
        )
        root_cohort_authority_sha256 = _validate_sha256(
            root_contract.get("cohort_authority_sha256"),
            location="feature cache cohort_authority_sha256",
        )
        root_cohort_authority_content_sha256 = _validate_sha256(
            root_contract.get("cohort_authority_content_sha256"),
            location="feature cache cohort_authority_content_sha256",
        )

    reconstruction_hash = _validate_sha256(
        root_contract.get("reconstruction_content_sha256"),
        location="feature cache reconstruction_content_sha256",
    )
    reconstruction_value = root_contract.get("reconstruction_manifest")
    if not isinstance(reconstruction_value, str) or not reconstruction_value.strip():
        raise ValueError("feature cache reconstruction_manifest is missing")
    reconstruction_path = Path(reconstruction_value)
    if not reconstruction_path.is_absolute():
        reconstruction_path = root / reconstruction_path
    if not reconstruction_path.is_file():
        raise FileNotFoundError(
            f"bound acquisition reconstruction manifest missing: {reconstruction_path}"
        )
    reconstruction = _read_strict_json(
        reconstruction_path, "acquisition reconstruction manifest"
    )
    if is_v2 and (
        reconstruction.get("schema_version")
        != ACQUISITION_RECONSTRUCTION_SCHEMA_VERSION_V2
    ):
        raise ValueError(
            "version-2 feature cache requires a version-2 acquisition reconstruction"
        )
    if reconstruction.get("content_sha256") != reconstruction_hash:
        raise ValueError("acquisition reconstruction/root contract hash mismatch")
    if _canonical_content_sha256(reconstruction) != reconstruction_hash:
        raise ValueError("acquisition reconstruction canonical content hash mismatch")
    verified_reconstruction = None
    if is_v2:
        # V2 is the only cache schema that may authorize a new scientific run.
        # Reuse the authoritative reconstruction consumer so that the cache
        # gate verifies the complete receipt/session/context graph rather than
        # trusting a self-consistent cache-side summary of that graph.
        try:
            verified_reconstruction = load_acquisition_reconstruction(
                reconstruction_path
            )
        except (ValueError, OSError) as error:
            raise ValueError(
                "version-2 acquisition reconstruction graph validation failed: "
                f"{error}"
            ) from error
        if verified_reconstruction.content_sha256 != reconstruction_hash:
            raise ValueError(
                "verified acquisition reconstruction/cache content hash mismatch"
            )
    if is_v2:
        assert root_cohort_authority_sha256 is not None
        assert root_cohort_authority_content_sha256 is not None
        reconstruction_cohort_authority_sha256 = _validate_sha256(
            reconstruction.get("cohort_authority_sha256"),
            location="acquisition reconstruction cohort_authority_sha256",
        )
        reconstruction_cohort_authority_content_sha256 = _validate_sha256(
            reconstruction.get("cohort_authority_content_sha256"),
            location="acquisition reconstruction cohort_authority_content_sha256",
        )
        if (
            reconstruction_cohort_authority_sha256
            != root_cohort_authority_sha256
            or reconstruction_cohort_authority_content_sha256
            != root_cohort_authority_content_sha256
        ):
            raise ValueError(
                "feature cache/reconstruction cohort authority binding mismatch"
            )
        reconstruction_execution_complete = _explicit_boolean(
            reconstruction.get("execution_complete"),
            location="acquisition reconstruction execution_complete",
        )
        if not reconstruction_execution_complete:
            raise ValueError("acquisition reconstruction selected execution is incomplete")
    elif reconstruction.get("complete") is not True:
        raise ValueError("acquisition reconstruction is not complete")
    reconstruction_eligible = _explicit_boolean(
        reconstruction.get("scientific_eligible"),
        location="acquisition reconstruction scientific_eligible",
    )
    if root_eligible and not reconstruction_eligible:
        raise ValueError("acquisition reconstruction/cache scientific eligibility mismatch")
    if is_v2:
        reconstruction_scope = reconstruction.get("selection_scope")
        reconstruction_full = _explicit_boolean(
            reconstruction.get("full_cohort_complete"),
            location="acquisition reconstruction full_cohort_complete",
        )
        reconstruction_complete = _explicit_boolean(
            reconstruction.get("complete"),
            location="acquisition reconstruction complete",
        )
        if reconstruction_complete != reconstruction_full:
            raise ValueError(
                "acquisition reconstruction complete/full_cohort_complete mismatch"
            )
        if reconstruction_full != reconstruction_full_claim:
            raise ValueError(
                "acquisition reconstruction/cache full-cohort statement mismatch"
            )
        if root_eligible and (
            reconstruction_scope != "full_cohort" or not reconstruction_full
        ):
            raise ValueError(
                "scientific acquisition reconstruction is not full-cohort complete"
            )
        reconstruction_expected_ids_hash = _validate_sha256(
            reconstruction.get("expected_usable_session_ids_sha256"),
            location="acquisition reconstruction expected_usable_session_ids_sha256",
        )
        if reconstruction_expected_ids_hash != expected_ids_hash:
            raise ValueError(
                "acquisition reconstruction/cache expected usable-session hash mismatch"
            )
        reconstruction_expected_ids = _unique_string_list(
            reconstruction.get("expected_usable_session_ids"),
            location="acquisition reconstruction expected_usable_session_ids",
        )
        if reconstruction_expected_ids != expected_ids:
            raise ValueError(
                "acquisition reconstruction/cache expected usable-session IDs mismatch"
            )
    reconstruction_sessions_raw = reconstruction.get("sessions")
    if not isinstance(reconstruction_sessions_raw, list):
        raise ValueError("acquisition reconstruction sessions list is missing")
    if any(
        not isinstance(item, dict)
        or not isinstance(item.get("session_id"), str)
        or not item.get("session_id")
        for item in reconstruction_sessions_raw
    ):
        raise ValueError("acquisition reconstruction contains an invalid session entry")
    reconstruction_sessions = {
        str(item.get("session_id")): item for item in reconstruction_sessions_raw
    }
    if len(reconstruction_sessions) != len(reconstruction_sessions_raw):
        raise ValueError("acquisition reconstruction contains duplicate session IDs")

    if is_v2:
        reconstruction_usable_ids: list[str] = []
        for item in reconstruction_sessions_raw:
            usable = item.get("usable")
            if type(usable) is not bool:
                raise ValueError(
                    "version-2 acquisition reconstruction session usable must be boolean"
                )
            if usable:
                reconstruction_usable_ids.append(str(item["session_id"]))
        available_ids = [str(item.get("session_id", "")) for item in available_items]
        reconstruction_usable_set = set(reconstruction_usable_ids)
        available_set = set(available_ids)
        if root_eligible and reconstruction_usable_set != available_set:
            raise ValueError(
                "feature cache/reconstruction usable-session cover mismatch"
            )
        if not root_eligible and not available_set <= reconstruction_usable_set:
            raise ValueError(
                "diagnostic feature cache contains a session outside its reconstruction"
            )

    if not available_items:
        raise ValueError("feature cache contains no usable acquisition sessions")

    # Inspect the complete usable catalogue, not only a requested subset.  A
    # session selection must never hide a legacy or unauthorized member of the
    # same cache root.
    for item in available_items:
        session_id = str(item.get("session_id", ""))
        if not session_id:
            raise ValueError("usable feature cache session is missing session_id")
        session_manifest_path = root / session_id / "manifest.json"
        if not session_manifest_path.is_file():
            raise FileNotFoundError(
                f"feature cache session manifest missing: {session_manifest_path}"
            )
        session_manifest = _read_strict_json(
            session_manifest_path,
            f"feature cache session manifest {session_id}",
        )
        declared_session_content = session_manifest.get("content_sha256")
        observed_session_content = _canonical_content_sha256(session_manifest)
        if is_v2:
            declared_session_content = _validate_sha256(
                declared_session_content,
                location=(
                    f"version-2 feature cache session {session_id} "
                    "content_sha256"
                ),
            )
        if (
            declared_session_content is not None
            and declared_session_content != observed_session_content
        ):
            raise ValueError(f"feature cache session content_sha256 mismatch: {session_id}")
        session_contract = session_manifest.get("acquisition_contract")
        if not isinstance(session_contract, dict) or not session_contract:
            if _has_acquisition_indicator(session_manifest):
                detail = "partial acquisition session manifest"
            else:
                detail = "legacy session mixed into acquisition cache"
            raise ValueError(f"{detail}: {session_id}")
        if session_contract.get("schema_version") != schema_version:
            raise ValueError(f"session acquisition schema_version mismatch: {session_id}")

        if is_v2:
            assert root_config_sha256 is not None
            assert root_pipeline_sha256 is not None
            for key, root_value in (
                ("config_sha256", root_config_sha256),
                ("pipeline_sha256", root_pipeline_sha256),
            ):
                session_value = _validate_sha256(
                    session_manifest.get(key),
                    location=f"session {session_id} {key}",
                )
                item_value = _validate_sha256(
                    item.get(key),
                    location=f"root catalogue {session_id} {key}",
                )
                if session_value != root_value or item_value != root_value:
                    raise ValueError(
                        f"feature cache root/session {key} mismatch: {session_id}"
                    )
            item_content = _validate_sha256(
                item.get("content_sha256"),
                location=f"root catalogue {session_id} content_sha256",
            )
            if item_content != observed_session_content:
                raise ValueError(
                    f"root catalogue/session content SHA-256 mismatch: {session_id}"
                )
            session_manifest_file_hash = _validate_sha256(
                item.get("session_manifest_sha256"),
                location=f"root catalogue {session_id} session_manifest_sha256",
            )
            if session_manifest_file_hash != _sha256_file(session_manifest_path):
                raise ValueError(f"session manifest file SHA-256 mismatch: {session_id}")
            session_manifest_content_hash = _validate_sha256(
                item.get("session_manifest_content_sha256"),
                location=f"root catalogue {session_id} session_manifest_content_sha256",
            )
            if session_manifest_content_hash != observed_session_content:
                raise ValueError(f"session manifest content SHA-256 mismatch: {session_id}")
            _validate_v2_session_inventory(
                session_manifest_path.parent,
                session_manifest,
                session_id=session_id,
                require_all_timing_valid=bool(
                    root_eligible
                    or _explicit_boolean(
                        session_contract.get("scientific_eligible"),
                        location=f"session {session_id} scientific_eligible",
                    )
                ),
            )
            if item.get("inventory_sha256") != session_manifest.get(
                "inventory_sha256"
            ):
                raise ValueError(
                    f"root catalogue/session inventory SHA-256 mismatch: {session_id}"
                )

        # The catalogue and on-disk session manifest must be byte-semantically
        # identical.  Unlike the root contract, these are per-session bindings.
        item_contract = item.get("acquisition_contract")
        if not isinstance(item_contract, dict) or not item_contract:
            raise ValueError(f"legacy root catalogue item mixed into acquisition cache: {session_id}")
        if _canonical_json(item_contract) != _canonical_json(session_contract):
            raise ValueError(f"root catalogue/session acquisition contract mismatch: {session_id}")

        session_annotation_columns = _annotation_columns(
            session_contract.get("annotation_only_columns"),
            location=f"session {session_id} annotation_only_columns",
        )
        if session_annotation_columns != root_annotation_columns:
            raise ValueError(f"root/session annotation-only column order mismatch: {session_id}")

        acquisition_session_hash = _validate_sha256(
            session_contract.get("acquisition_session_manifest_sha256"),
            location=f"session {session_id} acquisition_session_manifest_sha256",
        )
        sync_receipt_hash = _validate_sha256(
            session_contract.get("sync_receipt_content_sha256"),
            location=f"session {session_id} sync_receipt_content_sha256",
        )
        mapping_hash = _validate_sha256(
            session_contract.get("mapping_sha256"),
            location=f"session {session_id} mapping_sha256",
        )
        approval_hash = _validate_sha256(
            session_contract.get("manual_approval_content_sha256"),
            location=f"session {session_id} manual_approval_content_sha256",
            nullable=True,
        )
        range_hash = _validate_sha256(
            session_contract.get("range_artifact_sha256"),
            location=f"session {session_id} range_artifact_sha256",
            nullable=True,
        )
        if session_contract.get("protocol_annotation_schema_version") != "acquisition_protocol_v1":
            raise ValueError(f"protocol annotation schema mismatch: {session_id}")
        alignment_mode = session_contract.get("reference_alignment_mode")
        if alignment_mode not in {
            "authorized_marker_affine_v1",
            "diagnostic_unapproved_proposal_v1",
        }:
            raise ValueError(f"reference alignment mode is invalid: {session_id}")
        session_eligible = _explicit_boolean(
            session_contract.get("scientific_eligible"),
            location=f"session {session_id} scientific_eligible",
        )
        if session_eligible != root_eligible:
            raise ValueError(
                f"feature cache root/session scientific eligibility mismatch: {session_id}"
            )
        if session_eligible and alignment_mode != "authorized_marker_affine_v1":
            raise ValueError(f"eligible session lacks authorized marker alignment: {session_id}")
        if require_scientific_eligible and not session_eligible:
            raise ValueError(f"acquisition session is not scientifically eligible: {session_id}")

        source_item = reconstruction_sessions.get(session_id)
        if source_item is None:
            raise ValueError(f"session missing from acquisition reconstruction: {session_id}")
        if source_item.get("content_sha256") != acquisition_session_hash:
            raise ValueError(f"acquisition session/root reconstruction hash mismatch: {session_id}")
        source_item_eligible = _explicit_boolean(
            source_item.get("scientific_eligible"),
            location=f"acquisition reconstruction {session_id} scientific_eligible",
        )
        if session_eligible and not source_item_eligible:
            raise ValueError(f"acquisition session scientific eligibility mismatch: {session_id}")
        source_manifest_value = source_item.get("manifest")
        if not isinstance(source_manifest_value, str) or not source_manifest_value:
            raise ValueError(f"acquisition session manifest path is missing: {session_id}")
        source_manifest_path = Path(source_manifest_value)
        if not source_manifest_path.is_absolute():
            source_manifest_path = reconstruction_path.parent / source_manifest_path
        source_manifest_path = source_manifest_path.resolve()
        try:
            source_manifest_path.relative_to(reconstruction_path.parent.resolve())
        except ValueError as error:
            raise ValueError(
                f"acquisition session manifest escapes reconstruction root: {session_id}"
            ) from error
        if not source_manifest_path.is_file():
            raise FileNotFoundError(
                f"bound acquisition session manifest missing: {source_manifest_path}"
            )
        source_manifest = _read_strict_json(
            source_manifest_path,
            f"acquisition source session manifest {session_id}",
        )
        if source_manifest.get("content_sha256") != acquisition_session_hash or (
            _canonical_content_sha256(source_manifest) != acquisition_session_hash
        ):
            raise ValueError(f"acquisition session canonical content hash mismatch: {session_id}")
        source_manifest_eligible = _explicit_boolean(
            source_manifest.get("scientific_eligible"),
            location=f"acquisition source session {session_id} scientific_eligible",
        )
        if source_manifest_eligible != source_item_eligible:
            raise ValueError(
                f"acquisition root/source session scientific eligibility mismatch: {session_id}"
            )
        source_contract = None
        if is_v2:
            source_contract = (
                None
                if verified_reconstruction is None
                else verified_reconstruction.sessions.get(session_id)
            )
            if source_contract is None:
                raise ValueError(
                    f"verified acquisition session contract is missing: {session_id}"
                )
            source_sensor_summary = source_contract.manifest.get("sensor_summary")
            source_radar_summary = (
                source_sensor_summary.get("radar")
                if isinstance(source_sensor_summary, dict)
                else None
            )
            source_resampling_summary = (
                source_radar_summary.get("feature_resampling")
                if isinstance(source_radar_summary, dict)
                else None
            )
            cache_resampling_summary = session_manifest.get("radar_timing_summary")
            if (
                not isinstance(cache_resampling_summary, dict)
                or _canonical_json(cache_resampling_summary)
                != _canonical_json(source_resampling_summary)
            ):
                raise ValueError(
                    f"cache/source radar timing summary mismatch: {session_id}"
                )
            expected_cache_session_eligible = bool(
                mode == "strict" and source_contract.scientific_eligible
            )
            if session_eligible != expected_cache_session_eligible:
                raise ValueError(
                    f"cache mode/source session scientific eligibility mismatch: {session_id}"
                )
            expected_components = {
                "measured_timing_eligible": source_contract.measured_timing_eligible,
                "alignment_eligible": source_contract.alignment_eligible,
                "stage_metric_eligible": source_contract.stage_metric_eligible,
                "range_feature_eligible": source_contract.range_feature_eligible,
                "strict_cache_eligible": source_contract.strict_cache_eligible,
            }
            for key, expected_value in expected_components.items():
                cache_value = _explicit_boolean(
                    session_contract.get(key),
                    location=f"session {session_id} {key}",
                )
                if cache_value != expected_value:
                    raise ValueError(
                        f"cache/source session {key} mismatch: {session_id}"
                    )
        synchronization = source_manifest.get("synchronization")
        if not isinstance(synchronization, dict):
            raise ValueError(f"acquisition synchronization record is missing: {session_id}")
        if synchronization.get("receipt_content_sha256") != sync_receipt_hash:
            raise ValueError(f"synchronization receipt hash mismatch: {session_id}")
        if synchronization.get("manual_approval_content_sha256") != approval_hash:
            raise ValueError(f"manual synchronization approval hash mismatch: {session_id}")
        source_mapping = synchronization.get("mapping")
        if not isinstance(source_mapping, dict) or _canonical_sha256(source_mapping) != mapping_hash:
            raise ValueError(f"synchronization mapping hash mismatch: {session_id}")
        source_range = source_manifest.get("range_tracking")
        if not isinstance(source_range, dict):
            raise ValueError(f"acquisition range-tracking record is missing: {session_id}")
        if source_range.get("artifact_sha256") != range_hash:
            raise ValueError(f"range artifact hash mismatch: {session_id}")
        source_protocol_contract = source_manifest.get("protocol_contract")
        if not isinstance(source_protocol_contract, dict) or source_protocol_contract.get(
            "schema_version"
        ) != session_contract.get("protocol_annotation_schema_version"):
            raise ValueError(f"protocol annotation binding mismatch: {session_id}")

        metadata_path = root / session_id / "metadata.csv"
        if not metadata_path.is_file():
            raise FileNotFoundError(f"feature cache metadata missing: {metadata_path}")
        _validate_acquisition_metadata(
            pd.read_csv(metadata_path),
            session_id=session_id,
            session_scientific_eligible=session_eligible,
            synchronization_authorized=_explicit_boolean(
                synchronization.get("authorized"),
                location=f"acquisition synchronization {session_id} authorized",
            ),
            source_contract=source_contract,
            cache_session_manifest=session_manifest,
        )
    return root_contract


def fit_aux_scaler(aux: np.ndarray, indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fit a train-only robust scaler (median/IQR with finite guards)."""

    train = np.asarray(aux[np.asarray(indices)], dtype=np.float64)
    center = np.nanmedian(train, axis=0)
    q25, q75 = np.nanquantile(train, [0.25, 0.75], axis=0)
    scale = q75 - q25
    standard_deviation = np.nanstd(train, axis=0)
    scale = np.where(scale > 1e-6, scale, np.where(standard_deviation > 1e-6, standard_deviation, 1.0))
    center = np.where(np.isfinite(center), center, 0.0).astype(np.float32)
    scale = np.where(np.isfinite(scale), scale, 1.0).astype(np.float32)
    return center, scale


def transform_aux(aux: np.ndarray, center: np.ndarray, scale: np.ndarray) -> np.ndarray:
    transformed = (np.asarray(aux, dtype=np.float32) - center) / scale
    return np.clip(np.nan_to_num(transformed), -8.0, 8.0).astype(np.float32)


def causal_history_feature_names(
    *,
    lags: tuple[int, ...] = DEFAULT_HISTORY_LAGS,
    rolling_windows: tuple[int, ...] = DEFAULT_HISTORY_ROLLING_WINDOWS,
) -> list[str]:
    """Return the stable column order used by :func:`causal_history_features`."""

    lags = _positive_unique_ints(lags, "lags")
    rolling_windows = _positive_unique_ints(rolling_windows, "rolling_windows")
    names: list[str] = []
    for lag in lags:
        names.extend(
            [
                f"history_lag_{lag}_classical_rr_bpm",
                f"history_lag_{lag}_classical_confidence",
                f"history_lag_{lag}_radar_peak_spread_bpm",
                f"history_lag_{lag}_available",
            ]
        )
    for window in rolling_windows:
        names.extend(
            [
                f"history_roll_{window}_rr_median_bpm",
                f"history_roll_{window}_rr_mad_bpm",
                f"history_roll_{window}_rr_conf_weighted_mean_bpm",
                f"history_roll_{window}_rr_trend_bpm_per_window",
                f"history_roll_{window}_confidence_mean",
                f"history_roll_{window}_spread_median_bpm",
                f"history_roll_{window}_available_fraction",
                f"history_roll_{window}_sufficient",
            ]
        )
    return names


def causal_history_features(
    metadata: pd.DataFrame,
    *,
    lags: tuple[int, ...] = DEFAULT_HISTORY_LAGS,
    rolling_windows: tuple[int, ...] = DEFAULT_HISTORY_ROLLING_WINDOWS,
) -> tuple[np.ndarray, list[str]]:
    """Build label-free, strictly causal per-session history features.

    The function only reads the current cache's radar-derived
    ``classical_rr_bpm``, ``classical_confidence`` and
    ``radar_peak_spread_bpm`` columns.  Reference RR, validity and
    observability columns are deliberately never consulted.  An exact lag is
    available only when the same ``session_id`` contains
    ``window_number - lag``; rows from another session are never bridged.

    Rolling summaries likewise use only exact preceding window numbers and
    exclude the current row.  Consequently the result can be computed once on
    the full cache before a train/validation/test split: it contains the same
    information that would have been available online at each window and does
    not depend on split membership.  Missing history values are filled with
    zero and accompanied by explicit availability features.

    The returned rows preserve the input order, even when ``metadata`` is not
    sorted chronologically.
    """

    lags = _positive_unique_ints(lags, "lags")
    rolling_windows = _positive_unique_ints(rolling_windows, "rolling_windows")
    names = causal_history_feature_names(lags=lags, rolling_windows=rolling_windows)
    required = {
        "session_id",
        "window_number",
        "classical_rr_bpm",
        "classical_confidence",
        "radar_peak_spread_bpm",
    }
    missing = sorted(required - set(metadata.columns))
    if missing:
        raise KeyError(f"metadata missing causal-history columns: {missing}")
    if metadata.empty:
        return np.empty((0, len(names)), dtype=np.float32), names

    session_values = metadata["session_id"].to_numpy()
    if pd.isna(session_values).any():
        raise ValueError("session_id contains missing values")
    raw_windows = pd.to_numeric(metadata["window_number"], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(raw_windows).all() or not np.all(raw_windows == np.rint(raw_windows)):
        raise ValueError("window_number must contain finite integers")
    window_numbers = raw_windows.astype(np.int64)

    rr = pd.to_numeric(metadata["classical_rr_bpm"], errors="coerce").to_numpy(dtype=float)
    confidence = pd.to_numeric(metadata["classical_confidence"], errors="coerce").to_numpy(dtype=float)
    spread = pd.to_numeric(metadata["radar_peak_spread_bpm"], errors="coerce").to_numpy(dtype=float)
    values = np.zeros((len(metadata), len(names)), dtype=np.float32)

    # ``sort=False`` preserves first-seen session order, while all assignment
    # is by original row position so input row order is retained.
    sessions = pd.Series(np.arange(len(metadata))).groupby(session_values, sort=False).groups
    for session_id, row_index in sessions.items():
        rows = np.asarray(list(row_index), dtype=np.int64)
        numbers = window_numbers[rows]
        if len(np.unique(numbers)) != len(numbers):
            raise ValueError(f"duplicate window_number within session {session_id!r}")
        row_for_window = {int(number): int(row) for number, row in zip(numbers, rows, strict=True)}

        for row in rows:
            current_window = int(window_numbers[row])
            column = 0
            for lag in lags:
                previous_row = row_for_window.get(current_window - lag)
                if previous_row is not None:
                    history = np.asarray(
                        [rr[previous_row], confidence[previous_row], spread[previous_row]], dtype=float
                    )
                    finite = np.isfinite(history)
                    values[row, column : column + 3] = np.where(finite, history, 0.0)
                    values[row, column + 3] = float(finite.all())
                column += 4

            for window in rolling_windows:
                previous_rows = [
                    row_for_window[number]
                    for number in range(current_window - window, current_window)
                    if number in row_for_window
                ]
                if previous_rows:
                    previous_rows_array = np.asarray(previous_rows, dtype=np.int64)
                    rr_values = rr[previous_rows_array]
                    confidence_values = confidence[previous_rows_array]
                    spread_values = spread[previous_rows_array]
                    complete = (
                        np.isfinite(rr_values)
                        & np.isfinite(confidence_values)
                        & np.isfinite(spread_values)
                    )
                    rr_valid = rr_values[complete]
                    confidence_valid = confidence_values[complete]
                    spread_valid = spread_values[complete]
                    previous_windows = window_numbers[previous_rows_array][complete]
                else:
                    rr_valid = np.empty(0, dtype=float)
                    confidence_valid = np.empty(0, dtype=float)
                    spread_valid = np.empty(0, dtype=float)
                    previous_windows = np.empty(0, dtype=np.int64)

                available = len(rr_valid)
                if available:
                    rr_median = float(np.median(rr_valid))
                    rr_mad = float(np.median(np.abs(rr_valid - rr_median)))
                    nonnegative_weights = np.clip(confidence_valid, 0.0, None)
                    if float(nonnegative_weights.sum()) > 1e-12:
                        weighted_mean = float(np.average(rr_valid, weights=nonnegative_weights))
                    else:
                        weighted_mean = rr_median
                    if available >= 2 and np.ptp(previous_windows) > 0:
                        centered_window = previous_windows - previous_windows.mean()
                        trend = float(
                            np.dot(centered_window, rr_valid - rr_valid.mean())
                            / np.dot(centered_window, centered_window)
                        )
                    else:
                        trend = 0.0
                    values[row, column : column + 8] = (
                        rr_median,
                        rr_mad,
                        weighted_mean,
                        trend,
                        float(np.mean(confidence_valid)),
                        float(np.median(spread_valid)),
                        available / window,
                        float(available >= 2),
                    )
                # With no complete preceding radar estimate all eight fields
                # retain their initialized zero values.
                column += 8

    return values, names


def append_causal_history_features(
    aux: np.ndarray,
    metadata: pd.DataFrame,
    *,
    lags: tuple[int, ...] = DEFAULT_HISTORY_LAGS,
    rolling_windows: tuple[int, ...] = DEFAULT_HISTORY_ROLLING_WINDOWS,
) -> tuple[np.ndarray, list[str]]:
    """Append :func:`causal_history_features` to an auxiliary feature matrix."""

    aux_array = np.asarray(aux)
    if aux_array.ndim != 2 or len(aux_array) != len(metadata):
        raise ValueError("aux must be a 2-D array with one row per metadata row")
    history, names = causal_history_features(
        metadata, lags=lags, rolling_windows=rolling_windows
    )
    output_dtype = np.result_type(aux_array.dtype, np.float32)
    augmented = np.concatenate(
        [aux_array.astype(output_dtype, copy=False), history.astype(output_dtype, copy=False)], axis=1
    )
    return augmented, names


def _positive_unique_ints(values: tuple[int, ...], name: str) -> tuple[int, ...]:
    normalized = tuple(int(value) for value in values)
    if any(value <= 0 for value in normalized):
        raise ValueError(f"{name} must contain positive integers")
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"{name} must not contain duplicates")
    return normalized
