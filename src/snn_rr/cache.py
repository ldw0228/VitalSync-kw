"""Feature-cache loading for grouped training and evaluation."""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .acquisition_contract import (
    AcquisitionSessionContract,
    assign_stage_window,
    load_acquisition_reconstruction,
)
from .preprocess import identity_for_session


DEFAULT_HISTORY_LAGS = (1, 2, 4, 8)
DEFAULT_HISTORY_ROLLING_WINDOWS = (4, 8)
ACQUISITION_CACHE_SCHEMA_VERSION = "snn_rr.feature_cache_acquisition.v1"
ACQUISITION_CACHE_SCHEMA_VERSION_V2 = "snn_rr.feature_cache_acquisition.v2"
ACQUISITION_RECONSTRUCTION_SCHEMA_VERSION_V2 = "snn_rr.acquisition_reconstruction.v2"
SUPPORTED_ACQUISITION_CACHE_SCHEMA_VERSIONS = frozenset(
    {ACQUISITION_CACHE_SCHEMA_VERSION, ACQUISITION_CACHE_SCHEMA_VERSION_V2}
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

    Version-1 acquisition caches remain readable for historical diagnostics,
    but only a fully verified version-2 cache can be classified as
    ``acquisition_scientific``.  ``inventory_sha256`` binds the exact files
    loaded for ``selected_sessions`` even when an older manifest did not carry
    its own per-file inventory.
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
            require_scientific_eligible=require_scientific_eligible,
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
    # V1 remains a historical/diagnostic compatibility surface.  It lacks the
    # full-cohort and exact file-inventory guarantees required for a new
    # scientific run, even when its historical flag said eligible.
    effective_eligible = bool(
        schema_version == ACQUISITION_CACHE_SCHEMA_VERSION_V2
        and declared_eligible
        and sessions is None
    )
    if acquisition_contract is None:
        classification = "legacy"
    elif effective_eligible:
        classification = "acquisition_scientific"
    elif schema_version == ACQUISITION_CACHE_SCHEMA_VERSION:
        classification = "acquisition_historical_v1"
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


def _read_strict_json(path: Path, label: str) -> dict[str, Any]:
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
            path.read_text(encoding="utf-8"),
            object_pairs_hook=reject_duplicates,
            parse_constant=reject_nonfinite,
        )
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read {label}: {path} ({error})") from error
    if not isinstance(document, dict):
        raise ValueError(f"{label} must be a JSON object")
    return document


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


def _annotation_columns(value: Any, *, location: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{location} must be a non-empty list")
    columns = tuple(str(column) for column in value)
    if len(set(columns)) != len(columns):
        raise ValueError(f"{location} must not contain duplicates")
    if set(columns) != set(REQUIRED_ACQUISITION_ANNOTATION_COLUMNS):
        missing = sorted(REQUIRED_ACQUISITION_ANNOTATION_COLUMNS - set(columns))
        extra = sorted(set(columns) - REQUIRED_ACQUISITION_ANNOTATION_COLUMNS)
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
    if require_scientific_eligible and not is_v2:
        raise ValueError(
            "scientific training requires a version-2 full-cohort acquisition cache; "
            "version 1 is historical/diagnostic only"
        )
    if require_scientific_eligible and not root_eligible:
        raise ValueError("feature cache root is not scientifically eligible")

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
