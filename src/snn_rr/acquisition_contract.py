"""Strict consumers for versioned acquisition-reconstruction artifacts.

This module is deliberately separate from raw feature extraction.  It turns a
reconstruction manifest into a content-verified, fail-closed contract that may
be used for offline BIOPAC label alignment and retrospective stage metadata.
None of the BIOPAC-derived fields exposed here are deployable model inputs.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from zipfile import BadZipFile
from typing import Any, Mapping

import numpy as np
import yaml

from .acquisition_protocol import (
    AcquisitionProtocolConfig,
    _source_snapshot,
    load_dataset_issue_records,
    load_protocol_config,
)
from .data import (
    RADAR_BINS,
    XETHRU_RECORD_BYTES,
    XETHRU_RECORD_DTYPE,
    build_dataset_manifest,
)
from .preprocess import identity_for_session
from .synchronization import (
    SynchronizationConfig,
    TimeMapping,
    canonical_content_sha256,
    canonical_json_bytes,
    synchronization_is_authorized,
    validate_manual_approval,
    validate_sync_receipt,
)


ACQUISITION_SCHEMA = "snn_rr.acquisition_reconstruction.v2"
LEGACY_ACQUISITION_SCHEMA = "snn_rr.acquisition_reconstruction.v1"
ACQUISITION_COHORT_AUTHORITY_SCHEMA = "snn_rr.acquisition_cohort_authority.v1"
ACQUISITION_COHORT_V1_CONTENT_SHA256 = (
    "9fe44a6fd0b809dde063f845c3fdfdc60352eda462abc0d0a40e15207d814454"
)
SUPPORTED_ACQUISITION_SCHEMAS = frozenset(
    {ACQUISITION_SCHEMA, LEGACY_ACQUISITION_SCHEMA}
)
ANNOTATION_ONLY_COLUMNS: tuple[str, ...] = (
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
)


class AcquisitionContractError(ValueError):
    """Raised when acquisition provenance is absent, stale, or inconsistent."""


@dataclass(frozen=True, slots=True)
class AcquisitionCohortAuthority:
    """Immutable identity/session authority for the original 30-session cohort."""

    path: Path
    file_sha256: str
    content_sha256: str
    expected_session_ids: tuple[str, ...]
    expected_usable_session_ids: tuple[str, ...]
    excluded_sessions: tuple[tuple[str, str], ...]
    session_identities: tuple[tuple[str, str], ...]
    expected_physical_identities: tuple[str, ...]

    @property
    def session_identity_map(self) -> dict[str, str]:
        return dict(self.session_identities)

    def to_claims(self) -> dict[str, Any]:
        return {
            "expected_session_ids": list(self.expected_session_ids),
            "expected_usable_session_ids": list(self.expected_usable_session_ids),
            "excluded_sessions": [
                {"session_id": session_id, "reason": reason}
                for session_id, reason in self.excluded_sessions
            ],
            "session_identities": [
                {
                    "session_id": session_id,
                    "physical_identity": physical_identity,
                }
                for session_id, physical_identity in self.session_identities
            ],
            "expected_physical_identities": list(
                self.expected_physical_identities
            ),
        }


def _authority_string_array(
    document: Mapping[str, Any], key: str
) -> tuple[str, ...]:
    value = document.get(key)
    if not isinstance(value, list) or any(
        not isinstance(item, str) or not item for item in value
    ):
        raise AcquisitionContractError(
            f"cohort authority.{key} must be an array of non-empty strings"
        )
    result = tuple(value)
    if len(set(result)) != len(result):
        raise AcquisitionContractError(
            f"cohort authority.{key} contains duplicate IDs"
        )
    return result


def load_acquisition_cohort_authority(
    path: str | Path,
    *,
    expected_sha256: str | None = None,
) -> AcquisitionCohortAuthority:
    """Parse and validate the immutable v1 cohort authority artifact."""

    authority_path = Path(path).resolve()
    try:
        payload, _ = _source_snapshot(
            authority_path,
            label="cohort authority",
            expected_sha256=expected_sha256,
        )
        document = yaml.safe_load(payload.decode("utf-8"))
    except (OSError, UnicodeError, ValueError, yaml.YAMLError) as error:
        raise AcquisitionContractError(
            f"cannot read cohort authority {authority_path}: {error}"
        ) from error
    if not isinstance(document, Mapping):
        raise AcquisitionContractError("cohort authority must be a mapping")
    expected_keys = {
        "schema",
        "content_sha256",
        "expected_session_count",
        "expected_usable_session_count",
        "expected_physical_identity_count",
        "expected_session_ids",
        "expected_usable_session_ids",
        "excluded_sessions",
        "session_identities",
        "expected_physical_identities",
    }
    if set(document) != expected_keys:
        missing = sorted(expected_keys - set(document))
        extra = sorted(set(document) - expected_keys)
        raise AcquisitionContractError(
            f"cohort authority keys mismatch; missing={missing}, extra={extra}"
        )
    if document.get("schema") != ACQUISITION_COHORT_AUTHORITY_SCHEMA:
        raise AcquisitionContractError("unexpected cohort authority schema")
    content_sha256 = document.get("content_sha256")
    if not isinstance(content_sha256, str) or len(content_sha256) != 64:
        raise AcquisitionContractError("cohort authority content_sha256 is invalid")
    observed_content_sha256 = canonical_content_sha256(document)
    if content_sha256 != observed_content_sha256:
        raise AcquisitionContractError("cohort authority content_sha256 mismatch")

    session_ids = _authority_string_array(document, "expected_session_ids")
    usable_ids = _authority_string_array(
        document, "expected_usable_session_ids"
    )
    physical_identities = _authority_string_array(
        document, "expected_physical_identities"
    )
    count_claims = (
        ("expected_session_count", len(session_ids), 30),
        ("expected_usable_session_count", len(usable_ids), 29),
        ("expected_physical_identity_count", len(physical_identities), 18),
    )
    for key, observed, frozen in count_claims:
        claimed = document.get(key)
        if type(claimed) is not int or claimed != observed or claimed != frozen:
            raise AcquisitionContractError(
                f"cohort authority.{key} must equal the frozen value {frozen}"
            )
    usable_set = set(usable_ids)
    if not usable_set <= set(session_ids) or usable_ids != tuple(
        session_id for session_id in session_ids if session_id in usable_set
    ):
        raise AcquisitionContractError(
            "cohort authority usable IDs must be an ordered subset of session IDs"
        )
    excluded_ids = tuple(
        session_id for session_id in session_ids if session_id not in usable_set
    )
    if excluded_ids != ("S24_KHJ",):
        raise AcquisitionContractError(
            "cohort authority must exclude only S24_KHJ from usability"
        )

    excluded_raw = document.get("excluded_sessions")
    if not isinstance(excluded_raw, list):
        raise AcquisitionContractError(
            "cohort authority.excluded_sessions must be an array"
        )
    excluded: list[tuple[str, str]] = []
    for item in excluded_raw:
        if not isinstance(item, Mapping) or set(item) != {"session_id", "reason"}:
            raise AcquisitionContractError(
                "cohort authority excluded-session entry is invalid"
            )
        session_id = item.get("session_id")
        reason = item.get("reason")
        if not isinstance(session_id, str) or not isinstance(reason, str):
            raise AcquisitionContractError(
                "cohort authority excluded-session values are invalid"
            )
        excluded.append((session_id, reason))
    if tuple(excluded) != (("S24_KHJ", "empty_three_radar_streams"),):
        raise AcquisitionContractError(
            "cohort authority must bind S24_KHJ to empty_three_radar_streams"
        )

    assignments_raw = document.get("session_identities")
    if not isinstance(assignments_raw, list):
        raise AcquisitionContractError(
            "cohort authority.session_identities must be an array"
        )
    assignments: list[tuple[str, str]] = []
    for item in assignments_raw:
        if not isinstance(item, Mapping) or set(item) != {
            "session_id",
            "physical_identity",
        }:
            raise AcquisitionContractError(
                "cohort authority identity assignment is invalid"
            )
        session_id = item.get("session_id")
        identity = item.get("physical_identity")
        if (
            not isinstance(session_id, str)
            or not session_id
            or not isinstance(identity, str)
            or not identity
        ):
            raise AcquisitionContractError(
                "cohort authority identity assignment values are invalid"
            )
        assignments.append((session_id, identity))
    if tuple(session_id for session_id, _ in assignments) != session_ids:
        raise AcquisitionContractError(
            "cohort authority identity assignments must exactly follow session order"
        )
    if dict(assignments).get("S17_RJS") != "PJS":
        raise AcquisitionContractError(
            "cohort authority must map S17_RJS to physical identity PJS"
        )
    derived_identities: list[str] = []
    assignment_map = dict(assignments)
    for session_id in usable_ids:
        identity = assignment_map[session_id]
        if identity not in derived_identities:
            derived_identities.append(identity)
    if tuple(derived_identities) != physical_identities:
        raise AcquisitionContractError(
            "cohort authority physical identities do not match usable sessions"
        )
    for session_id, identity in assignments:
        try:
            canonical_identity = identity_for_session(session_id)
        except ValueError as error:
            raise AcquisitionContractError(
                f"cohort authority contains unknown session {session_id}"
            ) from error
        if identity != canonical_identity:
            raise AcquisitionContractError(
                f"cohort authority identity mismatch for {session_id}"
            )
    if content_sha256 != ACQUISITION_COHORT_V1_CONTENT_SHA256:
        raise AcquisitionContractError(
            "immutable cohort authority content does not match frozen v1"
        )

    return AcquisitionCohortAuthority(
        path=authority_path,
        file_sha256=_sha256_file(authority_path),
        content_sha256=content_sha256,
        expected_session_ids=session_ids,
        expected_usable_session_ids=usable_ids,
        excluded_sessions=tuple(excluded),
        session_identities=tuple(assignments),
        expected_physical_identities=physical_identities,
    )


@dataclass(frozen=True, slots=True)
class AcquisitionSessionContract:
    session_id: str
    reconstruction_root: Path
    manifest_path: Path
    manifest: Mapping[str, Any]
    receipt_path: Path
    receipt: Mapping[str, Any]
    mapping: TimeMapping | None
    manual_approval_path: Path | None
    manual_approval: Mapping[str, Any] | None
    authorized: bool
    measured_timing_eligible: bool
    alignment_eligible: bool
    stage_metric_eligible: bool
    range_feature_eligible: bool
    strict_cache_eligible: bool
    scientific_eligible: bool
    protocol: Mapping[str, Any]
    window_minimum_overlap_fraction: float
    transition_guard_s: float
    range_track_path: Path | None

    @property
    def content_sha256(self) -> str:
        return str(self.manifest["content_sha256"])

    @property
    def receipt_content_sha256(self) -> str:
        return str(self.receipt["content_sha256"])

    @property
    def mapping_sha256(self) -> str:
        return hashlib.sha256(
            canonical_json_bytes(self.receipt["result"]["mapping"])
        ).hexdigest()


@dataclass(frozen=True, slots=True)
class AcquisitionReconstruction:
    root: Path
    manifest_path: Path
    manifest: Mapping[str, Any]
    sessions: Mapping[str, AcquisitionSessionContract]
    selection_scope: str
    execution_complete: bool
    full_cohort_complete: bool
    scientific_eligible: bool

    @property
    def content_sha256(self) -> str:
        return str(self.manifest["content_sha256"])


@dataclass(frozen=True, slots=True)
class StageWindowAssignment:
    stage_id: str | None
    stage_name: str | None
    stage_status: str | None
    stage_confidence: float | None
    overlap_fraction: float
    transition_window: bool
    eligible_for_stage_metrics: bool
    phase7_assignment: str | None
    reason: str


def _read_strict_json_snapshot(path: Path) -> tuple[dict[str, Any], str]:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise AcquisitionContractError(
                    f"JSON document repeats key {key!r}: {path}"
                )
            result[key] = value
        return result

    def reject_nonfinite(value: str) -> None:
        raise AcquisitionContractError(
            f"JSON document contains non-finite number {value}: {path}"
        )

    try:
        payload, digest = _source_snapshot(path, label="acquisition JSON")
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=reject_duplicates,
            parse_constant=reject_nonfinite,
        )
    except (OSError, UnicodeError, ValueError, json.JSONDecodeError) as exc:
        raise AcquisitionContractError(f"cannot read acquisition JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise AcquisitionContractError(f"acquisition JSON must be an object: {path}")
    return value, digest


def _read_strict_json(path: Path) -> dict[str, Any]:
    return _read_strict_json_snapshot(path)[0]


def _require_content_hash(document: Mapping[str, Any], label: str) -> str:
    digest = document.get("content_sha256")
    if not isinstance(digest, str) or len(digest) != 64:
        raise AcquisitionContractError(f"{label} lacks a valid content_sha256")
    if canonical_content_sha256(document) != digest:
        raise AcquisitionContractError(f"{label} content_sha256 mismatch")
    return digest


def _resolve_inside(root: Path, relative: Any, label: str) -> Path:
    if not isinstance(relative, str) or not relative:
        raise AcquisitionContractError(f"{label} must be a non-empty relative path")
    candidate = (root / relative).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError as exc:
        raise AcquisitionContractError(f"{label} escapes reconstruction root") from exc
    if not candidate.is_file():
        raise AcquisitionContractError(f"{label} does not exist: {candidate}")
    return candidate


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _resolve_bound_external_file(
    document: Mapping[str, Any],
    *,
    reconstruction_root: Path,
    path_key: str,
    hash_key: str,
    label: str,
) -> tuple[Path, str]:
    value = document.get(path_key)
    if not isinstance(value, str) or not value:
        raise AcquisitionContractError(
            f"acquisition root.{path_key} must bind the {label} artifact"
        )
    path = Path(value)
    if not path.is_absolute():
        path = (reconstruction_root / path).resolve()
    if not path.is_file():
        raise AcquisitionContractError(f"bound {label} is missing: {path}")
    claimed = _require_sha256(document, hash_key, "acquisition root")
    observed = _sha256_file(path)
    if observed != claimed:
        raise AcquisitionContractError(
            f"acquisition root {label} file hash mismatch"
        )
    return path, observed


def build_v2_raw_input_binding_state(
    subject: Any,
    dataset_root: str | Path,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """Build the exact selected-session raw graph and content bindings."""

    root = Path(dataset_root).resolve()
    session_id = getattr(subject, "subject_id", None)
    selected_session = getattr(subject, "selected_session", None)
    biopac_path = getattr(subject, "biopac_path", None)
    if (
        not isinstance(session_id, str)
        or selected_session is None
        or biopac_path is None
    ):
        raise AcquisitionContractError(
            f"{session_id or 'unknown'} lacks a selected three-radar/BIOPAC graph"
        )
    paths: list[tuple[str, Path]] = [("biopac", Path(biopac_path))]
    radar_graph: list[dict[str, Any]] = []
    for radar_id in (1, 2, 3):
        stream = selected_session.radars.get(radar_id)
        if stream is None or stream.meta_path is None or not stream.data_paths:
            raise AcquisitionContractError(
                f"{session_id} selected-session graph lacks radar {radar_id}"
            )
        meta_key = f"radar{radar_id}_meta"
        paths.append((meta_key, Path(stream.meta_path)))
        data_keys: list[str] = []
        for chunk_index, data_path in enumerate(stream.data_paths):
            key = f"radar{radar_id}_data_{chunk_index:02d}"
            paths.append((key, Path(data_path)))
            data_keys.append(key)
        radar_graph.append(
            {
                "radar_id": radar_id,
                "metadata_binding": meta_key,
                "data_bindings": data_keys,
            }
        )

    bindings: dict[str, dict[str, Any]] = {}
    for key, raw_path in paths:
        resolved = raw_path.resolve()
        try:
            stored_path = str(resolved.relative_to(root))
        except ValueError:
            stored_path = str(resolved)
        bindings[key] = {
            "path": stored_path,
            "bytes": resolved.stat().st_size,
            "sha256": _sha256_file(resolved),
        }
    graph = {
        "schema": "snn_rr.selected_session_raw_input_graph.v1",
        "session_id": session_id,
        "selected_logical_session_id": str(selected_session.session_id),
        "binding_keys": list(bindings),
        "biopac_binding": "biopac",
        "radars": radar_graph,
    }
    return bindings, graph


def build_v2_xethru_record_contract(subject: Any) -> dict[str, Any]:
    """Recompute strict record-size, zero-header, and bin-count evidence."""

    session_id = getattr(subject, "subject_id", "unknown")
    selected_session = getattr(subject, "selected_session", None)
    if selected_session is None:
        raise AcquisitionContractError(
            f"{session_id} lacks a selected radar session"
        )
    views: list[dict[str, Any]] = []
    for radar_id in (1, 2, 3):
        stream = selected_session.radars.get(radar_id)
        if stream is None or not stream.data_paths:
            raise AcquisitionContractError(
                f"{session_id} selected-session graph lacks radar {radar_id} chunks"
            )
        chunks: list[dict[str, Any]] = []
        for chunk_index, raw_path in enumerate(stream.data_paths):
            path = Path(raw_path)
            byte_count = path.stat().st_size
            remainder = byte_count % XETHRU_RECORD_BYTES
            frame_count = byte_count // XETHRU_RECORD_BYTES
            if remainder or frame_count <= 0:
                raise AcquisitionContractError(
                    f"{session_id} radar {radar_id} chunk has invalid record geometry"
                )
            records = np.memmap(
                path,
                dtype=XETHRU_RECORD_DTYPE,
                mode="r",
                shape=(frame_count,),
            )
            chunks.append(
                {
                    "chunk_index": chunk_index,
                    "filename": path.name,
                    "bytes": byte_count,
                    "frame_count": frame_count,
                    "record_bytes": XETHRU_RECORD_BYTES,
                    "payload_bin_count": RADAR_BINS,
                    "record_size_remainder_bytes": remainder,
                    "zero_header_nonzero": int(
                        np.count_nonzero(records["zero"])
                    ),
                    "bin_count_invalid": int(
                        np.count_nonzero(records["bin_count"] != RADAR_BINS)
                    ),
                }
            )
        eligible = all(
            chunk["record_size_remainder_bytes"] == 0
            and chunk["zero_header_nonzero"] == 0
            and chunk["bin_count_invalid"] == 0
            for chunk in chunks
        )
        views.append(
            {"radar_id": radar_id, "chunks": chunks, "eligible": eligible}
        )
    evidence = {
        "schema": "snn_rr.xethru_record_contract.v1",
        "record_bytes": XETHRU_RECORD_BYTES,
        "payload_bin_count": RADAR_BINS,
        "views": views,
        "eligible": all(view["eligible"] for view in views),
    }
    if evidence["eligible"] is not True:
        raise AcquisitionContractError(
            f"{session_id} violates the strict XeThru header/bin-count contract"
        )
    return evidence


def _validate_v2_raw_input_graph(
    session: Mapping[str, Any],
    receipt: Mapping[str, Any],
    *,
    subject: Any,
    dataset_root: Path,
    session_id: str,
) -> None:
    expected_bindings, expected_graph = build_v2_raw_input_binding_state(
        subject, dataset_root
    )
    bindings = session.get("raw_input_bindings")
    if canonical_json_bytes(bindings) != canonical_json_bytes(expected_bindings):
        raise AcquisitionContractError(
            f"{session_id} raw bindings do not exactly cover the selected-session graph"
        )
    if canonical_json_bytes(receipt.get("input_bindings")) != canonical_json_bytes(
        expected_bindings
    ):
        raise AcquisitionContractError(
            f"{session_id} receipt bindings differ from the selected-session graph"
        )
    graph = session.get("raw_input_graph")
    if canonical_json_bytes(graph) != canonical_json_bytes(expected_graph):
        raise AcquisitionContractError(
            f"{session_id} raw input graph differs from dataset discovery"
        )
    graph_hash = _require_sha256(session, "raw_input_graph_sha256", session_id)
    if graph_hash != hashlib.sha256(canonical_json_bytes(expected_graph)).hexdigest():
        raise AcquisitionContractError(
            f"{session_id}.raw_input_graph_sha256 mismatch"
        )
    expected_record_contract = build_v2_xethru_record_contract(subject)
    sensor = session.get("sensor_summary")
    radar = sensor.get("radar") if isinstance(sensor, Mapping) else None
    if not isinstance(radar, Mapping) or canonical_json_bytes(
        radar.get("xethru_record_contract")
    ) != canonical_json_bytes(expected_record_contract):
        raise AcquisitionContractError(
            f"{session_id} XeThru record evidence differs from selected raw chunks"
        )
    expected_record_hash = hashlib.sha256(
        canonical_json_bytes(expected_record_contract)
    ).hexdigest()
    if radar.get("xethru_record_contract_evidence_sha256") != expected_record_hash:
        raise AcquisitionContractError(
            f"{session_id} XeThru record evidence hash mismatch"
        )


def _mapping_equal(left: Any, right: Any) -> bool:
    if left is None or right is None:
        return left is None and right is None
    return isinstance(left, Mapping) and isinstance(right, Mapping) and canonical_json_bytes(
        left
    ) == canonical_json_bytes(right)


def _require_bool(document: Mapping[str, Any], key: str, label: str) -> bool:
    value = document.get(key)
    if type(value) is not bool:
        raise AcquisitionContractError(f"{label}.{key} must be a JSON boolean")
    return value


def _require_nonnegative_int(
    document: Mapping[str, Any], key: str, label: str
) -> int:
    value = document.get(key)
    if type(value) is not int or value < 0:
        raise AcquisitionContractError(
            f"{label}.{key} must be a non-negative JSON integer"
        )
    return value


def _require_positive_int(
    document: Mapping[str, Any], key: str, label: str
) -> int:
    value = document.get(key)
    if type(value) is not int or value <= 0:
        raise AcquisitionContractError(
            f"{label}.{key} must be a positive JSON integer"
        )
    return value


def _require_sha256(document: Mapping[str, Any], key: str, label: str) -> str:
    value = document.get(key)
    if not isinstance(value, str) or len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise AcquisitionContractError(
            f"{label}.{key} must be a lowercase SHA-256 digest"
        )
    return value


def _require_finite_float(
    document: Mapping[str, Any], key: str, label: str, *, minimum: float | None = None
) -> float:
    value = document.get(key)
    # This validator consumes JSON, whose numeric domain is exactly int/float.
    # Do not let ``float(...)`` launder strings or booleans into authority-
    # relevant timing/confidence values.
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AcquisitionContractError(f"{label}.{key} must be a finite number")
    result = float(value)
    if not np.isfinite(result) or (minimum is not None and result < minimum):
        raise AcquisitionContractError(f"{label}.{key} must be a finite number")
    return result


def _protocol_decode_has_independent_verification(
    _session: Mapping[str, Any],
    _receipt: Mapping[str, Any],
    _protocol_config: AcquisitionProtocolConfig,
) -> bool:
    """Return whether raw-derived protocol decoding has external authority.

    Acquisition-reconstruction v2 stores a self-hashed decoded protocol but
    does not preserve a separately trusted replay receipt that re-derives the
    complete boundary path from the bound raw inputs.  Keep historical stage
    documents readable while making their metric eligibility fail closed.

    A successor generation must replace this boundary with a verifier whose
    trust root is outside the artifact being verified.  A local boolean or a
    second self-hash is not sufficient.
    """

    return False


def _require_unique_string_array(
    document: Mapping[str, Any], key: str, label: str
) -> tuple[str, ...]:
    value = document.get(key)
    if not isinstance(value, list) or any(
        not isinstance(item, str) or not item for item in value
    ):
        raise AcquisitionContractError(
            f"{label}.{key} must be an array of non-empty strings"
        )
    result = tuple(value)
    if len(set(result)) != len(result):
        raise AcquisitionContractError(f"{label}.{key} contains duplicate IDs")
    return result


def _timestamp_sources(session: Mapping[str, Any], session_id: str) -> tuple[str, ...]:
    sensor = session.get("sensor_summary")
    radar = sensor.get("radar") if isinstance(sensor, Mapping) else None
    sources = radar.get("timestamp_sources") if isinstance(radar, Mapping) else None
    if not isinstance(sources, list) or any(not isinstance(item, str) for item in sources):
        raise AcquisitionContractError(
            f"{session_id} lacks explicit per-radar timestamp sources"
        )
    return tuple(sources)


def _validate_timestamp_repair(
    repair: Any, *, session_id: str, view_index: int, frame_count: int
) -> tuple[float, int, int, int]:
    label = f"{session_id}.radar.per_view[{view_index}].timestamp_repair"
    if not isinstance(repair, Mapping):
        raise AcquisitionContractError(f"{label} must be an object")
    plateau_count = _require_nonnegative_int(
        repair, "timestamp_plateau_count", label
    )
    tie_count = _require_nonnegative_int(repair, "measured_tie_edge_count", label)
    reconstructed_count = _require_nonnegative_int(
        repair, "reconstructed_frame_count", label
    )
    maximum = _require_finite_float(
        repair, "maximum_timestamp_correction_s", label, minimum=0.0
    )
    method = repair.get("reconstruction_method")
    plateaus = repair.get("plateaus")
    if not isinstance(plateaus, list) or any(
        not isinstance(item, Mapping) for item in plateaus
    ):
        raise AcquisitionContractError(f"{label}.plateaus must be an object array")
    if plateau_count != len(plateaus):
        raise AcquisitionContractError(f"{label} plateau count mismatch")
    if plateau_count == 0:
        if (
            tie_count != 0
            or reconstructed_count != 0
            or maximum != 0.0
            or method != "none"
        ):
            raise AcquisitionContractError(
                f"{label} zero-plateau evidence is inconsistent"
            )
        return 0.0, 0, 0, 0
    if (
        method != "none_structural_mask_required"
        or tie_count < plateau_count
        or reconstructed_count != 0
        or maximum != 0.0
    ):
        raise AcquisitionContractError(
            f"{label} structural-mask repair evidence is inconsistent"
        )
    nominal_period = _require_finite_float(
        repair, "nominal_positive_period_s", label, minimum=0.0
    )
    if nominal_period <= 0.0:
        raise AcquisitionContractError(
            f"{label}.nominal_positive_period_s must be positive"
        )
    duplicate_edges = 0
    leading_duplicates = 0
    trailing_duplicates = 0
    expected_plateau_keys = {
        "first_affected_frame",
        "last_affected_frame",
        "measured_time_s",
        "duplicate_edge_count",
        "sequence_contiguous",
        "at_leading_boundary",
        "at_trailing_boundary",
    }
    for plateau_index, plateau in enumerate(plateaus):
        assert isinstance(plateau, Mapping)
        plateau_label = f"{label}.plateaus[{plateau_index}]"
        if set(plateau) != expected_plateau_keys:
            raise AcquisitionContractError(
                f"{plateau_label} fields do not match the structural-mask schema"
            )
        first = _require_nonnegative_int(
            plateau, "first_affected_frame", plateau_label
        )
        last = _require_nonnegative_int(
            plateau, "last_affected_frame", plateau_label
        )
        duplicate_count = _require_positive_int(
            plateau, "duplicate_edge_count", plateau_label
        )
        _require_finite_float(plateau, "measured_time_s", plateau_label)
        sequence_contiguous = _require_bool(
            plateau, "sequence_contiguous", plateau_label
        )
        at_leading = _require_bool(
            plateau, "at_leading_boundary", plateau_label
        )
        at_trailing = _require_bool(
            plateau, "at_trailing_boundary", plateau_label
        )
        if (
            last <= first
            or last >= frame_count
            or duplicate_count != last - first
            or at_leading != (first == 0)
            or at_trailing != (last == frame_count - 1)
        ):
            raise AcquisitionContractError(
                f"{plateau_label} affected-frame evidence is inconsistent"
            )
        # Sequence discontinuity is permitted only because its containing
        # interval is independently structurally invalidated.  It must remain
        # an explicit boolean rather than disappearing from the provenance.
        del sequence_contiguous
        duplicate_edges += duplicate_count
        leading_duplicates += duplicate_count if at_leading else 0
        trailing_duplicates += duplicate_count if at_trailing else 0
    if duplicate_edges != tie_count:
        raise AcquisitionContractError(
            f"{label}.measured_tie_edge_count is not derived from plateaus"
        )
    return maximum, plateau_count, leading_duplicates, trailing_duplicates


def _validate_v2_resampling_summary(
    summary: Any,
    *,
    session_id: str,
    label: str,
    timestamp_sources: tuple[str, ...],
) -> tuple[float, bool, int, int]:
    location = f"{session_id}.radar.{label}"
    if not isinstance(summary, Mapping):
        raise AcquisitionContractError(f"{location} must be an object")
    output_rate = _require_finite_float(
        summary, "output_rate_hz", location, minimum=0.0
    )
    interval = _require_finite_float(summary, "interval_s", location, minimum=0.0)
    maximum_gap = _require_finite_float(
        summary, "max_gap_s", location, minimum=0.0
    )
    if (
        summary.get("schema_version")
        != "snn_rr.causal_uniform_radar_resample.v1"
        or _require_bool(summary, "causal", location) is not True
        or summary.get("aggregation") != "half_open_interval_arithmetic_mean"
        or summary.get("timestamp_semantics") != "right_edge_exclusive"
        or summary.get("invalid_value_policy")
        != "exact_zero_with_structural_mask"
        or summary.get("gap_policy") != "mask"
        or output_rate != 10.0
        or not np.isclose(interval, 0.1, rtol=0.0, atol=1e-12)
        or not np.isclose(maximum_gap, 0.050, rtol=0.0, atol=1e-12)
    ):
        raise AcquisitionContractError(f"{location} contract is incompatible")
    time_arithmetic = summary.get("time_arithmetic")
    if not isinstance(time_arithmetic, Mapping):
        raise AcquisitionContractError(f"{location}.time_arithmetic must be an object")
    start_mode = time_arithmetic.get("start_epoch_arithmetic")
    bin_mode = time_arithmetic.get("bin_membership_arithmetic")
    cancellation_avoided = _require_bool(
        time_arithmetic, "start_offset_cancellation_avoided", f"{location}.time_arithmetic"
    )
    half_open_exact = _require_bool(
        time_arithmetic, "half_open_boundary_exact", f"{location}.time_arithmetic"
    )
    start_coordinates_exact = _require_bool(
        time_arithmetic,
        "start_offset_coordinates_exact",
        f"{location}.time_arithmetic",
    )
    start_quantization_error = _require_finite_float(
        time_arithmetic,
        "start_offset_quantization_max_abs_s",
        f"{location}.time_arithmetic",
        minimum=0.0,
    )
    timestamp_quantization_error = _require_finite_float(
        time_arithmetic,
        "timestamp_quantization_max_abs_s",
        f"{location}.time_arithmetic",
        minimum=0.0,
    )
    interval_quantization_error = _require_finite_float(
        time_arithmetic,
        "interval_quantization_max_abs_s",
        f"{location}.time_arithmetic",
        minimum=0.0,
    )
    gap_quantization_error = _require_finite_float(
        time_arithmetic,
        "max_gap_quantization_max_abs_s",
        f"{location}.time_arithmetic",
        minimum=0.0,
    )
    adaptive_arithmetic = _require_bool(
        time_arithmetic,
        "arithmetic_policy_selected_from_timestamp_values",
        f"{location}.time_arithmetic",
    )
    precision = time_arithmetic.get("start_epoch_precision_s")
    ticks = time_arithmetic.get("coordinate_ticks_per_second")
    if start_mode == "integer_millisecond_fixed_point":
        if (
            precision != 0.001
            or cancellation_avoided is not True
            or start_coordinates_exact is not True
            or start_quantization_error != 0.0
        ):
            raise AcquisitionContractError(
                f"{location}.time_arithmetic start-epoch evidence is inconsistent"
            )
    elif start_mode == "float64_centered_then_integer_nanosecond":
        if (
            precision is not None
            or cancellation_avoided is not False
        ):
            raise AcquisitionContractError(
                f"{location}.time_arithmetic start-epoch fallback is inconsistent"
            )
    else:
        raise AcquisitionContractError(
            f"{location}.time_arithmetic start mode is unsupported"
        )
    if (
        bin_mode != "integer_nanosecond_fixed_point"
        or ticks != 1_000_000_000
        or time_arithmetic.get("coordinate_quantization_policy")
        != "round_to_nearest_nanosecond_ties_to_even"
        or adaptive_arithmetic is not False
    ):
        raise AcquisitionContractError(
            f"{location}.time_arithmetic bin evidence is inconsistent"
        )

    content_hashes = summary.get("content_hashes")
    expected_content_hash_keys = {
        "hash_schema_version",
        "corrected_input_values_sha256",
        "aligned_input_time_coordinates_sha256",
        "frame_sequences_sha256",
        "output_times_sha256",
        "output_values_sha256",
        "valid_mask_sha256",
        "sample_counts_sha256",
    }
    if not isinstance(content_hashes, Mapping) or set(content_hashes) != expected_content_hash_keys:
        raise AcquisitionContractError(
            f"{location}.content_hashes fields do not match the transform schema"
        )
    if content_hashes.get("hash_schema_version") != "snn_rr.canonical_ndarray_sha256.v1":
        raise AcquisitionContractError(
            f"{location}.content_hashes hash schema is unsupported"
        )
    for key in expected_content_hash_keys - {"hash_schema_version"}:
        _require_sha256(content_hashes, key, f"{location}.content_hashes")
    transform_hash = _require_sha256(
        summary, "transform_evidence_sha256", location
    )
    observed_transform_hash = hashlib.sha256(
        canonical_json_bytes(content_hashes)
    ).hexdigest()
    if transform_hash != observed_transform_hash:
        raise AcquisitionContractError(
            f"{location}.transform_evidence_sha256 mismatch"
        )
    output_count = _require_positive_int(
        summary, "output_interval_count", location
    )
    all_valid_count = _require_nonnegative_int(
        summary, "all_views_valid_interval_count", location
    )
    any_invalid_count = _require_nonnegative_int(
        summary, "any_view_invalid_interval_count", location
    )
    if (
        all_valid_count > output_count
        or any_invalid_count > output_count
        or all_valid_count + any_invalid_count != output_count
    ):
        raise AcquisitionContractError(f"{location} aggregate validity counts mismatch")
    per_view = summary.get("per_view")
    if not isinstance(per_view, list) or len(per_view) != 3:
        raise AcquisitionContractError(
            f"{location}.per_view must contain exactly three radar views"
        )
    corrections: list[float] = []
    invalid_counts: list[int] = []
    unaccounted_counts: list[int] = []
    plateau_interval_counts: list[int] = []
    timestamp_coordinate_exactness: list[bool] = []
    per_view_quantization_errors: list[float] = []
    frame_accounting_proofs: list[bool] = []
    for view_index, view in enumerate(per_view):
        if not isinstance(view, Mapping):
            raise AcquisitionContractError(
                f"{location}.per_view[{view_index}] must be an object"
            )
        view_label = f"{location}.per_view[{view_index}]"
        if view.get("view_index") != view_index:
            raise AcquisitionContractError(f"{view_label}.view_index mismatch")
        if view.get("timestamp_source") != timestamp_sources[view_index]:
            raise AcquisitionContractError(f"{view_label}.timestamp_source mismatch")
        timestamp_coordinate_exactness.append(
            _require_bool(view, "timestamp_coordinates_exact", view_label)
        )
        per_view_quantization_errors.append(
            _require_finite_float(
                view,
                "timestamp_quantization_max_abs_s",
                view_label,
                minimum=0.0,
            )
        )
        original_count = _require_positive_int(
            view, "original_frame_count", view_label
        )
        frame_count = _require_positive_int(view, "frame_count", view_label)
        leading_trimmed = _require_nonnegative_int(
            view, "leading_boundary_frames_trimmed", view_label
        )
        trailing_trimmed = _require_nonnegative_int(
            view, "trailing_boundary_frames_trimmed", view_label
        )
        if (
            original_count != frame_count
            or leading_trimmed != 0
            or trailing_trimmed != 0
        ):
            raise AcquisitionContractError(
                f"{view_label} must retain every measured payload frame"
            )
        leading_duplicates = _require_nonnegative_int(
            view, "leading_boundary_duplicate_frame_count", view_label
        )
        trailing_duplicates = _require_nonnegative_int(
            view, "trailing_boundary_duplicate_frame_count", view_label
        )
        unaccounted = _require_nonnegative_int(
            view, "unaccounted_payload_frame_count", view_label
        )
        if (
            unaccounted != 0
            or view.get("boundary_plateau_policy")
            != "retain_all_and_structurally_mask_affected_interval"
        ):
            raise AcquisitionContractError(
                f"{view_label} boundary-plateau retention evidence is invalid"
            )
        frame_accounting = view.get("frame_accounting")
        if not isinstance(frame_accounting, Mapping):
            raise AcquisitionContractError(
                f"{view_label}.frame_accounting must be an object"
            )
        if (
            frame_accounting.get("schema_version")
            != "snn_rr.radar_frame_accounting.v1"
            or frame_accounting.get("coordinate_semantics")
            != "half_open_integer_nanosecond"
        ):
            raise AcquisitionContractError(
                f"{view_label}.frame_accounting schema is incompatible"
            )
        retained_count = _require_nonnegative_int(
            frame_accounting,
            "retained_input_frame_count",
            f"{view_label}.frame_accounting",
        )
        categories = frame_accounting.get("categories")
        category_keys = (
            "outside_common_intersection_prefix_frame_count",
            "leading_partial_edge_frame_count",
            "assigned_to_output_intervals_frame_count",
            "trailing_partial_edge_frame_count",
            "outside_common_intersection_suffix_frame_count",
        )
        if not isinstance(categories, Mapping) or set(categories) != set(category_keys):
            raise AcquisitionContractError(
                f"{view_label}.frame_accounting.categories fields are invalid"
            )
        category_counts = {
            key: _require_nonnegative_int(
                categories, key, f"{view_label}.frame_accounting.categories"
            )
            for key in category_keys
        }
        category_sum = _require_nonnegative_int(
            frame_accounting,
            "category_sum",
            f"{view_label}.frame_accounting",
        )
        before_count = _require_nonnegative_int(
            frame_accounting,
            "before_common_complete_support_frame_count",
            f"{view_label}.frame_accounting",
        )
        after_count = _require_nonnegative_int(
            frame_accounting,
            "after_common_complete_support_frame_count",
            f"{view_label}.frame_accounting",
        )
        accounting_residual = _require_nonnegative_int(
            frame_accounting,
            "unaccounted_payload_frame_count",
            f"{view_label}.frame_accounting",
        )
        categories_disjoint = _require_bool(
            frame_accounting,
            "categories_disjoint",
            f"{view_label}.frame_accounting",
        )
        coverage_complete = _require_bool(
            frame_accounting,
            "coverage_complete",
            f"{view_label}.frame_accounting",
        )
        assigned_matches = _require_bool(
            frame_accounting,
            "assigned_count_matches_sample_counts",
            f"{view_label}.frame_accounting",
        )
        derived_category_sum = sum(category_counts.values())
        if (
            retained_count != frame_count
            or category_sum != derived_category_sum
            or before_count
            != category_counts["outside_common_intersection_prefix_frame_count"]
            + category_counts["leading_partial_edge_frame_count"]
            or after_count
            != category_counts["trailing_partial_edge_frame_count"]
            + category_counts["outside_common_intersection_suffix_frame_count"]
            or accounting_residual != retained_count - category_sum
            or unaccounted != accounting_residual
            or coverage_complete != (accounting_residual == 0)
            or not categories_disjoint
            or not assigned_matches
        ):
            raise AcquisitionContractError(
                f"{view_label}.frame_accounting partition/proof mismatch"
            )
        frame_accounting_proofs.append(
            bool(categories_disjoint and coverage_complete and assigned_matches)
        )
        valid_count = _require_nonnegative_int(view, "valid_output_count", view_label)
        invalid_count = _require_nonnegative_int(
            view, "invalid_output_count", view_label
        )
        if valid_count + invalid_count != output_count:
            raise AcquisitionContractError(f"{view_label} output-count mismatch")
        plateau_interval_count = _require_nonnegative_int(
            view, "timestamp_plateau_interval_count", view_label
        )
        if plateau_interval_count > invalid_count:
            raise AcquisitionContractError(
                f"{view_label}.timestamp_plateau_interval_count exceeds invalid output"
            )
        for key in (
            "empty_interval_count",
            "temporal_gap_interval_count",
            "sequence_gap_interval_count",
            "nonfinite_interval_count",
        ):
            component = _require_nonnegative_int(view, key, view_label)
            if component > invalid_count:
                raise AcquisitionContractError(
                    f"{view_label}.{key} exceeds invalid_output_count"
                )
        correction, plateau_count, repair_leading, repair_trailing = (
            _validate_timestamp_repair(
                view.get("timestamp_repair"),
                session_id=session_id,
                view_index=view_index,
                frame_count=frame_count,
            )
        )
        if (
            leading_duplicates != repair_leading
            or trailing_duplicates != repair_trailing
            or (plateau_count == 0 and plateau_interval_count != 0)
        ):
            raise AcquisitionContractError(
                f"{view_label} plateau/boundary aggregate mismatch"
            )
        corrections.append(correction)
        invalid_counts.append(invalid_count)
        unaccounted_counts.append(unaccounted)
        plateau_interval_counts.append(plateau_interval_count)
    if any_invalid_count == 0 and any(invalid_counts):
        raise AcquisitionContractError(f"{location} per-view invalid counts mismatch")
    if any_invalid_count > 0 and not (
        max(invalid_counts, default=0)
        <= any_invalid_count
        <= sum(invalid_counts)
    ):
        raise AcquisitionContractError(f"{location} per-view invalid counts mismatch")
    if not np.isclose(
        timestamp_quantization_error,
        max(per_view_quantization_errors, default=0.0),
        rtol=0.0,
        atol=1e-18,
    ):
        raise AcquisitionContractError(
            f"{location}.time_arithmetic timestamp quantization aggregate mismatch"
        )
    interval_tolerance = max(
        1e-15, 4.0 * float(np.spacing(max(abs(interval), 1.0)))
    )
    gap_tolerance = max(
        1e-15, 4.0 * float(np.spacing(max(abs(maximum_gap), 1.0)))
    )
    derived_half_open_exact = bool(
        start_coordinates_exact
        and all(timestamp_coordinate_exactness)
        and interval_quantization_error <= interval_tolerance
        and gap_quantization_error <= gap_tolerance
    )
    if half_open_exact != derived_half_open_exact:
        raise AcquisitionContractError(
            f"{location}.time_arithmetic half-open exactness mismatch"
        )
    if len(frame_accounting_proofs) != 3:
        raise AcquisitionContractError(f"{location} frame-accounting proof count mismatch")
    return (
        max(corrections, default=0.0),
        half_open_exact,
        sum(unaccounted_counts),
        sum(plateau_interval_counts),
    )


def _v2_radar_metadata_warnings_are_eligible(
    radar: Mapping[str, Any],
    *,
    session_id: str,
    sync_config: Mapping[str, Any],
) -> bool:
    allowlist = sync_config.get("radar_metadata_warning_allowlist")
    if not isinstance(allowlist, Mapping):
        raise AcquisitionContractError(
            "bound synchronization config lacks radar metadata warning authority"
        )
    raw_allowed = allowlist.get(session_id, ())
    if not isinstance(raw_allowed, (list, tuple)) or any(
        not isinstance(value, str) or not value for value in raw_allowed
    ):
        raise AcquisitionContractError(
            f"{session_id} configured radar metadata warning allowlist is invalid"
        )
    allowed = list(raw_allowed)
    expected_policy = {
        "schema": "snn_rr.radar_metadata_warning_policy.v1",
        "mode": "ordered_exact_list_per_session_per_view",
        "config_field": "radar_metadata_warning_allowlist",
        "session_id": session_id,
        "session_allowlist_declared": session_id in allowlist,
        "unlisted_session_policy": "require_no_warnings",
        "allowed_warnings_per_view": allowed,
    }
    policy = radar.get("metadata_warning_policy")
    if canonical_json_bytes(policy) != canonical_json_bytes(expected_policy):
        raise AcquisitionContractError(
            f"{session_id}.radar.metadata_warning_policy/config mismatch"
        )
    views = radar.get("metadata_warning_views")
    if not isinstance(views, list) or len(views) != 3:
        raise AcquisitionContractError(
            f"{session_id}.radar.metadata_warning_views must contain three views"
        )
    recomputed_views: list[dict[str, Any]] = []
    for view_index, view in enumerate(views, start=1):
        if not isinstance(view, Mapping):
            raise AcquisitionContractError(
                f"{session_id}.radar.metadata_warning_views[{view_index - 1}] is invalid"
            )
        warnings = view.get("warnings")
        declared_allowed = view.get("allowed_warnings")
        if not isinstance(warnings, list) or any(
            not isinstance(value, str) or not value for value in warnings
        ):
            raise AcquisitionContractError(
                f"{session_id} radar {view_index} warning evidence is invalid"
            )
        if declared_allowed != allowed:
            raise AcquisitionContractError(
                f"{session_id} radar {view_index} allowed warnings mismatch config"
            )
        exact_match = warnings == allowed
        if (
            view.get("radar_id") != view_index
            or _require_bool(
                view,
                "exact_match",
                f"{session_id}.radar.metadata_warning_views[{view_index - 1}]",
            )
            != exact_match
        ):
            raise AcquisitionContractError(
                f"{session_id} radar {view_index} warning exact-match claim mismatch"
            )
        recomputed_views.append(
            {
                "radar_id": view_index,
                "warnings": list(warnings),
                "allowed_warnings": allowed,
                "exact_match": exact_match,
            }
        )
    expected_eligible = all(view["exact_match"] for view in recomputed_views)
    claimed_eligible = _require_bool(
        radar, "metadata_warnings_eligible", f"{session_id}.radar"
    )
    if claimed_eligible != expected_eligible:
        raise AcquisitionContractError(
            f"{session_id}.radar.metadata_warnings_eligible mismatch"
        )
    evidence = {
        "policy": expected_policy,
        "views": recomputed_views,
        "eligible": expected_eligible,
    }
    evidence_hash = _require_sha256(
        radar, "metadata_warning_evidence_sha256", f"{session_id}.radar"
    )
    if evidence_hash != hashlib.sha256(canonical_json_bytes(evidence)).hexdigest():
        raise AcquisitionContractError(
            f"{session_id}.radar.metadata_warning_evidence_sha256 mismatch"
        )
    return expected_eligible


def _v2_xethru_record_contract_is_eligible(
    radar: Mapping[str, Any], *, session_id: str
) -> bool:
    evidence = radar.get("xethru_record_contract")
    if not isinstance(evidence, Mapping):
        raise AcquisitionContractError(
            f"{session_id}.radar.xethru_record_contract must be an object"
        )
    if (
        evidence.get("schema") != "snn_rr.xethru_record_contract.v1"
        or evidence.get("record_bytes") != XETHRU_RECORD_BYTES
        or evidence.get("payload_bin_count") != RADAR_BINS
    ):
        raise AcquisitionContractError(
            f"{session_id}.radar XeThru record schema is incompatible"
        )
    views = evidence.get("views")
    if not isinstance(views, list) or len(views) != 3:
        raise AcquisitionContractError(
            f"{session_id}.radar XeThru record evidence must contain three views"
        )
    recomputed_views: list[dict[str, Any]] = []
    for radar_id, view in enumerate(views, start=1):
        if not isinstance(view, Mapping) or view.get("radar_id") != radar_id:
            raise AcquisitionContractError(
                f"{session_id} radar {radar_id} record evidence is invalid"
            )
        chunks = view.get("chunks")
        if not isinstance(chunks, list) or not chunks:
            raise AcquisitionContractError(
                f"{session_id} radar {radar_id} record chunks are missing"
            )
        recomputed_chunks: list[dict[str, Any]] = []
        for chunk_index, chunk in enumerate(chunks):
            if not isinstance(chunk, Mapping):
                raise AcquisitionContractError(
                    f"{session_id} radar {radar_id} chunk evidence is invalid"
                )
            filename = chunk.get("filename")
            if not isinstance(filename, str) or not filename:
                raise AcquisitionContractError(
                    f"{session_id} radar {radar_id} chunk filename is invalid"
                )
            numeric = {
                key: _require_nonnegative_int(
                    chunk,
                    key,
                    f"{session_id}.radar{radar_id}.chunks[{chunk_index}]",
                )
                for key in (
                    "bytes",
                    "frame_count",
                    "record_size_remainder_bytes",
                    "zero_header_nonzero",
                    "bin_count_invalid",
                )
            }
            if (
                chunk.get("chunk_index") != chunk_index
                or chunk.get("record_bytes") != XETHRU_RECORD_BYTES
                or chunk.get("payload_bin_count") != RADAR_BINS
                or numeric["frame_count"] <= 0
                or numeric["bytes"]
                != numeric["frame_count"] * XETHRU_RECORD_BYTES
            ):
                raise AcquisitionContractError(
                    f"{session_id} radar {radar_id} chunk record geometry mismatch"
                )
            recomputed_chunks.append(dict(chunk))
        eligible = all(
            chunk["record_size_remainder_bytes"] == 0
            and chunk["zero_header_nonzero"] == 0
            and chunk["bin_count_invalid"] == 0
            for chunk in recomputed_chunks
        )
        if _require_bool(
            view, "eligible", f"{session_id}.radar{radar_id}.record_contract"
        ) != eligible:
            raise AcquisitionContractError(
                f"{session_id} radar {radar_id} record eligibility mismatch"
            )
        recomputed_views.append(
            {"radar_id": radar_id, "chunks": recomputed_chunks, "eligible": eligible}
        )
    eligible = all(view["eligible"] for view in recomputed_views)
    if _require_bool(evidence, "eligible", f"{session_id}.radar.record_contract") != eligible:
        raise AcquisitionContractError(
            f"{session_id}.radar XeThru record eligibility mismatch"
        )
    recomputed = {
        "schema": "snn_rr.xethru_record_contract.v1",
        "record_bytes": XETHRU_RECORD_BYTES,
        "payload_bin_count": RADAR_BINS,
        "views": recomputed_views,
        "eligible": eligible,
    }
    evidence_hash = _require_sha256(
        radar,
        "xethru_record_contract_evidence_sha256",
        f"{session_id}.radar",
    )
    if evidence_hash != hashlib.sha256(canonical_json_bytes(recomputed)).hexdigest():
        raise AcquisitionContractError(
            f"{session_id}.radar XeThru record evidence hash mismatch"
        )
    return eligible


def _v2_measured_timing_is_eligible(
    session: Mapping[str, Any],
    session_id: str,
    *,
    sync_config: Mapping[str, Any],
) -> bool:
    sensor = session.get("sensor_summary")
    radar = sensor.get("radar") if isinstance(sensor, Mapping) else None
    if not isinstance(radar, Mapping):
        raise AcquisitionContractError(f"{session_id} lacks radar timing evidence")
    warning_evidence_eligible = _v2_radar_metadata_warnings_are_eligible(
        radar,
        session_id=session_id,
        sync_config=sync_config,
    )
    record_contract_eligible = _v2_xethru_record_contract_is_eligible(
        radar, session_id=session_id
    )
    sources = _timestamp_sources(session, session_id)
    if len(sources) != 3:
        raise AcquisitionContractError(
            f"{session_id}.radar.timestamp_sources must contain exactly three views"
        )
    sync_resampling = radar.get("sync_marker_resampling")
    feature_resampling = radar.get("feature_resampling")
    if canonical_json_bytes(sync_resampling) != canonical_json_bytes(
        feature_resampling
    ):
        raise AcquisitionContractError(
            f"{session_id} sync/feature resampling evidence diverges"
        )
    (
        sync_correction,
        sync_half_open_exact,
        sync_unaccounted,
        sync_plateau_intervals,
    ) = _validate_v2_resampling_summary(
        sync_resampling,
        session_id=session_id,
        label="sync_marker_resampling",
        timestamp_sources=sources,
    )
    (
        feature_correction,
        feature_half_open_exact,
        feature_unaccounted,
        feature_plateau_intervals,
    ) = _validate_v2_resampling_summary(
        feature_resampling,
        session_id=session_id,
        label="feature_resampling",
        timestamp_sources=sources,
    )
    correction_value = _require_finite_float(
        radar,
        "maximum_timestamp_correction_s",
        f"{session_id}.radar",
        minimum=0.0,
    )
    derived_correction = max(sync_correction, feature_correction)
    if not np.isclose(
        correction_value, derived_correction, rtol=0.0, atol=1e-12
    ):
        raise AcquisitionContractError(
            f"{session_id}.radar.maximum_timestamp_correction_s is not derived "
            "from per_view timestamp_repair evidence"
        )
    resampling_content_hashes = radar.get("resampling_content_hashes")
    if canonical_json_bytes(resampling_content_hashes) != canonical_json_bytes(
        feature_resampling.get("content_hashes")
    ):
        raise AcquisitionContractError(
            f"{session_id}.radar.resampling_content_hashes mismatch"
        )
    if radar.get("resampling_transform_evidence_sha256") != feature_resampling.get(
        "transform_evidence_sha256"
    ):
        raise AcquisitionContractError(
            f"{session_id}.radar.resampling_transform_evidence_sha256 mismatch"
        )
    if sync_unaccounted != feature_unaccounted or (
        sync_plateau_intervals != feature_plateau_intervals
    ):
        raise AcquisitionContractError(
            f"{session_id}.radar sync/feature timing aggregates diverge"
        )
    if _require_nonnegative_int(
        radar, "unaccounted_payload_frame_count", f"{session_id}.radar"
    ) != feature_unaccounted:
        raise AcquisitionContractError(
            f"{session_id}.radar.unaccounted_payload_frame_count aggregate mismatch"
        )
    if _require_nonnegative_int(
        radar, "timestamp_plateau_interval_count", f"{session_id}.radar"
    ) != feature_plateau_intervals:
        raise AcquisitionContractError(
            f"{session_id}.radar.timestamp_plateau_interval_count aggregate mismatch"
        )
    expected = bool(
        all(source == "meta_v13" for source in sources)
        and sync_resampling["any_view_invalid_interval_count"] == 0
        and sync_half_open_exact
        and feature_half_open_exact
        and correction_value <= 0.050
        and radar["unaccounted_payload_frame_count"] == 0
        and warning_evidence_eligible
        and record_contract_eligible
    )
    claimed = _require_bool(
        radar, "measured_timing_eligible", f"{session_id}.radar"
    )
    if claimed != expected:
        raise AcquisitionContractError(
            f"{session_id}.radar.measured_timing_eligible does not match evidence"
        )
    return expected


def load_acquisition_reconstruction(
    manifest_path: str | Path,
) -> AcquisitionReconstruction:
    """Read every session contract and verify the full content-addressed graph."""

    path = Path(manifest_path).resolve()
    root = path.parent
    document = _read_strict_json(path)
    schema = document.get("schema_version")
    if schema not in SUPPORTED_ACQUISITION_SCHEMAS:
        raise AcquisitionContractError("unexpected acquisition reconstruction schema")
    is_v2 = schema == ACQUISITION_SCHEMA
    _require_content_hash(document, "acquisition root manifest")
    entries = document.get("sessions")
    if not isinstance(entries, list):
        raise AcquisitionContractError("acquisition root sessions must be an array")
    session_ids = [entry.get("session_id") for entry in entries if isinstance(entry, Mapping)]
    if len(session_ids) != len(entries) or any(
        not isinstance(session_id, str) or not session_id for session_id in session_ids
    ):
        raise AcquisitionContractError("acquisition root contains an invalid session entry")
    if len(set(session_ids)) != len(session_ids):
        raise AcquisitionContractError("acquisition root contains duplicate session IDs")

    if is_v2:
        expected_ids = _require_unique_string_array(
            document, "expected_session_ids", "acquisition root"
        )
        expected_usable_ids = _require_unique_string_array(
            document, "expected_usable_session_ids", "acquisition root"
        )
        selected_ids = _require_unique_string_array(
            document, "selected_session_ids", "acquisition root"
        )
        if not expected_ids:
            raise AcquisitionContractError(
                "acquisition root.expected_session_ids must not be empty"
            )
        if not set(selected_ids) <= set(expected_ids):
            raise AcquisitionContractError(
                "acquisition root selected IDs are outside the dataset authority"
            )
        selected_set = set(selected_ids)
        if selected_ids != tuple(
            session_id for session_id in expected_ids if session_id in selected_set
        ):
            raise AcquisitionContractError(
                "acquisition root selected IDs are not in dataset-authority order"
            )
        for key, values in (
            ("expected_session_ids_sha256", expected_ids),
            ("expected_usable_session_ids_sha256", expected_usable_ids),
            ("selected_session_ids_sha256", selected_ids),
        ):
            claimed = document.get(key)
            observed = hashlib.sha256(
                canonical_json_bytes(list(values))
            ).hexdigest()
            if claimed != observed:
                raise AcquisitionContractError(f"acquisition root {key} mismatch")
        cohort_authority_value = document.get("cohort_authority")
        if not isinstance(cohort_authority_value, str) or not cohort_authority_value:
            raise AcquisitionContractError(
                "acquisition root.cohort_authority must bind the authority artifact"
            )
        cohort_authority_path = Path(cohort_authority_value)
        if not cohort_authority_path.is_absolute():
            cohort_authority_path = (root / cohort_authority_path).resolve()
        if not cohort_authority_path.is_file():
            raise AcquisitionContractError(
                f"bound cohort authority is missing: {cohort_authority_path}"
            )
        _require_sha256(
            document, "cohort_authority_sha256", "acquisition root"
        )
        cohort_authority = load_acquisition_cohort_authority(
            cohort_authority_path,
            expected_sha256=str(document["cohort_authority_sha256"]),
        )
        if document.get("cohort_authority_schema") != (
            ACQUISITION_COHORT_AUTHORITY_SCHEMA
        ):
            raise AcquisitionContractError(
                "acquisition root cohort authority schema mismatch"
            )
        if document.get("cohort_authority_content_sha256") != (
            cohort_authority.content_sha256
        ):
            raise AcquisitionContractError(
                "acquisition root cohort authority content hash mismatch"
            )
        if expected_ids != cohort_authority.expected_session_ids:
            raise AcquisitionContractError(
                "acquisition root expected session IDs differ from cohort authority"
            )
        if expected_usable_ids != cohort_authority.expected_usable_session_ids:
            raise AcquisitionContractError(
                "acquisition root expected usable IDs differ from cohort authority"
            )
        cohort_claim_arrays = (
            (
                "excluded_sessions",
                [
                    {"session_id": session_id, "reason": reason}
                    for session_id, reason in cohort_authority.excluded_sessions
                ],
            ),
            (
                "session_identities",
                [
                    {
                        "session_id": session_id,
                        "physical_identity": identity,
                    }
                    for session_id, identity in cohort_authority.session_identities
                ],
            ),
            (
                "expected_physical_identities",
                list(cohort_authority.expected_physical_identities),
            ),
        )
        for key, authoritative in cohort_claim_arrays:
            claimed = document.get(key)
            if canonical_json_bytes(claimed) != canonical_json_bytes(
                authoritative
            ):
                raise AcquisitionContractError(
                    f"acquisition root {key} differs from cohort authority"
                )
            digest_key = f"{key}_sha256"
            _require_sha256(document, digest_key, "acquisition root")
            if document[digest_key] != hashlib.sha256(
                canonical_json_bytes(authoritative)
            ).hexdigest():
                raise AcquisitionContractError(
                    f"acquisition root {digest_key} mismatch"
                )
        if tuple(session_ids) != selected_ids:
            raise AcquisitionContractError(
                "acquisition root selected IDs do not match its session catalogue"
            )
        if not set(expected_usable_ids) <= set(expected_ids):
            raise AcquisitionContractError(
                "acquisition root usable IDs are not a subset of expected IDs"
            )
        expected_usable_set = set(expected_usable_ids)
        if expected_usable_ids != tuple(
            session_id
            for session_id in expected_ids
            if session_id in expected_usable_set
        ):
            raise AcquisitionContractError(
                "acquisition root usable IDs are not in dataset-authority order"
            )
        _require_sha256(
            document, "dataset_manifest_sha256", "acquisition root"
        )
        for key in (
            "pipeline_sha256",
            "sync_config_sha256",
            "protocol_config_sha256",
            "spreadsheet_sha256",
        ):
            _require_sha256(document, key, "acquisition root")
        sync_config_value = document.get("sync_config")
        if not isinstance(sync_config_value, str) or not sync_config_value:
            raise AcquisitionContractError(
                "acquisition root.sync_config must bind the configuration artifact"
            )
        sync_config_path = Path(sync_config_value)
        if not sync_config_path.is_absolute():
            sync_config_path = (root / sync_config_path).resolve()
        if not sync_config_path.is_file():
            raise AcquisitionContractError(
                f"bound synchronization config is missing: {sync_config_path}"
            )
        try:
            sync_config_payload, _ = _source_snapshot(
                sync_config_path,
                label="synchronization configuration",
                expected_sha256=str(document["sync_config_sha256"]),
            )
            root_sync_config = SynchronizationConfig.from_mapping(
                yaml.safe_load(sync_config_payload.decode("utf-8"))
            )
        except (UnicodeError, ValueError, yaml.YAMLError) as error:
            raise AcquisitionContractError(
                f"bound synchronization config is invalid: {error}"
            ) from error
        root_sync_config_document = root_sync_config.to_dict()
        protocol_config_path, protocol_config_hash = _resolve_bound_external_file(
            document,
            reconstruction_root=root,
            path_key="protocol_config",
            hash_key="protocol_config_sha256",
            label="protocol configuration",
        )
        spreadsheet_path, spreadsheet_hash = _resolve_bound_external_file(
            document,
            reconstruction_root=root,
            path_key="spreadsheet",
            hash_key="spreadsheet_sha256",
            label="dataset issue spreadsheet",
        )
        try:
            root_protocol_config = load_protocol_config(
                protocol_config_path, expected_sha256=protocol_config_hash
            )
        except ValueError as error:
            raise AcquisitionContractError(
                f"bound protocol configuration is invalid: {error}"
            ) from error
        dataset_root_value = document.get("dataset_root")
        if not isinstance(dataset_root_value, str) or not dataset_root_value:
            raise AcquisitionContractError(
                "acquisition root.dataset_root must bind the discovered dataset"
            )
        dataset_root = Path(dataset_root_value).resolve()
        if not dataset_root.is_dir():
            raise AcquisitionContractError(
                f"bound dataset root is missing: {dataset_root}"
            )
        try:
            issue_records = load_dataset_issue_records(
                spreadsheet_path,
                dataset_root=dataset_root,
                config=root_protocol_config,
                expected_sha256=spreadsheet_hash,
            )
            discovered_dataset = build_dataset_manifest(dataset_root)
        except (OSError, ValueError, BadZipFile) as error:
            raise AcquisitionContractError(
                f"cannot re-derive the bound dataset authority: {error}"
            ) from error
        if _sha256_file(protocol_config_path) != protocol_config_hash:
            raise AcquisitionContractError(
                "bound protocol configuration changed while being parsed"
            )
        if _sha256_file(spreadsheet_path) != spreadsheet_hash:
            raise AcquisitionContractError(
                "bound dataset issue spreadsheet changed while being parsed"
            )
        discovered_ids = tuple(
            item.subject_id for item in discovered_dataset.subjects
        )
        discovered_usable_ids = tuple(
            item.subject_id
            for item in discovered_dataset.subjects
            if item.usable
        )
        if discovered_ids != cohort_authority.expected_session_ids:
            raise AcquisitionContractError(
                "discovered dataset sessions differ from cohort authority"
            )
        if discovered_usable_ids != cohort_authority.expected_usable_session_ids:
            raise AcquisitionContractError(
                "discovered dataset usability differs from cohort authority"
            )
        discovered_assignments = tuple(
            (session_id, identity_for_session(session_id))
            for session_id in discovered_ids
        )
        if discovered_assignments != cohort_authority.session_identities:
            raise AcquisitionContractError(
                "discovered dataset identities differ from cohort authority"
            )
        dataset_manifest_digest = hashlib.sha256(
            canonical_json_bytes(discovered_dataset.to_dict())
        ).hexdigest()
        if document.get("dataset_manifest_sha256") != dataset_manifest_digest:
            raise AcquisitionContractError(
                "acquisition root dataset manifest digest mismatch"
            )
        issue_record_ids = tuple(record.session_id for record in issue_records)
        if issue_record_ids != cohort_authority.expected_session_ids:
            raise AcquisitionContractError(
                "dataset issue spreadsheet sessions differ from cohort authority"
            )
        discovered_subjects_by_id = {
            item.subject_id: item for item in discovered_dataset.subjects
        }
        if _require_nonnegative_int(
            document, "dataset_session_count", "acquisition root"
        ) != len(expected_ids):
            raise AcquisitionContractError("acquisition root dataset session count mismatch")
        if _require_nonnegative_int(
            document, "dataset_usable_session_count", "acquisition root"
        ) != len(expected_usable_ids):
            raise AcquisitionContractError(
                "acquisition root dataset usable-session count mismatch"
            )
        if _require_nonnegative_int(
            document, "dataset_physical_identity_count", "acquisition root"
        ) != len(cohort_authority.expected_physical_identities):
            raise AcquisitionContractError(
                "acquisition root dataset physical-identity count mismatch"
            )
        if _require_nonnegative_int(
            document, "selected_session_count", "acquisition root"
        ) != len(selected_ids):
            raise AcquisitionContractError("acquisition root selected-session count mismatch")
        scope = document.get("selection_scope")
        if scope not in {"full_cohort", "diagnostic_subset"}:
            raise AcquisitionContractError("acquisition root selection_scope is invalid")
        subjects_filter_applied = _require_bool(
            document, "subjects_filter_applied", "acquisition root"
        )
        expected_scope = (
            "full_cohort"
            if not subjects_filter_applied and selected_ids == expected_ids
            else "diagnostic_subset"
        )
        if scope != expected_scope:
            raise AcquisitionContractError("acquisition root selection_scope mismatch")
        execution_complete = _require_bool(
            document, "execution_complete", "acquisition root"
        )
        full_cohort_complete = _require_bool(
            document, "full_cohort_complete", "acquisition root"
        )
        expected_full_complete = bool(
            execution_complete
            and not subjects_filter_applied
            and scope == "full_cohort"
            and selected_ids == expected_ids
        )
        if full_cohort_complete != expected_full_complete:
            raise AcquisitionContractError(
                "acquisition root full-cohort completion statement mismatch"
            )
        if _require_bool(document, "complete", "acquisition root") != full_cohort_complete:
            raise AcquisitionContractError(
                "acquisition root complete must mean full-cohort completion"
            )
    else:
        expected_ids = tuple(session_ids)
        expected_usable_ids = ()
        selected_ids = tuple(session_ids)
        scope = "legacy_unverified"
        execution_complete = document.get("complete") is True
        full_cohort_complete = False
        root_sync_config_document = None
        cohort_authority = None
        subjects_filter_applied = False

    sessions: dict[str, AcquisitionSessionContract] = {}
    observed_usable_ids: list[str] = []
    # Keep separately recorded historical claims so old self-hashed v2
    # artifacts remain inspectable.  Effective authority is derived below and
    # is necessarily stricter in the absence of an independent raw replay.
    declared_authorized_count = 0
    declared_eligible_count = 0
    authorized_count = 0
    eligible_count = 0
    for entry in entries:
        assert isinstance(entry, Mapping)
        session_id = str(entry["session_id"])
        session_path = _resolve_inside(root, entry.get("manifest"), "session manifest path")
        session, session_file_sha256 = _read_strict_json_snapshot(session_path)
        if session.get("schema_version") != schema:
            raise AcquisitionContractError(f"{session_id} has an incompatible session schema")
        if session.get("session_id") != session_id:
            raise AcquisitionContractError(f"{session_id} session manifest ID mismatch")
        session_hash = _require_content_hash(session, f"{session_id} session manifest")
        if entry.get("content_sha256") != session_hash:
            raise AcquisitionContractError(f"{session_id} root/session hash mismatch")
        if is_v2 and entry.get("manifest_sha256") != session_file_sha256:
            raise AcquisitionContractError(f"{session_id} root/session file hash mismatch")
        if is_v2:
            context = session.get("reconstruction_context")
            if not isinstance(context, Mapping):
                raise AcquisitionContractError(
                    f"{session_id} lacks a reconstruction context binding"
                )
            for key in (
                "pipeline_sha256",
                "sync_config_sha256",
                "protocol_config_sha256",
                "spreadsheet_sha256",
                "cohort_authority_sha256",
                "cohort_authority_content_sha256",
            ):
                _require_sha256(
                    context, key, f"{session_id}.reconstruction_context"
                )
                if context.get(key) != document.get(key):
                    raise AcquisitionContractError(
                        f"{session_id} reconstruction context {key} mismatch"
                    )
            if _require_bool(
                context,
                "subjects_filter_applied",
                f"{session_id}.reconstruction_context",
            ) != subjects_filter_applied:
                raise AcquisitionContractError(
                    f"{session_id} reconstruction context subjects_filter_applied mismatch"
                )
            _require_bool(
                context, "build_range_tracks", f"{session_id}.reconstruction_context"
            )
            _require_positive_int(
                context,
                "layout_maximum_frames",
                f"{session_id}.reconstruction_context",
            )
        usable = session.get("usable")
        if type(usable) is not bool:
            raise AcquisitionContractError(f"{session_id}.usable must be a JSON boolean")
        if is_v2 and entry.get("usable") is not usable:
            raise AcquisitionContractError(f"{session_id} root/session usability mismatch")
        if is_v2:
            assert cohort_authority is not None
            expected_identity = cohort_authority.session_identity_map[session_id]
            if session.get("physical_identity") != expected_identity:
                raise AcquisitionContractError(
                    f"{session_id} physical identity differs from cohort authority"
                )
            if entry.get("physical_identity") != expected_identity:
                raise AcquisitionContractError(
                    f"{session_id} root/session physical identity mismatch"
                )
        if not usable:
            if is_v2:
                eligibility = session.get("eligibility")
                if not isinstance(eligibility, Mapping):
                    raise AcquisitionContractError(
                        f"{session_id} lacks explicit eligibility components"
                    )
                for key in (
                    "measured_timing_eligible",
                    "alignment_eligible",
                    "stage_metric_eligible",
                    "range_feature_eligible",
                    "strict_cache_eligible",
                ):
                    if _require_bool(eligibility, key, f"{session_id}.eligibility"):
                        raise AcquisitionContractError(
                            f"{session_id} unusable session claims {key}"
                        )
                if _require_bool(session, "scientific_eligible", session_id):
                    raise AcquisitionContractError(
                        f"{session_id} unusable session claims scientific eligibility"
                    )
                if entry.get("scientific_eligible") is not False:
                    raise AcquisitionContractError(
                        f"{session_id} unusable root entry claims eligibility"
                    )
            continue
        observed_usable_ids.append(session_id)

        sync = session.get("synchronization")
        if not isinstance(sync, Mapping):
            raise AcquisitionContractError(f"{session_id} lacks synchronization metadata")
        receipt_path = _resolve_inside(root, sync.get("receipt"), "sync receipt path")
        receipt_document, receipt_file_sha256 = _read_strict_json_snapshot(
            receipt_path
        )
        receipt = validate_sync_receipt(receipt_document)
        if receipt.get("session_id") != session_id:
            raise AcquisitionContractError(f"{session_id} sync receipt ID mismatch")
        if sync.get("receipt_sha256") != receipt_file_sha256:
            raise AcquisitionContractError(f"{session_id} sync receipt file hash mismatch")
        if sync.get("receipt_content_sha256") != receipt.get("content_sha256"):
            raise AcquisitionContractError(f"{session_id} sync receipt content hash mismatch")
        if is_v2:
            receipt_algorithm = receipt.get("algorithm")
            receipt_config = (
                receipt_algorithm.get("config")
                if isinstance(receipt_algorithm, Mapping)
                else None
            )
            if canonical_json_bytes(receipt_config) != canonical_json_bytes(
                root_sync_config_document
            ):
                raise AcquisitionContractError(
                    f"{session_id} sync receipt/root configuration mismatch"
                )
        if is_v2:
            raw_bindings = session.get("raw_input_bindings")
            if not isinstance(raw_bindings, Mapping) or not raw_bindings:
                raise AcquisitionContractError(
                    f"{session_id}.raw_input_bindings must be a non-empty object"
                )
            raw_bindings_hash = _require_sha256(
                session, "raw_input_bindings_sha256", session_id
            )
            if hashlib.sha256(canonical_json_bytes(raw_bindings)).hexdigest() != (
                raw_bindings_hash
            ):
                raise AcquisitionContractError(
                    f"{session_id}.raw_input_bindings_sha256 mismatch"
                )
            if canonical_json_bytes(raw_bindings) != canonical_json_bytes(
                receipt.get("input_bindings")
            ):
                raise AcquisitionContractError(
                    f"{session_id} raw input/synchronization receipt bindings mismatch"
                )
            _validate_v2_raw_input_graph(
                session,
                receipt,
                subject=discovered_subjects_by_id[session_id],
                dataset_root=dataset_root,
                session_id=session_id,
            )
        receipt_mapping = receipt.get("result", {}).get("mapping")
        if not _mapping_equal(sync.get("mapping"), receipt_mapping):
            raise AcquisitionContractError(f"{session_id} duplicated sync mapping mismatch")
        if receipt_mapping is not None and not isinstance(receipt_mapping, Mapping):
            raise AcquisitionContractError(
                f"{session_id} synchronization mapping is malformed"
            )
        mapping = (
            None
            if receipt_mapping is None
            else TimeMapping.from_dict(receipt_mapping)
        )

        approval_path: Path | None = None
        approval: Mapping[str, Any] | None = None
        stated_approval_path = sync.get("manual_approval")
        if is_v2:
            approval_present = _require_bool(
                sync, "manual_approval_present", f"{session_id}.synchronization"
            )
            if approval_present != (stated_approval_path is not None):
                raise AcquisitionContractError(
                    f"{session_id} manual_approval_present/artifact mismatch"
                )
            if not approval_present and any(
                sync.get(key) is not None
                for key in (
                    "manual_approval",
                    "manual_approval_file_sha256",
                    "manual_approval_content_sha256",
                )
            ):
                raise AcquisitionContractError(
                    f"{session_id} absent manual approval has non-null bindings"
                )
        if stated_approval_path is not None:
            approval_path = _resolve_inside(
                root, stated_approval_path, "manual approval path"
            )
            approval, approval_file_sha256 = _read_strict_json_snapshot(
                approval_path
            )
            approval = validate_manual_approval(approval, receipt)
            if is_v2:
                _require_sha256(
                    sync,
                    "manual_approval_file_sha256",
                    f"{session_id}.synchronization",
                )
                _require_sha256(
                    sync,
                    "manual_approval_content_sha256",
                    f"{session_id}.synchronization",
                )
            if sync.get("manual_approval_file_sha256") != approval_file_sha256:
                raise AcquisitionContractError(
                    f"{session_id} manual approval file hash mismatch"
                )
            if sync.get("manual_approval_content_sha256") != approval.get(
                "content_sha256"
            ):
                raise AcquisitionContractError(
                    f"{session_id} manual approval content hash mismatch"
                )
        elif sync.get("manual_approval_present") is True:
            raise AcquisitionContractError(
                f"{session_id} states a manual approval but does not bind its artifact"
            )
        declared_authorized = _require_bool(
            sync, "authorized", f"{session_id}.synchronization"
        )
        receipt_result = receipt.get("result")
        assert isinstance(receipt_result, Mapping)
        receipt_decision = receipt_result.get("decision")
        legacy_decision_supports_claim = bool(
            (
                receipt_decision == "accepted"
                or (
                    receipt_decision == "manual_review_required"
                    and approval is not None
                    and approval.get("decision") == "approve"
                )
            )
            and (approval is None or approval.get("decision") != "reject")
        )
        if declared_authorized and not legacy_decision_supports_claim:
            raise AcquisitionContractError(
                f"{session_id} historical authorization exceeds its receipt"
            )
        # A stored v2 `authorized=true` value is preserved as historical
        # evidence only.  Scientific consumers require the independent
        # raw-derived verifier represented by synchronization_is_authorized;
        # current v1 receipts deliberately return False there.
        authorized = bool(
            declared_authorized
            and synchronization_is_authorized(
                receipt, manual_approval=approval
            )
        )
        declared_authorized_count += int(declared_authorized)
        authorized_count += int(authorized)

        protocol = session.get("protocol")
        protocol_contract = session.get("protocol_contract")
        if not isinstance(protocol, Mapping) or not isinstance(protocol_contract, Mapping):
            raise AcquisitionContractError(f"{session_id} lacks a protocol contract")
        if protocol.get("annotation_inference_feature_allowed") is not False:
            raise AcquisitionContractError(
                f"{session_id} protocol annotations are not inference-forbidden"
            )
        assignment = protocol_contract.get("window_assignment")
        if (
            protocol_contract.get("time_basis") != "seconds_from_biopac_start"
            or not isinstance(assignment, Mapping)
        ):
            raise AcquisitionContractError(f"{session_id} protocol time basis is invalid")
        minimum_overlap = _require_finite_float(
            assignment,
            "minimum_overlap_fraction",
            f"{session_id}.protocol_contract.window_assignment",
            minimum=0.0,
        )
        transition_guard = _require_finite_float(
            assignment,
            "transition_guard_s",
            f"{session_id}.protocol_contract.window_assignment",
            minimum=0.0,
        )
        if not 0 < minimum_overlap <= 1 or transition_guard < 0:
            raise AcquisitionContractError(f"{session_id} window assignment is invalid")
        protocol_status = protocol.get("status")
        if protocol_status not in {"auto", "uncertain", "review"}:
            raise AcquisitionContractError(f"{session_id} protocol status is invalid")
        protocol_stages = protocol.get("stages")
        if not isinstance(protocol_stages, list) or any(
            not isinstance(stage, Mapping)
            or stage.get("status") not in {"auto", "uncertain", "review"}
            for stage in protocol_stages
        ):
            raise AcquisitionContractError(f"{session_id} protocol stages are invalid")
        if is_v2:
            if protocol.get("session_id") != session_id:
                raise AcquisitionContractError(
                    f"{session_id} protocol session_id mismatch"
                )
            if protocol.get("annotation_schema_version") != protocol_contract.get(
                "schema_version"
            ):
                raise AcquisitionContractError(
                    f"{session_id} protocol annotation schema mismatch"
                )
            if (
                protocol_contract.get("schema_version")
                != root_protocol_config.schema_version
                or protocol_contract.get("time_basis")
                != root_protocol_config.time_basis
                or not np.isclose(
                    minimum_overlap,
                    root_protocol_config.window_assignment.minimum_overlap_fraction,
                    rtol=0.0,
                    atol=0.0,
                )
                or not np.isclose(
                    transition_guard,
                    root_protocol_config.window_assignment.transition_guard_s,
                    rtol=0.0,
                    atol=0.0,
                )
            ):
                raise AcquisitionContractError(
                    f"{session_id} protocol contract/root configuration mismatch"
                )
            duration = _require_finite_float(
                protocol, "duration_s", f"{session_id}.protocol", minimum=0.0
            )
            if duration <= 0:
                raise AcquisitionContractError(
                    f"{session_id}.protocol.duration_s must be positive"
                )
            protocol_confidence = _require_finite_float(
                protocol, "confidence", f"{session_id}.protocol", minimum=0.0
            )
            if protocol_confidence > 1.0:
                raise AcquisitionContractError(
                    f"{session_id}.protocol.confidence exceeds one"
                )
            if len(protocol_stages) != len(root_protocol_config.stages):
                raise AcquisitionContractError(
                    f"{session_id}.protocol must contain the exact ordered seven phases"
                )
            prior_end = 0.0
            stage_ids: list[str] = []
            stage_statuses: list[str] = []
            stage_confidences: list[float] = []
            for stage_index, stage in enumerate(protocol_stages):
                assert isinstance(stage, Mapping)
                stage_label = f"{session_id}.protocol.stages[{stage_index}]"
                stage_id = stage.get("stage_id")
                if not isinstance(stage_id, str) or not stage_id or stage_id in stage_ids:
                    raise AcquisitionContractError(
                        f"{stage_label}.stage_id is missing or duplicated"
                    )
                stage_ids.append(stage_id)
                expected_stage = root_protocol_config.stages[stage_index]
                if (
                    stage_id != expected_stage.stage_id
                    or stage.get("name") != expected_stage.name
                ):
                    raise AcquisitionContractError(
                        f"{stage_label} differs from the ordered protocol configuration"
                    )
                start_document = stage.get("start")
                end_document = stage.get("end")
                if not isinstance(start_document, Mapping) or not isinstance(
                    end_document, Mapping
                ):
                    raise AcquisitionContractError(
                        f"{stage_label} lacks explicit start/end time_s"
                    )
                stage_start = _require_finite_float(
                    start_document, "time_s", f"{stage_label}.start", minimum=0.0
                )
                stage_end = _require_finite_float(
                    end_document, "time_s", f"{stage_label}.end", minimum=0.0
                )
                stage_duration = _require_finite_float(
                    stage, "duration_s", stage_label, minimum=0.0
                )
                stage_confidence = _require_finite_float(
                    stage, "confidence", stage_label, minimum=0.0
                )
                stage_statuses.append(str(stage.get("status")))
                stage_confidences.append(stage_confidence)
                if (
                    stage_start < prior_end
                    or stage_end <= stage_start
                    or stage_end > duration
                    or stage_confidence > 1.0
                    or not np.isclose(
                        stage_duration,
                        stage_end - stage_start,
                        rtol=0.0,
                        atol=1e-9,
                    )
                ):
                    raise AcquisitionContractError(
                        f"{stage_label} timing/confidence evidence is invalid"
                    )
                prior_end = stage_end
            expected_stage_ids = [
                stage.stage_id for stage in root_protocol_config.stages
            ]
            if stage_ids != expected_stage_ids:
                raise AcquisitionContractError(
                    f"{session_id}.protocol must contain the exact ordered seven phases"
                )
            status_rank = {"auto": 0, "uncertain": 1, "review": 2}
            derived_status = max(stage_statuses, key=lambda item: status_rank[item])
            if protocol_status != derived_status:
                raise AcquisitionContractError(
                    f"{session_id}.protocol overall status is not derived from its stages"
                )
            derived_confidence = float(np.mean(stage_confidences))
            if not np.isclose(
                protocol_confidence, derived_confidence, rtol=0.0, atol=1e-12
            ):
                raise AcquisitionContractError(
                    f"{session_id}.protocol confidence is not the stage mean"
                )

        range_path: Path | None = None
        range_document = session.get("range_tracking")
        if isinstance(range_document, Mapping) and range_document.get("status") == "built":
            range_path = _resolve_inside(
                session_path.parent,
                range_document.get("artifact"),
                "range-track artifact path",
            )
            if range_document.get("artifact_sha256") != _sha256_file(range_path):
                raise AcquisitionContractError(f"{session_id} range-track hash mismatch")

        sources = _timestamp_sources(session, session_id)
        biopac_warning_free = True
        if is_v2:
            sensor_summary = session.get("sensor_summary")
            biopac_summary = (
                sensor_summary.get("biopac")
                if isinstance(sensor_summary, Mapping)
                else None
            )
            if not isinstance(biopac_summary, Mapping):
                raise AcquisitionContractError(
                    f"{session_id}.sensor_summary.biopac must be an object"
                )
            warnings = biopac_summary.get("warnings")
            if not isinstance(warnings, list) or any(
                not isinstance(warning, str) or not warning for warning in warnings
            ):
                raise AcquisitionContractError(
                    f"{session_id}.sensor_summary.biopac.warnings must be a string array"
                )
            biopac_warning_free = not warnings
        measured_timing_expected = (
            _v2_measured_timing_is_eligible(
                session,
                session_id,
                sync_config=root_sync_config_document,
            )
            if is_v2
            else len(sources) == 3
            and all(source == "meta_v13" for source in sources)
        )
        declared_alignment_expected = bool(
            measured_timing_expected
            and declared_authorized
            and biopac_warning_free
        )
        declared_stage_expected = bool(
            protocol_status == "auto" and declared_alignment_expected
        )
        alignment_expected = bool(
            measured_timing_expected and authorized and biopac_warning_free
        )
        stage_expected = bool(
            protocol_status == "auto"
            and alignment_expected
            and is_v2
            and _protocol_decode_has_independent_verification(
                session, receipt, root_protocol_config
            )
        )
        if is_v2:
            if not isinstance(range_document, Mapping):
                raise AcquisitionContractError(
                    f"{session_id}.range_tracking must be an object"
                )
            layout_selection_causal = _require_bool(
                range_document,
                "layout_selection_causal",
                f"{session_id}.range_tracking",
            )
            inference_feature_eligible = _require_bool(
                range_document,
                "inference_feature_eligible",
                f"{session_id}.range_tracking",
            )
            range_expected = bool(
                range_document.get("status") == "built"
                and range_document.get("selected_session_layout") == "split_halves"
                and layout_selection_causal
                and inference_feature_eligible
            )
        else:
            range_expected = bool(
                isinstance(range_document, Mapping)
                and range_document.get("status") == "built"
                and range_document.get("selected_session_layout") == "split_halves"
            )
        declared_strict_expected = declared_alignment_expected
        strict_expected = alignment_expected
        if is_v2:
            eligibility = session.get("eligibility")
            if not isinstance(eligibility, Mapping):
                raise AcquisitionContractError(
                    f"{session_id} lacks explicit eligibility components"
                )
            claimed_measured_timing_eligible = _require_bool(
                eligibility, "measured_timing_eligible", f"{session_id}.eligibility"
            )
            claimed_alignment_eligible = _require_bool(
                eligibility, "alignment_eligible", f"{session_id}.eligibility"
            )
            claimed_stage_metric_eligible = _require_bool(
                eligibility, "stage_metric_eligible", f"{session_id}.eligibility"
            )
            claimed_range_feature_eligible = _require_bool(
                eligibility, "range_feature_eligible", f"{session_id}.eligibility"
            )
            claimed_strict_cache_eligible = _require_bool(
                eligibility, "strict_cache_eligible", f"{session_id}.eligibility"
            )
            declared_expected = (
                measured_timing_expected,
                declared_alignment_expected,
                declared_stage_expected,
                range_expected,
                declared_strict_expected,
            )
            claimed = (
                claimed_measured_timing_eligible,
                claimed_alignment_eligible,
                claimed_stage_metric_eligible,
                claimed_range_feature_eligible,
                claimed_strict_cache_eligible,
            )
            if claimed != declared_expected:
                raise AcquisitionContractError(
                    f"{session_id} eligibility components do not match source evidence"
                )
            claimed_scientific_eligible = _require_bool(
                session, "scientific_eligible", session_id
            )
            if claimed_scientific_eligible != claimed_strict_cache_eligible:
                raise AcquisitionContractError(
                    f"{session_id} scientific eligibility alias mismatch"
                )
            if entry.get("scientific_eligible") is not claimed_scientific_eligible:
                raise AcquisitionContractError(
                    f"{session_id} root/session eligibility mismatch"
                )
            declared_eligible_count += int(claimed_scientific_eligible)
            measured_timing_eligible = measured_timing_expected
            alignment_eligible = alignment_expected
            stage_metric_eligible = stage_expected
            range_feature_eligible = range_expected
            strict_cache_eligible = strict_expected
            scientific_eligible = strict_cache_eligible
        else:
            # V1 mixed unrelated protocol/range hypotheses into one boolean and
            # did not prove full-cohort coverage.  It remains readable for
            # forensic inspection but can never authorize a new strict cache.
            measured_timing_eligible = measured_timing_expected
            alignment_eligible = alignment_expected
            stage_metric_eligible = stage_expected
            range_feature_eligible = range_expected
            strict_cache_eligible = False
            scientific_eligible = False
        eligible_count += int(strict_cache_eligible)

        sessions[session_id] = AcquisitionSessionContract(
            session_id=session_id,
            reconstruction_root=root,
            manifest_path=session_path,
            manifest=session,
            receipt_path=receipt_path,
            receipt=receipt,
            mapping=mapping,
            manual_approval_path=approval_path,
            manual_approval=approval,
            authorized=authorized,
            measured_timing_eligible=measured_timing_eligible,
            alignment_eligible=alignment_eligible,
            stage_metric_eligible=stage_metric_eligible,
            range_feature_eligible=range_feature_eligible,
            strict_cache_eligible=strict_cache_eligible,
            scientific_eligible=scientific_eligible,
            protocol=protocol,
            window_minimum_overlap_fraction=minimum_overlap,
            transition_guard_s=transition_guard,
            range_track_path=range_path,
        )

    if is_v2:
        if tuple(observed_usable_ids) != tuple(
            session_id for session_id in selected_ids if session_id in set(expected_usable_ids)
        ):
            raise AcquisitionContractError(
                "acquisition root expected usable IDs do not match session manifests"
            )
        assert cohort_authority is not None
        observed_identities: list[str] = []
        identity_map = cohort_authority.session_identity_map
        for session_id in observed_usable_ids:
            identity = identity_map[session_id]
            if identity not in observed_identities:
                observed_identities.append(identity)
        if full_cohort_complete and tuple(observed_identities) != (
            cohort_authority.expected_physical_identities
        ):
            raise AcquisitionContractError(
                "full-cohort physical identities do not match cohort authority"
            )
        if _require_nonnegative_int(
            document, "session_count", "acquisition root"
        ) != len(entries):
            raise AcquisitionContractError("acquisition root session count mismatch")
        if _require_nonnegative_int(
            document, "usable_session_count", "acquisition root"
        ) != len(observed_usable_ids):
            raise AcquisitionContractError("acquisition root usable-session count mismatch")
        if _require_nonnegative_int(
            document, "sync_authorized_session_count", "acquisition root"
        ) != declared_authorized_count:
            raise AcquisitionContractError(
                "acquisition root authorization count mismatch"
            )
        if _require_nonnegative_int(
            document, "scientific_eligible_session_count", "acquisition root"
        ) != declared_eligible_count:
            raise AcquisitionContractError(
                "acquisition root scientific-eligibility count mismatch"
            )
        claimed_root_scientific = _require_bool(
            document, "scientific_eligible", "acquisition root"
        )
        expected_claimed_root_scientific = bool(
            full_cohort_complete
            and observed_usable_ids
            and declared_eligible_count == len(observed_usable_ids)
        )
        if claimed_root_scientific != expected_claimed_root_scientific:
            raise AcquisitionContractError(
                "acquisition root scientific eligibility does not match its children"
            )
        # Historical claims remain available in ``manifest`` above.  The
        # effective property exposed to cache consumers is gated by the
        # independent raw-derived authorization boundary.
        root_scientific = bool(
            claimed_root_scientific
            and full_cohort_complete
            and observed_usable_ids
            and eligible_count == len(observed_usable_ids)
            and authorized_count == len(observed_usable_ids)
        )
    else:
        root_scientific = False

    return AcquisitionReconstruction(
        root=root,
        manifest_path=path,
        manifest=document,
        sessions=sessions,
        selection_scope=str(scope),
        execution_complete=execution_complete,
        full_cohort_complete=full_cohort_complete,
        scientific_eligible=root_scientific,
    )


def validate_raw_input_bindings(
    contract: AcquisitionSessionContract,
    dataset_root: str | Path,
) -> None:
    """Hash current raw inputs and require exact equality with the sync receipt."""

    root = Path(dataset_root).resolve()
    bindings = contract.receipt.get("input_bindings")
    if not isinstance(bindings, Mapping) or not bindings:
        raise AcquisitionContractError(f"{contract.session_id} lacks raw input bindings")
    for name, binding in bindings.items():
        if not isinstance(binding, Mapping):
            raise AcquisitionContractError(f"{contract.session_id} binding {name} is invalid")
        relative = binding.get("path")
        if not isinstance(relative, str) or not relative:
            raise AcquisitionContractError(f"{contract.session_id} binding {name} lacks a path")
        candidate = (root / relative).resolve()
        try:
            candidate.relative_to(root)
        except ValueError as exc:
            raise AcquisitionContractError(
                f"{contract.session_id} binding {name} escapes dataset root"
            ) from exc
        if not candidate.is_file():
            raise AcquisitionContractError(
                f"{contract.session_id} binding {name} is missing: {candidate}"
            )
        if "bytes" in binding and int(binding["bytes"]) != candidate.stat().st_size:
            raise AcquisitionContractError(
                f"{contract.session_id} binding {name} size mismatch"
            )
        if binding.get("sha256") != _sha256_file(candidate):
            raise AcquisitionContractError(
                f"{contract.session_id} binding {name} SHA-256 mismatch"
            )


def assign_stage_window(
    contract: AcquisitionSessionContract,
    window_start_biopac_s: float,
    window_end_biopac_s: float,
) -> StageWindowAssignment:
    """Assign one BIOPAC-time window with transition and review fail-closed gates."""

    start = float(window_start_biopac_s)
    end = float(window_end_biopac_s)
    duration = float(contract.protocol.get("duration_s"))
    if not np.isfinite([start, end]).all() or not 0 <= start < end <= duration:
        return StageWindowAssignment(
            None, None, None, None, 0.0, False, False, None, "outside_protocol_duration"
        )
    stages = contract.protocol.get("stages")
    if not isinstance(stages, list):
        raise AcquisitionContractError(f"{contract.session_id} protocol stages are invalid")
    window_duration = end - start
    best: Mapping[str, Any] | None = None
    best_overlap = 0.0
    transition = False
    for stage in stages:
        if not isinstance(stage, Mapping):
            raise AcquisitionContractError(
                f"{contract.session_id} protocol stage is not an object"
            )
        stage_start = float(stage["start"]["time_s"])
        stage_end = float(stage["end"]["time_s"])
        overlap = max(0.0, min(end, stage_end) - max(start, stage_start))
        fraction = overlap / window_duration
        if fraction > best_overlap:
            best_overlap = fraction
            best = stage
        guard = contract.transition_guard_s
        if (
            start < stage_start + guard
            and end > stage_start - guard
            or start < stage_end + guard
            and end > stage_end - guard
        ):
            transition = True
    if transition:
        return StageWindowAssignment(
            None,
            None,
            None,
            None,
            best_overlap,
            True,
            False,
            None,
            "transition_guard",
        )
    if best is None or best_overlap < contract.window_minimum_overlap_fraction:
        return StageWindowAssignment(
            None,
            None,
            None,
            None,
            best_overlap,
            transition,
            False,
            None,
            "insufficient_stage_overlap",
        )
    status = str(best.get("status"))
    # ``auto`` describes only the decoded stage's local status.  It is not an
    # authority token.  The reconstruction loader derives
    # ``stage_metric_eligible`` from synchronization and an independently
    # verified protocol replay; every public window assignment must retain that
    # exact session-level gate so callers cannot accidentally promote a
    # diagnostic stage document into evaluation evidence.
    eligible = bool(contract.stage_metric_eligible and status == "auto")
    stage_id = str(best.get("stage_id"))
    if eligible:
        reason = "assigned"
    elif status == "auto" and not contract.stage_metric_eligible:
        reason = "stage_metrics_not_authorized"
    elif status == "uncertain":
        reason = "stage_uncertain"
    elif status == "review":
        reason = "stage_requires_review"
    else:
        reason = "stage_status_invalid"
    return StageWindowAssignment(
        stage_id=stage_id,
        stage_name=str(best.get("name")),
        stage_status=status,
        stage_confidence=float(best.get("confidence")),
        overlap_fraction=best_overlap,
        transition_window=False,
        eligible_for_stage_metrics=eligible,
        phase7_assignment=(
            contract.protocol.get("phase7_assignment") if stage_id == "phase7" else None
        ),
        reason=reason,
    )


__all__ = [
    "ACQUISITION_SCHEMA",
    "LEGACY_ACQUISITION_SCHEMA",
    "ACQUISITION_COHORT_AUTHORITY_SCHEMA",
    "ACQUISITION_COHORT_V1_CONTENT_SHA256",
    "SUPPORTED_ACQUISITION_SCHEMAS",
    "ANNOTATION_ONLY_COLUMNS",
    "AcquisitionContractError",
    "AcquisitionCohortAuthority",
    "AcquisitionSessionContract",
    "AcquisitionReconstruction",
    "StageWindowAssignment",
    "load_acquisition_reconstruction",
    "load_acquisition_cohort_authority",
    "build_v2_raw_input_binding_state",
    "build_v2_xethru_record_contract",
    "validate_raw_input_bindings",
    "assign_stage_window",
]
