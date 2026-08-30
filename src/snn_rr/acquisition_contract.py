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
from typing import Any, Mapping

import numpy as np

from .synchronization import (
    TimeMapping,
    canonical_content_sha256,
    canonical_json_bytes,
    read_sync_receipt,
    synchronization_is_authorized,
    validate_manual_approval,
)


ACQUISITION_SCHEMA = "snn_rr.acquisition_reconstruction.v1"
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
class AcquisitionSessionContract:
    session_id: str
    reconstruction_root: Path
    manifest_path: Path
    manifest: Mapping[str, Any]
    receipt_path: Path
    receipt: Mapping[str, Any]
    mapping: TimeMapping
    manual_approval_path: Path | None
    manual_approval: Mapping[str, Any] | None
    authorized: bool
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


def _read_strict_json(path: Path) -> dict[str, Any]:
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
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=reject_duplicates,
            parse_constant=reject_nonfinite,
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise AcquisitionContractError(f"cannot read acquisition JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise AcquisitionContractError(f"acquisition JSON must be an object: {path}")
    return value


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


def _mapping_equal(left: Any, right: Any) -> bool:
    return isinstance(left, Mapping) and isinstance(right, Mapping) and canonical_json_bytes(
        left
    ) == canonical_json_bytes(right)


def load_acquisition_reconstruction(
    manifest_path: str | Path,
) -> AcquisitionReconstruction:
    """Read every session contract and verify the full content-addressed graph."""

    path = Path(manifest_path).resolve()
    root = path.parent
    document = _read_strict_json(path)
    if document.get("schema_version") != ACQUISITION_SCHEMA:
        raise AcquisitionContractError("unexpected acquisition reconstruction schema")
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

    sessions: dict[str, AcquisitionSessionContract] = {}
    for entry in entries:
        assert isinstance(entry, Mapping)
        session_id = str(entry["session_id"])
        session_path = _resolve_inside(root, entry.get("manifest"), "session manifest path")
        session = _read_strict_json(session_path)
        if session.get("schema_version") != ACQUISITION_SCHEMA:
            raise AcquisitionContractError(f"{session_id} has an incompatible session schema")
        if session.get("session_id") != session_id:
            raise AcquisitionContractError(f"{session_id} session manifest ID mismatch")
        session_hash = _require_content_hash(session, f"{session_id} session manifest")
        if entry.get("content_sha256") != session_hash:
            raise AcquisitionContractError(f"{session_id} root/session hash mismatch")
        if session.get("usable") is not True:
            continue

        sync = session.get("synchronization")
        if not isinstance(sync, Mapping):
            raise AcquisitionContractError(f"{session_id} lacks synchronization metadata")
        receipt_path = _resolve_inside(root, sync.get("receipt"), "sync receipt path")
        receipt = read_sync_receipt(receipt_path)
        if receipt.get("session_id") != session_id:
            raise AcquisitionContractError(f"{session_id} sync receipt ID mismatch")
        if sync.get("receipt_sha256") != _sha256_file(receipt_path):
            raise AcquisitionContractError(f"{session_id} sync receipt file hash mismatch")
        if sync.get("receipt_content_sha256") != receipt.get("content_sha256"):
            raise AcquisitionContractError(f"{session_id} sync receipt content hash mismatch")
        receipt_mapping = receipt.get("result", {}).get("mapping")
        if not _mapping_equal(sync.get("mapping"), receipt_mapping):
            raise AcquisitionContractError(f"{session_id} duplicated sync mapping mismatch")
        if not isinstance(receipt_mapping, Mapping):
            raise AcquisitionContractError(f"{session_id} has no usable synchronization mapping")
        mapping = TimeMapping.from_dict(receipt_mapping)

        approval_path: Path | None = None
        approval: Mapping[str, Any] | None = None
        stated_approval_path = sync.get("manual_approval")
        if stated_approval_path is not None:
            approval_path = _resolve_inside(
                root, stated_approval_path, "manual approval path"
            )
            approval = _read_strict_json(approval_path)
            approval = validate_manual_approval(approval, receipt)
            if sync.get("manual_approval_file_sha256") != _sha256_file(approval_path):
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
        authorized = synchronization_is_authorized(receipt, manual_approval=approval)
        if bool(sync.get("authorized")) != authorized:
            raise AcquisitionContractError(f"{session_id} authorization statement mismatch")

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
        minimum_overlap = float(assignment.get("minimum_overlap_fraction"))
        transition_guard = float(assignment.get("transition_guard_s"))
        if not 0 < minimum_overlap <= 1 or transition_guard < 0:
            raise AcquisitionContractError(f"{session_id} window assignment is invalid")

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
            scientific_eligible=bool(session.get("scientific_eligible")),
            protocol=protocol,
            window_minimum_overlap_fraction=minimum_overlap,
            transition_guard_s=transition_guard,
            range_track_path=range_path,
        )
    return AcquisitionReconstruction(
        root=root,
        manifest_path=path,
        manifest=document,
        sessions=sessions,
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
    eligible = bool(status != "review" and not transition)
    stage_id = str(best.get("stage_id"))
    return StageWindowAssignment(
        stage_id=stage_id,
        stage_name=str(best.get("name")),
        stage_status=status,
        stage_confidence=float(best.get("confidence")),
        overlap_fraction=best_overlap,
        transition_window=transition,
        eligible_for_stage_metrics=eligible,
        phase7_assignment=(
            contract.protocol.get("phase7_assignment") if stage_id == "phase7" else None
        ),
        reason="eligible" if eligible else ("transition_guard" if transition else "stage_review"),
    )


__all__ = [
    "ACQUISITION_SCHEMA",
    "ANNOTATION_ONLY_COLUMNS",
    "AcquisitionContractError",
    "AcquisitionSessionContract",
    "AcquisitionReconstruction",
    "StageWindowAssignment",
    "load_acquisition_reconstruction",
    "validate_raw_input_bindings",
    "assign_stage_window",
]
