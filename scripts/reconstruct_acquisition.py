#!/usr/bin/env python3
"""Reconstruct synchronization, protocol stages, and target-free range tracks.

This command is additive.  It never edits ``HAI_EXPERIMENT`` and never
overwrites the historical feature caches.  It emits content-bound sync
receipts, an offline seven-stage annotation, range-layout evidence, causal
range-bin tracks, and review plots under a new acquisition artifact root.

BIOPAC-derived timing and protocol annotations are label-construction/audit
metadata only.  They are explicitly forbidden as deployable model features.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from snn_rr.acquisition_protocol import (  # noqa: E402
    ANNOTATION_USAGE_CONTRACT,
    BoundaryCandidate,
    DecodedProtocol,
    SessionProtocolRecord,
    decode_ordered_protocol,
    load_dataset_issue_records,
    load_protocol_config,
    records_by_session,
)
from snn_rr.acquisition_contract import (  # noqa: E402
    AcquisitionCohortAuthority,
    build_v2_raw_input_binding_state,
    build_v2_xethru_record_contract,
    load_acquisition_cohort_authority,
)
from snn_rr.data import (  # noqa: E402
    SubjectManifest,
    build_dataset_manifest,
    load_biopac_mat,
    load_xethru_recording,
)
from snn_rr.range_tracking import (  # noqa: E402
    RangeTrack,
    causal_range_track,
    compare_iq_layouts,
)
from snn_rr.preprocess import (  # noqa: E402
    identity_for_session,
    replace_radar_outliers_past_only,
)
from snn_rr.radar_timing import (  # noqa: E402
    CausalUniformRadarResampleV1,
    causal_uniform_resample_radar_views_v1,
)
from snn_rr.synchronization import (  # noqa: E402
    MarkerCandidate,
    SynchronizationResult,
    build_sync_receipt,
    canonical_content_sha256,
    detect_radar_marker_candidates,
    detect_rsp_marker_candidates,
    epoch_prior_offset_from_starts,
    estimate_marker_time_mapping,
    load_synchronization_config,
    read_sync_receipt,
    robust_radar_motion_envelope,
    synchronization_is_authorized,
    validate_manual_approval,
)


SCHEMA_VERSION = "snn_rr.acquisition_reconstruction.v2"
DEFAULT_OUTPUT = PROJECT_ROOT / "artifacts/acquisition/reconstruction_v2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root", type=Path, default=PROJECT_ROOT / "HAI_EXPERIMENT"
    )
    parser.add_argument(
        "--spreadsheet",
        type=Path,
        default=PROJECT_ROOT / "HAI_EXPERIMENT/Dataset_issue.xlsx",
    )
    parser.add_argument(
        "--sync-config",
        type=Path,
        default=PROJECT_ROOT / "configs/sync_marker_affine_measured10_v2.yaml",
    )
    parser.add_argument(
        "--protocol-config",
        type=Path,
        default=PROJECT_ROOT / "configs/acquisition_protocol_v1.yaml",
    )
    parser.add_argument(
        "--cohort-authority",
        type=Path,
        default=PROJECT_ROOT / "configs/acquisition_cohort_v1.yaml",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--approval-dir", type=Path)
    parser.add_argument("--subjects", nargs="*", help="optional IDs such as S02_RJS")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--skip-range-tracks", action="store_true")
    parser.add_argument("--skip-review-plots", action="store_true")
    parser.add_argument("--layout-maximum-frames", type=int, default=24_000)
    return parser.parse_args()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_paths(paths: Iterable[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted({item.resolve() for item in paths}, key=str):
        digest.update(str(path).encode("utf-8"))
        digest.update(bytes.fromhex(_sha256_file(path)))
    return digest.hexdigest()


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


def _atomic_json(path: Path, document: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(document, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def _atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(path)


def _input_bindings(subject: SubjectManifest, dataset_root: Path) -> dict[str, dict[str, Any]]:
    if not subject.usable or subject.selected_session is None or subject.biopac_path is None:
        return {}
    bindings, _ = build_v2_raw_input_binding_state(subject, dataset_root)
    return bindings


def _validate_dataset_against_cohort_authority(
    dataset: Any,
    authority: AcquisitionCohortAuthority,
) -> None:
    """Require exact discovered session, usability, and identity authority."""

    session_ids = tuple(item.subject_id for item in dataset.subjects)
    if session_ids != authority.expected_session_ids:
        raise RuntimeError(
            "discovered dataset session IDs/order do not match cohort authority"
        )
    usable_ids = tuple(
        item.subject_id for item in dataset.subjects if item.usable
    )
    if usable_ids != authority.expected_usable_session_ids:
        raise RuntimeError(
            "discovered dataset usability does not match cohort authority"
        )
    authority_identities = authority.session_identity_map
    observed_assignments = tuple(
        (session_id, identity_for_session(session_id)) for session_id in session_ids
    )
    if observed_assignments != authority.session_identities:
        raise RuntimeError(
            "discovered dataset physical identities do not match cohort authority"
        )
    observed_physical_identities: list[str] = []
    for session_id in usable_ids:
        identity = authority_identities[session_id]
        if identity not in observed_physical_identities:
            observed_physical_identities.append(identity)
    if tuple(observed_physical_identities) != authority.expected_physical_identities:
        raise RuntimeError(
            "discovered physical-identity order does not match cohort authority"
        )


def _load_radar_views(
    subject: SubjectManifest,
    *,
    sync_config: Any,
) -> tuple[
    list[np.ndarray],
    list[np.ndarray],
    list[np.ndarray],
    list[float],
    list[str],
    dict[str, Any],
]:
    if subject.selected_session is None:
        raise ValueError(f"{subject.subject_id} has no selected radar session")
    payloads: list[np.ndarray] = []
    relative_times: list[np.ndarray] = []
    frame_sequences: list[np.ndarray] = []
    starts: list[float] = []
    timestamp_sources: list[str] = []
    timestamp_repairs: list[int] = []
    metadata_warnings: list[tuple[str, ...]] = []
    for radar_id in (1, 2, 3):
        stream = subject.selected_session.radars[radar_id]
        if stream.start_epoch_ms is None:
            raise ValueError(f"{subject.subject_id} radar{radar_id} lacks a start epoch")
        recording = load_xethru_recording(stream.recording_dir, strict=True)
        payloads.append(np.asarray(recording.records["bins"], dtype=np.float32))
        frame_sequences.append(
            np.asarray(recording.records["frame_sequence"], dtype=np.uint32)
        )
        starts.append(stream.start_epoch_ms / 1000.0)
        relative = np.asarray(recording.timestamps_ms, dtype=np.float64) / 1000.0
        relative_times.append(relative)
        timestamp_sources.append(recording.meta.timestamp_source if recording.meta else "fallback")
        timestamp_repairs.append(recording.meta.timestamp_repairs if recording.meta else 0)
        metadata_warnings.append(tuple(recording.warnings))
    warning_evidence = _radar_metadata_warning_evidence(
        session_id=subject.subject_id,
        warnings_by_view=metadata_warnings,
        sync_config=sync_config,
    )
    record_contract_evidence = build_v2_xethru_record_contract(subject)
    summary = {
        "timestamp_sources": timestamp_sources,
        "timestamp_repairs": timestamp_repairs,
        "start_epochs_s": starts,
        "start_spread_ms": 1000.0 * float(np.ptp(starts)),
        "per_radar_frame_counts": [len(item) for item in payloads],
        "xethru_record_contract": record_contract_evidence,
        "xethru_record_contract_evidence_sha256": _canonical_sha256(
            record_contract_evidence
        ),
        **warning_evidence,
    }
    return (
        payloads,
        relative_times,
        frame_sequences,
        starts,
        timestamp_sources,
        summary,
    )


def _radar_metadata_warning_evidence(
    *,
    session_id: str,
    warnings_by_view: Iterable[Iterable[str]],
    sync_config: Any,
) -> dict[str, Any]:
    """Bind parser warnings to the exact per-session config declaration.

    A declaration is session-wide but evaluated independently for each radar
    view.  An unlisted session therefore requires three empty warning lists;
    a listed session requires every view to equal the declared ordered list.
    No substring/category matching is permitted.
    """

    observed_by_view = [tuple(values) for values in warnings_by_view]
    if len(observed_by_view) != 3:
        raise ValueError("radar metadata warning policy requires exactly three views")
    for warnings in observed_by_view:
        if any(not isinstance(value, str) or not value for value in warnings):
            raise ValueError("radar metadata warnings must be non-empty strings")

    allowlist = getattr(sync_config, "radar_metadata_warning_allowlist", None)
    if not isinstance(allowlist, Mapping):
        raise ValueError("radar metadata warning allowlist is not a mapping")
    declared = session_id in allowlist
    allowed = tuple(allowlist.get(session_id, ()))
    views = [
        {
            "radar_id": radar_id,
            "warnings": list(observed),
            "allowed_warnings": list(allowed),
            "exact_match": observed == allowed,
        }
        for radar_id, observed in enumerate(observed_by_view, start=1)
    ]
    eligible = all(view["exact_match"] is True for view in views)
    policy = {
        "schema": "snn_rr.radar_metadata_warning_policy.v1",
        "mode": "ordered_exact_list_per_session_per_view",
        "config_field": "radar_metadata_warning_allowlist",
        "session_id": session_id,
        "session_allowlist_declared": declared,
        "unlisted_session_policy": "require_no_warnings",
        "allowed_warnings_per_view": list(allowed),
    }
    evidence = {
        "policy": policy,
        "views": views,
        "eligible": eligible,
    }
    return {
        "metadata_warning_policy": policy,
        "metadata_warning_views": views,
        "metadata_warnings_eligible": eligible,
        "metadata_warning_evidence_sha256": _canonical_sha256(evidence),
    }


def _marker_score(candidate: MarkerCandidate) -> float:
    return float(np.clip(1.0 - np.exp(-max(float(candidate.score), 0.0) / 6.0), 0.1, 1.0))


def protocol_boundary_candidates(result: SynchronizationResult) -> tuple[BoundaryCandidate, ...]:
    """Build BIOPAC-time candidates without importing an unapproved mapping.

    Protocol stages live natively on the BIOPAC clock.  A sync proposal that
    has not passed authorization must not move radar peaks onto that clock and
    then feed them back into stage decoding.  Direct RSP marker evidence is
    sufficient here and avoids both duplication and circular confirmation.
    """

    candidates: list[BoundaryCandidate] = []
    for marker in result.rsp_markers:
        candidates.append(
            BoundaryCandidate(
                time_s=float(marker.time_s),
                score=0.65 * _marker_score(marker),
                source="biopac_marker",
                biopac_derived=True,
            )
        )
    return tuple(sorted(candidates, key=lambda item: item.time_s))


def _track_arrays(prefix: str, track: RangeTrack) -> dict[str, np.ndarray]:
    return {
        f"{prefix}_bin_index": track.bin_index,
        f"{prefix}_confidence": track.confidence,
        f"{prefix}_normalized_entropy": track.normalized_entropy,
        f"{prefix}_missing": track.missing,
        f"{prefix}_multimodal": track.multimodal,
        f"{prefix}_evidence_strength": track.evidence_strength,
    }


def _build_range_tracks(
    resampled: CausalUniformRadarResampleV1,
    *,
    output_path: Path,
    layout_maximum_frames: int,
) -> dict[str, Any]:
    downsampled = resampled.values
    times = resampled.times_s
    arrays: dict[str, np.ndarray] = {
        "radar_times_s": times.astype(np.float64),
        "radar_valid_mask": resampled.valid_mask.astype(bool),
        "radar_sample_counts": resampled.sample_counts.astype(np.int32),
    }
    evidence_documents: list[dict[str, Any]] = []
    selected: list[str] = []
    for radar_index in range(3):
        values = downsampled[radar_index]
        evidence = compare_iq_layouts(
            values,
            fs=10.0,
            maximum_frames=layout_maximum_frames,
        )
        evidence_documents.append(asdict(evidence))
        selected.append(evidence.selected_layout)
        for layout in ("split_halves", "interleaved"):
            track = causal_range_track(values, fs=10.0, layout=layout)
            arrays.update(_track_arrays(f"radar{radar_index + 1}_{layout}", track))
    _atomic_npz(output_path, **arrays)
    if all(item == "split_halves" for item in selected):
        session_layout = "split_halves"
    elif all(item == "interleaved" for item in selected):
        session_layout = "interleaved"
    else:
        session_layout = "unknown"
    return {
        "artifact": output_path.name,
        "artifact_sha256": _sha256_file(output_path),
        "sample_rate_hz": 10.0,
        "physical_range_calibrated": False,
        "range_unit": "unscaled_bin_index",
        "per_radar_layout_evidence": evidence_documents,
        "selected_session_layout": session_layout,
        "selection_fail_closed": session_layout == "unknown",
        "target_or_biopac_used": False,
        "samplewise_tracking_causal": True,
        # Layout comparison pools a bounded prefix/session context and is an
        # offline representation audit, not a predeclared streaming input
        # contract.  It cannot authorize range_aux as an inference feature.
        "layout_selection_causal": False,
        "inference_feature_eligible": False,
        "radar_resampling": resampled.summary,
    }


def _find_manual_approval(approval_dir: Path | None, session_id: str) -> dict[str, Any] | None:
    if approval_dir is None:
        return None
    path = approval_dir / f"{session_id}.approval.json"
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read approval {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"approval must be a JSON object: {path}")
    return value


def _plot_review(
    path: Path,
    *,
    session_id: str,
    rsp: np.ndarray,
    rsp_times_s: np.ndarray,
    radar_times_s: np.ndarray,
    radar_motion_z: np.ndarray,
    sync_result: SynchronizationResult,
    protocol: DecodedProtocol,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(2, 1, figsize=(16, 9), constrained_layout=True)
    axes[0].plot(rsp_times_s, rsp, color="#1b7f3a", linewidth=0.45)
    for marker in sync_result.rsp_markers:
        axes[0].axvline(marker.time_s, color="#d62728", alpha=0.25, linewidth=0.7)
    for stage in protocol.stages:
        color = "#3366cc" if stage.status == "auto" else "#ff9900" if stage.status == "uncertain" else "#cc0000"
        axes[0].axvspan(stage.start.time_s, stage.end.time_s, color=color, alpha=0.06)
        axes[0].text(
            0.5 * (stage.start.time_s + stage.end.time_s),
            0.98,
            stage.stage_id,
            transform=axes[0].get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=8,
        )
    axes[0].set_ylabel("BIOPAC RSP (V)")
    axes[0].set_title(f"{session_id}: RSP markers and reconstructed stages")
    axes[0].grid(alpha=0.15)

    axes[1].plot(radar_times_s, radar_motion_z, color="#111111", linewidth=0.55)
    for marker in sync_result.radar_markers:
        axes[1].axvline(marker.time_s, color="#17becf", alpha=0.3, linewidth=0.7)
    if sync_result.mapping is not None:
        for stage in protocol.stages:
            start, end = sync_result.mapping.rsp_to_radar(
                [stage.start.time_s, stage.end.time_s]
            )
            color = "#3366cc" if stage.status == "auto" else "#ff9900" if stage.status == "uncertain" else "#cc0000"
            axes[1].axvspan(start, end, color=color, alpha=0.06)
    axes[1].set_xlabel("radar relative time (s)")
    axes[1].set_ylabel("robust motion z")
    axes[1].set_title(
        f"sync={sync_result.decision}, confidence={sync_result.confidence:.3f}, "
        f"matches={len(sync_result.matches)}"
    )
    axes[1].grid(alpha=0.15)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=140)
    plt.close(figure)


def _session_rows(document: Mapping[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    sync = document.get("synchronization", {})
    protocol = document.get("protocol", {})
    range_doc = document.get("range_tracking", {})
    eligibility = document.get("eligibility", {})
    session_row = {
        "session_id": document["session_id"],
        "usable": document.get("usable", False),
        "scientific_eligible": document.get("scientific_eligible", False),
        "measured_timing_eligible": eligibility.get(
            "measured_timing_eligible", False
        ),
        "alignment_eligible": eligibility.get("alignment_eligible", False),
        "stage_metric_eligible": eligibility.get("stage_metric_eligible", False),
        "range_feature_eligible": eligibility.get("range_feature_eligible", False),
        "strict_cache_eligible": eligibility.get("strict_cache_eligible", False),
        "sync_decision": sync.get("decision"),
        "sync_authorized": sync.get("authorized", False),
        "sync_confidence": sync.get("confidence"),
        "sync_offset_s": sync.get("mapping", {}).get("offset_s") if isinstance(sync.get("mapping"), Mapping) else None,
        "sync_drift_ppm": sync.get("mapping", {}).get("drift_ppm") if isinstance(sync.get("mapping"), Mapping) else None,
        "sync_match_count": sync.get("match_count", 0),
        "protocol_status": protocol.get("status"),
        "protocol_confidence": protocol.get("confidence"),
        "phase7_assignment": protocol.get("phase7_assignment"),
        "range_layout": range_doc.get("selected_session_layout"),
        "review_reasons": "|".join(document.get("review_reasons", [])),
    }
    stage_rows: list[dict[str, Any]] = []
    for stage in protocol.get("stages", []):
        stage_rows.append(
            {
                "session_id": document["session_id"],
                "stage_id": stage["stage_id"],
                "stage_name": stage["name"],
                "start_biopac_s": stage["start"]["time_s"],
                "end_biopac_s": stage["end"]["time_s"],
                "duration_s": stage["duration_s"],
                "status": stage["status"],
                "confidence": stage["confidence"],
                "phase7_assignment": protocol.get("phase7_assignment") if stage["stage_id"] == "phase7" else None,
                "qc_flags": "|".join(stage.get("qc_flags", [])),
            }
        )
    marker_rows: list[dict[str, Any]] = []
    for modality, markers in (
        ("radar", sync.get("radar_markers", [])),
        ("biopac_rsp", sync.get("rsp_markers", [])),
    ):
        for marker in markers:
            marker_rows.append(
                {
                    "session_id": document["session_id"],
                    "modality": modality,
                    "time_s": marker["time_s"],
                    "score": marker["score"],
                    "source": marker["source"],
                }
            )
    return session_row, stage_rows, marker_rows


def reconstruct_subject(
    subject: SubjectManifest,
    *,
    dataset_root: Path,
    output_root: Path,
    sync_config: Any,
    protocol_config: Any,
    session_record: SessionProtocolRecord | None,
    approval_dir: Path | None,
    build_range_tracks: bool,
    build_review_plot: bool,
    layout_maximum_frames: int,
    reconstruction_context: Mapping[str, Any],
    force: bool,
) -> dict[str, Any]:
    session_dir = output_root / "sessions" / subject.subject_id
    manifest_path = session_dir / "session_manifest.json"
    if manifest_path.is_file() and not force:
        document = json.loads(manifest_path.read_text(encoding="utf-8"))
        if document.get("content_sha256") != canonical_content_sha256(document):
            raise ValueError(
                f"existing session manifest content hash mismatch: {manifest_path}"
            )
        if not subject.usable:
            expected_unusable = {
                "schema_version": SCHEMA_VERSION,
                "session_id": subject.subject_id,
                "physical_identity": identity_for_session(subject.subject_id),
                "usable": False,
                "reconstruction_context": dict(reconstruction_context),
                "scientific_eligible": False,
                "eligibility": {
                    "measured_timing_eligible": False,
                    "alignment_eligible": False,
                    "stage_metric_eligible": False,
                    "range_feature_eligible": False,
                    "strict_cache_eligible": False,
                },
                "reason": "missing paired three-radar/BIOPAC recording",
                "session_record": (
                    None if session_record is None else session_record.to_dict()
                ),
                "review_reasons": ["raw_session_unusable"],
            }
            expected_unusable["content_sha256"] = canonical_content_sha256(
                expected_unusable
            )
            if document == expected_unusable:
                return document
            if document.get("schema_version") == SCHEMA_VERSION:
                raise ValueError(
                    f"stale reconstruction context for {subject.subject_id}; use --force"
                )
            raise ValueError(
                f"existing incompatible artifact: {manifest_path}; use --force"
            )
        if subject.usable:
            current_bindings, current_raw_graph = (
                build_v2_raw_input_binding_state(subject, dataset_root)
            )
        source_approval = _find_manual_approval(approval_dir, subject.subject_id)
        stored_sync = document.get("synchronization", {})
        receipt_relative = stored_sync.get("receipt")
        if not isinstance(receipt_relative, str) or not receipt_relative:
            raise ValueError(
                f"existing reconstruction lacks a receipt: {subject.subject_id}"
            )
        receipt_path = (output_root / receipt_relative).resolve()
        try:
            receipt_path.relative_to(output_root.resolve())
        except ValueError as error:
            raise ValueError(
                f"existing receipt escapes reconstruction root: {subject.subject_id}"
            ) from error
        current_receipt = read_sync_receipt(receipt_path)
        if source_approval is not None:
            source_approval = validate_manual_approval(
                source_approval, current_receipt
            )
        source_approval_hash = (
            None if source_approval is None else source_approval.get("content_sha256")
        )
        stored_approval_hash = stored_sync.get(
            "manual_approval_content_sha256"
        )
        if (
            document.get("schema_version") == SCHEMA_VERSION
            and document.get("reconstruction_context") == dict(reconstruction_context)
            and document.get("raw_input_bindings") == current_bindings
            and (
                not subject.usable
                or document.get("raw_input_graph") == current_raw_graph
            )
            and stored_sync.get("manual_approval_present", False)
            is (source_approval is not None)
            and stored_sync.get("receipt_sha256") == _sha256_file(receipt_path)
            and stored_sync.get("receipt_content_sha256")
            == current_receipt.get("content_sha256")
            and stored_approval_hash == source_approval_hash
        ):
            return document
        if document.get("schema_version") == SCHEMA_VERSION:
            raise ValueError(
                f"stale reconstruction context for {subject.subject_id}; use --force"
            )
        raise ValueError(f"existing incompatible artifact: {manifest_path}; use --force")

    if not subject.usable:
        document = {
            "schema_version": SCHEMA_VERSION,
            "session_id": subject.subject_id,
            "physical_identity": identity_for_session(subject.subject_id),
            "usable": False,
            "reconstruction_context": dict(reconstruction_context),
            "scientific_eligible": False,
            "eligibility": {
                "measured_timing_eligible": False,
                "alignment_eligible": False,
                "stage_metric_eligible": False,
                "range_feature_eligible": False,
                "strict_cache_eligible": False,
            },
            "reason": "missing paired three-radar/BIOPAC recording",
            "session_record": None if session_record is None else session_record.to_dict(),
            "review_reasons": ["raw_session_unusable"],
        }
        document["content_sha256"] = canonical_content_sha256(document)
        _atomic_json(manifest_path, document)
        return document

    preload_bindings, preload_raw_graph = build_v2_raw_input_binding_state(
        subject, dataset_root
    )
    biopac = load_biopac_mat(subject.biopac_path, strict=False)
    biopac_reference_eligible = len(biopac.warnings) == 0
    (
        payloads,
        relative_times,
        frame_sequences,
        radar_starts,
        timestamp_sources,
        radar_summary,
    ) = _load_radar_views(subject, sync_config=sync_config)
    radar_outlier_replacements: list[int] = []
    corrected_payloads: list[np.ndarray] = []
    for values in payloads:
        corrected, replacement_count = replace_radar_outliers_past_only(values)
        corrected_payloads.append(corrected)
        radar_outlier_replacements.append(replacement_count)
    payloads = corrected_payloads
    radar_summary["past_only_outlier_replacements"] = radar_outlier_replacements
    feature_resample = causal_uniform_resample_radar_views_v1(
        payloads,
        relative_times,
        radar_starts,
        frame_sequences,
        output_hz=10.0,
        max_gap_s=0.050,
        gap_policy="mask",
        timestamp_sources=timestamp_sources,
        require_measured_timestamps=False,
    )
    payload = feature_resample.values
    radar_times_s = feature_resample.times_s
    timestamp_corrections = [
        float(
            item["timestamp_repair"].get("maximum_timestamp_correction_s", 0.0)
        )
        for item in feature_resample.summary["per_view"]
    ]
    unaccounted_payload_frames = sum(
        int(item.get("unaccounted_payload_frame_count", 0))
        for item in feature_resample.summary["per_view"]
    )
    timestamp_plateau_intervals = sum(
        int(item.get("timestamp_plateau_interval_count", 0))
        for item in feature_resample.summary["per_view"]
    )
    time_arithmetic = feature_resample.summary.get("time_arithmetic", {})
    measured_timing_eligible = bool(
        len(timestamp_sources) == 3
        and all(source == "meta_v13" for source in timestamp_sources)
        and radar_summary.get("metadata_warnings_eligible") is True
        and feature_resample.valid_mask.all()
        and isinstance(time_arithmetic, dict)
        and time_arithmetic.get("half_open_boundary_exact") is True
        and max(timestamp_corrections, default=0.0) <= 0.050
        and unaccounted_payload_frames == 0
    )
    radar_summary.update(
        {
            "origin_epoch_s": feature_resample.origin_epoch_s,
            "sync_marker_resampling": feature_resample.summary,
            "feature_resampling": feature_resample.summary,
            "resampling_content_hashes": feature_resample.summary[
                "content_hashes"
            ],
            "resampling_transform_evidence_sha256": feature_resample.summary[
                "transform_evidence_sha256"
            ],
            "maximum_timestamp_correction_s": max(
                timestamp_corrections, default=0.0
            ),
            "unaccounted_payload_frame_count": unaccounted_payload_frames,
            "timestamp_plateau_interval_count": timestamp_plateau_intervals,
            "measured_timing_eligible": measured_timing_eligible,
        }
    )
    rsp_times_s = np.arange(len(biopac.rsp), dtype=np.float64) / biopac.sample_rate_hz
    prior = epoch_prior_offset_from_starts(
        radar_start_epoch_s=radar_summary["origin_epoch_s"],
        rsp_start_epoch_s=biopac.start_datetime.timestamp(),
    )
    envelope = robust_radar_motion_envelope(
        payload,
        radar_times_s=radar_times_s,
        radar_valid_mask=feature_resample.valid_mask,
        config=sync_config,
    )
    radar_markers = detect_radar_marker_candidates(envelope, config=sync_config)
    rsp_markers = detect_rsp_marker_candidates(
        biopac.rsp, rsp_times_s=rsp_times_s, config=sync_config
    )
    base_result = estimate_marker_time_mapping(
        radar_markers,
        rsp_markers,
        epoch_prior_offset_s=prior,
        config=sync_config,
    )
    diagnostics = dict(base_result.diagnostics)
    diagnostics.update(
        {
            "valid_radar_view_count": envelope.valid_view_count,
            "radar_nonfinite_values_repaired": envelope.repaired_nonfinite_values,
            "radar_envelope_target_free": True,
            "radar_payload_interpretation": "182_real_float_payload_values",
            "radar_resample_valid_mask_applied": True,
            "radar_resample_valid_mask_sha256": hashlib.sha256(
                np.ascontiguousarray(feature_resample.valid_mask).tobytes()
            ).hexdigest(),
            "radar_envelope_valid_mask_sha256": hashlib.sha256(
                np.ascontiguousarray(envelope.valid_mask).tobytes()
            ).hexdigest(),
            "radar_invalid_guard_policy": (
                "adjacent_pair_smoothing_support_and_view_topology"
            ),
            "radar_invalid_guard_radius_frames": max(
                1,
                int(
                    np.ceil(
                        round(
                            sync_config.motion_smoothing_s
                            * sync_config.radar_sample_rate_hz
                        )
                        / 2.0
                    )
                ),
            ),
            "radar_envelope_valid_sample_count": int(envelope.valid_mask.sum()),
        }
    )
    sync_result = SynchronizationResult(
        decision=base_result.decision,
        reasons=base_result.reasons,
        mapping=base_result.mapping,
        matches=base_result.matches,
        confidence=base_result.confidence,
        residual_rmse_s=base_result.residual_rmse_s,
        residual_max_abs_s=base_result.residual_max_abs_s,
        marker_span_s=base_result.marker_span_s,
        ambiguous=base_result.ambiguous,
        prior_offset_s=base_result.prior_offset_s,
        radar_markers=base_result.radar_markers,
        rsp_markers=base_result.rsp_markers,
        diagnostics=diagnostics,
    )
    bindings, raw_input_graph = build_v2_raw_input_binding_state(
        subject, dataset_root
    )
    if bindings != preload_bindings or raw_input_graph != preload_raw_graph:
        raise RuntimeError(
            f"raw acquisition inputs changed while loading {subject.subject_id}"
        )
    receipt = build_sync_receipt(
        sync_result,
        session_id=subject.subject_id,
        config=sync_config,
        input_bindings=bindings,
    )
    receipt_path = session_dir / "sync_receipt.json"
    _atomic_json(receipt_path, receipt)
    # Re-read through the strict duplicate-key/tamper validator.
    validated_receipt = read_sync_receipt(receipt_path)
    manual_approval = _find_manual_approval(approval_dir, subject.subject_id)
    approval_artifact_path: Path | None = None
    if manual_approval is not None:
        manual_approval = validate_manual_approval(manual_approval, validated_receipt)
        approval_artifact_path = session_dir / "manual_approval.json"
        _atomic_json(approval_artifact_path, manual_approval)
    authorized = synchronization_is_authorized(
        validated_receipt, manual_approval=manual_approval
    )

    # Radar-only candidates are mapped through a proposal learned from noisy
    # markers and can occasionally land just beyond the recorded reference
    # interval.  The protocol decoder intentionally rejects out-of-domain
    # candidates, so clip the candidate *set* (not candidate timestamps) here.
    # This preserves the original measurements while keeping the decoder's
    # input contract explicit.
    candidates = tuple(
        candidate
        for candidate in protocol_boundary_candidates(sync_result)
        if 0.0 <= candidate.time_s <= biopac.duration_seconds
    )
    protocol = decode_ordered_protocol(
        duration_s=biopac.duration_seconds,
        config=protocol_config,
        candidates=candidates,
        session_record=session_record,
    )

    _atomic_npz(
        session_dir / "sync_signals.npz",
        radar_times_s=radar_times_s.astype(np.float64),
        radar_motion_robust_z=envelope.robust_z.astype(np.float32),
        radar_motion_valid_mask=envelope.valid_mask.astype(bool),
        rsp_marker_times_s=np.asarray([item.time_s for item in rsp_markers]),
        radar_marker_times_s=np.asarray([item.time_s for item in radar_markers]),
    )
    range_document: dict[str, Any] = {
        "status": "not_built",
        "physical_range_calibrated": False,
        "layout_selection_causal": False,
        "inference_feature_eligible": False,
    }
    if build_range_tracks:
        range_document = {
            "status": "built",
            **_build_range_tracks(
                feature_resample,
                output_path=session_dir / "range_tracks.npz",
                layout_maximum_frames=layout_maximum_frames,
            ),
        }

    review_reasons = list(sync_result.reasons)
    if not authorized:
        review_reasons.append("synchronization_not_authorized")
    if protocol.status != "auto":
        review_reasons.append(f"protocol_{protocol.status}")
    if range_document.get("selected_session_layout") == "unknown":
        review_reasons.append("range_layout_ambiguous")
    if range_document.get("status") == "built":
        review_reasons.append("range_layout_not_causal_inference_authority")
    if not measured_timing_eligible:
        review_reasons.append("measured_radar_timing_not_strictly_eligible")
    if radar_summary.get("metadata_warnings_eligible") is not True:
        review_reasons.append("radar_metadata_warning_policy_mismatch")
    if not biopac_reference_eligible:
        review_reasons.append("biopac_reference_metadata_warning")
    if timestamp_plateau_intervals:
        review_reasons.append("radar_timestamp_plateau_structurally_masked")
    review_reasons.append("independent_raw_derived_sync_verifier_absent")
    review_reasons.append("independent_protocol_decode_verifier_absent")
    alignment_eligible = bool(
        authorized and measured_timing_eligible and biopac_reference_eligible
    )
    # The current protocol artifact is self-hashed reconstruction evidence,
    # not a separately trusted replay from its raw sources.  Retain the decode
    # for diagnostic inspection but never let it authorize stage metrics.
    stage_metric_eligible = False
    range_feature_eligible = bool(
        range_document.get("status") == "built"
        and range_document.get("layout_selection_causal") is True
        and range_document.get("inference_feature_eligible") is True
    )
    # The base RR feature cache does not consume the unconfirmed I/Q-derived
    # range tracker or protocol stage labels.  Its scientific gate therefore
    # depends only on measured timing and an authorized radar/BIOPAC mapping.
    strict_cache_eligible = alignment_eligible
    scientific_eligible = strict_cache_eligible

    document: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": _utc_now(),
        "session_id": subject.subject_id,
        "physical_identity": identity_for_session(subject.subject_id),
        "usable": True,
        "reconstruction_context": dict(reconstruction_context),
        "scientific_eligible": scientific_eligible,
        "eligibility": {
            "measured_timing_eligible": measured_timing_eligible,
            "alignment_eligible": alignment_eligible,
            "stage_metric_eligible": stage_metric_eligible,
            "range_feature_eligible": range_feature_eligible,
            "strict_cache_eligible": strict_cache_eligible,
        },
        "raw_input_bindings_sha256": hashlib.sha256(
            json.dumps(bindings, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
        "raw_input_bindings": bindings,
        "raw_input_graph": raw_input_graph,
        "raw_input_graph_sha256": _canonical_sha256(raw_input_graph),
        "sensor_summary": {
            "radar": radar_summary,
            "biopac": biopac.summary(),
        },
        "synchronization": {
            **sync_result.to_dict(),
            "authorized": authorized,
            "receipt": str(receipt_path.relative_to(output_root)),
            "receipt_sha256": _sha256_file(receipt_path),
            "receipt_content_sha256": receipt["content_sha256"],
            "manual_approval_present": manual_approval is not None,
            "manual_approval": (
                None
                if approval_artifact_path is None
                else str(approval_artifact_path.relative_to(output_root))
            ),
            "manual_approval_file_sha256": (
                None
                if approval_artifact_path is None
                else _sha256_file(approval_artifact_path)
            ),
            "manual_approval_content_sha256": (
                None if manual_approval is None else manual_approval.get("content_sha256")
            ),
            "match_count": len(sync_result.matches),
        },
        "protocol": protocol.to_dict(),
        "protocol_contract": {
            "schema_version": protocol_config.schema_version,
            "time_basis": protocol_config.time_basis,
            "window_assignment": asdict(protocol_config.window_assignment),
            "annotation_inference_feature_allowed": False,
        },
        "session_record": None if session_record is None else session_record.to_dict(),
        "range_tracking": range_document,
        "annotation_contract": ANNOTATION_USAGE_CONTRACT,
        "review_reasons": sorted(set(review_reasons)),
    }
    document["content_sha256"] = canonical_content_sha256(document)
    _atomic_json(manifest_path, document)
    if build_review_plot:
        _plot_review(
            session_dir / "review.png",
            session_id=subject.subject_id,
            rsp=np.asarray(biopac.rsp),
            rsp_times_s=rsp_times_s,
            radar_times_s=radar_times_s,
            radar_motion_z=envelope.robust_z,
            sync_result=sync_result,
            protocol=protocol,
        )
    return document


def _resolve_publication_artifact(
    base: Path,
    relative: Any,
    *,
    label: str,
) -> Path:
    if not isinstance(relative, str) or not relative:
        raise RuntimeError(f"{label} path is missing")
    path = (base / relative).resolve()
    try:
        path.relative_to(base.resolve())
    except ValueError as error:
        raise RuntimeError(f"{label} escapes its artifact root") from error
    if not path.is_file():
        raise RuntimeError(f"{label} is missing: {path}")
    return path


def _validate_session_publication(
    subject: SubjectManifest,
    in_memory_document: Mapping[str, Any],
    *,
    dataset_root: Path,
    output_root: Path,
    approval_dir: Path | None,
    reconstruction_context: Mapping[str, Any],
) -> dict[str, Any]:
    """Revalidate one complete child graph immediately before root publish."""

    session_id = subject.subject_id
    session_dir = output_root / "sessions" / session_id
    manifest_path = session_dir / "session_manifest.json"
    disk_document = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(disk_document, dict):
        raise RuntimeError(f"{session_id} session manifest is not an object")
    if disk_document.get("content_sha256") != canonical_content_sha256(
        disk_document
    ):
        raise RuntimeError(f"{session_id} session manifest content hash mismatch")
    if _canonical_sha256(disk_document) != _canonical_sha256(
        dict(in_memory_document)
    ):
        raise RuntimeError(f"{session_id} in-memory/session manifest mismatch")
    if disk_document.get("reconstruction_context") != dict(
        reconstruction_context
    ):
        raise RuntimeError(f"{session_id} reconstruction context changed")
    if disk_document.get("usable") is not subject.usable:
        raise RuntimeError(f"{session_id} raw usability statement changed")
    if not subject.usable:
        return disk_document

    current_bindings, current_raw_graph = build_v2_raw_input_binding_state(
        subject, dataset_root
    )
    if disk_document.get("raw_input_bindings") != current_bindings:
        raise RuntimeError(f"{session_id} raw input bindings changed")
    if disk_document.get("raw_input_bindings_sha256") != _canonical_sha256(
        current_bindings
    ):
        raise RuntimeError(f"{session_id} raw input binding hash mismatch")
    if disk_document.get("raw_input_graph") != current_raw_graph:
        raise RuntimeError(f"{session_id} raw input graph changed")
    if disk_document.get("raw_input_graph_sha256") != _canonical_sha256(
        current_raw_graph
    ):
        raise RuntimeError(f"{session_id} raw input graph hash mismatch")

    synchronization = disk_document.get("synchronization")
    if not isinstance(synchronization, dict):
        raise RuntimeError(f"{session_id} synchronization document is missing")
    receipt_path = _resolve_publication_artifact(
        output_root,
        synchronization.get("receipt"),
        label=f"{session_id} synchronization receipt",
    )
    receipt = read_sync_receipt(receipt_path)
    if (
        synchronization.get("receipt_sha256") != _sha256_file(receipt_path)
        or synchronization.get("receipt_content_sha256")
        != receipt.get("content_sha256")
    ):
        raise RuntimeError(f"{session_id} synchronization receipt binding changed")
    if receipt.get("input_bindings") != current_bindings:
        raise RuntimeError(f"{session_id} receipt/raw input binding mismatch")

    approval_present = synchronization.get("manual_approval_present") is True
    source_approval = _find_manual_approval(approval_dir, session_id)
    if source_approval is not None:
        source_approval = validate_manual_approval(source_approval, receipt)
    if approval_present != (source_approval is not None):
        raise RuntimeError(f"{session_id} source approval presence changed")
    if approval_present:
        approval_path = _resolve_publication_artifact(
            output_root,
            synchronization.get("manual_approval"),
            label=f"{session_id} manual approval",
        )
        approval_document = json.loads(approval_path.read_text(encoding="utf-8"))
        approval_document = validate_manual_approval(approval_document, receipt)
        if (
            synchronization.get("manual_approval_file_sha256")
            != _sha256_file(approval_path)
            or synchronization.get("manual_approval_content_sha256")
            != approval_document.get("content_sha256")
            or approval_document.get("content_sha256")
            != source_approval.get("content_sha256")
        ):
            raise RuntimeError(f"{session_id} manual approval binding changed")
    elif any(
        synchronization.get(key) is not None
        for key in (
            "manual_approval",
            "manual_approval_file_sha256",
            "manual_approval_content_sha256",
        )
    ):
        raise RuntimeError(f"{session_id} absent approval has non-null bindings")

    range_document = disk_document.get("range_tracking")
    if not isinstance(range_document, dict):
        raise RuntimeError(f"{session_id} range-tracking document is missing")
    if range_document.get("status") == "built":
        range_path = _resolve_publication_artifact(
            session_dir,
            range_document.get("artifact"),
            label=f"{session_id} range-track artifact",
        )
        if range_document.get("artifact_sha256") != _sha256_file(range_path):
            raise RuntimeError(f"{session_id} range-track artifact changed")
    return disk_document


def main() -> int:
    args = parse_args()
    dataset_root = args.dataset_root.resolve()
    output_root = args.output_dir.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    sync_config_path = args.sync_config.resolve()
    protocol_config_path = args.protocol_config.resolve()
    spreadsheet_path = args.spreadsheet.resolve()
    cohort_authority_path = args.cohort_authority.resolve()
    parsed_input_hashes = {
        "sync_config_sha256": _sha256_file(sync_config_path),
        "protocol_config_sha256": _sha256_file(protocol_config_path),
        "spreadsheet_sha256": _sha256_file(spreadsheet_path),
        "cohort_authority_sha256": _sha256_file(cohort_authority_path),
    }
    sync_config = load_synchronization_config(
        sync_config_path,
        expected_sha256=parsed_input_hashes["sync_config_sha256"],
    )
    protocol_config = load_protocol_config(
        protocol_config_path,
        expected_sha256=parsed_input_hashes["protocol_config_sha256"],
    )
    cohort_authority = load_acquisition_cohort_authority(
        cohort_authority_path,
        expected_sha256=parsed_input_hashes["cohort_authority_sha256"],
    )
    records = load_dataset_issue_records(
        spreadsheet_path,
        dataset_root=dataset_root,
        config=protocol_config,
        expected_sha256=parsed_input_hashes["spreadsheet_sha256"],
    )
    for key, source_path in (
        ("sync_config_sha256", sync_config_path),
        ("protocol_config_sha256", protocol_config_path),
        ("spreadsheet_sha256", spreadsheet_path),
        ("cohort_authority_sha256", cohort_authority_path),
    ):
        if _sha256_file(source_path) != parsed_input_hashes[key]:
            raise RuntimeError(
                f"acquisition input changed while being parsed: {source_path}"
            )
    record_map = records_by_session(records)
    dataset = build_dataset_manifest(dataset_root)
    _validate_dataset_against_cohort_authority(dataset, cohort_authority)
    pipeline_paths = [
        Path(__file__),
        SOURCE_ROOT / "snn_rr/acquisition_contract.py",
        SOURCE_ROOT / "snn_rr/data.py",
        SOURCE_ROOT / "snn_rr/synchronization.py",
        SOURCE_ROOT / "snn_rr/acquisition_protocol.py",
        SOURCE_ROOT / "snn_rr/radar_timing.py",
        SOURCE_ROOT / "snn_rr/range_tracking.py",
        SOURCE_ROOT / "snn_rr/preprocess.py",
    ]
    pipeline_digest = _sha256_paths(pipeline_paths)
    reconstruction_context = {
        "pipeline_sha256": pipeline_digest,
        **parsed_input_hashes,
        "cohort_authority_content_sha256": cohort_authority.content_sha256,
        "subjects_filter_applied": args.subjects is not None,
        "build_range_tracks": not args.skip_range_tracks,
        "layout_maximum_frames": args.layout_maximum_frames,
    }
    expected_session_ids = cohort_authority.expected_session_ids
    expected_usable_session_ids = cohort_authority.expected_usable_session_ids
    selected = dataset.subjects
    if args.subjects:
        wanted = set(args.subjects)
        selected = tuple(item for item in selected if item.subject_id in wanted)
        missing = wanted - {item.subject_id for item in selected}
        if missing:
            raise KeyError(f"unknown subjects: {sorted(missing)}")

    documents: list[dict[str, Any]] = []
    for subject in selected:
        print(f"[{subject.subject_id}] reconstructing", flush=True)
        documents.append(
            reconstruct_subject(
                subject,
                dataset_root=dataset_root,
                output_root=output_root,
                sync_config=sync_config,
                protocol_config=protocol_config,
                session_record=record_map.get(subject.subject_id),
                approval_dir=None if args.approval_dir is None else args.approval_dir.resolve(),
                build_range_tracks=not args.skip_range_tracks,
                build_review_plot=not args.skip_review_plots,
                layout_maximum_frames=args.layout_maximum_frames,
                reconstruction_context=reconstruction_context,
                force=args.force,
            )
        )

    if _sha256_paths(pipeline_paths) != pipeline_digest:
        raise RuntimeError("acquisition pipeline source changed during reconstruction")
    for path, key in (
        (sync_config_path, "sync_config_sha256"),
        (protocol_config_path, "protocol_config_sha256"),
        (spreadsheet_path, "spreadsheet_sha256"),
        (cohort_authority_path, "cohort_authority_sha256"),
    ):
        if _sha256_file(path) != reconstruction_context[key]:
            raise RuntimeError(f"acquisition input changed during reconstruction: {path}")
    final_dataset = build_dataset_manifest(dataset_root)
    _validate_dataset_against_cohort_authority(
        final_dataset, cohort_authority
    )
    if _canonical_sha256(final_dataset.to_dict()) != _canonical_sha256(
        dataset.to_dict()
    ):
        raise RuntimeError("dataset manifest changed during reconstruction")
    final_documents = [
        _validate_session_publication(
            subject,
            document,
            dataset_root=dataset_root,
            output_root=output_root,
            approval_dir=(
                None if args.approval_dir is None else args.approval_dir.resolve()
            ),
            reconstruction_context=reconstruction_context,
        )
        for subject, document in zip(selected, documents, strict=True)
    ]
    documents = final_documents

    session_rows: list[dict[str, Any]] = []
    stage_rows: list[dict[str, Any]] = []
    marker_rows: list[dict[str, Any]] = []
    for document in documents:
        session_row, current_stages, current_markers = _session_rows(document)
        session_rows.append(session_row)
        stage_rows.extend(current_stages)
        marker_rows.extend(current_markers)
    _atomic_csv(output_root / "sessions.csv", pd.DataFrame(session_rows))
    _atomic_csv(output_root / "stages.csv", pd.DataFrame(stage_rows))
    _atomic_csv(output_root / "markers.csv", pd.DataFrame(marker_rows))

    selected_session_ids = tuple(item.subject_id for item in selected)
    document_session_ids = tuple(str(item.get("session_id")) for item in documents)
    selection_scope = (
        "full_cohort"
        if args.subjects is None
        and selected_session_ids == expected_session_ids
        else "diagnostic_subset"
    )
    execution_complete = document_session_ids == selected_session_ids
    full_cohort_complete = bool(
        execution_complete
        and selection_scope == "full_cohort"
        and selected_session_ids == expected_session_ids
    )
    usable = [item for item in documents if item.get("usable") is True]
    authorized = [
        item
        for item in usable
        if item.get("synchronization", {}).get("authorized") is True
    ]
    eligible = [item for item in usable if item.get("scientific_eligible") is True]
    root_scientific_eligible = bool(
        full_cohort_complete and usable and len(eligible) == len(usable)
    )
    strict_failure_reasons: list[str] = []
    if selection_scope != "full_cohort":
        strict_failure_reasons.append("diagnostic_subset_reconstruction")
    if not execution_complete:
        strict_failure_reasons.append("selected_execution_incomplete")
    if not full_cohort_complete:
        strict_failure_reasons.append("full_cohort_incomplete")
    if any(
        item.get("eligibility", {}).get("measured_timing_eligible") is not True
        for item in usable
    ):
        strict_failure_reasons.append("not_all_usable_sessions_have_measured_timing")
    if len(authorized) != len(usable):
        strict_failure_reasons.append("not_all_usable_sessions_have_authorized_sync")
    if len(eligible) != len(usable):
        strict_failure_reasons.append("not_all_usable_sessions_are_strict_cache_eligible")
    strict_failure_reasons.extend(
        (
            "independent_raw_derived_sync_verifier_absent",
            "independent_protocol_decode_verifier_absent",
        )
    )
    root: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": _utc_now(),
        "dataset_root": str(dataset_root),
        "dataset_private_physiological_data": True,
        "sync_config": str(sync_config_path),
        "sync_config_sha256": _sha256_file(sync_config_path),
        "protocol_config": str(protocol_config_path),
        "protocol_config_sha256": _sha256_file(protocol_config_path),
        "spreadsheet": str(spreadsheet_path),
        "spreadsheet_sha256": _sha256_file(spreadsheet_path),
        "cohort_authority": str(cohort_authority_path),
        "cohort_authority_sha256": _sha256_file(cohort_authority_path),
        "cohort_authority_content_sha256": cohort_authority.content_sha256,
        "cohort_authority_schema": "snn_rr.acquisition_cohort_authority.v1",
        "pipeline_sha256": pipeline_digest,
        "dataset_manifest_sha256": _canonical_sha256(dataset.to_dict()),
        "expected_session_ids": list(expected_session_ids),
        "expected_session_ids_sha256": _canonical_sha256(expected_session_ids),
        "expected_usable_session_ids": list(expected_usable_session_ids),
        "expected_usable_session_ids_sha256": _canonical_sha256(
            expected_usable_session_ids
        ),
        "excluded_sessions": [
            {"session_id": session_id, "reason": reason}
            for session_id, reason in cohort_authority.excluded_sessions
        ],
        "excluded_sessions_sha256": _canonical_sha256(
            [
                {"session_id": session_id, "reason": reason}
                for session_id, reason in cohort_authority.excluded_sessions
            ]
        ),
        "session_identities": [
            {
                "session_id": session_id,
                "physical_identity": identity,
            }
            for session_id, identity in cohort_authority.session_identities
        ],
        "session_identities_sha256": _canonical_sha256(
            [
                {
                    "session_id": session_id,
                    "physical_identity": identity,
                }
                for session_id, identity in cohort_authority.session_identities
            ]
        ),
        "expected_physical_identities": list(
            cohort_authority.expected_physical_identities
        ),
        "expected_physical_identities_sha256": _canonical_sha256(
            cohort_authority.expected_physical_identities
        ),
        "selected_session_ids": list(selected_session_ids),
        "selected_session_ids_sha256": _canonical_sha256(selected_session_ids),
        "selection_scope": selection_scope,
        "subjects_filter_applied": args.subjects is not None,
        "dataset_session_count": len(expected_session_ids),
        "dataset_usable_session_count": len(expected_usable_session_ids),
        "dataset_physical_identity_count": len(
            cohort_authority.expected_physical_identities
        ),
        "selected_session_count": len(selected_session_ids),
        "session_count": len(documents),
        "usable_session_count": len(usable),
        "sync_authorized_session_count": len(authorized),
        "scientific_eligible_session_count": len(eligible),
        "execution_complete": execution_complete,
        "full_cohort_complete": full_cohort_complete,
        "complete": full_cohort_complete,
        "scientific_eligible": root_scientific_eligible,
        "strict_failure_reasons": sorted(set(strict_failure_reasons)),
        "annotation_contract": ANNOTATION_USAGE_CONTRACT,
        "sessions": [
            {
                "session_id": item["session_id"],
                "physical_identity": item["physical_identity"],
                "usable": item.get("usable", False),
                "scientific_eligible": item.get("scientific_eligible", False),
                "manifest": f"sessions/{item['session_id']}/session_manifest.json",
                "manifest_sha256": _sha256_file(
                    output_root
                    / "sessions"
                    / str(item["session_id"])
                    / "session_manifest.json"
                ),
                "content_sha256": item.get("content_sha256"),
            }
            for item in documents
        ],
    }
    root["content_sha256"] = canonical_content_sha256(root)
    _atomic_json(output_root / "manifest.json", root)
    print(
        f"Reconstructed {len(documents)} sessions: {len(usable)} usable, "
        f"{len(authorized)} sync-authorized, {len(eligible)} scientific-eligible",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
