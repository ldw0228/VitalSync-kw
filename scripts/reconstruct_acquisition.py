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
from snn_rr.radar_timing import (  # noqa: E402
    block_mean_times,
    fuse_common_radar_timeline,
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


SCHEMA_VERSION = "snn_rr.acquisition_reconstruction.v1"
DEFAULT_OUTPUT = PROJECT_ROOT / "artifacts/acquisition/reconstruction_v1"


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
        default=PROJECT_ROOT / "configs/sync_marker_affine_v1.yaml",
    )
    parser.add_argument(
        "--protocol-config",
        type=Path,
        default=PROJECT_ROOT / "configs/acquisition_protocol_v1.yaml",
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


def _relative_or_absolute(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path.resolve())


def _input_bindings(subject: SubjectManifest, dataset_root: Path) -> dict[str, dict[str, Any]]:
    if not subject.usable or subject.selected_session is None or subject.biopac_path is None:
        return {}
    paths: list[tuple[str, Path]] = [("biopac_mat", Path(subject.biopac_path))]
    for radar_id in (1, 2, 3):
        stream = subject.selected_session.radars[radar_id]
        if stream.meta_path is not None:
            paths.append((f"radar{radar_id}_meta", Path(stream.meta_path)))
        for chunk_index, data_path in enumerate(stream.data_paths):
            paths.append((f"radar{radar_id}_data_{chunk_index:02d}", Path(data_path)))
    bindings: dict[str, dict[str, Any]] = {}
    for name, path in paths:
        bindings[name] = {
            "path": _relative_or_absolute(path, dataset_root),
            "bytes": path.stat().st_size,
            "sha256": _sha256_file(path),
        }
    return bindings


def _load_common_radar(subject: SubjectManifest) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    if subject.selected_session is None:
        raise ValueError(f"{subject.subject_id} has no selected radar session")
    payloads: list[np.ndarray] = []
    relative_times: list[np.ndarray] = []
    frame_sequences: list[np.ndarray] = []
    starts: list[float] = []
    timestamp_sources: list[str] = []
    timestamp_repairs: list[int] = []
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
    common = min(map(len, payloads))
    payload = np.stack([values[:common] for values in payloads])
    timeline = fuse_common_radar_timeline(relative_times, starts, frame_sequences)
    summary = {
        **timeline.summary,
        "timestamp_sources": timestamp_sources,
        "timestamp_repairs": timestamp_repairs,
    }
    return payload, timeline.times_s, summary


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


def _downsample_payload(payload: np.ndarray, times_s: np.ndarray, factor: int = 4) -> tuple[np.ndarray, np.ndarray]:
    usable = payload.shape[1] - payload.shape[1] % factor
    if usable < factor:
        raise ValueError("radar recording is too short to downsample")
    downsampled = payload[:, :usable].reshape(
        payload.shape[0], usable // factor, factor, payload.shape[2]
    ).mean(axis=2, dtype=np.float32)
    downsampled_times = block_mean_times(times_s, factor)
    return downsampled, downsampled_times


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
    payload: np.ndarray,
    radar_times_s: np.ndarray,
    *,
    output_path: Path,
    layout_maximum_frames: int,
) -> dict[str, Any]:
    downsampled, times = _downsample_payload(payload, radar_times_s)
    arrays: dict[str, np.ndarray] = {"radar_times_s": times.astype(np.float64)}
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
    session_row = {
        "session_id": document["session_id"],
        "usable": document.get("usable", False),
        "scientific_eligible": document.get("scientific_eligible", False),
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
    force: bool,
) -> dict[str, Any]:
    session_dir = output_root / "sessions" / subject.subject_id
    manifest_path = session_dir / "session_manifest.json"
    if manifest_path.is_file() and not force:
        document = json.loads(manifest_path.read_text(encoding="utf-8"))
        if document.get("schema_version") == SCHEMA_VERSION:
            return document
        raise ValueError(f"existing incompatible artifact: {manifest_path}; use --force")

    if not subject.usable:
        document = {
            "schema_version": SCHEMA_VERSION,
            "session_id": subject.subject_id,
            "usable": False,
            "scientific_eligible": False,
            "reason": "missing paired three-radar/BIOPAC recording",
            "session_record": None if session_record is None else session_record.to_dict(),
            "review_reasons": ["raw_session_unusable"],
        }
        document["content_sha256"] = canonical_content_sha256(document)
        _atomic_json(manifest_path, document)
        return document

    biopac = load_biopac_mat(subject.biopac_path, strict=False)
    payload, radar_times_s, radar_summary = _load_common_radar(subject)
    rsp_times_s = np.arange(len(biopac.rsp), dtype=np.float64) / biopac.sample_rate_hz
    prior = epoch_prior_offset_from_starts(
        radar_start_epoch_s=radar_summary["origin_epoch_s"],
        rsp_start_epoch_s=biopac.start_datetime.timestamp(),
    )
    envelope = robust_radar_motion_envelope(
        payload, radar_times_s=radar_times_s, config=sync_config
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
    timestamp_correction_s = float(
        radar_summary.get("maximum_timestamp_correction_s", 0.0)
    )
    if timestamp_correction_s > 0.050:
        timestamp_reasons = tuple(
            dict.fromkeys(
                [*sync_result.reasons, "large_radar_timestamp_plateau_reconstruction"]
            )
        )
        sync_result = SynchronizationResult(
            decision=(
                "manual_review_required"
                if sync_result.mapping is not None
                else "rejected"
            ),
            reasons=timestamp_reasons,
            mapping=sync_result.mapping,
            matches=sync_result.matches,
            confidence=min(sync_result.confidence, 0.49),
            residual_rmse_s=sync_result.residual_rmse_s,
            residual_max_abs_s=sync_result.residual_max_abs_s,
            marker_span_s=sync_result.marker_span_s,
            ambiguous=sync_result.ambiguous,
            prior_offset_s=sync_result.prior_offset_s,
            radar_markers=sync_result.radar_markers,
            rsp_markers=sync_result.rsp_markers,
            diagnostics={
                **sync_result.diagnostics,
                "timestamp_plateau_forced_manual_review": True,
                "maximum_timestamp_correction_s": timestamp_correction_s,
            },
        )

    bindings = _input_bindings(subject, dataset_root)
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
        rsp_marker_times_s=np.asarray([item.time_s for item in rsp_markers]),
        radar_marker_times_s=np.asarray([item.time_s for item in radar_markers]),
    )
    range_document: dict[str, Any] = {
        "status": "not_built",
        "physical_range_calibrated": False,
    }
    if build_range_tracks:
        range_document = {
            "status": "built",
            **_build_range_tracks(
                payload,
                radar_times_s,
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
    scientific_eligible = bool(
        authorized
        and protocol.status != "review"
        and range_document.get("selected_session_layout") == "split_halves"
    )

    document: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": _utc_now(),
        "session_id": subject.subject_id,
        "usable": True,
        "scientific_eligible": scientific_eligible,
        "raw_input_bindings_sha256": hashlib.sha256(
            json.dumps(bindings, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
        "raw_input_bindings": bindings,
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


def main() -> int:
    args = parse_args()
    dataset_root = args.dataset_root.resolve()
    output_root = args.output_dir.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    sync_config = load_synchronization_config(args.sync_config)
    protocol_config = load_protocol_config(args.protocol_config)
    records = load_dataset_issue_records(
        args.spreadsheet,
        dataset_root=dataset_root,
        config=protocol_config,
    )
    record_map = records_by_session(records)
    dataset = build_dataset_manifest(dataset_root)
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
                force=args.force,
            )
        )

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

    usable = [item for item in documents if item.get("usable")]
    authorized = [
        item for item in usable if item.get("synchronization", {}).get("authorized")
    ]
    eligible = [item for item in usable if item.get("scientific_eligible")]
    root: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": _utc_now(),
        "dataset_root": str(dataset_root),
        "dataset_private_physiological_data": True,
        "sync_config": str(args.sync_config.resolve()),
        "sync_config_sha256": _sha256_file(args.sync_config),
        "protocol_config": str(args.protocol_config.resolve()),
        "protocol_config_sha256": _sha256_file(args.protocol_config),
        "spreadsheet": str(args.spreadsheet.resolve()),
        "spreadsheet_sha256": _sha256_file(args.spreadsheet),
        "pipeline_sha256": _sha256_paths(
            [
                Path(__file__),
                SOURCE_ROOT / "snn_rr/synchronization.py",
                SOURCE_ROOT / "snn_rr/acquisition_protocol.py",
                SOURCE_ROOT / "snn_rr/range_tracking.py",
            ]
        ),
        "session_count": len(documents),
        "usable_session_count": len(usable),
        "sync_authorized_session_count": len(authorized),
        "scientific_eligible_session_count": len(eligible),
        "complete": len(documents) == len(selected),
        "scientific_eligible": bool(usable and len(eligible) == len(usable)),
        "strict_failure_reasons": (
            []
            if usable and len(eligible) == len(usable)
            else ["not_all_usable_sessions_have_authorized_sync_and_reviewed_protocol"]
        ),
        "annotation_contract": ANNOTATION_USAGE_CONTRACT,
        "sessions": [
            {
                "session_id": item["session_id"],
                "usable": item.get("usable", False),
                "scientific_eligible": item.get("scientific_eligible", False),
                "manifest": f"sessions/{item['session_id']}/session_manifest.json",
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
