from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import snn_rr.acquisition_contract as acquisition_contract_module
import snn_rr.cache as cache_module
from snn_rr.acquisition_contract import load_acquisition_cohort_authority
from snn_rr.acquisition_protocol import load_protocol_config
from snn_rr.data import build_dataset_manifest
from snn_rr.cache import (
    ACQUISITION_CACHE_SCHEMA_VERSION,
    ACQUISITION_CACHE_SCHEMA_VERSION_V2,
    REQUIRED_ACQUISITION_ANNOTATION_COLUMNS,
    load_feature_cache,
)
from snn_rr.preprocess import identity_for_session
from snn_rr.synchronization import (
    MarkerCandidate,
    MarkerMatch,
    SynchronizationConfig,
    SynchronizationResult,
    TimeMapping,
    build_sync_receipt,
)


_RECONSTRUCTION_CONTEXT = {
    "pipeline_sha256": "a" * 64,
    "sync_config_sha256": "b" * 64,
    "protocol_config_sha256": "c" * 64,
    "spreadsheet_sha256": "d" * 64,
    "build_range_tracks": False,
    "layout_maximum_frames": 5_000_000,
}

_SESSION_ID = "S12_KDH"
_COHORT_AUTHORITY_PATH = (
    Path(__file__).resolve().parents[1]
    / "configs"
    / "acquisition_cohort_v1.yaml"
)
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_PROTOCOL_CONFIG_PATH = _PROJECT_ROOT / "configs" / "acquisition_protocol_v1.yaml"
_PROTOCOL_CONFIG = load_protocol_config(_PROTOCOL_CONFIG_PATH)
_SPREADSHEET_PATH = _PROJECT_ROOT / "HAI_EXPERIMENT" / "Dataset_issue.xlsx"
_DATASET_ROOT = _PROJECT_ROOT / "HAI_EXPERIMENT"
_DISCOVERED_DATASET = build_dataset_manifest(_DATASET_ROOT)
_REAL_SYNC_AUTHORIZER = acquisition_contract_module.synchronization_is_authorized
_REAL_PROTOCOL_VERIFIER = (
    acquisition_contract_module._protocol_decode_has_independent_verification
)


@pytest.fixture(autouse=True)
def _reuse_stable_dataset_discovery(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "snn_rr.acquisition_contract.build_dataset_manifest",
        lambda _root: _DISCOVERED_DATASET,
    )
    monkeypatch.setattr(
        "snn_rr.acquisition_contract._validate_v2_raw_input_graph",
        lambda *_args, **_kwargs: None,
    )
    # Strict-cache fixtures model a future independently verified acquisition
    # generation.  The current production v1/v2 boundary is tested in
    # test_acquisition_contract_v2 and remains fail-closed.
    monkeypatch.setattr(
        acquisition_contract_module,
        "synchronization_is_authorized",
        lambda *_args, **_kwargs: True,
    )
    monkeypatch.setattr(
        acquisition_contract_module,
        "_protocol_decode_has_independent_verification",
        lambda *_args, **_kwargs: True,
    )


def _content_sha256(document: dict[str, object]) -> str:
    payload = dict(document)
    payload.pop("content_sha256", None)
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _value_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _write_json(path: Path, document: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, sort_keys=True), encoding="utf-8")


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _half_open_index(time_s: float, sample_rate_hz: float = 250.0) -> int:
    coordinate = float(time_s) * float(sample_rate_hz)
    nearest = float(np.rint(coordinate))
    tolerance = max(
        1.0e-9,
        8.0 * abs(float(np.spacing(max(abs(coordinate), 1.0)))),
    )
    return math.ceil(
        nearest if abs(coordinate - nearest) <= tolerance else coordinate
    )


def _sync_receipt(
    session_id: str,
    *,
    authorized: bool,
    mapping_scale: float = 1.0,
    config: SynchronizationConfig | None = None,
) -> dict[str, object]:
    radar_times = (0.0, 200.0, 400.0)
    rsp_times = tuple(2.0 + mapping_scale * value for value in radar_times)
    radar_markers = tuple(
        MarkerCandidate(index=index, time_s=time, score=10.0, source="motion")
        for index, time in enumerate(radar_times)
    )
    rsp_markers = tuple(
        MarkerCandidate(index=index, time_s=time, score=10.0, source="fixed_high")
        for index, time in enumerate(rsp_times)
    )
    result = SynchronizationResult(
        decision="accepted" if authorized else "manual_review_required",
        reasons=() if authorized else ("confidence_gate_failed",),
        mapping=TimeMapping(
            mode="constant" if mapping_scale == 1.0 else "affine",
            offset_s=2.0,
            scale=mapping_scale,
        ),
        matches=tuple(
            MarkerMatch(
                radar_index=index,
                rsp_index=index,
                radar_time_s=radar_time,
                rsp_time_s=rsp_time,
                residual_s=0.0,
            )
            for index, (radar_time, rsp_time) in enumerate(
                zip(radar_times, rsp_times, strict=True)
            )
        ),
        confidence=0.95 if authorized else 0.5,
        residual_rmse_s=0.0,
        residual_max_abs_s=0.0,
        marker_span_s=400.0,
        ambiguous=False,
        prior_offset_s=2.0,
        radar_markers=radar_markers,
        rsp_markers=rsp_markers,
        diagnostics={"synthetic_test_fixture": True},
    )
    return build_sync_receipt(
        result,
        session_id=session_id,
        config=SynchronizationConfig() if config is None else config,
        input_bindings={
            "signals": {
                "path": f"raw/{session_id}.bin",
                "sha256": hashlib.sha256(session_id.encode("utf-8")).hexdigest(),
            }
        },
        created_at_utc="2026-08-30T00:00:00Z",
    )


def _resampling_evidence() -> dict[str, object]:
    content_hashes = {
        "hash_schema_version": "snn_rr.canonical_ndarray_sha256.v1",
        "corrected_input_values_sha256": "1" * 64,
        "aligned_input_time_coordinates_sha256": "2" * 64,
        "frame_sequences_sha256": "3" * 64,
        "output_times_sha256": "4" * 64,
        "output_values_sha256": "5" * 64,
        "valid_mask_sha256": "6" * 64,
        "sample_counts_sha256": "7" * 64,
    }
    summary: dict[str, object] = {
        "schema_version": "snn_rr.causal_uniform_radar_resample.v1",
        "aggregation": "half_open_interval_arithmetic_mean",
        "causal": True,
        "timestamp_semantics": "right_edge_exclusive",
        "invalid_value_policy": "exact_zero_with_structural_mask",
        "gap_policy": "mask",
        "output_rate_hz": 10.0,
        "interval_s": 0.1,
        "max_gap_s": 0.050,
        "time_arithmetic": {
            "start_epoch_arithmetic": "integer_millisecond_fixed_point",
            "start_epoch_precision_s": 0.001,
            "start_offset_cancellation_avoided": True,
            "start_offset_coordinates_exact": True,
            "start_offset_quantization_max_abs_s": 0.0,
            "bin_membership_arithmetic": "integer_nanosecond_fixed_point",
            "coordinate_ticks_per_second": 1_000_000_000,
            "coordinate_quantization_policy": (
                "round_to_nearest_nanosecond_ties_to_even"
            ),
            "timestamp_quantization_max_abs_s": 0.0,
            "interval_quantization_max_abs_s": 0.0,
            "max_gap_quantization_max_abs_s": 0.0,
            "arithmetic_policy_selected_from_timestamp_values": False,
            "half_open_boundary_exact": True,
        },
        "output_interval_count": 1_200,
        "all_views_valid_interval_count": 1_200,
        "any_view_invalid_interval_count": 0,
        "content_hashes": content_hashes,
        "transform_evidence_sha256": _value_sha256(content_hashes),
        "per_view": [
            {
                "view_index": view_index,
                "timestamp_source": "meta_v13",
                "original_frame_count": 4_800,
                "frame_count": 4_800,
                "leading_boundary_frames_trimmed": 0,
                "trailing_boundary_frames_trimmed": 0,
                "leading_boundary_duplicate_frame_count": 0,
                "trailing_boundary_duplicate_frame_count": 0,
                "unaccounted_payload_frame_count": 0,
                "boundary_plateau_policy": (
                    "retain_all_and_structurally_mask_affected_interval"
                ),
                "timestamp_coordinates_exact": True,
                "timestamp_quantization_max_abs_s": 0.0,
                "frame_accounting": {
                    "schema_version": "snn_rr.radar_frame_accounting.v1",
                    "coordinate_semantics": "half_open_integer_nanosecond",
                    "retained_input_frame_count": 4_800,
                    "categories": {
                        "outside_common_intersection_prefix_frame_count": 0,
                        "leading_partial_edge_frame_count": 0,
                        "assigned_to_output_intervals_frame_count": 4_800,
                        "trailing_partial_edge_frame_count": 0,
                        "outside_common_intersection_suffix_frame_count": 0,
                    },
                    "category_sum": 4_800,
                    "before_common_complete_support_frame_count": 0,
                    "after_common_complete_support_frame_count": 0,
                    "unaccounted_payload_frame_count": 0,
                    "categories_disjoint": True,
                    "coverage_complete": True,
                    "assigned_count_matches_sample_counts": True,
                },
                "valid_output_count": 1_200,
                "invalid_output_count": 0,
                "empty_interval_count": 0,
                "temporal_gap_interval_count": 0,
                "sequence_gap_interval_count": 0,
                "timestamp_plateau_interval_count": 0,
                "nonfinite_interval_count": 0,
                "timestamp_repair": {
                    "timestamp_plateau_count": 0,
                    "measured_tie_edge_count": 0,
                    "reconstructed_frame_count": 0,
                    "maximum_timestamp_correction_s": 0.0,
                    "reconstruction_method": "none",
                    "plateaus": [],
                },
            }
            for view_index in range(3)
        ],
    }
    return summary


def _warning_evidence(
    session_id: str, config: SynchronizationConfig
) -> dict[str, object]:
    allowlist = config.radar_metadata_warning_allowlist
    allowed = list(allowlist.get(session_id, ()))
    policy = {
        "schema": "snn_rr.radar_metadata_warning_policy.v1",
        "mode": "ordered_exact_list_per_session_per_view",
        "config_field": "radar_metadata_warning_allowlist",
        "session_id": session_id,
        "session_allowlist_declared": session_id in allowlist,
        "unlisted_session_policy": "require_no_warnings",
        "allowed_warnings_per_view": allowed,
    }
    views = [
        {
            "radar_id": radar_id,
            "warnings": allowed,
            "allowed_warnings": allowed,
            "exact_match": True,
        }
        for radar_id in (1, 2, 3)
    ]
    evidence = {"policy": policy, "views": views, "eligible": True}
    return {
        "metadata_warning_policy": policy,
        "metadata_warning_views": views,
        "metadata_warnings_eligible": True,
        "metadata_warning_evidence_sha256": _value_sha256(evidence),
    }


def _record_contract_evidence() -> dict[str, object]:
    views = [
        {
            "radar_id": radar_id,
            "chunks": [
                {
                    "chunk_index": 0,
                    "filename": f"radar{radar_id}.dat",
                    "bytes": 4_800 * 740,
                    "frame_count": 4_800,
                    "record_bytes": 740,
                    "payload_bin_count": 182,
                    "record_size_remainder_bytes": 0,
                    "zero_header_nonzero": 0,
                    "bin_count_invalid": 0,
                }
            ],
            "eligible": True,
        }
        for radar_id in (1, 2, 3)
    ]
    evidence = {
        "schema": "snn_rr.xethru_record_contract.v1",
        "record_bytes": 740,
        "payload_bin_count": 182,
        "views": views,
        "eligible": True,
    }
    return {
        "xethru_record_contract": evidence,
        "xethru_record_contract_evidence_sha256": _value_sha256(evidence),
    }


def _metadata(
    session_id: str,
    *,
    eligible: bool,
    sync_authorized: bool | None = None,
    mapping_scale: float = 1.0,
) -> pd.DataFrame:
    authorized = eligible if sync_authorized is None else sync_authorized
    radar_start_s = 8.0
    radar_end_s = 40.0
    reference_start_s = 2.0 + mapping_scale * radar_start_s
    reference_end_s = 2.0 + mapping_scale * radar_end_s
    values: dict[str, list[object]] = {
        "session_id": [session_id],
        "identity": [identity_for_session(session_id)],
        "window_number": [0],
        "window_start_s": [reference_start_s],
        "window_end_s": [reference_end_s],
        "reference_start_sample": [_half_open_index(reference_start_s)],
        "reference_end_sample": [_half_open_index(reference_end_s)],
        "reference_window_start_biopac_s": [reference_start_s],
        "reference_window_end_biopac_s": [reference_end_s],
        "radar_window_start_relative_s": [radar_start_s],
        "radar_window_end_relative_s": [radar_end_s],
        "sync_authorized": [authorized],
        "sync_confidence": [0.95 if authorized else 0.5],
        "alignment_scientific_eligible": [eligible],
        "acquisition_phase": ["phase1"],
        "acquisition_phase_name": [_PROTOCOL_CONFIG.stages[0].name],
        "acquisition_phase_status": ["auto"],
        "acquisition_phase_confidence": [0.95],
        "phase_overlap_fraction": [1.0],
        "transition_window": [False],
        "eligible_for_stage_metrics": [eligible],
        "phase7_assignment": [None],
        "acquisition_batch": ["v2"],
    }
    assert REQUIRED_ACQUISITION_ANNOTATION_COLUMNS <= set(values)
    return pd.DataFrame(values)


def _protocol_document(session_id: str) -> dict[str, object]:
    """Build a complete synthetic protocol that obeys the bound V1 config."""

    stages: list[dict[str, object]] = []
    stage_start = 0.0
    for spec in _PROTOCOL_CONFIG.stages:
        stage_end = stage_start + spec.nominal_duration_s
        stages.append(
            {
                "stage_id": spec.stage_id,
                "name": spec.name,
                "status": "auto",
                "confidence": 0.95,
                "start": {"time_s": stage_start},
                "end": {"time_s": stage_end},
                "duration_s": spec.nominal_duration_s,
            }
        )
        stage_start = stage_end
    return {
        "session_id": session_id,
        "annotation_schema_version": _PROTOCOL_CONFIG.schema_version,
        "duration_s": stage_start,
        "status": "auto",
        "confidence": 0.95,
        "annotation_inference_feature_allowed": False,
        "stages": stages,
    }


def _write_cache(
    tmp_path: Path,
    *,
    eligible: bool,
    version: int = 1,
    reconstruction_eligible: bool | None = None,
    expected_usable_session_ids: tuple[str, ...] | None = None,
    mapping_scale: float = 1.0,
) -> tuple[Path, Path]:
    source_eligible = (
        eligible if reconstruction_eligible is None else reconstruction_eligible
    )
    session_id = _SESSION_ID
    acquisition_root = tmp_path / "acquisition"
    reconstruction_schema = f"snn_rr.acquisition_reconstruction.v{version}"
    if version != 2:
        source_session_path = (
            acquisition_root
            / "sessions"
            / session_id
            / "session_manifest.json"
        )
        mapping = {"mode": "affine", "offset_s": 1.25, "scale": 1.0001}
        sync_receipt_hash = "1" * 64
        approval_hash = "3" * 64 if source_eligible else None
        range_hash = "4" * 64
        source_session = {
            "schema_version": reconstruction_schema,
            "session_id": session_id,
            "scientific_eligible": source_eligible,
            "synchronization": {
                "authorized": source_eligible,
                "receipt_content_sha256": sync_receipt_hash,
                "manual_approval_content_sha256": approval_hash,
                "mapping": mapping,
            },
            "range_tracking": {"artifact_sha256": range_hash},
            "protocol_contract": {"schema_version": "acquisition_protocol_v1"},
        }
        source_session["content_sha256"] = _content_sha256(source_session)
        _write_json(source_session_path, source_session)
        reconstruction = {
            "schema_version": reconstruction_schema,
            "complete": True,
            "scientific_eligible": source_eligible,
            "sessions": [
                {
                    "session_id": session_id,
                    "usable": True,
                    "manifest": f"sessions/{session_id}/session_manifest.json",
                    "content_sha256": source_session["content_sha256"],
                    "scientific_eligible": source_eligible,
                }
            ],
        }
        reconstruction["content_sha256"] = _content_sha256(reconstruction)
        reconstruction_path = acquisition_root / "manifest.json"
        _write_json(reconstruction_path, reconstruction)
        annotation_columns = sorted(REQUIRED_ACQUISITION_ANNOTATION_COLUMNS)
        session_contract = {
            "schema_version": ACQUISITION_CACHE_SCHEMA_VERSION,
            "acquisition_session_manifest_sha256": source_session[
                "content_sha256"
            ],
            "sync_receipt_content_sha256": sync_receipt_hash,
            "mapping_sha256": _value_sha256(mapping),
            "manual_approval_content_sha256": approval_hash,
            "protocol_annotation_schema_version": "acquisition_protocol_v1",
            "range_artifact_sha256": range_hash,
            "reference_alignment_mode": "authorized_marker_affine_v1",
            "scientific_eligible": eligible,
            "annotation_only_columns": annotation_columns,
        }
        cache = tmp_path / "cache"
        session_dir = cache / session_id
        session_dir.mkdir(parents=True)
        np.save(session_dir / "maps.npy", np.zeros((1, 3, 2, 4), dtype=np.float16))
        np.save(session_dir / "aux.npy", np.zeros((1, 2), dtype=np.float32))
        np.save(session_dir / "frequencies_hz.npy", np.asarray([0.1, 0.2]))
        _metadata(session_id, eligible=eligible).to_csv(
            session_dir / "metadata.csv", index=False
        )
        _write_json(
            session_dir / "manifest.json",
            {"acquisition_contract": session_contract},
        )
        _write_json(
            cache / "manifest.json",
            {
                "acquisition_contract": {
                    "schema_version": ACQUISITION_CACHE_SCHEMA_VERSION,
                    "reconstruction_manifest": str(reconstruction_path),
                    "reconstruction_content_sha256": reconstruction[
                        "content_sha256"
                    ],
                    "mode": "strict" if eligible else "diagnostic",
                    "annotation_only_columns": annotation_columns,
                    "scientific_eligible": eligible,
                },
                "sessions": [
                    {
                        "session_id": session_id,
                        "status": "ok",
                        "acquisition_contract": session_contract,
                    }
                ],
            },
        )
        return cache, reconstruction_path

    cohort = load_acquisition_cohort_authority(_COHORT_AUTHORITY_PATH)
    expected_session_ids = cohort.expected_session_ids
    expected_usable_ids = cohort.expected_usable_session_ids
    reconstruction_ids = (
        expected_session_ids if source_eligible else (session_id,)
    )
    subjects_filter_applied = not source_eligible
    sync_config = SynchronizationConfig()
    sync_config_path = acquisition_root / "sync_config.yaml"
    _write_json(sync_config_path, sync_config.to_dict())
    reconstruction_context = {
        **_RECONSTRUCTION_CONTEXT,
        "sync_config_sha256": _file_sha256(sync_config_path),
        "protocol_config_sha256": _file_sha256(_PROTOCOL_CONFIG_PATH),
        "spreadsheet_sha256": _file_sha256(_SPREADSHEET_PATH),
        "cohort_authority_sha256": cohort.file_sha256,
        "cohort_authority_content_sha256": cohort.content_sha256,
        "subjects_filter_applied": subjects_filter_applied,
    }
    resampling = _resampling_evidence()
    source_sessions: dict[str, dict[str, object]] = {}
    mappings: dict[str, dict[str, object]] = {}
    receipt_hashes: dict[str, str] = {}
    for current_id in reconstruction_ids:
        source_path = (
            acquisition_root
            / "sessions"
            / current_id
            / "session_manifest.json"
        )
        if current_id == "S24_KHJ":
            source = {
                "schema_version": reconstruction_schema,
                "session_id": current_id,
                "physical_identity": identity_for_session(current_id),
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
                "review_reasons": ["raw_session_unusable"],
            }
        else:
            current_scale = mapping_scale if current_id == session_id else 1.0
            receipt = _sync_receipt(
                current_id,
                authorized=source_eligible,
                mapping_scale=current_scale,
                config=sync_config,
            )
            receipt_path = source_path.parent / "sync_receipt.json"
            _write_json(receipt_path, receipt)
            result = receipt["result"]
            assert isinstance(result, dict)
            current_mapping = result["mapping"]
            assert isinstance(current_mapping, dict)
            raw_bindings = receipt["input_bindings"]
            source = {
                "schema_version": reconstruction_schema,
                "session_id": current_id,
                "physical_identity": identity_for_session(current_id),
                "usable": True,
                "reconstruction_context": dict(reconstruction_context),
                "raw_input_bindings": raw_bindings,
                "raw_input_bindings_sha256": _value_sha256(raw_bindings),
                "scientific_eligible": source_eligible,
                "eligibility": {
                    "measured_timing_eligible": True,
                    "alignment_eligible": source_eligible,
                    "stage_metric_eligible": source_eligible,
                    "range_feature_eligible": False,
                    "strict_cache_eligible": source_eligible,
                },
                "sensor_summary": {
                    "radar": {
                        **_warning_evidence(current_id, sync_config),
                        **_record_contract_evidence(),
                        "timestamp_sources": ["meta_v13"] * 3,
                        "sync_marker_resampling": resampling,
                        "feature_resampling": resampling,
                        "resampling_content_hashes": resampling["content_hashes"],
                        "resampling_transform_evidence_sha256": resampling[
                            "transform_evidence_sha256"
                        ],
                        "maximum_timestamp_correction_s": 0.0,
                        "unaccounted_payload_frame_count": 0,
                        "timestamp_plateau_interval_count": 0,
                        "measured_timing_eligible": True,
                    },
                    "biopac": {"sample_rate_hz": 250.0, "warnings": []},
                },
                "synchronization": {
                    **result,
                    "authorized": source_eligible,
                    "receipt": f"sessions/{current_id}/sync_receipt.json",
                    "receipt_sha256": _file_sha256(receipt_path),
                    "receipt_content_sha256": receipt["content_sha256"],
                    "manual_approval_present": False,
                    "manual_approval": None,
                    "manual_approval_file_sha256": None,
                    "manual_approval_content_sha256": None,
                    "match_count": len(result["matches"]),
                },
                "protocol": _protocol_document(current_id),
                "range_tracking": {
                    "status": "not_built",
                    "selected_session_layout": None,
                    "physical_range_calibrated": False,
                    "layout_selection_causal": False,
                    "inference_feature_eligible": False,
                },
                "protocol_contract": {
                    "schema_version": "acquisition_protocol_v1",
                    "time_basis": "seconds_from_biopac_start",
                    "window_assignment": {
                        "minimum_overlap_fraction": 0.8,
                        "transition_guard_s": 2.0,
                    },
                    "annotation_inference_feature_allowed": False,
                },
                "review_reasons": [],
            }
            mappings[current_id] = current_mapping
            receipt_hashes[current_id] = str(receipt["content_sha256"])
        source["content_sha256"] = _content_sha256(source)
        _write_json(source_path, source)
        source_sessions[current_id] = source

    claims = cohort.to_claims()
    reconstruction_full = reconstruction_ids == expected_session_ids
    reconstruction_entries = []
    for current_id in reconstruction_ids:
        source = source_sessions[current_id]
        source_path = (
            acquisition_root
            / "sessions"
            / current_id
            / "session_manifest.json"
        )
        reconstruction_entries.append(
            {
                "session_id": current_id,
                "physical_identity": identity_for_session(current_id),
                "usable": source["usable"],
                "manifest": f"sessions/{current_id}/session_manifest.json",
                "manifest_sha256": _file_sha256(source_path),
                "content_sha256": source["content_sha256"],
                "scientific_eligible": source["scientific_eligible"],
            }
        )
    reconstruction: dict[str, object] = {
        "schema_version": reconstruction_schema,
        **reconstruction_context,
        "sync_config": str(sync_config_path),
        "protocol_config": str(_PROTOCOL_CONFIG_PATH),
        "spreadsheet": str(_SPREADSHEET_PATH),
        "dataset_root": str(_DATASET_ROOT),
        "cohort_authority": str(_COHORT_AUTHORITY_PATH),
        "cohort_authority_schema": "snn_rr.acquisition_cohort_authority.v1",
        "dataset_manifest_sha256": _value_sha256(_DISCOVERED_DATASET.to_dict()),
        "expected_session_ids": list(expected_session_ids),
        "expected_session_ids_sha256": _value_sha256(list(expected_session_ids)),
        "expected_usable_session_ids": list(expected_usable_ids),
        "expected_usable_session_ids_sha256": _value_sha256(
            list(expected_usable_ids)
        ),
        "excluded_sessions": claims["excluded_sessions"],
        "excluded_sessions_sha256": _value_sha256(claims["excluded_sessions"]),
        "session_identities": claims["session_identities"],
        "session_identities_sha256": _value_sha256(claims["session_identities"]),
        "expected_physical_identities": claims["expected_physical_identities"],
        "expected_physical_identities_sha256": _value_sha256(
            claims["expected_physical_identities"]
        ),
        "selected_session_ids": list(reconstruction_ids),
        "selected_session_ids_sha256": _value_sha256(list(reconstruction_ids)),
        "dataset_session_count": 30,
        "dataset_usable_session_count": 29,
        "dataset_physical_identity_count": 18,
        "selected_session_count": len(reconstruction_ids),
        "selection_scope": (
            "full_cohort" if reconstruction_full else "diagnostic_subset"
        ),
        "subjects_filter_applied": subjects_filter_applied,
        "execution_complete": True,
        "full_cohort_complete": reconstruction_full,
        "complete": reconstruction_full,
        "session_count": len(reconstruction_ids),
        "usable_session_count": sum(
            source["usable"] is True for source in source_sessions.values()
        ),
        "sync_authorized_session_count": (
            len(expected_usable_ids) if reconstruction_full else 0
        ),
        "scientific_eligible_session_count": (
            len(expected_usable_ids) if reconstruction_full else 0
        ),
        "scientific_eligible": reconstruction_full,
        "sessions": reconstruction_entries,
    }
    reconstruction["content_sha256"] = _content_sha256(reconstruction)
    reconstruction_path = acquisition_root / "manifest.json"
    _write_json(reconstruction_path, reconstruction)

    annotation_columns = sorted(REQUIRED_ACQUISITION_ANNOTATION_COLUMNS)
    cache_ids = expected_usable_ids if eligible else (session_id,)
    cache = tmp_path / "cache"
    root_items: list[dict[str, object]] = []
    inventory_bindings: dict[str, str] = {}
    for current_id in cache_ids:
        source = source_sessions[current_id]
        current_mapping = mappings[current_id]
        current_scale = mapping_scale if current_id == session_id else 1.0
        session_contract: dict[str, object] = {
            "schema_version": ACQUISITION_CACHE_SCHEMA_VERSION_V2,
            "acquisition_session_manifest_sha256": source["content_sha256"],
            "sync_receipt_content_sha256": receipt_hashes[current_id],
            "mapping_sha256": _value_sha256(current_mapping),
            "manual_approval_content_sha256": None,
            "protocol_annotation_schema_version": "acquisition_protocol_v1",
            "range_artifact_sha256": None,
            "reference_alignment_mode": (
                "authorized_marker_affine_v1"
                if eligible
                else "diagnostic_unapproved_proposal_v1"
            ),
            "scientific_eligible": eligible,
            "annotation_only_columns": annotation_columns,
            "measured_timing_eligible": True,
            "alignment_eligible": source_eligible,
            "stage_metric_eligible": source_eligible,
            "range_feature_eligible": False,
            "strict_cache_eligible": source_eligible,
        }
        session_dir = cache / current_id
        session_dir.mkdir(parents=True)
        np.save(session_dir / "maps.npy", np.zeros((1, 3, 2, 4), dtype=np.float16))
        np.save(session_dir / "aux.npy", np.zeros((1, 2), dtype=np.float32))
        np.save(session_dir / "frequencies_hz.npy", np.asarray([0.1, 0.2]))
        np.save(
            session_dir / "radar_timing_valid_mask.npy",
            np.ones((1, 3, 320), dtype=np.bool_),
            allow_pickle=False,
        )
        cache_metadata = _metadata(
            current_id,
            eligible=eligible,
            sync_authorized=source_eligible,
            mapping_scale=current_scale,
        )
        cache_metadata.to_csv(session_dir / "metadata.csv", index=False)
        inventory: dict[str, object] = {}
        for name, filename, shape, dtype in (
            ("maps", "maps.npy", [1, 3, 2, 4], "float16"),
            ("aux", "aux.npy", [1, 2], "float32"),
            (
                "frequencies_hz",
                "frequencies_hz.npy",
                [2],
                "float64",
            ),
            (
                "radar_timing_valid_mask",
                "radar_timing_valid_mask.npy",
                [1, 3, 320],
                "bool",
            ),
        ):
            artifact = session_dir / filename
            inventory[name] = {
                "path": filename,
                "sha256": _file_sha256(artifact),
                "bytes": artifact.stat().st_size,
                "shape": shape,
                "dtype": dtype,
            }
        metadata_path = session_dir / "metadata.csv"
        inventory["metadata"] = {
            "path": "metadata.csv",
            "sha256": _file_sha256(metadata_path),
            "bytes": metadata_path.stat().st_size,
            "shape": [1, len(cache_metadata.columns)],
            "dtype": "csv",
        }
        session_manifest: dict[str, object] = {
            "acquisition_contract": session_contract,
            "config_sha256": "5" * 64,
            "pipeline_sha256": "6" * 64,
            "window_count": 1,
            "measured_window_support": {
                "timestamp_semantics": "right_edge_exclusive",
                "window_interval_count": 320,
                "window_duration_s": 32.0,
                "stride_interval_count": 10,
                "stride_duration_s": 1.0,
                "reference_sample_indexing": {
                    "sample_timestamp_semantics": "i / sample_rate_hz",
                    "support_membership": "start_s <= i / sample_rate_hz < end_s",
                    "slice_boundary_rule": "ceil_both_boundaries",
                    "near_integer_canonicalization": (
                        "abs(coordinate-rint(coordinate)) <= "
                        "max(1e-9,8*spacing(max(abs(coordinate),1)))"
                    ),
                },
            },
            "radar_timing_valid_mask_shape": [1, 3, 320],
            "radar_timing_invalid_interval_count": 0,
            "radar_timing_summary": resampling,
            "file_inventory": inventory,
            "inventory_sha256": _value_sha256(inventory),
        }
        session_manifest["content_sha256"] = _content_sha256(session_manifest)
        session_manifest_path = session_dir / "manifest.json"
        _write_json(session_manifest_path, session_manifest)
        inventory_bindings[current_id] = str(session_manifest["inventory_sha256"])
        root_items.append(
            {
                "session_id": current_id,
                "status": "ok",
                "acquisition_contract": session_contract,
                "inventory_sha256": session_manifest["inventory_sha256"],
                "config_sha256": "5" * 64,
                "pipeline_sha256": "6" * 64,
                "content_sha256": session_manifest["content_sha256"],
                "session_manifest_sha256": _file_sha256(session_manifest_path),
                "session_manifest_content_sha256": session_manifest[
                    "content_sha256"
                ],
            }
        )

    cache_full = eligible and tuple(cache_ids) == expected_usable_ids
    cache_subjects_filter_applied = not eligible
    root_contract = {
        "schema_version": ACQUISITION_CACHE_SCHEMA_VERSION_V2,
        "reconstruction_manifest": str(reconstruction_path),
        "reconstruction_content_sha256": reconstruction["content_sha256"],
        "cohort_authority_sha256": cohort.file_sha256,
        "cohort_authority_content_sha256": cohort.content_sha256,
        "mode": "strict" if eligible else "diagnostic",
        "annotation_only_columns": annotation_columns,
        "scientific_eligible": eligible,
        "subjects_filter_applied": cache_subjects_filter_applied,
        "selection_scope": "full_cohort" if cache_full else "diagnostic_subset",
        "reconstruction_full_cohort_complete": reconstruction_full,
        "full_cohort_complete": bool(cache_full and reconstruction_full),
        "expected_usable_session_ids": list(expected_usable_ids),
        "expected_usable_session_ids_sha256": _value_sha256(
            list(expected_usable_ids)
        ),
        "cache_usable_session_ids": list(cache_ids),
        "cache_usable_session_ids_sha256": _value_sha256(list(cache_ids)),
        "cache_inventory_aggregate_sha256": _value_sha256(inventory_bindings),
    }
    root_manifest: dict[str, object] = {
        "config_sha256": "5" * 64,
        "pipeline_sha256": "6" * 64,
        "subjects_filter_applied": cache_subjects_filter_applied,
        "acquisition_contract": root_contract,
        "sessions": root_items,
    }
    root_manifest["content_sha256"] = _content_sha256(root_manifest)
    _write_json(cache / "manifest.json", root_manifest)
    return cache, reconstruction_path


def _reseal_cache_files(cache: Path, session_id: str = _SESSION_ID) -> None:
    """Rebind an intentionally mutated synthetic cache without hiding semantics."""

    session_dir = cache / session_id
    session_path = session_dir / "manifest.json"
    session = json.loads(session_path.read_text(encoding="utf-8"))
    inventory = session["file_inventory"]
    for binding in inventory.values():
        path = session_dir / binding["path"]
        binding["sha256"] = _file_sha256(path)
        binding["bytes"] = path.stat().st_size
        if binding["dtype"] == "csv":
            binding["shape"] = list(pd.read_csv(path).shape)
        else:
            array = np.load(path, allow_pickle=False)
            binding["shape"] = list(array.shape)
            binding["dtype"] = str(array.dtype)
    session["inventory_sha256"] = _value_sha256(inventory)
    timing_mask = np.load(
        session_dir / "radar_timing_valid_mask.npy", allow_pickle=False
    )
    session["radar_timing_valid_mask_shape"] = list(timing_mask.shape)
    session["radar_timing_invalid_interval_count"] = int(
        np.size(timing_mask) - np.count_nonzero(timing_mask)
    )
    session["content_sha256"] = _content_sha256(session)
    _write_json(session_path, session)

    root_path = cache / "manifest.json"
    root = json.loads(root_path.read_text(encoding="utf-8"))
    root_item = next(
        item for item in root["sessions"] if item.get("session_id") == session_id
    )
    root_item["acquisition_contract"] = session["acquisition_contract"]
    root_item["inventory_sha256"] = session["inventory_sha256"]
    root_item["content_sha256"] = session["content_sha256"]
    root_item["session_manifest_sha256"] = _file_sha256(session_path)
    root_item["session_manifest_content_sha256"] = session["content_sha256"]
    root["acquisition_contract"]["cache_inventory_aggregate_sha256"] = (
        _value_sha256(
            {
                str(item["session_id"]): item["inventory_sha256"]
                for item in root["sessions"]
                if item.get("status") == "ok"
            }
        )
    )
    root["content_sha256"] = _content_sha256(root)
    _write_json(root_path, root)


def test_legacy_cache_default_behavior_is_unchanged(tmp_path: Path) -> None:
    cache = tmp_path / "legacy"
    session = cache / "LEGACY"
    session.mkdir(parents=True)
    np.save(session / "maps.npy", np.zeros((1, 3, 2, 4), dtype=np.float16))
    np.save(session / "aux.npy", np.zeros((1, 2), dtype=np.float32))
    np.save(session / "frequencies_hz.npy", np.asarray([0.1, 0.2]))
    pd.DataFrame({"session_id": ["LEGACY"]}).to_csv(session / "metadata.csv", index=False)
    _write_json(
        cache / "manifest.json",
        {"sessions": [{"session_id": "LEGACY", "status": "ok"}]},
    )

    loaded = load_feature_cache(cache)
    assert loaded.maps.shape == (1, 3, 2, 4)
    assert loaded.radar_timing_valid_mask is None
    assert loaded.provenance is not None
    assert loaded.provenance.classification == "legacy"
    with pytest.raises(ValueError, match="root acquisition_contract"):
        load_feature_cache(cache, require_acquisition_contract=True)


def test_valid_acquisition_contract_and_scientific_gate(tmp_path: Path) -> None:
    eligible_cache, _ = _write_cache(
        tmp_path / "eligible", eligible=True, version=2
    )
    loaded = load_feature_cache(
        eligible_cache,
        require_acquisition_contract=True,
        require_scientific_eligible=True,
    )
    assert loaded.metadata.loc[0, "acquisition_phase"] == "phase1"
    assert loaded.provenance is not None
    assert loaded.provenance.classification == "acquisition_scientific"
    assert loaded.provenance.scientific_eligible is True
    assert loaded.radar_timing_valid_mask is not None
    assert loaded.radar_timing_valid_mask.shape == (29, 3, 320)
    assert loaded.radar_timing_valid_mask.all()

    diagnostic_cache, _ = _write_cache(
        tmp_path / "diagnostic", eligible=False, version=2
    )
    diagnostic = load_feature_cache(
        diagnostic_cache, require_acquisition_contract=True
    )
    assert diagnostic.provenance is not None
    assert diagnostic.provenance.classification == "acquisition_diagnostic"
    with pytest.raises(ValueError, match="root is not scientifically eligible"):
        load_feature_cache(diagnostic_cache, require_scientific_eligible=True)

    downgraded_cache, _ = _write_cache(
        tmp_path / "downgraded",
        eligible=False,
        reconstruction_eligible=True,
        version=2,
    )
    downgraded = load_feature_cache(
        downgraded_cache, require_acquisition_contract=True
    )
    assert downgraded.provenance is not None
    assert downgraded.provenance.classification == "acquisition_diagnostic"

    historical_cache, _ = _write_cache(tmp_path / "historical", eligible=True)
    historical = load_feature_cache(historical_cache)
    assert historical.provenance is not None
    assert historical.provenance.classification == "acquisition_historical_v1"
    with pytest.raises(ValueError, match="version-2 full-cohort"):
        load_feature_cache(
            historical_cache,
            require_scientific_eligible=True,
        )


def test_contract_mixing_annotation_and_source_hash_tampering_fail_closed(
    tmp_path: Path,
) -> None:
    cache, reconstruction_path = _write_cache(tmp_path, eligible=True)
    session_manifest_path = cache / _SESSION_ID / "manifest.json"
    session_manifest = json.loads(session_manifest_path.read_text(encoding="utf-8"))
    del session_manifest["acquisition_contract"]
    _write_json(session_manifest_path, session_manifest)
    with pytest.raises(ValueError, match="legacy session mixed"):
        load_feature_cache(cache, require_acquisition_contract=True)

    cache, reconstruction_path = _write_cache(tmp_path / "columns", eligible=True)
    metadata_path = cache / _SESSION_ID / "metadata.csv"
    frame = pd.read_csv(metadata_path).drop(columns=["acquisition_phase_status"])
    frame.to_csv(metadata_path, index=False)
    with pytest.raises(ValueError, match="metadata columns missing"):
        load_feature_cache(cache, require_acquisition_contract=True)

    cache, reconstruction_path = _write_cache(tmp_path / "hash", eligible=True)
    reconstruction = json.loads(reconstruction_path.read_text(encoding="utf-8"))
    reconstruction["complete"] = False
    _write_json(reconstruction_path, reconstruction)
    with pytest.raises(ValueError, match="canonical content hash mismatch"):
        load_feature_cache(cache, require_acquisition_contract=True)


def test_acquisition_indicator_auto_enables_graph_validation(tmp_path: Path) -> None:
    cache, _ = _write_cache(tmp_path, eligible=False)
    session_manifest_path = cache / _SESSION_ID / "manifest.json"
    session_manifest = json.loads(session_manifest_path.read_text(encoding="utf-8"))
    del session_manifest["acquisition_contract"]
    _write_json(session_manifest_path, session_manifest)

    # No strict flag is supplied: the root acquisition indicator itself must
    # force fail-closed graph validation.
    with pytest.raises(ValueError, match="legacy session mixed"):
        load_feature_cache(cache)


def test_v2_exact_inventory_tampering_fails_before_load(tmp_path: Path) -> None:
    cache, _ = _write_cache(tmp_path, eligible=True, version=2)
    np.save(
        cache / _SESSION_ID / "aux.npy",
        np.ones((1, 2), dtype=np.float32),
        allow_pickle=False,
    )
    with pytest.raises(ValueError, match="file_inventory SHA-256 mismatch"):
        load_feature_cache(cache, require_scientific_eligible=True)


def test_v2_inventory_generation_change_during_load_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache, _ = _write_cache(tmp_path, eligible=True, version=2)
    original = cache_module._inventory_sha256
    calls = 0

    def changing_inventory(root: Path, files: list[Path]) -> tuple[str, int]:
        nonlocal calls
        calls += 1
        digest, count = original(root, files)
        return (("0" * 64) if calls == 2 else digest, count)

    monkeypatch.setattr(cache_module, "_inventory_sha256", changing_inventory)
    with pytest.raises(ValueError, match="inventory changed during load"):
        load_feature_cache(cache, require_scientific_eligible=True)


def test_v2_inventory_cannot_redirect_away_from_loader_target(tmp_path: Path) -> None:
    cache, _ = _write_cache(tmp_path, eligible=True, version=2)
    session_dir = cache / _SESSION_ID
    redirected_path = session_dir / "unbound_maps.npy"
    np.save(
        redirected_path,
        np.load(session_dir / "maps.npy", allow_pickle=False),
        allow_pickle=False,
    )
    session_manifest_path = session_dir / "manifest.json"
    session_manifest = json.loads(
        session_manifest_path.read_text(encoding="utf-8")
    )
    maps_binding = session_manifest["file_inventory"]["maps"]
    maps_binding.update(
        path=redirected_path.name,
        sha256=_file_sha256(redirected_path),
        bytes=redirected_path.stat().st_size,
    )
    session_manifest["inventory_sha256"] = _value_sha256(
        session_manifest["file_inventory"]
    )
    session_manifest["content_sha256"] = _content_sha256(session_manifest)
    _write_json(session_manifest_path, session_manifest)

    root_path = cache / "manifest.json"
    root = json.loads(root_path.read_text(encoding="utf-8"))
    root_item = next(
        item for item in root["sessions"] if item["session_id"] == _SESSION_ID
    )
    root_item["inventory_sha256"] = session_manifest["inventory_sha256"]
    root_item["content_sha256"] = session_manifest["content_sha256"]
    root_item["session_manifest_sha256"] = _file_sha256(session_manifest_path)
    root_item["session_manifest_content_sha256"] = _content_sha256(
        session_manifest
    )
    root["acquisition_contract"]["cache_inventory_aggregate_sha256"] = (
        _value_sha256(
            {
                item["session_id"]: item["inventory_sha256"]
                for item in root["sessions"]
            }
        )
    )
    root["content_sha256"] = _content_sha256(root)
    _write_json(root_path, root)

    with pytest.raises(ValueError, match="does not bind the loader target"):
        load_feature_cache(cache, require_scientific_eligible=True)


def test_v2_bound_reconstruction_graph_tampering_fails_closed(
    tmp_path: Path,
) -> None:
    cache, reconstruction_path = _write_cache(
        tmp_path, eligible=True, version=2
    )
    receipt_path = (
        reconstruction_path.parent
        / "sessions"
        / _SESSION_ID
        / "sync_receipt.json"
    )
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["result"]["confidence"] = 0.1
    _write_json(receipt_path, receipt)

    with pytest.raises(ValueError, match="reconstruction graph validation failed"):
        load_feature_cache(cache, require_scientific_eligible=True)


def test_v2_diagnostic_subset_is_bound_but_not_promoted(tmp_path: Path) -> None:
    cache, _ = _write_cache(
        tmp_path,
        eligible=False,
        version=2,
        expected_usable_session_ids=(_SESSION_ID, "S02_RJS"),
    )
    loaded = load_feature_cache(cache, require_acquisition_contract=True)
    assert loaded.provenance is not None
    assert loaded.provenance.classification == "acquisition_diagnostic"
    assert loaded.provenance.scientific_eligible is False
    with pytest.raises(ValueError, match="root is not scientifically eligible"):
        load_feature_cache(cache, require_scientific_eligible=True)


def test_v2_explicit_all_session_filter_is_valid_only_as_diagnostic(
    tmp_path: Path,
) -> None:
    cache, _ = _write_cache(tmp_path, eligible=True, version=2)
    root_path = cache / "manifest.json"
    root = json.loads(root_path.read_text(encoding="utf-8"))
    for item in root["sessions"]:
        session_path = cache / item["session_id"] / "manifest.json"
        session = json.loads(session_path.read_text(encoding="utf-8"))
        metadata_path = cache / item["session_id"] / "metadata.csv"
        metadata = pd.read_csv(metadata_path)
        metadata["alignment_scientific_eligible"] = False
        metadata["eligible_for_stage_metrics"] = False
        metadata.to_csv(metadata_path, index=False)
        metadata_binding = session["file_inventory"]["metadata"]
        metadata_binding["sha256"] = _file_sha256(metadata_path)
        metadata_binding["bytes"] = metadata_path.stat().st_size
        session["inventory_sha256"] = _value_sha256(
            session["file_inventory"]
        )
        contract = session["acquisition_contract"]
        contract["scientific_eligible"] = False
        contract["reference_alignment_mode"] = (
            "diagnostic_unapproved_proposal_v1"
        )
        session["content_sha256"] = _content_sha256(session)
        _write_json(session_path, session)
        item["acquisition_contract"] = contract
        item["content_sha256"] = session["content_sha256"]
        item["session_manifest_content_sha256"] = session["content_sha256"]
        item["session_manifest_sha256"] = _file_sha256(session_path)
        item["inventory_sha256"] = session["inventory_sha256"]
    root["subjects_filter_applied"] = True
    contract = root["acquisition_contract"]
    contract["subjects_filter_applied"] = True
    contract["selection_scope"] = "diagnostic_subset"
    contract["full_cohort_complete"] = False
    contract["mode"] = "diagnostic"
    contract["scientific_eligible"] = False
    contract["cache_inventory_aggregate_sha256"] = _value_sha256(
        {
            item["session_id"]: item["inventory_sha256"]
            for item in root["sessions"]
        }
    )
    root["content_sha256"] = _content_sha256(root)
    _write_json(root_path, root)

    loaded = load_feature_cache(cache, require_acquisition_contract=True)

    assert loaded.provenance is not None
    assert loaded.provenance.classification == "acquisition_diagnostic"
    assert loaded.provenance.scientific_eligible is False


@pytest.mark.parametrize(
    "field",
    ("cohort_authority_sha256", "cohort_authority_content_sha256"),
)
def test_v2_cache_must_inherit_exact_cohort_authority_binding(
    tmp_path: Path,
    field: str,
) -> None:
    cache, _ = _write_cache(tmp_path, eligible=True, version=2)
    root_path = cache / "manifest.json"
    root = json.loads(root_path.read_text(encoding="utf-8"))
    root["acquisition_contract"][field] = "f" * 64
    root["content_sha256"] = _content_sha256(root)
    _write_json(root_path, root)

    with pytest.raises(
        ValueError,
        match="cache/reconstruction cohort authority binding mismatch",
    ):
        load_feature_cache(cache)


def test_v2_cache_root_session_eligibility_mismatch_fails(tmp_path: Path) -> None:
    cache, _ = _write_cache(tmp_path, eligible=True, version=2)
    root_path = cache / "manifest.json"
    root = json.loads(root_path.read_text(encoding="utf-8"))
    root["acquisition_contract"]["scientific_eligible"] = False
    root["content_sha256"] = _content_sha256(root)
    _write_json(root_path, root)
    with pytest.raises(ValueError, match="root/session scientific eligibility mismatch"):
        load_feature_cache(cache)


def test_scientific_session_filter_and_in_memory_subset_are_never_promoted(
    tmp_path: Path,
) -> None:
    cache, _ = _write_cache(tmp_path, eligible=True, version=2)

    with pytest.raises(ValueError, match="forbids any sessions filter"):
        load_feature_cache(
            cache,
            sessions=[_SESSION_ID],
            require_scientific_eligible=True,
        )

    selected = load_feature_cache(cache, sessions=[_SESSION_ID])
    assert selected.provenance is not None
    assert selected.provenance.classification == "acquisition_diagnostic"
    assert not selected.provenance.scientific_eligible

    complete = load_feature_cache(cache, require_scientific_eligible=True)
    subset = complete.subset(np.asarray([0], dtype=np.int64))
    assert subset.provenance is not None
    assert subset.provenance.classification == "acquisition_diagnostic"
    assert not subset.provenance.scientific_eligible
    assert subset.radar_timing_valid_mask is not None
    np.testing.assert_array_equal(
        subset.radar_timing_valid_mask,
        complete.radar_timing_valid_mask[[0]],
    )


def test_timing_mask_is_exposed_for_diagnostics_but_all_true_for_scientific(
    tmp_path: Path,
) -> None:
    diagnostic_cache, _ = _write_cache(
        tmp_path / "diagnostic", eligible=False, version=2
    )
    diagnostic_mask_path = (
        diagnostic_cache / _SESSION_ID / "radar_timing_valid_mask.npy"
    )
    diagnostic_mask = np.ones((1, 3, 320), dtype=np.bool_)
    diagnostic_mask[0, 1, 17] = False
    np.save(diagnostic_mask_path, diagnostic_mask, allow_pickle=False)
    _reseal_cache_files(diagnostic_cache)

    diagnostic = load_feature_cache(diagnostic_cache)
    assert diagnostic.radar_timing_valid_mask is not None
    assert not diagnostic.radar_timing_valid_mask[0, 1, 17]
    assert diagnostic.provenance is not None
    assert diagnostic.provenance.classification == "acquisition_diagnostic"

    scientific_cache, _ = _write_cache(
        tmp_path / "scientific", eligible=True, version=2
    )
    scientific_mask_path = (
        scientific_cache / _SESSION_ID / "radar_timing_valid_mask.npy"
    )
    scientific_mask = np.ones((1, 3, 320), dtype=np.bool_)
    scientific_mask[0, 0, 0] = False
    np.save(scientific_mask_path, scientific_mask, allow_pickle=False)
    _reseal_cache_files(scientific_cache)

    with pytest.raises(ValueError, match="contains invalid radar timing"):
        load_feature_cache(scientific_cache)


def test_affine_drift_preserves_radar_32s_and_scaled_reference_support(
    tmp_path: Path,
) -> None:
    cache, _ = _write_cache(
        tmp_path, eligible=True, version=2, mapping_scale=1.0001
    )

    loaded = load_feature_cache(cache, require_scientific_eligible=True)
    frame = loaded.metadata.loc[
        loaded.metadata["session_id"] == _SESSION_ID
    ].iloc[0]
    assert frame["radar_window_end_relative_s"] - frame[
        "radar_window_start_relative_s"
    ] == pytest.approx(32.0)
    assert frame["reference_window_end_biopac_s"] - frame[
        "reference_window_start_biopac_s"
    ] == pytest.approx(32.0 * 1.0001)
    assert int(frame["reference_start_sample"]) == 2_501
    assert int(frame["reference_end_sample"]) == 10_501

    metadata_path = cache / _SESSION_ID / "metadata.csv"
    metadata = pd.read_csv(metadata_path)
    metadata.loc[0, "reference_window_end_biopac_s"] += 0.01
    metadata.loc[0, "window_end_s"] += 0.01
    metadata.loc[0, "reference_end_sample"] = _half_open_index(
        metadata.loc[0, "reference_window_end_biopac_s"]
    )
    metadata.to_csv(metadata_path, index=False)
    _reseal_cache_files(cache)

    with pytest.raises(ValueError, match="exact 32s support"):
        load_feature_cache(cache)


@pytest.mark.parametrize(
    ("column", "replacement", "message"),
    (
        ("identity", "SUBSTITUTED", "canonical physical identity mismatch"),
        ("window_number", 1, "unique, ordered, and consecutive"),
        (
            "radar_window_start_relative_s",
            8.25,
            "exact 32s support|synchronization mapping mismatch",
        ),
        (
            "acquisition_phase_status",
            "review",
            "phase assignment does not match reconstructed protocol",
        ),
    ),
)
def test_rehashed_metadata_authority_tampering_fails_closed(
    tmp_path: Path,
    column: str,
    replacement: object,
    message: str,
) -> None:
    cache, _ = _write_cache(tmp_path, eligible=True, version=2)
    metadata_path = cache / _SESSION_ID / "metadata.csv"
    metadata = pd.read_csv(metadata_path)
    metadata.loc[0, column] = replacement
    metadata.to_csv(metadata_path, index=False)
    _reseal_cache_files(cache)

    with pytest.raises(ValueError, match=message):
        load_feature_cache(cache)


def test_rehashed_session_config_and_missing_content_hash_fail_closed(
    tmp_path: Path,
) -> None:
    config_cache, _ = _write_cache(
        tmp_path / "config", eligible=True, version=2
    )
    session_path = config_cache / _SESSION_ID / "manifest.json"
    session = json.loads(session_path.read_text(encoding="utf-8"))
    session["config_sha256"] = "f" * 64
    _write_json(session_path, session)
    _reseal_cache_files(config_cache)
    with pytest.raises(ValueError, match="root/session config_sha256 mismatch"):
        load_feature_cache(config_cache)

    content_cache, _ = _write_cache(
        tmp_path / "content", eligible=True, version=2
    )
    session_path = content_cache / _SESSION_ID / "manifest.json"
    session = json.loads(session_path.read_text(encoding="utf-8"))
    del session["content_sha256"]
    _write_json(session_path, session)
    root_path = content_cache / "manifest.json"
    root = json.loads(root_path.read_text(encoding="utf-8"))
    observed_content = _content_sha256(session)
    root_item = next(
        item for item in root["sessions"] if item["session_id"] == _SESSION_ID
    )
    root_item["content_sha256"] = observed_content
    root_item["session_manifest_content_sha256"] = observed_content
    root_item["session_manifest_sha256"] = _file_sha256(session_path)
    root["content_sha256"] = _content_sha256(root)
    _write_json(root_path, root)
    with pytest.raises(ValueError, match="session .* content_sha256"):
        load_feature_cache(content_cache)


def test_rehashed_cache_timing_summary_cannot_diverge_from_reconstruction(
    tmp_path: Path,
) -> None:
    cache, _ = _write_cache(tmp_path, eligible=True, version=2)
    session_path = cache / _SESSION_ID / "manifest.json"
    session = json.loads(session_path.read_text(encoding="utf-8"))
    session["radar_timing_summary"]["output_interval_count"] -= 1
    _write_json(session_path, session)
    _reseal_cache_files(cache)

    with pytest.raises(ValueError, match="cache/source radar timing summary mismatch"):
        load_feature_cache(cache)
