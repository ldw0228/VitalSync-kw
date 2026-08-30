from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

import snn_rr.acquisition_contract as acquisition_contract_module
from snn_rr.acquisition_contract import (
    ACQUISITION_SCHEMA,
    AcquisitionContractError,
    load_acquisition_cohort_authority,
    load_acquisition_reconstruction,
)
from snn_rr.preprocess import identity_for_session
from snn_rr.data import XETHRU_RECORD_DTYPE, build_dataset_manifest
from snn_rr.synchronization import (
    MarkerCandidate,
    MarkerMatch,
    SynchronizationConfig,
    SynchronizationResult,
    TimeMapping,
    build_sync_receipt,
    canonical_content_sha256,
    canonical_json_bytes,
)


_RECONSTRUCTION_CONTEXT = {
    "pipeline_sha256": "1" * 64,
    "sync_config_sha256": "2" * 64,
    "protocol_config_sha256": "3" * 64,
    "spreadsheet_sha256": "4" * 64,
    "build_range_tracks": False,
    "layout_maximum_frames": 5_000_000,
}

_COHORT_AUTHORITY_PATH = (
    Path(__file__).resolve().parents[1]
    / "configs"
    / "acquisition_cohort_v1.yaml"
)
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_PROTOCOL_CONFIG_PATH = _PROJECT_ROOT / "configs" / "acquisition_protocol_v1.yaml"
_SPREADSHEET_PATH = _PROJECT_ROOT / "HAI_EXPERIMENT" / "Dataset_issue.xlsx"
_DATASET_ROOT = _PROJECT_ROOT / "HAI_EXPERIMENT"
_DISCOVERED_DATASET = build_dataset_manifest(_DATASET_ROOT)
_REAL_RAW_GRAPH_VALIDATOR = acquisition_contract_module._validate_v2_raw_input_graph


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


def _write_json(path: Path, document: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            document,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n",
        encoding="utf-8",
    )


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _seal(document: dict[str, Any]) -> None:
    document["content_sha256"] = canonical_content_sha256(document)


def _bind_root_id_hashes(root_document: dict[str, Any]) -> None:
    for key in (
        "expected_session_ids",
        "expected_usable_session_ids",
        "selected_session_ids",
    ):
        root_document[f"{key}_sha256"] = hashlib.sha256(
            canonical_json_bytes(root_document[key])
        ).hexdigest()


def _accepted_receipt(
    session_id: str, *, config: SynchronizationConfig | None = None
) -> dict[str, Any]:
    radar_times = (0.0, 200.0, 400.0)
    rsp_times = (2.0, 202.0, 402.0)
    radar_markers = tuple(
        MarkerCandidate(index=index, time_s=time, score=10.0, source="motion")
        for index, time in enumerate(radar_times)
    )
    rsp_markers = tuple(
        MarkerCandidate(index=index, time_s=time, score=10.0, source="fixed_high")
        for index, time in enumerate(rsp_times)
    )
    result = SynchronizationResult(
        decision="accepted",
        reasons=(),
        mapping=TimeMapping(mode="constant", offset_s=2.0, scale=1.0),
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
        confidence=0.95,
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


def _resampling_evidence(output_rate_hz: float) -> dict[str, Any]:
    content_hashes = {
        "hash_schema_version": "snn_rr.canonical_ndarray_sha256.v1",
        "corrected_input_values_sha256": "5" * 64,
        "aligned_input_time_coordinates_sha256": "6" * 64,
        "frame_sequences_sha256": "7" * 64,
        "output_times_sha256": "8" * 64,
        "output_values_sha256": "9" * 64,
        "valid_mask_sha256": "a" * 64,
        "sample_counts_sha256": "b" * 64,
    }
    per_view = [
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
    ]
    summary = {
        "schema_version": "snn_rr.causal_uniform_radar_resample.v1",
        "aggregation": "half_open_interval_arithmetic_mean",
        "causal": True,
        "timestamp_semantics": "right_edge_exclusive",
        "invalid_value_policy": "exact_zero_with_structural_mask",
        "gap_policy": "mask",
        "output_rate_hz": output_rate_hz,
        "interval_s": 1.0 / output_rate_hz,
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
        "transform_evidence_sha256": hashlib.sha256(
            canonical_json_bytes(content_hashes)
        ).hexdigest(),
        "per_view": per_view,
    }
    return summary


def _warning_evidence(
    session_id: str, config: SynchronizationConfig
) -> dict[str, Any]:
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
        "metadata_warning_evidence_sha256": hashlib.sha256(
            canonical_json_bytes(evidence)
        ).hexdigest(),
    }


def _record_contract_evidence() -> dict[str, Any]:
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
        "xethru_record_contract_evidence_sha256": hashlib.sha256(
            canonical_json_bytes(evidence)
        ).hexdigest(),
    }


def _session_document(
    root: Path,
    session_id: str,
    *,
    reconstruction_context: dict[str, Any],
    sync_config: SynchronizationConfig,
) -> dict[str, Any]:
    receipt = _accepted_receipt(session_id, config=sync_config)
    receipt_path = root / "sessions" / session_id / "sync_receipt.json"
    _write_json(receipt_path, receipt)
    result = receipt["result"]
    raw_bindings = receipt["input_bindings"]
    resampling = _resampling_evidence(10.0)
    return {
        "schema_version": ACQUISITION_SCHEMA,
        "session_id": session_id,
        "physical_identity": identity_for_session(session_id),
        "usable": True,
        "reconstruction_context": dict(reconstruction_context),
        "raw_input_bindings": raw_bindings,
        "raw_input_bindings_sha256": hashlib.sha256(
            canonical_json_bytes(raw_bindings)
        ).hexdigest(),
        "scientific_eligible": True,
        "eligibility": {
            "measured_timing_eligible": True,
            "alignment_eligible": True,
            "stage_metric_eligible": True,
            "range_feature_eligible": False,
            "strict_cache_eligible": True,
        },
        "sensor_summary": {
            "radar": {
                **_warning_evidence(session_id, sync_config),
                **_record_contract_evidence(),
                "timestamp_sources": ["meta_v13", "meta_v13", "meta_v13"],
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
            "authorized": True,
            "receipt": f"sessions/{session_id}/sync_receipt.json",
            "receipt_sha256": _sha256_file(receipt_path),
            "receipt_content_sha256": receipt["content_sha256"],
            "manual_approval_present": False,
            "manual_approval": None,
            "manual_approval_file_sha256": None,
            "manual_approval_content_sha256": None,
            "match_count": len(result["matches"]),
        },
        "protocol": {
            "session_id": session_id,
            "annotation_schema_version": "acquisition_protocol_v1",
            "duration_s": 120.0,
            "status": "auto",
            "confidence": 0.95,
            "annotation_inference_feature_allowed": False,
            "stages": [
                {
                    "stage_id": "phase1",
                    "name": "synthetic_stage",
                    "status": "auto",
                    "confidence": 0.95,
                    "start": {"time_s": 0.0},
                    "end": {"time_s": 120.0},
                    "duration_s": 120.0,
                }
            ],
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
        "range_tracking": {
            "status": "not_built",
            "selected_session_layout": None,
            "physical_range_calibrated": False,
            "layout_selection_causal": False,
            "inference_feature_eligible": False,
        },
        "review_reasons": [],
    }


def _unusable_session_document(
    session_id: str,
    *,
    reconstruction_context: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": ACQUISITION_SCHEMA,
        "session_id": session_id,
        "physical_identity": identity_for_session(session_id),
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
        "session_record": None,
        "review_reasons": ["raw_session_unusable"],
    }


def _reseal_bundle(
    root_path: Path,
    root_document: dict[str, Any],
    sessions: dict[str, dict[str, Any]],
) -> None:
    for entry in root_document["sessions"]:
        session_id = entry["session_id"]
        session = sessions[session_id]
        _seal(session)
        session_path = root_path.parent / entry["manifest"]
        _write_json(session_path, session)
        entry["content_sha256"] = session["content_sha256"]
        entry["manifest_sha256"] = _sha256_file(session_path)
    _bind_root_id_hashes(root_document)
    _seal(root_document)
    _write_json(root_path, root_document)


def _reseal_root(root_path: Path, root_document: dict[str, Any]) -> None:
    _bind_root_id_hashes(root_document)
    _seal(root_document)
    _write_json(root_path, root_document)


def _build_reconstruction(
    tmp_path: Path,
    *,
    selected_ids: tuple[str, ...] | None = None,
) -> tuple[Path, dict[str, Any], dict[str, dict[str, Any]]]:
    cohort = load_acquisition_cohort_authority(_COHORT_AUTHORITY_PATH)
    expected_ids = cohort.expected_session_ids
    expected_usable_ids = cohort.expected_usable_session_ids
    selected = expected_ids if selected_ids is None else selected_ids
    subjects_filter_applied = selected_ids is not None
    root_dir = tmp_path / "reconstruction"
    sync_config = SynchronizationConfig()
    sync_config_path = root_dir / "sync_config.yaml"
    _write_json(sync_config_path, sync_config.to_dict())
    reconstruction_context = {
        **_RECONSTRUCTION_CONTEXT,
        "sync_config_sha256": _sha256_file(sync_config_path),
        "protocol_config_sha256": _sha256_file(_PROTOCOL_CONFIG_PATH),
        "spreadsheet_sha256": _sha256_file(_SPREADSHEET_PATH),
        "cohort_authority_sha256": cohort.file_sha256,
        "cohort_authority_content_sha256": cohort.content_sha256,
        "subjects_filter_applied": subjects_filter_applied,
    }
    sessions = {
        session_id: (
            _session_document(
                root_dir,
                session_id,
                reconstruction_context=reconstruction_context,
                sync_config=sync_config,
            )
            if session_id in set(expected_usable_ids)
            else _unusable_session_document(
                session_id,
                reconstruction_context=reconstruction_context,
            )
        )
        for session_id in selected
    }
    selected_usable = tuple(
        session_id for session_id in selected if session_id in set(expected_usable_ids)
    )
    full_cohort = not subjects_filter_applied and selected == expected_ids
    cohort_claims = cohort.to_claims()
    root_document: dict[str, Any] = {
        "schema_version": ACQUISITION_SCHEMA,
        **reconstruction_context,
        "sync_config": str(sync_config_path),
        "protocol_config": str(_PROTOCOL_CONFIG_PATH),
        "spreadsheet": str(_SPREADSHEET_PATH),
        "dataset_root": str(_DATASET_ROOT),
        "cohort_authority": str(_COHORT_AUTHORITY_PATH),
        "cohort_authority_schema": "snn_rr.acquisition_cohort_authority.v1",
        "dataset_manifest_sha256": hashlib.sha256(
            canonical_json_bytes(_DISCOVERED_DATASET.to_dict())
        ).hexdigest(),
        "expected_session_ids": list(expected_ids),
        "expected_usable_session_ids": list(expected_usable_ids),
        "excluded_sessions": cohort_claims["excluded_sessions"],
        "excluded_sessions_sha256": hashlib.sha256(
            canonical_json_bytes(cohort_claims["excluded_sessions"])
        ).hexdigest(),
        "session_identities": cohort_claims["session_identities"],
        "session_identities_sha256": hashlib.sha256(
            canonical_json_bytes(cohort_claims["session_identities"])
        ).hexdigest(),
        "expected_physical_identities": cohort_claims[
            "expected_physical_identities"
        ],
        "expected_physical_identities_sha256": hashlib.sha256(
            canonical_json_bytes(cohort_claims["expected_physical_identities"])
        ).hexdigest(),
        "selected_session_ids": list(selected),
        "dataset_session_count": len(expected_ids),
        "dataset_usable_session_count": len(expected_usable_ids),
        "dataset_physical_identity_count": len(
            cohort.expected_physical_identities
        ),
        "selected_session_count": len(selected),
        "selection_scope": "full_cohort" if full_cohort else "diagnostic_subset",
        "subjects_filter_applied": subjects_filter_applied,
        "execution_complete": True,
        "full_cohort_complete": full_cohort,
        "complete": full_cohort,
        "session_count": len(selected),
        "usable_session_count": len(selected_usable),
        "sync_authorized_session_count": len(selected_usable),
        "scientific_eligible_session_count": len(selected_usable),
        "scientific_eligible": full_cohort,
        "sessions": [
            {
                "session_id": session_id,
                "physical_identity": identity_for_session(session_id),
                "usable": sessions[session_id]["usable"],
                "scientific_eligible": sessions[session_id][
                    "scientific_eligible"
                ],
                "manifest": f"sessions/{session_id}/session_manifest.json",
                "content_sha256": None,
                "manifest_sha256": None,
            }
            for session_id in selected
        ],
    }
    root_path = root_dir / "manifest.json"
    _reseal_bundle(root_path, root_document, sessions)
    return root_path, root_document, sessions


def test_v2_content_bound_full_cohort_fixture_is_scientifically_eligible(
    tmp_path: Path,
) -> None:
    path, _, _ = _build_reconstruction(tmp_path)

    loaded = load_acquisition_reconstruction(path)

    assert loaded.selection_scope == "full_cohort"
    assert loaded.execution_complete
    assert loaded.full_cohort_complete
    assert loaded.scientific_eligible
    assert all(contract.strict_cache_eligible for contract in loaded.sessions.values())


def test_v2_explicit_all_session_filter_remains_diagnostic(tmp_path: Path) -> None:
    cohort = load_acquisition_cohort_authority(_COHORT_AUTHORITY_PATH)
    path, root, _ = _build_reconstruction(
        tmp_path, selected_ids=cohort.expected_session_ids
    )

    loaded = load_acquisition_reconstruction(path)

    assert root["subjects_filter_applied"] is True
    assert loaded.selection_scope == "diagnostic_subset"
    assert loaded.execution_complete
    assert not loaded.full_cohort_complete
    assert not loaded.scientific_eligible


@pytest.mark.parametrize("mutation", ("missing", "extra", "order", "duplicate"))
def test_v2_expected_session_claim_must_equal_frozen_cohort_authority(
    tmp_path: Path,
    mutation: str,
) -> None:
    path, root, _ = _build_reconstruction(tmp_path)
    ids = root["expected_session_ids"]
    if mutation == "missing":
        ids.pop()
    elif mutation == "extra":
        ids.append("S31_FORGED")
    elif mutation == "order":
        ids[0], ids[1] = ids[1], ids[0]
    else:
        ids[-1] = ids[0]
    root["dataset_session_count"] = len(ids)
    _reseal_root(path, root)

    message = {
        "missing": "outside the dataset authority",
        "extra": "expected session IDs differ from cohort authority",
        "order": "not in dataset-authority order",
        "duplicate": "duplicate IDs",
    }[mutation]
    with pytest.raises(AcquisitionContractError, match=message):
        load_acquisition_reconstruction(path)


def test_v2_usable_session_claim_must_equal_frozen_cohort_authority(
    tmp_path: Path,
) -> None:
    path, root, _ = _build_reconstruction(tmp_path)
    usable_ids = root["expected_usable_session_ids"]
    usable_ids[22] = "S24_KHJ"
    _reseal_root(path, root)

    with pytest.raises(
        AcquisitionContractError,
        match="expected usable IDs differ from cohort authority",
    ):
        load_acquisition_reconstruction(path)


def test_v2_s17_physical_identity_forgery_fails_against_authority(
    tmp_path: Path,
) -> None:
    path, root, _ = _build_reconstruction(tmp_path)
    assignment = next(
        item
        for item in root["session_identities"]
        if item["session_id"] == "S17_RJS"
    )
    assignment["physical_identity"] = "RJS"
    root["session_identities_sha256"] = hashlib.sha256(
        canonical_json_bytes(root["session_identities"])
    ).hexdigest()
    _reseal_root(path, root)

    with pytest.raises(
        AcquisitionContractError,
        match="session_identities differs from cohort authority",
    ):
        load_acquisition_reconstruction(path)


@pytest.mark.parametrize(
    ("artifact", "message"),
    (
        ("protocol", "bound protocol configuration is invalid"),
        ("spreadsheet", "cannot re-derive the bound dataset authority"),
    ),
)
def test_v2_bound_external_inputs_are_rehashed_and_reparsed(
    tmp_path: Path,
    artifact: str,
    message: str,
) -> None:
    path, root, sessions = _build_reconstruction(tmp_path)
    forged = tmp_path / ("forged.yaml" if artifact == "protocol" else "forged.xlsx")
    forged.write_bytes(b"not a valid bound acquisition artifact")
    digest = _sha256_file(forged)
    if artifact == "protocol":
        root["protocol_config"] = str(forged)
        root["protocol_config_sha256"] = digest
        context_key = "protocol_config_sha256"
    else:
        root["spreadsheet"] = str(forged)
        root["spreadsheet_sha256"] = digest
        context_key = "spreadsheet_sha256"
    for session in sessions.values():
        session["reconstruction_context"][context_key] = digest
    _reseal_bundle(path, root, sessions)

    with pytest.raises(AcquisitionContractError, match=message):
        load_acquisition_reconstruction(path)


def test_v2_dataset_manifest_digest_is_recomputed_not_syntax_checked(
    tmp_path: Path,
) -> None:
    path, root, _ = _build_reconstruction(tmp_path)
    root["dataset_manifest_sha256"] = "f" * 64
    _reseal_root(path, root)

    with pytest.raises(
        AcquisitionContractError, match="dataset manifest digest mismatch"
    ):
        load_acquisition_reconstruction(path)


@pytest.mark.parametrize("field", ("zero_header_nonzero", "bin_count_invalid"))
def test_v2_xethru_header_and_bin_count_contract_fails_closed(
    tmp_path: Path,
    field: str,
) -> None:
    path, root, sessions = _build_reconstruction(tmp_path)
    radar = sessions["S01_CMS"]["sensor_summary"]["radar"]
    evidence = radar["xethru_record_contract"]
    evidence["views"][0]["chunks"][0][field] = 1
    evidence["views"][0]["eligible"] = False
    evidence["eligible"] = False
    radar["xethru_record_contract_evidence_sha256"] = hashlib.sha256(
        canonical_json_bytes(evidence)
    ).hexdigest()
    _reseal_bundle(path, root, sessions)

    with pytest.raises(
        AcquisitionContractError,
        match="measured_timing_eligible does not match evidence",
    ):
        load_acquisition_reconstruction(path)


@pytest.mark.parametrize("mutation", ("missing", "extra", "path"))
def test_v2_raw_binding_graph_requires_exact_selected_session_coverage(
    tmp_path: Path,
    mutation: str,
) -> None:
    dataset_root = tmp_path / "raw"
    biopac_path = dataset_root / "S01_CMS" / "BIOPAC" / "recording.mat"
    biopac_path.parent.mkdir(parents=True)
    biopac_path.write_bytes(b"synthetic biopac")
    radars: dict[int, SimpleNamespace] = {}
    for radar_id in (1, 2, 3):
        recording_dir = dataset_root / "S01_CMS" / str(radar_id) / "recording"
        recording_dir.mkdir(parents=True)
        meta_path = recording_dir / "xethru_recording_meta.dat"
        meta_path.write_bytes(f"meta-{radar_id}".encode())
        data_path = recording_dir / f"xethru_datafloat_{radar_id}.dat"
        record = np.zeros(1, dtype=XETHRU_RECORD_DTYPE)
        record["bin_count"] = 182
        record.tofile(data_path)
        radars[radar_id] = SimpleNamespace(
            meta_path=meta_path,
            data_paths=(data_path,),
        )
    subject = SimpleNamespace(
        subject_id="S01_CMS",
        biopac_path=biopac_path,
        selected_session=SimpleNamespace(
            session_id="synthetic-selected-session",
            radars=radars,
        ),
    )
    bindings, graph = acquisition_contract_module.build_v2_raw_input_binding_state(
        subject, dataset_root
    )
    record_contract = (
        acquisition_contract_module.build_v2_xethru_record_contract(subject)
    )
    session = {
        "raw_input_bindings": json.loads(json.dumps(bindings)),
        "raw_input_graph": graph,
        "raw_input_graph_sha256": hashlib.sha256(
            canonical_json_bytes(graph)
        ).hexdigest(),
        "sensor_summary": {
            "radar": {
                "xethru_record_contract": record_contract,
                "xethru_record_contract_evidence_sha256": hashlib.sha256(
                    canonical_json_bytes(record_contract)
                ).hexdigest(),
            }
        },
    }
    receipt = {"input_bindings": json.loads(json.dumps(bindings))}
    _REAL_RAW_GRAPH_VALIDATOR(
        session,
        receipt,
        subject=subject,
        dataset_root=dataset_root,
        session_id="S01_CMS",
    )

    if mutation == "missing":
        del session["raw_input_bindings"]["radar3_data_00"]
    elif mutation == "extra":
        session["raw_input_bindings"]["radar4_data_00"] = dict(
            bindings["radar3_data_00"]
        )
    else:
        session["raw_input_bindings"]["biopac"]["path"] = "substituted.mat"

    with pytest.raises(
        AcquisitionContractError,
        match="do not exactly cover the selected-session graph",
    ):
        _REAL_RAW_GRAPH_VALIDATOR(
            session,
            receipt,
            subject=subject,
            dataset_root=dataset_root,
            session_id="S01_CMS",
        )


def test_v2_subset_completion_is_execution_only_and_never_scientific(
    tmp_path: Path,
) -> None:
    path, root, _ = _build_reconstruction(
        tmp_path, selected_ids=("S01_CMS",)
    )

    loaded = load_acquisition_reconstruction(path)

    assert root["execution_complete"] is True
    assert root["full_cohort_complete"] is False
    assert root["complete"] is False
    assert root["scientific_eligible"] is False
    assert loaded.selection_scope == "diagnostic_subset"
    assert loaded.execution_complete
    assert not loaded.full_cohort_complete
    assert not loaded.scientific_eligible
    assert loaded.sessions["S01_CMS"].strict_cache_eligible


@pytest.mark.parametrize(
    ("location", "field"),
    (
        ("root", "execution_complete"),
        ("root", "scientific_eligible"),
        ("session", "usable"),
        ("eligibility", "alignment_eligible"),
    ),
)
def test_v2_explicit_boolean_fields_reject_truthy_strings(
    tmp_path: Path,
    location: str,
    field: str,
) -> None:
    path, root, sessions = _build_reconstruction(tmp_path)
    if location == "root":
        root[field] = "true"
        _reseal_root(path, root)
    elif location == "session":
        sessions["S01_CMS"][field] = "true"
        _reseal_bundle(path, root, sessions)
    else:
        sessions["S01_CMS"]["eligibility"][field] = "true"
        _reseal_bundle(path, root, sessions)

    with pytest.raises(AcquisitionContractError, match="JSON boolean"):
        load_acquisition_reconstruction(path)


@pytest.mark.parametrize(
    ("field", "message"),
    (
        ("complete", "complete must mean full-cohort completion"),
        ("scientific_eligible", "does not match its children"),
    ),
)
def test_v2_subset_cannot_claim_full_completion_or_scientific_eligibility(
    tmp_path: Path,
    field: str,
    message: str,
) -> None:
    path, root, _ = _build_reconstruction(
        tmp_path, selected_ids=("S01_CMS",)
    )
    root[field] = True
    _reseal_root(path, root)

    with pytest.raises(AcquisitionContractError, match=message):
        load_acquisition_reconstruction(path)


def test_v2_full_cohort_count_mismatch_is_rejected_after_rehash(
    tmp_path: Path,
) -> None:
    path, root, _ = _build_reconstruction(tmp_path)
    root["dataset_session_count"] += 1
    _reseal_root(path, root)

    with pytest.raises(AcquisitionContractError, match="dataset session count mismatch"):
        load_acquisition_reconstruction(path)


def test_v2_full_cohort_id_mismatch_is_rejected_after_rehash(
    tmp_path: Path,
) -> None:
    path, root, _ = _build_reconstruction(tmp_path)
    root["selected_session_ids"][-1] = "S99_SUBSTITUTED"
    _reseal_root(path, root)

    with pytest.raises(AcquisitionContractError, match="selected IDs"):
        load_acquisition_reconstruction(path)


def test_v2_root_child_eligibility_mismatch_is_rejected_after_rehash(
    tmp_path: Path,
) -> None:
    path, root, _ = _build_reconstruction(tmp_path)
    root["sessions"][0]["scientific_eligible"] = False
    _reseal_root(path, root)

    with pytest.raises(AcquisitionContractError, match="root/session eligibility mismatch"):
        load_acquisition_reconstruction(path)


@pytest.mark.parametrize(
    ("target", "message"),
    (
        ("protocol", "protocol status is invalid"),
        ("stage", "protocol stages are invalid"),
    ),
)
def test_v2_unknown_protocol_status_is_rejected_after_rehash(
    tmp_path: Path,
    target: str,
    message: str,
) -> None:
    path, root, sessions = _build_reconstruction(tmp_path)
    protocol = sessions["S01_CMS"]["protocol"]
    if target == "protocol":
        protocol["status"] = "unknown"
    else:
        protocol["stages"][0]["status"] = "unknown"
    _reseal_bundle(path, root, sessions)

    with pytest.raises(AcquisitionContractError, match=message):
        load_acquisition_reconstruction(path)


@pytest.mark.parametrize(
    ("timing_defect", "message"),
    (
        ("invalid_interval", "measured_timing_eligible does not match evidence"),
        ("large_correction", "is not derived from per_view"),
    ),
)
def test_v2_timing_eligibility_is_recomputed_from_bound_evidence(
    tmp_path: Path,
    timing_defect: str,
    message: str,
) -> None:
    path, root, sessions = _build_reconstruction(tmp_path)
    radar = sessions["S01_CMS"]["sensor_summary"]["radar"]
    if timing_defect == "invalid_interval":
        summary = radar["feature_resampling"]
        summary["all_views_valid_interval_count"] = 1_199
        summary["any_view_invalid_interval_count"] = 1
        summary["per_view"][0]["valid_output_count"] = 1_199
        summary["per_view"][0]["invalid_output_count"] = 1
        summary["per_view"][0]["empty_interval_count"] = 1
    else:
        radar["maximum_timestamp_correction_s"] = 0.051
    _reseal_bundle(path, root, sessions)

    with pytest.raises(AcquisitionContractError, match=message):
        load_acquisition_reconstruction(path)


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        ("missing_dataset_authority", "dataset_manifest_sha256"),
        ("manual_approval_presence", "manual_approval_present/artifact mismatch"),
        ("raw_binding", "raw_input_bindings_sha256 mismatch"),
        ("session_pipeline", "reconstruction context pipeline_sha256 mismatch"),
        ("frame_accounting", "frame_accounting partition/proof mismatch"),
        ("half_open", "half-open exactness mismatch"),
        ("radar_warning", "radar 1 warning exact-match claim mismatch"),
        ("biopac_warning", "eligibility components"),
        ("range_promotion", "eligibility components"),
    ),
)
def test_v2_rehashed_provenance_forgery_fails_closed(
    tmp_path: Path,
    mutation: str,
    message: str,
) -> None:
    path, root, sessions = _build_reconstruction(tmp_path)
    session = sessions["S01_CMS"]
    if mutation == "missing_dataset_authority":
        del root["dataset_manifest_sha256"]
        _reseal_root(path, root)
    elif mutation == "manual_approval_presence":
        session["synchronization"]["manual_approval_present"] = True
        _reseal_bundle(path, root, sessions)
    elif mutation == "raw_binding":
        session["raw_input_bindings"]["signals"]["path"] = "raw/substituted.bin"
        _reseal_bundle(path, root, sessions)
    elif mutation == "session_pipeline":
        session["reconstruction_context"]["pipeline_sha256"] = "f" * 64
        _reseal_bundle(path, root, sessions)
    elif mutation == "frame_accounting":
        accounting = session["sensor_summary"]["radar"][
            "feature_resampling"
        ]["per_view"][0]["frame_accounting"]
        accounting["category_sum"] -= 1
        _reseal_bundle(path, root, sessions)
    elif mutation == "half_open":
        arithmetic = session["sensor_summary"]["radar"][
            "feature_resampling"
        ]["time_arithmetic"]
        arithmetic["half_open_boundary_exact"] = False
        _reseal_bundle(path, root, sessions)
    elif mutation == "radar_warning":
        warning_view = session["sensor_summary"]["radar"][
            "metadata_warning_views"
        ][0]
        warning_view["warnings"] = ["unexpected warning"]
        _reseal_bundle(path, root, sessions)
    elif mutation == "biopac_warning":
        session["sensor_summary"]["biopac"]["warnings"] = [
            "synthetic acquisition warning"
        ]
        _reseal_bundle(path, root, sessions)
    else:
        session["eligibility"]["range_feature_eligible"] = True
        _reseal_bundle(path, root, sessions)

    with pytest.raises(AcquisitionContractError, match=message):
        load_acquisition_reconstruction(path)


def test_v2_retained_plateau_is_accepted_only_as_structurally_masked_diagnostic(
    tmp_path: Path,
) -> None:
    path, root, sessions = _build_reconstruction(tmp_path)
    session = sessions["S01_CMS"]
    radar = session["sensor_summary"]["radar"]
    summary = radar["feature_resampling"]
    summary["all_views_valid_interval_count"] = 1_199
    summary["any_view_invalid_interval_count"] = 1
    view = summary["per_view"][0]
    view["valid_output_count"] = 1_199
    view["invalid_output_count"] = 1
    view["timestamp_plateau_interval_count"] = 1
    view["timestamp_repair"] = {
        "timestamp_plateau_count": 1,
        "measured_tie_edge_count": 1,
        "reconstructed_frame_count": 0,
        "maximum_timestamp_correction_s": 0.0,
        "reconstruction_method": "none_structural_mask_required",
        "nominal_positive_period_s": 0.025,
        "plateaus": [
            {
                "first_affected_frame": 100,
                "last_affected_frame": 101,
                "measured_time_s": 2.5,
                "duplicate_edge_count": 1,
                "sequence_contiguous": True,
                "at_leading_boundary": False,
                "at_trailing_boundary": False,
            }
        ],
    }
    radar["timestamp_plateau_interval_count"] = 1
    radar["measured_timing_eligible"] = False
    session["eligibility"]["measured_timing_eligible"] = False
    session["eligibility"]["alignment_eligible"] = False
    session["eligibility"]["strict_cache_eligible"] = False
    session["scientific_eligible"] = False
    root["sessions"][0]["scientific_eligible"] = False
    root["scientific_eligible_session_count"] -= 1
    root["scientific_eligible"] = False
    _reseal_bundle(path, root, sessions)

    loaded = load_acquisition_reconstruction(path)

    assert not loaded.scientific_eligible
    assert not loaded.sessions["S01_CMS"].measured_timing_eligible
