from __future__ import annotations

import copy
import hashlib

import numpy as np
import pytest

from snn_rr.synchronization import (
    MarkerCandidate,
    SynchronizationConfig,
    SynchronizationError,
    TimeMapping,
    apply_affine_sample_index,
    build_affine_sample_index,
    build_manual_approval,
    build_sync_receipt,
    canonical_json_bytes,
    detect_radar_marker_candidates,
    detect_rsp_marker_candidates,
    epoch_prior_offset_from_starts,
    estimate_marker_time_mapping,
    load_synchronization_config,
    robust_radar_motion_envelope,
    synchronization_is_authorized,
    synchronize_from_signals,
    validate_manual_approval,
    validate_sync_receipt,
)


def _markers(times, *, source="motion", score=12.0):
    return tuple(
        MarkerCandidate(index=index, time_s=float(time), score=score, source=source)
        for index, time in enumerate(times)
    )


def _decision_config(**overrides):
    values = SynchronizationConfig().to_dict()
    values.update(
        {
            "min_marker_pairs": 3,
            "good_marker_pairs": 4,
            "min_marker_span_s": 100.0,
            "min_affine_span_s": 250.0,
            "min_affine_pairs": 3,
            "accept_min_confidence": 0.75,
        }
    )
    values.update(overrides)
    return SynchronizationConfig(**values)


def _accepted_result():
    radar = _markers([20.0, 180.0, 360.0, 540.0])
    rsp = _markers([25.25, 185.25, 365.25, 545.25], source="adaptive_high+fixed_high")
    return estimate_marker_time_mapping(
        radar, rsp, epoch_prior_offset_s=5.0, config=_decision_config()
    )


def test_exact_182_payload_and_target_free_motion_marker_detection():
    rng = np.random.default_rng(20260830)
    config = _decision_config(
        radar_sample_rate_hz=10.0,
        motion_smoothing_s=0.3,
        radar_marker_merge_s=3.0,
        radar_marker_z=5.0,
    )
    frames = rng.normal(0.0, 0.02, size=(3, 900, 182)).astype(np.float32)
    # Large, short body motions affect broad but different range regions.
    for frame_index in (150, 450, 750):
        frames[0, frame_index : frame_index + 5, 20:100] += 4.0
        frames[1, frame_index : frame_index + 5, 60:160] -= 3.0
    envelope = robust_radar_motion_envelope(frames, config=config)
    markers = detect_radar_marker_candidates(envelope, config=config)

    assert envelope.valid_view_count == 3
    assert envelope.repaired_nonfinite_values == 0
    assert len(markers) == 3
    assert np.allclose([marker.time_s for marker in markers], [15.0, 45.0, 75.0], atol=0.5)

    with pytest.raises(SynchronizationError, match="182"):
        robust_radar_motion_envelope(frames[..., :181], config=config)


def test_rsp_candidates_union_adaptive_and_fixed_regions():
    config = _decision_config(
        rsp_sample_rate_hz=20.0,
        rsp_marker_merge_s=2.0,
        rsp_adaptive_z=5.0,
    )
    times = np.arange(0.0, 60.0, 1.0 / config.rsp_sample_rate_hz)
    rsp = 4.0 + 0.35 * np.sin(2 * np.pi * 0.25 * times)
    rsp[(times >= 10.0) & (times < 10.5)] = 10.0
    rsp[(times >= 40.0) & (times < 40.5)] = 8.7
    candidates = detect_rsp_marker_candidates(rsp, rsp_times_s=times, config=config)

    assert len(candidates) == 2
    assert np.allclose([candidate.time_s for candidate in candidates], [10.0, 40.0], atol=0.1)
    assert all("adaptive_high" in candidate.source for candidate in candidates)
    assert all("fixed_high" in candidate.source for candidate in candidates)


def test_known_constant_offset_is_accepted():
    assert epoch_prior_offset_from_starts(
        radar_start_epoch_s=1_000_005.25,
        rsp_start_epoch_s=1_000_000.0,
    ) == pytest.approx(5.25)
    result = _accepted_result()

    assert result.decision == "accepted"
    assert result.mapping is not None
    assert result.mapping.mode == "constant"
    assert result.mapping.offset_s == pytest.approx(5.25, abs=1e-9)
    assert result.mapping.drift_ppm == 0.0
    assert len(result.matches) == 4
    assert result.residual_rmse_s == pytest.approx(0.0, abs=1e-10)


def test_plus_200ppm_drift_selects_affine_fit():
    radar_times = np.asarray([20.0, 350.0, 710.0, 1100.0, 1500.0])
    offset = 3.4
    scale = 1.0 + 200e-6
    rsp_times = offset + scale * radar_times
    result = estimate_marker_time_mapping(
        _markers(radar_times),
        _markers(rsp_times, source="adaptive_high+fixed_high"),
        epoch_prior_offset_s=3.3,
        config=_decision_config(),
    )

    assert result.decision == "accepted"
    assert result.mapping is not None
    assert result.mapping.mode == "affine"
    assert result.mapping.offset_s == pytest.approx(offset, abs=1e-9)
    assert result.mapping.drift_ppm == pytest.approx(200.0, abs=1e-6)
    assert result.residual_max_abs_s == pytest.approx(0.0, abs=1e-9)


def test_false_and_missing_markers_retain_monotonic_mapping():
    radar_true = np.asarray([40.0, 240.0, 520.0, 820.0, 1180.0])
    offset = -2.75
    radar = _markers([40.0, 240.0, 400.0, 520.0, 820.0, 1180.0])
    # RSP misses the 240-s marker and has two false candidates.
    rsp = _markers(
        [-1.0, *(offset + radar_true[[0, 2, 3, 4]]), 1300.0],
        source="adaptive_high+fixed_high",
    )
    result = estimate_marker_time_mapping(
        radar,
        rsp,
        epoch_prior_offset_s=-2.5,
        config=_decision_config(),
    )

    assert result.decision == "accepted"
    assert result.mapping is not None
    assert result.mapping.offset_s == pytest.approx(offset, abs=1e-9)
    assert len(result.matches) == 4
    assert [match.radar_time_s for match in result.matches] == [40.0, 520.0, 820.0, 1180.0]


def test_insufficient_marker_span_fails_closed():
    radar = _markers([100.0, 120.0, 140.0])
    rsp = _markers([104.0, 124.0, 144.0], source="fixed_high")
    result = estimate_marker_time_mapping(
        radar,
        rsp,
        epoch_prior_offset_s=4.0,
        config=_decision_config(min_marker_span_s=100.0),
    )

    assert result.decision == "manual_review_required"
    assert not result.automatically_authorized
    assert "insufficient_marker_span" in result.reasons


def test_equal_quality_distinct_offsets_are_ambiguous_and_fail_closed():
    config = _decision_config(
        prior_tolerance_s=20.0,
        match_residual_gate_s=0.3,
        ambiguity_score_margin=1.0,
    )
    radar = _markers([100.0, 300.0, 500.0, 700.0])
    # Two complete marker trains are symmetrically placed around the epoch prior.
    rsp = _markers(
        [95.0, 105.0, 295.0, 305.0, 495.0, 505.0, 695.0, 705.0],
        source="adaptive_high+fixed_high",
    )
    result = estimate_marker_time_mapping(
        radar, rsp, epoch_prior_offset_s=0.0, config=config
    )

    assert result.ambiguous
    assert result.decision == "manual_review_required"
    assert "ambiguous_marker_mapping" in result.reasons
    assert result.confidence == 0.0


def test_affine_mapping_index_and_resampling():
    mapping = TimeMapping(mode="affine", offset_s=0.125, scale=1.0 + 200e-6)
    radar_times = np.asarray([-1.0, 0.0, 0.75, 2.0, 20.0])
    rsp_times = np.arange(0.0, 10.0, 0.01)
    rsp_values = 2.0 * rsp_times + 1.0
    index = build_affine_sample_index(mapping, radar_times, rsp_times)
    aligned = apply_affine_sample_index(rsp_values, index)
    expected_times = mapping.radar_to_rsp(radar_times)

    assert index.valid.tolist() == [False, True, True, True, False]
    assert np.isnan(aligned[[0, 4]]).all()
    assert np.allclose(aligned[1:4], 2.0 * expected_times[1:4] + 1.0, atol=1e-12)
    assert np.all(index.lower[index.valid] <= index.upper[index.valid])


def test_receipt_is_canonical_tamper_evident_and_manual_approval_is_bound():
    result = _accepted_result()
    config = _decision_config()
    receipt = build_sync_receipt(
        result,
        session_id="SYNTHETIC_01",
        config=config,
        input_bindings={
            "radar": {"sha256": "a" * 64, "bytes": 1234},
            "rsp": {"sha256": "b" * 64, "bytes": 5678},
        },
        created_at_utc="2026-08-30T12:00:00Z",
    )
    validated = validate_sync_receipt(receipt, expected_session_id="SYNTHETIC_01")
    canonical_file_hash = hashlib.sha256(canonical_json_bytes(receipt) + b"\n").hexdigest()
    assert validated == receipt
    assert len(canonical_file_hash) == 64

    approval = build_manual_approval(
        receipt,
        reviewer_id="reviewer-01",
        decision="approve",
        reviewed_at_utc="2026-08-30T12:05:00Z",
        rationale="Three marker pairs were visually confirmed in both modalities.",
    )
    assert validate_manual_approval(approval, receipt) == approval
    assert synchronization_is_authorized(receipt, manual_approval=approval)

    tampered = copy.deepcopy(receipt)
    tampered["result"]["mapping"]["offset_s"] += 0.1
    with pytest.raises(SynchronizationError, match="content SHA-256"):
        validate_sync_receipt(tampered)

    other = copy.deepcopy(receipt)
    other["session_id"] = "SYNTHETIC_02"
    other["content_sha256"] = hashlib.sha256(
        canonical_json_bytes({key: value for key, value in other.items() if key != "content_sha256"})
    ).hexdigest()
    with pytest.raises(SynchronizationError, match="(session|binding) mismatch"):
        validate_manual_approval(approval, other)

    approval_tampered = copy.deepcopy(approval)
    approval_tampered["mapping_sha256"] = "f" * 64
    with pytest.raises(SynchronizationError, match="content SHA-256"):
        validate_manual_approval(approval_tampered, receipt)


def test_manual_approval_is_required_for_review_only_mapping():
    config = _decision_config(min_marker_span_s=1_000.0)
    radar = _markers([0.0, 100.0, 200.0])
    rsp = _markers([2.0, 102.0, 202.0], source="fixed_high")
    result = estimate_marker_time_mapping(
        radar, rsp, epoch_prior_offset_s=2.0, config=config
    )
    receipt = build_sync_receipt(
        result,
        session_id="SYNTHETIC_REVIEW",
        config=config,
        input_bindings={"signals": {"sha256": "c" * 64}},
        created_at_utc="2026-08-30T00:00:00Z",
    )

    assert not synchronization_is_authorized(receipt)
    approval = build_manual_approval(
        receipt,
        reviewer_id="reviewer-02",
        decision="approve",
        reviewed_at_utc="2026-08-30T00:01:00Z",
        rationale="The complete plots and acquisition notes were reviewed.",
    )
    assert synchronization_is_authorized(receipt, manual_approval=approval)


def test_config_contract_loads_and_end_to_end_signal_path():
    loaded = load_synchronization_config("configs/sync_marker_affine_v1.yaml")
    assert loaded.expected_payload_bins == 182
    assert loaded.rsp_fixed_high == 8.5

    rng = np.random.default_rng(7)
    config = _decision_config(
        radar_sample_rate_hz=10.0,
        rsp_sample_rate_hz=20.0,
        min_marker_span_s=20.0,
        min_affine_span_s=40.0,
        radar_marker_merge_s=2.0,
        rsp_marker_merge_s=2.0,
        radar_marker_z=5.0,
    )
    radar_times = np.arange(0.0, 100.0, 0.1)
    rsp_times = np.arange(0.0, 110.0, 0.05)
    radar = rng.normal(0.0, 0.02, size=(3, radar_times.size, 182)).astype(np.float32)
    rsp = 4.0 + 0.3 * np.sin(2 * np.pi * 0.25 * rsp_times)
    for radar_event in [10.0, 45.0, 80.0]:
        radar_index = int(round(radar_event * config.radar_sample_rate_hz))
        radar[:, radar_index, 30:150] += 4.0
        rsp_event = radar_event + 3.0
        rsp[int(round(rsp_event * config.rsp_sample_rate_hz))] = 10.0

    result = synchronize_from_signals(
        radar,
        rsp,
        epoch_prior_offset_s=3.1,
        radar_times_s=radar_times,
        rsp_times_s=rsp_times,
        config=config,
    )
    assert result.decision == "accepted"
    assert result.mapping is not None
    assert result.mapping.offset_s == pytest.approx(3.0, abs=0.15)
    assert result.diagnostics["radar_payload_interpretation"] == (
        "182_real_float_payload_values"
    )
