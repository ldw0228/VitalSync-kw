from __future__ import annotations

import copy
import hashlib

import numpy as np
import pytest
import snn_rr.synchronization as synchronization_module

from snn_rr.synchronization import (
    MANUAL_APPROVAL_SCHEMA,
    MarkerCandidate,
    SynchronizationConfig,
    SynchronizationError,
    TimeMapping,
    apply_affine_sample_index,
    build_affine_sample_index,
    build_manual_approval,
    build_sync_receipt,
    canonical_content_sha256,
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


@pytest.mark.parametrize(
    "field",
    ["motion_range_quantile", "accept_min_confidence"],
)
def test_fractional_sync_gates_reject_yaml_booleans(field: str) -> None:
    values = SynchronizationConfig().to_dict()
    values[field] = True
    with pytest.raises(SynchronizationError):
        SynchronizationConfig(**values).validate()


def test_motion_guard_kernel_longer_than_signal_fails_cleanly() -> None:
    frames = np.zeros((1, 3, 182), dtype=np.float32)
    config = _decision_config(
        radar_sample_rate_hz=10.0,
        motion_smoothing_s=10.0,
    )
    with pytest.raises(SynchronizationError, match="no radar view"):
        robust_radar_motion_envelope(frames, config=config)


def _content_bound_manual_decision(receipt, *, decision="approve"):
    """Model a pre-existing v1 artifact without using the guarded builder."""

    mapping = receipt["result"]["mapping"]
    assert mapping is not None
    approval = {
        "schema": MANUAL_APPROVAL_SCHEMA,
        "session_id": receipt["session_id"],
        "reviewed_at_utc": "2026-08-30T00:01:00Z",
        "reviewer_id": "historical-reviewer",
        "decision": decision,
        "rationale": "Content-bound historical review artifact.",
        "sync_receipt_content_sha256": receipt["content_sha256"],
        "mapping_sha256": hashlib.sha256(canonical_json_bytes(mapping)).hexdigest(),
    }
    approval["content_sha256"] = canonical_content_sha256(approval)
    return approval


def _accepted_receipt():
    config = _decision_config()
    return build_sync_receipt(
        _accepted_result(),
        session_id="SYNTHETIC_ACCEPTED",
        config=config,
        input_bindings={"signals": {"sha256": "9" * 64}},
        created_at_utc="2026-08-30T00:00:00Z",
    )


def _rehash(document):
    document["content_sha256"] = canonical_content_sha256(document)
    return document


def _refresh_match_residual_summaries(receipt):
    mapping = receipt["result"]["mapping"]
    residuals = []
    for match in receipt["result"]["matches"]:
        residual = match["rsp_time_s"] - (
            mapping["offset_s"] + mapping["scale"] * match["radar_time_s"]
        )
        match["residual_s"] = residual
        residuals.append(residual)
    receipt["result"]["residual_rmse_s"] = float(
        np.sqrt(np.mean(np.square(residuals)))
    )
    receipt["result"]["residual_max_abs_s"] = float(np.max(np.abs(residuals)))


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


def test_resample_invalid_zero_boundaries_cannot_become_radar_markers():
    rng = np.random.default_rng(73)
    config = _decision_config(
        radar_sample_rate_hz=10.0,
        motion_smoothing_s=0.1,
        radar_marker_merge_s=1.0,
        radar_marker_z=4.0,
        radar_marker_prominence_z=1.0,
    )
    frames = rng.normal(2.0, 0.002, size=(3, 600, 182)).astype(np.float32)
    valid = np.ones((3, 600), dtype=np.bool_)
    # Causal resampling writes invalid intervals as exact zero.  Entering and
    # leaving this interval must not look like a broad body-motion marker.
    frames[:, 200:240] = 0.0
    valid[:, 200:240] = False
    envelope = robust_radar_motion_envelope(
        frames, radar_valid_mask=valid, config=config
    )
    markers = detect_radar_marker_candidates(envelope, config=config)
    guard_radius = int(np.ceil(config.motion_smoothing_s * config.radar_sample_rate_hz / 2.0))
    assert not envelope.valid_mask[200 - guard_radius : 241 + guard_radius].any()
    assert all(not 19.5 <= marker.time_s <= 24.5 for marker in markers)
    with pytest.raises(SynchronizationError, match="boolean"):
        robust_radar_motion_envelope(
            frames, radar_valid_mask=valid.astype(np.int8), config=config
        )


def test_numeric_zero_remains_available_when_structural_mask_is_true():
    config = _decision_config(radar_sample_rate_hz=10.0, motion_smoothing_s=0.2)
    frames = np.zeros((3, 100, 182), dtype=np.float32)
    valid = np.ones((3, 100), dtype=np.bool_)
    envelope = robust_radar_motion_envelope(
        frames, radar_valid_mask=valid, config=config
    )
    first_valid = int(np.flatnonzero(envelope.valid_mask)[0])
    assert first_valid <= 5
    assert envelope.valid_mask[first_valid:].all()
    assert not detect_radar_marker_candidates(envelope, config=config)


def test_every_valid_smoothed_output_is_invariant_to_invalid_payload_bytes():
    rng = np.random.default_rng(808)
    config = _decision_config(
        radar_sample_rate_hz=10.0,
        motion_smoothing_s=0.5,
        radar_marker_z=4.0,
    )
    frames = rng.normal(1.0, 0.01, size=(3, 240, 182)).astype(np.float32)
    valid = np.ones((3, 240), dtype=np.bool_)
    valid[:, 80:100] = False
    zero_placeholder = frames.copy()
    zero_placeholder[:, 80:100] = 0.0
    hostile_placeholder = frames.copy()
    hostile_placeholder[:, 80:100] = 1.0e6

    first = robust_radar_motion_envelope(
        zero_placeholder, radar_valid_mask=valid, config=config
    )
    second = robust_radar_motion_envelope(
        hostile_placeholder, radar_valid_mask=valid, config=config
    )
    np.testing.assert_array_equal(first.valid_mask, second.valid_mask)
    np.testing.assert_array_equal(
        first.robust_z[first.valid_mask], second.robust_z[second.valid_mask]
    )
    assert not first.valid_mask[75:105].any()


def test_partial_view_availability_topology_change_is_guarded() -> None:
    rng = np.random.default_rng(902)
    config = _decision_config(
        radar_sample_rate_hz=10.0,
        motion_smoothing_s=0.5,
    )
    frames = rng.normal(0.0, 0.01, size=(3, 240, 182)).astype(np.float32)
    valid = np.ones((3, 240), dtype=np.bool_)
    valid[0, 80:120] = False
    envelope = robust_radar_motion_envelope(
        frames, radar_valid_mask=valid, config=config
    )
    radius = int(
        np.ceil(config.motion_smoothing_s * config.radar_sample_rate_hz / 2.0)
    )
    entry_guard = np.flatnonzero(~envelope.valid_mask[70:90]) + 70
    exit_guard = np.flatnonzero(~envelope.valid_mask[110:140]) + 110
    assert (
        entry_guard.min() < 80 < entry_guard.max()
        and len(entry_guard) >= 2 * radius + 1
    )
    assert (
        exit_guard.min() < 120 < exit_guard.max()
        and len(exit_guard) >= 2 * radius + 1
    )
    assert envelope.valid_mask[95:105].all()


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


def test_ambiguity_search_checks_later_dangerous_solution(monkeypatch):
    radar = _markers([0.0, 100.0, 200.0])
    rsp = _markers([0.0, 100.0, 200.0], source="fixed_high")
    hypotheses = [
        TimeMapping("constant", 0.0),
        TimeMapping("constant", 10.0),
        TimeMapping("constant", 20.0),
    ]
    solutions = {
        0.0: synchronization_module._CandidateSolution(
            hypotheses[0], ((0, 0), (1, 1), (2, 2)), np.zeros(3), 0.0, 0.0, 10.0
        ),
        # This first separated candidate is harmless because it has fewer pairs.
        10.0: synchronization_module._CandidateSolution(
            hypotheses[1], ((0, 0), (1, 1)), np.zeros(2), 0.0, 0.0, 5.0
        ),
        # A later same-count, close-score solution must still trigger ambiguity.
        20.0: synchronization_module._CandidateSolution(
            hypotheses[2], ((0, 0), (1, 1), (2, 2)), np.zeros(3), 0.0, 0.0, 9.9
        ),
    }
    monkeypatch.setattr(
        synchronization_module,
        "_initial_hypotheses",
        lambda *args, **kwargs: hypotheses,
    )
    monkeypatch.setattr(
        synchronization_module,
        "_refine_hypothesis",
        lambda radar_values, rsp_values, mapping, prior, config: solutions[
            mapping.offset_s
        ],
    )
    config = _decision_config(
        prior_tolerance_s=100.0,
        ambiguity_mapping_separation_s=1.0,
        ambiguity_score_margin=0.2,
    )
    result = estimate_marker_time_mapping(
        radar, rsp, epoch_prior_offset_s=0.0, config=config
    )
    assert result.ambiguous
    assert result.decision == "manual_review_required"
    assert result.diagnostics["alternative_mapping"]["offset_s"] == 20.0


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
    config = _decision_config(min_marker_span_s=1_000.0)
    result = estimate_marker_time_mapping(
        _markers([0.0, 100.0, 200.0]),
        _markers([2.0, 102.0, 202.0], source="fixed_high"),
        epoch_prior_offset_s=2.0,
        config=config,
    )
    assert result.decision == "manual_review_required"
    assert not result.ambiguous
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
    assert not synchronization_is_authorized(receipt, manual_approval=approval)

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


def test_accepted_receipt_recomputes_epoch_prior_and_drift_gates():
    prior_tampered = copy.deepcopy(_accepted_receipt())
    prior_tampered["result"]["prior_offset_s"] += 100.0
    _rehash(prior_tampered)
    with pytest.raises(SynchronizationError, match="epoch prior gate"):
        validate_sync_receipt(prior_tampered)

    drift_tampered = copy.deepcopy(_accepted_receipt())
    mapping = drift_tampered["result"]["mapping"]
    mapping["mode"] = "affine"
    mapping["scale"] = 1.0 + 5_000e-6
    mapping["drift_ppm"] = (mapping["scale"] - 1.0) * 1_000_000.0
    for match in drift_tampered["result"]["matches"]:
        rsp_time = mapping["offset_s"] + mapping["scale"] * match["radar_time_s"]
        match["rsp_time_s"] = rsp_time
        drift_tampered["result"]["rsp_markers"][match["rsp_index"]][
            "time_s"
        ] = rsp_time
    _refresh_match_residual_summaries(drift_tampered)
    _rehash(drift_tampered)
    with pytest.raises(SynchronizationError, match="drift gate"):
        validate_sync_receipt(drift_tampered)


@pytest.mark.parametrize(
    ("field", "value"),
    [("automatically_authorized", "true"), ("ambiguous", "false")],
)
def test_receipt_authority_flags_reject_truthy_strings(field, value):
    tampered = copy.deepcopy(_accepted_receipt())
    tampered["result"][field] = value
    _rehash(tampered)

    with pytest.raises(SynchronizationError, match="must be boolean"):
        validate_sync_receipt(tampered)


@pytest.mark.parametrize(
    ("field_path", "value", "message"),
    [
        (("result", "confidence"), True, "finite JSON number"),
        (("result", "confidence"), "0.95", "finite JSON number"),
        (("result", "mapping", "offset_s"), "2.0", "finite JSON number"),
        (("result", "radar_markers", 0, "time_s"), False, "finite JSON number"),
        (("result", "matches", 0, "radar_index"), "0", "JSON integer"),
        (
            ("input_bindings", "signals", "sha256"),
            int("9" * 64),
            "lowercase SHA-256",
        ),
    ],
)
def test_receipt_rejects_bool_string_and_nonstring_authority_values(
    field_path, value, message
):
    tampered = copy.deepcopy(_accepted_receipt())
    target = tampered
    for key in field_path[:-1]:
        target = target[key]
    target[field_path[-1]] = value
    _rehash(tampered)

    with pytest.raises(SynchronizationError, match=message):
        validate_sync_receipt(tampered)


@pytest.mark.parametrize("value", [True, "2.0"])
def test_time_mapping_from_dict_rejects_numeric_coercion(value):
    with pytest.raises(SynchronizationError, match="finite JSON number"):
        TimeMapping.from_dict(
            {"mode": "constant", "offset_s": value, "scale": 1.0}
        )
    with pytest.raises(SynchronizationError, match="real numbers"):
        TimeMapping(mode="constant", offset_s=value, scale=1.0)


def test_match_indices_times_and_residuals_bind_exactly_to_markers_and_mapping():
    swapped_indices = copy.deepcopy(_accepted_receipt())
    swapped_indices["result"]["matches"][1]["radar_index"], swapped_indices[
        "result"
    ]["matches"][2]["radar_index"] = (
        swapped_indices["result"]["matches"][2]["radar_index"],
        swapped_indices["result"]["matches"][1]["radar_index"],
    )
    _rehash(swapped_indices)
    with pytest.raises(SynchronizationError, match="does not bind"):
        validate_sync_receipt(swapped_indices)

    time_tampered = copy.deepcopy(_accepted_receipt())
    time_tampered["result"]["matches"][0]["radar_time_s"] += 5e-10
    _rehash(time_tampered)
    with pytest.raises(SynchronizationError, match="does not bind"):
        validate_sync_receipt(time_tampered)

    residual_tampered = copy.deepcopy(_accepted_receipt())
    residual_tampered["result"]["matches"][0]["residual_s"] += 5e-10
    _rehash(residual_tampered)
    with pytest.raises(SynchronizationError, match="residual is inconsistent"):
        validate_sync_receipt(residual_tampered)


def test_mapping_must_equal_deterministic_proposal_even_if_residuals_are_rehashed():
    tampered = copy.deepcopy(_accepted_receipt())
    tampered["result"]["mapping"]["offset_s"] += 0.1
    _refresh_match_residual_summaries(tampered)
    _rehash(tampered)

    with pytest.raises(SynchronizationError, match="marker proposal"):
        validate_sync_receipt(tampered)


def test_legacy_manual_approval_cannot_authorize_review_only_mapping():
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
    assert validate_manual_approval(approval, receipt) == approval
    assert not synchronization_is_authorized(receipt, manual_approval=approval)


def test_rejected_mapping_cannot_be_manually_approved():
    config = _decision_config()
    result = estimate_marker_time_mapping(
        _markers([100.0]),
        _markers([102.0], source="fixed_high"),
        epoch_prior_offset_s=2.0,
        config=config,
    )
    assert result.decision == "rejected"
    assert result.mapping is not None
    receipt = build_sync_receipt(
        result,
        session_id="SYNTHETIC_REJECTED",
        config=config,
        input_bindings={"signals": {"sha256": "d" * 64}},
        created_at_utc="2026-08-30T00:00:00Z",
    )

    with pytest.raises(SynchronizationError, match="review-required"):
        build_manual_approval(
            receipt,
            reviewer_id="reviewer-03",
            decision="approve",
            reviewed_at_utc="2026-08-30T00:01:00Z",
            rationale="Audit-only attempted override.",
        )
    historical_approval = _content_bound_manual_decision(receipt)
    with pytest.raises(SynchronizationError, match="review-required"):
        validate_manual_approval(historical_approval, receipt)
    with pytest.raises(SynchronizationError, match="review-required"):
        synchronization_is_authorized(
            receipt, manual_approval=historical_approval
        )
    assert not synchronization_is_authorized(receipt)


def test_ambiguous_mapping_cannot_be_manually_approved():
    config = _decision_config(
        prior_tolerance_s=20.0,
        match_residual_gate_s=0.3,
        ambiguity_score_margin=1.0,
    )
    result = estimate_marker_time_mapping(
        _markers([100.0, 300.0, 500.0, 700.0]),
        _markers(
            [95.0, 105.0, 295.0, 305.0, 495.0, 505.0, 695.0, 705.0],
            source="adaptive_high+fixed_high",
        ),
        epoch_prior_offset_s=0.0,
        config=config,
    )
    assert result.decision == "manual_review_required"
    assert result.ambiguous
    receipt = build_sync_receipt(
        result,
        session_id="SYNTHETIC_AMBIGUOUS",
        config=config,
        input_bindings={"signals": {"sha256": "e" * 64}},
        created_at_utc="2026-08-30T00:00:00Z",
    )

    with pytest.raises(SynchronizationError, match="ambiguous"):
        build_manual_approval(
            receipt,
            reviewer_id="reviewer-04",
            decision="approve",
            reviewed_at_utc="2026-08-30T00:01:00Z",
            rationale="Audit-only attempted override.",
        )
    historical_approval = _content_bound_manual_decision(receipt)
    with pytest.raises(SynchronizationError, match="ambiguous"):
        validate_manual_approval(historical_approval, receipt)
    with pytest.raises(SynchronizationError, match="ambiguous"):
        synchronization_is_authorized(
            receipt, manual_approval=historical_approval
        )
    assert not synchronization_is_authorized(receipt)


def test_manual_rejection_is_retained_but_v1_mapping_never_authorizes():
    config = _decision_config()
    receipt = build_sync_receipt(
        _accepted_result(),
        session_id="SYNTHETIC_REVOKED",
        config=config,
        input_bindings={"signals": {"sha256": "f" * 64}},
        created_at_utc="2026-08-30T00:00:00Z",
    )
    rejection = build_manual_approval(
        receipt,
        reviewer_id="reviewer-05",
        decision="reject",
        reviewed_at_utc="2026-08-30T00:01:00Z",
        rationale="Visual review identified an acquisition mismatch.",
    )

    assert not synchronization_is_authorized(receipt)
    assert not synchronization_is_authorized(receipt, manual_approval=rejection)
    with pytest.raises(SynchronizationError, match="review-required"):
        build_manual_approval(
            receipt,
            reviewer_id="reviewer-05",
            decision="approve",
            reviewed_at_utc="2026-08-30T00:02:00Z",
            rationale="Automatic acceptance does not require a manual approval.",
        )


def test_self_consistent_shifted_marker_receipt_remains_diagnostic_only():
    receipt = build_sync_receipt(
        _accepted_result(),
        session_id="SYNTHETIC_SHIFTED",
        config=_decision_config(),
        input_bindings={"signals": {"sha256": "e" * 64}},
        created_at_utc="2026-08-30T00:00:00Z",
    )
    shifted = copy.deepcopy(receipt)
    shift_s = 1_000_000.0
    for marker in shifted["result"]["radar_markers"]:
        marker["time_s"] += shift_s
    for marker in shifted["result"]["rsp_markers"]:
        marker["time_s"] += shift_s
    for match in shifted["result"]["matches"]:
        match["radar_time_s"] += shift_s
        match["rsp_time_s"] += shift_s
    shifted["content_sha256"] = canonical_content_sha256(shifted)

    # Internal receipt consistency is intentionally still auditable, but this
    # self-hashed document cannot prove that markers came from the bound raw
    # bytes and therefore cannot grant synchronization authority.
    assert validate_sync_receipt(shifted)["result"]["decision"] == "accepted"
    assert not synchronization_is_authorized(shifted)


def test_pre_replay_semantics_receipt_remains_readable_but_unauthorized():
    historical = copy.deepcopy(_accepted_receipt())
    historical["algorithm"].pop("proposal_replay_semantics")
    _rehash(historical)

    assert validate_sync_receipt(historical)["result"]["decision"] == "accepted"
    assert not synchronization_is_authorized(historical)


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


def test_config_loader_binds_the_exact_consumed_snapshot_hash(tmp_path):
    source = tmp_path / "sync.yaml"
    payload = b"expected_payload_bins: 182\n"
    source.write_bytes(payload)

    loaded = load_synchronization_config(
        source, expected_sha256=hashlib.sha256(payload).hexdigest()
    )
    assert loaded.expected_payload_bins == 182
    with pytest.raises(SynchronizationError, match="consumed-byte SHA-256 mismatch"):
        load_synchronization_config(source, expected_sha256="0" * 64)


def test_measured_timing_config_binds_exact_radar_metadata_warning_allowlist():
    loaded = load_synchronization_config(
        "configs/sync_marker_affine_measured10_v2.yaml"
    )

    assert loaded.radar_metadata_warning_allowlist == {
        "S07_KDM": ["unwrapped 1 relative-timestamp counter reset(s)"]
    }
    assert loaded.to_dict()["radar_metadata_warning_allowlist"] == {
        "S07_KDM": ["unwrapped 1 relative-timestamp counter reset(s)"]
    }


def test_radar_metadata_warning_allowlist_rejects_duplicate_or_empty_content():
    with pytest.raises(SynchronizationError, match="repeats"):
        SynchronizationConfig(
            radar_metadata_warning_allowlist={"S07_KDM": ["warning", "warning"]}
        ).validate()
    with pytest.raises(SynchronizationError, match="invalid"):
        SynchronizationConfig(
            radar_metadata_warning_allowlist={"S07_KDM": [""]}
        ).validate()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("radar_marker_z", float("nan")),
        ("ambiguity_score_margin", float("nan")),
        ("rsp_fixed_high", float("inf")),
        ("min_marker_pairs", True),
    ],
)
def test_sync_config_rejects_nonfinite_and_wrong_numeric_types(field, value):
    document = SynchronizationConfig().to_dict()
    document[field] = value
    with pytest.raises(SynchronizationError):
        SynchronizationConfig.from_mapping(document)
