from __future__ import annotations

from dataclasses import replace
import hashlib
import json

import numpy as np
import pytest

from snn_rr.radar_timing import (
    CAUSAL_UNIFORM_INVALID_REASON_SCHEMA_V1,
    CAUSAL_UNIFORM_INVALID_REASON_SEMANTICS_SHA256_V1,
    CAUSAL_UNIFORM_RESAMPLE_SCHEMA_V1,
    CausalUniformInvalidReasonV1,
    CausalUniformRadarResampleV1,
    RadarTimingError,
    canonical_ndarray_sha256,
    causal_uniform_invalid_reason_semantics_v1,
    causal_uniform_resample_radar_views_v1,
    validate_causal_uniform_invalid_reason_contract_v1,
)


def _resample(
    values: list[np.ndarray],
    times: list[np.ndarray],
    *,
    starts: list[float] | None = None,
    sequences: list[np.ndarray] | None = None,
    output_hz: object = 10.0,
    max_gap_s: float = 0.050,
    gap_policy: str = "mask",
):
    view_count = len(values)
    return causal_uniform_resample_radar_views_v1(
        values,
        times,
        [1_800_000_000.0] * view_count if starts is None else starts,
        (
            [np.arange(len(item), dtype=np.uint32) for item in values]
            if sequences is None
            else sequences
        ),
        output_hz=output_hz,
        max_gap_s=max_gap_s,
        gap_policy=gap_policy,
        timestamp_sources=["meta_v13"] * view_count,
        require_measured_timestamps=True,
    )


def test_regular_40hz_is_legacy_four_frame_mean_with_right_edge_times() -> None:
    frames = np.arange(24, dtype=np.float32).reshape(12, 2)
    values = [frames, frames + 100.0, frames - 50.0]
    times = [np.arange(12, dtype=np.float64) / 40.0 for _ in values]

    result = _resample(values, times, max_gap_s=0.030)
    expected = np.stack(
        [item.reshape(3, 4, 2).mean(axis=1, dtype=np.float32) for item in values]
    )

    np.testing.assert_array_equal(result.values, expected)
    np.testing.assert_allclose(result.times_s, [0.1, 0.2, 0.3], atol=1e-12)
    np.testing.assert_array_equal(result.sample_counts, np.full((3, 3), 4))
    assert result.valid_mask.all()
    np.testing.assert_array_equal(
        result.invalid_reason_mask,
        np.zeros(result.valid_mask.shape, dtype=np.uint8),
    )
    assert result.summary["schema_version"] == CAUSAL_UNIFORM_RESAMPLE_SCHEMA_V1
    assert result.summary["timestamp_semantics"] == "right_edge_exclusive"
    contract = result.summary["invalid_reason_contract"]
    assert contract["semantics"]["schema_version"] == (
        CAUSAL_UNIFORM_INVALID_REASON_SCHEMA_V1
    )
    assert contract["semantics_sha256"] == (
        CAUSAL_UNIFORM_INVALID_REASON_SEMANTICS_SHA256_V1
    )
    assert contract["all_invalid_explained"]
    assert validate_causal_uniform_invalid_reason_contract_v1(result)[
        "all_invalid_explained"
    ]


def test_finite_float32_mean_cannot_overflow_a_valid_output() -> None:
    maximum = np.finfo(np.float32).max
    values = np.full((8, 2), maximum, dtype=np.float32)
    values[:, 1] *= np.asarray(
        [1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0],
        dtype=np.float32,
    )
    times = np.arange(8, dtype=np.float64) / 40.0

    with np.errstate(over="raise", invalid="raise"):
        result = _resample([values], [times], max_gap_s=0.030)

    assert result.valid_mask.all()
    assert np.isfinite(result.values[result.valid_mask]).all()
    np.testing.assert_array_equal(result.values[0, :, 0], maximum)
    np.testing.assert_array_equal(result.values[0, :, 1], 0.0)
    assert result.invalid_reason_mask[0].tolist() == [0, 0]
    validate_causal_uniform_invalid_reason_contract_v1(result)


def test_finite_but_float32_unrepresentable_mean_is_structurally_masked() -> None:
    beyond_float32 = np.float64(np.finfo(np.float32).max) * 2.0
    values = np.full((8, 1), beyond_float32, dtype=np.float64)
    times = np.arange(8, dtype=np.float64) / 40.0

    result = _resample([values], [times], max_gap_s=0.030)

    expected_reason = int(CausalUniformInvalidReasonV1.NONFINITE_PAYLOAD)
    assert result.invalid_reason_mask[0].tolist() == [expected_reason, expected_reason]
    assert not result.valid_mask.any()
    np.testing.assert_array_equal(result.values, np.zeros_like(result.values))
    assert not np.signbit(result.values).any()
    semantics = causal_uniform_invalid_reason_semantics_v1()
    nonfinite = next(
        item for item in semantics["flags"] if item["name"] == "nonfinite_payload"
    )
    assert "represented as finite float32" in nonfinite["condition"]
    validate_causal_uniform_invalid_reason_contract_v1(result)


def test_invalid_reason_validator_rejects_nonfinite_valid_output_even_if_resealed() -> None:
    times = np.arange(8, dtype=np.float64) / 40.0
    result = _resample(
        [np.arange(8, dtype=np.float32)[:, None]],
        [times],
        max_gap_s=0.030,
    )
    tampered_values = result.values.copy()
    tampered_values[0, 0, 0] = np.inf
    content_hashes = {
        **result.summary["content_hashes"],
        "output_values_sha256": canonical_ndarray_sha256(tampered_values),
    }
    tampered_summary = {
        **result.summary,
        "content_hashes": content_hashes,
        "transform_evidence_sha256": hashlib.sha256(
            json.dumps(
                content_hashes,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest(),
    }

    with pytest.raises(RadarTimingError, match="valid resampling cells.*finite"):
        validate_causal_uniform_invalid_reason_contract_v1(
            replace(result, values=tampered_values, summary=tampered_summary)
        )


def test_future_values_and_boundary_impulse_cannot_change_earlier_outputs() -> None:
    times = np.arange(16, dtype=np.float64) / 40.0
    baseline = np.zeros((16, 1), dtype=np.float32)
    boundary_impulse = baseline.copy()
    boundary_impulse[4] = 8.0  # t=0.1 belongs to [0.1, 0.2), not [0.0, 0.1).

    impulse_result = _resample([boundary_impulse], [times], max_gap_s=0.030)
    assert impulse_result.values[0, 0, 0] == 0.0
    assert impulse_result.values[0, 1, 0] == 2.0

    changed_future = boundary_impulse.copy()
    changed_future[times >= 0.2] = 1000.0
    changed_result = _resample([changed_future], [times], max_gap_s=0.030)
    np.testing.assert_array_equal(
        impulse_result.values[:, :2], changed_result.values[:, :2]
    )


def test_independent_view_offsets_and_jitter_align_on_physical_grid() -> None:
    starts = [1_800_000_000.000, 1_800_000_000.006, 1_799_999_999.995]
    origin = float(np.median(starts))
    base = np.arange(28, dtype=np.float64) / 40.0
    jitters = [
        0.0007 * np.sin(np.arange(28)),
        0.0006 * np.cos(np.arange(28)),
        0.0005 * np.sin(0.7 * np.arange(28)),
    ]
    times = [base + jitter - jitter[0] for jitter in jitters]
    aligned = [
        item + (start - origin) for item, start in zip(times, starts, strict=True)
    ]
    # A value determined by physical 100 ms interval should align even though
    # raw frame indices and timestamps differ between views.
    values = [
        np.floor((physical + 1e-10) / 0.1).astype(np.float32)[:, None]
        for physical in aligned
    ]

    result = _resample(
        values,
        times,
        starts=starts,
        max_gap_s=0.040,
    )

    assert result.valid_mask.all()
    np.testing.assert_allclose(result.values[0], result.values[1], atol=0.0)
    np.testing.assert_allclose(result.values[0], result.values[2], atol=0.0)
    np.testing.assert_allclose(np.diff(result.times_s), 0.1, atol=1e-12)
    assert result.summary["first_grid_left_edge_s"] >= max(item[0] for item in aligned)


def test_timestamp_gap_is_exact_zero_mask_or_fail_closed() -> None:
    complete_times = np.arange(12, dtype=np.float64) / 40.0
    keep = np.ones(12, dtype=bool)
    keep[2] = False  # 50 ms between adjacent retained frames in the first bin.
    times = complete_times[keep]
    values = np.arange(12, dtype=np.float32)[keep, None]
    sequences = [np.arange(len(times), dtype=np.uint32)]

    masked = _resample(
        [values],
        [times],
        sequences=sequences,
        max_gap_s=0.040,
    )
    assert not masked.valid_mask[0, 0]
    assert np.count_nonzero(masked.values[0, 0]) == 0
    assert masked.invalid_reason_mask[0, 0] == int(
        CausalUniformInvalidReasonV1.TEMPORAL_GAP
    )
    assert masked.summary["per_view"][0]["temporal_gap_interval_count"] == 1

    with pytest.raises(RadarTimingError, match="invalid view-intervals"):
        _resample(
            [values],
            [times],
            sequences=sequences,
            max_gap_s=0.040,
            gap_policy="raise",
        )


def test_sequence_gap_is_masked_even_when_timestamps_look_regular() -> None:
    times = np.arange(12, dtype=np.float64) / 40.0
    values = np.ones((12, 1), dtype=np.float32)
    sequence = np.arange(12, dtype=np.uint32)
    sequence[4:] += 1

    result = _resample(
        [values],
        [times],
        sequences=[sequence],
        max_gap_s=0.030,
    )

    assert result.valid_mask[0].tolist() == [True, False, True]
    assert result.values[0, 1, 0] == 0.0
    assert result.invalid_reason_mask[0, 1] == int(
        CausalUniformInvalidReasonV1.FRAME_SEQUENCE_GAP
    )
    assert result.summary["per_view"][0]["frame_sequence_gap_count"] == 1
    assert result.summary["per_view"][0]["sequence_gap_interval_count"] == 1


@pytest.mark.parametrize(
    "sequence",
    [
        np.asarray([0.0, 1.0, 2.0, 3.0]),
        np.asarray([0.0, 1.0, 2.5, 3.0]),
        np.asarray([0.0, 1.0, np.nan, 3.0]),
        np.asarray([0, 1, 2, 3], dtype=np.uint64)
        + np.uint64(np.iinfo(np.int64).max),
        np.asarray([False, True, True, False]),
        np.asarray([0 + 0j, 1 + 0j, 2 + 1j, 3 + 0j]),
        np.asarray([0.0, 1.0, float(2**63), 3.0], dtype=np.float64),
    ],
)
def test_frame_sequence_must_be_exact_finite_int64(sequence: np.ndarray) -> None:
    times = np.arange(sequence.size, dtype=np.float64) * 0.025
    values = np.arange(sequence.size, dtype=np.float32)[:, None]
    with pytest.raises(RadarTimingError, match="not integral"):
        _resample([values], [times], sequences=[sequence], max_gap_s=0.030)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("output_hz", True),
        ("output_hz", "10.0"),
        ("max_gap_s", True),
        ("max_gap_s", "0.03"),
    ],
)
def test_resampling_controls_reject_bool_and_string_numbers(
    field: str, value: object
) -> None:
    times = np.arange(12, dtype=np.float64) / 40.0
    values = np.ones((12, 1), dtype=np.float32)
    kwargs = {field: value}
    with pytest.raises(RadarTimingError, match="must be"):
        _resample([values], [times], **kwargs)


@pytest.mark.parametrize(
    ("kind", "value"),
    [
        ("starts", [True]),
        ("starts", ["1800000000.0"]),
        ("times", np.asarray(["0", "0.025", "0.050", "0.075"])),
        ("times", np.asarray([False, True, True, True])),
        ("payload", np.asarray([[False], [True], [False], [True]])),
        ("payload", np.asarray([["0"], ["1"], ["2"], ["3"]])),
    ],
)
def test_timing_arrays_reject_bool_and_string_numeric_coercion(
    kind: str, value: object
) -> None:
    times: object = np.arange(4, dtype=np.float64) / 40.0
    payload: object = np.ones((4, 1), dtype=np.float32)
    starts: object = [1_800_000_000.0]
    if kind == "starts":
        starts = value
    elif kind == "times":
        times = value
    else:
        payload = value
    with pytest.raises(RadarTimingError, match="real numeric|finite real"):
        _resample([payload], [times], starts=starts, max_gap_s=0.030)


def test_1280_regular_frames_form_exactly_one_320_sample_32s_support() -> None:
    frame_count = 32 * 40
    times = np.arange(frame_count, dtype=np.float64) / 40.0
    values = [np.ones((frame_count, 1), dtype=np.float32) for _ in range(3)]

    result = _resample(values, [times, times, times], max_gap_s=0.030)

    assert result.values.shape == (3, 320, 1)
    assert result.valid_mask.all()
    assert result.summary["first_grid_left_edge_s"] == pytest.approx(0.0)
    assert result.times_s[0] == pytest.approx(0.1)
    assert result.times_s[-1] == pytest.approx(32.0)
    assert result.times_s[-1] - result.summary["first_grid_left_edge_s"] == pytest.approx(
        32.0
    )


def test_bounded_plateau_is_retained_and_structurally_masked() -> None:
    times = np.asarray([0.0, 0.025, 0.050, 0.050, 0.100, 0.125, 0.150, 0.175])
    values = np.arange(8, dtype=np.float32)[:, None]
    result = _resample([values], [times], max_gap_s=0.030)
    repair = result.summary["per_view"][0]["timestamp_repair"]
    assert repair["timestamp_plateau_count"] == 1
    assert repair["reconstructed_frame_count"] == 0
    assert repair["maximum_timestamp_correction_s"] == 0.0
    assert repair["reconstruction_method"] == "none_structural_mask_required"
    assert result.valid_mask[0].tolist() == [False, True]
    assert result.sample_counts[0].tolist() == [4, 4]
    assert result.values[0, 0, 0] == 0.0
    assert result.values[0, 1, 0] == pytest.approx(5.5)
    assert result.summary["per_view"][0]["timestamp_plateau_interval_count"] == 1
    assert result.invalid_reason_mask[0, 0] & int(
        CausalUniformInvalidReasonV1.TIMESTAMP_PLATEAU
    )


def test_measured_timestamp_fallback_can_be_rejected() -> None:
    values = np.arange(8, dtype=np.float32)[:, None]

    with pytest.raises(RadarTimingError, match="rejected source"):
        causal_uniform_resample_radar_views_v1(
            [values],
            [np.arange(8, dtype=np.float64) / 40.0],
            [1_800_000_000.0],
            [np.arange(8, dtype=np.uint32)],
            timestamp_sources=["fallback_40hz"],
            require_measured_timestamps=True,
        )


def test_boundary_plateaus_retain_payload_and_fail_closed_by_mask() -> None:
    times = np.asarray(
        [0.0, 0.0, 0.025, 0.050, 0.075, 0.100, 0.125, 0.150, 0.175, 0.175]
    )
    values = np.asarray(
        [100.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 200.0],
        dtype=np.float32,
    )[:, None]

    result = _resample([values], [times], max_gap_s=0.030)

    np.testing.assert_array_equal(result.values[0, :, 0], [0.0, 0.0])
    assert result.valid_mask[0].tolist() == [False, False]
    assert result.sample_counts[0].tolist() == [5, 5]
    view = result.summary["per_view"][0]
    assert view["original_frame_count"] == 10
    assert view["frame_count"] == 10
    assert view["leading_boundary_frames_trimmed"] == 0
    assert view["trailing_boundary_frames_trimmed"] == 0
    assert view["leading_boundary_duplicate_frame_count"] == 1
    assert view["trailing_boundary_duplicate_frame_count"] == 1
    assert view["unaccounted_payload_frame_count"] == 0
    repair = view["timestamp_repair"]
    assert repair["leading_boundary_frames_trimmed"] == 0
    assert repair["trailing_boundary_frames_trimmed"] == 0
    assert repair["boundary_plateau_policy"].startswith("retain_all")


def test_plateau_with_sequence_gap_is_structurally_masked_not_interpolated() -> None:
    times = [0.0, 0.025, 0.025, 0.050, 0.075, 0.100, 0.125, 0.150]
    sequence = [0, 1, 3, 4, 5, 6, 7, 8]
    values = np.arange(len(times), dtype=np.float32)[:, None]
    result = _resample(
        [values],
        [np.asarray(times, dtype=np.float64)],
        sequences=[np.asarray(sequence, dtype=np.uint32)],
        max_gap_s=0.030,
    )
    assert not result.valid_mask[0, 0]
    assert result.values[0, 0, 0] == 0.0
    assert result.invalid_reason_mask[0, 0] & int(
        CausalUniformInvalidReasonV1.TIMESTAMP_PLATEAU
    )
    assert result.invalid_reason_mask[0, 0] & int(
        CausalUniformInvalidReasonV1.FRAME_SEQUENCE_GAP
    )
    view = result.summary["per_view"][0]
    assert view["timestamp_plateau_interval_count"] == 1
    assert view["sequence_gap_interval_count"] == 1


def test_large_epoch_exact_millisecond_boundary_never_enters_earlier_bin() -> None:
    base_epoch = 1_800_000_000.0
    starts = [base_epoch, base_epoch + 0.100, base_epoch + 0.200]
    times = np.arange(24, dtype=np.float64) / 40.0
    values = [np.zeros((24, 1), dtype=np.float32) for _ in range(3)]
    # View 0's frame at local 300 ms is physically at common-clock 200 ms.
    # It belongs to [200, 300), never the output ending at 200 ms.
    values[0][12, 0] = 8.0

    result = _resample(values, [times, times, times], starts=starts, max_gap_s=0.030)

    assert result.summary["time_arithmetic"]["start_epoch_arithmetic"] == (
        "integer_millisecond_fixed_point"
    )
    assert result.summary["time_arithmetic"]["bin_membership_arithmetic"] == (
        "integer_nanosecond_fixed_point"
    )
    np.testing.assert_array_equal(result.times_s[:2], [0.2, 0.3])
    assert result.values[0, 0, 0] == 0.0
    assert result.values[0, 1, 0] == 2.0


def test_future_submicrosecond_timestamp_cannot_switch_earlier_bin_arithmetic() -> None:
    times = np.arange(24, dtype=np.float64) / 40.0
    changed_future_times = times.copy()
    changed_future_times[-1] += 0.3e-6
    values = np.zeros((24, 1), dtype=np.float32)
    values[12, 0] = 8.0  # t=0.3; belongs to [0.3, 0.4), not [0.2, 0.3).

    baseline = _resample([values], [times], max_gap_s=0.030)
    changed = _resample([values], [changed_future_times], max_gap_s=0.030)

    assert baseline.summary["time_arithmetic"]["bin_membership_arithmetic"] == (
        "integer_nanosecond_fixed_point"
    )
    assert changed.summary["time_arithmetic"]["bin_membership_arithmetic"] == (
        "integer_nanosecond_fixed_point"
    )
    assert not baseline.summary["time_arithmetic"][
        "arithmetic_policy_selected_from_timestamp_values"
    ]
    assert not changed.summary["time_arithmetic"][
        "arithmetic_policy_selected_from_timestamp_values"
    ]
    np.testing.assert_array_equal(baseline.values[:, :5], changed.values[:, :5])
    np.testing.assert_array_equal(
        baseline.sample_counts[:, :5], changed.sample_counts[:, :5]
    )
    np.testing.assert_array_equal(
        baseline.valid_mask[:, :5], changed.valid_mask[:, :5]
    )
    np.testing.assert_array_equal(
        baseline.invalid_reason_mask[:, :5],
        changed.invalid_reason_mask[:, :5],
    )
    assert baseline.values[0, 2, 0] == 0.0
    assert baseline.sample_counts[0, 2] == 4
    assert baseline.values[0, 3, 0] == 2.0

    nonexact_future_times = times.copy()
    nonexact_future_times[-1] += 0.3e-9
    nonexact = _resample([values], [nonexact_future_times], max_gap_s=0.030)
    assert not nonexact.summary["time_arithmetic"]["half_open_boundary_exact"]
    assert nonexact.summary["time_arithmetic"][
        "timestamp_quantization_max_abs_s"
    ] > 0.0
    np.testing.assert_array_equal(baseline.values[:, :5], nonexact.values[:, :5])
    np.testing.assert_array_equal(
        baseline.sample_counts[:, :5], nonexact.sample_counts[:, :5]
    )


def test_asymmetric_common_support_accounts_for_all_retained_frames() -> None:
    base_epoch = 1_800_000_000.0
    starts = [base_epoch, base_epoch + 0.050, base_epoch]
    times = np.arange(12, dtype=np.float64) / 40.0
    values = [np.arange(12, dtype=np.float32)[:, None] for _ in range(3)]

    result = _resample(
        values,
        [times, times, times],
        starts=starts,
        max_gap_s=0.030,
    )

    np.testing.assert_array_equal(result.sample_counts, np.full((3, 2), 4))
    for view in result.summary["per_view"]:
        accounting = view["frame_accounting"]
        categories = accounting["categories"]
        assert accounting["schema_version"] == "snn_rr.radar_frame_accounting.v1"
        assert accounting["retained_input_frame_count"] == 12
        assert categories["assigned_to_output_intervals_frame_count"] == 8
        assert accounting["category_sum"] == 12
        assert accounting["unaccounted_payload_frame_count"] == 0
        assert view["unaccounted_payload_frame_count"] == 0
        assert accounting["categories_disjoint"]
        assert accounting["coverage_complete"]
        assert accounting["assigned_count_matches_sample_counts"]

    first = result.summary["per_view"][0]["frame_accounting"]
    shifted = result.summary["per_view"][1]["frame_accounting"]
    assert first["before_common_complete_support_frame_count"] == 4
    assert first["after_common_complete_support_frame_count"] == 0
    assert shifted["before_common_complete_support_frame_count"] == 2
    assert shifted["after_common_complete_support_frame_count"] == 2
    assert first["categories"] == {
        "outside_common_intersection_prefix_frame_count": 2,
        "leading_partial_edge_frame_count": 2,
        "assigned_to_output_intervals_frame_count": 8,
        "trailing_partial_edge_frame_count": 0,
        "outside_common_intersection_suffix_frame_count": 0,
    }
    assert shifted["categories"] == {
        "outside_common_intersection_prefix_frame_count": 0,
        "leading_partial_edge_frame_count": 2,
        "assigned_to_output_intervals_frame_count": 8,
        "trailing_partial_edge_frame_count": 0,
        "outside_common_intersection_suffix_frame_count": 2,
    }


def test_plateau_classification_is_timestamp_prefix_invariant() -> None:
    prefix_times = np.asarray(
        [0.0, 0.025, 0.025, 0.050, 0.075, 0.100, 0.125, 0.150, 0.175],
        dtype=np.float64,
    )
    future_a = np.asarray([0.200, 0.225, 0.250, 0.275, 0.300, 0.325, 0.350, 0.375])
    future_b = np.asarray([0.200, 0.228, 0.257, 0.289, 0.320, 0.351, 0.383, 0.414])
    values = np.arange(len(prefix_times) + len(future_a), dtype=np.float32)[:, None]

    first = _resample(
        [values],
        [np.concatenate([prefix_times, future_a])],
        max_gap_s=0.040,
    )
    second = _resample(
        [values],
        [np.concatenate([prefix_times, future_b])],
        max_gap_s=0.040,
    )

    # Both calls have identical timestamp/value prefixes through 175 ms.  No
    # future anchor may alter either of the first two completed intervals.
    np.testing.assert_array_equal(first.values[:, :2], second.values[:, :2])
    np.testing.assert_array_equal(first.valid_mask[:, :2], second.valid_mask[:, :2])
    np.testing.assert_array_equal(first.sample_counts[:, :2], second.sample_counts[:, :2])
    np.testing.assert_array_equal(
        first.invalid_reason_mask[:, :2], second.invalid_reason_mask[:, :2]
    )
    assert first.summary["per_view"][0]["timestamp_repair"]["plateaus"] == (
        second.summary["per_view"][0]["timestamp_repair"]["plateaus"]
    )


def test_transform_content_hashes_bind_values_mask_and_counts() -> None:
    times = np.arange(12, dtype=np.float64) / 40.0
    values = np.arange(12, dtype=np.float32)[:, None]
    result = _resample([values], [times], max_gap_s=0.030)
    hashes = result.summary["content_hashes"]

    assert hashes["output_values_sha256"] == canonical_ndarray_sha256(result.values)
    assert hashes["valid_mask_sha256"] == canonical_ndarray_sha256(result.valid_mask)
    assert hashes["sample_counts_sha256"] == canonical_ndarray_sha256(
        result.sample_counts
    )
    assert hashes["invalid_reason_mask_sha256"] == canonical_ndarray_sha256(
        result.invalid_reason_mask
    )
    assert hashes["invalid_reason_semantics_sha256"] == (
        CAUSAL_UNIFORM_INVALID_REASON_SEMANTICS_SHA256_V1
    )
    assert len(result.summary["transform_evidence_sha256"]) == 64

    changed_values = values.copy()
    changed_values[1, 0] += 1.0
    changed = _resample([changed_values], [times], max_gap_s=0.030)
    changed_hashes = changed.summary["content_hashes"]
    assert changed_hashes["corrected_input_values_sha256"] != hashes[
        "corrected_input_values_sha256"
    ]
    assert changed_hashes["output_values_sha256"] != hashes["output_values_sha256"]
    assert changed_hashes["valid_mask_sha256"] == hashes["valid_mask_sha256"]
    assert changed_hashes["sample_counts_sha256"] == hashes["sample_counts_sha256"]

    tampered_output = result.values.copy()
    tampered_output[0, 0, 0] += 1.0
    assert canonical_ndarray_sha256(tampered_output) != hashes["output_values_sha256"]


def test_invalid_reason_bits_preserve_plateau_gap_sequence_nonfinite_overlap() -> None:
    times = np.asarray(
        [0.0, 0.025, 0.025, 0.075, 0.100, 0.125, 0.150, 0.175],
        dtype=np.float64,
    )
    sequence = np.asarray([0, 1, 3, 4, 5, 6, 7, 8], dtype=np.uint32)
    values = np.arange(times.size, dtype=np.float32)[:, None]
    values[1, 0] = np.nan

    result = _resample(
        [values],
        [times],
        sequences=[sequence],
        max_gap_s=0.030,
    )

    expected = int(
        CausalUniformInvalidReasonV1.TEMPORAL_GAP
        | CausalUniformInvalidReasonV1.FRAME_SEQUENCE_GAP
        | CausalUniformInvalidReasonV1.TIMESTAMP_PLATEAU
        | CausalUniformInvalidReasonV1.NONFINITE_PAYLOAD
    )
    assert result.invalid_reason_mask.dtype == np.uint8
    assert result.invalid_reason_mask[0].tolist() == [expected, 0]
    assert result.valid_mask[0].tolist() == [False, True]
    assert result.values[0, 0, 0] == 0.0
    assert not np.signbit(result.values[0, 0, 0])
    evidence = validate_causal_uniform_invalid_reason_contract_v1(result)
    assert evidence["overlap_interval_count"] == 1
    assert evidence["maximum_reason_multiplicity"] == 4
    assert evidence["per_reason_interval_counts"] == {
        "empty_interval": 0,
        "temporal_gap": 1,
        "frame_sequence_gap": 1,
        "timestamp_plateau": 1,
        "nonfinite_payload": 1,
    }


def test_empty_interval_preserves_overlapping_temporal_gap_reason() -> None:
    # The frame at 200 ms is right-edge exclusive, leaving [100, 200) empty.
    times = np.asarray([0.0, 0.200, 0.225, 0.250, 0.275], dtype=np.float64)
    values = np.arange(times.size, dtype=np.float32)[:, None]

    result = _resample([values], [times], max_gap_s=0.030)

    empty_and_gap = int(
        CausalUniformInvalidReasonV1.EMPTY_INTERVAL
        | CausalUniformInvalidReasonV1.TEMPORAL_GAP
    )
    assert result.sample_counts[0].tolist() == [1, 0, 4]
    assert result.invalid_reason_mask[0].tolist() == [
        int(CausalUniformInvalidReasonV1.TEMPORAL_GAP),
        empty_and_gap,
        0,
    ]
    assert result.values[0, 1, 0] == 0.0
    assert not np.signbit(result.values[0, 1, 0])
    contract = result.summary["invalid_reason_contract"]
    assert contract["per_reason_interval_counts"]["empty_interval"] == 1
    assert contract["per_reason_interval_counts"]["temporal_gap"] == 2
    assert contract["overlap_interval_count"] == 1
    assert contract["all_invalid_explained"]


def test_invalid_reason_validator_rejects_reason_union_hash_and_zero_tampering() -> None:
    times = np.asarray(
        [0.0, 0.025, 0.025, 0.075, 0.100, 0.125, 0.150, 0.175],
        dtype=np.float64,
    )
    values = np.arange(times.size, dtype=np.float32)[:, None]
    result = _resample([values], [times], max_gap_s=0.030)

    missing_union = result.invalid_reason_mask.copy()
    missing_union[0, 0] = 0
    with pytest.raises(RadarTimingError, match="exactly equal"):
        validate_causal_uniform_invalid_reason_contract_v1(
            replace(result, invalid_reason_mask=missing_union)
        )

    wrong_known_reason = result.invalid_reason_mask.copy()
    wrong_known_reason[0, 0] = np.uint8(
        CausalUniformInvalidReasonV1.FRAME_SEQUENCE_GAP
    )
    with pytest.raises(RadarTimingError, match="content hash mismatch"):
        validate_causal_uniform_invalid_reason_contract_v1(
            replace(result, invalid_reason_mask=wrong_known_reason)
        )

    unknown_reason = result.invalid_reason_mask.copy()
    unknown_reason[0, 0] |= np.uint8(1 << 7)
    with pytest.raises(RadarTimingError, match="unknown reason bit"):
        validate_causal_uniform_invalid_reason_contract_v1(
            replace(result, invalid_reason_mask=unknown_reason)
        )

    negative_zero = result.values.copy()
    negative_zero[0, 0] = np.float32(-0.0)
    with pytest.raises(RadarTimingError, match="exact positive zero"):
        validate_causal_uniform_invalid_reason_contract_v1(
            replace(result, values=negative_zero)
        )

    bool_laundered_summary = {
        **result.summary,
        "invalid_reason_contract": {
            **result.summary["invalid_reason_contract"],
            "invalid_interval_count": True,
        },
    }
    with pytest.raises(RadarTimingError, match="exact integer"):
        validate_causal_uniform_invalid_reason_contract_v1(
            replace(result, summary=bool_laundered_summary)
        )


def test_invalid_reason_validator_rejects_frame_accounting_tamper() -> None:
    times = np.arange(12, dtype=np.float64) / 40.0
    values = np.arange(12, dtype=np.float32)[:, None]
    result = _resample([values], [times], max_gap_s=0.030)
    tampered_summary = {
        **result.summary,
        "per_view": [
            {
                **result.summary["per_view"][0],
                "frame_accounting": {
                    **result.summary["per_view"][0]["frame_accounting"],
                    "unaccounted_payload_frame_count": 1,
                    "coverage_complete": False,
                },
            }
        ],
    }

    with pytest.raises(RadarTimingError, match="exactly accounted"):
        validate_causal_uniform_invalid_reason_contract_v1(
            replace(result, summary=tampered_summary)
        )


def test_s07_style_causally_unwrapped_counter_reset_remains_compatible() -> None:
    # S07's raw counter drops by seconds.  The parser repairs it from the past
    # edge and nominal 25 ms period before this resampler sees it.
    raw_ms = np.asarray(
        [4574, 4598, 4623, 243, 269, 294, 319, 344, 369, 394, 419, 444, 469],
        dtype=np.int64,
    )
    repaired_ms = raw_ms.astype(np.float64)
    offset = 0.0
    previous = repaired_ms[0]
    for index in range(1, repaired_ms.size):
        candidate = repaired_ms[index] + offset
        if candidate < previous - 100.0:
            offset += previous + 25.0 - candidate
            candidate = repaired_ms[index] + offset
        repaired_ms[index] = candidate
        previous = candidate
    relative_s = (repaired_ms - repaired_ms[0]) / 1000.0
    values = np.arange(relative_s.size, dtype=np.float32)[:, None]

    result = _resample([values], [relative_s], max_gap_s=0.030)

    np.testing.assert_array_equal(np.diff(repaired_ms) >= 0.0, True)
    assert result.valid_mask.all()
    np.testing.assert_array_equal(
        result.invalid_reason_mask,
        np.zeros_like(result.invalid_reason_mask),
    )
    assert result.summary["per_view"][0]["unaccounted_payload_frame_count"] == 0


def test_legacy_seven_field_result_stays_readable_but_has_no_reason_authority() -> None:
    times = np.arange(8, dtype=np.float64) / 40.0
    current = _resample(
        [np.arange(8, dtype=np.float32)[:, None]],
        [times],
        max_gap_s=0.030,
    )
    legacy = CausalUniformRadarResampleV1(
        current.origin_epoch_s,
        current.times_s,
        current.values,
        current.valid_mask,
        current.sample_counts,
        current.interval_s,
        current.summary,
    )

    assert legacy.invalid_reason_mask is None
    with pytest.raises(RadarTimingError, match="legacy resample result"):
        validate_causal_uniform_invalid_reason_contract_v1(legacy)

    mutated = causal_uniform_invalid_reason_semantics_v1()
    mutated["flags"][0]["value"] = 255
    assert causal_uniform_invalid_reason_semantics_v1()["flags"][0]["value"] == int(
        CausalUniformInvalidReasonV1.EMPTY_INTERVAL
    )
