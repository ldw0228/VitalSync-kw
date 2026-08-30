from __future__ import annotations

import numpy as np
import pytest

from snn_rr.radar_timing import (
    CAUSAL_UNIFORM_RESAMPLE_SCHEMA_V1,
    RadarTimingError,
    canonical_ndarray_sha256,
    causal_uniform_resample_radar_views_v1,
)


def _resample(
    values: list[np.ndarray],
    times: list[np.ndarray],
    *,
    starts: list[float] | None = None,
    sequences: list[np.ndarray] | None = None,
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
        output_hz=10.0,
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
    assert result.summary["schema_version"] == CAUSAL_UNIFORM_RESAMPLE_SCHEMA_V1
    assert result.summary["timestamp_semantics"] == "right_edge_exclusive"


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
    assert result.summary["per_view"][0]["frame_sequence_gap_count"] == 1
    assert result.summary["per_view"][0]["sequence_gap_interval_count"] == 1


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
