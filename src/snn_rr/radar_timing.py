"""Measured multi-radar timeline fusion with auditable plateau repair."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np


class RadarTimingError(ValueError):
    """Raised when a common measured radar clock cannot be reconstructed safely."""


@dataclass(frozen=True, slots=True)
class CommonRadarTimeline:
    origin_epoch_s: float
    times_s: np.ndarray
    summary: dict[str, Any]


def repair_common_timestamp_plateaus(
    measured_times_s: np.ndarray,
    frame_sequences: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Interpolate only bounded, sequence-contiguous duplicate-clock plateaus."""

    measured = np.asarray(measured_times_s, dtype=np.float64)
    sequences = np.asarray(frame_sequences)
    if measured.ndim != 1 or measured.size < 2:
        raise RadarTimingError("common radar time must contain at least two frames")
    if sequences.ndim != 2 or sequences.shape[1] != measured.size:
        raise RadarTimingError("frame_sequences must have shape [radar, frame]")
    delta = np.diff(measured)
    if np.any(delta < 0):
        first = int(np.flatnonzero(delta < 0)[0])
        raise RadarTimingError(f"fused radar time moves backwards at frame {first}")
    tied_edges = np.flatnonzero(delta == 0)
    positive = delta[delta > 0]
    if not tied_edges.size:
        return measured.copy(), {
            "timestamp_plateau_count": 0,
            "measured_tie_edge_count": 0,
            "reconstructed_frame_count": 0,
            "maximum_timestamp_correction_s": 0.0,
            "reconstruction_method": "none",
            "plateaus": [],
        }
    if not positive.size:
        raise RadarTimingError("common radar timestamps contain no positive clock interval")
    nominal_period = float(np.median(positive))
    groups = np.split(tied_edges, np.flatnonzero(np.diff(tied_edges) > 1) + 1)
    repaired = measured.copy()
    correction = np.zeros_like(repaired)
    reconstructed = 0
    plateau_documents: list[dict[str, Any]] = []
    for group in groups:
        first_plateau_frame = int(group[0])
        last_plateau_frame = int(group[-1] + 1)
        left = first_plateau_frame - 1
        right = last_plateau_frame + 1
        if left < 0 or right >= measured.size:
            raise RadarTimingError(
                "unbounded radar timestamp plateau cannot be reconstructed"
            )
        sequence_window = sequences[:, left : right + 1].astype(np.int64, copy=False)
        if np.any(np.diff(sequence_window, axis=1) != 1):
            raise RadarTimingError(
                "radar timestamp plateau does not have contiguous frame sequences"
            )
        interval_count = right - left
        bracket_step = float((measured[right] - measured[left]) / interval_count)
        if not 0.25 * nominal_period <= bracket_step <= 4.0 * nominal_period:
            raise RadarTimingError(
                "radar timestamp plateau is not bounded by a plausible clock span"
            )
        interior = measured[left] + bracket_step * np.arange(1, interval_count)
        repaired[left + 1 : right] = interior
        correction[left + 1 : right] = interior - measured[left + 1 : right]
        reconstructed += interval_count - 1
        plateau_documents.append(
            {
                "left_anchor_frame": left,
                "first_reconstructed_frame": left + 1,
                "last_reconstructed_frame": right - 1,
                "right_anchor_frame": right,
                "left_anchor_time_s": float(measured[left]),
                "right_anchor_time_s": float(measured[right]),
                "interpolated_period_s": bracket_step,
                "maximum_correction_s": float(
                    np.max(np.abs(interior - measured[left + 1 : right]))
                ),
            }
        )
    if np.any(np.diff(repaired) <= 0):
        raise RadarTimingError("reconstructed common radar time is not strictly increasing")
    return repaired, {
        "timestamp_plateau_count": len(groups),
        "measured_tie_edge_count": int(tied_edges.size),
        "reconstructed_frame_count": reconstructed,
        "maximum_timestamp_correction_s": float(np.max(np.abs(correction))),
        "reconstruction_method": "bounded_linear_interpolation",
        "nominal_positive_period_s": nominal_period,
        "plateaus": plateau_documents,
    }


def fuse_common_radar_timeline(
    relative_times_s: Sequence[np.ndarray],
    start_epochs_s: Sequence[float],
    frame_sequences: Sequence[np.ndarray],
) -> CommonRadarTimeline:
    """Fuse measured per-view clocks on a median relative timeline."""

    if not (
        len(relative_times_s) == len(start_epochs_s) == len(frame_sequences)
        and len(relative_times_s) >= 1
    ):
        raise RadarTimingError("radar timing inputs must have equal non-zero view counts")
    relative = [np.asarray(item, dtype=np.float64) for item in relative_times_s]
    sequences = [np.asarray(item) for item in frame_sequences]
    for view_index, (times, sequence) in enumerate(zip(relative, sequences, strict=True)):
        if times.ndim != 1 or sequence.ndim != 1 or len(times) != len(sequence):
            raise RadarTimingError(f"radar view {view_index} timing vectors are inconsistent")
        if not np.isfinite(times).all():
            raise RadarTimingError(f"radar view {view_index} time contains non-finite values")
        delta = np.diff(times)
        if np.any(delta < 0):
            first = int(np.flatnonzero(delta < 0)[0])
            raise RadarTimingError(
                f"radar view {view_index} time moves backwards at frame {first}"
            )
    starts = np.asarray(start_epochs_s, dtype=np.float64)
    if not np.isfinite(starts).all():
        raise RadarTimingError("radar start epochs must be finite")
    common = min(map(len, relative))
    if common < 2:
        raise RadarTimingError("radar recording has fewer than two common frames")
    origin = float(np.median(starts))
    aligned = np.stack(
        [
            times[:common] + (float(start) - origin)
            for times, start in zip(relative, starts, strict=True)
        ]
    )
    measured_common = np.median(aligned, axis=0)
    repaired, plateau_summary = repair_common_timestamp_plateaus(
        measured_common, np.stack([item[:common] for item in sequences])
    )
    periods = np.diff(aligned, axis=1)
    positive = periods[periods > 0]
    if not positive.size:
        raise RadarTimingError("radar views contain no positive frame interval")
    summary = {
        "origin_epoch_s": origin,
        "start_epochs_s": starts.tolist(),
        "start_spread_ms": 1000.0 * float(np.ptp(starts)),
        "common_frame_count": common,
        "per_radar_timestamp_ties": [
            int(np.count_nonzero(np.diff(times[:common]) == 0)) for times in relative
        ],
        "median_frame_period_ms": 1000.0 * float(np.median(positive)),
        "median_frame_rate_hz": 1.0 / float(np.median(positive)),
        "nominal_grid_endpoint_error_s": float(
            repaired[-1] - (common - 1) / 40.0
        ),
        **plateau_summary,
    }
    return CommonRadarTimeline(origin_epoch_s=origin, times_s=repaired, summary=summary)


def block_mean_times(times_s: np.ndarray, factor: int) -> np.ndarray:
    """Return the timestamp of non-overlapping causal block-mean samples."""

    values = np.asarray(times_s, dtype=np.float64)
    if values.ndim != 1 or factor <= 0:
        raise RadarTimingError("times must be a vector and factor must be positive")
    usable = len(values) - len(values) % factor
    if usable < factor:
        raise RadarTimingError("timeline is too short for one block")
    return values[:usable].reshape(-1, factor).mean(axis=1)


__all__ = [
    "RadarTimingError",
    "CommonRadarTimeline",
    "repair_common_timestamp_plateaus",
    "fuse_common_radar_timeline",
    "block_mean_times",
]
