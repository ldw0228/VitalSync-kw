"""Target-free, causal range-bin tracking for XeThru baseband frames.

The HAI recordings contain 182 float payload values per frame but no persisted
range calibration.  XeThru's baseband API normally exposes equally-sized I and
Q vectors, which is consistent with a ``[I[0:91], Q[0:91]]`` flattened frame.
An old local MATLAB utility instead paired adjacent floats.  This module keeps
both interpretations explicit, never converts a bin index to metres, and
retains a raw-payload fallback for format audits.

All sample-wise tracking is causal: output ``t`` depends only on frames
``[:t+1]``.  Session-level layout comparison is an offline, target-free audit
and is deliberately separate from the per-frame tracker so it cannot be
mistaken for a deployable hardware-format contract.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal

import numpy as np


IQLayout = Literal["split_halves", "interleaved", "raw_payload"]
COMPLEX_LAYOUTS: tuple[IQLayout, ...] = ("split_halves", "interleaved")


@dataclass(frozen=True, slots=True)
class RangeTrack:
    """Causal posterior summary for one radar and one layout hypothesis."""

    layout: IQLayout
    bin_index: np.ndarray
    confidence: np.ndarray
    normalized_entropy: np.ndarray
    missing: np.ndarray
    multimodal: np.ndarray
    evidence_strength: np.ndarray
    bin_count: int
    sample_rate_hz: float

    def __post_init__(self) -> None:
        length = len(self.bin_index)
        fields = (
            self.confidence,
            self.normalized_entropy,
            self.missing,
            self.multimodal,
            self.evidence_strength,
        )
        if any(len(item) != length for item in fields):
            raise ValueError("range-track arrays must have equal length")
        if self.bin_count < 2:
            raise ValueError("bin_count must be at least two")
        if self.sample_rate_hz <= 0:
            raise ValueError("sample_rate_hz must be positive")


@dataclass(frozen=True, slots=True)
class LayoutEvidence:
    """Target-free evidence comparing the two possible 91-bin I/Q layouts."""

    selected_layout: Literal["split_halves", "interleaved", "unknown"]
    split_halves_score: float
    interleaved_score: float
    score_margin: float
    minimum_margin: float
    reasons: tuple[str, ...]


def _validate_frames(frames: np.ndarray) -> np.ndarray:
    values = np.asarray(frames, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != 182:
        raise ValueError("radar frames must have shape [time, 182]")
    if len(values) < 2:
        raise ValueError("at least two radar frames are required")
    return np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)


def complex_frames(frames: np.ndarray, layout: IQLayout) -> np.ndarray:
    """Return a complex 91-bin view for an explicit I/Q layout hypothesis."""

    values = _validate_frames(frames)
    if layout == "split_halves":
        return values[:, :91].astype(np.complex64) + 1j * values[:, 91:]
    if layout == "interleaved":
        return values[:, 0::2].astype(np.complex64) + 1j * values[:, 1::2]
    raise ValueError("raw_payload has no complex interpretation")


def range_amplitude(frames: np.ndarray, layout: IQLayout) -> np.ndarray:
    """Map raw payloads to non-negative coordinates without range calibration."""

    values = _validate_frames(frames)
    if layout == "raw_payload":
        return np.abs(values).astype(np.float32)
    return np.abs(complex_frames(values, layout)).astype(np.float32)


def _spatial_smooth(values: np.ndarray) -> np.ndarray:
    if len(values) < 3:
        return values.copy()
    result = values.copy()
    result[1:-1] = 0.25 * values[:-2] + 0.50 * values[1:-1] + 0.25 * values[2:]
    result[0] = 0.75 * values[0] + 0.25 * values[1]
    result[-1] = 0.25 * values[-2] + 0.75 * values[-1]
    return result


def _transition_kernel(radius: int) -> np.ndarray:
    radius = max(1, int(radius))
    coordinate = np.arange(-radius, radius + 1, dtype=np.float64)
    sigma = max(1.0, radius / 2.0)
    kernel = np.exp(-0.5 * (coordinate / sigma) ** 2)
    return kernel / kernel.sum()


def _local_peak_ratio(probability: np.ndarray, primary: int, exclusion: int = 3) -> float:
    if len(probability) <= 2 * exclusion + 1:
        return 0.0
    alternate = probability.copy()
    alternate[max(0, primary - exclusion) : min(len(alternate), primary + exclusion + 1)] = 0
    return float(np.max(alternate) / max(float(probability[primary]), 1e-12))


def causal_range_track(
    frames: np.ndarray,
    *,
    fs: float = 40.0,
    layout: IQLayout = "split_halves",
    background_seconds: float = 8.0,
    noise_seconds: float = 20.0,
    evidence_smoothing_seconds: float = 0.35,
    max_jump_bins_per_second: float = 16.0,
    missing_strength: float = 1.35,
    multimodal_ratio: float = 0.68,
) -> RangeTrack:
    """Track the most likely active range coordinate with a causal Bayes filter.

    The measurement combines frame-to-frame motion and deviation from a slow
    background.  Per-bin scale estimates and the posterior are updated only
    after the current output has been emitted.  Low-evidence frames retain a
    propagated prior and are explicitly marked ``missing`` rather than
    manufacturing a confident location.
    """

    if fs <= 0:
        raise ValueError("fs must be positive")
    if min(background_seconds, noise_seconds, evidence_smoothing_seconds) <= 0:
        raise ValueError("time constants must be positive")
    values = _validate_frames(frames)
    if layout == "raw_payload":
        signal = values.astype(np.float64)
    else:
        signal = complex_frames(values, layout).astype(np.complex128)
    length, bins = signal.shape

    background_alpha = 1.0 - np.exp(-1.0 / (fs * background_seconds))
    noise_alpha = 1.0 - np.exp(-1.0 / (fs * noise_seconds))
    smooth_alpha = 1.0 - np.exp(-1.0 / (fs * evidence_smoothing_seconds))
    radius = max(1, int(round(max_jump_bins_per_second / fs * 4.0)))
    kernel = _transition_kernel(radius)

    path = np.zeros(length, dtype=np.int16)
    confidence = np.zeros(length, dtype=np.float32)
    entropy = np.ones(length, dtype=np.float32)
    missing = np.ones(length, dtype=bool)
    multimodal = np.zeros(length, dtype=bool)
    strength_out = np.zeros(length, dtype=np.float32)

    background = signal[0].copy()
    previous_signal = signal[0].copy()
    smooth = np.zeros(bins, dtype=np.float64)
    initial_scale = max(float(np.median(np.abs(signal[1] - signal[0]))), 1e-9)
    noise = np.full(bins, initial_scale, dtype=np.float64)
    posterior = np.full(bins, 1.0 / bins, dtype=np.float64)
    log_bins = np.log(float(bins))

    for index in range(length):
        current = signal[index]
        motion = np.abs(current - previous_signal) if index else np.zeros(bins)
        foreground = np.abs(current - background)
        measurement = motion + 0.30 * foreground
        smooth += smooth_alpha * (measurement - smooth)

        standardized = smooth / np.maximum(noise, initial_scale * 0.05)
        standardized = _spatial_smooth(np.clip(standardized, 0.0, 40.0))
        center = float(np.median(standardized))
        mad = float(np.median(np.abs(standardized - center)))
        robust_scale = max(1.4826 * mad, 0.15)
        zscore = np.clip((standardized - center) / robust_scale, -4.0, 12.0)
        strength = max(0.0, float(np.quantile(zscore, 0.95)))

        prior = np.convolve(posterior, kernel, mode="same")
        prior = 0.995 * prior + 0.005 / bins
        logits = np.clip(zscore / 1.8, -8.0, 8.0)
        likelihood = np.exp(logits - float(np.max(logits)))
        likelihood /= max(float(likelihood.sum()), 1e-20)
        if strength < missing_strength:
            updated = prior
        else:
            updated = prior * (0.02 / bins + 0.98 * likelihood)
        posterior = updated / max(float(updated.sum()), 1e-20)

        primary = int(np.argmax(posterior))
        normalized_entropy = float(
            -np.sum(posterior * np.log(np.maximum(posterior, 1e-20))) / log_bins
        )
        posterior_peak_ratio = _local_peak_ratio(posterior, primary)
        measurement_primary = int(np.argmax(likelihood))
        measurement_peak_ratio = _local_peak_ratio(likelihood, measurement_primary)
        separation = 1.0 - min(1.0, posterior_peak_ratio)
        detection = float(np.clip((strength - missing_strength) / 4.0, 0.0, 1.0))
        current_confidence = (1.0 - normalized_entropy) * (0.35 + 0.65 * separation) * detection

        path[index] = primary
        entropy[index] = normalized_entropy
        confidence[index] = current_confidence
        strength_out[index] = strength
        missing[index] = bool(strength < missing_strength or current_confidence < 0.015)
        multimodal[index] = bool(
            not missing[index]
            and measurement_peak_ratio >= multimodal_ratio
        )

        # State updates occur after emitting the current result.  This ordering
        # is what makes a prefix bit-identical when future frames are appended.
        background += background_alpha * (current - background)
        noise += noise_alpha * (measurement - noise)
        previous_signal = current

    return RangeTrack(
        layout=layout,
        bin_index=path,
        confidence=confidence,
        normalized_entropy=entropy,
        missing=missing,
        multimodal=multimodal,
        evidence_strength=strength_out,
        bin_count=bins,
        sample_rate_hz=float(fs),
    )


def _layout_score(frames: np.ndarray, track: RangeTrack, layout: IQLayout) -> tuple[float, dict[str, float]]:
    amplitude = range_amplitude(frames, layout).astype(np.float64)
    median_profile = np.median(amplitude, axis=0)
    spatial_jump = float(
        np.median(np.abs(np.diff(median_profile)))
        / max(float(np.median(np.abs(median_profile))), 1e-12)
    )
    valid = ~track.missing
    valid_fraction = float(np.mean(valid))
    mean_confidence = float(np.mean(track.confidence[valid])) if np.any(valid) else 0.0
    multimodal_fraction = float(np.mean(track.multimodal[valid])) if np.any(valid) else 0.0
    if len(track.bin_index) > 1:
        continuity = float(np.mean(np.abs(np.diff(track.bin_index.astype(float))) <= 4.0))
    else:
        continuity = 0.0
    smoothness = 1.0 / (1.0 + spatial_jump)
    score = (
        0.45 * smoothness
        + 0.25 * valid_fraction
        + 0.20 * mean_confidence
        + 0.10 * continuity
        - 0.15 * multimodal_fraction
    )
    return float(score), {
        "spatial_jump": spatial_jump,
        "spatial_smoothness": smoothness,
        "valid_fraction": valid_fraction,
        "mean_confidence": mean_confidence,
        "continuity": continuity,
        "multimodal_fraction": multimodal_fraction,
    }


def compare_iq_layouts(
    frames: np.ndarray,
    *,
    fs: float = 40.0,
    minimum_margin: float = 0.035,
    maximum_frames: int = 24_000,
) -> LayoutEvidence:
    """Compare layout hypotheses without BIOPAC, labels, or physical distance.

    The comparison may inspect a session-wide deterministic subsample and is
    therefore an offline format audit.  A small margin returns ``unknown``;
    callers must not silently choose whichever score is microscopically larger.
    """

    values = _validate_frames(frames)
    if maximum_frames < 2:
        raise ValueError("maximum_frames must be at least two")
    if len(values) > maximum_frames:
        step = int(np.ceil(len(values) / maximum_frames))
        values = values[::step]
        effective_fs = fs / step
    else:
        effective_fs = fs

    split_track = causal_range_track(values, fs=effective_fs, layout="split_halves")
    interleaved_track = causal_range_track(values, fs=effective_fs, layout="interleaved")
    split_score, split_parts = _layout_score(values, split_track, "split_halves")
    interleaved_score, interleaved_parts = _layout_score(
        values, interleaved_track, "interleaved"
    )
    margin = abs(split_score - interleaved_score)
    reasons = [
        f"split_spatial_jump={split_parts['spatial_jump']:.6g}",
        f"interleaved_spatial_jump={interleaved_parts['spatial_jump']:.6g}",
        f"split_valid_fraction={split_parts['valid_fraction']:.6g}",
        f"interleaved_valid_fraction={interleaved_parts['valid_fraction']:.6g}",
        f"split_multimodal_fraction={split_parts['multimodal_fraction']:.6g}",
        f"interleaved_multimodal_fraction={interleaved_parts['multimodal_fraction']:.6g}",
    ]
    if margin < minimum_margin:
        selected: Literal["split_halves", "interleaved", "unknown"] = "unknown"
        reasons.append("score_margin_below_fail_safe_threshold")
    elif split_score > interleaved_score:
        selected = "split_halves"
        reasons.append("split_halves_has_higher_target_free_score")
    else:
        selected = "interleaved"
        reasons.append("interleaved_has_higher_target_free_score")
    return LayoutEvidence(
        selected_layout=selected,
        split_halves_score=split_score,
        interleaved_score=interleaved_score,
        score_margin=margin,
        minimum_margin=float(minimum_margin),
        reasons=tuple(reasons),
    )


RANGE_TRACK_FEATURE_NAMES: tuple[str, ...] = (
    "range_bin_median_normalized",
    "range_bin_iqr_normalized",
    "range_bin_displacement_normalized",
    "range_bin_speed_normalized_per_second",
    "range_confidence_mean",
    "range_confidence_p10",
    "range_entropy_mean",
    "range_missing_fraction",
    "range_multimodal_fraction",
    "range_track_available",
)


def range_track_window_features(
    track: RangeTrack, start: int, stop: int
) -> tuple[np.ndarray, tuple[str, ...]]:
    """Summarize a half-open causal track interval for model-side auxiliary use."""

    if not 0 <= start < stop <= len(track.bin_index):
        raise ValueError("range-track window is outside the available samples")
    bins = track.bin_index[start:stop].astype(np.float64)
    confidence = track.confidence[start:stop].astype(np.float64)
    entropy = track.normalized_entropy[start:stop].astype(np.float64)
    missing = track.missing[start:stop]
    multimodal = track.multimodal[start:stop]
    valid = ~missing
    denominator = max(1.0, track.bin_count - 1.0)
    if np.any(valid):
        selected = bins[valid]
        q25, q75 = np.quantile(selected, [0.25, 0.75])
        median = float(np.median(selected))
        displacement = float(selected[-1] - selected[0]) if len(selected) > 1 else 0.0
        speed = (
            float(np.mean(np.abs(np.diff(selected)))) * track.sample_rate_hz
            if len(selected) > 1
            else 0.0
        )
        conf_mean = float(np.mean(confidence[valid]))
        conf_p10 = float(np.quantile(confidence[valid], 0.10))
        available = 1.0
    else:
        median = q25 = q75 = displacement = speed = conf_mean = conf_p10 = 0.0
        available = 0.0
    features = np.asarray(
        [
            median / denominator,
            (q75 - q25) / denominator,
            displacement / denominator,
            speed / denominator,
            conf_mean,
            conf_p10,
            float(np.mean(entropy)),
            float(np.mean(missing)),
            float(np.mean(multimodal)),
            available,
        ],
        dtype=np.float32,
    )
    return features, RANGE_TRACK_FEATURE_NAMES


def fuse_range_track_window_features(
    tracks: Iterable[RangeTrack], start: int, stop: int
) -> tuple[np.ndarray, tuple[str, ...]]:
    """Concatenate three view summaries and target-free cross-view reliability."""

    items = tuple(tracks)
    if len(items) != 3:
        raise ValueError("exactly three radar tracks are required")
    per_view: list[np.ndarray] = []
    names: list[str] = []
    reliabilities: list[float] = []
    availabilities: list[float] = []
    for radar_index, item in enumerate(items, start=1):
        values, base_names = range_track_window_features(item, start, stop)
        per_view.append(values)
        names.extend(f"radar{radar_index}_{name}" for name in base_names)
        reliabilities.append(float(values[4] * (1.0 - values[7])))
        availabilities.append(float(values[-1]))
    reliability = np.asarray(reliabilities, dtype=np.float32)
    cross = np.asarray(
        [
            float(np.sum(availabilities)),
            float(np.max(reliability)),
            float(np.mean(reliability)),
            float(np.std(reliability)),
            float(np.all(np.asarray(availabilities) == 0.0)),
        ],
        dtype=np.float32,
    )
    names.extend(
        [
            "range_available_view_count",
            "range_view_reliability_max",
            "range_view_reliability_mean",
            "range_view_reliability_std",
            "range_all_views_missing",
        ]
    )
    return np.concatenate([*per_view, cross]), tuple(names)


__all__ = [
    "COMPLEX_LAYOUTS",
    "IQLayout",
    "LayoutEvidence",
    "RANGE_TRACK_FEATURE_NAMES",
    "RangeTrack",
    "causal_range_track",
    "compare_iq_layouts",
    "complex_frames",
    "fuse_range_track_window_features",
    "range_amplitude",
    "range_track_window_features",
]
