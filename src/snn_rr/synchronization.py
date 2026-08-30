"""Fail-closed marker synchronization for XeThru radar and BIOPAC RSP.

The HAI acquisition has no common hardware trigger.  A deliberately large
body motion was recorded by the three radars while a chest press produced an
RSP excursion.  This module reconstructs that relation without treating the
182 XeThru payload floats as 92 complex values and without using respiratory
rate or any other prediction target.

The time mapping used throughout is explicit::

    rsp_time_seconds = offset_seconds + scale * radar_time_seconds

where ``drift_ppm = (scale - 1) * 1e6``.  Automated synchronization is
fail-closed: a proposed mapping is usable only when the result passes the
configured gates, or when a separately content-addressed manual approval
binds that exact proposal and mapping.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.signal import find_peaks
import yaml


EXPECTED_RADAR_PAYLOAD_BINS = 182
SYNC_CONFIG_SCHEMA = "snn_rr.sync_marker_affine.config.v1"
SYNC_RECEIPT_SCHEMA = "snn_rr.sync_marker_affine.v1"
MANUAL_APPROVAL_SCHEMA = "snn_rr.sync_marker_manual_approval.v1"


class SynchronizationError(ValueError):
    """Raised when synchronization input or provenance fails validation."""


@dataclass(frozen=True, slots=True)
class SynchronizationConfig:
    """Locked numerical and decision thresholds for marker synchronization."""

    expected_payload_bins: int = EXPECTED_RADAR_PAYLOAD_BINS
    radar_sample_rate_hz: float = 40.0
    rsp_sample_rate_hz: float = 250.0

    motion_smoothing_s: float = 0.25
    motion_clip_z: float = 30.0
    motion_range_quantile: float = 0.90
    radar_marker_z: float = 6.0
    radar_marker_prominence_z: float = 2.0
    radar_marker_merge_s: float = 4.0

    rsp_adaptive_z: float = 6.0
    rsp_fixed_high: float | None = 8.5
    rsp_fixed_low: float | None = None
    rsp_marker_merge_s: float = 4.0

    prior_tolerance_s: float = 12.0
    match_residual_gate_s: float = 0.80
    max_drift_ppm: float = 1_000.0
    min_marker_pairs: int = 3
    good_marker_pairs: int = 5
    min_marker_span_s: float = 120.0
    min_affine_span_s: float = 300.0
    min_affine_pairs: int = 3
    min_affine_improvement_s: float = 0.015
    min_affine_drift_ppm: float = 25.0

    accept_max_rmse_s: float = 0.30
    accept_max_abs_residual_s: float = 0.75
    accept_min_confidence: float = 0.80
    ambiguity_mapping_separation_s: float = 0.40
    ambiguity_rmse_margin_s: float = 0.05
    ambiguity_score_margin: float = 1.0
    manual_review_min_pairs: int = 2

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "SynchronizationConfig":
        """Load either a flat mapping or the canonical nested YAML layout."""

        if not isinstance(raw, Mapping):
            raise SynchronizationError("synchronization config must be a mapping")
        root: Mapping[str, Any] = raw
        if "synchronization" in root:
            unknown_document_keys = sorted(set(root) - {"schema", "synchronization"})
            if unknown_document_keys:
                raise SynchronizationError(
                    f"unknown config document keys: {unknown_document_keys}"
                )
            if "schema" in root and root["schema"] != SYNC_CONFIG_SCHEMA:
                raise SynchronizationError("unexpected synchronization config schema")
            nested = root["synchronization"]
            if not isinstance(nested, Mapping):
                raise SynchronizationError("synchronization must be a mapping")
            root = nested

        flattened: dict[str, Any] = {}
        scalar_names = set(cls.__dataclass_fields__)
        category_names = {"input", "radar_motion", "rsp_marker", "matching", "decision"}
        unknown_root = sorted(set(root) - scalar_names - category_names)
        if unknown_root:
            raise SynchronizationError(f"unknown synchronization keys: {unknown_root}")
        for key, value in root.items():
            if key in scalar_names:
                flattened[key] = value
            elif isinstance(value, Mapping):
                unknown_nested = sorted(set(value) - scalar_names)
                if unknown_nested:
                    raise SynchronizationError(
                        f"unknown synchronization keys in {key}: {unknown_nested}"
                    )
                for nested_key, nested_value in value.items():
                    if nested_key in scalar_names:
                        if nested_key in flattened:
                            raise SynchronizationError(
                                f"duplicate synchronization setting: {nested_key}"
                            )
                        flattened[nested_key] = nested_value
            else:
                raise SynchronizationError(f"synchronization category {key} must be a mapping")
        unknown = sorted(set(flattened) - scalar_names)
        if unknown:  # Defensive; currently impossible because of the filter above.
            raise SynchronizationError(f"unknown synchronization keys: {unknown}")
        config = cls(**flattened)
        config.validate()
        return config

    def validate(self) -> None:
        if self.expected_payload_bins != EXPECTED_RADAR_PAYLOAD_BINS:
            raise SynchronizationError(
                "sync v1 is bound to the exact 182-float XeThru payload"
            )
        positive = {
            "radar_sample_rate_hz": self.radar_sample_rate_hz,
            "rsp_sample_rate_hz": self.rsp_sample_rate_hz,
            "motion_smoothing_s": self.motion_smoothing_s,
            "motion_clip_z": self.motion_clip_z,
            "radar_marker_merge_s": self.radar_marker_merge_s,
            "rsp_marker_merge_s": self.rsp_marker_merge_s,
            "prior_tolerance_s": self.prior_tolerance_s,
            "match_residual_gate_s": self.match_residual_gate_s,
            "max_drift_ppm": self.max_drift_ppm,
            "accept_max_rmse_s": self.accept_max_rmse_s,
            "accept_max_abs_residual_s": self.accept_max_abs_residual_s,
        }
        bad = sorted(name for name, value in positive.items() if float(value) <= 0)
        if bad:
            raise SynchronizationError(f"config fields must be positive: {bad}")
        if not 0.0 < self.motion_range_quantile <= 1.0:
            raise SynchronizationError("motion_range_quantile must be in (0, 1]")
        if not 0.0 <= self.accept_min_confidence <= 1.0:
            raise SynchronizationError("accept_min_confidence must be in [0, 1]")
        if self.min_marker_pairs < 2 or self.min_affine_pairs < 2:
            raise SynchronizationError("marker pair gates must be at least two")
        if self.good_marker_pairs < self.min_marker_pairs:
            raise SynchronizationError("good_marker_pairs cannot be below min_marker_pairs")
        if self.manual_review_min_pairs < 2:
            raise SynchronizationError("manual_review_min_pairs must be at least two")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)


def load_synchronization_config(path: str | Path) -> SynchronizationConfig:
    """Read and validate a YAML synchronization contract."""

    config_path = Path(path)
    try:
        value = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise SynchronizationError(f"cannot read synchronization config: {exc}") from exc
    return SynchronizationConfig.from_mapping(value)


@dataclass(frozen=True, slots=True)
class MarkerCandidate:
    index: int
    time_s: float
    score: float
    source: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": int(self.index),
            "time_s": float(self.time_s),
            "score": float(self.score),
            "source": self.source,
        }


@dataclass(frozen=True, slots=True)
class RadarMotionEnvelope:
    times_s: np.ndarray = field(repr=False, compare=False)
    robust_z: np.ndarray = field(repr=False, compare=False)
    valid_view_count: int
    repaired_nonfinite_values: int


@dataclass(frozen=True, slots=True)
class TimeMapping:
    """Constant or affine mapping from radar-relative to RSP-relative time."""

    mode: Literal["constant", "affine"]
    offset_s: float
    scale: float = 1.0

    def __post_init__(self) -> None:
        if self.mode not in {"constant", "affine"}:
            raise SynchronizationError(f"unknown time mapping mode: {self.mode!r}")
        if not np.isfinite(self.offset_s) or not np.isfinite(self.scale):
            raise SynchronizationError("time mapping parameters must be finite")
        if self.scale <= 0:
            raise SynchronizationError("time mapping scale must be positive")
        if self.mode == "constant" and self.scale != 1.0:
            raise SynchronizationError("constant mappings must have scale exactly 1")

    @property
    def drift_ppm(self) -> float:
        return (float(self.scale) - 1.0) * 1_000_000.0

    def radar_to_rsp(self, radar_time_s: Any) -> np.ndarray:
        return self.offset_s + self.scale * np.asarray(radar_time_s, dtype=np.float64)

    def rsp_to_radar(self, rsp_time_s: Any) -> np.ndarray:
        return (np.asarray(rsp_time_s, dtype=np.float64) - self.offset_s) / self.scale

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "offset_s": float(self.offset_s),
            "scale": float(self.scale),
            "drift_ppm": float(self.drift_ppm),
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "TimeMapping":
        if not isinstance(raw, Mapping):
            raise SynchronizationError("mapping must be a JSON object")
        return cls(
            mode=str(raw.get("mode")),  # type: ignore[arg-type]
            offset_s=float(raw.get("offset_s")),
            scale=float(raw.get("scale")),
        )


@dataclass(frozen=True, slots=True)
class MarkerMatch:
    radar_index: int
    rsp_index: int
    radar_time_s: float
    rsp_time_s: float
    residual_s: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class SynchronizationResult:
    """Automated synchronization proposal and fail-closed decision."""

    decision: Literal["accepted", "manual_review_required", "rejected"]
    reasons: tuple[str, ...]
    mapping: TimeMapping | None
    matches: tuple[MarkerMatch, ...]
    confidence: float
    residual_rmse_s: float | None
    residual_max_abs_s: float | None
    marker_span_s: float
    ambiguous: bool
    prior_offset_s: float
    radar_markers: tuple[MarkerCandidate, ...]
    rsp_markers: tuple[MarkerCandidate, ...]
    diagnostics: Mapping[str, Any] = field(default_factory=dict)

    @property
    def automatically_authorized(self) -> bool:
        return self.decision == "accepted"

    def to_dict(self) -> dict[str, Any]:
        return {
            "decision": self.decision,
            "automatically_authorized": self.automatically_authorized,
            "reasons": list(self.reasons),
            "mapping": None if self.mapping is None else self.mapping.to_dict(),
            "matches": [match.to_dict() for match in self.matches],
            "confidence": float(self.confidence),
            "residual_rmse_s": self.residual_rmse_s,
            "residual_max_abs_s": self.residual_max_abs_s,
            "marker_span_s": float(self.marker_span_s),
            "ambiguous": bool(self.ambiguous),
            "prior_offset_s": float(self.prior_offset_s),
            "radar_markers": [marker.to_dict() for marker in self.radar_markers],
            "rsp_markers": [marker.to_dict() for marker in self.rsp_markers],
            "diagnostics": dict(self.diagnostics),
        }


@dataclass(frozen=True, slots=True)
class AffineSampleIndex:
    """Linear interpolation index for mapping RSP samples onto radar times."""

    lower: np.ndarray = field(repr=False, compare=False)
    upper: np.ndarray = field(repr=False, compare=False)
    upper_weight: np.ndarray = field(repr=False, compare=False)
    valid: np.ndarray = field(repr=False, compare=False)
    mapped_rsp_times_s: np.ndarray = field(repr=False, compare=False)


def epoch_prior_offset_from_starts(
    *, radar_start_epoch_s: float, rsp_start_epoch_s: float
) -> float:
    """Convert absolute start epochs to the module's relative-time offset.

    For the same physical event,
    ``radar_start_epoch_s + radar_time_s == rsp_start_epoch_s + rsp_time_s``;
    consequently the prior offset is radar start minus RSP start.  Keeping the
    sign conversion here prevents a common integration error.
    """

    radar_start = float(radar_start_epoch_s)
    rsp_start = float(rsp_start_epoch_s)
    if not np.isfinite([radar_start, rsp_start]).all():
        raise SynchronizationError("sensor start epochs must be finite")
    return radar_start - rsp_start


def _as_increasing_times(
    values: Sequence[float] | np.ndarray | None,
    count: int,
    sample_rate_hz: float,
    label: str,
) -> np.ndarray:
    if values is None:
        return np.arange(count, dtype=np.float64) / float(sample_rate_hz)
    times = np.asarray(values, dtype=np.float64)
    if times.ndim != 1 or times.size != count:
        raise SynchronizationError(f"{label} must be a length-{count} vector")
    if not np.isfinite(times).all() or (times.size > 1 and np.any(np.diff(times) <= 0)):
        raise SynchronizationError(f"{label} must be finite and strictly increasing")
    return times


def _robust_location_scale(values: np.ndarray) -> tuple[float, float]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return 0.0, 1.0
    center = float(np.median(finite))
    mad = float(np.median(np.abs(finite - center)))
    scale = max(1.4826 * mad, np.finfo(np.float64).eps)
    return center, scale


def robust_radar_motion_envelope(
    radar_payload: np.ndarray,
    *,
    radar_times_s: Sequence[float] | np.ndarray | None = None,
    config: SynchronizationConfig | None = None,
) -> RadarMotionEnvelope:
    """Build a target-free motion envelope from exact 182-float payloads.

    Input is ``[time, 182]`` or ``[radar, time, 182]``.  Values are never
    reinterpreted as I/Q pairs.  Each range cell is robustly normalized from
    frame differences, range cells are combined with a high quantile, and
    views are combined by a median/max blend so one failed/noisy view cannot
    dominate while a marker visible to only one view remains detectable.
    """

    cfg = config or SynchronizationConfig()
    cfg.validate()
    payload = np.asarray(radar_payload)
    if payload.ndim == 2:
        payload = payload[None, ...]
    if payload.ndim != 3 or payload.shape[-1] != cfg.expected_payload_bins:
        raise SynchronizationError(
            "radar_payload must have shape [time,182] or [radar,time,182]; "
            f"observed {payload.shape}"
        )
    if payload.shape[0] < 1 or payload.shape[1] < 3:
        raise SynchronizationError("radar_payload contains too few views or frames")

    frames = np.asarray(payload, dtype=np.float64).copy()
    repaired = int((~np.isfinite(frames)).sum())
    valid_views: list[np.ndarray] = []
    for view in frames:
        finite_fraction = np.isfinite(view).mean()
        if finite_fraction < 0.95:
            continue
        if not np.isfinite(view).all():
            column_median = np.nanmedian(np.where(np.isfinite(view), view, np.nan), axis=0)
            column_median = np.where(np.isfinite(column_median), column_median, 0.0)
            bad_row, bad_col = np.nonzero(~np.isfinite(view))
            view[bad_row, bad_col] = column_median[bad_col]

        delta = np.abs(np.diff(view, axis=0, prepend=view[[0]]))
        per_bin_center = np.median(delta, axis=0)
        per_bin_mad = np.median(np.abs(delta - per_bin_center[None, :]), axis=0)
        per_bin_scale = 1.4826 * per_bin_mad
        positive = per_bin_scale[per_bin_scale > np.finfo(np.float64).eps]
        scale_floor = (
            max(float(np.quantile(positive, 0.10)) * 0.10, np.finfo(np.float64).eps)
            if positive.size
            else 1.0
        )
        standardized = np.maximum(delta - per_bin_center[None, :], 0.0) / np.maximum(
            per_bin_scale[None, :], scale_floor
        )
        standardized = np.clip(standardized, 0.0, cfg.motion_clip_z)
        valid_views.append(
            np.quantile(standardized, cfg.motion_range_quantile, axis=1)
        )

    if not valid_views:
        raise SynchronizationError("no radar view has at least 95% finite payload values")
    per_view = np.stack(valid_views, axis=0)
    if per_view.shape[0] == 1:
        combined = per_view[0]
    else:
        combined = 0.5 * np.median(per_view, axis=0) + 0.5 * np.max(per_view, axis=0)
    smoothing_frames = max(1, int(round(cfg.motion_smoothing_s * cfg.radar_sample_rate_hz)))
    smoothed = uniform_filter1d(combined, size=smoothing_frames, mode="nearest")
    center, scale = _robust_location_scale(smoothed)
    robust_z = np.maximum((smoothed - center) / scale, 0.0)
    times = _as_increasing_times(
        radar_times_s,
        payload.shape[1],
        cfg.radar_sample_rate_hz,
        "radar_times_s",
    )
    return RadarMotionEnvelope(
        times_s=times,
        robust_z=robust_z,
        valid_view_count=len(valid_views),
        repaired_nonfinite_values=repaired,
    )


def _merge_marker_indices(
    indices_with_sources: Sequence[tuple[int, str]],
    scores: np.ndarray,
    times_s: np.ndarray,
    merge_s: float,
) -> tuple[MarkerCandidate, ...]:
    if not indices_with_sources:
        return ()
    ordered = sorted(indices_with_sources, key=lambda item: item[0])
    groups: list[list[tuple[int, str]]] = [[ordered[0]]]
    for item in ordered[1:]:
        if float(times_s[item[0]] - times_s[groups[-1][-1][0]]) <= merge_s:
            groups[-1].append(item)
        else:
            groups.append([item])
    result: list[MarkerCandidate] = []
    for group in groups:
        candidates = sorted({index for index, _ in group})
        chosen = max(candidates, key=lambda index: (float(scores[index]), -index))
        sources = sorted({source for _, source in group})
        result.append(
            MarkerCandidate(
                index=int(chosen),
                time_s=float(times_s[chosen]),
                score=float(scores[chosen]),
                source="+".join(sources),
            )
        )
    return tuple(result)


def detect_radar_marker_candidates(
    envelope: RadarMotionEnvelope,
    *,
    config: SynchronizationConfig | None = None,
) -> tuple[MarkerCandidate, ...]:
    cfg = config or SynchronizationConfig()
    cfg.validate()
    z = np.asarray(envelope.robust_z, dtype=np.float64)
    if z.size != envelope.times_s.size or z.ndim != 1:
        raise SynchronizationError("radar motion envelope vectors are inconsistent")
    dt = float(np.median(np.diff(envelope.times_s)))
    distance = max(1, int(round(cfg.radar_marker_merge_s / dt)))
    peaks, _ = find_peaks(
        z,
        height=cfg.radar_marker_z,
        prominence=cfg.radar_marker_prominence_z,
        distance=distance,
    )
    return _merge_marker_indices(
        [(int(index), "motion") for index in peaks],
        z,
        envelope.times_s,
        cfg.radar_marker_merge_s,
    )


def _threshold_region_peaks(values: np.ndarray, mask: np.ndarray) -> list[int]:
    padded = np.pad(mask.astype(np.int8), (1, 1))
    starts = np.flatnonzero(np.diff(padded) == 1)
    stops = np.flatnonzero(np.diff(padded) == -1)
    result: list[int] = []
    for start, stop in zip(starts, stops, strict=True):
        if stop <= start:
            continue
        result.append(int(start + np.argmax(values[start:stop])))
    return result


def detect_rsp_marker_candidates(
    rsp_values: Sequence[float] | np.ndarray,
    *,
    rsp_times_s: Sequence[float] | np.ndarray | None = None,
    config: SynchronizationConfig | None = None,
) -> tuple[MarkerCandidate, ...]:
    """Detect chest-press candidates with adaptive and fixed-voltage gates."""

    cfg = config or SynchronizationConfig()
    cfg.validate()
    rsp = np.asarray(rsp_values, dtype=np.float64)
    if rsp.ndim != 1 or rsp.size < 3:
        raise SynchronizationError("rsp_values must be a one-dimensional signal")
    if not np.isfinite(rsp).all():
        raise SynchronizationError("rsp_values contains non-finite samples")
    times = _as_increasing_times(
        rsp_times_s, rsp.size, cfg.rsp_sample_rate_hz, "rsp_times_s"
    )
    center, scale = _robust_location_scale(rsp)
    high_z = (rsp - center) / scale
    low_z = (center - rsp) / scale
    score = np.maximum(high_z, low_z)
    raw: list[tuple[int, str]] = []

    for index in _threshold_region_peaks(rsp, high_z >= cfg.rsp_adaptive_z):
        raw.append((index, "adaptive_high"))
    for index in _threshold_region_peaks(-rsp, low_z >= cfg.rsp_adaptive_z):
        raw.append((index, "adaptive_low"))
    if cfg.rsp_fixed_high is not None:
        for index in _threshold_region_peaks(rsp, rsp >= cfg.rsp_fixed_high):
            raw.append((index, "fixed_high"))
    if cfg.rsp_fixed_low is not None:
        for index in _threshold_region_peaks(-rsp, rsp <= cfg.rsp_fixed_low):
            raw.append((index, "fixed_low"))

    return _merge_marker_indices(raw, score, times, cfg.rsp_marker_merge_s)


@dataclass(frozen=True, slots=True)
class _CandidateSolution:
    mapping: TimeMapping
    pairs: tuple[tuple[int, int], ...]
    residuals: np.ndarray = field(repr=False, compare=False)
    rmse_s: float
    max_abs_s: float
    score: float


def _monotonic_match(
    radar: Sequence[MarkerCandidate],
    rsp: Sequence[MarkerCandidate],
    mapping: TimeMapping,
    residual_gate_s: float,
) -> tuple[tuple[int, int], ...]:
    """Maximum-cardinality, minimum-residual ordered one-to-one matching."""

    nr, ns = len(radar), len(rsp)
    # Lexicographic objective represented as (pair count, quality).
    counts = np.zeros((nr + 1, ns + 1), dtype=np.int32)
    quality = np.zeros((nr + 1, ns + 1), dtype=np.float64)
    action = np.zeros((nr + 1, ns + 1), dtype=np.int8)
    predicted = mapping.radar_to_rsp([candidate.time_s for candidate in radar])

    for i in range(1, nr + 1):
        for j in range(1, ns + 1):
            options: list[tuple[int, float, int]] = [
                (int(counts[i - 1, j]), float(quality[i - 1, j]), 1),
                (int(counts[i, j - 1]), float(quality[i, j - 1]), 2),
            ]
            residual = float(rsp[j - 1].time_s - predicted[i - 1])
            if abs(residual) <= residual_gate_s:
                strength = min(radar[i - 1].score, 20.0) + min(rsp[j - 1].score, 20.0)
                pair_quality = 0.01 * strength - abs(residual) / residual_gate_s
                options.append(
                    (
                        int(counts[i - 1, j - 1]) + 1,
                        float(quality[i - 1, j - 1]) + pair_quality,
                        3,
                    )
                )
            best = max(options, key=lambda item: (item[0], item[1], item[2] == 3))
            counts[i, j], quality[i, j], action[i, j] = best

    pairs: list[tuple[int, int]] = []
    i, j = nr, ns
    while i > 0 and j > 0:
        code = int(action[i, j])
        if code == 3:
            pairs.append((i - 1, j - 1))
            i -= 1
            j -= 1
        elif code == 1:
            i -= 1
        else:
            j -= 1
    pairs.reverse()
    return tuple(pairs)


def _fit_constant(
    radar: Sequence[MarkerCandidate],
    rsp: Sequence[MarkerCandidate],
    pairs: Sequence[tuple[int, int]],
) -> TimeMapping:
    offsets = np.asarray(
        [rsp[j].time_s - radar[i].time_s for i, j in pairs], dtype=np.float64
    )
    return TimeMapping(mode="constant", offset_s=float(np.median(offsets)), scale=1.0)


def _fit_affine_irls(
    radar: Sequence[MarkerCandidate],
    rsp: Sequence[MarkerCandidate],
    pairs: Sequence[tuple[int, int]],
) -> TimeMapping:
    x = np.asarray([radar[i].time_s for i, _ in pairs], dtype=np.float64)
    y = np.asarray([rsp[j].time_s for _, j in pairs], dtype=np.float64)
    design = np.column_stack((np.ones_like(x), x))
    weights = np.ones_like(x)
    beta = np.linalg.lstsq(design, y, rcond=None)[0]
    for _ in range(20):
        residual = y - design @ beta
        _, scale = _robust_location_scale(residual)
        cutoff = max(1.345 * scale, 1e-6)
        new_weights = np.minimum(1.0, cutoff / np.maximum(np.abs(residual), 1e-12))
        weighted_design = design * np.sqrt(new_weights)[:, None]
        weighted_y = y * np.sqrt(new_weights)
        updated = np.linalg.lstsq(weighted_design, weighted_y, rcond=None)[0]
        if np.max(np.abs(updated - beta)) < 1e-12:
            beta = updated
            break
        beta = updated
        weights = new_weights
    del weights
    return TimeMapping(mode="affine", offset_s=float(beta[0]), scale=float(beta[1]))


def _evaluate_solution(
    radar: Sequence[MarkerCandidate],
    rsp: Sequence[MarkerCandidate],
    mapping: TimeMapping,
    pairs: Sequence[tuple[int, int]],
    prior_offset_s: float,
    cfg: SynchronizationConfig,
) -> _CandidateSolution:
    residuals = np.asarray(
        [
            rsp[j].time_s
            - float(mapping.offset_s + mapping.scale * radar[i].time_s)
            for i, j in pairs
        ],
        dtype=np.float64,
    )
    if residuals.size:
        rmse = float(np.sqrt(np.mean(np.square(residuals))))
        max_abs = float(np.max(np.abs(residuals)))
        strength = float(
            np.mean(
                [
                    min(radar[i].score, 20.0) + min(rsp[j].score, 20.0)
                    for i, j in pairs
                ]
            )
        )
    else:
        rmse = float("inf")
        max_abs = float("inf")
        strength = 0.0
    prior_penalty = abs(mapping.offset_s - prior_offset_s) / cfg.prior_tolerance_s
    score = 100.0 * len(pairs) - 10.0 * rmse - prior_penalty + 0.01 * strength
    return _CandidateSolution(
        mapping=mapping,
        pairs=tuple(pairs),
        residuals=residuals,
        rmse_s=rmse,
        max_abs_s=max_abs,
        score=float(score),
    )


def _refine_hypothesis(
    radar: Sequence[MarkerCandidate],
    rsp: Sequence[MarkerCandidate],
    initial: TimeMapping,
    prior_offset_s: float,
    cfg: SynchronizationConfig,
) -> _CandidateSolution | None:
    mapping = initial
    pairs: tuple[tuple[int, int], ...] = ()
    for _ in range(4):
        updated_pairs = _monotonic_match(
            radar, rsp, mapping, cfg.match_residual_gate_s
        )
        if not updated_pairs:
            return None
        if mapping.mode == "affine" and len(updated_pairs) >= 2:
            fitted = _fit_affine_irls(radar, rsp, updated_pairs)
            if abs(fitted.drift_ppm) > cfg.max_drift_ppm:
                return None
            mapping = fitted
        else:
            mapping = _fit_constant(radar, rsp, updated_pairs)
        if updated_pairs == pairs:
            break
        pairs = updated_pairs
    final_pairs = _monotonic_match(radar, rsp, mapping, cfg.match_residual_gate_s)
    return _evaluate_solution(radar, rsp, mapping, final_pairs, prior_offset_s, cfg)


def _initial_hypotheses(
    radar: Sequence[MarkerCandidate],
    rsp: Sequence[MarkerCandidate],
    prior_offset_s: float,
    cfg: SynchronizationConfig,
) -> list[TimeMapping]:
    pairings: list[tuple[int, int, float]] = []
    for i, radar_marker in enumerate(radar):
        for j, rsp_marker in enumerate(rsp):
            offset = rsp_marker.time_s - radar_marker.time_s
            if abs(offset - prior_offset_s) <= cfg.prior_tolerance_s:
                pairings.append((i, j, offset))
    hypotheses: list[TimeMapping] = [
        TimeMapping(mode="constant", offset_s=float(prior_offset_s), scale=1.0)
    ]
    hypotheses.extend(
        TimeMapping(mode="constant", offset_s=float(offset), scale=1.0)
        for _, _, offset in pairings
    )

    # Pair-of-pairs hypotheses make the matching robust to missing/false
    # markers and expose small recorder drift before the final IRLS fit.
    max_pairings = 160
    if len(pairings) <= max_pairings:
        for left_index, (i1, j1, _) in enumerate(pairings):
            for i2, j2, _ in pairings[left_index + 1 :]:
                if i2 <= i1 or j2 <= j1:
                    continue
                radar_span = radar[i2].time_s - radar[i1].time_s
                if radar_span < cfg.min_affine_span_s:
                    continue
                scale = (rsp[j2].time_s - rsp[j1].time_s) / radar_span
                offset = rsp[j1].time_s - scale * radar[i1].time_s
                mapping = TimeMapping(mode="affine", offset_s=float(offset), scale=float(scale))
                if (
                    abs(mapping.drift_ppm) <= cfg.max_drift_ppm
                    and abs(mapping.offset_s - prior_offset_s) <= cfg.prior_tolerance_s
                ):
                    hypotheses.append(mapping)

    # Canonical de-duplication keeps complexity bounded without changing the
    # meaningful 0.1-ms / 0.1-ppm resolution of a proposal.
    unique: dict[tuple[str, int, int], TimeMapping] = {}
    for mapping in hypotheses:
        key = (
            mapping.mode,
            int(round(mapping.offset_s * 10_000)),
            int(round(mapping.drift_ppm * 10)),
        )
        unique[key] = mapping
    return list(unique.values())


def estimate_marker_time_mapping(
    radar_markers: Sequence[MarkerCandidate],
    rsp_markers: Sequence[MarkerCandidate],
    *,
    epoch_prior_offset_s: float,
    config: SynchronizationConfig | None = None,
) -> SynchronizationResult:
    """Match ordered markers and fit a gated constant/affine clock relation."""

    cfg = config or SynchronizationConfig()
    cfg.validate()
    if not np.isfinite(epoch_prior_offset_s):
        raise SynchronizationError("epoch_prior_offset_s must be finite")
    radar = tuple(sorted(radar_markers, key=lambda marker: marker.time_s))
    rsp = tuple(sorted(rsp_markers, key=lambda marker: marker.time_s))
    if any(b.time_s <= a.time_s for a, b in zip(radar, radar[1:], strict=False)):
        raise SynchronizationError("radar marker times must be unique")
    if any(b.time_s <= a.time_s for a, b in zip(rsp, rsp[1:], strict=False)):
        raise SynchronizationError("RSP marker times must be unique")

    hypotheses = _initial_hypotheses(radar, rsp, epoch_prior_offset_s, cfg)
    candidates = [
        solution
        for hypothesis in hypotheses
        if (
            solution := _refine_hypothesis(
                radar, rsp, hypothesis, epoch_prior_offset_s, cfg
            )
        )
        is not None
    ]
    if not candidates:
        return SynchronizationResult(
            decision="rejected",
            reasons=("no_monotonic_marker_mapping",),
            mapping=None,
            matches=(),
            confidence=0.0,
            residual_rmse_s=None,
            residual_max_abs_s=None,
            marker_span_s=0.0,
            ambiguous=False,
            prior_offset_s=float(epoch_prior_offset_s),
            radar_markers=radar,
            rsp_markers=rsp,
            diagnostics={"hypothesis_count": len(hypotheses)},
        )

    # Collapse solutions that converge to the same effective mapping.
    deduplicated: dict[tuple[str, int, int, tuple[tuple[int, int], ...]], _CandidateSolution] = {}
    for solution in candidates:
        key = (
            solution.mapping.mode,
            int(round(solution.mapping.offset_s * 10_000)),
            int(round(solution.mapping.drift_ppm * 10)),
            solution.pairs,
        )
        previous = deduplicated.get(key)
        if previous is None or solution.score > previous.score:
            deduplicated[key] = solution
    candidates = sorted(deduplicated.values(), key=lambda item: item.score, reverse=True)

    constants = [candidate for candidate in candidates if candidate.mapping.mode == "constant"]
    affines = [candidate for candidate in candidates if candidate.mapping.mode == "affine"]
    best_constant = constants[0] if constants else None
    best_affine = affines[0] if affines else None
    best = candidates[0]
    if best_affine is not None and best_constant is not None:
        pair_times = [radar[i].time_s for i, _ in best_affine.pairs]
        affine_span = max(pair_times) - min(pair_times) if pair_times else 0.0
        improvement = best_constant.rmse_s - best_affine.rmse_s
        affine_eligible = (
            len(best_affine.pairs) >= cfg.min_affine_pairs
            and affine_span >= cfg.min_affine_span_s
            and improvement >= cfg.min_affine_improvement_s
            and abs(best_affine.mapping.drift_ppm) >= cfg.min_affine_drift_ppm
        )
        best = best_affine if affine_eligible else best_constant

    matched_radar_times = [radar[i].time_s for i, _ in best.pairs]
    marker_span = (
        float(max(matched_radar_times) - min(matched_radar_times))
        if len(matched_radar_times) >= 2
        else 0.0
    )

    alternative: _CandidateSolution | None = None
    reference_times = np.asarray(matched_radar_times or [0.0], dtype=np.float64)
    best_projection = best.mapping.radar_to_rsp(reference_times)
    for candidate in candidates:
        if candidate is best:
            continue
        separation = float(
            np.max(np.abs(candidate.mapping.radar_to_rsp(reference_times) - best_projection))
        )
        if separation >= cfg.ambiguity_mapping_separation_s:
            alternative = candidate
            break
    ambiguous = False
    if alternative is not None:
        same_count = len(alternative.pairs) == len(best.pairs)
        close_rmse = alternative.rmse_s <= best.rmse_s + cfg.ambiguity_rmse_margin_s
        close_score = alternative.score >= best.score - cfg.ambiguity_score_margin
        ambiguous = same_count and close_rmse and close_score

    pair_score = min(1.0, len(best.pairs) / float(cfg.good_marker_pairs))
    span_score = min(1.0, marker_span / max(cfg.min_marker_span_s, 1e-12))
    residual_score = max(0.0, 1.0 - best.rmse_s / cfg.accept_max_rmse_s)
    prior_score = max(
        0.0,
        1.0 - abs(best.mapping.offset_s - epoch_prior_offset_s) / cfg.prior_tolerance_s,
    )
    fixed_fraction = (
        float(
            np.mean(
                ["fixed" in rsp[j].source for _, j in best.pairs], dtype=np.float64
            )
        )
        if best.pairs
        else 0.0
    )
    confidence = float(
        np.clip(
            0.30 * pair_score
            + 0.20 * span_score
            + 0.30 * residual_score
            + 0.15 * prior_score
            + 0.05 * fixed_fraction,
            0.0,
            1.0,
        )
    )
    if ambiguous:
        confidence = 0.0

    reasons: list[str] = []
    if len(best.pairs) < cfg.min_marker_pairs:
        reasons.append("insufficient_marker_pairs")
    if marker_span < cfg.min_marker_span_s:
        reasons.append("insufficient_marker_span")
    if best.rmse_s > cfg.accept_max_rmse_s:
        reasons.append("residual_rmse_gate_failed")
    if best.max_abs_s > cfg.accept_max_abs_residual_s:
        reasons.append("maximum_residual_gate_failed")
    if abs(best.mapping.offset_s - epoch_prior_offset_s) > cfg.prior_tolerance_s:
        reasons.append("epoch_prior_gate_failed")
    if abs(best.mapping.drift_ppm) > cfg.max_drift_ppm:
        reasons.append("drift_gate_failed")
    if ambiguous:
        reasons.append("ambiguous_marker_mapping")
    if confidence < cfg.accept_min_confidence:
        reasons.append("confidence_gate_failed")

    if not reasons:
        decision: Literal["accepted", "manual_review_required", "rejected"] = "accepted"
    elif len(best.pairs) >= cfg.manual_review_min_pairs:
        decision = "manual_review_required"
    else:
        decision = "rejected"

    matches = tuple(
        MarkerMatch(
            radar_index=i,
            rsp_index=j,
            radar_time_s=float(radar[i].time_s),
            rsp_time_s=float(rsp[j].time_s),
            residual_s=float(
                rsp[j].time_s
                - (best.mapping.offset_s + best.mapping.scale * radar[i].time_s)
            ),
        )
        for i, j in best.pairs
    )
    diagnostics: dict[str, Any] = {
        "hypothesis_count": len(hypotheses),
        "distinct_solution_count": len(candidates),
        "fixed_rsp_match_fraction": fixed_fraction,
        "best_constant_rmse_s": None if best_constant is None else best_constant.rmse_s,
        "best_affine_rmse_s": None if best_affine is None else best_affine.rmse_s,
        "alternative_mapping": None if alternative is None else alternative.mapping.to_dict(),
        "alternative_score": None if alternative is None else alternative.score,
    }
    return SynchronizationResult(
        decision=decision,
        reasons=tuple(reasons),
        mapping=best.mapping,
        matches=matches,
        confidence=confidence,
        residual_rmse_s=best.rmse_s,
        residual_max_abs_s=best.max_abs_s,
        marker_span_s=marker_span,
        ambiguous=ambiguous,
        prior_offset_s=float(epoch_prior_offset_s),
        radar_markers=radar,
        rsp_markers=rsp,
        diagnostics=diagnostics,
    )


def synchronize_from_signals(
    radar_payload: np.ndarray,
    rsp_values: Sequence[float] | np.ndarray,
    *,
    epoch_prior_offset_s: float,
    radar_times_s: Sequence[float] | np.ndarray | None = None,
    rsp_times_s: Sequence[float] | np.ndarray | None = None,
    config: SynchronizationConfig | None = None,
) -> SynchronizationResult:
    """End-to-end target-free radar envelope and marker mapping proposal."""

    cfg = config or SynchronizationConfig()
    envelope = robust_radar_motion_envelope(
        radar_payload, radar_times_s=radar_times_s, config=cfg
    )
    radar_markers = detect_radar_marker_candidates(envelope, config=cfg)
    rsp_markers = detect_rsp_marker_candidates(
        rsp_values, rsp_times_s=rsp_times_s, config=cfg
    )
    result = estimate_marker_time_mapping(
        radar_markers,
        rsp_markers,
        epoch_prior_offset_s=epoch_prior_offset_s,
        config=cfg,
    )
    diagnostics = dict(result.diagnostics)
    diagnostics.update(
        {
            "valid_radar_view_count": envelope.valid_view_count,
            "radar_nonfinite_values_repaired": envelope.repaired_nonfinite_values,
            "radar_envelope_target_free": True,
            "radar_payload_interpretation": "182_real_float_payload_values",
        }
    )
    return SynchronizationResult(
        decision=result.decision,
        reasons=result.reasons,
        mapping=result.mapping,
        matches=result.matches,
        confidence=result.confidence,
        residual_rmse_s=result.residual_rmse_s,
        residual_max_abs_s=result.residual_max_abs_s,
        marker_span_s=result.marker_span_s,
        ambiguous=result.ambiguous,
        prior_offset_s=result.prior_offset_s,
        radar_markers=result.radar_markers,
        rsp_markers=result.rsp_markers,
        diagnostics=diagnostics,
    )


def build_affine_sample_index(
    mapping: TimeMapping,
    radar_times_s: Sequence[float] | np.ndarray,
    rsp_times_s: Sequence[float] | np.ndarray,
) -> AffineSampleIndex:
    """Create exact bracketing indices for affine RSP-to-radar resampling."""

    radar_times = np.asarray(radar_times_s, dtype=np.float64)
    rsp_times = np.asarray(rsp_times_s, dtype=np.float64)
    if radar_times.ndim != 1 or not np.isfinite(radar_times).all():
        raise SynchronizationError("radar_times_s must be a finite vector")
    if (
        rsp_times.ndim != 1
        or rsp_times.size < 2
        or not np.isfinite(rsp_times).all()
        or np.any(np.diff(rsp_times) <= 0)
    ):
        raise SynchronizationError("rsp_times_s must be a finite, increasing vector")
    mapped = mapping.radar_to_rsp(radar_times)
    upper = np.searchsorted(rsp_times, mapped, side="right")
    lower = upper - 1
    valid = (mapped >= rsp_times[0]) & (mapped <= rsp_times[-1])
    exact_last = valid & (upper == rsp_times.size)
    upper = np.where(exact_last, rsp_times.size - 1, upper)
    lower = np.where(exact_last, rsp_times.size - 1, lower)
    safe_lower = np.clip(lower, 0, rsp_times.size - 1)
    safe_upper = np.clip(upper, 0, rsp_times.size - 1)
    denominator = rsp_times[safe_upper] - rsp_times[safe_lower]
    upper_weight = np.divide(
        mapped - rsp_times[safe_lower],
        denominator,
        out=np.zeros_like(mapped),
        where=denominator > 0,
    )
    lower = np.where(valid, safe_lower, -1).astype(np.int64)
    upper = np.where(valid, safe_upper, -1).astype(np.int64)
    upper_weight = np.where(valid, np.clip(upper_weight, 0.0, 1.0), 0.0)
    return AffineSampleIndex(
        lower=lower,
        upper=upper,
        upper_weight=upper_weight,
        valid=valid,
        mapped_rsp_times_s=mapped,
    )


def apply_affine_sample_index(
    rsp_values: Sequence[float] | np.ndarray,
    index: AffineSampleIndex,
    *,
    fill_value: float = float("nan"),
) -> np.ndarray:
    """Linearly resample RSP values using a prevalidated affine index."""

    values = np.asarray(rsp_values, dtype=np.float64)
    if values.ndim != 1:
        raise SynchronizationError("rsp_values must be a vector")
    valid_positions = np.flatnonzero(index.valid)
    if valid_positions.size:
        maximum = int(
            max(index.lower[valid_positions].max(), index.upper[valid_positions].max())
        )
        if maximum >= values.size:
            raise SynchronizationError("affine sample index exceeds rsp_values")
    output = np.full(index.valid.shape, fill_value, dtype=np.float64)
    lower = index.lower[valid_positions]
    upper = index.upper[valid_positions]
    weight = index.upper_weight[valid_positions]
    output[valid_positions] = (1.0 - weight) * values[lower] + weight * values[upper]
    return output


def _json_compatible(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_compatible(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_compatible(item) for item in value]
    return value


def canonical_json_bytes(value: Any) -> bytes:
    """Serialize JSON in the single form used by receipts and approvals."""

    try:
        return json.dumps(
            _json_compatible(value),
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise SynchronizationError(f"value is not canonical JSON: {exc}") from exc


def canonical_content_sha256(document: Mapping[str, Any]) -> str:
    payload = dict(document)
    payload.pop("content_sha256", None)
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def _require_sha256(value: Any, label: str) -> str:
    digest = str(value)
    if len(digest) != 64 or digest.lower() != digest or any(
        character not in "0123456789abcdef" for character in digest
    ):
        raise SynchronizationError(f"{label} must be a lowercase SHA-256")
    return digest


def _utc_timestamp(value: str | None) -> str:
    if value is None:
        return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    if not isinstance(value, str) or not value.endswith("Z"):
        raise SynchronizationError("timestamp must be an explicit UTC ISO-8601 value ending Z")
    try:
        datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as exc:
        raise SynchronizationError("timestamp is not valid ISO-8601") from exc
    return value


def build_sync_receipt(
    result: SynchronizationResult,
    *,
    session_id: str,
    config: SynchronizationConfig,
    input_bindings: Mapping[str, Mapping[str, Any]],
    created_at_utc: str | None = None,
) -> dict[str, Any]:
    """Create a canonical, self-hashed proposal receipt.

    ``input_bindings`` must contain content hashes (and may additionally carry
    byte counts or non-sensitive relative identifiers).  Raw signal samples
    are intentionally never serialized into the receipt.
    """

    if not isinstance(session_id, str) or not session_id.strip() or session_id != session_id.strip():
        raise SynchronizationError("session_id must be a non-empty trimmed string")
    if not isinstance(input_bindings, Mapping) or not input_bindings:
        raise SynchronizationError("input_bindings must be a non-empty mapping")
    bindings: dict[str, dict[str, Any]] = {}
    for name, binding in sorted(input_bindings.items()):
        if not isinstance(name, str) or not name or not isinstance(binding, Mapping):
            raise SynchronizationError("input bindings must have string names and objects")
        item = dict(_json_compatible(binding))
        _require_sha256(item.get("sha256"), f"input_bindings.{name}.sha256")
        bindings[name] = item
    document: dict[str, Any] = {
        "schema": SYNC_RECEIPT_SCHEMA,
        "session_id": session_id,
        "created_at_utc": _utc_timestamp(created_at_utc),
        "algorithm": {
            "radar_payload_bins": EXPECTED_RADAR_PAYLOAD_BINS,
            "radar_motion_target_free": True,
            "time_mapping_equation": "rsp_time_s=offset_s+scale*radar_time_s",
            "config": config.to_dict(),
            "config_sha256": hashlib.sha256(canonical_json_bytes(config.to_dict())).hexdigest(),
        },
        "input_bindings": bindings,
        "result": result.to_dict(),
    }
    document["content_sha256"] = canonical_content_sha256(document)
    return document


def validate_sync_receipt(
    receipt: Mapping[str, Any],
    *,
    expected_session_id: str | None = None,
) -> dict[str, Any]:
    """Validate exact receipt bytes semantically and fail on any tampering."""

    if not isinstance(receipt, Mapping):
        raise SynchronizationError("sync receipt must be a JSON object")
    document = dict(_json_compatible(receipt))
    if document.get("schema") != SYNC_RECEIPT_SCHEMA:
        raise SynchronizationError("unexpected sync receipt schema")
    digest = _require_sha256(document.get("content_sha256"), "content_sha256")
    if canonical_content_sha256(document) != digest:
        raise SynchronizationError("sync receipt content SHA-256 mismatch")
    session_id = document.get("session_id")
    if not isinstance(session_id, str) or not session_id:
        raise SynchronizationError("sync receipt has invalid session_id")
    if expected_session_id is not None and session_id != expected_session_id:
        raise SynchronizationError("sync receipt session_id mismatch")
    _utc_timestamp(document.get("created_at_utc"))
    algorithm = document.get("algorithm")
    if not isinstance(algorithm, Mapping):
        raise SynchronizationError("sync receipt algorithm must be an object")
    if algorithm.get("radar_payload_bins") != EXPECTED_RADAR_PAYLOAD_BINS:
        raise SynchronizationError("sync receipt is not bound to 182 payload floats")
    if algorithm.get("radar_motion_target_free") is not True:
        raise SynchronizationError("sync receipt lacks target-free radar declaration")
    config = algorithm.get("config")
    if not isinstance(config, Mapping):
        raise SynchronizationError("sync receipt config must be an object")
    config_object = SynchronizationConfig.from_mapping(config)
    config_hash = _require_sha256(algorithm.get("config_sha256"), "config_sha256")
    if hashlib.sha256(canonical_json_bytes(config)).hexdigest() != config_hash:
        raise SynchronizationError("sync receipt config SHA-256 mismatch")
    bindings = document.get("input_bindings")
    if not isinstance(bindings, Mapping) or not bindings:
        raise SynchronizationError("sync receipt lacks input bindings")
    for name, binding in bindings.items():
        if not isinstance(binding, Mapping):
            raise SynchronizationError(f"input binding {name!r} must be an object")
        _require_sha256(binding.get("sha256"), f"input_bindings.{name}.sha256")
    result = document.get("result")
    if not isinstance(result, Mapping):
        raise SynchronizationError("sync receipt result must be an object")
    decision = result.get("decision")
    if decision not in {"accepted", "manual_review_required", "rejected"}:
        raise SynchronizationError("sync receipt decision is invalid")
    if bool(result.get("automatically_authorized")) != (decision == "accepted"):
        raise SynchronizationError("sync receipt authorization/decision mismatch")
    confidence = result.get("confidence")
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
        raise SynchronizationError("sync receipt confidence must be numeric")
    confidence = float(confidence)
    if not np.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
        raise SynchronizationError("sync receipt confidence must be finite and in [0, 1]")
    reasons = result.get("reasons")
    if not isinstance(reasons, list) or any(
        not isinstance(reason, str) or not reason for reason in reasons
    ):
        raise SynchronizationError("sync receipt reasons must be an array of strings")
    if decision == "accepted" and reasons:
        raise SynchronizationError("accepted sync receipt cannot contain failure reasons")
    if decision != "accepted" and not reasons:
        raise SynchronizationError("non-accepted sync receipt must contain failure reasons")
    if decision == "accepted" and confidence < config_object.accept_min_confidence:
        raise SynchronizationError("accepted sync receipt violates confidence gate")
    ambiguous = result.get("ambiguous")
    if not isinstance(ambiguous, bool):
        raise SynchronizationError("sync receipt ambiguous must be boolean")
    if decision == "accepted" and ambiguous:
        raise SynchronizationError("ambiguous sync receipt cannot be accepted")

    mapping_raw = result.get("mapping")
    mapping: TimeMapping | None = None
    if mapping_raw is not None:
        mapping = TimeMapping.from_dict(mapping_raw)
        stated_drift = float(mapping_raw.get("drift_ppm"))
        if not np.isclose(stated_drift, mapping.drift_ppm, rtol=0.0, atol=1e-9):
            raise SynchronizationError("sync receipt mapping drift is inconsistent")
    elif decision != "rejected":
        raise SynchronizationError("accepted/review sync receipt must contain a mapping")

    matches = result.get("matches")
    if not isinstance(matches, list):
        raise SynchronizationError("sync receipt matches must be an array")
    residuals: list[float] = []
    radar_match_times: list[float] = []
    previous_pair = (-1, -1)
    for index, match in enumerate(matches):
        if not isinstance(match, Mapping):
            raise SynchronizationError(f"sync receipt match {index} must be an object")
        try:
            radar_index = int(match["radar_index"])
            rsp_index = int(match["rsp_index"])
            radar_time = float(match["radar_time_s"])
            rsp_time = float(match["rsp_time_s"])
            residual = float(match["residual_s"])
        except (KeyError, TypeError, ValueError) as exc:
            raise SynchronizationError(f"sync receipt match {index} is invalid") from exc
        if (
            radar_index < 0
            or rsp_index < 0
            or radar_index <= previous_pair[0]
            or rsp_index <= previous_pair[1]
        ):
            raise SynchronizationError("sync receipt matches are not strictly monotonic")
        if not np.isfinite([radar_time, rsp_time, residual]).all():
            raise SynchronizationError("sync receipt match values must be finite")
        if mapping is not None:
            expected_residual = rsp_time - (
                mapping.offset_s + mapping.scale * radar_time
            )
            if not np.isclose(residual, expected_residual, rtol=0.0, atol=1e-9):
                raise SynchronizationError("sync receipt match residual is inconsistent")
        residuals.append(residual)
        radar_match_times.append(radar_time)
        previous_pair = (radar_index, rsp_index)

    reported_rmse = result.get("residual_rmse_s")
    reported_max = result.get("residual_max_abs_s")
    if residuals:
        expected_rmse = float(np.sqrt(np.mean(np.square(residuals))))
        expected_max = float(np.max(np.abs(residuals)))
        if reported_rmse is None or not np.isclose(
            float(reported_rmse), expected_rmse, rtol=0.0, atol=1e-9
        ):
            raise SynchronizationError("sync receipt residual RMSE is inconsistent")
        if reported_max is None or not np.isclose(
            float(reported_max), expected_max, rtol=0.0, atol=1e-9
        ):
            raise SynchronizationError("sync receipt maximum residual is inconsistent")
        expected_span = (
            max(radar_match_times) - min(radar_match_times)
            if len(radar_match_times) >= 2
            else 0.0
        )
        if not np.isclose(
            float(result.get("marker_span_s")), expected_span, rtol=0.0, atol=1e-9
        ):
            raise SynchronizationError("sync receipt marker span is inconsistent")
    elif reported_rmse is not None or reported_max is not None:
        raise SynchronizationError("sync receipt reports residuals without matches")

    if decision == "accepted":
        if len(matches) < config_object.min_marker_pairs:
            raise SynchronizationError("accepted sync receipt violates marker pair gate")
        if float(result.get("marker_span_s")) < config_object.min_marker_span_s:
            raise SynchronizationError("accepted sync receipt violates marker span gate")
        if float(reported_rmse) > config_object.accept_max_rmse_s:
            raise SynchronizationError("accepted sync receipt violates residual RMSE gate")
        if float(reported_max) > config_object.accept_max_abs_residual_s:
            raise SynchronizationError("accepted sync receipt violates maximum residual gate")

    for marker_label in ("radar_markers", "rsp_markers"):
        markers = result.get(marker_label)
        if not isinstance(markers, list):
            raise SynchronizationError(f"sync receipt {marker_label} must be an array")
        previous_time = -float("inf")
        for marker in markers:
            if not isinstance(marker, Mapping):
                raise SynchronizationError(f"sync receipt {marker_label} entries must be objects")
            try:
                marker_index = int(marker["index"])
                marker_time = float(marker["time_s"])
                marker_score = float(marker["score"])
            except (KeyError, TypeError, ValueError) as exc:
                raise SynchronizationError(f"sync receipt {marker_label} entry is invalid") from exc
            if (
                marker_index < 0
                or not np.isfinite([marker_time, marker_score]).all()
                or marker_time <= previous_time
                or not isinstance(marker.get("source"), str)
                or not marker["source"]
            ):
                raise SynchronizationError(f"sync receipt {marker_label} entry is invalid")
            previous_time = marker_time
    return document


def write_canonical_json(path: str | Path, document: Mapping[str, Any]) -> None:
    """Write canonical JSON plus one newline; callers choose a new artifact path."""

    Path(path).write_bytes(canonical_json_bytes(document) + b"\n")


def read_sync_receipt(path: str | Path) -> dict[str, Any]:
    def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise SynchronizationError(f"sync receipt repeats JSON key {key!r}")
            result[key] = value
        return result

    def reject_nonfinite(value: str) -> None:
        raise SynchronizationError(f"sync receipt contains non-finite JSON number {value}")

    try:
        value = json.loads(
            Path(path).read_text(encoding="utf-8"),
            object_pairs_hook=reject_duplicate_keys,
            parse_constant=reject_nonfinite,
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise SynchronizationError(f"cannot read sync receipt: {exc}") from exc
    return validate_sync_receipt(value)


def _mapping_sha256(receipt: Mapping[str, Any]) -> str:
    result = receipt.get("result")
    if not isinstance(result, Mapping) or not isinstance(result.get("mapping"), Mapping):
        raise SynchronizationError("sync receipt has no approvable mapping")
    return hashlib.sha256(canonical_json_bytes(result["mapping"])).hexdigest()


def build_manual_approval(
    receipt: Mapping[str, Any],
    *,
    reviewer_id: str,
    decision: Literal["approve", "reject"],
    reviewed_at_utc: str,
    rationale: str,
) -> dict[str, Any]:
    """Build an explicit human decision bound to one exact sync proposal."""

    validated = validate_sync_receipt(receipt)
    if not isinstance(reviewer_id, str) or not reviewer_id.strip() or reviewer_id != reviewer_id.strip():
        raise SynchronizationError("reviewer_id must be a non-empty trimmed string")
    if decision not in {"approve", "reject"}:
        raise SynchronizationError("manual decision must be approve or reject")
    if not isinstance(rationale, str) or not rationale.strip():
        raise SynchronizationError("manual approval requires a non-empty rationale")
    approval: dict[str, Any] = {
        "schema": MANUAL_APPROVAL_SCHEMA,
        "session_id": validated["session_id"],
        "reviewed_at_utc": _utc_timestamp(reviewed_at_utc),
        "reviewer_id": reviewer_id,
        "decision": decision,
        "rationale": rationale,
        "sync_receipt_content_sha256": validated["content_sha256"],
        "mapping_sha256": _mapping_sha256(validated),
    }
    approval["content_sha256"] = canonical_content_sha256(approval)
    return approval


def validate_manual_approval(
    approval: Mapping[str, Any],
    receipt: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate a manual decision and its exact receipt/mapping binding."""

    validated_receipt = validate_sync_receipt(receipt)
    if not isinstance(approval, Mapping):
        raise SynchronizationError("manual approval must be a JSON object")
    document = dict(_json_compatible(approval))
    if document.get("schema") != MANUAL_APPROVAL_SCHEMA:
        raise SynchronizationError("unexpected manual approval schema")
    digest = _require_sha256(document.get("content_sha256"), "manual content_sha256")
    if canonical_content_sha256(document) != digest:
        raise SynchronizationError("manual approval content SHA-256 mismatch")
    if document.get("session_id") != validated_receipt["session_id"]:
        raise SynchronizationError("manual approval session mismatch")
    if document.get("sync_receipt_content_sha256") != validated_receipt["content_sha256"]:
        raise SynchronizationError("manual approval receipt binding mismatch")
    if document.get("mapping_sha256") != _mapping_sha256(validated_receipt):
        raise SynchronizationError("manual approval mapping binding mismatch")
    if document.get("decision") not in {"approve", "reject"}:
        raise SynchronizationError("manual approval decision is invalid")
    if not isinstance(document.get("reviewer_id"), str) or not document["reviewer_id"]:
        raise SynchronizationError("manual approval reviewer_id is invalid")
    if not isinstance(document.get("rationale"), str) or not document["rationale"].strip():
        raise SynchronizationError("manual approval rationale is invalid")
    _utc_timestamp(document.get("reviewed_at_utc"))
    return document


def synchronization_is_authorized(
    receipt: Mapping[str, Any],
    *,
    manual_approval: Mapping[str, Any] | None = None,
) -> bool:
    """Return whether a mapping may be consumed, validating all bindings."""

    validated = validate_sync_receipt(receipt)
    result = validated["result"]
    if result["decision"] == "accepted":
        if manual_approval is not None:
            decision = validate_manual_approval(manual_approval, validated)["decision"]
            return decision == "approve"
        return True
    if manual_approval is None:
        return False
    approval = validate_manual_approval(manual_approval, validated)
    return approval["decision"] == "approve"


__all__ = [
    "EXPECTED_RADAR_PAYLOAD_BINS",
    "SYNC_CONFIG_SCHEMA",
    "SYNC_RECEIPT_SCHEMA",
    "MANUAL_APPROVAL_SCHEMA",
    "SynchronizationError",
    "SynchronizationConfig",
    "MarkerCandidate",
    "RadarMotionEnvelope",
    "TimeMapping",
    "MarkerMatch",
    "SynchronizationResult",
    "AffineSampleIndex",
    "epoch_prior_offset_from_starts",
    "load_synchronization_config",
    "robust_radar_motion_envelope",
    "detect_radar_marker_candidates",
    "detect_rsp_marker_candidates",
    "estimate_marker_time_mapping",
    "synchronize_from_signals",
    "build_affine_sample_index",
    "apply_affine_sample_index",
    "canonical_json_bytes",
    "canonical_content_sha256",
    "build_sync_receipt",
    "validate_sync_receipt",
    "write_canonical_json",
    "read_sync_receipt",
    "build_manual_approval",
    "validate_manual_approval",
    "synchronization_is_authorized",
]
