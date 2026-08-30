"""Deterministic, label-free evidence for harmonic candidate-set SNNs.

This module is intentionally limited to quantities available at deployment:
frozen proposer modes, causal classical radar estimates, and cached radar/SVD
spectra.  Reference RR and reference-quality fields are not accepted by any
candidate or evidence builder.

The large range-frequency support is exposed only through batch-oriented
functions.  A canonical RF batch has shape ``[N, R, F, 182]``; the last axis
is *viewed*, never pooled, as two branches of 91 original range indices.  The
corresponding harmonic support has shape ``[N, K, R, Q, 2, 91]``.  Callers
should normally use :func:`iter_compact_node_feature_batches`, whose small
default batch size avoids materializing this tensor for the whole cohort.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
import hashlib
import json
from collections.abc import Iterator, Sequence
from typing import Any

import numpy as np


RR_MIN_BPM = 6.0
RR_MAX_BPM = 45.0
DEFAULT_MAX_CANDIDATES = 12
DEFAULT_MERGE_RADIUS_BPM = 0.5
HARMONIC_RATIOS = (0.25, 1.0 / 3.0, 0.5, 1.0, 2.0, 3.0, 4.0)
RF_BRANCH_NAMES = ("raw_power", "candidate_iq_phase_power")
RF_RANGE_BINS_PER_BRANCH = 91
VERIFIED_SVD_VARIANT_INDICES = (0, 1, 2, 3, 4, 5)
VERIFIED_SVD_VARIANT_NAMES = (
    "raw",
    "raw_standardized",
    "temporal_velocity",
    "temporal_velocity_standardized",
    "range_difference",
    "range_difference_standardized",
)


class CandidateSource(IntEnum):
    """Stable source channels used by :class:`CandidateBank`."""

    BASE = 0
    DIRECT_MODE = 1
    CLASSICAL_X1 = 2
    CLASSICAL_X2 = 3
    CLASSICAL_X3 = 4
    CLASSICAL_X4 = 5
    RADAR_PEAK_1 = 6
    RADAR_PEAK_2 = 7
    RADAR_PEAK_3 = 8


CANDIDATE_SOURCE_NAMES = tuple(source.name.lower() for source in CandidateSource)

# Only these metadata fields may be selected for a model-forward path.  In
# particular, identity/session/protocol are useful for grouping and lineage,
# but must never become learned features.
FORWARD_METADATA_ALLOWLIST = (
    "window_number",
    "window_start_s",
    "classical_rr_bpm",
    "classical_confidence",
    "radar_peak_1_bpm",
    "radar_peak_2_bpm",
    "radar_peak_3_bpm",
    "radar_peak_spread_bpm",
)
CANDIDATE_METADATA_FIELDS = (
    "classical_rr_bpm",
    "classical_confidence",
    "radar_peak_1_bpm",
    "radar_peak_2_bpm",
    "radar_peak_3_bpm",
)
FORBIDDEN_TARGET_QC_FIELDS = frozenset(
    {
        "rr_bpm",
        "rr_spectral_bpm",
        "rr_phase_bpm",
        "rr_events_bpm",
        "reference_valid",
        "reference_quality",
        "reference_sigma_bpm",
        "spectral_concentration",
        "periodicity",
        "interval_cv",
        "estimator_disagreement_bpm",
        "phase_residual_rad",
        "clip_fraction",
        "guard_clip_fraction",
        "plateau_fraction",
        "breath_count",
        "classical_error_bpm",
        "radar_observable",
        "classical_acceptable_within_2bpm",
        "identity",
        "session_id",
        "session_number",
        "protocol",
        "fold",
    }
)

SEMANTIC_ROW_FIELDS = (
    "cache_index",
    "session_id",
    "identity",
    "protocol",
    "fold",
    "window_number",
    "window_start_s",
    "window_end_s",
)
_SEMANTIC_STRING_FIELDS = frozenset(("session_id", "identity", "protocol"))
_SEMANTIC_INTEGER_FIELDS = frozenset(
    ("cache_index", "fold", "window_number", "session_number")
)


def _readonly(array: np.ndarray) -> np.ndarray:
    """Return ``array`` with accidental in-place mutation disabled."""

    result = np.asarray(array)
    result.setflags(write=False)
    return result


@dataclass(frozen=True, slots=True)
class CandidateBank:
    """A fixed-width bank of deployment-time respiratory-rate proposals.

    ``source_mask`` is multi-hot because a stable merge may show that several
    independent sources support the same anchor.  ``primary_source`` is the
    first (highest-priority) source that created the anchor; padding uses -1.
    """

    bpm: np.ndarray
    mask: np.ndarray
    confidence: np.ndarray
    source_mask: np.ndarray
    primary_source: np.ndarray
    merge_radius_bpm: float = DEFAULT_MERGE_RADIUS_BPM
    rr_min_bpm: float = RR_MIN_BPM
    rr_max_bpm: float = RR_MAX_BPM
    source_confidence: np.ndarray | None = None

    def __post_init__(self) -> None:
        bpm = np.asarray(self.bpm)
        if bpm.ndim != 2:
            raise ValueError("candidate bpm must have shape [rows, candidates]")
        if np.asarray(self.mask).shape != bpm.shape:
            raise ValueError("candidate mask shape mismatch")
        if np.asarray(self.confidence).shape != bpm.shape:
            raise ValueError("candidate confidence shape mismatch")
        if np.asarray(self.primary_source).shape != bpm.shape:
            raise ValueError("candidate primary_source shape mismatch")
        if np.asarray(self.source_mask).shape != (
            *bpm.shape,
            len(CANDIDATE_SOURCE_NAMES),
        ):
            raise ValueError("candidate source_mask shape mismatch")
        if self.source_confidence is None:
            # Keep construction through the public dataclass backwards compatible.
            # Older callers only retained the maximum merged confidence, so the
            # best faithful expansion is to attach it to every asserted source.
            expanded = (
                np.asarray(self.source_mask, dtype=np.float32)
                * np.asarray(self.confidence, dtype=np.float32)[..., None]
            )
            object.__setattr__(self, "source_confidence", _readonly(expanded))
        elif np.asarray(self.source_confidence).shape != (
            *bpm.shape,
            len(CANDIDATE_SOURCE_NAMES),
        ):
            raise ValueError("candidate source_confidence shape mismatch")
        elif not np.isfinite(np.asarray(self.source_confidence)).all():
            raise ValueError("candidate source_confidence contains non-finite values")
        if not np.isfinite(self.merge_radius_bpm) or self.merge_radius_bpm < 0:
            raise ValueError("candidate merge radius must be non-negative")
        if not (
            np.isfinite(self.rr_min_bpm)
            and np.isfinite(self.rr_max_bpm)
            and self.rr_min_bpm < self.rr_max_bpm
        ):
            raise ValueError("candidate RR bounds are invalid")

    @property
    def rows(self) -> int:
        return int(self.bpm.shape[0])

    @property
    def max_candidates(self) -> int:
        return int(self.bpm.shape[1])

    def subset(self, index: slice | np.ndarray | Sequence[int]) -> "CandidateBank":
        """Return a row subset without altering candidate order."""

        return CandidateBank(
            bpm=_readonly(np.asarray(self.bpm[index])),
            mask=_readonly(np.asarray(self.mask[index])),
            confidence=_readonly(np.asarray(self.confidence[index])),
            source_mask=_readonly(np.asarray(self.source_mask[index])),
            primary_source=_readonly(np.asarray(self.primary_source[index])),
            merge_radius_bpm=float(self.merge_radius_bpm),
            rr_min_bpm=float(self.rr_min_bpm),
            rr_max_bpm=float(self.rr_max_bpm),
            source_confidence=_readonly(np.asarray(self.source_confidence[index])),
        )

    def manifest(self) -> dict[str, Any]:
        """Return the settings and content digest needed for stack provenance."""

        digest = hashlib.sha256()
        for name, value in (
            ("bpm", self.bpm),
            ("mask", self.mask),
            ("confidence", self.confidence),
            ("source_mask", self.source_mask),
            ("primary_source", self.primary_source),
            ("source_confidence", self.source_confidence),
        ):
            array = np.ascontiguousarray(value)
            digest.update(name.encode("utf-8"))
            digest.update(str(array.dtype).encode("utf-8"))
            digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode())
            digest.update(array.view(np.uint8))
        return {
            "schema_version": 2,
            "rows": self.rows,
            "maximum_candidates": self.max_candidates,
            "valid_candidate_count": int(np.asarray(self.mask).sum()),
            "rr_min_bpm": float(self.rr_min_bpm),
            "rr_max_bpm": float(self.rr_max_bpm),
            "merge_radius_bpm": float(self.merge_radius_bpm),
            "merge_anchor_policy": "first_source_anchor_never_moves",
            "candidate_priority": [
                "supplied_proposer_order",
                "classical_x1_x2_x3_x4",
                "radar_peak_1_2_3",
            ],
            "source_channels": list(CANDIDATE_SOURCE_NAMES),
            "content_sha256": digest.hexdigest(),
        }


@dataclass(frozen=True, slots=True)
class RFHarmonicSupport:
    """Range-preserving RF support for every candidate and harmonic ratio."""

    values: np.ndarray  # [N, K, R, Q, branch=2, range=91]
    mask: np.ndarray  # [N, K, R, Q]
    centers_bpm: np.ndarray  # [N, K, Q]
    radar_mask: np.ndarray  # [N, R]
    ratios: tuple[float, ...]

    def __post_init__(self) -> None:
        values = np.asarray(self.values)
        if values.ndim != 6 or values.shape[-2:] != (
            len(RF_BRANCH_NAMES),
            RF_RANGE_BINS_PER_BRANCH,
        ):
            raise ValueError(
                "RF support must have shape [N,K,R,Q,2,91]"
            )
        if np.asarray(self.mask).shape != values.shape[:4]:
            raise ValueError("RF support mask shape mismatch")
        if np.asarray(self.centers_bpm).shape != (
            values.shape[0],
            values.shape[1],
            values.shape[3],
        ):
            raise ValueError("RF support center shape mismatch")
        if np.asarray(self.radar_mask).shape != (values.shape[0], values.shape[2]):
            raise ValueError("RF radar mask shape mismatch")
        if len(self.ratios) != values.shape[3]:
            raise ValueError("RF ratio count mismatch")


@dataclass(frozen=True, slots=True)
class SVDHarmonicSupport:
    """Verified, component-preserving SVD harmonic support."""

    values: np.ndarray  # [N, K, R, Q, verified_variant=6, component=6|12]
    mask: np.ndarray  # [N, K, R, Q]
    centers_bpm: np.ndarray  # [N, K, Q]
    reliability: np.ndarray  # [N, R, 6, component]
    component_mask: np.ndarray  # [N, R, 6, component]
    component_peak_bpm: np.ndarray  # [N, R, 6, component]
    radar_mask: np.ndarray  # [N, R]
    ratios: tuple[float, ...]

    def __post_init__(self) -> None:
        values = np.asarray(self.values)
        component_count = int(values.shape[-1]) if values.ndim == 6 else -1
        expected_tail = (len(VERIFIED_SVD_VARIANT_INDICES), component_count)
        if component_count not in (6, 12):
            raise ValueError("SVD support must retain either 6 or 12 components")
        if values.ndim != 6 or values.shape[-2:] != expected_tail:
            raise ValueError("SVD support must have shape [N,K,R,Q,6,6|12]")
        if np.asarray(self.mask).shape != values.shape[:4]:
            raise ValueError("SVD support mask shape mismatch")
        component_shape = (values.shape[0], values.shape[2], *expected_tail)
        for name, array in (
            ("reliability", self.reliability),
            ("component_mask", self.component_mask),
            ("component_peak_bpm", self.component_peak_bpm),
        ):
            if np.asarray(array).shape != component_shape:
                raise ValueError(f"SVD {name} shape mismatch")
        if np.asarray(self.radar_mask).shape != (values.shape[0], values.shape[2]):
            raise ValueError("SVD radar mask shape mismatch")
        if np.asarray(self.centers_bpm).shape != (
            values.shape[0],
            values.shape[1],
            values.shape[3],
        ):
            raise ValueError("SVD support center shape mismatch")
        if len(self.ratios) != values.shape[3]:
            raise ValueError("SVD ratio count mismatch")


@dataclass(frozen=True, slots=True)
class NodeFeatureBatch:
    """Compact fixed-width graph-node features."""

    features: np.ndarray
    mask: np.ndarray
    feature_names: tuple[str, ...]

    def __post_init__(self) -> None:
        features = np.asarray(self.features)
        if features.ndim != 3:
            raise ValueError("node features must have shape [N,K,F]")
        if np.asarray(self.mask).shape != features.shape[:2]:
            raise ValueError("node feature mask shape mismatch")
        if len(self.feature_names) != features.shape[-1]:
            raise ValueError("node feature name count mismatch")


@dataclass(frozen=True, slots=True)
class HarmonicFeatureBatch:
    """One bounded-memory evidence batch yielded by the lazy iterator."""

    row_slice: slice
    candidates: CandidateBank
    rf_support: RFHarmonicSupport
    svd_support: SVDHarmonicSupport
    nodes: NodeFeatureBatch


def _as_1d(values: Any, *, name: str, rows: int | None = None) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or (rows is not None and len(array) != rows):
        expected = "[rows]" if rows is None else f"[{rows}]"
        raise ValueError(f"{name} must have shape {expected}")
    return array


def _as_proposals(values: Any | None, rows: int, *, name: str) -> np.ndarray:
    if values is None:
        return np.empty((rows, 0), dtype=np.float64)
    array = np.asarray(values, dtype=np.float64)
    if array.ndim == 1:
        if len(array) != rows:
            raise ValueError(f"{name} one-dimensional input must have one value per row")
        array = array[:, None]
    if array.ndim != 2 or array.shape[0] != rows:
        raise ValueError(f"{name} must have shape [rows, proposals]")
    return array


def _proposal_sources(
    source: Any | None,
    *,
    rows: int,
    proposals: int,
) -> np.ndarray:
    if proposals == 0:
        return np.empty((rows, 0), dtype=np.int16)
    if source is None:
        default = np.full(proposals, int(CandidateSource.DIRECT_MODE), dtype=np.int16)
        default[0] = int(CandidateSource.BASE)
        return np.broadcast_to(default, (rows, proposals)).copy()
    raw = np.asarray(source)
    if raw.ndim == 1:
        if len(raw) != proposals:
            raise ValueError("proposal_source must have one entry per proposal")
        raw = np.broadcast_to(raw, (rows, proposals))
    if raw.shape != (rows, proposals):
        raise ValueError("proposal_source must have shape [rows, proposals]")
    if raw.dtype.kind in "OUS":
        lookup = {name: index for index, name in enumerate(CANDIDATE_SOURCE_NAMES)}
        try:
            normalized = np.asarray(
                [[lookup[str(value).lower()] for value in row] for row in raw],
                dtype=np.int16,
            )
        except KeyError as error:
            raise ValueError(f"unknown proposal source {error.args[0]!r}") from error
    else:
        normalized = raw.astype(np.int16, copy=False)
    allowed = (normalized == int(CandidateSource.BASE)) | (
        normalized == int(CandidateSource.DIRECT_MODE)
    )
    if not np.all(allowed):
        raise ValueError("generic proposals may only be base or direct_mode sources")
    return normalized


def build_candidate_bank(
    *,
    classical_rr_bpm: np.ndarray,
    classical_confidence: np.ndarray,
    radar_peaks_bpm: np.ndarray,
    proposal_bpm: np.ndarray | None = None,
    proposal_confidence: np.ndarray | None = None,
    proposal_mask: np.ndarray | None = None,
    proposal_source: np.ndarray | Sequence[str] | None = None,
    radar_peak_confidence: np.ndarray | None = None,
    rr_min_bpm: float = RR_MIN_BPM,
    rr_max_bpm: float = RR_MAX_BPM,
    merge_radius_bpm: float = DEFAULT_MERGE_RADIUS_BPM,
    max_candidates: int = DEFAULT_MAX_CANDIDATES,
) -> CandidateBank:
    """Build a deterministic candidate set without consulting any label.

    Candidates are admitted in a fixed priority order: supplied base/direct
    proposals, classical x1..x4, then radar peaks 1..3.  A later candidate
    within ``merge_radius_bpm`` of an existing anchor adds its source bit and
    may raise the anchor confidence, but never moves the anchor.  This both
    keeps a frozen base proposal bit-exact and prevents order-dependent drift.
    Valid anchors are finally stably sorted by BPM; padding follows them.
    """

    classical = _as_1d(classical_rr_bpm, name="classical_rr_bpm")
    rows = len(classical)
    classical_conf = _as_1d(
        classical_confidence, name="classical_confidence", rows=rows
    )
    peaks = np.asarray(radar_peaks_bpm, dtype=np.float64)
    if peaks.ndim != 2 or peaks.shape[0] != rows or not 1 <= peaks.shape[1] <= 3:
        raise ValueError("radar_peaks_bpm must have shape [rows, 1..3]")
    if not (
        np.isfinite(rr_min_bpm)
        and np.isfinite(rr_max_bpm)
        and rr_min_bpm < rr_max_bpm
    ):
        raise ValueError("candidate RR bounds are invalid")
    if not np.isfinite(merge_radius_bpm) or merge_radius_bpm < 0:
        raise ValueError("merge_radius_bpm must be non-negative")
    if int(max_candidates) < 1:
        raise ValueError("max_candidates must be positive")
    max_candidates = int(max_candidates)

    proposals = _as_proposals(proposal_bpm, rows, name="proposal_bpm")
    if proposal_confidence is None:
        proposal_conf = np.ones_like(proposals)
    else:
        proposal_conf = _as_proposals(
            proposal_confidence, rows, name="proposal_confidence"
        )
        if proposal_conf.shape != proposals.shape:
            raise ValueError("proposal_confidence shape mismatch")
    if proposal_mask is None:
        proposal_available = np.ones(proposals.shape, dtype=bool)
    else:
        proposal_available = np.asarray(proposal_mask, dtype=bool)
        if proposal_available.shape != proposals.shape:
            raise ValueError("proposal_mask shape mismatch")
    proposal_sources = _proposal_sources(
        proposal_source, rows=rows, proposals=proposals.shape[1]
    )

    if radar_peak_confidence is None:
        peak_conf = np.broadcast_to(classical_conf[:, None], peaks.shape)
    else:
        peak_conf = np.asarray(radar_peak_confidence, dtype=np.float64)
        if peak_conf.shape != peaks.shape:
            raise ValueError("radar_peak_confidence shape mismatch")

    output_bpm = np.zeros((rows, max_candidates), dtype=np.float32)
    output_mask = np.zeros((rows, max_candidates), dtype=bool)
    output_conf = np.zeros((rows, max_candidates), dtype=np.float32)
    output_source = np.zeros(
        (rows, max_candidates, len(CANDIDATE_SOURCE_NAMES)), dtype=bool
    )
    output_source_confidence = np.zeros(
        (rows, max_candidates, len(CANDIDATE_SOURCE_NAMES)), dtype=np.float32
    )
    output_primary = np.full((rows, max_candidates), -1, dtype=np.int16)

    for row in range(rows):
        anchors: list[float] = []
        confidences: list[float] = []
        sources: list[np.ndarray] = []
        source_confidences: list[np.ndarray] = []
        primary: list[int] = []

        def admit(value: float, confidence: float, source_index: int) -> None:
            if not np.isfinite(value) or not rr_min_bpm <= value <= rr_max_bpm:
                return
            confidence = (
                float(np.clip(confidence, 0.0, 1.0))
                if np.isfinite(confidence)
                else 0.0
            )
            if anchors:
                distance = np.abs(np.asarray(anchors, dtype=np.float64) - value)
                eligible = np.flatnonzero(distance <= merge_radius_bpm + 1.0e-12)
            else:
                eligible = np.empty(0, dtype=np.int64)
            if len(eligible):
                chosen = int(eligible[np.argmin(distance[eligible])])
                confidences[chosen] = max(confidences[chosen], confidence)
                sources[chosen][source_index] = True
                source_confidences[chosen][source_index] = max(
                    float(source_confidences[chosen][source_index]), confidence
                )
                return
            if len(anchors) >= max_candidates:
                return
            anchors.append(float(value))
            confidences.append(confidence)
            source_bits = np.zeros(len(CANDIDATE_SOURCE_NAMES), dtype=bool)
            source_bits[source_index] = True
            sources.append(source_bits)
            per_source_confidence = np.zeros(
                len(CANDIDATE_SOURCE_NAMES), dtype=np.float32
            )
            per_source_confidence[source_index] = confidence
            source_confidences.append(per_source_confidence)
            primary.append(int(source_index))

        for proposal in range(proposals.shape[1]):
            if proposal_available[row, proposal]:
                admit(
                    float(proposals[row, proposal]),
                    float(proposal_conf[row, proposal]),
                    int(proposal_sources[row, proposal]),
                )
        for factor_index, factor in enumerate((1.0, 2.0, 3.0, 4.0)):
            admit(
                float(classical[row] * factor),
                float(classical_conf[row]),
                int(CandidateSource.CLASSICAL_X1) + factor_index,
            )
        for radar in range(peaks.shape[1]):
            admit(
                float(peaks[row, radar]),
                float(peak_conf[row, radar]),
                int(CandidateSource.RADAR_PEAK_1) + radar,
            )

        if anchors:
            order = np.argsort(np.asarray(anchors), kind="stable")
            count = len(order)
            output_bpm[row, :count] = np.asarray(anchors, np.float32)[order]
            output_conf[row, :count] = np.asarray(confidences, np.float32)[order]
            output_source[row, :count] = np.asarray(sources, bool)[order]
            output_source_confidence[row, :count] = np.asarray(
                source_confidences, np.float32
            )[order]
            output_primary[row, :count] = np.asarray(primary, np.int16)[order]
            output_mask[row, :count] = True

    return CandidateBank(
        bpm=_readonly(output_bpm),
        mask=_readonly(output_mask),
        confidence=_readonly(output_conf),
        source_mask=_readonly(output_source),
        primary_source=_readonly(output_primary),
        merge_radius_bpm=float(merge_radius_bpm),
        rr_min_bpm=float(rr_min_bpm),
        rr_max_bpm=float(rr_max_bpm),
        source_confidence=_readonly(output_source_confidence),
    )


def _column(metadata: Any, name: str) -> np.ndarray:
    try:
        values = metadata[name]
    except (KeyError, TypeError) as error:
        raise KeyError(f"metadata is missing required field {name!r}") from error
    if hasattr(values, "to_numpy"):
        return np.asarray(values.to_numpy())
    return np.asarray(values)


def select_forward_metadata(
    metadata: Any,
    *,
    fields: Sequence[str] = CANDIDATE_METADATA_FIELDS,
) -> dict[str, np.ndarray]:
    """Select model inputs through an explicit deployment-safe allow-list.

    Extra columns in ``metadata`` are ignored.  Requesting a reference, QC,
    identity, protocol or unregistered field is a hard error rather than an
    invitation to silently expand the model boundary.
    """

    selected = tuple(str(field) for field in fields)
    if not selected or len(set(selected)) != len(selected):
        raise ValueError("forward metadata fields must be non-empty and unique")
    disallowed = sorted(set(selected) - set(FORWARD_METADATA_ALLOWLIST))
    if disallowed:
        raise ValueError(f"fields are outside the forward allow-list: {disallowed}")
    return {name: _column(metadata, name).copy() for name in selected}


def candidate_bank_from_metadata(
    metadata: Any,
    **kwargs: Any,
) -> CandidateBank:
    """Convenience wrapper whose metadata reads are statically allow-listed."""

    fields = select_forward_metadata(metadata)
    peaks = np.stack(
        [fields[f"radar_peak_{radar}_bpm"] for radar in (1, 2, 3)], axis=1
    )
    return build_candidate_bank(
        classical_rr_bpm=fields["classical_rr_bpm"],
        classical_confidence=fields["classical_confidence"],
        radar_peaks_bpm=peaks,
        **kwargs,
    )


def _frequency_grid(frequencies_hz: np.ndarray, expected: int) -> np.ndarray:
    grid = np.asarray(frequencies_hz, dtype=np.float64)
    if grid.ndim != 1 or len(grid) != expected or len(grid) < 2:
        raise ValueError("frequency grid must be one-dimensional and match spectra")
    if not np.isfinite(grid).all() or np.any(np.diff(grid) <= 0):
        raise ValueError("frequency grid must be finite and strictly increasing")
    return grid


def _triangular_brackets(
    frequencies_hz: np.ndarray,
    centers_bpm: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    centers_hz = np.asarray(centers_bpm, dtype=np.float64) / 60.0
    # Cache grids are float32 while callers commonly express an exact BPM in
    # float64.  Treat only round-off-sized endpoint differences as equality;
    # genuinely out-of-band centers remain invalid and are zeroed below.
    endpoint_tolerance = max(
        np.finfo(np.float32).eps * max(1.0, abs(float(frequencies_hz[-1]))),
        float(np.min(np.diff(frequencies_hz))) * 1.0e-7,
    )
    valid = (
        np.isfinite(centers_hz)
        & (centers_hz >= frequencies_hz[0] - endpoint_tolerance)
        & (centers_hz <= frequencies_hz[-1] + endpoint_tolerance)
    )
    safe = np.where(
        valid,
        np.clip(centers_hz, frequencies_hz[0], frequencies_hz[-1]),
        frequencies_hz[0],
    )
    right = np.searchsorted(frequencies_hz, safe, side="right")
    right = np.clip(right, 1, len(frequencies_hz) - 1)
    left = right - 1
    denominator = frequencies_hz[right] - frequencies_hz[left]
    right_weight = np.clip(
        (safe - frequencies_hz[left]) / denominator, 0.0, 1.0
    )
    return left.astype(np.int64), right.astype(np.int64), right_weight, valid


def triangular_sample_native_grid(
    values: np.ndarray,
    frequencies_hz: np.ndarray,
    centers_bpm: np.ndarray | float,
) -> tuple[np.ndarray, np.ndarray]:
    """Linearly sample a native frequency grid with a triangular two-bin kernel.

    ``values`` is either ``[F]`` or ``[N, ..., F]``.  For a batched input,
    ``centers_bpm`` must be ``[N, ...]`` and the result is
    ``[N, value_channels..., center_axes...]``.  Centers outside the grid
    produce exactly zero and ``False``; they are never clamped to an edge bin.
    """

    array = np.asarray(values)
    if array.ndim < 1:
        raise ValueError("values must have a frequency axis")
    grid = _frequency_grid(frequencies_hz, array.shape[-1])
    if not np.isfinite(array).all():
        raise ValueError("values contain non-finite spectral support")
    unbatched = array.ndim == 1
    if unbatched:
        batch_values = array[None, :]
        centers = np.asarray(centers_bpm, dtype=np.float64)[None, ...]
    else:
        batch_values = array
        centers = np.asarray(centers_bpm, dtype=np.float64)
        if centers.ndim < 1 or centers.shape[0] != array.shape[0]:
            raise ValueError("batched centers must share the values row axis")
    left, right, weight, valid = _triangular_brackets(grid, centers)
    channel_shape = batch_values.shape[1:-1]
    center_shape = centers.shape[1:]
    output = np.zeros(
        (batch_values.shape[0], *channel_shape, *center_shape), dtype=np.float32
    )
    for row in range(batch_values.shape[0]):
        flat = np.asarray(batch_values[row], dtype=np.float32).reshape(-1, len(grid))
        row_left = left[row].reshape(-1)
        row_right = right[row].reshape(-1)
        row_weight = weight[row].reshape(1, -1).astype(np.float32)
        sampled = flat[:, row_left] * (1.0 - row_weight)
        sampled += flat[:, row_right] * row_weight
        sampled[:, ~valid[row].reshape(-1)] = 0.0
        output[row] = sampled.reshape(*channel_shape, *center_shape)
    if unbatched:
        return output[0], np.asarray(valid[0], dtype=bool)
    return output, np.asarray(valid, dtype=bool)


def reshape_rf_branches(
    maps: np.ndarray,
    *,
    range_bins: int = RF_RANGE_BINS_PER_BRANCH,
) -> np.ndarray:
    """View the flattened RF range axis as ``[branch=2, raw_range=91]``.

    No pooling, sorting, or range-index permutation is performed.
    """

    array = np.asarray(maps)
    if array.ndim < 1 or array.shape[-1] != len(RF_BRANCH_NAMES) * int(range_bins):
        raise ValueError(
            f"RF map final axis must contain 2*{int(range_bins)} branch/range bins"
        )
    return array.reshape(*array.shape[:-1], len(RF_BRANCH_NAMES), int(range_bins))


def resolve_joint_radar_mask(
    rf_maps: np.ndarray,
    svd_spectra: np.ndarray,
    *,
    svd_attributes: np.ndarray | None = None,
    explicit_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Fail closed unless each radar has finite, nonzero RF *and* SVD evidence."""

    rf = np.asarray(rf_maps)
    svd = np.asarray(svd_spectra)
    if rf.ndim != 4 or svd.ndim != 5 or rf.shape[:2] != svd.shape[:2]:
        raise ValueError("RF/SVD inputs must have shapes [N,R,F,X]/[N,R,V,C,F]")
    reduce_rf = tuple(range(2, rf.ndim))
    reduce_svd = tuple(range(2, svd.ndim))
    rf_ok = np.isfinite(rf).all(axis=reduce_rf) & np.any(rf != 0, axis=reduce_rf)
    svd_ok = np.isfinite(svd).all(axis=reduce_svd) & np.any(
        svd != 0, axis=reduce_svd
    )
    available = rf_ok & svd_ok
    if svd_attributes is not None:
        attributes = np.asarray(svd_attributes)
        if attributes.shape[:2] != svd.shape[:2] or attributes.ndim != 5:
            raise ValueError("SVD attributes must have shape [N,R,V,C,A]")
        available &= np.isfinite(attributes).all(axis=tuple(range(2, attributes.ndim)))
    if explicit_mask is not None:
        explicit = np.asarray(explicit_mask, dtype=bool)
        if explicit.shape != available.shape:
            raise ValueError("explicit radar mask shape mismatch")
        available &= explicit
    return available.astype(bool, copy=False)


def _candidate_centers(
    candidates: CandidateBank,
    ratios: Sequence[float],
) -> tuple[np.ndarray, tuple[float, ...]]:
    normalized = tuple(float(value) for value in ratios)
    if not normalized or not np.isfinite(normalized).all() or min(normalized) <= 0:
        raise ValueError("harmonic ratios must be finite and positive")
    centers = np.asarray(candidates.bpm, dtype=np.float32)[..., None] * np.asarray(
        normalized, dtype=np.float32
    )
    centers *= np.asarray(candidates.mask, dtype=np.float32)[..., None]
    return centers, normalized


def sample_rf_harmonic_support(
    maps: np.ndarray,
    frequencies_hz: np.ndarray,
    candidates: CandidateBank,
    *,
    radar_mask: np.ndarray | None = None,
    ratios: Sequence[float] = HARMONIC_RATIOS,
) -> RFHarmonicSupport:
    """Sample candidate harmonics while retaining both 91-bin range branches."""

    raw = np.asarray(maps)
    if raw.ndim != 4 or raw.shape[0] != candidates.rows:
        raise ValueError("RF maps must have shape [rows, radars, frequencies, 182]")
    branches = reshape_rf_branches(raw)
    grid = _frequency_grid(frequencies_hz, raw.shape[2])
    centers, ratio_tuple = _candidate_centers(candidates, ratios)
    left, right, weight, in_band = _triangular_brackets(grid, centers)
    if radar_mask is None:
        available = np.isfinite(raw).all(axis=(2, 3)) & np.any(raw != 0, axis=(2, 3))
    else:
        available = np.asarray(radar_mask, dtype=bool)
        if available.shape != raw.shape[:2]:
            raise ValueError("RF radar mask shape mismatch")
        available = (
            available
            & np.isfinite(raw).all(axis=(2, 3))
            & np.any(raw != 0, axis=(2, 3))
        )
    rows, radars = raw.shape[:2]
    count = candidates.max_candidates
    ratio_count = len(ratio_tuple)
    output = np.zeros(
        (
            rows,
            count,
            radars,
            ratio_count,
            len(RF_BRANCH_NAMES),
            RF_RANGE_BINS_PER_BRANCH,
        ),
        dtype=np.float32,
    )
    safe_branches = np.nan_to_num(
        np.asarray(branches, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0
    )
    for row in range(rows):
        flattened_left = left[row].reshape(-1)
        flattened_right = right[row].reshape(-1)
        row_values = safe_branches[row]  # [R,F,branch,range]
        lower = np.take(row_values, flattened_left, axis=1)
        upper = np.take(row_values, flattened_right, axis=1)
        row_weight = weight[row].reshape(1, -1, 1, 1).astype(np.float32)
        sampled = lower * (1.0 - row_weight) + upper * row_weight
        sampled = sampled.reshape(
            radars,
            count,
            ratio_count,
            len(RF_BRANCH_NAMES),
            RF_RANGE_BINS_PER_BRANCH,
        ).transpose(1, 0, 2, 3, 4)
        row_mask = (
            np.asarray(candidates.mask[row])[:, None, None]
            & available[row][None, :, None]
            & in_band[row][:, None, :]
        )
        sampled *= row_mask[..., None, None]
        output[row] = sampled
    mask = (
        np.asarray(candidates.mask)[:, :, None, None]
        & available[:, None, :, None]
        & in_band[:, :, None, :]
    )
    return RFHarmonicSupport(
        values=_readonly(output),
        mask=_readonly(mask.astype(bool, copy=False)),
        centers_bpm=_readonly(centers.astype(np.float32, copy=False)),
        radar_mask=_readonly(available.astype(bool, copy=False)),
        ratios=ratio_tuple,
    )


def _svd_reliability(
    spectra: np.ndarray,
    attributes: np.ndarray,
    radar_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    diagnostics = np.asarray(attributes, dtype=np.float32)
    energy = np.clip(diagnostics[..., 0], 0.0, 1.0)
    band = np.clip(diagnostics[..., 1], 0.0, 1.0)
    concentration = np.clip(diagnostics[..., 2], 0.0, 1.0)
    entropy = np.clip(diagnostics[..., 3], 0.0, 1.0)
    energy_band = energy * band
    quality = (
        np.sqrt(np.maximum(energy_band, 1.0e-8))
        * (energy_band > 0)
        * concentration
        * (1.0 - 0.5 * entropy)
    )
    has_power = np.asarray(spectra, dtype=np.float32).sum(axis=-1) > 0
    component_mask = (
        np.isfinite(diagnostics).all(axis=-1)
        & has_power
        & (quality > 0)
        & radar_mask[:, :, None, None]
    )
    quality = np.where(component_mask, quality, 0.0)
    positive_count = component_mask.sum(axis=(-2, -1), keepdims=True)
    denominator = np.divide(
        quality.sum(axis=(-2, -1), keepdims=True),
        np.maximum(positive_count, 1),
    )
    normalized = np.divide(
        quality,
        np.maximum(denominator, 1.0e-8),
        out=np.zeros_like(quality),
        where=component_mask,
    )
    normalized = np.clip(normalized, 0.0, 4.0) * component_mask
    peak_bpm = np.where(
        component_mask, diagnostics[..., 4] * 60.0, 0.0
    ).astype(np.float32)
    return (
        normalized.astype(np.float32, copy=False),
        component_mask.astype(bool, copy=False),
        peak_bpm,
    )


def sample_svd_harmonic_support(
    spectra: np.ndarray,
    attributes: np.ndarray,
    frequencies_hz: np.ndarray,
    candidates: CandidateBank,
    *,
    radar_mask: np.ndarray | None = None,
    ratios: Sequence[float] = HARMONIC_RATIOS,
    components: int = 6,
) -> SVDHarmonicSupport:
    """Sample the six physically verified SVD variants component by component.

    Split-amplitude and split-phase hypotheses (cache variants 6..9) are never
    read here.  The first six SVD components are preserved as separate support
    values, accompanied by deterministic reliability derived from the cached
    energy/band/concentration/entropy attributes.
    """

    raw = np.asarray(spectra)
    attrs = np.asarray(attributes)
    if raw.ndim != 5 or raw.shape[0] != candidates.rows:
        raise ValueError("SVD spectra must have shape [rows,radars,variants,components,F]")
    if attrs.shape != (*raw.shape[:-1], 5):
        raise ValueError("SVD attributes must match spectra and have five diagnostics")
    if raw.shape[2] < len(VERIFIED_SVD_VARIANT_INDICES) or int(components) not in (6, 12):
        raise ValueError(
            "exactly the first six variants and either six or twelve components are required"
        )
    if raw.shape[3] < int(components):
        raise ValueError("SVD cache contains fewer than six components")
    grid = _frequency_grid(frequencies_hz, raw.shape[-1])
    selected = np.asarray(
        raw[:, :, VERIFIED_SVD_VARIANT_INDICES, :components, :], dtype=np.float32
    )
    selected_attrs = np.asarray(
        attrs[:, :, VERIFIED_SVD_VARIANT_INDICES, :components, :], dtype=np.float32
    )
    if radar_mask is None:
        available = (
            np.isfinite(selected).all(axis=(2, 3, 4))
            & np.isfinite(selected_attrs).all(axis=(2, 3, 4))
            & np.any(selected != 0, axis=(2, 3, 4))
        )
    else:
        available = np.asarray(radar_mask, dtype=bool)
        if available.shape != raw.shape[:2]:
            raise ValueError("SVD radar mask shape mismatch")
        available = (
            available
            & np.isfinite(selected).all(axis=(2, 3, 4))
            & np.isfinite(selected_attrs).all(axis=(2, 3, 4))
            & np.any(selected != 0, axis=(2, 3, 4))
        )
    safe = np.clip(
        np.nan_to_num(selected, nan=0.0, posinf=0.0, neginf=0.0), 0.0, None
    )
    reliability, component_mask, peak_bpm = _svd_reliability(
        safe, np.nan_to_num(selected_attrs), available
    )
    available &= component_mask.any(axis=(2, 3))
    component_mask &= available[:, :, None, None]
    reliability *= component_mask
    peak_bpm *= component_mask

    centers, ratio_tuple = _candidate_centers(candidates, ratios)
    left, right, weight, in_band = _triangular_brackets(grid, centers)
    rows, radars = raw.shape[:2]
    count = candidates.max_candidates
    ratio_count = len(ratio_tuple)
    variants = len(VERIFIED_SVD_VARIANT_INDICES)
    output = np.zeros(
        (rows, count, radars, ratio_count, variants, components), dtype=np.float32
    )
    for row in range(rows):
        flattened_left = left[row].reshape(-1)
        flattened_right = right[row].reshape(-1)
        row_values = safe[row]  # [R,V,C,F]
        lower = np.take(row_values, flattened_left, axis=-1)
        upper = np.take(row_values, flattened_right, axis=-1)
        row_weight = weight[row].reshape(1, 1, 1, -1).astype(np.float32)
        sampled = lower * (1.0 - row_weight) + upper * row_weight
        sampled = sampled.reshape(
            radars, variants, components, count, ratio_count
        ).transpose(3, 0, 4, 1, 2)
        row_mask = (
            np.asarray(candidates.mask[row])[:, None, None]
            & available[row][None, :, None]
            & in_band[row][:, None, :]
        )
        sampled *= row_mask[..., None, None]
        sampled *= component_mask[row][None, :, None, :, :]
        output[row] = sampled
    mask = (
        np.asarray(candidates.mask)[:, :, None, None]
        & available[:, None, :, None]
        & in_band[:, :, None, :]
    )
    return SVDHarmonicSupport(
        values=_readonly(output),
        mask=_readonly(mask.astype(bool, copy=False)),
        centers_bpm=_readonly(centers.astype(np.float32, copy=False)),
        reliability=_readonly(reliability.astype(np.float32, copy=False)),
        component_mask=_readonly(component_mask.astype(bool, copy=False)),
        component_peak_bpm=_readonly(peak_bpm.astype(np.float32, copy=False)),
        radar_mask=_readonly(available.astype(bool, copy=False)),
        ratios=ratio_tuple,
    )


def _ratio_name(value: float) -> str:
    common = {
        0.25: "r1_4",
        1.0 / 3.0: "r1_3",
        0.5: "r1_2",
        1.0: "r1",
        2.0: "r2",
        3.0: "r3",
        4.0: "r4",
    }
    return common.get(float(value), "r" + format(float(value), ".6g").replace(".", "p"))


def _stable_topk(profiles: np.ndarray, top_k: int) -> tuple[np.ndarray, np.ndarray]:
    flat = np.asarray(profiles, dtype=np.float32).reshape(-1, profiles.shape[-1])
    values = np.zeros((len(flat), top_k), dtype=np.float32)
    indices = np.zeros((len(flat), top_k), dtype=np.float32)
    normalizer = float(max(1, profiles.shape[-1] - 1))
    for row, profile in enumerate(flat):
        # Stable sorting makes equal-power bins prefer their original raw range
        # index, so cache/replay results do not depend on an unstable partition.
        chosen = np.argsort(-profile, kind="stable")[:top_k]
        values[row] = profile[chosen]
        indices[row] = chosen.astype(np.float32) / normalizer
    output_shape = (*profiles.shape[:-1], top_k)
    return values.reshape(output_shape), indices.reshape(output_shape)


def _rf_consensus(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Mean cosine agreement with the other available radars."""

    rows, candidates, radars, ratios, branches, _ = values.shape
    result = np.zeros((rows, candidates, radars, ratios, branches), np.float32)
    norm = np.sqrt(np.square(values, dtype=np.float64).sum(axis=-1))
    count = np.zeros_like(result, dtype=np.int16)
    for first in range(radars):
        for second in range(radars):
            if first == second:
                continue
            pair = mask[:, :, first, :] & mask[:, :, second, :]
            denominator = norm[:, :, first] * norm[:, :, second]
            dot = (values[:, :, first] * values[:, :, second]).sum(
                axis=-1, dtype=np.float64
            )
            cosine = np.divide(
                dot,
                np.maximum(denominator, 1.0e-12),
                out=np.zeros_like(dot),
                where=pair[..., None] & (denominator > 0),
            )
            result[:, :, first] += cosine.astype(np.float32)
            count[:, :, first] += (pair[..., None] & (denominator > 0)).astype(
                np.int16
            )
    return np.divide(
        result,
        np.maximum(count, 1),
        out=np.zeros_like(result),
        where=count > 0,
    )


def build_compact_node_features(
    candidates: CandidateBank,
    rf_support: RFHarmonicSupport,
    svd_support: SVDHarmonicSupport,
    *,
    rf_top_k: int = 2,
    proposer_node_features: np.ndarray | None = None,
    proposer_feature_names: Sequence[str] | None = None,
    include_source_confidence: bool = False,
    rr_min_bpm: float = RR_MIN_BPM,
    rr_max_bpm: float = RR_MAX_BPM,
) -> NodeFeatureBatch:
    """Compress label-free support into fixed-width candidate-node features.

    RF compression retains stable top-k range values *and their original
    normalized 0..90 indices* for every radar/ratio/branch.  SVD compression
    is performed only after component-preserving sampling and uses cached
    diagnostic reliability.  Padding is exactly zero.
    """

    rf = np.asarray(rf_support.values, dtype=np.float32)
    svd = np.asarray(svd_support.values, dtype=np.float32)
    if rf.shape[:4] != svd.shape[:4] or rf.shape[:2] != candidates.bpm.shape:
        raise ValueError("candidate/RF/SVD node axes do not match")
    if tuple(rf_support.ratios) != tuple(svd_support.ratios):
        raise ValueError("RF and SVD ratio grids do not match")
    if int(rf_top_k) < 1 or int(rf_top_k) > rf.shape[-1]:
        raise ValueError("rf_top_k is outside the preserved range axis")
    rf_top_k = int(rf_top_k)
    if not rr_min_bpm < rr_max_bpm:
        raise ValueError("node RR bounds are invalid")

    arrays: list[np.ndarray] = []
    names: list[str] = []

    def append(name: str, value: np.ndarray) -> None:
        array = np.asarray(value, dtype=np.float32)
        if array.shape != candidates.bpm.shape:
            raise RuntimeError(f"node feature {name} has inconsistent shape")
        arrays.append(array)
        names.append(name)

    bpm = np.asarray(candidates.bpm, dtype=np.float32)
    candidate_mask = np.asarray(candidates.mask, dtype=bool)
    append("candidate_bpm", bpm)
    append("candidate_bpm_unit", (bpm - rr_min_bpm) / (rr_max_bpm - rr_min_bpm))
    append("candidate_confidence", candidates.confidence)
    for source, name in enumerate(CANDIDATE_SOURCE_NAMES):
        append(f"source_{name}", candidates.source_mask[..., source])
    if include_source_confidence:
        for source, name in enumerate(CANDIDATE_SOURCE_NAMES):
            append(
                f"source_confidence_{name}",
                np.asarray(candidates.source_confidence)[..., source],
            )
    if proposer_node_features is None:
        if proposer_feature_names not in (None, (), []):
            raise ValueError("proposer feature names were supplied without values")
    else:
        proposer_values = np.asarray(proposer_node_features, dtype=np.float32)
        proposed_names = tuple(str(name) for name in (proposer_feature_names or ()))
        if proposer_values.ndim != 3 or proposer_values.shape[:2] != bpm.shape:
            raise ValueError("proposer node features must have shape [N,K,P]")
        if proposer_values.shape[-1] != len(proposed_names) or not proposed_names:
            raise ValueError("proposer node feature names do not match values")
        if len(set(proposed_names)) != len(proposed_names):
            raise ValueError("proposer node feature names must be unique")
        if not np.isfinite(proposer_values[candidate_mask]).all():
            raise ValueError("available proposer node features must be finite")
        for feature_index, name in enumerate(proposed_names):
            append(name, proposer_values[..., feature_index])
    count_fraction = candidate_mask.sum(axis=1, keepdims=True) / float(
        candidates.max_candidates
    )
    append("candidate_count_fraction", np.broadcast_to(count_fraction, bpm.shape))
    previous_gap = np.zeros_like(bpm)
    next_gap = np.zeros_like(bpm)
    if bpm.shape[1] > 1:
        previous_valid = candidate_mask[:, 1:] & candidate_mask[:, :-1]
        previous_gap[:, 1:] = np.where(
            previous_valid, bpm[:, 1:] - bpm[:, :-1], 0.0
        )
        next_gap[:, :-1] = np.where(
            previous_valid, bpm[:, 1:] - bpm[:, :-1], 0.0
        )
    append("previous_candidate_gap_bpm", previous_gap)
    append("next_candidate_gap_bpm", next_gap)

    nonnegative_rf = np.clip(np.nan_to_num(rf), 0.0, None)
    rf_sum = nonnegative_rf.sum(axis=-1, dtype=np.float64)
    rf_mean = nonnegative_rf.mean(axis=-1, dtype=np.float64)
    rf_max = nonnegative_rf.max(axis=-1)
    probability = np.divide(
        nonnegative_rf,
        np.maximum(rf_sum[..., None], 1.0e-12),
        out=np.zeros_like(nonnegative_rf),
        where=rf_sum[..., None] > 0,
    )
    rf_entropy = -(
        probability * np.log(np.maximum(probability, 1.0e-20))
    ).sum(axis=-1, dtype=np.float64) / np.log(nonnegative_rf.shape[-1])
    rf_concentration = np.divide(
        rf_max,
        np.maximum(rf_sum, 1.0e-12),
        out=np.zeros_like(rf_max),
        where=rf_sum > 0,
    )
    top_values, top_indices = _stable_topk(nonnegative_rf, rf_top_k)
    consensus = _rf_consensus(nonnegative_rf, np.asarray(rf_support.mask, bool))
    for radar in range(rf.shape[2]):
        for ratio_index, ratio in enumerate(rf_support.ratios):
            ratio_name = _ratio_name(ratio)
            valid = np.asarray(rf_support.mask[:, :, radar, ratio_index], np.float32)
            for branch, branch_name in enumerate(RF_BRANCH_NAMES):
                prefix = f"rf_radar{radar + 1}_{ratio_name}_{branch_name}"
                append(prefix + "_mean", rf_mean[:, :, radar, ratio_index, branch] * valid)
                append(prefix + "_max", rf_max[:, :, radar, ratio_index, branch] * valid)
                append(
                    prefix + "_entropy",
                    rf_entropy[:, :, radar, ratio_index, branch] * valid,
                )
                append(
                    prefix + "_peak_concentration",
                    rf_concentration[:, :, radar, ratio_index, branch] * valid,
                )
                for rank in range(rf_top_k):
                    append(
                        f"{prefix}_top{rank + 1}_value",
                        top_values[:, :, radar, ratio_index, branch, rank] * valid,
                    )
                    append(
                        f"{prefix}_top{rank + 1}_range_index_unit",
                        top_indices[:, :, radar, ratio_index, branch, rank] * valid,
                    )
                append(
                    prefix + "_cross_radar_consensus",
                    consensus[:, :, radar, ratio_index, branch] * valid,
                )

    reliability = np.asarray(svd_support.reliability, dtype=np.float32)
    component_mask = np.asarray(svd_support.component_mask, dtype=bool)
    peaks = np.asarray(svd_support.component_peak_bpm, dtype=np.float32)
    svd_nonnegative = np.clip(np.nan_to_num(svd), 0.0, None)
    for radar in range(svd.shape[2]):
        radar_reliability = reliability[:, radar]
        radar_component_mask = component_mask[:, radar]
        base_weight = radar_reliability * radar_component_mask
        base_weight_sum = base_weight.sum(axis=(-2, -1))
        base_reliability_mean = np.divide(
            base_weight_sum,
            np.maximum(radar_component_mask.sum(axis=(-2, -1)), 1),
        )
        base_reliability_max = base_weight.max(axis=(-2, -1))
        for ratio_index, ratio in enumerate(svd_support.ratios):
            ratio_name = _ratio_name(ratio)
            valid = np.asarray(
                svd_support.mask[:, :, radar, ratio_index], dtype=np.float32
            )
            support = svd_nonnegative[:, :, radar, ratio_index]
            weight = base_weight[:, None, :, :]
            weighted = support * weight
            weight_sum = np.broadcast_to(
                base_weight_sum[:, None], support.shape[:2]
            )
            weighted_mean = np.divide(
                weighted.sum(axis=(-2, -1)),
                np.maximum(weight_sum, 1.0e-12),
                out=np.zeros(support.shape[:2], dtype=np.float32),
                where=weight_sum > 0,
            )
            maximum = support.max(axis=(-2, -1))
            support_sum = weighted.sum(axis=(-2, -1))
            distribution = np.divide(
                weighted,
                np.maximum(support_sum[..., None, None], 1.0e-12),
                out=np.zeros_like(weighted),
                where=support_sum[..., None, None] > 0,
            )
            entropy = -(
                distribution * np.log(np.maximum(distribution, 1.0e-20))
            ).sum(axis=(-2, -1), dtype=np.float64) / np.log(
                support.shape[-2] * support.shape[-1]
            )
            center = svd_support.centers_bpm[:, :, ratio_index]
            distance = np.abs(center[..., None, None] - peaks[:, None, radar])
            mean_distance = np.divide(
                (distance * weight).sum(axis=(-2, -1)),
                np.maximum(weight_sum, 1.0e-12),
                out=np.zeros(support.shape[:2], dtype=np.float32),
                where=weight_sum > 0,
            )
            masked_distance = np.where(
                radar_component_mask[:, None], distance, np.inf
            )
            closest_distance = masked_distance.min(axis=(-2, -1))
            closest_distance = np.where(np.isfinite(closest_distance), closest_distance, 0.0)
            prefix = f"svd_radar{radar + 1}_{ratio_name}"
            append(prefix + "_reliability_weighted_mean", weighted_mean * valid)
            append(prefix + "_max", maximum * valid)
            append(prefix + "_component_entropy", entropy * valid)
            append(prefix + "_reliability_weighted_peak_distance_bpm", mean_distance * valid)
            append(prefix + "_closest_peak_distance_bpm", closest_distance * valid)
            append(
                prefix + "_reliability_mean",
                np.broadcast_to(base_reliability_mean[:, None], support.shape[:2])
                * valid,
            )
            append(
                prefix + "_reliability_max",
                np.broadcast_to(base_reliability_max[:, None], support.shape[:2])
                * valid,
            )

    features = np.stack(arrays, axis=-1).astype(np.float32, copy=False)
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
    features *= candidate_mask[..., None]
    return NodeFeatureBatch(
        features=_readonly(features),
        mask=_readonly(candidate_mask.copy()),
        feature_names=tuple(names),
    )


def iter_compact_node_feature_batches(
    rf_maps: np.ndarray,
    rf_frequencies_hz: np.ndarray,
    svd_spectra: np.ndarray,
    svd_attributes: np.ndarray,
    svd_frequencies_hz: np.ndarray,
    candidates: CandidateBank,
    *,
    explicit_radar_mask: np.ndarray | None = None,
    ratios: Sequence[float] = HARMONIC_RATIOS,
    batch_size: int = 8,
    rf_top_k: int = 2,
    svd_components: int = 6,
    proposer_node_features: np.ndarray | None = None,
    proposer_feature_names: Sequence[str] | None = None,
    include_source_confidence: bool = False,
) -> Iterator[HarmonicFeatureBatch]:
    """Yield raw support and compact nodes without cohort-sized allocation."""

    rows = candidates.rows
    if int(batch_size) < 1:
        raise ValueError("batch_size must be positive")
    if not (
        np.asarray(rf_maps).shape[0]
        == np.asarray(svd_spectra).shape[0]
        == np.asarray(svd_attributes).shape[0]
        == rows
    ):
        raise ValueError("candidate and evidence row counts do not match")
    if explicit_radar_mask is not None and np.asarray(explicit_radar_mask).shape != (
        rows,
        np.asarray(rf_maps).shape[1],
    ):
        raise ValueError("explicit radar mask shape mismatch")
    if int(svd_components) not in (6, 12):
        raise ValueError("svd_components must be 6 or 12")
    if proposer_node_features is not None:
        proposer_shape = np.asarray(proposer_node_features).shape
        if len(proposer_shape) != 3 or proposer_shape[:2] != (
            rows,
            candidates.max_candidates,
        ):
            raise ValueError("proposer node features must have shape [N,K,P]")
    for start in range(0, rows, int(batch_size)):
        stop = min(rows, start + int(batch_size))
        row_slice = slice(start, stop)
        rf_batch = np.asarray(rf_maps[row_slice])
        svd_batch = np.asarray(svd_spectra[row_slice])
        attribute_batch = np.asarray(svd_attributes[row_slice])
        explicit = (
            None
            if explicit_radar_mask is None
            else np.asarray(explicit_radar_mask[row_slice], dtype=bool)
        )
        selected_svd_for_mask = svd_batch[
            :, :, VERIFIED_SVD_VARIANT_INDICES, : int(svd_components), :
        ]
        selected_attributes_for_mask = attribute_batch[
            :, :, VERIFIED_SVD_VARIANT_INDICES, : int(svd_components), :
        ]
        joint = resolve_joint_radar_mask(
            rf_batch,
            selected_svd_for_mask,
            svd_attributes=selected_attributes_for_mask,
            explicit_mask=explicit,
        )
        candidate_batch = candidates.subset(row_slice)
        rf_support = sample_rf_harmonic_support(
            rf_batch,
            rf_frequencies_hz,
            candidate_batch,
            radar_mask=joint,
            ratios=ratios,
        )
        svd_support = sample_svd_harmonic_support(
            svd_batch,
            attribute_batch,
            svd_frequencies_hz,
            candidate_batch,
            radar_mask=joint,
            ratios=ratios,
            components=int(svd_components),
        )
        nodes = build_compact_node_features(
            candidate_batch,
            rf_support,
            svd_support,
            rf_top_k=rf_top_k,
            proposer_node_features=(
                None
                if proposer_node_features is None
                else np.asarray(proposer_node_features[row_slice])
            ),
            proposer_feature_names=proposer_feature_names,
            include_source_confidence=include_source_confidence,
        )
        yield HarmonicFeatureBatch(
            row_slice=row_slice,
            candidates=candidate_batch,
            rf_support=rf_support,
            svd_support=svd_support,
            nodes=nodes,
        )


def _semantic_value(field: str, value: Any) -> str | int | float:
    if field in _SEMANTIC_STRING_FIELDS:
        result = str(value)
        if not result or result.lower() == "nan":
            raise ValueError(f"semantic field {field} contains an empty value")
        return result
    try:
        numeric = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"semantic field {field} is not numeric") from error
    if not np.isfinite(numeric):
        raise ValueError(f"semantic field {field} contains a non-finite value")
    if field in _SEMANTIC_INTEGER_FIELDS:
        rounded = int(round(numeric))
        if numeric != rounded:
            raise ValueError(f"semantic integer field {field} is fractional")
        return rounded
    # Microsecond-level canonicalization is tighter than the cache's 10 Hz
    # alignment tolerance while surviving a CSV text round-trip.
    return round(numeric, 6)


def semantic_row_binding_sha256(
    metadata: Any,
    *,
    fields: Sequence[str] = SEMANTIC_ROW_FIELDS,
) -> str:
    """Hash ordered semantic row meaning, not merely a set of cache indices."""

    selected = tuple(str(field) for field in fields)
    if not selected or len(set(selected)) != len(selected):
        raise ValueError("semantic fields must be non-empty and unique")
    columns = {field: _column(metadata, field) for field in selected}
    lengths = {len(column) for column in columns.values()}
    if len(lengths) != 1:
        raise ValueError("semantic row columns have inconsistent lengths")
    rows = lengths.pop()
    records: list[list[str | int | float]] = []
    for row in range(rows):
        records.append([_semantic_value(field, columns[field][row]) for field in selected])
    encoded = json.dumps(
        {"fields": selected, "rows": records},
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def assert_exact_semantic_row_binding(
    source: Any,
    expected: Any,
    *,
    label: str = "source",
    fields: Sequence[str] = SEMANTIC_ROW_FIELDS,
) -> str:
    """Fail closed if row order or any semantic field differs.

    The returned digest can be recorded in a provenance manifest.  It is safe
    for lineage/audit use, but semantic fields remain outside the model-forward
    allow-list.
    """

    try:
        source_digest = semantic_row_binding_sha256(source, fields=fields)
        expected_digest = semantic_row_binding_sha256(expected, fields=fields)
    except (KeyError, ValueError) as error:
        raise RuntimeError(f"{label} semantic row binding cannot be verified: {error}") from error
    if source_digest != expected_digest:
        raise RuntimeError(f"{label} semantic row binding mismatch")
    indices = _column(source, "cache_index")
    if len(np.unique(indices)) != len(indices):
        raise RuntimeError(f"{label} semantic row binding has duplicate cache_index")
    return expected_digest


__all__ = [
    "CANDIDATE_METADATA_FIELDS",
    "CANDIDATE_SOURCE_NAMES",
    "DEFAULT_MAX_CANDIDATES",
    "DEFAULT_MERGE_RADIUS_BPM",
    "FORBIDDEN_TARGET_QC_FIELDS",
    "FORWARD_METADATA_ALLOWLIST",
    "HARMONIC_RATIOS",
    "RF_BRANCH_NAMES",
    "RF_RANGE_BINS_PER_BRANCH",
    "RR_MAX_BPM",
    "RR_MIN_BPM",
    "SEMANTIC_ROW_FIELDS",
    "VERIFIED_SVD_VARIANT_INDICES",
    "VERIFIED_SVD_VARIANT_NAMES",
    "CandidateBank",
    "CandidateSource",
    "HarmonicFeatureBatch",
    "NodeFeatureBatch",
    "RFHarmonicSupport",
    "SVDHarmonicSupport",
    "assert_exact_semantic_row_binding",
    "build_candidate_bank",
    "build_compact_node_features",
    "candidate_bank_from_metadata",
    "iter_compact_node_feature_batches",
    "reshape_rf_branches",
    "resolve_joint_radar_mask",
    "sample_rf_harmonic_support",
    "sample_svd_harmonic_support",
    "select_forward_metadata",
    "semantic_row_binding_sha256",
    "triangular_sample_native_grid",
]
