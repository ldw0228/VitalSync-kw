"""Measured multi-radar timeline fusion with causal, content-bound resampling."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntFlag
import hashlib
import json
from typing import Any, Sequence

import numpy as np


class RadarTimingError(ValueError):
    """Raised when a common measured radar clock cannot be reconstructed safely."""


@dataclass(frozen=True, slots=True)
class CommonRadarTimeline:
    origin_epoch_s: float
    times_s: np.ndarray
    summary: dict[str, Any]


CAUSAL_UNIFORM_RESAMPLE_SCHEMA_V1 = "snn_rr.causal_uniform_radar_resample.v1"
CAUSAL_UNIFORM_INVALID_REASON_SCHEMA_V1 = (
    "snn_rr.causal_uniform_radar_invalid_reason_mask.v1"
)
_FIXED_POINT_TICKS_PER_SECOND = 1_000_000_000
_CANONICAL_ARRAY_HASH_SCHEMA = "snn_rr.canonical_ndarray_sha256.v1"


class CausalUniformInvalidReasonV1(IntFlag):
    """Independent invalidity causes for one radar view/output interval.

    Values are stable serialized bits, not ordinal labels.  Multiple causes
    are preserved with bitwise OR; a zero mask is the sole valid state.
    """

    EMPTY_INTERVAL = 1 << 0
    TEMPORAL_GAP = 1 << 1
    FRAME_SEQUENCE_GAP = 1 << 2
    TIMESTAMP_PLATEAU = 1 << 3
    NONFINITE_PAYLOAD = 1 << 4


_INVALID_REASON_FLAGS_V1: tuple[
    tuple[str, CausalUniformInvalidReasonV1, str], ...
] = (
    (
        "empty_interval",
        CausalUniformInvalidReasonV1.EMPTY_INTERVAL,
        "the half-open output interval contains no retained frame",
    ),
    (
        "temporal_gap",
        CausalUniformInvalidReasonV1.TEMPORAL_GAP,
        "the maximum support gap, including interval edges, exceeds max_gap_s",
    ),
    (
        "frame_sequence_gap",
        CausalUniformInvalidReasonV1.FRAME_SEQUENCE_GAP,
        "the first frame after a non-unit frame-sequence edge lies in the interval",
    ),
    (
        "timestamp_plateau",
        CausalUniformInvalidReasonV1.TIMESTAMP_PLATEAU,
        "a retained frame on a duplicate measured-timestamp edge lies in the interval",
    ),
    (
        "nonfinite_payload",
        CausalUniformInvalidReasonV1.NONFINITE_PAYLOAD,
        (
            "at least one selected payload component is non-finite, or the "
            "finite interval mean cannot be represented as finite float32 "
            "after stable scaled-float64 aggregation"
        ),
    ),
)
_INVALID_REASON_KNOWN_BITS_V1 = sum(
    int(flag) for _, flag, _ in _INVALID_REASON_FLAGS_V1
)
_INVALID_REASON_SEMANTICS_DOCUMENT_V1: dict[str, Any] = {
    "schema_version": CAUSAL_UNIFORM_INVALID_REASON_SCHEMA_V1,
    "mask_dtype": "uint8",
    "mask_layout": ["radar_view", "output_interval"],
    "validity_semantics": "valid_mask == (invalid_reason_mask == 0)",
    "overlap_policy": "independent_causes_preserved_by_bitwise_or",
    "invalid_output_policy": "all_components_exact_positive_zero",
    "valid_output_policy": "all_components_finite_float32",
    "aggregation_policy": (
        "scaled_float64_mean_then_checked_float32_conversion"
    ),
    "empty_interval_temporal_gap_policy": (
        "empty intervals also carry temporal_gap when their edge-to-edge support "
        "gap exceeds max_gap_s"
    ),
    "timestamp_plateau_policy": "retain_measured_timestamps_and_payload; never_interpolate",
    "flags": [
        {
            "name": name,
            "value": int(flag),
            "bit_index": int(flag).bit_length() - 1,
            "condition": condition,
        }
        for name, flag, condition in _INVALID_REASON_FLAGS_V1
    ],
    "known_bits_mask": _INVALID_REASON_KNOWN_BITS_V1,
}
CAUSAL_UNIFORM_INVALID_REASON_SEMANTICS_SHA256_V1 = hashlib.sha256(
    json.dumps(
        _INVALID_REASON_SEMANTICS_DOCUMENT_V1,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
).hexdigest()


def _strict_finite_real(value: Any, label: str) -> float:
    """Accept a real numeric scalar without laundering bool/string values."""

    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise RadarTimingError(f"{label} must be a finite real number")
    result = float(value)
    if not np.isfinite(result):
        raise RadarTimingError(f"{label} must be a finite real number")
    return result


def _strict_real_array(value: Any, label: str) -> np.ndarray:
    """Reject bool, complex, string, and object arrays before numeric casts."""

    array = np.asarray(value)
    try:
        valid_dtype = bool(
            np.issubdtype(array.dtype, np.number)
            and not np.issubdtype(array.dtype, np.bool_)
            and not np.issubdtype(array.dtype, np.complexfloating)
        )
    except TypeError:
        valid_dtype = False
    if not valid_dtype:
        raise RadarTimingError(f"{label} must contain real numeric values")
    return array


def _strict_int64_array(value: Any, label: str) -> np.ndarray:
    """Return exact int64 coordinates without bool/float/string laundering."""

    array = np.asarray(value)
    try:
        if (
            np.issubdtype(array.dtype, np.bool_)
            or np.issubdtype(array.dtype, np.complexfloating)
            or not np.issubdtype(array.dtype, np.number)
        ):
            raise ValueError("dtype is not numeric integral")
        int64_info = np.iinfo(np.int64)
        if np.issubdtype(array.dtype, np.floating):
            raise ValueError("floating frame counters are forbidden")
        if np.issubdtype(array.dtype, np.unsignedinteger):
            if array.size and int(array.max()) > int64_info.max:
                raise OverflowError("unsigned values exceed int64")
        elif np.issubdtype(array.dtype, np.signedinteger):
            if array.size and (
                int(array.min()) < int64_info.min
                or int(array.max()) > int64_info.max
            ):
                raise OverflowError("signed values exceed int64")
        return array.astype(np.int64, copy=False)
    except (TypeError, ValueError, OverflowError) as exc:
        raise RadarTimingError(f"{label} is not integral") from exc


def _strict_nonnegative_evidence_count(value: Any, label: str) -> int:
    """Read a serialized count without accepting bool/float/string aliases."""

    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise RadarTimingError(f"{label} must be an exact integer")
    result = int(value)
    if result < 0:
        raise RadarTimingError(f"{label} cannot be negative")
    return result


@dataclass(frozen=True, slots=True)
class CausalUniformRadarResampleV1:
    """Uniform, right-edge-timestamped radar samples and their validity mask.

    ``values[:, k]`` is the checked float32 arithmetic mean of the measured frames whose
    timestamps fall in ``[times_s[k] - interval_s, times_s[k])``.  The
    half-open interval makes the causal boundary explicit: a frame stamped
    exactly at an output edge belongs to the following output, never the one
    ending at that edge.  Invalid intervals are exact zero and must be
    distinguished from real zeros with ``valid_mask``.
    """

    origin_epoch_s: float
    times_s: np.ndarray
    values: np.ndarray
    valid_mask: np.ndarray
    sample_counts: np.ndarray
    interval_s: float
    summary: dict[str, Any]
    # Added additively so callers constructing the historical seven-field
    # result remain source-compatible.  Results produced by the current
    # resampler always carry the versioned uint8 mask; ``None`` denotes a
    # legacy result with no invalid-reason authority.
    invalid_reason_mask: np.ndarray | None = None


def causal_uniform_invalid_reason_semantics_v1() -> dict[str, Any]:
    """Return an independent copy of the stable invalid-reason semantics."""

    return json.loads(
        json.dumps(
            _INVALID_REASON_SEMANTICS_DOCUMENT_V1,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    )


def canonical_ndarray_sha256(value: np.ndarray) -> str:
    """Hash ndarray content with canonical shape, dtype, byte order, and layout.

    The hash is independent of host endianness and memory strides.  Numeric
    bit patterns are otherwise preserved, including signed zero and NaN
    payloads, so an audit can detect a one-bit change in a bound transform.
    """

    array = np.asarray(value)
    if array.dtype.hasobject:
        raise RadarTimingError("object arrays cannot be content-bound")
    canonical_dtype = array.dtype.newbyteorder("<")
    canonical = np.ascontiguousarray(array.astype(canonical_dtype, copy=False))
    header = json.dumps(
        {
            "schema_version": _CANONICAL_ARRAY_HASH_SCHEMA,
            "dtype": canonical.dtype.str,
            "shape": list(canonical.shape),
        },
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    digest = hashlib.sha256()
    digest.update(len(header).to_bytes(8, byteorder="little", signed=False))
    digest.update(header)
    digest.update(memoryview(canonical).cast("B"))
    return digest.hexdigest()


def canonical_ndarray_sequence_sha256(values: Sequence[np.ndarray]) -> str:
    """Hash an ordered, variably shaped sequence of numeric ndarrays."""

    document = {
        "schema_version": "snn_rr.canonical_ndarray_sequence_sha256.v1",
        "items": [canonical_ndarray_sha256(item) for item in values],
    }
    return hashlib.sha256(
        json.dumps(
            document,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _invalid_reason_array_evidence_v1(
    invalid_reason_mask: np.ndarray,
    valid_mask: np.ndarray,
    output_values: np.ndarray,
    sample_counts: np.ndarray,
) -> dict[str, Any]:
    """Validate exact array-level reason invariants and return count evidence."""

    reasons = np.asarray(invalid_reason_mask)
    valid = np.asarray(valid_mask)
    output = np.asarray(output_values)
    counts = np.asarray(sample_counts)
    if reasons.dtype != np.dtype(np.uint8) or reasons.ndim != 2:
        raise RadarTimingError("invalid_reason_mask must be a uint8 [view, output] array")
    if valid.dtype != np.dtype(bool) or valid.shape != reasons.shape:
        raise RadarTimingError("valid_mask must be boolean and match invalid_reason_mask")
    if counts.shape != reasons.shape or not np.issubdtype(counts.dtype, np.integer):
        raise RadarTimingError("sample_counts must be integral and match invalid_reason_mask")
    if output.ndim < 2 or output.shape[:2] != reasons.shape:
        raise RadarTimingError("output values do not match invalid_reason_mask")
    if np.any(counts < 0):
        raise RadarTimingError("sample_counts cannot be negative")

    unknown_bits = np.bitwise_and(
        reasons,
        np.uint8(0xFF ^ _INVALID_REASON_KNOWN_BITS_V1),
    )
    if np.any(unknown_bits != 0):
        raise RadarTimingError("invalid_reason_mask contains an unknown reason bit")

    reason_union = reasons != 0
    if not np.array_equal(reason_union, ~valid):
        raise RadarTimingError("invalid reason union does not exactly equal ~valid_mask")

    empty_bits = (
        np.bitwise_and(
            reasons,
            np.uint8(CausalUniformInvalidReasonV1.EMPTY_INTERVAL),
        )
        != 0
    )
    if not np.array_equal(empty_bits, counts == 0):
        raise RadarTimingError("empty-interval reason does not exactly match sample_counts")

    invalid_values = output[reason_union]
    if invalid_values.size and (
        np.any(invalid_values != 0.0) or np.any(np.signbit(invalid_values))
    ):
        raise RadarTimingError(
            "invalid resampling cells must contain exact positive zero"
        )
    valid_values = output[valid]
    if valid_values.size and not np.isfinite(valid_values).all():
        raise RadarTimingError(
            "valid resampling cells must contain only finite values"
        )

    per_reason_counts: dict[str, int] = {}
    multiplicity = np.zeros(reasons.shape, dtype=np.uint8)
    for name, flag, _ in _INVALID_REASON_FLAGS_V1:
        present = np.bitwise_and(reasons, np.uint8(flag)) != 0
        per_reason_counts[name] = int(np.count_nonzero(present))
        multiplicity += present.astype(np.uint8, copy=False)
    invalid_count = int(np.count_nonzero(reason_union))
    return {
        "invalid_interval_count": invalid_count,
        "valid_interval_count": int(reasons.size - invalid_count),
        "per_reason_interval_counts": per_reason_counts,
        "overlap_interval_count": int(np.count_nonzero(multiplicity > 1)),
        "maximum_reason_multiplicity": int(np.max(multiplicity, initial=0)),
        "all_invalid_explained": bool(
            np.array_equal(reason_union, ~valid)
            and not np.any(unknown_bits)
        ),
        "invalid_outputs_exact_positive_zero": True,
    }


def _stable_float32_interval_mean_v1(selected: np.ndarray) -> np.ndarray | None:
    """Return a finite float32 mean, or ``None`` when it is unrepresentable.

    NumPy's float32 mean first forms a float32 sum, so several finite values
    near ``finfo(float32).max`` can overflow even when their arithmetic mean
    is representable.  Scaling each component before a float64 accumulation
    prevents that intermediate overflow.  The conversion back to the public
    float32 result is allowed only after an explicit representability check.
    """

    values = np.asarray(selected)
    if values.ndim < 1 or values.shape[0] <= 0 or not np.isfinite(values).all():
        return None
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        working = values.astype(np.float64, copy=False)
        if not np.isfinite(working).all():
            return None
        scale = np.max(np.abs(working), axis=0)
        denominator = np.where(scale == 0.0, 1.0, scale)
        normalized_mean = np.mean(
            working / denominator,
            axis=0,
            dtype=np.float64,
        )
        representability_limit = np.float64(np.finfo(np.float32).max) / denominator
        if (
            not np.isfinite(normalized_mean).all()
            or np.any(np.abs(normalized_mean) > representability_limit)
        ):
            return None
        mean64 = normalized_mean * scale
        if not np.isfinite(mean64).all():
            return None
        mean32 = np.asarray(mean64, dtype=np.float32)
    if not np.isfinite(mean32).all():
        return None
    return mean32


def _validate_frame_accounting_documents_v1(
    *,
    per_view_documents: Any,
    sample_counts: np.ndarray,
) -> None:
    """Fail closed unless each retained frame has one support category."""

    if not isinstance(per_view_documents, list) or len(per_view_documents) != len(
        sample_counts
    ):
        raise RadarTimingError("per-view frame-accounting evidence is incomplete")
    category_names = (
        "outside_common_intersection_prefix_frame_count",
        "leading_partial_edge_frame_count",
        "assigned_to_output_intervals_frame_count",
        "trailing_partial_edge_frame_count",
        "outside_common_intersection_suffix_frame_count",
    )

    for view_index, document in enumerate(per_view_documents):
        if not isinstance(document, dict):
            raise RadarTimingError("per-view timing evidence must be a mapping")
        accounting = document.get("frame_accounting")
        if not isinstance(accounting, dict) or accounting.get("schema_version") != (
            "snn_rr.radar_frame_accounting.v1"
        ):
            raise RadarTimingError("frame-accounting schema is missing or unsupported")
        categories = accounting.get("categories")
        if not isinstance(categories, dict) or set(categories) != set(category_names):
            raise RadarTimingError("frame-accounting categories are incomplete")
        try:
            category_values = [
                _strict_nonnegative_evidence_count(
                    categories[name], f"frame-accounting category {name}"
                )
                for name in category_names
            ]
            retained_count = _strict_nonnegative_evidence_count(
                accounting["retained_input_frame_count"],
                "retained input frame count",
            )
            category_sum = _strict_nonnegative_evidence_count(
                accounting["category_sum"], "frame-accounting category sum"
            )
            unaccounted = _strict_nonnegative_evidence_count(
                accounting["unaccounted_payload_frame_count"],
                "unaccounted payload frame count",
            )
            document_unaccounted = _strict_nonnegative_evidence_count(
                document["unaccounted_payload_frame_count"],
                "per-view unaccounted payload frame count",
            )
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            raise RadarTimingError("frame-accounting counts are invalid") from exc
        expected_sum = sum(category_values)
        assigned = int(
            np.sum(np.asarray(sample_counts[view_index]), dtype=np.int64)
        )
        if not (
            expected_sum == category_sum == retained_count
            and unaccounted == 0
            and category_values[2] == assigned
            and accounting.get("categories_disjoint") is True
            and accounting.get("coverage_complete") is True
            and accounting.get("assigned_count_matches_sample_counts") is True
            and document_unaccounted == 0
        ):
            raise RadarTimingError("not every retained frame is exactly accounted")


def validate_causal_uniform_invalid_reason_contract_v1(
    result: CausalUniformRadarResampleV1,
) -> dict[str, Any]:
    """Validate a produced result's v1 reason, hash, and accounting evidence.

    Historical results without ``invalid_reason_mask`` remain readable through
    :class:`CausalUniformRadarResampleV1`, but deliberately fail this stronger
    authority check rather than being upgraded by inference.
    """

    if not isinstance(result, CausalUniformRadarResampleV1):
        raise RadarTimingError("invalid-reason validation requires a resample result")
    if result.invalid_reason_mask is None:
        raise RadarTimingError("legacy resample result has no invalid-reason contract")
    if not isinstance(result.summary, dict):
        raise RadarTimingError("resample summary must be a mapping")

    evidence = _invalid_reason_array_evidence_v1(
        result.invalid_reason_mask,
        result.valid_mask,
        result.values,
        result.sample_counts,
    )
    contract = result.summary.get("invalid_reason_contract")
    if not isinstance(contract, dict):
        raise RadarTimingError("invalid-reason contract evidence is missing")
    expected_semantics = causal_uniform_invalid_reason_semantics_v1()
    if contract.get("semantics") != expected_semantics:
        raise RadarTimingError("invalid-reason semantics do not match v1")
    encoded_semantics = json.dumps(
        contract["semantics"],
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    if (
        hashlib.sha256(encoded_semantics).hexdigest()
        != CAUSAL_UNIFORM_INVALID_REASON_SEMANTICS_SHA256_V1
        or contract.get("semantics_sha256")
        != CAUSAL_UNIFORM_INVALID_REASON_SEMANTICS_SHA256_V1
    ):
        raise RadarTimingError("invalid-reason semantics hash does not match v1")

    reason_sha256 = canonical_ndarray_sha256(result.invalid_reason_mask)
    content_hashes = result.summary.get("content_hashes")
    if not isinstance(content_hashes, dict):
        raise RadarTimingError("resample content hashes are missing")
    expected_array_hashes = {
        "output_times_sha256": canonical_ndarray_sha256(result.times_s),
        "output_values_sha256": canonical_ndarray_sha256(result.values),
        "valid_mask_sha256": canonical_ndarray_sha256(result.valid_mask),
        "sample_counts_sha256": canonical_ndarray_sha256(result.sample_counts),
        "invalid_reason_mask_sha256": reason_sha256,
        "invalid_reason_semantics_sha256": (
            CAUSAL_UNIFORM_INVALID_REASON_SEMANTICS_SHA256_V1
        ),
    }
    for key, expected in expected_array_hashes.items():
        if content_hashes.get(key) != expected:
            raise RadarTimingError(f"resample content hash mismatch: {key}")
    if contract.get("invalid_reason_mask_sha256") != reason_sha256:
        raise RadarTimingError("invalid-reason contract mask hash mismatch")
    for key in (
        "invalid_interval_count",
        "valid_interval_count",
        "overlap_interval_count",
        "maximum_reason_multiplicity",
    ):
        recorded = _strict_nonnegative_evidence_count(
            contract.get(key), f"invalid-reason evidence {key}"
        )
        if recorded != evidence[key]:
            raise RadarTimingError(f"invalid-reason contract evidence mismatch: {key}")
    for key in ("all_invalid_explained", "invalid_outputs_exact_positive_zero"):
        if contract.get(key) is not evidence[key]:
            raise RadarTimingError(f"invalid-reason contract evidence mismatch: {key}")
    recorded_reason_counts = contract.get("per_reason_interval_counts")
    if not isinstance(recorded_reason_counts, dict) or set(recorded_reason_counts) != {
        name for name, _, _ in _INVALID_REASON_FLAGS_V1
    }:
        raise RadarTimingError("invalid-reason aggregate counts do not match v1")
    for name, expected_count in evidence["per_reason_interval_counts"].items():
        if _strict_nonnegative_evidence_count(
            recorded_reason_counts.get(name),
            f"invalid-reason aggregate count {name}",
        ) != int(expected_count):
            raise RadarTimingError(
                f"invalid-reason contract evidence mismatch: {name}"
            )

    transform_evidence_sha256 = hashlib.sha256(
        json.dumps(
            content_hashes,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    if result.summary.get("transform_evidence_sha256") != transform_evidence_sha256:
        raise RadarTimingError("transform evidence hash does not bind content hashes")

    per_view = result.summary.get("per_view")
    _validate_frame_accounting_documents_v1(
        per_view_documents=per_view,
        sample_counts=np.asarray(result.sample_counts),
    )
    assert isinstance(per_view, list)
    legacy_count_keys = {
        "empty_interval": "empty_interval_count",
        "temporal_gap": "temporal_gap_interval_count",
        "frame_sequence_gap": "sequence_gap_interval_count",
        "timestamp_plateau": "timestamp_plateau_interval_count",
        "nonfinite_payload": "nonfinite_interval_count",
    }
    reasons = np.asarray(result.invalid_reason_mask)
    for view_index, document in enumerate(per_view):
        view_reasons = reasons[view_index]
        recorded_valid_count = _strict_nonnegative_evidence_count(
            document.get("valid_output_count"), "per-view valid output count"
        )
        recorded_invalid_count = _strict_nonnegative_evidence_count(
            document.get("invalid_output_count"), "per-view invalid output count"
        )
        if recorded_valid_count != int(
            np.count_nonzero(view_reasons == 0)
        ) or recorded_invalid_count != int(np.count_nonzero(view_reasons != 0)):
            raise RadarTimingError("per-view valid/invalid counts do not match reasons")
        recorded_reason_counts = document.get("invalid_reason_interval_counts")
        if not isinstance(recorded_reason_counts, dict):
            raise RadarTimingError("per-view invalid-reason counts are missing")
        for name, flag, _ in _INVALID_REASON_FLAGS_V1:
            expected_count = int(
                np.count_nonzero(np.bitwise_and(view_reasons, np.uint8(flag)))
            )
            if (
                _strict_nonnegative_evidence_count(
                    document.get(legacy_count_keys[name]),
                    f"per-view legacy reason count {name}",
                )
                != expected_count
                or _strict_nonnegative_evidence_count(
                    recorded_reason_counts.get(name),
                    f"per-view reason count {name}",
                )
                != expected_count
            ):
                raise RadarTimingError(f"per-view reason count mismatch: {name}")
        if set(recorded_reason_counts) != {
            name for name, _, _ in _INVALID_REASON_FLAGS_V1
        }:
            raise RadarTimingError("per-view invalid-reason count keys do not match v1")

    all_views_valid = np.all(np.asarray(result.valid_mask), axis=0)
    if _strict_nonnegative_evidence_count(
        result.summary.get("all_views_valid_interval_count"),
        "all-views-valid interval count",
    ) != int(
        np.count_nonzero(all_views_valid)
    ) or _strict_nonnegative_evidence_count(
        result.summary.get("any_view_invalid_interval_count"),
        "any-view-invalid interval count",
    ) != int(
        np.count_nonzero(~all_views_valid)
    ):
        raise RadarTimingError("cross-view validity counts do not match reasons")
    return {
        "schema_version": CAUSAL_UNIFORM_INVALID_REASON_SCHEMA_V1,
        "semantics_sha256": CAUSAL_UNIFORM_INVALID_REASON_SEMANTICS_SHA256_V1,
        "invalid_reason_mask_sha256": reason_sha256,
        **evidence,
    }


def repair_common_timestamp_plateaus(
    measured_times_s: np.ndarray,
    frame_sequences: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Classify duplicate-clock plateaus without inventing timestamps.

    Historical code interpolated a plateau between past and future anchors.
    That made an earlier timestamp depend on a later observation.  This
    routine now preserves every measured timestamp and describes each tied
    group.  Callers with a structural validity mask can invalidate the
    affected half-open interval; callers without one must fail closed.
    """

    measured = _strict_real_array(
        measured_times_s, "common radar timestamps"
    ).astype(np.float64, copy=False)
    sequences = _strict_int64_array(frame_sequences, "frame_sequences")
    if measured.ndim != 1 or measured.size < 2:
        raise RadarTimingError("common radar time must contain at least two frames")
    if sequences.ndim != 2 or sequences.shape[1] != measured.size:
        raise RadarTimingError("frame_sequences must have shape [radar, frame]")
    delta = np.diff(measured)
    if np.any(delta < 0):
        first = int(np.flatnonzero(delta < 0)[0])
        raise RadarTimingError(f"fused radar time moves backwards at frame {first}")
    tied_edges = np.flatnonzero(delta == 0)
    if not tied_edges.size:
        return measured.copy(), {
            "timestamp_plateau_count": 0,
            "measured_tie_edge_count": 0,
            "reconstructed_frame_count": 0,
            "maximum_timestamp_correction_s": 0.0,
            "reconstruction_method": "none",
            "plateaus": [],
        }
    positive = delta[delta > 0]
    if not positive.size:
        raise RadarTimingError("common radar timestamps contain no positive clock interval")
    nominal_period = float(np.median(positive))
    groups = np.split(tied_edges, np.flatnonzero(np.diff(tied_edges) > 1) + 1)
    plateau_documents: list[dict[str, Any]] = []
    for group in groups:
        first_frame = int(group[0])
        last_frame = int(group[-1] + 1)
        sequence_window = sequences[:, first_frame : last_frame + 1].astype(
            np.int64, copy=False
        )
        plateau_documents.append(
            {
                "first_affected_frame": first_frame,
                "last_affected_frame": last_frame,
                "measured_time_s": float(measured[first_frame]),
                "duplicate_edge_count": int(len(group)),
                "sequence_contiguous": bool(
                    np.all(np.diff(sequence_window, axis=1) == 1)
                ),
                "at_leading_boundary": first_frame == 0,
                "at_trailing_boundary": last_frame == measured.size - 1,
            }
        )
    return measured.copy(), {
        "timestamp_plateau_count": len(groups),
        "measured_tie_edge_count": int(tied_edges.size),
        "reconstructed_frame_count": 0,
        "maximum_timestamp_correction_s": 0.0,
        "reconstruction_method": "none_structural_mask_required",
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
    relative = [
        _strict_real_array(item, f"radar view {index} timestamps").astype(
            np.float64, copy=False
        )
        for index, item in enumerate(relative_times_s)
    ]
    sequences = [
        _strict_int64_array(item, f"radar view {index} frame sequence")
        for index, item in enumerate(frame_sequences)
    ]
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
    starts = _strict_real_array(
        start_epochs_s, "radar start epochs"
    ).astype(np.float64, copy=False)
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
    if plateau_summary["timestamp_plateau_count"]:
        raise RadarTimingError(
            "common timeline cannot represent structurally masked timestamp plateaus"
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


def _exact_millisecond_start_offsets_v1(
    starts_s: np.ndarray,
) -> tuple[float, np.ndarray, np.ndarray, dict[str, Any]]:
    """Subtract millisecond start epochs without large-float cancellation."""

    scaled = starts_s * 1000.0
    rounded = np.rint(scaled)
    reconstructed = rounded / 1000.0
    tolerance_s = np.maximum(
        1e-12,
        4.0 * np.spacing(np.maximum(np.abs(starts_s), 1.0)),
    )
    fits_i64 = bool(
        np.all(np.abs(rounded) <= np.iinfo(np.int64).max // 2)
    )
    if fits_i64 and np.all(np.abs(starts_s - reconstructed) <= tolerance_s):
        starts_ms = rounded.astype(np.int64)
        ordered = np.sort(starts_ms)
        middle = len(ordered) // 2
        if len(ordered) % 2:
            twice_origin_ms = 2 * int(ordered[middle])
        else:
            twice_origin_ms = int(ordered[middle - 1]) + int(ordered[middle])
        # Work in half-millisecond integer units until the final conversion.
        twice_offsets_ms = 2 * starts_ms - twice_origin_ms
        offsets_s = twice_offsets_ms.astype(np.float64) / 2000.0
        offsets_ticks = (
            twice_offsets_ms.astype(np.int64)
            * (_FIXED_POINT_TICKS_PER_SECOND // 2000)
        )
        origin_s = float(twice_origin_ms / 2000.0)
        return origin_s, offsets_s, offsets_ticks, {
            "start_epoch_arithmetic": "integer_millisecond_fixed_point",
            "start_epoch_precision_s": 0.001,
            "start_offset_cancellation_avoided": True,
            "start_offset_coordinates_exact": True,
            "start_offset_quantization_max_abs_s": 0.0,
        }

    origin_s = float(np.median(starts_s))
    offsets_s = starts_s - origin_s
    offsets_ticks, maximum_error_s, exact = _as_fixed_point_ticks_v1(offsets_s)
    return origin_s, offsets_s, offsets_ticks, {
        "start_epoch_arithmetic": "float64_centered_then_integer_nanosecond",
        "start_epoch_precision_s": None,
        "start_offset_cancellation_avoided": False,
        "start_offset_coordinates_exact": exact,
        "start_offset_quantization_max_abs_s": maximum_error_s,
    }


def _as_fixed_point_ticks_v1(
    values_s: np.ndarray,
) -> tuple[np.ndarray, float, bool]:
    """Canonicalize finite relative seconds to a nanosecond coordinate axis.

    Conversion is element-local and unconditional: future values can change
    their own quantization evidence but can never select a different arithmetic
    path for earlier samples.
    """

    values = np.asarray(values_s, dtype=np.float64)
    scaled = values * _FIXED_POINT_TICKS_PER_SECOND
    if not np.isfinite(scaled).all() or np.any(
        np.abs(scaled) > np.iinfo(np.int64).max
    ):
        raise RadarTimingError("relative timestamps exceed fixed-point coordinate range")
    rounded = np.rint(scaled)
    ticks = rounded.astype(np.int64)
    reconstructed = ticks.astype(np.float64) / _FIXED_POINT_TICKS_PER_SECOND
    errors_s = np.abs(values - reconstructed)
    tolerance_s = np.maximum(
        1e-15,
        4.0 * np.spacing(np.maximum(np.abs(values), 1.0)),
    )
    maximum_error_s = float(np.max(errors_s, initial=0.0))
    exact = bool(np.all(errors_s <= tolerance_s))
    return ticks, maximum_error_s, exact


def _uniform_grid_bounds_ticks_v1(
    common_start: int,
    common_end: int,
    interval: int,
) -> tuple[int, np.ndarray]:
    """Integer counterpart of :func:`_uniform_grid_bounds_v1`."""

    first_left = -((-int(common_start)) // int(interval)) * int(interval)
    first_right = first_left + int(interval)
    if first_right > common_end:
        return first_left, np.empty(0, dtype=np.int64)
    count = (int(common_end) - first_right) // int(interval) + 1
    right_edges = first_right + int(interval) * np.arange(count, dtype=np.int64)
    return first_left, right_edges


def _frame_accounting_v1(
    coordinates: np.ndarray,
    *,
    common_start: int,
    first_complete_left: int,
    last_complete_right: int,
    common_end: int,
    assigned_sample_count: int,
) -> dict[str, Any]:
    """Partition every retained frame across disjoint support categories."""

    if not (
        common_start <= first_complete_left < last_complete_right <= common_end
    ):
        raise RadarTimingError("frame-accounting support boundaries are inconsistent")
    intersection_start_index = int(
        np.searchsorted(coordinates, common_start, side="left")
    )
    complete_start_index = int(
        np.searchsorted(coordinates, first_complete_left, side="left")
    )
    complete_end_index = int(
        np.searchsorted(coordinates, last_complete_right, side="left")
    )
    intersection_end_index = int(
        np.searchsorted(coordinates, common_end, side="left")
    )
    categories = {
        "outside_common_intersection_prefix_frame_count": (
            intersection_start_index
        ),
        "leading_partial_edge_frame_count": (
            complete_start_index - intersection_start_index
        ),
        "assigned_to_output_intervals_frame_count": (
            complete_end_index - complete_start_index
        ),
        "trailing_partial_edge_frame_count": (
            intersection_end_index - complete_end_index
        ),
        "outside_common_intersection_suffix_frame_count": (
            len(coordinates) - intersection_end_index
        ),
    }
    category_sum = int(sum(categories.values()))
    retained_count = int(len(coordinates))
    unaccounted = retained_count - category_sum
    assigned_matches = bool(
        categories["assigned_to_output_intervals_frame_count"]
        == int(assigned_sample_count)
    )
    disjoint = bool(
        0
        <= intersection_start_index
        <= complete_start_index
        <= complete_end_index
        <= intersection_end_index
        <= retained_count
    )
    return {
        "schema_version": "snn_rr.radar_frame_accounting.v1",
        "coordinate_semantics": "half_open_integer_nanosecond",
        "retained_input_frame_count": retained_count,
        "categories": categories,
        "category_sum": category_sum,
        "before_common_complete_support_frame_count": int(complete_start_index),
        "after_common_complete_support_frame_count": int(
            retained_count - complete_end_index
        ),
        "unaccounted_payload_frame_count": int(unaccounted),
        "categories_disjoint": disjoint,
        "coverage_complete": unaccounted == 0,
        "assigned_count_matches_sample_counts": assigned_matches,
    }


def _boundary_plateau_summary_v1(times_s: np.ndarray) -> dict[str, Any]:
    """Describe, but never discard, payload at boundary timestamp plateaus."""

    ties = np.diff(times_s) == 0
    leading = 0
    while leading < len(ties) and ties[leading]:
        leading += 1
    trailing = 0
    while trailing < len(ties) and ties[len(ties) - 1 - trailing]:
        trailing += 1
    return {
        "leading_boundary_duplicate_frame_count": int(leading),
        "trailing_boundary_duplicate_frame_count": int(trailing),
        "leading_boundary_frames_trimmed": 0,
        "trailing_boundary_frames_trimmed": 0,
        "boundary_plateau_policy": "retain_all_and_structurally_mask_affected_interval",
    }


def causal_uniform_resample_radar_views_v1(
    values_by_view: Sequence[np.ndarray],
    relative_times_s: Sequence[np.ndarray],
    start_epochs_s: Sequence[float],
    frame_sequences: Sequence[np.ndarray],
    *,
    output_hz: float = 10.0,
    max_gap_s: float = 0.050,
    gap_policy: str = "mask",
    timestamp_sources: Sequence[str] | None = None,
    require_measured_timestamps: bool = False,
) -> CausalUniformRadarResampleV1:
    """Resample independently timed radar views onto a causal uniform grid.

    Version-1 uses a scaled-float64 arithmetic mean with checked float32
    conversion over half-open intervals
    ``[right_edge - 1/output_hz, right_edge)``.  Consequently regular 40 Hz
    samples at ``0, 25, 50, 75, ...`` milliseconds reproduce the legacy
    non-overlapping four-frame mean exactly at 10 Hz, while the returned time
    is the causal *right edge* (100, 200, ... milliseconds), not the midpoint.

    Each view is validated independently before its absolute
    start is expressed relative to the median start epoch.  Only complete
    intervals in the intersection of all view coverages are returned.  An
    interval is invalid when it is empty, contains non-finite payload, has a
    timestamp coverage gap larger than ``max_gap_s``, or contains the first
    frame after a sequence discontinuity or any duplicate-clock plateau.
    ``gap_policy='mask'`` writes exact zero and exposes validity structurally;
    ``gap_policy='raise'`` fails the complete call if any output interval is
    invalid.  The additive ``invalid_reason_mask`` records every independent
    cause with the versioned uint8 bit semantics; it never infers a cause from
    a numeric zero.

    No sample stamped at or after an output edge contributes to that output.
    This is a target-free timing transform and performs no interpolation from
    future frames.
    """

    view_count = len(values_by_view)
    if not (
        view_count
        and len(relative_times_s) == view_count
        and len(start_epochs_s) == view_count
        and len(frame_sequences) == view_count
    ):
        raise RadarTimingError(
            "radar resampling inputs must have equal non-zero view counts"
        )
    if timestamp_sources is not None and len(timestamp_sources) != view_count:
        raise RadarTimingError("timestamp_sources must match the radar view count")
    if require_measured_timestamps:
        if timestamp_sources is None:
            raise RadarTimingError(
                "measured timestamp enforcement requires timestamp_sources"
            )
        rejected = [
            str(source)
            for source in timestamp_sources
            if str(source) != "meta_v13"
        ]
        if rejected:
            raise RadarTimingError(
                "measured timestamp enforcement rejected source(s): "
                + ", ".join(rejected)
            )

    rate = _strict_finite_real(output_hz, "output_hz")
    maximum_gap = _strict_finite_real(max_gap_s, "max_gap_s")
    if rate <= 0:
        raise RadarTimingError("output_hz must be finite and positive")
    if maximum_gap <= 0:
        raise RadarTimingError("max_gap_s must be finite and positive")
    if gap_policy not in {"mask", "raise"}:
        raise RadarTimingError("gap_policy must be 'mask' or 'raise'")
    interval_s = 1.0 / rate

    starts = _strict_real_array(
        start_epochs_s, "radar start epochs"
    ).astype(np.float64, copy=False)
    if starts.ndim != 1 or not np.isfinite(starts).all():
        raise RadarTimingError("radar start epochs must be a finite vector")
    origin, start_offsets_s, start_offset_ticks, start_arithmetic = (
        _exact_millisecond_start_offsets_v1(starts)
    )

    payloads: list[np.ndarray] = []
    relative_times: list[np.ndarray] = []
    integer_sequences: list[np.ndarray] = []
    per_view_documents: list[dict[str, Any]] = []
    trailing_shape: tuple[int, ...] | None = None
    for view_index, (raw_values, raw_times, raw_sequence) in enumerate(
        zip(
            values_by_view,
            relative_times_s,
            frame_sequences,
            strict=True,
        )
    ):
        payload = _strict_real_array(
            raw_values, f"radar view {view_index} payload"
        )
        times = _strict_real_array(
            raw_times, f"radar view {view_index} timestamps"
        ).astype(np.float64, copy=False)
        sequence = np.asarray(raw_sequence)
        if payload.ndim < 1 or times.ndim != 1 or sequence.ndim != 1:
            raise RadarTimingError(
                f"radar view {view_index} values/times/sequence dimensions are invalid"
            )
        if len(payload) != len(times) or len(times) != len(sequence):
            raise RadarTimingError(
                f"radar view {view_index} values/times/sequence lengths differ"
            )
        if len(times) < 2:
            raise RadarTimingError(
                f"radar view {view_index} contains fewer than two frames"
            )
        if not np.isfinite(times).all():
            raise RadarTimingError(
                f"radar view {view_index} contains non-finite timestamps"
            )
        if trailing_shape is None:
            trailing_shape = payload.shape[1:]
        elif payload.shape[1:] != trailing_shape:
            raise RadarTimingError("all radar views must have the same payload shape")
        sequence_i64 = _strict_int64_array(
            sequence, f"radar view {view_index} frame sequence"
        )
        boundary_summary = _boundary_plateau_summary_v1(times)
        measured, repair_summary = repair_common_timestamp_plateaus(
            times, sequence_i64[None, :]
        )
        repair_summary = {**repair_summary, **boundary_summary}
        delta = np.diff(measured)
        if np.any(delta < 0):
            raise RadarTimingError(
                f"radar view {view_index} time moves backwards"
            )
        positive_delta = delta[delta > 0]
        if not positive_delta.size:
            raise RadarTimingError(
                f"radar view {view_index} has no positive timestamp interval"
            )
        terminal_period = float(np.median(positive_delta))
        aligned = measured + float(start_offsets_s[view_index])
        sequence_gap_edges = np.flatnonzero(np.diff(sequence_i64) != 1)
        payloads.append(payload)
        relative_times.append(measured)
        integer_sequences.append(sequence_i64)
        per_view_documents.append(
            {
                "view_index": view_index,
                "timestamp_source": (
                    None if timestamp_sources is None else str(timestamp_sources[view_index])
                ),
                "original_frame_count": int(len(raw_times)),
                "frame_count": int(len(times)),
                "leading_boundary_frames_trimmed": boundary_summary[
                    "leading_boundary_frames_trimmed"
                ],
                "trailing_boundary_frames_trimmed": boundary_summary[
                    "trailing_boundary_frames_trimmed"
                ],
                "aligned_first_time_s": float(aligned[0]),
                "aligned_last_time_s": float(aligned[-1]),
                "median_frame_period_s": terminal_period,
                "minimum_positive_frame_period_s": float(np.min(positive_delta)),
                "maximum_positive_frame_period_s": float(np.max(positive_delta)),
                "frame_sequence_gap_count": int(sequence_gap_edges.size),
                "timestamp_repair": repair_summary,
                **boundary_summary,
            }
        )

    relative_coordinate_results = [
        _as_fixed_point_ticks_v1(item) for item in relative_times
    ]
    relative_coordinate_arrays = [item[0] for item in relative_coordinate_results]
    relative_quantization_errors = [item[1] for item in relative_coordinate_results]
    relative_coordinate_exact = [item[2] for item in relative_coordinate_results]
    interval_ticks, interval_error_s, interval_exact = _as_fixed_point_ticks_v1(
        np.asarray([interval_s], dtype=np.float64)
    )
    gap_ticks, gap_error_s, gap_exact = _as_fixed_point_ticks_v1(
        np.asarray([maximum_gap], dtype=np.float64)
    )
    interval_coordinate = int(interval_ticks[0])
    maximum_gap_coordinate = int(gap_ticks[0])
    aligned_coordinates = [
        item + int(start_offset_ticks[index])
        for index, item in enumerate(relative_coordinate_arrays)
    ]
    for view_index, coordinates in enumerate(aligned_coordinates):
        per_view_documents[view_index].update(
            {
                "aligned_first_time_s": (
                    int(coordinates[0]) / _FIXED_POINT_TICKS_PER_SECOND
                ),
                "aligned_last_time_s": (
                    int(coordinates[-1]) / _FIXED_POINT_TICKS_PER_SECOND
                ),
                "timestamp_coordinates_exact": relative_coordinate_exact[
                    view_index
                ],
                "timestamp_quantization_max_abs_s": relative_quantization_errors[
                    view_index
                ],
            }
        )
    terminal_coordinates = [
        int(np.rint(np.median(np.diff(item)[np.diff(item) > 0])))
        for item in aligned_coordinates
    ]
    common_start_coordinate = max(int(item[0]) for item in aligned_coordinates)
    common_end_coordinate = min(
        int(item[-1]) + terminal
        for item, terminal in zip(
            aligned_coordinates, terminal_coordinates, strict=True
        )
    )
    first_left_coordinate, right_edge_coordinates = _uniform_grid_bounds_ticks_v1(
        common_start_coordinate,
        common_end_coordinate,
        interval_coordinate,
    )
    common_start = common_start_coordinate / _FIXED_POINT_TICKS_PER_SECOND
    common_end = common_end_coordinate / _FIXED_POINT_TICKS_PER_SECOND
    first_left = first_left_coordinate / _FIXED_POINT_TICKS_PER_SECOND
    right_edges = (
        right_edge_coordinates.astype(np.float64) / _FIXED_POINT_TICKS_PER_SECOND
    )
    timestamp_quantization_max_abs_s = max(
        relative_quantization_errors,
        default=0.0,
    )
    half_open_boundary_exact = bool(
        start_arithmetic["start_offset_coordinates_exact"]
        and all(relative_coordinate_exact)
        and interval_exact
        and gap_exact
    )
    time_arithmetic = {
        **start_arithmetic,
        "bin_membership_arithmetic": "integer_nanosecond_fixed_point",
        "coordinate_ticks_per_second": _FIXED_POINT_TICKS_PER_SECOND,
        "coordinate_quantization_policy": "round_to_nearest_nanosecond_ties_to_even",
        "timestamp_quantization_max_abs_s": timestamp_quantization_max_abs_s,
        "interval_quantization_max_abs_s": interval_error_s,
        "max_gap_quantization_max_abs_s": gap_error_s,
        "arithmetic_policy_selected_from_timestamp_values": False,
        "half_open_boundary_exact": half_open_boundary_exact,
    }
    if right_edges.size == 0:
        raise RadarTimingError(
            "radar views have no complete common output interval"
        )

    assert trailing_shape is not None
    output = np.zeros(
        (view_count, len(right_edges), *trailing_shape), dtype=np.float32
    )
    valid_mask = np.zeros((view_count, len(right_edges)), dtype=bool)
    sample_counts = np.zeros((view_count, len(right_edges)), dtype=np.int32)
    invalid_reason_mask = np.zeros(
        (view_count, len(right_edges)), dtype=np.uint8
    )

    for view_index, (payload, coordinates, sequence) in enumerate(
        zip(
            payloads,
            aligned_coordinates,
            integer_sequences,
            strict=True,
        )
    ):
        sequence_gap_coordinates = coordinates[1:][np.diff(sequence) != 1]
        plateau_coordinates = coordinates[1:][np.diff(coordinates) == 0]
        maximum_observed_bin_gap = 0.0
        for output_index, right_coordinate in enumerate(right_edge_coordinates):
            left_coordinate = right_coordinate - interval_coordinate
            lower = int(
                np.searchsorted(coordinates, left_coordinate, side="left")
            )
            upper = int(
                np.searchsorted(coordinates, right_coordinate, side="left")
            )
            count = upper - lower
            sample_counts[view_index, output_index] = count
            reason = CausalUniformInvalidReasonV1(0)
            if count <= 0:
                reason |= CausalUniformInvalidReasonV1.EMPTY_INTERVAL

            bin_coordinates = coordinates[lower:upper]
            coverage_points = np.concatenate(
                (
                    np.asarray([left_coordinate], dtype=coordinates.dtype),
                    bin_coordinates,
                    np.asarray([right_coordinate], dtype=coordinates.dtype),
                )
            )
            observed_gap_coordinate = int(np.max(np.diff(coverage_points)))
            observed_gap = (
                observed_gap_coordinate / _FIXED_POINT_TICKS_PER_SECOND
            )
            maximum_observed_bin_gap = max(maximum_observed_bin_gap, observed_gap)
            gap_invalid = bool(
                observed_gap_coordinate > maximum_gap_coordinate
            )
            if gap_invalid:
                reason |= CausalUniformInvalidReasonV1.TEMPORAL_GAP

            sequence_invalid = bool(
                np.any(
                    (sequence_gap_coordinates >= left_coordinate)
                    & (sequence_gap_coordinates < right_coordinate)
                )
            )
            if sequence_invalid:
                reason |= CausalUniformInvalidReasonV1.FRAME_SEQUENCE_GAP

            plateau_invalid = bool(
                np.any(
                    (plateau_coordinates >= left_coordinate)
                    & (plateau_coordinates < right_coordinate)
                )
            )
            if plateau_invalid:
                reason |= CausalUniformInvalidReasonV1.TIMESTAMP_PLATEAU

            selected = np.asarray(payload[lower:upper])
            selected_finite = bool(np.isfinite(selected).all())
            aggregate = (
                _stable_float32_interval_mean_v1(selected)
                if count > 0 and selected_finite
                else None
            )
            if not selected_finite or (count > 0 and aggregate is None):
                reason |= CausalUniformInvalidReasonV1.NONFINITE_PAYLOAD
            invalid_reason_mask[view_index, output_index] = np.uint8(reason)
            valid = reason == 0
            valid_mask[view_index, output_index] = valid
            if valid:
                assert aggregate is not None
                output[view_index, output_index] = aggregate

        counts = sample_counts[view_index]
        frame_accounting = _frame_accounting_v1(
            coordinates,
            common_start=common_start_coordinate,
            first_complete_left=first_left_coordinate,
            last_complete_right=int(right_edge_coordinates[-1]),
            common_end=common_end_coordinate,
            assigned_sample_count=int(np.sum(counts, dtype=np.int64)),
        )
        if not frame_accounting["categories_disjoint"]:
            raise RadarTimingError("frame-accounting categories overlap")
        if not frame_accounting["assigned_count_matches_sample_counts"]:
            raise RadarTimingError(
                "frame accounting disagrees with output-interval sample counts"
            )
        if (
            not frame_accounting["coverage_complete"]
            or frame_accounting["unaccounted_payload_frame_count"] != 0
            or frame_accounting["category_sum"]
            != frame_accounting["retained_input_frame_count"]
        ):
            raise RadarTimingError("not every retained frame is exactly accounted")
        view_reasons = invalid_reason_mask[view_index]
        reason_counts = {
            name: int(
                np.count_nonzero(np.bitwise_and(view_reasons, np.uint8(flag)))
            )
            for name, flag, _ in _INVALID_REASON_FLAGS_V1
        }
        per_view_documents[view_index].update(
            {
                "valid_output_count": int(valid_mask[view_index].sum()),
                "invalid_output_count": int((~valid_mask[view_index]).sum()),
                "empty_interval_count": reason_counts["empty_interval"],
                "temporal_gap_interval_count": reason_counts["temporal_gap"],
                "sequence_gap_interval_count": reason_counts[
                    "frame_sequence_gap"
                ],
                "timestamp_plateau_interval_count": reason_counts[
                    "timestamp_plateau"
                ],
                "nonfinite_interval_count": reason_counts["nonfinite_payload"],
                "invalid_reason_interval_counts": reason_counts,
                "minimum_samples_per_interval": int(np.min(counts)),
                "maximum_samples_per_interval": int(np.max(counts)),
                "maximum_observed_interval_gap_s": maximum_observed_bin_gap,
                "frame_accounting": frame_accounting,
                "unaccounted_payload_frame_count": frame_accounting[
                    "unaccounted_payload_frame_count"
                ],
            }
        )

    invalid_reason_evidence = _invalid_reason_array_evidence_v1(
        invalid_reason_mask,
        valid_mask,
        output,
        sample_counts,
    )
    invalid_count = int(invalid_reason_evidence["invalid_interval_count"])
    if gap_policy == "raise" and invalid_count:
        raise RadarTimingError(
            f"causal uniform resampling produced {invalid_count} invalid view-intervals"
        )

    all_views_valid = np.all(valid_mask, axis=0)
    invalid_reason_mask_sha256 = canonical_ndarray_sha256(invalid_reason_mask)
    content_hashes = {
        "hash_schema_version": _CANONICAL_ARRAY_HASH_SCHEMA,
        "corrected_input_values_sha256": canonical_ndarray_sequence_sha256(
            payloads
        ),
        "aligned_input_time_coordinates_sha256": (
            canonical_ndarray_sequence_sha256(aligned_coordinates)
        ),
        "frame_sequences_sha256": canonical_ndarray_sequence_sha256(
            integer_sequences
        ),
        "output_times_sha256": canonical_ndarray_sha256(right_edges),
        "output_values_sha256": canonical_ndarray_sha256(output),
        "valid_mask_sha256": canonical_ndarray_sha256(valid_mask),
        "sample_counts_sha256": canonical_ndarray_sha256(sample_counts),
        "invalid_reason_mask_sha256": invalid_reason_mask_sha256,
        "invalid_reason_semantics_sha256": (
            CAUSAL_UNIFORM_INVALID_REASON_SEMANTICS_SHA256_V1
        ),
    }
    transform_evidence_sha256 = hashlib.sha256(
        json.dumps(
            content_hashes,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    summary = {
        "schema_version": CAUSAL_UNIFORM_RESAMPLE_SCHEMA_V1,
        "aggregation": "half_open_interval_arithmetic_mean",
        "causal": True,
        "timestamp_semantics": "right_edge_exclusive",
        "invalid_value_policy": "exact_zero_with_structural_mask",
        "gap_policy": gap_policy,
        "output_rate_hz": rate,
        "interval_s": interval_s,
        "max_gap_s": maximum_gap,
        "origin_epoch_s": origin,
        "time_arithmetic": time_arithmetic,
        "common_coverage_start_s": common_start,
        "common_coverage_end_s": common_end,
        "first_grid_left_edge_s": first_left,
        "first_grid_right_edge_s": float(right_edges[0]),
        "last_grid_right_edge_s": float(right_edges[-1]),
        "output_interval_count": int(len(right_edges)),
        "all_views_valid_interval_count": int(all_views_valid.sum()),
        "any_view_invalid_interval_count": int((~all_views_valid).sum()),
        "content_hashes": content_hashes,
        "transform_evidence_sha256": transform_evidence_sha256,
        "invalid_reason_contract": {
            "semantics": causal_uniform_invalid_reason_semantics_v1(),
            "semantics_sha256": (
                CAUSAL_UNIFORM_INVALID_REASON_SEMANTICS_SHA256_V1
            ),
            "invalid_reason_mask_sha256": invalid_reason_mask_sha256,
            **invalid_reason_evidence,
        },
        "per_view": per_view_documents,
    }
    result = CausalUniformRadarResampleV1(
        origin_epoch_s=origin,
        times_s=right_edges,
        values=output,
        valid_mask=valid_mask,
        sample_counts=sample_counts,
        interval_s=interval_s,
        summary=summary,
        invalid_reason_mask=invalid_reason_mask,
    )
    validate_causal_uniform_invalid_reason_contract_v1(result)
    return result


def block_mean_times(times_s: np.ndarray, factor: int) -> np.ndarray:
    """Return the timestamp of non-overlapping causal block-mean samples."""

    values = _strict_real_array(times_s, "times").astype(np.float64, copy=False)
    if values.ndim != 1 or type(factor) is not int or factor <= 0:
        raise RadarTimingError("times must be a vector and factor must be positive")
    usable = len(values) - len(values) % factor
    if usable < factor:
        raise RadarTimingError("timeline is too short for one block")
    return values[:usable].reshape(-1, factor).mean(axis=1)


__all__ = [
    "RadarTimingError",
    "CommonRadarTimeline",
    "CAUSAL_UNIFORM_RESAMPLE_SCHEMA_V1",
    "CAUSAL_UNIFORM_INVALID_REASON_SCHEMA_V1",
    "CAUSAL_UNIFORM_INVALID_REASON_SEMANTICS_SHA256_V1",
    "CausalUniformInvalidReasonV1",
    "CausalUniformRadarResampleV1",
    "canonical_ndarray_sha256",
    "canonical_ndarray_sequence_sha256",
    "causal_uniform_invalid_reason_semantics_v1",
    "validate_causal_uniform_invalid_reason_contract_v1",
    "repair_common_timestamp_plateaus",
    "fuse_common_radar_timeline",
    "causal_uniform_resample_radar_views_v1",
    "block_mean_times",
]
