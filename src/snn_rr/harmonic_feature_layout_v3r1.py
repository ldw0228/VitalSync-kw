"""Frozen feature layout and outer-train scaling for DHFER-SNN-v3r1.

The v3r1 cache contains 571 values per candidate node.  A numeric zero in
that cache is not, by itself, evidence that a cell is available: availability
is determined from the candidate, radar, and harmonic-ratio geometry.  This
module centralises that target-free structural rule and makes the scaler use
the same rule during both fitting and inference.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Final

import numpy as np


TOTAL_FEATURE_WIDTH: Final[int] = 571
CORE_OFFSET: Final[int] = 0
CORE_WIDTH: Final[int] = 46
RF_OFFSET: Final[int] = 46
RF_WIDTH: Final[int] = 378
SVD_OFFSET: Final[int] = 424
SVD_WIDTH: Final[int] = 147
RF_SVD_RATIOS: Final[tuple[float, ...]] = (
    0.25,
    1.0 / 3.0,
    0.5,
    1.0,
    2.0,
    3.0,
    4.0,
)
EXPECTED_FEATURE_NAMES_FILE_SHA256: Final[str] = (
    "97e18a11ec7672d96ec41ae90c22e5e9c7d8744e265d146b10b72895f306da54"
)
EXPECTED_FEATURE_NAMES_SEMANTIC_SHA256: Final[str] = (
    "d7553f8b11733903393575d02bc6acd4a8edefd5ce0e538491295ec84d938f05"
)

FEATURE_LAYOUT: Final[dict[str, object]] = {
    "total_width": TOTAL_FEATURE_WIDTH,
    "concatenation_order": ["core", "rf", "svd"],
    "core": {"offset": CORE_OFFSET, "width": CORE_WIDTH, "shape": [46]},
    "rf": {
        "offset": RF_OFFSET,
        "width": RF_WIDTH,
        "shape": [3, 7, 2, 9],
        "axis_order": ["radar", "ratio", "branch", "statistic"],
    },
    "svd": {
        "offset": SVD_OFFSET,
        "width": SVD_WIDTH,
        "shape": [3, 7, 7],
        "axis_order": ["radar", "ratio", "statistic"],
    },
    "ordered_names_semantic_sha256": EXPECTED_FEATURE_NAMES_SEMANTIC_SHA256,
}


class FeatureLayoutContractError(ValueError):
    """Raised when schema, shape, availability, or scaler state drifts."""


def semantic_sha256(value: object) -> str:
    """Return the repository's canonical JSON semantic digest."""

    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


FEATURE_LAYOUT_SEMANTIC_SHA256: Final[str] = semantic_sha256(FEATURE_LAYOUT)


def _reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise FeatureLayoutContractError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_nonfinite_json_constant(value: str) -> None:
    raise FeatureLayoutContractError(f"non-finite JSON constant is forbidden: {value}")


def validate_ordered_feature_names(names: Sequence[str]) -> str:
    """Validate the exact ordered 571-name schema and return its digest."""

    normalized = tuple(str(name) for name in names)
    if len(normalized) != TOTAL_FEATURE_WIDTH:
        raise FeatureLayoutContractError(
            f"feature schema must contain exactly {TOTAL_FEATURE_WIDTH} names"
        )
    if len(set(normalized)) != len(normalized):
        raise FeatureLayoutContractError("feature names must be unique")
    digest = semantic_sha256(list(normalized))
    if digest != EXPECTED_FEATURE_NAMES_SEMANTIC_SHA256:
        raise FeatureLayoutContractError("ordered feature-name semantic digest drifted")
    return digest


def load_and_validate_feature_names(path: str | Path) -> tuple[str, ...]:
    """Load the immutable cache schema and fail closed on byte/name drift."""

    source = Path(path)
    raw = source.read_bytes()
    if hashlib.sha256(raw).hexdigest() != EXPECTED_FEATURE_NAMES_FILE_SHA256:
        raise FeatureLayoutContractError("feature_names.json byte digest drifted")
    try:
        document = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=_reject_nonfinite_json_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise FeatureLayoutContractError("feature_names.json is not strict JSON") from error
    if not isinstance(document, Mapping):
        raise FeatureLayoutContractError("feature_names.json must be an object")
    names = document.get("node_feature_names")
    if not isinstance(names, list) or not all(isinstance(name, str) for name in names):
        raise FeatureLayoutContractError("node_feature_names must be a string array")
    validate_ordered_feature_names(names)
    return tuple(names)


def _as_bool_array(name: str, value: np.ndarray | Sequence[object]) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype != np.bool_:
        raise FeatureLayoutContractError(f"{name} must have boolean dtype")
    return array


def build_structural_availability_mask(
    candidate_rr_bpm: np.ndarray,
    candidate_mask: np.ndarray,
    joint_radar_mask: np.ndarray,
    rr_min_bpm: float = 6.0,
    rr_max_bpm: float = 45.0,
) -> np.ndarray:
    """Build the target-free structural mask with shape ``[..., K, 571]``.

    Core cells follow candidate availability.  RF and SVD cells additionally
    require an available radar and an in-band candidate-times-ratio location.
    Only RF ``raw_power`` is present in the frozen cache; every IQ-branch cell
    is structurally false even if stale/non-zero bytes are supplied.
    """

    candidate_rr = np.asarray(candidate_rr_bpm)
    candidates = _as_bool_array("candidate_mask", candidate_mask)
    radars = _as_bool_array("joint_radar_mask", joint_radar_mask)
    if candidate_rr.ndim < 1 or candidate_rr.shape != candidates.shape:
        raise FeatureLayoutContractError(
            "candidate_rr_bpm and candidate_mask must share shape [..., K]"
        )
    if candidate_rr.shape[-1] < 1 or candidate_rr.shape[-1] > 12:
        raise FeatureLayoutContractError("candidate count must be in [1, 12]")
    if radars.shape != (*candidate_rr.shape[:-1], 3):
        raise FeatureLayoutContractError(
            "joint_radar_mask must have shape candidate_rr_bpm.shape[:-1] + (3,)"
        )
    if not np.issubdtype(candidate_rr.dtype, np.number):
        raise FeatureLayoutContractError("candidate_rr_bpm must be numeric")
    if not np.isfinite([rr_min_bpm, rr_max_bpm]).all() or not rr_min_bpm < rr_max_bpm:
        raise FeatureLayoutContractError("RR bounds must be finite and increasing")
    valid_candidate = (
        np.isfinite(candidate_rr)
        & (candidate_rr >= float(rr_min_bpm))
        & (candidate_rr <= float(rr_max_bpm))
    )
    if np.any(candidates & ~valid_candidate):
        raise FeatureLayoutContractError(
            "available candidates require finite in-range RR"
        )
    node_available = candidates
    ratios = np.asarray(RF_SVD_RATIOS, dtype=np.float64)
    ratio_rr = candidate_rr.astype(np.float64, copy=False)[..., None] * ratios
    ratio_available = (
        node_available[..., None]
        & (ratio_rr >= float(rr_min_bpm))
        & (ratio_rr <= float(rr_max_bpm))
    )
    cell_available = (
        node_available[..., None, None]
        & radars[..., None, :, None]
        & ratio_available[..., None, :]
    )

    core = np.broadcast_to(
        node_available[..., None], (*node_available.shape, CORE_WIDTH)
    )
    raw_rf = cell_available[..., None]
    iq_rf = np.zeros_like(raw_rf)
    rf = np.concatenate((raw_rf, iq_rf), axis=-1)
    rf = np.broadcast_to(rf[..., None], (*rf.shape, 9)).reshape(
        *node_available.shape, RF_WIDTH
    )
    svd = np.broadcast_to(
        cell_available[..., None], (*cell_available.shape, 7)
    ).reshape(*node_available.shape, SVD_WIDTH)
    result = np.concatenate((core, rf, svd), axis=-1)
    if result.shape != (*candidate_rr.shape, TOTAL_FEATURE_WIDTH):
        raise RuntimeError("internal structural mask layout drifted")
    return np.asarray(result, dtype=np.bool_)


def sanitize_structural_features(
    features: np.ndarray,
    availability: np.ndarray,
    *,
    output_dtype: np.dtype[Any] | type[np.floating[Any]] = np.float32,
) -> np.ndarray:
    """Validate available cells and overwrite every unavailable cell by +0."""

    values = np.asarray(features)
    mask = _as_bool_array("availability", availability)
    if values.ndim < 2 or values.shape[-1] != TOTAL_FEATURE_WIDTH:
        raise FeatureLayoutContractError("features must end in width 571")
    if mask.shape != values.shape:
        raise FeatureLayoutContractError("availability shape must equal feature shape")
    if not np.issubdtype(values.dtype, np.number):
        raise FeatureLayoutContractError("features must be numeric")
    if not np.isfinite(values[mask]).all():
        raise FeatureLayoutContractError("available feature cells must be finite")
    output = np.zeros(values.shape, dtype=output_dtype)
    output[mask] = values[mask]
    if not np.isfinite(output).all():
        raise FeatureLayoutContractError("sanitized features must be finite")
    if np.count_nonzero(output[~mask]):
        raise RuntimeError("unavailable feature sanitization is not exact zero")
    return output


@dataclass(frozen=True)
class OuterTrainFeatureStandardizer:
    """Deterministic per-column scaler fitted only at declared train positions."""

    mean: np.ndarray
    scale: np.ndarray
    observed_count: np.ndarray
    fit_position_count: int
    fit_available_cell_count: int
    minimum_scale: float = 1.0e-6
    fit_scope: str = "outer_train_only"
    schema_version: int = 1

    def __post_init__(self) -> None:
        mean = np.array(self.mean, dtype=np.float64, copy=True)
        scale = np.array(self.scale, dtype=np.float64, copy=True)
        count = np.array(self.observed_count, dtype=np.int64, copy=True)
        if mean.shape != (TOTAL_FEATURE_WIDTH,) or scale.shape != mean.shape:
            raise FeatureLayoutContractError("scaler mean/scale must be length 571")
        if count.shape != mean.shape:
            raise FeatureLayoutContractError("scaler observed_count must be length 571")
        if not np.isfinite(mean).all() or not np.isfinite(scale).all():
            raise FeatureLayoutContractError("scaler statistics must be finite")
        if np.any(scale <= 0.0) or np.any(count < 0):
            raise FeatureLayoutContractError("scaler scale/count values are invalid")
        if not np.isfinite(self.minimum_scale) or self.minimum_scale <= 0.0:
            raise FeatureLayoutContractError("minimum_scale must be finite and positive")
        if self.fit_scope != "outer_train_only":
            raise FeatureLayoutContractError("scaler fit_scope must be outer_train_only")
        if int(self.schema_version) != 1:
            raise FeatureLayoutContractError("unknown scaler schema version")
        if int(self.fit_position_count) < 1 or int(self.fit_available_cell_count) < 1:
            raise FeatureLayoutContractError("scaler fit receipt cannot be empty")
        mean.setflags(write=False)
        scale.setflags(write=False)
        count.setflags(write=False)
        object.__setattr__(self, "mean", mean)
        object.__setattr__(self, "scale", scale)
        object.__setattr__(self, "observed_count", count)

    @classmethod
    def fit(
        cls,
        features: np.ndarray,
        availability: np.ndarray,
        fit_positions: np.ndarray | None = None,
        *,
        fit_scope: str = "outer_train_only",
        minimum_scale: float = 1.0e-6,
    ) -> "OuterTrainFeatureStandardizer":
        """Fit on positions explicitly supplied by the outer-training caller.

        ``fit_positions`` may address every prefix position (``[N,K]`` for
        ``[N,K,571]``), or each row before the candidate axis (``[N]``); the
        latter is broadcast across candidates.  If omitted, all supplied
        positions are declared outer-train positions, so callers must pass an
        already outer-train-only tensor.
        """

        if fit_scope != "outer_train_only":
            raise FeatureLayoutContractError(
                "standardizer may only be fitted with fit_scope='outer_train_only'"
            )
        if not np.isfinite(minimum_scale) or minimum_scale <= 0.0:
            raise FeatureLayoutContractError("minimum_scale must be finite and positive")
        values = np.asarray(features)
        mask = _as_bool_array("availability", availability)
        if values.ndim < 2 or values.shape[-1] != TOTAL_FEATURE_WIDTH:
            raise FeatureLayoutContractError("features must end in width 571")
        if values.shape != mask.shape:
            raise FeatureLayoutContractError("availability shape must equal feature shape")
        if not np.issubdtype(values.dtype, np.number):
            raise FeatureLayoutContractError("features must be numeric")
        prefix_shape = values.shape[:-1]
        if fit_positions is None:
            selected = np.ones(prefix_shape, dtype=np.bool_)
        else:
            selected = _as_bool_array("fit_positions", fit_positions)
            if selected.shape == prefix_shape[:-1]:
                selected = np.broadcast_to(selected[..., None], prefix_shape)
            elif selected.shape != prefix_shape:
                raise FeatureLayoutContractError(
                    "fit_positions must address rows or all feature-prefix positions"
                )
        if not selected.any():
            raise FeatureLayoutContractError("at least one outer-train position is required")
        effective = mask & selected[..., None]
        if not effective.any():
            raise FeatureLayoutContractError("outer-train positions contain no available cells")
        if not np.isfinite(values[effective]).all():
            raise FeatureLayoutContractError(
                "available outer-train feature cells must be finite"
            )

        flat_values = values.reshape(-1, TOTAL_FEATURE_WIDTH).astype(
            np.float64, copy=False
        )
        flat_mask = effective.reshape(-1, TOTAL_FEATURE_WIDTH)
        counts = flat_mask.sum(axis=0, dtype=np.int64)
        finite_values = np.zeros_like(flat_values)
        finite_values[flat_mask] = flat_values[flat_mask]
        sums = finite_values.sum(axis=0, dtype=np.float64)
        means = np.divide(
            sums,
            counts,
            out=np.zeros(TOTAL_FEATURE_WIDTH, dtype=np.float64),
            where=counts > 0,
        )
        centered = np.where(flat_mask, finite_values - means, 0.0)
        variances = np.divide(
            np.square(centered).sum(axis=0, dtype=np.float64),
            counts,
            out=np.zeros(TOTAL_FEATURE_WIDTH, dtype=np.float64),
            where=counts > 0,
        )
        scales = np.sqrt(np.maximum(variances, 0.0))
        scales = np.where((counts > 0) & (scales >= minimum_scale), scales, 1.0)
        return cls(
            mean=means,
            scale=scales,
            observed_count=counts,
            fit_position_count=int(selected.sum(dtype=np.int64)),
            fit_available_cell_count=int(effective.sum(dtype=np.int64)),
            minimum_scale=float(minimum_scale),
            fit_scope=fit_scope,
        )

    def transform(self, features: np.ndarray, availability: np.ndarray) -> np.ndarray:
        values = np.asarray(features)
        mask = _as_bool_array("availability", availability)
        if values.ndim < 2 or values.shape[-1] != TOTAL_FEATURE_WIDTH:
            raise FeatureLayoutContractError("features must end in width 571")
        if values.shape != mask.shape:
            raise FeatureLayoutContractError("availability shape must equal feature shape")
        if not np.issubdtype(values.dtype, np.number):
            raise FeatureLayoutContractError("features must be numeric")
        if not np.isfinite(values[mask]).all():
            raise FeatureLayoutContractError("available feature cells must be finite")
        standardized = (
            values.astype(np.float64, copy=False) - self.mean
        ) / self.scale
        output = np.zeros(values.shape, dtype=np.float32)
        output[mask] = standardized[mask].astype(np.float32, copy=False)
        if not np.isfinite(output).all():
            raise FeatureLayoutContractError("transformed features must be finite")
        if np.count_nonzero(output[~mask]):
            raise RuntimeError("masked transformed cells are not exact zero")
        return output

    def to_state(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "class": "OuterTrainFeatureStandardizer",
            "fit_scope": self.fit_scope,
            "feature_width": TOTAL_FEATURE_WIDTH,
            "feature_names_semantic_sha256": EXPECTED_FEATURE_NAMES_SEMANTIC_SHA256,
            "feature_layout_semantic_sha256": FEATURE_LAYOUT_SEMANTIC_SHA256,
            "algorithm": "available_population_mean_std_float64",
            "minimum_scale": self.minimum_scale,
            "fit_position_count": self.fit_position_count,
            "fit_available_cell_count": self.fit_available_cell_count,
            "mean": self.mean.tolist(),
            "scale": self.scale.tolist(),
            "observed_count": self.observed_count.tolist(),
        }

    def state_receipt(self) -> dict[str, object]:
        state = self.to_state()
        return {
            "schema_version": 1,
            "artifact_type": "outer_train_feature_standardizer_v3r1",
            "semantic_sha256": semantic_sha256(state),
            "feature_width": TOTAL_FEATURE_WIDTH,
            "fit_scope": self.fit_scope,
            "fit_position_count": self.fit_position_count,
            "fit_available_cell_count": self.fit_available_cell_count,
        }

    def save_json(self, path: str | Path) -> dict[str, object]:
        state = self.to_state()
        document = {**state, "semantic_receipt": self.state_receipt()}
        destination = Path(path)
        destination.write_text(
            json.dumps(
                document,
                indent=2,
                sort_keys=True,
                ensure_ascii=False,
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        )
        return document["semantic_receipt"]  # type: ignore[return-value]

    @classmethod
    def from_state(cls, document: Mapping[str, object]) -> "OuterTrainFeatureStandardizer":
        state = dict(document)
        receipt = state.pop("semantic_receipt", None)
        required = {
            "schema_version",
            "class",
            "fit_scope",
            "feature_width",
            "feature_names_semantic_sha256",
            "feature_layout_semantic_sha256",
            "algorithm",
            "minimum_scale",
            "fit_position_count",
            "fit_available_cell_count",
            "mean",
            "scale",
            "observed_count",
        }
        if set(state) != required:
            raise FeatureLayoutContractError("scaler JSON state keys drifted")
        if state["class"] != "OuterTrainFeatureStandardizer":
            raise FeatureLayoutContractError("scaler class binding drifted")
        if state["feature_width"] != TOTAL_FEATURE_WIDTH:
            raise FeatureLayoutContractError("scaler feature width drifted")
        if (
            state["feature_names_semantic_sha256"]
            != EXPECTED_FEATURE_NAMES_SEMANTIC_SHA256
            or state["feature_layout_semantic_sha256"]
            != FEATURE_LAYOUT_SEMANTIC_SHA256
        ):
            raise FeatureLayoutContractError("scaler feature schema binding drifted")
        if state["algorithm"] != "available_population_mean_std_float64":
            raise FeatureLayoutContractError("scaler algorithm binding drifted")
        if receipt is not None:
            if not isinstance(receipt, Mapping):
                raise FeatureLayoutContractError("semantic_receipt must be an object")
            if receipt.get("semantic_sha256") != semantic_sha256(state):
                raise FeatureLayoutContractError("scaler semantic receipt drifted")
        try:
            result = cls(
                mean=np.asarray(state["mean"], dtype=np.float64),
                scale=np.asarray(state["scale"], dtype=np.float64),
                observed_count=np.asarray(state["observed_count"], dtype=np.int64),
                fit_position_count=int(state["fit_position_count"]),
                fit_available_cell_count=int(state["fit_available_cell_count"]),
                minimum_scale=float(state["minimum_scale"]),
                fit_scope=str(state["fit_scope"]),
                schema_version=int(state["schema_version"]),
            )
        except (TypeError, ValueError, OverflowError) as error:
            if isinstance(error, FeatureLayoutContractError):
                raise
            raise FeatureLayoutContractError("invalid scaler JSON values") from error
        if receipt is not None and dict(receipt) != result.state_receipt():
            raise FeatureLayoutContractError("scaler semantic receipt fields drifted")
        return result

    @classmethod
    def load_json(cls, path: str | Path) -> "OuterTrainFeatureStandardizer":
        try:
            document = json.loads(
                Path(path).read_text(encoding="utf-8"),
                object_pairs_hook=_reject_duplicate_pairs,
                parse_constant=_reject_nonfinite_json_constant,
            )
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise FeatureLayoutContractError("scaler state is not strict JSON") from error
        if not isinstance(document, Mapping):
            raise FeatureLayoutContractError("scaler state must be a JSON object")
        if "semantic_receipt" not in document:
            raise FeatureLayoutContractError("scaler JSON semantic_receipt is required")
        return cls.from_state(document)


__all__ = [
    "CORE_OFFSET",
    "CORE_WIDTH",
    "EXPECTED_FEATURE_NAMES_FILE_SHA256",
    "EXPECTED_FEATURE_NAMES_SEMANTIC_SHA256",
    "FEATURE_LAYOUT",
    "FEATURE_LAYOUT_SEMANTIC_SHA256",
    "FeatureLayoutContractError",
    "OuterTrainFeatureStandardizer",
    "RF_OFFSET",
    "RF_SVD_RATIOS",
    "RF_WIDTH",
    "SVD_OFFSET",
    "SVD_WIDTH",
    "TOTAL_FEATURE_WIDTH",
    "build_structural_availability_mask",
    "load_and_validate_feature_names",
    "sanitize_structural_features",
    "semantic_sha256",
    "validate_ordered_feature_names",
]
