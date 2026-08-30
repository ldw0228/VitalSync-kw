#!/usr/bin/env python3
"""Validation-locked causal decoder for respiratory harmonic aliases.

For every outer fold, this experiment reconstructs the source checkpoint's
validation predictions, selects every blend/classifier/decoder setting using
only those validation identities, writes an immutable lock, and only then
opens the corresponding outer-test prediction files.  The decoder combines
two SNN estimates with classical candidates ``RR * {1,2,3,4}``, posterior
alias evidence, trainable-on-validation binary alias evidence, uncertainty,
and a forward-only state transition.

An optional alias-gated SNN contributes its RR estimate only when a
validation-selected three-way simplex clears a declared improvement guard;
its alias probability remains an independently selectable evidence source.
The saved source predictions contain reference-valid rows only.  Recursive
state is consequently reset across any window-number gap.  Independently, the
classifier receives the cache's strictly causal history features, which were
built from every preceding radar window and never use reference RR.  This is a
conservative deployment emulation: it never bridges an unevaluated interval
with hidden knowledge of its labels or future observations.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
for search_path in (PROJECT_ROOT, SRC_ROOT):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

from snn_rr.metrics import (  # noqa: E402
    grouped_oof_metrics,
    identity_macro_metrics,
    risk_coverage_curve,
)
from scripts.ensemble import (  # noqa: E402
    align_prediction_bundles,
    infer_validation_checkpoint,
    prepare_source_cache,
    verify_checkpoint_scaler,
)
from scripts.train import PredictionBundle, load_prediction_bundle  # noqa: E402


PIPELINE_VERSION = "causal-alias-decoder-v2"
DIVISORS = np.asarray([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
DEFAULT_COVERAGES = (1.0, 0.9, 0.8, 0.7, 0.5)
HISTORY_FEATURES = (
    "history_lag_1_classical_rr_bpm",
    "history_lag_1_classical_confidence",
    "history_lag_1_radar_peak_spread_bpm",
    "history_lag_1_available",
    "history_roll_4_rr_median_bpm",
    "history_roll_4_rr_mad_bpm",
    "history_roll_4_rr_trend_bpm_per_window",
    "history_roll_4_confidence_mean",
    "history_roll_4_spread_median_bpm",
    "history_roll_4_available_fraction",
    "history_roll_4_sufficient",
    "history_roll_8_rr_median_bpm",
    "history_roll_8_rr_mad_bpm",
    "history_roll_8_rr_trend_bpm_per_window",
    "history_roll_8_confidence_mean",
    "history_roll_8_spread_median_bpm",
    "history_roll_8_available_fraction",
    "history_roll_8_sufficient",
)


@dataclass(frozen=True, slots=True)
class DecoderParams:
    enabled: bool = True
    alias_route: str = "multiplier"
    alias_threshold: float = 0.5
    alias_pull: float = 0.75
    state_decay: float = 0.5
    continuity: float = 0.25
    uncertainty_gain: float = 0.5
    confidence_scale: float = 0.20
    rr_min: float = 6.0
    rr_max: float = 45.0


@dataclass(frozen=True, slots=True)
class LinearAliasModel:
    feature_names: tuple[str, ...]
    mean: np.ndarray
    scale: np.ndarray
    coefficient: np.ndarray
    intercept: float
    constant_probability: float | None
    regularization_c: float
    fit_rows: int
    positive_rows: int

    def predict_probability(self, features: np.ndarray) -> np.ndarray:
        values = np.asarray(features, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != len(self.feature_names):
            raise ValueError("alias features have the wrong shape")
        if self.constant_probability is not None:
            return np.full(len(values), self.constant_probability, dtype=np.float64)
        standardized = (values - self.mean) / self.scale
        logits = standardized @ self.coefficient + self.intercept
        logits = np.clip(logits, -40.0, 40.0)
        return 1.0 / (1.0 + np.exp(-logits))

    def to_json(self) -> dict[str, Any]:
        return {
            "feature_names": list(self.feature_names),
            "mean": self.mean.tolist(),
            "scale": self.scale.tolist(),
            "coefficient": self.coefficient.tolist(),
            "intercept": self.intercept,
            "constant_probability": self.constant_probability,
            "regularization_c": self.regularization_c,
            "fit_rows": self.fit_rows,
            "positive_rows": self.positive_rows,
        }


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_checkpoint(
    path: Path,
    *,
    fold: int,
    expected_run_signature: str | None = None,
) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    required = {
        "model_type",
        "model_kwargs",
        "model_state",
        "fold",
        "split",
        "aux_center",
        "aux_scale",
        "run_signature",
    }
    missing = sorted(required - set(checkpoint))
    if missing:
        raise KeyError(f"checkpoint missing fields {missing}: {path}")
    if checkpoint["model_type"] != "snn" or int(checkpoint["fold"]) != fold:
        raise ValueError(f"checkpoint model/fold mismatch: {path}")
    if (
        expected_run_signature is not None
        and str(checkpoint["run_signature"]) != str(expected_run_signature)
    ):
        raise RuntimeError(
            f"checkpoint/run_config signature mismatch for fold {fold}: {path}"
        )
    return checkpoint


def choose_deployment_variant(
    *,
    static_stack_supported: bool,
    causal_decoder_supported: bool,
) -> tuple[str, str]:
    """Return the independently supported locked variant and report decision."""

    if causal_decoder_supported:
        return "accept", "validation_locked_causal_alias_decoder"
    if static_stack_supported:
        return "accept", "validation_locked_blend"
    return "reject", "validation_locked_two_way_blend"


def indices_for_identities(
    metadata: pd.DataFrame,
    identities: Sequence[str],
    *,
    valid_only: bool = True,
) -> np.ndarray:
    identity = metadata["identity"].astype(str).to_numpy()
    selected = np.isin(identity, np.asarray(identities, dtype=str))
    if valid_only:
        selected &= metadata["reference_valid"].to_numpy(dtype=bool)
    return np.flatnonzero(selected)


def audit_fold_split(split: Mapping[str, Sequence[str]]) -> None:
    groups = {
        name: set(map(str, split[f"{name}_identities"]))
        for name in ("train", "validation", "test")
    }
    if any(not values for values in groups.values()):
        raise ValueError("fold train/validation/test identity groups must be non-empty")
    if (
        groups["train"] & groups["validation"]
        or groups["train"] & groups["test"]
        or groups["validation"] & groups["test"]
    ):
        raise RuntimeError("fold identities are not disjoint")


def _sorted_bundle(bundle: PredictionBundle) -> PredictionBundle:
    order = np.argsort(bundle.index, kind="stable")
    return PredictionBundle(
        **{
            field: np.asarray(getattr(bundle, field))[order]
            for field in PredictionBundle.__dataclass_fields__
        }
    )


def load_saved_bundle(
    path: Path,
    checkpoint: Mapping[str, Any],
    metadata: pd.DataFrame,
    *,
    fold: int,
    role: str,
) -> PredictionBundle:
    bundle, folds, signature = load_prediction_bundle(path)
    if signature != checkpoint["run_signature"] or not np.all(folds == fold):
        raise RuntimeError(f"prediction provenance mismatch: {path}")
    identities = checkpoint["split"][f"{role}_identities"]
    expected = indices_for_identities(metadata, identities)
    ordered = _sorted_bundle(bundle)
    if not np.array_equal(ordered.index, expected):
        raise RuntimeError(f"prediction identity/index mismatch: {path}")
    return ordered


def candidate_matrix(classical_rr: np.ndarray) -> np.ndarray:
    values = np.asarray(classical_rr, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError("classical RR must be one-dimensional")
    return values[:, None] * DIVISORS[None, :]


def alias_targets(
    target: np.ndarray,
    classical_rr: np.ndarray,
    *,
    tolerance_bpm: float = 2.0,
    rr_range: tuple[float, float] = (6.0, 45.0),
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return binary alias labels, confidence mask, and best divisor index."""

    target = np.asarray(target, dtype=np.float64)
    candidates = candidate_matrix(classical_rr)
    if target.shape != candidates.shape[:1]:
        raise ValueError("target/classical vectors must align")
    valid_candidate = (
        np.isfinite(candidates)
        & (candidates >= float(rr_range[0]))
        & (candidates <= float(rr_range[1]))
    )
    error = np.where(valid_candidate, np.abs(candidates - target[:, None]), np.inf)
    best = np.argmin(error, axis=1)
    minimum = error[np.arange(len(target)), best]
    confident = np.isfinite(target) & (minimum <= tolerance_bpm)
    return best > 0, confident, best


def posterior_candidate_support(
    posterior: np.ndarray,
    classical_rr: np.ndarray,
    *,
    rr_min: float,
    rr_max: float,
    radius_bpm: float = 0.5,
) -> np.ndarray:
    probabilities = np.asarray(posterior, dtype=np.float64)
    if probabilities.ndim != 2 or probabilities.shape[1] < 2:
        raise ValueError("a full posterior [row, RR-bin] is required")
    candidates = candidate_matrix(classical_rr)
    grid = np.linspace(rr_min, rr_max, probabilities.shape[1], dtype=np.float64)
    support = np.zeros_like(candidates)
    for column in range(candidates.shape[1]):
        distance = np.abs(grid[None, :] - candidates[:, column, None])
        support[:, column] = np.sum(
            np.where(distance <= radius_bpm, probabilities, 0.0), axis=1
        )
    invalid = (
        ~np.isfinite(candidates)
        | (candidates < rr_min)
        | (candidates > rr_max)
    )
    support[invalid] = 0.0
    return support


def _metadata_numeric(rows: pd.DataFrame, column: str, default: float = 0.0) -> np.ndarray:
    if column not in rows:
        return np.full(len(rows), default, dtype=np.float64)
    return pd.to_numeric(rows[column], errors="coerce").to_numpy(dtype=np.float64)


def build_alias_features(
    bundle_a: PredictionBundle,
    bundle_b: PredictionBundle,
    metadata: pd.DataFrame,
    causal_auxiliary: np.ndarray,
    history_names: Sequence[str],
    *,
    weight_a: float,
    rr_min: float,
    rr_max: float,
    external_alias_probability: np.ndarray | None = None,
    stack_prediction: np.ndarray | None = None,
    stack_uncertainty: np.ndarray | None = None,
) -> tuple[np.ndarray, tuple[str, ...], dict[str, np.ndarray]]:
    """Build current-and-past radar-only features; no target is accepted."""

    first, second = align_prediction_bundles(bundle_a, bundle_b)
    index = np.asarray(first.index, dtype=np.int64)
    rows = metadata.iloc[index]
    classical = _metadata_numeric(rows, "classical_rr_bpm")
    confidence = _metadata_numeric(rows, "classical_confidence")
    spread = _metadata_numeric(rows, "radar_peak_spread_bpm")
    peaks = np.column_stack(
        [_metadata_numeric(rows, f"radar_peak_{item}_bpm") for item in (1, 2, 3)]
    )
    prediction_a = np.asarray(first.prediction, dtype=np.float64)
    prediction_b = np.asarray(second.prediction, dtype=np.float64)
    two_way_blend = weight_a * prediction_a + (1.0 - weight_a) * prediction_b
    blend = (
        two_way_blend
        if stack_prediction is None
        else np.asarray(stack_prediction, dtype=np.float64)
    )
    if blend.shape != prediction_a.shape or not np.isfinite(blend).all():
        raise ValueError("stack prediction must be a finite aligned vector")
    uncertainty_a = np.asarray(first.uncertainty, dtype=np.float64)
    uncertainty_b = np.asarray(second.uncertainty, dtype=np.float64)
    disagreement = np.abs(prediction_a - prediction_b)
    two_way_uncertainty = (
        weight_a * np.log1p(np.maximum(uncertainty_a, 0.0))
        + (1.0 - weight_a) * np.log1p(np.maximum(uncertainty_b, 0.0))
        + 0.25 * disagreement
    )
    raw_uncertainty = (
        two_way_uncertainty
        if stack_uncertainty is None
        else np.asarray(stack_uncertainty, dtype=np.float64)
    )
    if raw_uncertainty.shape != prediction_a.shape or not np.isfinite(
        raw_uncertainty
    ).all():
        raise ValueError("stack uncertainty must be a finite aligned vector")
    posterior = np.asarray(second.posterior_probability, dtype=np.float64)
    support = posterior_candidate_support(
        posterior,
        classical,
        rr_min=rr_min,
        rr_max=rr_max,
    )
    candidates = candidate_matrix(classical)
    candidate_distance = np.abs(candidates - blend[:, None])
    ratio = blend / np.maximum(classical, 1e-3)
    map_prediction = np.asarray(second.map_prediction, dtype=np.float64)

    columns: list[np.ndarray] = [
        classical,
        confidence,
        spread,
        peaks[:, 0],
        peaks[:, 1],
        peaks[:, 2],
        prediction_a,
        prediction_b,
        blend,
        prediction_a - prediction_b,
        ratio,
        raw_uncertainty,
        np.asarray(first.rr_std, dtype=np.float64),
        np.asarray(second.rr_std, dtype=np.float64),
        np.asarray(first.quality, dtype=np.float64),
        np.asarray(second.quality, dtype=np.float64),
        map_prediction,
        map_prediction - blend,
        np.asarray(second.posterior_entropy, dtype=np.float64),
    ]
    names = [
        "classical_rr_bpm",
        "classical_confidence",
        "radar_peak_spread_bpm",
        "radar_peak_1_bpm",
        "radar_peak_2_bpm",
        "radar_peak_3_bpm",
        "prediction_a_bpm",
        "prediction_b_bpm",
        "blend_bpm",
        "model_signed_disagreement_bpm",
        "blend_to_classical_ratio",
        "combined_uncertainty",
        "rr_std_a_bpm",
        "rr_std_b_bpm",
        "quality_a",
        "quality_b",
        "exact_map_prediction_bpm",
        "exact_map_minus_blend_bpm",
        "exact_posterior_entropy",
    ]
    for divisor_index, divisor in enumerate(DIVISORS):
        columns.extend(
            [
                np.log(np.maximum(support[:, divisor_index], 1e-8)),
                candidate_distance[:, divisor_index],
            ]
        )
        names.extend(
            [
                f"log_posterior_support_x{int(divisor)}",
                f"blend_distance_x{int(divisor)}_bpm",
            ]
        )
    log_alias_support = np.log(np.maximum(np.sum(support[:, 1:], axis=1), 1e-8))
    log_direct_support = np.log(np.maximum(support[:, 0], 1e-8))
    columns.append(log_alias_support - log_direct_support)
    names.append("log_alias_to_direct_support_ratio")
    if external_alias_probability is not None:
        external = np.asarray(external_alias_probability, dtype=np.float64)
        if external.shape != (len(index),) or not np.isfinite(external).all():
            raise ValueError("external alias probability must be a finite aligned vector")
        columns.append(np.clip(external, 0.0, 1.0))
        names.append("alias_head_probability")

    history_lookup = {str(name): position for position, name in enumerate(history_names)}
    history_width = len(history_names)
    history_start = causal_auxiliary.shape[1] - history_width
    if history_start < 0:
        raise ValueError("causal auxiliary matrix is narrower than history schema")
    for name in HISTORY_FEATURES:
        if name not in history_lookup:
            raise RuntimeError(f"required causal feature is absent: {name}")
        values = np.asarray(
            causal_auxiliary[index, history_start + history_lookup[name]],
            dtype=np.float64,
        )
        columns.append(values)
        names.append(name)

    matrix = np.column_stack(columns)
    matrix = np.nan_to_num(matrix, nan=0.0, posinf=50.0, neginf=-50.0)
    matrix = np.clip(matrix, -1e4, 1e4)
    context = {
        "index": index,
        "classical_rr": classical,
        "classical_confidence": confidence,
        "blend": blend,
        "raw_uncertainty": raw_uncertainty,
        "posterior_support": support,
    }
    return matrix, tuple(names), context


def identity_balanced_weights(identities: Sequence[str] | np.ndarray) -> np.ndarray:
    groups = np.asarray(identities, dtype=str)
    if groups.ndim != 1 or not len(groups):
        raise ValueError("identities must be a non-empty vector")
    weights = np.zeros(len(groups), dtype=np.float64)
    unique, counts = np.unique(groups, return_counts=True)
    for identity, count in zip(unique, counts, strict=True):
        weights[groups == identity] = 1.0 / float(count)
    return weights / np.mean(weights)


def fit_alias_model(
    features: np.ndarray,
    alias_label: np.ndarray,
    selected: np.ndarray,
    identities: np.ndarray,
    feature_names: Sequence[str],
    *,
    regularization_c: float,
) -> LinearAliasModel:
    values = np.asarray(features, dtype=np.float64)
    labels = np.asarray(alias_label, dtype=bool)
    use = np.asarray(selected, dtype=bool)
    groups = np.asarray(identities, dtype=str)
    if values.ndim != 2 or labels.shape != (len(values),) or use.shape != labels.shape:
        raise ValueError("alias fit arrays do not align")
    if groups.shape != labels.shape or len(feature_names) != values.shape[1]:
        raise ValueError("alias fit identity/schema does not align")
    if regularization_c <= 0:
        raise ValueError("regularization C must be positive")
    if not use.any():
        return LinearAliasModel(
            tuple(feature_names),
            np.zeros(values.shape[1]),
            np.ones(values.shape[1]),
            np.zeros(values.shape[1]),
            0.0,
            0.0,
            regularization_c,
            0,
            0,
        )

    fit_x = values[use]
    fit_y = labels[use]
    fit_groups = groups[use]
    weights = identity_balanced_weights(fit_groups)
    # Equalize direct/alias total mass after identity balancing.  This affects
    # only the validation-trained evidence model, never outer-test labels.
    total_mass = float(weights.sum())
    for value in (False, True):
        class_rows = fit_y == value
        if class_rows.any():
            weights[class_rows] *= 0.5 * total_mass / weights[class_rows].sum()
    weights /= np.mean(weights)
    mean = np.average(fit_x, axis=0, weights=weights)
    variance = np.average(np.square(fit_x - mean), axis=0, weights=weights)
    scale = np.sqrt(np.maximum(variance, 1e-8))
    positive_rows = int(fit_y.sum())
    if np.unique(fit_y).size < 2:
        probability = float((positive_rows + 1.0) / (len(fit_y) + 2.0))
        return LinearAliasModel(
            tuple(feature_names),
            mean,
            scale,
            np.zeros(values.shape[1]),
            0.0,
            probability,
            regularization_c,
            int(len(fit_y)),
            positive_rows,
        )
    estimator = LogisticRegression(
        C=regularization_c,
        solver="lbfgs",
        max_iter=1000,
        random_state=20260828,
    )
    estimator.fit((fit_x - mean) / scale, fit_y.astype(int), sample_weight=weights)
    return LinearAliasModel(
        tuple(feature_names),
        mean,
        scale,
        np.asarray(estimator.coef_[0], dtype=np.float64),
        float(estimator.intercept_[0]),
        None,
        regularization_c,
        int(len(fit_y)),
        positive_rows,
    )


def fit_alias_multiplier(
    target: np.ndarray,
    classical_rr: np.ndarray,
    identities: np.ndarray,
    selected_alias: np.ndarray,
) -> float:
    selected = np.asarray(selected_alias, dtype=bool)
    ratio = np.asarray(target, dtype=np.float64) / np.maximum(
        np.asarray(classical_rr, dtype=np.float64), 1e-3
    )
    selected &= np.isfinite(ratio)
    if not selected.any():
        return 3.5
    values = np.clip(ratio[selected], 2.0, 4.0)
    weights = identity_balanced_weights(np.asarray(identities, dtype=str)[selected])
    # A continuous divisor is deliberate: validation analysis cannot reliably
    # distinguish hard x3 from x4 routing, while their convex interpolation is
    # stable and remains inside the declared candidate set.
    return float(np.clip(np.average(values, weights=weights), 2.0, 4.0))


def fit_high_alias_prior(
    target: np.ndarray,
    identities: np.ndarray,
    selected_alias: np.ndarray,
) -> float:
    """Fit an identity-balanced robust high-alias RR center on validation only."""

    values = np.asarray(target, dtype=np.float64)
    groups = np.asarray(identities, dtype=str)
    selected = (
        np.asarray(selected_alias, dtype=bool)
        & np.isfinite(values)
        & (values >= 25.0)
        & (values <= 35.0)
    )
    if not selected.any():
        # Predeclared fallback is used only when a fold has no confident alias
        # supervision; the validation guard will normally disable correction.
        return 27.5
    chosen = values[selected]
    weights = identity_balanced_weights(groups[selected])
    order = np.argsort(chosen, kind="stable")
    chosen = chosen[order]
    weights = weights[order]
    midpoint = 0.5 * float(weights.sum())
    position = int(np.searchsorted(np.cumsum(weights), midpoint, side="left"))
    return float(np.clip(chosen[min(position, len(chosen) - 1)], 20.0, 35.0))


def crossfit_alias_evidence(
    features: np.ndarray,
    alias_label: np.ndarray,
    confident: np.ndarray,
    identities: np.ndarray,
    target: np.ndarray,
    classical_rr: np.ndarray,
    feature_names: Sequence[str],
    *,
    regularization_c: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]]]:
    groups = np.asarray(identities, dtype=str)
    probability = np.zeros(len(groups), dtype=np.float64)
    multiplier = np.full(len(groups), 3.5, dtype=np.float64)
    high_alias_prior = np.full(len(groups), 27.5, dtype=np.float64)
    audit: list[dict[str, Any]] = []
    for held_out in np.unique(groups):
        holdout_rows = groups == held_out
        train_rows = ~holdout_rows
        model = fit_alias_model(
            features,
            alias_label,
            confident & train_rows,
            groups,
            feature_names,
            regularization_c=regularization_c,
        )
        probability[holdout_rows] = model.predict_probability(features[holdout_rows])
        selected_alias = confident & alias_label & train_rows
        multiplier[holdout_rows] = fit_alias_multiplier(
            target, classical_rr, groups, selected_alias
        )
        high_alias_prior[holdout_rows] = fit_high_alias_prior(
            target, groups, selected_alias
        )
        train_identities = sorted(set(groups[train_rows]))
        if held_out in train_identities:
            raise RuntimeError("LOIO alias evidence included its held-out identity")
        audit.append(
            {
                "held_out_identity": str(held_out),
                "train_identities": train_identities,
                "fit_rows": model.fit_rows,
                "positive_rows": model.positive_rows,
                "alias_multiplier": float(multiplier[holdout_rows][0]),
                "high_alias_rr_prior_bpm": float(high_alias_prior[holdout_rows][0]),
            }
        )
    return probability, multiplier, high_alias_prior, audit


def normalize_uncertainty(
    raw_uncertainty: np.ndarray,
    *,
    lower: float,
    upper: float,
) -> np.ndarray:
    values = np.asarray(raw_uncertainty, dtype=np.float64)
    width = max(float(upper - lower), 1e-8)
    return np.clip((values - lower) / width, 0.0, 1.0)


def causal_decode(
    base_prediction: np.ndarray,
    classical_rr: np.ndarray,
    classical_confidence: np.ndarray,
    alias_probability: np.ndarray,
    uncertainty: np.ndarray,
    session_id: np.ndarray,
    window_number: np.ndarray,
    alias_multiplier: float | np.ndarray,
    high_alias_prior: float | np.ndarray,
    params: DecoderParams,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply a strictly forward state machine independently within sessions."""

    base = np.asarray(base_prediction, dtype=np.float64)
    classical = np.asarray(classical_rr, dtype=np.float64)
    confidence = np.asarray(classical_confidence, dtype=np.float64)
    probability = np.clip(np.asarray(alias_probability, dtype=np.float64), 0.0, 1.0)
    uncertainty = np.clip(np.asarray(uncertainty, dtype=np.float64), 0.0, 1.0)
    sessions = np.asarray(session_id, dtype=str)
    windows = np.asarray(window_number, dtype=np.int64)
    multiplier = np.broadcast_to(
        np.asarray(alias_multiplier, dtype=np.float64), base.shape
    )
    prior = np.broadcast_to(np.asarray(high_alias_prior, dtype=np.float64), base.shape)
    arrays = (
        classical,
        confidence,
        probability,
        uncertainty,
        sessions,
        windows,
        multiplier,
        prior,
    )
    if base.ndim != 1 or any(np.asarray(value).shape != base.shape for value in arrays):
        raise ValueError("causal decoder inputs must be aligned vectors")
    if not 0.0 <= params.alias_threshold < 1.0:
        raise ValueError("alias threshold must be in [0,1)")
    if params.alias_route not in {"multiplier", "high_alias_prior", "hybrid"}:
        raise ValueError(f"unsupported alias route: {params.alias_route}")

    output = base.copy()
    state_probability = np.zeros(len(base), dtype=np.float64)
    correction_strength = np.zeros(len(base), dtype=np.float64)
    order = np.lexsort((windows, sessions))
    previous_session: str | None = None
    previous_window: int | None = None
    previous_output = 0.0
    previous_probability = 0.0
    for row in order:
        same_stream = bool(
            previous_session == sessions[row]
            and previous_window is not None
            and windows[row] == previous_window + 1
        )
        if same_stream:
            state = (
                params.state_decay * previous_probability
                + (1.0 - params.state_decay) * probability[row]
            )
        else:
            state = probability[row]
        state_probability[row] = state
        gate = np.clip(
            (state - params.alias_threshold) / (1.0 - params.alias_threshold),
            0.0,
            1.0,
        )
        reliability = np.clip(
            confidence[row] / max(params.confidence_scale, 1e-8), 0.0, 1.0
        )
        uncertainty_trust = (
            1.0 - params.uncertainty_gain
            + params.uncertainty_gain * uncertainty[row]
        )
        strength = params.alias_pull * gate * reliability * uncertainty_trust
        multiplier_candidate = np.clip(
            classical[row] * multiplier[row], params.rr_min, params.rr_max
        )
        if params.alias_route == "multiplier":
            candidate = multiplier_candidate
        elif params.alias_route == "high_alias_prior":
            candidate = prior[row]
        else:
            candidate = 0.5 * (multiplier_candidate + prior[row])
        candidate = np.clip(candidate, params.rr_min, params.rr_max)
        if not np.isfinite(candidate):
            strength = 0.0
            candidate = base[row]
        proposal = base[row] + strength * (candidate - base[row])
        if same_stream and params.continuity > 0.0:
            continuity = params.continuity * (0.5 + 0.5 * uncertainty[row])
            proposal = (1.0 - continuity) * proposal + continuity * previous_output
        decoded = float(np.clip(proposal, params.rr_min, params.rr_max))
        if params.enabled:
            output[row] = decoded
            correction_strength[row] = strength
            previous_output = decoded
        else:
            output[row] = base[row]
            previous_output = float(base[row])
        previous_probability = float(state)
        previous_session = str(sessions[row])
        previous_window = int(windows[row])
    return output, state_probability, correction_strength


def identity_macro_mae(
    target: np.ndarray, prediction: np.ndarray, identities: np.ndarray
) -> float:
    return float(identity_macro_metrics(target, prediction, identities)["macro_mae"])


def identity_macro_rate(values: np.ndarray, identities: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    groups = np.asarray(identities, dtype=str)
    return float(np.mean([np.mean(values[groups == group]) for group in np.unique(groups)]))


def validation_objective(
    target: np.ndarray,
    prediction: np.ndarray,
    identities: np.ndarray,
) -> dict[str, float]:
    target = np.asarray(target, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    groups = np.asarray(identities, dtype=str)
    full = identity_macro_mae(target, prediction, groups)
    high = (target >= 25.0) & (target <= 35.0)
    high_macro = (
        identity_macro_mae(target[high], prediction[high], groups[high])
        if high.any()
        else full
    )
    catastrophic = identity_macro_rate(np.abs(target - prediction) > 5.0, groups)
    return {
        "objective": float(full + 0.35 * high_macro + 0.5 * catastrophic),
        "macro_mae": full,
        "high_rr_25_35_macro_mae": high_macro,
        "catastrophic_over_5_macro_rate": catastrophic,
        "high_rr_rows": int(high.sum()),
    }


def select_blend(
    target: np.ndarray,
    prediction_a: np.ndarray,
    prediction_b: np.ndarray,
    identities: np.ndarray,
    *,
    step: float,
) -> tuple[float, np.ndarray, dict[str, Any]]:
    if not 0.0 < step <= 1.0:
        raise ValueError("blend step must be in (0,1]")
    grid = np.arange(0.0, 1.0 + step / 2.0, step)
    grid = np.unique(np.clip(grid, 0.0, 1.0))
    rows: list[dict[str, float]] = []
    for weight in grid:
        prediction = weight * prediction_a + (1.0 - weight) * prediction_b
        rows.append(
            {
                "weight_a": float(weight),
                "validation_macro_mae": identity_macro_mae(
                    target, prediction, identities
                ),
            }
        )
    selected = min(
        rows,
        key=lambda row: (
            row["validation_macro_mae"],
            abs(row["weight_a"] - 0.5),
            row["weight_a"],
        ),
    )
    weight = float(selected["weight_a"])
    return (
        weight,
        weight * prediction_a + (1.0 - weight) * prediction_b,
        {"selected_weight_a": weight, "search": rows},
    )


def stacked_raw_uncertainty(
    predictions: np.ndarray,
    uncertainties: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    prediction_matrix = np.asarray(predictions, dtype=np.float64)
    uncertainty_matrix = np.asarray(uncertainties, dtype=np.float64)
    weight = np.asarray(weights, dtype=np.float64)
    if (
        prediction_matrix.ndim != 2
        or uncertainty_matrix.shape != prediction_matrix.shape
        or weight.shape != (prediction_matrix.shape[1],)
        or not np.isfinite(prediction_matrix).all()
        or not np.isfinite(uncertainty_matrix).all()
        or not np.isfinite(weight).all()
        or np.any(weight < 0.0)
        or not np.isclose(weight.sum(), 1.0)
    ):
        raise ValueError("stacked uncertainty inputs are invalid")
    mean = prediction_matrix @ weight
    disagreement = np.sum(
        weight[None, :] * np.abs(prediction_matrix - mean[:, None]), axis=1
    )
    return (
        np.log1p(np.maximum(uncertainty_matrix, 0.0)) @ weight
        + 0.5 * disagreement
    )


def select_optional_third_component(
    target: np.ndarray,
    prediction_a: np.ndarray,
    prediction_b: np.ndarray,
    prediction_third: np.ndarray,
    identities: np.ndarray,
    *,
    step: float,
    minimum_improvement: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Lock a 3-way simplex only if it beats the best 2-way validation blend."""

    two_weight, two_prediction, two_report = select_blend(
        target, prediction_a, prediction_b, identities, step=step
    )
    baseline = identity_macro_mae(target, two_prediction, identities)
    levels = int(round(1.0 / step))
    if levels < 1 or not np.isclose(levels * step, 1.0, atol=1e-9):
        raise ValueError("three-way blend step must divide 1.0 exactly")
    matrix = np.column_stack([prediction_a, prediction_b, prediction_third]).astype(
        np.float64
    )
    rows: list[dict[str, Any]] = []
    for first in range(levels + 1):
        for second in range(levels - first + 1):
            third = levels - first - second
            weights = np.asarray([first, second, third], dtype=np.float64) / levels
            prediction = matrix @ weights
            rows.append(
                {
                    "weights": weights.tolist(),
                    "validation_macro_mae": identity_macro_mae(
                        target, prediction, identities
                    ),
                }
            )
    best = min(
        rows,
        key=lambda row: (
            row["validation_macro_mae"],
            row["weights"][2],
            abs(row["weights"][0] - two_weight),
            row["weights"],
        ),
    )
    improvement = float(baseline - best["validation_macro_mae"])
    selected = improvement >= minimum_improvement and best["weights"][2] > 0.0
    weights = (
        np.asarray(best["weights"], dtype=np.float64)
        if selected
        else np.asarray([two_weight, 1.0 - two_weight, 0.0], dtype=np.float64)
    )
    prediction = matrix @ weights
    return weights, prediction, {
        "source_order": ["structured_aux", "structured_exact", "alias_gate_snn"],
        "two_way": two_report,
        "two_way_validation_macro_mae": baseline,
        "three_way_grid_best": best,
        "three_way_improvement_bpm": improvement,
        "minimum_improvement_bpm": minimum_improvement,
        "third_component_selected": bool(selected),
        "selected_weights": weights.tolist(),
        "selected_validation_macro_mae": identity_macro_mae(
            target, prediction, identities
        ),
        "search": rows,
    }
def decoder_parameter_grid() -> Iterable[DecoderParams]:
    for route in ("multiplier", "high_alias_prior", "hybrid"):
        for threshold in (0.30, 0.50, 0.70, 0.85, 0.95):
            for pull in (0.50, 0.75, 1.00):
                for decay in (0.0, 0.50, 0.75):
                    for continuity in (0.0, 0.25):
                        for uncertainty_gain in (0.0, 0.50):
                            yield DecoderParams(
                                alias_route=route,
                                alias_threshold=threshold,
                                alias_pull=pull,
                                state_decay=decay,
                                continuity=continuity,
                                uncertainty_gain=uncertainty_gain,
                                confidence_scale=0.20,
                            )


def select_decoder(
    target: np.ndarray,
    base_prediction: np.ndarray,
    classical_rr: np.ndarray,
    classical_confidence: np.ndarray,
    identities: np.ndarray,
    session_id: np.ndarray,
    window_number: np.ndarray,
    raw_uncertainty: np.ndarray,
    features: np.ndarray,
    feature_names: Sequence[str],
    alias_label: np.ndarray,
    confident: np.ndarray,
    *,
    regularization_grid: Sequence[float],
    minimum_improvement: float,
    external_alias_probability: np.ndarray | None = None,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray, np.ndarray]:
    lower, upper = np.quantile(raw_uncertainty, [0.10, 0.90])
    normalized_uncertainty = normalize_uncertainty(
        raw_uncertainty, lower=float(lower), upper=float(upper)
    )
    baseline = validation_objective(target, base_prediction, identities)
    search: list[dict[str, Any]] = []
    crossfit_by_c: dict[
        float, tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]]]
    ] = {}
    for value in regularization_grid:
        crossfit_by_c[float(value)] = crossfit_alias_evidence(
            features,
            alias_label,
            confident,
            identities,
            target,
            classical_rr,
            feature_names,
            regularization_c=float(value),
        )
        meta_probability, multiplier, high_alias_prior, _ = crossfit_by_c[float(value)]
        evidence = {"validation_meta_classifier": meta_probability}
        if external_alias_probability is not None:
            external = np.clip(
                np.asarray(external_alias_probability, dtype=np.float64), 0.0, 1.0
            )
            if external.shape != meta_probability.shape or not np.isfinite(external).all():
                raise ValueError("external alias evidence must be finite and aligned")
            evidence["alias_head_probability"] = external
            evidence["equal_meta_alias_head_fusion"] = 0.5 * (
                meta_probability + external
            )
        for evidence_source, probability in evidence.items():
            for params in decoder_parameter_grid():
                prediction, _, _ = causal_decode(
                    base_prediction,
                    classical_rr,
                    classical_confidence,
                    probability,
                    normalized_uncertainty,
                    session_id,
                    window_number,
                    multiplier,
                    high_alias_prior,
                    params,
                )
                metrics = validation_objective(target, prediction, identities)
                search.append(
                    {
                        "regularization_c": float(value),
                        "evidence_source": evidence_source,
                        "params": asdict(params),
                        **metrics,
                    }
                )
    best = min(
        search,
        key=lambda row: (
            row["objective"],
            row["macro_mae"],
            row["params"]["alias_pull"],
            row["params"]["state_decay"],
            row["params"]["continuity"],
            row["params"]["alias_route"],
            row["evidence_source"],
            -row["params"]["alias_threshold"],
            row["regularization_c"],
        ),
    )
    improvement = float(baseline["objective"] - best["objective"])
    high_guard = bool(
        best["high_rr_25_35_macro_mae"]
        <= baseline["high_rr_25_35_macro_mae"] + 1e-12
    )
    full_guard = bool(best["macro_mae"] <= baseline["macro_mae"] + 0.01)
    enabled = bool(
        improvement >= minimum_improvement and high_guard and full_guard
    )
    selected_params = DecoderParams(**{**best["params"], "enabled": enabled})
    selected_c = float(best["regularization_c"])
    meta_probability, multiplier, high_alias_prior, crossfit_audit = crossfit_by_c[
        selected_c
    ]
    evidence_source = str(best["evidence_source"])
    if evidence_source == "validation_meta_classifier":
        probability = meta_probability
    elif evidence_source == "alias_head_probability":
        if external_alias_probability is None:
            raise RuntimeError("selected alias head evidence is unavailable")
        probability = np.asarray(external_alias_probability, dtype=np.float64)
    elif evidence_source == "equal_meta_alias_head_fusion":
        if external_alias_probability is None:
            raise RuntimeError("selected alias fusion evidence is unavailable")
        probability = 0.5 * (
            meta_probability + np.asarray(external_alias_probability, dtype=np.float64)
        )
    else:
        raise RuntimeError(f"unknown selected evidence source: {evidence_source}")
    crossfit_prediction, state_probability, strength = causal_decode(
        base_prediction,
        classical_rr,
        classical_confidence,
        probability,
        normalized_uncertainty,
        session_id,
        window_number,
        multiplier,
        high_alias_prior,
        selected_params,
    )
    if not enabled:
        crossfit_prediction = np.asarray(base_prediction, dtype=np.float64).copy()

    final_model = fit_alias_model(
        features,
        alias_label,
        confident,
        identities,
        feature_names,
        regularization_c=selected_c,
    )
    final_multiplier = fit_alias_multiplier(
        target,
        classical_rr,
        identities,
        confident & alias_label,
    )
    final_high_alias_prior = fit_high_alias_prior(
        target,
        identities,
        confident & alias_label,
    )
    correction_scale = float(
        max(np.quantile(np.abs(crossfit_prediction - base_prediction), 0.90), 0.25)
    )
    lock = {
        "selection_method": "leave-one-validation-identity-out alias evidence",
        "selected_params": asdict(selected_params),
        "selected_regularization_c": selected_c,
        "selected_evidence_source": evidence_source,
        "uncertainty_normalization": {
            "lower_q10": float(lower),
            "upper_q90": float(upper),
        },
        "alias_multiplier": final_multiplier,
        "high_alias_rr_prior_bpm": final_high_alias_prior,
        "high_alias_prior_fit": (
            "identity-balanced weighted median of confident validation alias targets "
            "in 25-35 bpm; predeclared 27.5 bpm fallback only when none exist"
        ),
        "high_alias_prior_support": {
            "rows": int(
                np.sum(
                    confident
                    & alias_label
                    & (np.asarray(target) >= 25.0)
                    & (np.asarray(target) <= 35.0)
                )
            ),
            "identities": sorted(
                set(
                    np.asarray(identities, dtype=str)[
                        confident
                        & alias_label
                        & (np.asarray(target) >= 25.0)
                        & (np.asarray(target) <= 35.0)
                    ]
                )
            ),
        },
        "alias_classifier": final_model.to_json(),
        "crossfit_audit": crossfit_audit,
        "confident_alias_supervision": {
            "rows": int(confident.sum()),
            "positive_rows": int((confident & alias_label).sum()),
        },
        "baseline_validation": baseline,
        "candidate_validation": {
            key: best[key]
            for key in (
                "objective",
                "macro_mae",
                "high_rr_25_35_macro_mae",
                "catastrophic_over_5_macro_rate",
                "high_rr_rows",
            )
        },
        "selected_validation": validation_objective(
            target, crossfit_prediction, identities
        ),
        "objective_improvement": improvement,
        "guards": {
            "minimum_improvement": minimum_improvement,
            "objective_passed": improvement >= minimum_improvement,
            "full_macro_degradation_le_0_01": full_guard,
            "high_rr_not_worse": high_guard,
            "decoder_enabled": enabled,
        },
        "uncertainty_correction_scale_bpm": correction_scale,
        "search": sorted(search, key=lambda row: row["objective"]),
    }
    return lock, crossfit_prediction, state_probability, strength


def locked_alias_model(specification: Mapping[str, Any]) -> LinearAliasModel:
    return LinearAliasModel(
        feature_names=tuple(specification["feature_names"]),
        mean=np.asarray(specification["mean"], dtype=np.float64),
        scale=np.asarray(specification["scale"], dtype=np.float64),
        coefficient=np.asarray(specification["coefficient"], dtype=np.float64),
        intercept=float(specification["intercept"]),
        constant_probability=(
            None
            if specification["constant_probability"] is None
            else float(specification["constant_probability"])
        ),
        regularization_c=float(specification["regularization_c"]),
        fit_rows=int(specification["fit_rows"]),
        positive_rows=int(specification["positive_rows"]),
    )


def apply_decoder_lock(
    lock: Mapping[str, Any],
    features: np.ndarray,
    context: Mapping[str, np.ndarray],
    session_id: np.ndarray,
    window_number: np.ndarray,
    external_alias_probability: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    model = locked_alias_model(lock["alias_classifier"])
    meta_probability = model.predict_probability(features)
    evidence_source = str(lock.get("selected_evidence_source", "validation_meta_classifier"))
    if evidence_source == "validation_meta_classifier":
        probability = meta_probability
    elif evidence_source == "alias_head_probability":
        if external_alias_probability is None:
            raise ValueError("locked decoder requires external alias-head probability")
        probability = np.asarray(external_alias_probability, dtype=np.float64)
    elif evidence_source == "equal_meta_alias_head_fusion":
        if external_alias_probability is None:
            raise ValueError("locked decoder requires external alias-head probability")
        probability = 0.5 * (
            meta_probability + np.asarray(external_alias_probability, dtype=np.float64)
        )
    else:
        raise ValueError(f"unknown locked evidence source: {evidence_source}")
    if probability.shape != (len(features),) or not np.isfinite(probability).all():
        raise ValueError("locked alias evidence is not finite/aligned")
    probability = np.clip(probability, 0.0, 1.0)
    normalization = lock["uncertainty_normalization"]
    uncertainty = normalize_uncertainty(
        context["raw_uncertainty"],
        lower=float(normalization["lower_q10"]),
        upper=float(normalization["upper_q90"]),
    )
    params = DecoderParams(**lock["selected_params"])
    prediction, state_probability, strength = causal_decode(
        context["blend"],
        context["classical_rr"],
        context["classical_confidence"],
        probability,
        uncertainty,
        session_id,
        window_number,
        float(lock["alias_multiplier"]),
        float(lock["high_alias_rr_prior_bpm"]),
        params,
    )
    final_uncertainty = (
        uncertainty
        + 2.0 * 4.0 * probability * (1.0 - probability)
        + np.abs(prediction - context["blend"])
        / float(lock["uncertainty_correction_scale_bpm"])
    )
    return prediction, final_uncertainty, probability, state_probability, strength


def fixed_nonoverlap_mask(
    session_id: np.ndarray,
    window_number: np.ndarray,
    *,
    windows_apart: int = 8,
) -> np.ndarray:
    sessions = np.asarray(session_id, dtype=str)
    windows = np.asarray(window_number, dtype=int)
    selected = np.zeros(len(sessions), dtype=bool)
    for session in np.unique(sessions):
        rows = np.flatnonzero(sessions == session)
        first = int(np.min(windows[rows]))
        selected[rows] = ((windows[rows] - first) % windows_apart) == 0
    return selected


def _subset_metrics(
    target: np.ndarray,
    prediction: np.ndarray,
    identities: np.ndarray,
    folds: np.ndarray,
    selected: np.ndarray,
    *,
    bootstrap_samples: int,
) -> dict[str, Any]:
    selected = np.asarray(selected, dtype=bool)
    if not selected.any():
        return {"available": False, "n": 0}
    error = np.abs(target[selected] - prediction[selected])
    return {
        "available": True,
        "n": int(selected.sum()),
        **grouped_oof_metrics(
            target[selected],
            prediction[selected],
            identities[selected],
            fold_ids=folds[selected],
            bootstrap_samples=bootstrap_samples,
            bootstrap_seed=20260828,
        ),
        "catastrophic_over_5": {
            "count": int(np.sum(error > 5.0)),
            "rate": float(np.mean(error > 5.0)),
            "identity_macro_rate": identity_macro_rate(
                error > 5.0, identities[selected]
            ),
        },
    }


def evaluation_report(
    target: np.ndarray,
    prediction: np.ndarray,
    uncertainty: np.ndarray,
    identities: np.ndarray,
    folds: np.ndarray,
    session_id: np.ndarray,
    window_number: np.ndarray,
    *,
    bootstrap_samples: int,
    coverages: Sequence[float],
) -> dict[str, Any]:
    full = np.ones(len(target), dtype=bool)
    high = (target >= 25.0) & (target <= 35.0)
    nonoverlap = fixed_nonoverlap_mask(session_id, window_number)
    bands = {
        "rr_6_15": (target >= 6.0) & (target < 15.0),
        "rr_15_25": (target >= 15.0) & (target < 25.0),
        "rr_25_35": high,
        "rr_35_45": (target > 35.0) & (target <= 45.0),
    }
    return {
        "full": _subset_metrics(
            target,
            prediction,
            identities,
            folds,
            full,
            bootstrap_samples=bootstrap_samples,
        ),
        "high_rr_25_35": _subset_metrics(
            target,
            prediction,
            identities,
            folds,
            high,
            bootstrap_samples=bootstrap_samples,
        ),
        "fixed_nonoverlap_32s": _subset_metrics(
            target,
            prediction,
            identities,
            folds,
            nonoverlap,
            bootstrap_samples=bootstrap_samples,
        ),
        "high_rr_nonoverlap_32s": _subset_metrics(
            target,
            prediction,
            identities,
            folds,
            high & nonoverlap,
            bootstrap_samples=bootstrap_samples,
        ),
        "rr_bands": {
            name: _subset_metrics(
                target,
                prediction,
                identities,
                folds,
                selected,
                bootstrap_samples=bootstrap_samples,
            )
            for name, selected in bands.items()
        },
        "selective_risk": risk_coverage_curve(
            target,
            prediction,
            uncertainty,
            coverages=coverages,
            identities=identities,
        ),
    }


def _json_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return _json_value(value.tolist())
    if isinstance(value, np.generic):
        return _json_value(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(_json_value(value), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def write_npz(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    temporary.replace(path)


def _source_validation_bundle(
    run_dir: Path,
    checkpoint: Mapping[str, Any],
    cache: Any,
    validation_index: np.ndarray,
    *,
    fold: int,
    device: torch.device,
    batch_size: int,
    workers: int,
    amp: bool,
) -> PredictionBundle:
    saved = run_dir / f"fold_{fold}" / "snn_validation_predictions.npz"
    if saved.is_file():
        return load_saved_bundle(
            saved, checkpoint, cache.metadata, fold=fold, role="validation"
        )
    return _sorted_bundle(
        infer_validation_checkpoint(
            checkpoint,
            cache,
            validation_index,
            device=device,
            batch_size=batch_size,
            workers=workers,
            amp=amp,
        )
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    run_a = args.run_a.resolve()
    run_b = args.run_b.resolve()
    alias_run = args.alias_run.resolve() if args.alias_run is not None else None
    config_a = _load_json(run_a / "run_config.json")
    config_b = _load_json(run_b / "run_config.json")
    alias_config = (
        _load_json(alias_run / "run_config.json") if alias_run is not None else None
    )
    n_folds = int(config_a["arguments"]["folds"])
    if n_folds != int(config_b["arguments"]["folds"]):
        raise ValueError("source runs use different fold counts")
    cache = prepare_source_cache(config_a, config_b, cache_dir_override=args.cache_dir)
    if alias_config is not None:
        if int(alias_config["arguments"]["folds"]) != n_folds:
            raise ValueError("alias-evidence run uses a different fold count")
        alias_cache_audit = prepare_source_cache(
            config_a, alias_config, cache_dir_override=args.cache_dir
        )
        if (
            alias_cache_audit.maps.shape != cache.maps.shape
            or alias_cache_audit.aux.shape != cache.aux.shape
            or not alias_cache_audit.metadata.index.equals(cache.metadata.index)
        ):
            raise RuntimeError("alias-evidence run reconstructs a different cache")
        del alias_cache_audit
    history_names = tuple(config_a.get("causal_history_feature_names", []))
    if history_names != tuple(config_b.get("causal_history_feature_names", [])):
        raise RuntimeError("source causal history schemas differ")

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    amp = bool(args.amp and device.type == "cuda")
    if args.num_threads:
        torch.set_num_threads(args.num_threads)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_paths = {
        "source_a": {
            str(fold): run_a / f"fold_{fold}" / "snn_best.pt"
            for fold in range(n_folds)
        },
        "source_b": {
            str(fold): run_b / f"fold_{fold}" / "snn_best.pt"
            for fold in range(n_folds)
        },
    }
    if alias_run is not None:
        checkpoint_paths["alias_source"] = {
            str(fold): alias_run / f"fold_{fold}" / "snn_best.pt"
            for fold in range(n_folds)
        }
    checkpoint_sha256 = {
        source: {fold: _sha256_file(path) for fold, path in paths.items()}
        for source, paths in checkpoint_paths.items()
    }

    signature_payload = {
        "pipeline_version": PIPELINE_VERSION,
        "source_a": config_a["run_signature"],
        "source_b": config_b["run_signature"],
        "alias_source": alias_config["run_signature"] if alias_config else None,
        "checkpoint_sha256": checkpoint_sha256,
        "blend_step": args.blend_step,
        "regularization_grid": args.regularization_grid,
        "alias_tolerance": args.alias_tolerance,
        "minimum_improvement": args.minimum_improvement,
        "third_component_minimum_improvement": (
            args.third_component_minimum_improvement
        ),
    }
    signature = hashlib.sha256(
        json.dumps(signature_payload, sort_keys=True).encode("utf-8")
    ).hexdigest()[:16]
    collected: dict[str, list[np.ndarray]] = {
        name: []
        for name in (
            "index",
            "target",
            "fold",
            "prediction_a",
            "prediction_b",
            "prediction_alias_snn",
            "two_way_prediction",
            "blend_prediction",
            "decoded_prediction",
            "uncertainty_a",
            "uncertainty_b",
            "uncertainty_alias_snn",
            "two_way_uncertainty",
            "base_uncertainty",
            "final_uncertainty",
            "alias_probability",
            "alias_state_probability",
            "alias_strength",
            "alias_head_probability",
        )
    }
    locks: dict[str, Any] = {}

    for fold in range(n_folds):
        checkpoint_a = _load_checkpoint(
            run_a / f"fold_{fold}" / "snn_best.pt",
            fold=fold,
            expected_run_signature=str(config_a["run_signature"]),
        )
        checkpoint_b = _load_checkpoint(
            run_b / f"fold_{fold}" / "snn_best.pt",
            fold=fold,
            expected_run_signature=str(config_b["run_signature"]),
        )
        alias_checkpoint = (
            _load_checkpoint(
                alias_run / f"fold_{fold}" / "snn_best.pt",
                fold=fold,
                expected_run_signature=str(alias_config["run_signature"]),
            )
            if alias_run is not None
            else None
        )
        if checkpoint_a["split"] != checkpoint_b["split"]:
            raise RuntimeError(f"source fold {fold} splits differ")
        if alias_checkpoint is not None and alias_checkpoint["split"] != checkpoint_a["split"]:
            raise RuntimeError(f"alias source fold {fold} split differs")
        split = checkpoint_a["split"]
        audit_fold_split(split)
        validation_index = indices_for_identities(
            cache.metadata, split["validation_identities"]
        )
        scaler_audit = {
            "source_a": verify_checkpoint_scaler(checkpoint_a, cache),
            "source_b": verify_checkpoint_scaler(checkpoint_b, cache),
        }
        if alias_checkpoint is not None:
            scaler_audit["alias_source"] = verify_checkpoint_scaler(
                alias_checkpoint, cache
            )

        # Selection phase: only checkpoint validation identities are inferred
        # or read.  No outer-test prediction path has been opened yet.
        validation_a = _source_validation_bundle(
            run_a,
            checkpoint_a,
            cache,
            validation_index,
            fold=fold,
            device=device,
            batch_size=args.batch_size,
            workers=args.workers,
            amp=amp,
        )
        validation_b = _source_validation_bundle(
            run_b,
            checkpoint_b,
            cache,
            validation_index,
            fold=fold,
            device=device,
            batch_size=args.batch_size,
            workers=args.workers,
            amp=amp,
        )
        validation_a, validation_b = align_prediction_bundles(
            validation_a, validation_b
        )
        validation_alias: PredictionBundle | None = None
        validation_alias_probability: np.ndarray | None = None
        if alias_checkpoint is not None and alias_run is not None:
            validation_alias = _source_validation_bundle(
                alias_run,
                alias_checkpoint,
                cache,
                validation_index,
                fold=fold,
                device=device,
                batch_size=args.batch_size,
                workers=args.workers,
                amp=amp,
            )
            validation_a, validation_alias = align_prediction_bundles(
                validation_a, validation_alias
            )
            validation_a, validation_b = align_prediction_bundles(
                validation_a, validation_b
            )
            validation_alias_probability = np.asarray(
                validation_alias.alias_probability, dtype=np.float64
            )
            if not np.isfinite(validation_alias_probability).all():
                raise RuntimeError(
                    f"alias source fold {fold} validation probability is non-finite"
                )
        validation_rows = cache.metadata.iloc[validation_a.index]
        validation_target = np.asarray(validation_a.target, dtype=np.float64)
        validation_identity = validation_rows["identity"].astype(str).to_numpy()
        validation_session = validation_rows["session_id"].astype(str).to_numpy()
        validation_window = validation_rows["window_number"].to_numpy(dtype=int)
        validation_prediction_a = np.asarray(
            validation_a.prediction, dtype=np.float64
        )
        validation_prediction_b = np.asarray(
            validation_b.prediction, dtype=np.float64
        )
        if validation_alias is not None:
            stack_weights, validation_blend, blend_report = (
                select_optional_third_component(
                    validation_target,
                    validation_prediction_a,
                    validation_prediction_b,
                    np.asarray(validation_alias.prediction, dtype=np.float64),
                    validation_identity,
                    step=args.blend_step,
                    minimum_improvement=args.third_component_minimum_improvement,
                )
            )
            validation_prediction_matrix = np.column_stack(
                [
                    validation_prediction_a,
                    validation_prediction_b,
                    np.asarray(validation_alias.prediction, dtype=np.float64),
                ]
            )
            validation_uncertainty_matrix = np.column_stack(
                [
                    validation_a.uncertainty,
                    validation_b.uncertainty,
                    validation_alias.uncertainty,
                ]
            )
            two_way_weight_a = float(
                blend_report["two_way"]["selected_weight_a"]
            )
        else:
            two_way_weight_a, validation_blend, two_way_report = select_blend(
                validation_target,
                validation_prediction_a,
                validation_prediction_b,
                validation_identity,
                step=args.blend_step,
            )
            stack_weights = np.asarray(
                [two_way_weight_a, 1.0 - two_way_weight_a], dtype=np.float64
            )
            validation_prediction_matrix = np.column_stack(
                [validation_prediction_a, validation_prediction_b]
            )
            validation_uncertainty_matrix = np.column_stack(
                [validation_a.uncertainty, validation_b.uncertainty]
            )
            blend_report = {
                "source_order": ["structured_aux", "structured_exact"],
                "two_way": two_way_report,
                "third_component_selected": False,
                "selected_weights": stack_weights.tolist(),
                "selected_validation_macro_mae": identity_macro_mae(
                    validation_target, validation_blend, validation_identity
                ),
                "reason": "no optional alias SNN run supplied",
            }
        validation_stack_uncertainty = stacked_raw_uncertainty(
            validation_prediction_matrix,
            validation_uncertainty_matrix,
            stack_weights,
        )
        features, feature_names, context = build_alias_features(
            validation_a,
            validation_b,
            cache.metadata,
            cache.aux,
            history_names,
            weight_a=two_way_weight_a,
            rr_min=float(checkpoint_b["model_kwargs"]["rr_min"]),
            rr_max=float(checkpoint_b["model_kwargs"]["rr_max"]),
            external_alias_probability=validation_alias_probability,
            stack_prediction=validation_blend,
            stack_uncertainty=validation_stack_uncertainty,
        )
        alias_label, confident, best_divisor = alias_targets(
            validation_target,
            context["classical_rr"],
            tolerance_bpm=args.alias_tolerance,
        )
        decoder_lock, validation_decoded, _, _ = select_decoder(
            validation_target,
            validation_blend,
            context["classical_rr"],
            context["classical_confidence"],
            validation_identity,
            validation_session,
            validation_window,
            context["raw_uncertainty"],
            features,
            feature_names,
            alias_label,
            confident,
            regularization_grid=args.regularization_grid,
            minimum_improvement=args.minimum_improvement,
            external_alias_probability=validation_alias_probability,
        )
        lock = {
            "fold": fold,
            "pipeline_version": PIPELINE_VERSION,
            "selection_data": "validation identities only",
            "validation_identities": list(split["validation_identities"]),
            "outer_test_identities": list(split["test_identities"]),
            "outer_test_loaded_after_lock": True,
            "source_a": {
                "path": str(run_a),
                "run_signature": checkpoint_a["run_signature"],
            },
            "source_b": {
                "path": str(run_b),
                "run_signature": checkpoint_b["run_signature"],
            },
            "alias_source": (
                {
                    "path": str(alias_run),
                    "run_signature": alias_checkpoint["run_signature"],
                    "role": (
                        "alias probability evidence plus validation-guarded optional "
                        "third RR blend component"
                    ),
                }
                if alias_checkpoint is not None and alias_run is not None
                else None
            ),
            "scaler_audit": scaler_audit,
            "validation_rows": int(len(validation_target)),
            "blend": blend_report,
            "decoder": decoder_lock,
            "alias_best_divisor_counts_confident": {
                str(divisor): int(np.sum(confident & (best_divisor == position)))
                for position, divisor in enumerate((1, 2, 3, 4))
            },
            "validation_crossfit_metrics": {
                "blend": validation_objective(
                    validation_target, validation_blend, validation_identity
                ),
                "decoder": validation_objective(
                    validation_target, validation_decoded, validation_identity
                ),
            },
            "lock_written_before_outer_test_open": True,
        }
        lock_path = output_dir / f"fold_{fold}" / "lock.json"
        write_json(lock_path, lock)
        if not lock_path.is_file():
            raise RuntimeError("fold lock was not persisted before test application")
        locks[str(fold)] = lock

        # Application phase starts after the immutable fold lock exists.
        test_a = load_saved_bundle(
            run_a / f"fold_{fold}" / "snn_test_predictions.npz",
            checkpoint_a,
            cache.metadata,
            fold=fold,
            role="test",
        )
        test_b = load_saved_bundle(
            run_b / f"fold_{fold}" / "snn_test_predictions.npz",
            checkpoint_b,
            cache.metadata,
            fold=fold,
            role="test",
        )
        test_a, test_b = align_prediction_bundles(test_a, test_b)
        test_alias: PredictionBundle | None = None
        test_alias_probability: np.ndarray | None = None
        if alias_checkpoint is not None and alias_run is not None:
            test_alias = load_saved_bundle(
                alias_run / f"fold_{fold}" / "snn_test_predictions.npz",
                alias_checkpoint,
                cache.metadata,
                fold=fold,
                role="test",
            )
            test_a, test_alias = align_prediction_bundles(test_a, test_alias)
            test_a, test_b = align_prediction_bundles(test_a, test_b)
            test_alias_probability = np.asarray(
                test_alias.alias_probability, dtype=np.float64
            )
            if not np.isfinite(test_alias_probability).all():
                raise RuntimeError(
                    f"alias source fold {fold} test probability is non-finite"
                )
        test_rows = cache.metadata.iloc[test_a.index]
        test_prediction_a = np.asarray(test_a.prediction, dtype=np.float64)
        test_prediction_b = np.asarray(test_b.prediction, dtype=np.float64)
        if test_alias is not None:
            test_prediction_matrix = np.column_stack(
                [test_prediction_a, test_prediction_b, test_alias.prediction]
            )
            test_uncertainty_matrix = np.column_stack(
                [test_a.uncertainty, test_b.uncertainty, test_alias.uncertainty]
            )
        else:
            test_prediction_matrix = np.column_stack(
                [test_prediction_a, test_prediction_b]
            )
            test_uncertainty_matrix = np.column_stack(
                [test_a.uncertainty, test_b.uncertainty]
            )
        if test_prediction_matrix.shape[1] != len(stack_weights):
            raise RuntimeError("locked stack weight/source count differs on test")
        test_stack_prediction = test_prediction_matrix @ stack_weights
        test_stack_uncertainty = stacked_raw_uncertainty(
            test_prediction_matrix, test_uncertainty_matrix, stack_weights
        )
        test_two_way_weights = np.asarray(
            [two_way_weight_a, 1.0 - two_way_weight_a], dtype=np.float64
        )
        test_two_way_prediction = (
            test_prediction_matrix[:, :2] @ test_two_way_weights
        )
        test_two_way_uncertainty = stacked_raw_uncertainty(
            test_prediction_matrix[:, :2],
            test_uncertainty_matrix[:, :2],
            test_two_way_weights,
        )
        test_features, test_names, test_context = build_alias_features(
            test_a,
            test_b,
            cache.metadata,
            cache.aux,
            history_names,
            weight_a=two_way_weight_a,
            rr_min=float(checkpoint_b["model_kwargs"]["rr_min"]),
            rr_max=float(checkpoint_b["model_kwargs"]["rr_max"]),
            external_alias_probability=test_alias_probability,
            stack_prediction=test_stack_prediction,
            stack_uncertainty=test_stack_uncertainty,
        )
        if tuple(test_names) != tuple(feature_names):
            raise RuntimeError("validation/test alias feature schemas differ")
        decoded, final_uncertainty, probability, state_probability, strength = (
            apply_decoder_lock(
                decoder_lock,
                test_features,
                test_context,
                test_rows["session_id"].astype(str).to_numpy(),
                test_rows["window_number"].to_numpy(dtype=int),
                external_alias_probability=test_alias_probability,
            )
        )
        collected["index"].append(np.asarray(test_a.index, dtype=np.int64))
        collected["target"].append(np.asarray(test_a.target, dtype=np.float32))
        collected["fold"].append(np.full(len(test_a), fold, dtype=np.int16))
        collected["prediction_a"].append(
            np.asarray(test_a.prediction, dtype=np.float32)
        )
        collected["prediction_b"].append(
            np.asarray(test_b.prediction, dtype=np.float32)
        )
        collected["prediction_alias_snn"].append(
            (
                np.asarray(test_alias.prediction, dtype=np.float32)
                if test_alias is not None
                else np.full(len(test_a), np.nan, dtype=np.float32)
            )
        )
        collected["two_way_prediction"].append(
            test_two_way_prediction.astype(np.float32)
        )
        collected["blend_prediction"].append(
            np.asarray(test_context["blend"], dtype=np.float32)
        )
        collected["decoded_prediction"].append(decoded.astype(np.float32))
        collected["uncertainty_a"].append(
            np.asarray(test_a.uncertainty, dtype=np.float32)
        )
        collected["uncertainty_b"].append(
            np.asarray(test_b.uncertainty, dtype=np.float32)
        )
        collected["uncertainty_alias_snn"].append(
            (
                np.asarray(test_alias.uncertainty, dtype=np.float32)
                if test_alias is not None
                else np.full(len(test_a), np.nan, dtype=np.float32)
            )
        )
        collected["two_way_uncertainty"].append(
            test_two_way_uncertainty.astype(np.float32)
        )
        collected["base_uncertainty"].append(
            np.asarray(test_context["raw_uncertainty"], dtype=np.float32)
        )
        collected["final_uncertainty"].append(final_uncertainty.astype(np.float32))
        collected["alias_probability"].append(probability.astype(np.float32))
        collected["alias_state_probability"].append(
            state_probability.astype(np.float32)
        )
        collected["alias_strength"].append(strength.astype(np.float32))
        collected["alias_head_probability"].append(
            (
                test_alias_probability.astype(np.float32)
                if test_alias_probability is not None
                else np.full(len(test_a), np.nan, dtype=np.float32)
            )
        )
        print(
            f"fold={fold} stack={np.round(stack_weights, 2).tolist()} "
            f"decoder={decoder_lock['guards']['decoder_enabled']} "
            f"val_macro={decoder_lock['selected_validation']['macro_mae']:.4f}",
            flush=True,
        )

    concatenated = {key: np.concatenate(value) for key, value in collected.items()}
    order = np.argsort(concatenated["index"], kind="stable")
    arrays = {key: value[order] for key, value in concatenated.items()}
    expected = np.flatnonzero(cache.metadata["reference_valid"].to_numpy(dtype=bool))
    if not np.array_equal(arrays["index"], expected):
        raise RuntimeError("outer-fold rows are not a complete unique valid OOF")
    rows = cache.metadata.iloc[arrays["index"]]
    identities = rows["identity"].astype(str).to_numpy()
    sessions = rows["session_id"].astype(str).to_numpy()
    windows = rows["window_number"].to_numpy(dtype=int)
    target = arrays["target"].astype(np.float64)
    folds = arrays["fold"].astype(int)

    write_npz(
        output_dir / "causal_alias_decoder_oof.npz",
        **arrays,
        run_signature=np.asarray(signature),
        source_labels=np.asarray(
            ["structured_aux", "structured_exact"]
            + (["alias_gate_snn"] if alias_run is not None else [])
        ),
        alias_source=np.asarray(str(alias_run) if alias_run is not None else ""),
    )
    columns = [
        "session_id",
        "identity",
        "protocol",
        "window_number",
        "window_start_s",
        "window_end_s",
        "rr_bpm",
        "reference_quality",
        "radar_observable",
        "classical_rr_bpm",
        "classical_confidence",
        "radar_peak_spread_bpm",
    ]
    table = rows[columns].reset_index(drop=True)
    table.insert(0, "cache_index", arrays["index"])
    table.insert(1, "fold", arrays["fold"])
    for name in (
        "prediction_a",
        "prediction_b",
        "prediction_alias_snn",
        "two_way_prediction",
        "blend_prediction",
        "decoded_prediction",
        "uncertainty_a",
        "uncertainty_b",
        "uncertainty_alias_snn",
        "two_way_uncertainty",
        "base_uncertainty",
        "final_uncertainty",
        "alias_probability",
        "alias_state_probability",
        "alias_strength",
        "alias_head_probability",
    ):
        table[name] = arrays[name]
    table.to_csv(output_dir / "causal_alias_decoder_oof.csv", index=False)

    variants = {
        "structured_aux": (arrays["prediction_a"], arrays["uncertainty_a"]),
        "structured_exact": (arrays["prediction_b"], arrays["uncertainty_b"]),
        "validation_locked_two_way_blend": (
            arrays["two_way_prediction"],
            arrays["two_way_uncertainty"],
        ),
        "validation_locked_blend": (
            arrays["blend_prediction"],
            arrays["base_uncertainty"],
        ),
        "validation_locked_causal_alias_decoder": (
            arrays["decoded_prediction"],
            arrays["final_uncertainty"],
        ),
    }
    if alias_run is not None:
        variants["alias_gate_snn_rr_diagnostic"] = (
            arrays["prediction_alias_snn"],
            arrays["uncertainty_alias_snn"],
        )
    metrics = {
        name: evaluation_report(
            target,
            np.asarray(prediction, dtype=np.float64),
            np.asarray(uncertainty, dtype=np.float64),
            identities,
            folds,
            sessions,
            windows,
            bootstrap_samples=args.bootstrap_samples,
            coverages=args.coverages,
        )
        for name, (prediction, uncertainty) in variants.items()
    }
    baseline = metrics["validation_locked_blend"]
    two_way_baseline = metrics["validation_locked_two_way_blend"]
    decoder = metrics["validation_locked_causal_alias_decoder"]
    deltas = {
        "full_macro_mae_bpm": float(
            decoder["full"]["identity_macro"]["macro_mae"]
            - baseline["full"]["identity_macro"]["macro_mae"]
        ),
        "high_rr_25_35_macro_mae_bpm": float(
            decoder["high_rr_25_35"]["identity_macro"]["macro_mae"]
            - baseline["high_rr_25_35"]["identity_macro"]["macro_mae"]
        ),
        "nonoverlap_macro_mae_bpm": float(
            decoder["fixed_nonoverlap_32s"]["identity_macro"]["macro_mae"]
            - baseline["fixed_nonoverlap_32s"]["identity_macro"]["macro_mae"]
        ),
        "catastrophic_over_5_rate": float(
            decoder["full"]["catastrophic_over_5"]["rate"]
            - baseline["full"]["catastrophic_over_5"]["rate"]
        ),
    }
    selected_folds = int(
        sum(lock["decoder"]["guards"]["decoder_enabled"] for lock in locks.values())
    )
    third_component_selected_folds = int(
        sum(lock["blend"]["third_component_selected"] for lock in locks.values())
    )
    static_stack_delta = float(
        baseline["full"]["identity_macro"]["macro_mae"]
        - two_way_baseline["full"]["identity_macro"]["macro_mae"]
    )
    static_stack_deltas = {
        "full_macro_mae_bpm": static_stack_delta,
        "high_rr_25_35_macro_mae_bpm": float(
            baseline["high_rr_25_35"]["identity_macro"]["macro_mae"]
            - two_way_baseline["high_rr_25_35"]["identity_macro"]["macro_mae"]
        ),
        "nonoverlap_macro_mae_bpm": float(
            baseline["fixed_nonoverlap_32s"]["identity_macro"]["macro_mae"]
            - two_way_baseline["fixed_nonoverlap_32s"]["identity_macro"][
                "macro_mae"
            ]
        ),
        "catastrophic_over_5_rate": float(
            baseline["full"]["catastrophic_over_5"]["rate"]
            - two_way_baseline["full"]["catastrophic_over_5"]["rate"]
        ),
    }
    static_stack_supported = bool(
        third_component_selected_folds > 0
        and static_stack_deltas["full_macro_mae_bpm"] <= -0.01
        and static_stack_deltas["high_rr_25_35_macro_mae_bpm"] <= 0.0
        and static_stack_deltas["nonoverlap_macro_mae_bpm"] <= 0.0
        and static_stack_deltas["catastrophic_over_5_rate"] <= 0.0
    )
    final_vs_two_way_deltas = {
        "full_macro_mae_bpm": float(
            decoder["full"]["identity_macro"]["macro_mae"]
            - two_way_baseline["full"]["identity_macro"]["macro_mae"]
        ),
        "high_rr_25_35_macro_mae_bpm": float(
            decoder["high_rr_25_35"]["identity_macro"]["macro_mae"]
            - two_way_baseline["high_rr_25_35"]["identity_macro"]["macro_mae"]
        ),
        "nonoverlap_macro_mae_bpm": float(
            decoder["fixed_nonoverlap_32s"]["identity_macro"]["macro_mae"]
            - two_way_baseline["fixed_nonoverlap_32s"]["identity_macro"][
                "macro_mae"
            ]
        ),
        "catastrophic_over_5_rate": float(
            decoder["full"]["catastrophic_over_5"]["rate"]
            - two_way_baseline["full"]["catastrophic_over_5"]["rate"]
        ),
    }
    causal_decoder_supported = bool(
        selected_folds > 0
        and deltas["full_macro_mae_bpm"] <= -0.01
        and deltas["high_rr_25_35_macro_mae_bpm"] <= 0.0
        and deltas["nonoverlap_macro_mae_bpm"] <= 0.0
        and deltas["catastrophic_over_5_rate"] <= 0.0
        and final_vs_two_way_deltas["full_macro_mae_bpm"] <= -0.01
        and final_vs_two_way_deltas["high_rr_25_35_macro_mae_bpm"] <= 0.0
        and final_vs_two_way_deltas["nonoverlap_macro_mae_bpm"] <= 0.0
        and final_vs_two_way_deltas["catastrophic_over_5_rate"] <= 0.0
    )
    decision, deployment_variant = choose_deployment_variant(
        static_stack_supported=static_stack_supported,
        causal_decoder_supported=causal_decoder_supported,
    )
    conclusion = {
        "decision": decision,
        "deployment_variant": deployment_variant,
        "deployment_prediction_column": {
            "validation_locked_causal_alias_decoder": "decoded_prediction",
            "validation_locked_blend": "blend_prediction",
            "validation_locked_two_way_blend": "two_way_prediction",
        }[deployment_variant],
        "selected_folds": selected_folds,
        "n_folds": n_folds,
        "third_component_selected_folds": third_component_selected_folds,
        "outer_oof_selected_static_minus_two_way_macro_mae_bpm": static_stack_delta,
        "static_stack_evidence": {
            "deltas_selected_static_minus_two_way": static_stack_deltas,
            "supported": static_stack_supported,
            "outer_oof_is_pristine_confirmatory": False,
        },
        "causal_decoder_evidence": {
            "supported": causal_decoder_supported,
            "outer_oof_is_pristine_confirmatory": False,
        },
        "outer_oof_role": (
            "retrospective acceptance/rejection audit after model iteration; this is "
            "selection evidence and not a pristine prospective confirmatory test"
        ),
        "deltas_decoder_minus_locked_blend": deltas,
        "deltas_final_decoder_minus_locked_two_way": final_vs_two_way_deltas,
        "acceptance_rule": (
            "the static stack and causal decoder are assessed independently; a candidate "
            "must improve full macro >=0.01 bpm against its required locked baseline "
            "without regression in 25-35, fixed non-overlap, or >5 bpm failure rate"
        ),
        "reason": (
            f"{deployment_variant} satisfied all retrospective deployment-evidence guards"
            if decision == "accept"
            else "neither locked alias candidate passed all retrospective outer-OOF "
            "guards; both are rejected without post-hoc retuning"
        ),
    }
    report = {
        "method": "validation-locked causal alias-state decoder",
        "pipeline_version": PIPELINE_VERSION,
        "run_signature": signature,
        "source_checkpoint_sha256": checkpoint_sha256,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "selection_guarantee": (
            "Every fold lock is selected from validation identities and atomically "
            "written before that fold's outer-test prediction files are opened."
        ),
        "causal_contract": (
            "Features contain current radar/model observations and strictly past cache "
            "history only. Recursive state advances in window order and resets at session "
            "or window gaps; targets are never accepted by the application API."
        ),
        "device": str(device),
        "amp": amp,
        "n_rows": int(len(target)),
        "n_identities": int(len(np.unique(identities))),
        "fold_locks": locks,
        "metrics": metrics,
        "conclusion": conclusion,
    }
    write_json(output_dir / "metrics.json", report)
    write_json(
        output_dir / "run_config.json",
        {
            "run_signature": signature,
            "pipeline_version": PIPELINE_VERSION,
            "source_a": str(run_a),
            "source_b": str(run_b),
            "alias_source": str(alias_run) if alias_run is not None else None,
            "source_checkpoint_sha256": checkpoint_sha256,
            "arguments": vars(args),
            "device": str(device),
            "amp": amp,
        },
    )
    print(
        f"completed decision={conclusion['decision']} "
        f"macro_delta={deltas['full_macro_mae_bpm']:+.4f} "
        f"high_rr_delta={deltas['high_rr_25_35_macro_mae_bpm']:+.4f}",
        flush=True,
    )
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validation-locked causal harmonic-alias decoder"
    )
    parser.add_argument(
        "--run-a",
        type=Path,
        default=Path("artifacts/runs/final_structured_aux_s12"),
    )
    parser.add_argument(
        "--run-b",
        type=Path,
        default=Path("artifacts/runs/final_structured_exact_s12_deterministic"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/runs/causal_alias_decoder"),
    )
    parser.add_argument(
        "--alias-run",
        type=Path,
        help=(
            "optional full alias-gated SNN run; its alias probability is evidence and "
            "its RR is admitted only by a validation-guarded static simplex"
        ),
    )
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--num-threads", type=int, default=4)
    parser.add_argument("--blend-step", type=float, default=0.05)
    parser.add_argument(
        "--regularization-grid", type=float, nargs="+", default=[0.03, 0.10, 0.30]
    )
    parser.add_argument("--alias-tolerance", type=float, default=2.0)
    parser.add_argument("--minimum-improvement", type=float, default=0.01)
    parser.add_argument(
        "--third-component-minimum-improvement", type=float, default=0.005
    )
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument(
        "--coverages", type=float, nargs="+", default=list(DEFAULT_COVERAGES)
    )
    parser.add_argument("--seed", type=int, default=20260828)
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.batch_size < 1 or args.workers < 0 or args.num_threads < 1:
        parser.error("batch size/threads must be positive and workers non-negative")
    if not 0 < args.blend_step <= 1:
        parser.error("--blend-step must be in (0,1]")
    if any(value <= 0 for value in args.regularization_grid):
        parser.error("regularization values must be positive")
    if (
        args.alias_tolerance <= 0
        or args.minimum_improvement < 0
        or args.third_component_minimum_improvement < 0
    ):
        parser.error("alias tolerance must be positive and improvement non-negative")
    if args.bootstrap_samples < 1:
        parser.error("bootstrap samples must be positive")
    if any(not 0 < coverage <= 1 for coverage in args.coverages):
        parser.error("coverages must be in (0,1]")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    run(parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
