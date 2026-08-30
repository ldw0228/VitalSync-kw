#!/usr/bin/env python3
"""Leakage-safe commercial-candidate stack over four trained RR models.

This experiment combines the default/compact SNN and ANN-teacher checkpoints.
For each outer fold every weight, calibration rule, uncertainty rule, and causal
state-space parameter is selected using that fold's validation identities.
Only after the complete rule is frozen are outer-test prediction files opened.

The output is intentionally an evaluation artifact, not a commercial or
medical claim.  It reports rejected stages as well as selected stages and
keeps full-coverage, identity-macro, high-RR, non-overlap, and selective-risk
metrics separate.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
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
from sklearn.isotonic import IsotonicRegression


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
for search_path in (PROJECT_ROOT, SRC_ROOT):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

from snn_rr.metrics import (  # noqa: E402
    grouped_oof_metrics,
    identity_macro_metrics,
    regression_metrics,
    risk_coverage_curve,
)
from scripts.ensemble import (  # noqa: E402
    PredictionBundle,
    align_prediction_bundles,
    infer_validation_checkpoint,
    prepare_source_cache,
    verify_checkpoint_scaler,
)
from scripts.train import load_prediction_bundle, write_json  # noqa: E402


DEFAULT_COVERAGES = (1.0, 0.9, 0.8, 0.7, 0.5)
SELECTION_COVERAGES = (0.9, 0.8, 0.7, 0.5)
PIPELINE_VERSION = 2


@dataclass(frozen=True, slots=True)
class SourceSpec:
    label: str
    run_dir: Path
    model_type: str


def _finite_vector(value: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.ndim != 1 or not len(array):
        raise ValueError(f"{name} must be a non-empty vector")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite")
    return array


def identity_balanced_weights(identities: Sequence[str] | np.ndarray) -> np.ndarray:
    """Return sample weights whose total mass is equal for every identity."""

    groups = np.asarray(identities, dtype=str)
    if groups.ndim != 1 or not len(groups):
        raise ValueError("identities must be a non-empty vector")
    weights = np.zeros(len(groups), dtype=float)
    names = np.unique(groups)
    for name in names:
        selected = groups == name
        weights[selected] = 1.0 / (len(names) * int(selected.sum()))
    return weights


def identity_macro_mae_fast(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    identities: Sequence[str] | np.ndarray,
) -> float:
    target = _finite_vector(y_true, "target")
    prediction = _finite_vector(y_pred, "prediction")
    if target.shape != prediction.shape:
        raise ValueError("target and prediction shapes differ")
    weights = identity_balanced_weights(identities)
    if len(weights) != len(target):
        raise ValueError("identity count differs from predictions")
    return float(np.sum(weights * np.abs(prediction - target)))


def simplex_grid(n_models: int, step: float) -> np.ndarray:
    """Enumerate a deterministic non-negative simplex grid."""

    if n_models < 1 or not 0 < step <= 1:
        raise ValueError("n_models and step must be positive")
    units = int(round(1.0 / step))
    if not math.isclose(units * step, 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError("step must divide one exactly")
    rows: list[list[int]] = []

    def recurse(prefix: list[int], remaining: int, dimensions: int) -> None:
        if dimensions == 1:
            rows.append([*prefix, remaining])
            return
        for value in range(remaining + 1):
            recurse([*prefix, value], remaining - value, dimensions - 1)

    recurse([], units, n_models)
    return np.asarray(rows, dtype=float) / float(units)


def apply_static_stack(predictions: np.ndarray, weights: np.ndarray) -> np.ndarray:
    matrix = np.asarray(predictions, dtype=float)
    coefficients = np.asarray(weights, dtype=float)
    if matrix.ndim != 2 or coefficients.shape != (matrix.shape[1],):
        raise ValueError("prediction matrix and stack weights are incompatible")
    if np.any(coefficients < 0) or not np.isclose(coefficients.sum(), 1.0):
        raise ValueError("stack weights must lie on the simplex")
    return matrix @ coefficients


def select_convex_stack(
    y_true: np.ndarray,
    predictions: np.ndarray,
    identities: Sequence[str] | np.ndarray,
    *,
    step: float = 0.025,
    minimum_improvement: float = 0.005,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Select a convex stack by validation identity-macro MAE.

    A small improvement guard prefers the best single component when a dense
    blend does not materially improve validation risk.
    """

    target = _finite_vector(y_true, "target")
    matrix = np.asarray(predictions, dtype=float)
    if matrix.ndim != 2 or len(matrix) != len(target) or not np.isfinite(matrix).all():
        raise ValueError("predictions must be a finite [N, M] matrix")
    sample_weight = identity_balanced_weights(identities)
    if len(sample_weight) != len(target):
        raise ValueError("identity count differs from predictions")
    candidates = simplex_grid(matrix.shape[1], step)
    best_objective = float("inf")
    best_weights: np.ndarray | None = None
    scored: list[tuple[float, np.ndarray]] = []
    # Chunking bounds peak memory for finer future grids.
    for start in range(0, len(candidates), 2048):
        block = candidates[start : start + 2048]
        block_prediction = matrix @ block.T
        objectives = np.sum(
            sample_weight[:, None] * np.abs(block_prediction - target[:, None]),
            axis=0,
        )
        for local in np.argsort(objectives)[:10]:
            scored.append((float(objectives[local]), block[local].copy()))
        local_best = int(np.argmin(objectives))
        objective = float(objectives[local_best])
        if objective < best_objective - 1e-12:
            best_objective = objective
            best_weights = block[local_best].copy()
    assert best_weights is not None
    component_objectives = np.asarray(
        [
            identity_macro_mae_fast(target, matrix[:, column], identities)
            for column in range(matrix.shape[1])
        ]
    )
    best_component = int(np.argmin(component_objectives))
    component_weights = np.eye(matrix.shape[1], dtype=float)[best_component]
    improvement = float(component_objectives[best_component] - best_objective)
    guard_selected_component = improvement < minimum_improvement
    selected = component_weights if guard_selected_component else best_weights
    selected_objective = identity_macro_mae_fast(
        target, apply_static_stack(matrix, selected), identities
    )
    top = sorted(scored, key=lambda item: (item[0], tuple(item[1])))[:20]
    return selected, {
        "selected_weights": selected.tolist(),
        "grid_best_weights": best_weights.tolist(),
        "grid_best_macro_mae": best_objective,
        "best_component": best_component,
        "best_component_macro_mae": float(component_objectives[best_component]),
        "component_macro_mae": component_objectives.tolist(),
        "grid_improvement_over_best_component": improvement,
        "minimum_improvement": minimum_improvement,
        "guard_selected_single_component": guard_selected_component,
        "selected_macro_mae": selected_objective,
        "grid_size": int(len(candidates)),
        "top_grid_candidates": [
            {"macro_mae": objective, "weights": weights.tolist()}
            for objective, weights in top
        ],
    }


def uncertainty_gated_weights(
    base_weights: np.ndarray,
    uncertainty: np.ndarray,
    score_scale: np.ndarray,
    *,
    alpha: float,
    uniform_floor: float,
) -> np.ndarray:
    """Reliability-gate a static stack without target information."""

    base = np.asarray(base_weights, dtype=float)
    scores = np.asarray(uncertainty, dtype=float)
    scale = np.asarray(score_scale, dtype=float)
    if scores.ndim != 2 or base.shape != (scores.shape[1],):
        raise ValueError("base weights and uncertainty matrix are incompatible")
    if scale.shape != base.shape or np.any(scale <= 0):
        raise ValueError("score_scale must be positive per model")
    if alpha < 0 or not 0 <= uniform_floor < 1:
        raise ValueError("invalid gating alpha or floor")
    prior = (1.0 - uniform_floor) * base + uniform_floor / len(base)
    relative = np.clip(np.maximum(scores, 1e-6) / scale[None, :], 0.05, 20.0)
    reliability = np.exp(-alpha * np.clip(np.log(relative), -3.0, 3.0))
    dynamic = prior[None, :] * reliability
    normalizer = dynamic.sum(axis=1, keepdims=True)
    if np.any(normalizer <= 0):
        raise RuntimeError("uncertainty gating produced zero total weight")
    return dynamic / normalizer


def select_uncertainty_gate(
    y_true: np.ndarray,
    predictions: np.ndarray,
    uncertainties: np.ndarray,
    identities: Sequence[str] | np.ndarray,
    base_weights: np.ndarray,
    *,
    alphas: Sequence[float] = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0),
    floors: Sequence[float] = (0.0, 0.02, 0.05, 0.10),
    minimum_improvement: float = 0.005,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    """Select uncertainty-aware dynamic weights with a validation guard."""

    target = _finite_vector(y_true, "target")
    matrix = np.asarray(predictions, dtype=float)
    scores = np.asarray(uncertainties, dtype=float)
    if matrix.shape != scores.shape or matrix.ndim != 2 or len(matrix) != len(target):
        raise ValueError("prediction and uncertainty matrices must match")
    score_scale = np.maximum(np.median(scores, axis=0), 1e-6)
    baseline_dynamic = uncertainty_gated_weights(
        base_weights, scores, score_scale, alpha=0.0, uniform_floor=0.0
    )
    baseline_prediction = np.sum(baseline_dynamic * matrix, axis=1)
    baseline_objective = identity_macro_mae_fast(
        target, baseline_prediction, identities
    )
    rows: list[dict[str, float]] = []
    best: tuple[float, float, float, np.ndarray, np.ndarray] | None = None
    for alpha in alphas:
        for floor in floors:
            dynamic = uncertainty_gated_weights(
                base_weights,
                scores,
                score_scale,
                alpha=float(alpha),
                uniform_floor=float(floor),
            )
            prediction = np.sum(dynamic * matrix, axis=1)
            objective = identity_macro_mae_fast(target, prediction, identities)
            rows.append(
                {
                    "alpha": float(alpha),
                    "uniform_floor": float(floor),
                    "validation_macro_mae": objective,
                }
            )
            key = (objective, float(alpha), float(floor), dynamic, prediction)
            if best is None or key[:3] < best[:3]:
                best = key
    assert best is not None
    improvement = baseline_objective - best[0]
    guarded = improvement < minimum_improvement
    alpha = 0.0 if guarded else best[1]
    floor = 0.0 if guarded else best[2]
    selected_dynamic = uncertainty_gated_weights(
        base_weights, scores, score_scale, alpha=alpha, uniform_floor=floor
    )
    selected_prediction = np.sum(selected_dynamic * matrix, axis=1)
    rule = {
        "alpha": alpha,
        "uniform_floor": floor,
        "score_scale": score_scale.tolist(),
        "baseline_macro_mae": baseline_objective,
        "candidate_best_macro_mae": best[0],
        "candidate_improvement": improvement,
        "minimum_improvement": minimum_improvement,
        "guard_rejected_gating": guarded,
        "selected_macro_mae": identity_macro_mae_fast(
            target, selected_prediction, identities
        ),
        "search": sorted(rows, key=lambda row: row["validation_macro_mae"]),
    }
    return rule, selected_dynamic, selected_prediction


def weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    values = _finite_vector(values, "values")
    weights = _finite_vector(weights, "weights")
    if values.shape != weights.shape or np.any(weights < 0) or weights.sum() <= 0:
        raise ValueError("invalid weighted-median inputs")
    order = np.argsort(values, kind="stable")
    cumulative = np.cumsum(weights[order])
    index = int(np.searchsorted(cumulative, 0.5 * weights.sum(), side="left"))
    return float(values[order[min(index, len(values) - 1)]])


def _fit_identity_calibrator(
    prediction: np.ndarray,
    target: np.ndarray,
    identities: np.ndarray,
) -> dict[str, Any]:
    del prediction, target, identities
    return {"kind": "identity"}


def _fit_robust_affine(
    prediction: np.ndarray,
    target: np.ndarray,
    identities: np.ndarray,
    *,
    slope_bounds: tuple[float, float] = (0.7, 1.5),
    intercept_bounds: tuple[float, float] = (-5.0, 5.0),
) -> dict[str, Any]:
    weights = identity_balanced_weights(identities)
    best: tuple[float, float, float] | None = None
    for slope in np.linspace(slope_bounds[0], slope_bounds[1], 33):
        intercept = float(
            np.clip(
                weighted_median(target - slope * prediction, weights),
                intercept_bounds[0],
                intercept_bounds[1],
            )
        )
        calibrated = slope * prediction + intercept
        objective = float(np.sum(weights * np.abs(calibrated - target)))
        key = (objective, abs(float(slope) - 1.0), abs(intercept))
        if best is None or key < (best[0], abs(best[1] - 1.0), abs(best[2])):
            best = (objective, float(slope), intercept)
    assert best is not None
    return {
        "kind": "robust_affine",
        "slope": best[1],
        "intercept": best[2],
        "fit_macro_mae": best[0],
        "slope_bounds": list(slope_bounds),
        "intercept_bounds": list(intercept_bounds),
    }


def _fit_robust_hinge(
    prediction: np.ndarray,
    target: np.ndarray,
    identities: np.ndarray,
) -> dict[str, Any]:
    """Fit a monotone two-slope piecewise-linear calibration."""

    weights = identity_balanced_weights(identities)
    best: tuple[float, float, float, float, float] | None = None
    for knot in (16.0, 18.0, 20.0, 22.0, 24.0):
        for low_slope in (0.8, 0.9, 1.0, 1.1, 1.2):
            for high_slope in (0.9, 1.0, 1.1, 1.25, 1.5):
                base = (
                    low_slope * prediction
                    + (high_slope - low_slope) * np.maximum(prediction - knot, 0.0)
                )
                intercept = float(
                    np.clip(weighted_median(target - base, weights), -5.0, 5.0)
                )
                calibrated = base + intercept
                objective = float(np.sum(weights * np.abs(calibrated - target)))
                key = (
                    objective,
                    abs(low_slope - 1.0) + abs(high_slope - 1.0),
                    abs(intercept),
                    knot,
                )
                if best is None or key < (
                    best[0],
                    abs(best[2] - 1.0) + abs(best[3] - 1.0),
                    abs(best[4]),
                    best[1],
                ):
                    best = (
                        objective,
                        knot,
                        low_slope,
                        high_slope,
                        intercept,
                    )
    assert best is not None
    return {
        "kind": "robust_hinge",
        "knot": best[1],
        "low_slope": best[2],
        "high_slope": best[3],
        "intercept": best[4],
        "fit_macro_mae": best[0],
    }


def _fit_isotonic(
    prediction: np.ndarray,
    target: np.ndarray,
    identities: np.ndarray,
) -> dict[str, Any]:
    weights = identity_balanced_weights(identities)
    estimator = IsotonicRegression(
        y_min=6.0,
        y_max=45.0,
        increasing=True,
        out_of_bounds="clip",
    )
    estimator.fit(prediction, target, sample_weight=weights)
    return {
        "kind": "isotonic_piecewise",
        "x_thresholds": np.asarray(estimator.X_thresholds_, dtype=float).tolist(),
        "y_thresholds": np.asarray(estimator.y_thresholds_, dtype=float).tolist(),
    }


CALIBRATOR_FITTERS = {
    "identity": _fit_identity_calibrator,
    "robust_affine": _fit_robust_affine,
    "robust_hinge": _fit_robust_hinge,
    "isotonic_piecewise": _fit_isotonic,
}


def fit_calibrator(
    kind: str,
    prediction: np.ndarray,
    target: np.ndarray,
    identities: Sequence[str] | np.ndarray,
) -> dict[str, Any]:
    prediction = _finite_vector(prediction, "prediction")
    target = _finite_vector(target, "target")
    groups = np.asarray(identities, dtype=str)
    if prediction.shape != target.shape or groups.shape != target.shape:
        raise ValueError("calibration inputs have different shapes")
    try:
        fitter = CALIBRATOR_FITTERS[kind]
    except KeyError as exc:
        raise ValueError(f"unknown calibrator: {kind}") from exc
    return fitter(prediction, target, groups)


def apply_calibrator(prediction: np.ndarray, spec: Mapping[str, Any]) -> np.ndarray:
    value = np.asarray(prediction, dtype=float)
    kind = spec["kind"]
    if kind == "identity":
        calibrated = value
    elif kind == "robust_affine":
        calibrated = float(spec["slope"]) * value + float(spec["intercept"])
    elif kind == "robust_hinge":
        knot = float(spec["knot"])
        low = float(spec["low_slope"])
        high = float(spec["high_slope"])
        calibrated = (
            low * value
            + (high - low) * np.maximum(value - knot, 0.0)
            + float(spec["intercept"])
        )
    elif kind == "isotonic_piecewise":
        calibrated = np.interp(
            value,
            np.asarray(spec["x_thresholds"], dtype=float),
            np.asarray(spec["y_thresholds"], dtype=float),
        )
    else:
        raise ValueError(f"unknown calibrator: {kind}")
    return np.clip(calibrated, 6.0, 45.0)


def select_calibrator_leave_one_identity_out(
    y_true: np.ndarray,
    prediction: np.ndarray,
    identities: Sequence[str] | np.ndarray,
    *,
    candidates: Sequence[str] = (
        "identity",
        "robust_affine",
        "robust_hinge",
        "isotonic_piecewise",
    ),
    minimum_improvement: float = 0.01,
) -> tuple[dict[str, Any], np.ndarray, dict[str, Any]]:
    """Choose calibration by leave-one-validation-identity-out risk."""

    target = _finite_vector(y_true, "target")
    raw = _finite_vector(prediction, "prediction")
    groups = np.asarray(identities, dtype=str)
    if target.shape != raw.shape or groups.shape != raw.shape:
        raise ValueError("calibration inputs have different shapes")
    names = np.unique(groups)
    if len(names) < 3:
        raise ValueError("LOIO calibration requires at least three identities")
    rows: list[dict[str, Any]] = []
    for kind in candidates:
        cross_prediction = np.empty_like(raw)
        fold_specs: dict[str, Any] = {}
        for held_out in names:
            train = groups != held_out
            held = ~train
            spec = fit_calibrator(kind, raw[train], target[train], groups[train])
            cross_prediction[held] = apply_calibrator(raw[held], spec)
            fold_specs[str(held_out)] = spec
        objective = identity_macro_mae_fast(target, cross_prediction, groups)
        rows.append(
            {
                "kind": kind,
                "loio_macro_mae": objective,
                "held_out_specs": fold_specs,
            }
        )
    baseline = next(row for row in rows if row["kind"] == "identity")
    candidate = min(rows, key=lambda row: (row["loio_macro_mae"], row["kind"]))
    improvement = float(baseline["loio_macro_mae"] - candidate["loio_macro_mae"])
    guarded = candidate["kind"] == "identity" or improvement < minimum_improvement
    selected_kind = "identity" if guarded else str(candidate["kind"])
    final_spec = fit_calibrator(selected_kind, raw, target, groups)
    calibrated = apply_calibrator(raw, final_spec)
    report = {
        "selected_kind": selected_kind,
        "selected_spec": final_spec,
        "loio_best_kind": candidate["kind"],
        "loio_best_macro_mae": candidate["loio_macro_mae"],
        "loio_identity_macro_mae": baseline["loio_macro_mae"],
        "loio_improvement": improvement,
        "minimum_improvement": minimum_improvement,
        "guard_rejected_calibration": guarded,
        "validation_macro_mae_before": identity_macro_mae_fast(
            target, raw, groups
        ),
        "validation_macro_mae_after_refit": identity_macro_mae_fast(
            target, calibrated, groups
        ),
        "loio_search": rows,
    }
    return final_spec, calibrated, report


def stacked_uncertainty(
    predictions: np.ndarray,
    uncertainties: np.ndarray,
    dynamic_weights: np.ndarray,
    disagreement_coefficient: float,
) -> np.ndarray:
    predictions = np.asarray(predictions, dtype=float)
    uncertainties = np.asarray(uncertainties, dtype=float)
    weights = np.asarray(dynamic_weights, dtype=float)
    if predictions.shape != uncertainties.shape or predictions.shape != weights.shape:
        raise ValueError("stack uncertainty inputs must have equal [N, M] shapes")
    if disagreement_coefficient < 0:
        raise ValueError("disagreement coefficient must be non-negative")
    mean = np.sum(weights * predictions, axis=1)
    disagreement = np.sqrt(
        np.maximum(np.sum(weights * np.square(predictions - mean[:, None]), axis=1), 0.0)
    )
    base = np.sum(weights * uncertainties, axis=1)
    return base + disagreement_coefficient * disagreement


def select_disagreement_score(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    identities: Sequence[str] | np.ndarray,
    predictions: np.ndarray,
    uncertainties: np.ndarray,
    dynamic_weights: np.ndarray,
    *,
    coefficients: Sequence[float] = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0),
    coverages: Sequence[float] = SELECTION_COVERAGES,
) -> tuple[dict[str, Any], np.ndarray]:
    target = _finite_vector(y_true, "target")
    estimate = _finite_vector(y_pred, "prediction")
    groups = np.asarray(identities, dtype=str)
    rows: list[dict[str, Any]] = []
    best: tuple[float, float, np.ndarray] | None = None
    for coefficient in coefficients:
        score = stacked_uncertainty(
            predictions, uncertainties, dynamic_weights, float(coefficient)
        )
        curve = risk_coverage_curve(
            target,
            estimate,
            score,
            coverages=coverages,
            identities=groups,
        )
        objective = float(np.mean([row["macro_mae"] for row in curve]))
        rows.append(
            {
                "coefficient": float(coefficient),
                "selective_macro_mae": objective,
                "risk_coverage": curve,
            }
        )
        key = (objective, float(coefficient), score)
        if best is None or key[:2] < best[:2]:
            best = key
    assert best is not None
    return {
        "coefficient": best[1],
        "validation_selective_macro_mae": best[0],
        "search": rows,
    }, best[2]


def fit_empirical_uncertainty_scale(
    uncertainty: np.ndarray,
    *,
    quantile_count: int = 101,
) -> dict[str, Any]:
    """Fit a validation-only percentile map for cross-fold score comparability."""

    score = _finite_vector(uncertainty, "uncertainty")
    if quantile_count < 3:
        raise ValueError("quantile_count must be at least three")
    probability = np.linspace(0.0, 1.0, quantile_count)
    threshold = np.quantile(score, probability)
    unique_threshold, first = np.unique(threshold, return_index=True)
    unique_probability = probability[first]
    # Assign the maximum quantile represented by duplicate thresholds.
    for index, value in enumerate(unique_threshold):
        unique_probability[index] = probability[threshold == value].max()
    return {
        "kind": "validation_empirical_percentile",
        "thresholds": unique_threshold.tolist(),
        "percentiles": unique_probability.tolist(),
        "quantile_count": quantile_count,
    }


def apply_empirical_uncertainty_scale(
    uncertainty: np.ndarray,
    rule: Mapping[str, Any],
) -> np.ndarray:
    score = np.asarray(uncertainty, dtype=float)
    percentile = np.interp(
        score,
        np.asarray(rule["thresholds"], dtype=float),
        np.asarray(rule["percentiles"], dtype=float),
        left=0.0,
        right=1.0,
    )
    # A small positive floor keeps downstream variance calculations defined.
    return 0.05 + percentile


def causal_kalman_filter(
    prediction: np.ndarray,
    uncertainty: np.ndarray,
    session_id: Sequence[str] | np.ndarray,
    window_number: np.ndarray,
    *,
    process_noise: float,
    measurement_scale: float,
    innovation_clip: float | None,
    score_median: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Strictly causal random-walk Kalman filter, reset per session.

    Missing valid-reference windows are represented by gaps in window_number;
    the prior covariance grows with that gap.  No future observation is used.
    """

    raw = _finite_vector(prediction, "prediction")
    score = _finite_vector(uncertainty, "uncertainty")
    sessions = np.asarray(session_id, dtype=str)
    windows = np.asarray(window_number)
    if raw.shape != score.shape or sessions.shape != raw.shape or windows.shape != raw.shape:
        raise ValueError("causal filter inputs have different shapes")
    if not np.all(np.isfinite(windows)) or not np.all(windows == np.rint(windows)):
        raise ValueError("window_number must contain finite integers")
    if process_noise <= 0 or measurement_scale <= 0 or score_median <= 0:
        raise ValueError("state-space scales must be positive")
    if innovation_clip is not None and innovation_clip <= 0:
        raise ValueError("innovation_clip must be positive or None")

    filtered = np.empty_like(raw)
    filtered_score = np.empty_like(score)
    integer_windows = windows.astype(np.int64)
    for session in pd.unique(sessions):
        rows = np.flatnonzero(sessions == session)
        order = rows[np.argsort(integer_windows[rows], kind="stable")]
        state = float(raw[order[0]])
        normalized = float(np.clip(score[order[0]] / score_median, 0.10, 10.0))
        measurement_sd = measurement_scale * normalized
        covariance = measurement_sd**2
        filtered[order[0]] = state
        filtered_score[order[0]] = score[order[0]]
        previous_window = int(integer_windows[order[0]])
        for row in order[1:]:
            current_window = int(integer_windows[row])
            if current_window <= previous_window:
                raise ValueError(f"non-increasing window numbers in session {session}")
            gap = current_window - previous_window
            prior_covariance = covariance + process_noise**2 * gap
            normalized = float(np.clip(score[row] / score_median, 0.10, 10.0))
            measurement_sd = measurement_scale * normalized
            measurement_variance = measurement_sd**2
            gain = prior_covariance / (prior_covariance + measurement_variance)
            innovation = float(raw[row] - state)
            if innovation_clip is not None:
                innovation = float(
                    np.clip(innovation, -innovation_clip, innovation_clip)
                )
            state = state + gain * innovation
            covariance = max((1.0 - gain) * prior_covariance, 1e-9)
            filtered[row] = state
            # Preserve the source ranking signal and explicitly flag large
            # corrections made by the state model.
            filtered_score[row] = score[row] + abs(float(raw[row] - state))
            previous_window = current_window
    return filtered, filtered_score


def select_causal_state_space(
    y_true: np.ndarray,
    prediction: np.ndarray,
    uncertainty: np.ndarray,
    identities: Sequence[str] | np.ndarray,
    session_id: Sequence[str] | np.ndarray,
    window_number: np.ndarray,
    *,
    process_noises: Sequence[float] = (0.1, 0.25, 0.5, 1.0, 2.0),
    measurement_scales: Sequence[float] = (0.25, 0.5, 1.0, 2.0),
    innovation_clips: Sequence[float | None] = (2.0, 4.0, 8.0, None),
    minimum_improvement: float = 0.01,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    """Select a causal state-space rule on validation identities only."""

    target = _finite_vector(y_true, "target")
    raw = _finite_vector(prediction, "prediction")
    score = _finite_vector(uncertainty, "uncertainty")
    groups = np.asarray(identities, dtype=str)
    baseline = identity_macro_mae_fast(target, raw, groups)
    score_median = float(max(np.median(score), 1e-6))
    rows: list[dict[str, Any]] = []
    best: tuple[float, float, float, float, np.ndarray, np.ndarray] | None = None
    for process_noise in process_noises:
        for measurement_scale in measurement_scales:
            for innovation_clip in innovation_clips:
                filtered, filtered_score = causal_kalman_filter(
                    raw,
                    score,
                    session_id,
                    window_number,
                    process_noise=float(process_noise),
                    measurement_scale=float(measurement_scale),
                    innovation_clip=innovation_clip,
                    score_median=score_median,
                )
                objective = identity_macro_mae_fast(target, filtered, groups)
                rows.append(
                    {
                        "process_noise": float(process_noise),
                        "measurement_scale": float(measurement_scale),
                        "innovation_clip": innovation_clip,
                        "validation_macro_mae": objective,
                    }
                )
                clip_key = float("inf") if innovation_clip is None else float(innovation_clip)
                key = (
                    objective,
                    -float(process_noise),
                    float(measurement_scale),
                    -clip_key,
                    filtered,
                    filtered_score,
                )
                if best is None or key[:4] < best[:4]:
                    best = key
    assert best is not None
    improvement = baseline - best[0]
    guarded = improvement < minimum_improvement
    if guarded:
        rule = {
            "enabled": False,
            "process_noise": None,
            "measurement_scale": None,
            "innovation_clip": None,
            "score_median": score_median,
        }
        selected_prediction = raw.copy()
        selected_score = score.copy()
    else:
        selected_row = min(rows, key=lambda row: row["validation_macro_mae"])
        rule = {
            "enabled": True,
            "process_noise": selected_row["process_noise"],
            "measurement_scale": selected_row["measurement_scale"],
            "innovation_clip": selected_row["innovation_clip"],
            "score_median": score_median,
        }
        selected_prediction, selected_score = causal_kalman_filter(
            raw,
            score,
            session_id,
            window_number,
            process_noise=float(rule["process_noise"]),
            measurement_scale=float(rule["measurement_scale"]),
            innovation_clip=rule["innovation_clip"],
            score_median=score_median,
        )
    rule.update(
        {
            "baseline_validation_macro_mae": baseline,
            "candidate_best_validation_macro_mae": best[0],
            "candidate_improvement": improvement,
            "minimum_improvement": minimum_improvement,
            "guard_rejected_state_space": guarded,
            "selected_validation_macro_mae": identity_macro_mae_fast(
                target, selected_prediction, groups
            ),
            "search": sorted(rows, key=lambda row: row["validation_macro_mae"]),
        }
    )
    return rule, selected_prediction, selected_score


def apply_causal_rule(
    prediction: np.ndarray,
    uncertainty: np.ndarray,
    session_id: Sequence[str] | np.ndarray,
    window_number: np.ndarray,
    rule: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    if not rule["enabled"]:
        return np.asarray(prediction, dtype=float), np.asarray(uncertainty, dtype=float)
    return causal_kalman_filter(
        prediction,
        uncertainty,
        session_id,
        window_number,
        process_noise=float(rule["process_noise"]),
        measurement_scale=float(rule["measurement_scale"]),
        innovation_clip=rule["innovation_clip"],
        score_median=float(rule["score_median"]),
    )


def fixed_nonoverlap_mask(
    session_id: Sequence[str] | np.ndarray,
    window_number: np.ndarray,
    *,
    windows_apart: int = 8,
) -> np.ndarray:
    """Select a fixed first-then-every-N row per session without targets."""

    if windows_apart < 1:
        raise ValueError("windows_apart must be positive")
    sessions = np.asarray(session_id, dtype=str)
    windows = np.asarray(window_number)
    if sessions.shape != windows.shape or sessions.ndim != 1:
        raise ValueError("session and window vectors must match")
    selected = np.zeros(len(sessions), dtype=bool)
    for session in pd.unique(sessions):
        rows = np.flatnonzero(sessions == session)
        first = int(np.min(windows[rows]))
        selected[rows] = ((windows[rows].astype(int) - first) % windows_apart) == 0
    return selected


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _load_checkpoint(path: Path, expected_model_type: str) -> dict[str, Any]:
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
    if checkpoint["model_type"] != expected_model_type:
        raise ValueError(f"wrong checkpoint model type: {path}")
    return checkpoint


def _indices_for_identities(
    metadata: pd.DataFrame, identities: Sequence[str]
) -> np.ndarray:
    names = metadata["identity"].astype(str).to_numpy()
    valid = metadata["reference_valid"].to_numpy(dtype=bool)
    return np.flatnonzero(valid & np.isin(names, np.asarray(identities, dtype=str)))


def _sort_bundle(bundle: PredictionBundle) -> PredictionBundle:
    order = np.argsort(bundle.index, kind="stable")
    return PredictionBundle(
        **{
            field: np.asarray(getattr(bundle, field))[order]
            for field in PredictionBundle.__dataclass_fields__
        }
    )


def align_many_bundles(
    bundles: Sequence[PredictionBundle],
) -> list[PredictionBundle]:
    if not bundles:
        raise ValueError("no bundles supplied")
    aligned = [_sort_bundle(bundle) for bundle in bundles]
    reference = aligned[0]
    for bundle in aligned[1:]:
        align_prediction_bundles(reference, bundle)
    return aligned


def _load_locked_test_prediction(
    source: SourceSpec,
    fold: int,
    checkpoint: Mapping[str, Any],
    metadata: pd.DataFrame,
) -> PredictionBundle:
    path = source.run_dir / f"fold_{fold}" / f"{source.model_type}_test_predictions.npz"
    bundle, folds, signature = load_prediction_bundle(path)
    if signature != checkpoint["run_signature"] or not np.all(folds == fold):
        raise RuntimeError(f"test prediction metadata mismatch: {path}")
    expected = _indices_for_identities(metadata, checkpoint["split"]["test_identities"])
    if not np.array_equal(np.sort(bundle.index), expected):
        raise RuntimeError(f"test prediction identities mismatch: {path}")
    return bundle


def _prediction_matrices(
    bundles: Sequence[PredictionBundle],
) -> tuple[np.ndarray, np.ndarray]:
    predictions = np.column_stack([bundle.prediction for bundle in bundles]).astype(float)
    uncertainties = np.column_stack([bundle.uncertainty for bundle in bundles]).astype(float)
    if not np.isfinite(predictions).all() or not np.isfinite(uncertainties).all():
        raise ValueError("source predictions contain non-finite values")
    return predictions, uncertainties


def _apply_gate_rule(
    predictions: np.ndarray,
    uncertainties: np.ndarray,
    base_weights: np.ndarray,
    rule: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    dynamic = uncertainty_gated_weights(
        base_weights,
        uncertainties,
        np.asarray(rule["score_scale"], dtype=float),
        alpha=float(rule["alpha"]),
        uniform_floor=float(rule["uniform_floor"]),
    )
    return dynamic, np.sum(dynamic * predictions, axis=1)


def _strict_npz(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    temporary.replace(path)


def _subset_grouped_metrics(
    target: np.ndarray,
    prediction: np.ndarray,
    identities: np.ndarray,
    folds: np.ndarray,
    selected: np.ndarray,
    *,
    bootstrap_samples: int,
) -> dict[str, Any]:
    if not selected.any():
        return {"available": False, "n": 0}
    return {
        "available": True,
        **grouped_oof_metrics(
            target[selected],
            prediction[selected],
            identities[selected],
            fold_ids=folds[selected],
            bootstrap_samples=bootstrap_samples,
            bootstrap_seed=20260828,
        ),
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
    full = grouped_oof_metrics(
        target,
        prediction,
        identities,
        fold_ids=folds,
        bootstrap_samples=bootstrap_samples,
        bootstrap_seed=20260828,
    )
    high_rr = (target >= 25.0) & (target <= 35.0)
    nonoverlap = fixed_nonoverlap_mask(session_id, window_number, windows_apart=8)
    return {
        "full": full,
        "high_rr_25_35": _subset_grouped_metrics(
            target,
            prediction,
            identities,
            folds,
            high_rr,
            bootstrap_samples=bootstrap_samples,
        ),
        "fixed_nonoverlap_32s": _subset_grouped_metrics(
            target,
            prediction,
            identities,
            folds,
            nonoverlap,
            bootstrap_samples=bootstrap_samples,
        ),
        "selective_risk": risk_coverage_curve(
            target,
            prediction,
            uncertainty,
            coverages=coverages,
            identities=identities,
        ),
        "subset_counts": {
            "full": int(len(target)),
            "high_rr_25_35": int(high_rr.sum()),
            "fixed_nonoverlap_32s": int(nonoverlap.sum()),
        },
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    default_run = args.default_run.resolve()
    compact_run = args.compact_run.resolve()
    default_config = _load_json(default_run / "run_config.json")
    compact_config = _load_json(compact_run / "run_config.json")
    n_folds = int(default_config["arguments"]["folds"])
    if n_folds != int(compact_config["arguments"]["folds"]):
        raise ValueError("source runs have different fold counts")
    cache = prepare_source_cache(
        default_config,
        compact_config,
        cache_dir_override=args.cache_dir,
    )
    sources = [
        SourceSpec("default_snn", default_run, "snn"),
        SourceSpec("compact_snn", compact_run, "snn"),
        SourceSpec("default_teacher", default_run, "teacher"),
        SourceSpec("compact_teacher", compact_run, "teacher"),
    ]

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    amp = bool(args.amp and device.type == "cuda")
    if args.num_threads:
        torch.set_num_threads(args.num_threads)
    torch.set_float32_matmul_precision("high")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    signature_payload = {
        "pipeline_version": PIPELINE_VERSION,
        "default": default_config["run_signature"],
        "compact": compact_config["run_signature"],
        "stack_step": args.stack_step,
        "stack_minimum_improvement": args.stack_minimum_improvement,
        "gate_minimum_improvement": args.gate_minimum_improvement,
        "calibration_minimum_improvement": args.calibration_minimum_improvement,
        "state_minimum_improvement": args.state_minimum_improvement,
    }
    signature = hashlib.sha256(
        json.dumps(signature_payload, sort_keys=True).encode("utf-8")
    ).hexdigest()[:16]

    collected: dict[str, list[np.ndarray]] = {
        "index": [],
        "target": [],
        "fold": [],
        "stack_prediction": [],
        "calibrated_prediction": [],
        "final_prediction": [],
        "pre_state_uncertainty": [],
        "uncertainty": [],
        "dynamic_weights": [],
        "component_predictions": [],
        "component_uncertainties": [],
    }
    fold_locks: dict[str, Any] = {}

    for fold in range(n_folds):
        checkpoints: list[dict[str, Any]] = []
        for source in sources:
            checkpoint_path = (
                source.run_dir / f"fold_{fold}" / f"{source.model_type}_best.pt"
            )
            checkpoint = _load_checkpoint(checkpoint_path, source.model_type)
            if int(checkpoint["fold"]) != fold:
                raise RuntimeError(f"checkpoint fold mismatch: {checkpoint_path}")
            expected_signature = (
                default_config["run_signature"]
                if source.run_dir == default_run
                else compact_config["run_signature"]
            )
            if checkpoint["run_signature"] != expected_signature:
                raise RuntimeError(f"checkpoint signature mismatch: {checkpoint_path}")
            checkpoints.append(checkpoint)
        split = checkpoints[0]["split"]
        if any(checkpoint["split"] != split for checkpoint in checkpoints[1:]):
            raise RuntimeError(f"source split mismatch in fold {fold}")
        validation_index = _indices_for_identities(
            cache.metadata, split["validation_identities"]
        )
        scaler_audits = {
            source.label: verify_checkpoint_scaler(checkpoint, cache)
            for source, checkpoint in zip(sources, checkpoints, strict=True)
        }

        # Validation-only fitting/selection.  No outer-test NPZ has been opened.
        validation_bundles = [
            infer_validation_checkpoint(
                checkpoint,
                cache,
                validation_index,
                device=device,
                batch_size=args.batch_size,
                workers=args.workers,
                amp=amp,
            )
            for checkpoint in checkpoints
        ]
        validation_bundles = align_many_bundles(validation_bundles)
        validation_predictions, validation_uncertainties = _prediction_matrices(
            validation_bundles
        )
        validation_target = validation_bundles[0].target.astype(float)
        validation_rows = cache.metadata.iloc[validation_bundles[0].index]
        validation_identity = validation_rows["identity"].astype(str).to_numpy()
        validation_session = validation_rows["session_id"].astype(str).to_numpy()
        validation_window = validation_rows["window_number"].to_numpy(dtype=int)

        static_weights, stack_report = select_convex_stack(
            validation_target,
            validation_predictions,
            validation_identity,
            step=args.stack_step,
            minimum_improvement=args.stack_minimum_improvement,
        )
        gate_rule, validation_dynamic, validation_stack = select_uncertainty_gate(
            validation_target,
            validation_predictions,
            validation_uncertainties,
            validation_identity,
            static_weights,
            minimum_improvement=args.gate_minimum_improvement,
        )
        calibration_spec, validation_calibrated, calibration_report = (
            select_calibrator_leave_one_identity_out(
                validation_target,
                validation_stack,
                validation_identity,
                minimum_improvement=args.calibration_minimum_improvement,
            )
        )
        disagreement_rule, validation_score_raw = select_disagreement_score(
            validation_target,
            validation_calibrated,
            validation_identity,
            validation_predictions,
            validation_uncertainties,
            validation_dynamic,
        )
        empirical_score_rule = fit_empirical_uncertainty_scale(
            validation_score_raw
        )
        validation_score_normalized = apply_empirical_uncertainty_scale(
            validation_score_raw, empirical_score_rule
        )
        disagreement_rule["cross_fold_score_calibration"] = empirical_score_rule
        state_rule, validation_final, validation_final_score = (
            select_causal_state_space(
                validation_target,
                validation_calibrated,
                validation_score_raw,
                validation_identity,
                validation_session,
                validation_window,
                minimum_improvement=args.state_minimum_improvement,
            )
        )
        correction_scale = float(
            max(
                np.quantile(
                    np.abs(validation_calibrated - validation_final), 0.90
                ),
                0.25,
            )
        )
        state_rule["uncertainty_correction_scale_bpm"] = correction_scale
        validation_final_score = validation_score_normalized + np.abs(
            validation_calibrated - validation_final
        ) / correction_scale
        lock = {
            "fold": fold,
            "selection_data": "validation identities only",
            "outer_test_loaded_after_this_lock": True,
            "validation_identities": list(split["validation_identities"]),
            "outer_test_identities": list(split["test_identities"]),
            "source_order": [source.label for source in sources],
            "validation_rows": int(len(validation_target)),
            "scaler_audits": scaler_audits,
            "model_kwargs": {
                source.label: checkpoint["model_kwargs"]
                for source, checkpoint in zip(sources, checkpoints, strict=True)
            },
            "static_stack": stack_report,
            "uncertainty_gate": gate_rule,
            "calibration": calibration_report,
            "uncertainty_disagreement": disagreement_rule,
            "causal_state_space": state_rule,
            "validation_stage_macro_mae": {
                "stack": identity_macro_mae_fast(
                    validation_target, validation_stack, validation_identity
                ),
                "calibrated": identity_macro_mae_fast(
                    validation_target, validation_calibrated, validation_identity
                ),
                "final": identity_macro_mae_fast(
                    validation_target, validation_final, validation_identity
                ),
            },
        }
        # Persist the immutable rule before any outer-test prediction is read.
        write_json(output_dir / f"fold_{fold}_lock.json", lock)
        fold_locks[str(fold)] = lock

        # Outer-test application begins only after every stage above is locked.
        test_bundles = [
            _load_locked_test_prediction(
                source, fold, checkpoint, cache.metadata
            )
            for source, checkpoint in zip(sources, checkpoints, strict=True)
        ]
        test_bundles = align_many_bundles(test_bundles)
        test_predictions, test_uncertainties = _prediction_matrices(test_bundles)
        test_dynamic, test_stack = _apply_gate_rule(
            test_predictions, test_uncertainties, static_weights, gate_rule
        )
        test_calibrated = apply_calibrator(test_stack, calibration_spec)
        test_score_raw = stacked_uncertainty(
            test_predictions,
            test_uncertainties,
            test_dynamic,
            float(disagreement_rule["coefficient"]),
        )
        test_score_normalized = apply_empirical_uncertainty_scale(
            test_score_raw, empirical_score_rule
        )
        test_rows = cache.metadata.iloc[test_bundles[0].index]
        test_session = test_rows["session_id"].astype(str).to_numpy()
        test_window = test_rows["window_number"].to_numpy(dtype=int)
        test_final, _ = apply_causal_rule(
            test_calibrated,
            test_score_raw,
            test_session,
            test_window,
            state_rule,
        )
        test_final_score = test_score_normalized + np.abs(
            test_calibrated - test_final
        ) / correction_scale

        collected["index"].append(test_bundles[0].index.astype(np.int64))
        collected["target"].append(test_bundles[0].target.astype(np.float32))
        collected["fold"].append(np.full(len(test_final), fold, dtype=np.int16))
        collected["stack_prediction"].append(test_stack.astype(np.float32))
        collected["calibrated_prediction"].append(
            test_calibrated.astype(np.float32)
        )
        collected["final_prediction"].append(test_final.astype(np.float32))
        collected["pre_state_uncertainty"].append(
            test_score_normalized.astype(np.float32)
        )
        collected["uncertainty"].append(test_final_score.astype(np.float32))
        collected["dynamic_weights"].append(test_dynamic.astype(np.float32))
        collected["component_predictions"].append(
            test_predictions.astype(np.float32)
        )
        collected["component_uncertainties"].append(
            test_uncertainties.astype(np.float32)
        )
        print(
            f"fold={fold} weights={np.round(static_weights, 3).tolist()} "
            f"gate={gate_rule['alpha']}/{gate_rule['uniform_floor']} "
            f"cal={calibration_spec['kind']} state={state_rule['enabled']} "
            f"val_macro={lock['validation_stage_macro_mae']['final']:.4f}",
            flush=True,
        )

    raw_index = np.concatenate(collected["index"])
    order = np.argsort(raw_index, kind="stable")
    arrays = {
        key: np.concatenate(value, axis=0)[order]
        for key, value in collected.items()
    }
    if len(np.unique(arrays["index"])) != len(arrays["index"]):
        raise RuntimeError("duplicate outer-test cache indices")
    expected_rows = int(cache.metadata["reference_valid"].to_numpy(dtype=bool).sum())
    if len(arrays["index"]) != expected_rows:
        raise RuntimeError(
            f"incomplete OOF: {len(arrays['index'])} rows, expected {expected_rows}"
        )
    oof_rows = cache.metadata.iloc[arrays["index"]]
    identities = oof_rows["identity"].astype(str).to_numpy()
    sessions = oof_rows["session_id"].astype(str).to_numpy()
    windows = oof_rows["window_number"].to_numpy(dtype=int)
    folds = arrays["fold"].astype(int)

    _strict_npz(
        output_dir / "commercial_stack_oof.npz",
        **arrays,
        source_labels=np.asarray([source.label for source in sources]),
        run_signature=np.asarray(signature),
    )
    csv_columns = [
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
    ]
    table = oof_rows[csv_columns].reset_index(drop=True)
    table.insert(0, "cache_index", arrays["index"])
    table.insert(1, "fold", folds)
    for column, source in enumerate(sources):
        table[f"prediction_{source.label}_bpm"] = arrays[
            "component_predictions"
        ][:, column]
        table[f"uncertainty_{source.label}"] = arrays[
            "component_uncertainties"
        ][:, column]
        table[f"dynamic_weight_{source.label}"] = arrays["dynamic_weights"][:, column]
    table["prediction_stack_bpm"] = arrays["stack_prediction"]
    table["prediction_calibrated_bpm"] = arrays["calibrated_prediction"]
    table["prediction_final_bpm"] = arrays["final_prediction"]
    table["uncertainty_pre_state"] = arrays["pre_state_uncertainty"]
    table["uncertainty_final"] = arrays["uncertainty"]
    table.to_csv(output_dir / "commercial_stack_oof.csv", index=False)

    variants: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for column, source in enumerate(sources):
        variants[source.label] = (
            arrays["component_predictions"][:, column],
            arrays["component_uncertainties"][:, column],
        )
    variants.update(
        {
            "uncalibrated_stack": (
                arrays["stack_prediction"],
                arrays["pre_state_uncertainty"],
            ),
            "post_calibration": (
                arrays["calibrated_prediction"],
                arrays["pre_state_uncertainty"],
            ),
            "locked_final": (
                arrays["final_prediction"],
                arrays["uncertainty"],
            ),
        }
    )
    variant_metrics = {
        name: evaluation_report(
            arrays["target"].astype(float),
            prediction.astype(float),
            uncertainty.astype(float),
            identities,
            folds,
            sessions,
            windows,
            bootstrap_samples=args.bootstrap_samples,
            coverages=args.coverages,
        )
        for name, (prediction, uncertainty) in variants.items()
    }
    final_overall = variant_metrics["locked_final"]["full"]["overall"]
    final_macro = variant_metrics["locked_final"]["full"]["identity_macro"]
    commercial_target_audit = {
        "macro_mae_le_1_bpm": bool(final_macro["macro_mae"] <= 1.0),
        "overall_rmse_le_1_5_bpm": bool(final_overall["rmse"] <= 1.5),
        "p95_ae_le_3_bpm": bool(final_overall["p95_ae"] <= 3.0),
        "within_2_ge_95_percent": bool(final_overall["within_2"] >= 0.95),
        "catastrophic_over_5_le_1_percent": bool(
            final_overall["catastrophic_over_5"] <= 0.01
        ),
        "all_internal_targets_met": False,
        "claim": "research-only grouped OOF; prospective locked validation required",
    }
    commercial_target_audit["all_internal_targets_met"] = bool(
        all(
            value
            for key, value in commercial_target_audit.items()
            if key not in {"all_internal_targets_met", "claim"}
        )
    )
    stage_counts = {
        "static_blend_selected": int(
            sum(
                not lock["static_stack"]["guard_selected_single_component"]
                for lock in fold_locks.values()
            )
        ),
        "uncertainty_gate_selected": int(
            sum(
                not lock["uncertainty_gate"]["guard_rejected_gating"]
                for lock in fold_locks.values()
            )
        ),
        "calibration_selected": int(
            sum(
                not lock["calibration"]["guard_rejected_calibration"]
                for lock in fold_locks.values()
            )
        ),
        "causal_state_space_selected": int(
            sum(
                lock["causal_state_space"]["enabled"]
                for lock in fold_locks.values()
            )
        ),
        "n_folds": n_folds,
    }
    baseline_metrics = variant_metrics["default_snn"]
    stack_metrics = variant_metrics["uncalibrated_stack"]
    final_metrics = variant_metrics["locked_final"]
    stack_macro_delta = float(
        stack_metrics["full"]["identity_macro"]["macro_mae"]
        - baseline_metrics["full"]["identity_macro"]["macro_mae"]
    )
    state_macro_delta = float(
        final_metrics["full"]["identity_macro"]["macro_mae"]
        - stack_metrics["full"]["identity_macro"]["macro_mae"]
    )
    state_nonoverlap_delta = float(
        final_metrics["fixed_nonoverlap_32s"]["identity_macro"]["macro_mae"]
        - stack_metrics["fixed_nonoverlap_32s"]["identity_macro"]["macro_mae"]
    )
    high_rr_macro_delta = float(
        final_metrics["high_rr_25_35"]["identity_macro"]["macro_mae"]
        - baseline_metrics["high_rr_25_35"]["identity_macro"]["macro_mae"]
    )
    candidate_conclusions = {
        "outer_oof_is_evaluation_not_a_new_selection_stage": True,
        "four_model_stack": {
            "macro_mae_delta_vs_default_snn_bpm": stack_macro_delta,
            "supported_by_outer_oof": stack_macro_delta < -0.01,
        },
        "calibration": {
            "selected_folds": stage_counts["calibration_selected"],
            "conclusion": (
                "rejected by validation LOIO guard in every fold"
                if stage_counts["calibration_selected"] == 0
                else "selected only in folds recorded above"
            ),
        },
        "causal_state_space": {
            "full_macro_mae_delta_vs_stack_bpm": state_macro_delta,
            "nonoverlap_macro_mae_delta_vs_stack_bpm": state_nonoverlap_delta,
            "commercial_evidence_supported": bool(
                state_macro_delta < -0.01 and state_nonoverlap_delta <= 0.0
            ),
            "conclusion": (
                "supported by both full and fixed non-overlap outer OOF"
                if state_macro_delta < -0.01 and state_nonoverlap_delta <= 0.0
                else "not supported for commercial deployment: full-macro gain is "
                "not practically meaningful and/or fixed non-overlap performance regressed"
            ),
        },
        "high_rr_25_35": {
            "macro_mae_delta_vs_default_snn_bpm": high_rr_macro_delta,
            "commercial_evidence_supported": bool(
                final_metrics["high_rr_25_35"]["identity_macro"]["macro_mae"]
                <= 1.5
            ),
            "conclusion": "high-RR mean-regression failure remains unresolved",
        },
    }
    report = {
        "method": "validation-only locked four-model commercial-candidate stack",
        "selection_guarantee": (
            "Every fold's static stack, uncertainty gate, LOIO calibration, "
            "disagreement score, and causal state-space rule was frozen before "
            "that fold's outer-test prediction files were opened."
        ),
        "run_signature": signature,
        "pipeline_version": PIPELINE_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "device": str(device),
        "amp": amp,
        "source_order": [source.label for source in sources],
        "complete_oof": True,
        "n_rows": int(len(arrays["target"])),
        "n_identities": int(len(np.unique(identities))),
        "fold_locks": fold_locks,
        "stage_selection_counts": stage_counts,
        "candidate_conclusions": candidate_conclusions,
        "metrics": variant_metrics,
        "commercial_target_audit": commercial_target_audit,
    }
    write_json(output_dir / "metrics.json", report)
    write_json(
        output_dir / "run_config.json",
        {
            "run_signature": signature,
            "pipeline_version": PIPELINE_VERSION,
            "default_run": str(default_run),
            "compact_run": str(compact_run),
            "arguments": vars(args),
            "source_order": [source.label for source in sources],
            "device": str(device),
            "amp": amp,
        },
    )
    print(
        f"completed output={output_dir} "
        f"overall_mae={final_overall['mae']:.4f} "
        f"macro_mae={final_macro['macro_mae']:.4f}",
        flush=True,
    )
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Leakage-safe four-model RR stacking experiment"
    )
    parser.add_argument(
        "--default-run",
        type=Path,
        default=Path("artifacts/runs/final_default_s12"),
    )
    parser.add_argument(
        "--compact-run",
        type=Path,
        default=Path("artifacts/runs/final_compact_s8"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/experiments/commercial_stack"),
    )
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--num-threads", type=int)
    parser.add_argument("--stack-step", type=float, default=0.025)
    parser.add_argument("--stack-minimum-improvement", type=float, default=0.005)
    parser.add_argument("--gate-minimum-improvement", type=float, default=0.005)
    parser.add_argument(
        "--calibration-minimum-improvement", type=float, default=0.01
    )
    parser.add_argument("--state-minimum-improvement", type=float, default=0.01)
    parser.add_argument(
        "--coverages",
        type=float,
        nargs="+",
        default=list(DEFAULT_COVERAGES),
    )
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.batch_size < 1 or args.workers < 0:
        parser.error("batch size must be positive and workers non-negative")
    if args.bootstrap_samples < 1:
        parser.error("bootstrap samples must be positive")
    if any(not 0 < coverage <= 1 for coverage in args.coverages):
        parser.error("coverages must be in (0,1]")
    for name in (
        "stack_minimum_improvement",
        "gate_minimum_improvement",
        "calibration_minimum_improvement",
        "state_minimum_improvement",
    ):
        if getattr(args, name) < 0:
            parser.error(f"--{name.replace('_', '-')} must be non-negative")
    try:
        simplex_grid(4, args.stack_step)
    except ValueError as exc:
        parser.error(str(exc))
    return args


def main(argv: Sequence[str] | None = None) -> None:
    run(parse_args(argv))


if __name__ == "__main__":
    main()
