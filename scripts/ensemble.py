#!/usr/bin/env python3
"""Locked, validation-selected ensemble of two grouped SNN runs.

For every outer fold this script performs the following operations in order:

1. Reconstruct the cached inputs (including strictly causal history).
2. Load each fold checkpoint with its own train-only auxiliary scaler.
3. Infer *only* the identities assigned to that fold's validation split.
4. Select the prediction blend weight and uncertainty-disagreement coefficient.
5. Lock those values and only then load/apply them to the saved outer-test
   predictions.

Consequently no outer-test target or prediction participates in selection.
The final files are a genuine nested/locked combination of the two source
runs, although the original model checkpoints themselves still used their
validation identities for early stopping.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import sys
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
for search_path in (PROJECT_ROOT, SRC_ROOT):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

from snn_rr.cache import (  # noqa: E402
    FeatureCache,
    append_causal_history_features,
    fit_aux_scaler,
    load_feature_cache,
    transform_aux,
)
from snn_rr.metrics import (  # noqa: E402
    grouped_oof_metrics,
    identity_macro_metrics,
    regression_metrics,
    risk_coverage_curve,
)
from scripts.train import (  # noqa: E402
    PredictionBundle,
    build_model,
    load_prediction_bundle,
    make_loader,
    predict,
    write_json,
)


DEFAULT_WEIGHT_GRID = tuple(np.linspace(0.0, 1.0, 101).tolist())
DEFAULT_DISAGREEMENT_GRID = (0.0, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0)
DEFAULT_COVERAGES = (1.0, 0.9, 0.8, 0.7, 0.5)
DEFAULT_SELECTION_COVERAGES = (0.9, 0.8, 0.7, 0.5)


def _finite_vectors(*values: np.ndarray) -> tuple[np.ndarray, ...]:
    arrays = tuple(np.asarray(value, dtype=float) for value in values)
    if not arrays or arrays[0].ndim != 1:
        raise ValueError("inputs must be one-dimensional vectors")
    if any(array.shape != arrays[0].shape for array in arrays):
        raise ValueError("inputs must have identical shapes")
    if not len(arrays[0]):
        raise ValueError("inputs must not be empty")
    if any(not np.isfinite(array).all() for array in arrays):
        raise ValueError("inputs must contain only finite values")
    return arrays


def blend_prediction(
    prediction_a: np.ndarray,
    prediction_b: np.ndarray,
    weight_a: float,
) -> np.ndarray:
    """Convex prediction blend where ``weight_a`` belongs to source A."""

    prediction_a, prediction_b = _finite_vectors(prediction_a, prediction_b)
    if not 0.0 <= weight_a <= 1.0:
        raise ValueError("weight_a must be in [0, 1]")
    return weight_a * prediction_a + (1.0 - weight_a) * prediction_b


def ensemble_uncertainty(
    score_a: np.ndarray,
    score_b: np.ndarray,
    prediction_a: np.ndarray,
    prediction_b: np.ndarray,
    *,
    weight_a: float,
    disagreement_coefficient: float,
) -> np.ndarray:
    """Weighted source score plus a locked model-disagreement penalty."""

    score_a, score_b, prediction_a, prediction_b = _finite_vectors(
        score_a, score_b, prediction_a, prediction_b
    )
    if not 0.0 <= weight_a <= 1.0:
        raise ValueError("weight_a must be in [0, 1]")
    if disagreement_coefficient < 0 or not math.isfinite(disagreement_coefficient):
        raise ValueError("disagreement_coefficient must be finite and non-negative")
    return (
        weight_a * score_a
        + (1.0 - weight_a) * score_b
        + disagreement_coefficient * np.abs(prediction_a - prediction_b)
    )


def select_blend_weight(
    y_true: np.ndarray,
    prediction_a: np.ndarray,
    prediction_b: np.ndarray,
    identities: Sequence[str] | np.ndarray,
    *,
    grid: Iterable[float] = DEFAULT_WEIGHT_GRID,
) -> tuple[float, list[dict[str, float]]]:
    """Select a convex weight by validation identity-macro MAE only."""

    y_true, prediction_a, prediction_b = _finite_vectors(
        y_true, prediction_a, prediction_b
    )
    identities = np.asarray(identities, dtype=str)
    if identities.shape != y_true.shape:
        raise ValueError("identity count does not match validation predictions")
    candidates = sorted({float(value) for value in grid})
    if not candidates or candidates[0] < 0.0 or candidates[-1] > 1.0:
        raise ValueError("blend grid must contain values in [0, 1]")

    rows: list[dict[str, float]] = []
    for weight in candidates:
        candidate = blend_prediction(prediction_a, prediction_b, weight)
        macro_mae = identity_macro_metrics(
            y_true, candidate, identities
        )["macro_mae"]
        rows.append({"weight_a": weight, "validation_macro_mae": macro_mae})
    # Prefer a less extreme blend when numerical ties occur; this rule is
    # declared before test inference and is therefore part of the lock.
    selected = min(
        rows,
        key=lambda row: (
            row["validation_macro_mae"],
            abs(row["weight_a"] - 0.5),
            row["weight_a"],
        ),
    )
    return float(selected["weight_a"]), rows


def select_disagreement_coefficient(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    identities: Sequence[str] | np.ndarray,
    score_a: np.ndarray,
    score_b: np.ndarray,
    prediction_a: np.ndarray,
    prediction_b: np.ndarray,
    *,
    weight_a: float,
    coefficient_grid: Iterable[float] = DEFAULT_DISAGREEMENT_GRID,
    coverages: Iterable[float] = DEFAULT_SELECTION_COVERAGES,
) -> tuple[float, list[dict[str, Any]]]:
    """Choose disagreement scaling by validation selective macro risk.

    The objective is the mean identity-macro MAE over the declared validation
    coverages.  Full coverage is intentionally ignored if supplied because it
    cannot depend on uncertainty ranking.
    """

    y_true, y_pred, score_a, score_b, prediction_a, prediction_b = _finite_vectors(
        y_true, y_pred, score_a, score_b, prediction_a, prediction_b
    )
    identities = np.asarray(identities, dtype=str)
    if identities.shape != y_true.shape:
        raise ValueError("identity count does not match validation predictions")
    candidates = sorted({float(value) for value in coefficient_grid})
    if not candidates or candidates[0] < 0.0 or not np.isfinite(candidates).all():
        raise ValueError("coefficient grid must contain finite non-negative values")
    selection_coverages = tuple(float(value) for value in coverages if value < 1.0)
    if not selection_coverages:
        raise ValueError("at least one selection coverage below 1.0 is required")

    rows: list[dict[str, Any]] = []
    for coefficient in candidates:
        uncertainty = ensemble_uncertainty(
            score_a,
            score_b,
            prediction_a,
            prediction_b,
            weight_a=weight_a,
            disagreement_coefficient=coefficient,
        )
        curve = risk_coverage_curve(
            y_true,
            y_pred,
            uncertainty,
            coverages=selection_coverages,
            identities=identities,
        )
        objective = float(np.mean([row["macro_mae"] for row in curve]))
        rows.append(
            {
                "disagreement_coefficient": coefficient,
                "validation_selective_macro_mae": objective,
                "risk_coverage": curve,
            }
        )
    selected = min(
        rows,
        key=lambda row: (
            row["validation_selective_macro_mae"],
            row["disagreement_coefficient"],
        ),
    )
    return float(selected["disagreement_coefficient"]), rows


def fit_identity_balanced_affine(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    identities: Sequence[str] | np.ndarray,
    *,
    slope_bounds: tuple[float, float] = (0.7, 1.5),
    intercept_bounds: tuple[float, float] = (-5.0, 5.0),
    min_macro_mae_improvement: float = 1e-6,
) -> dict[str, Any]:
    """Fit a conservative identity-balanced affine validation calibrator.

    Each identity receives the same total least-squares mass regardless of its
    number of overlapping windows.  The candidate slope/intercept are clipped
    to conservative bounds.  A validation identity-macro-MAE guard then locks
    the identity mapping whenever applying the affine candidate would not
    improve validation performance.
    """

    y_true, y_pred = _finite_vectors(y_true, y_pred)
    identities = np.asarray(identities, dtype=str)
    if identities.shape != y_true.shape:
        raise ValueError("identity count does not match calibration predictions")
    slope_low, slope_high = map(float, slope_bounds)
    intercept_low, intercept_high = map(float, intercept_bounds)
    if not 0 < slope_low <= slope_high:
        raise ValueError("slope bounds must be positive and ordered")
    if intercept_low > intercept_high:
        raise ValueError("intercept bounds must be ordered")

    weights = np.zeros(len(y_true), dtype=float)
    for identity in np.unique(identities):
        selected = identities == identity
        weights[selected] = 1.0 / float(selected.sum())
    weights /= weights.sum()
    x_mean = float(np.sum(weights * y_pred))
    y_mean = float(np.sum(weights * y_true))
    denominator = float(np.sum(weights * np.square(y_pred - x_mean)))
    if denominator <= 1e-12:
        raw_slope = 1.0
    else:
        raw_slope = float(
            np.sum(weights * (y_pred - x_mean) * (y_true - y_mean))
            / denominator
        )
    candidate_slope = float(np.clip(raw_slope, slope_low, slope_high))
    raw_intercept = y_mean - candidate_slope * x_mean
    candidate_intercept = float(
        np.clip(raw_intercept, intercept_low, intercept_high)
    )
    candidate_prediction = candidate_slope * y_pred + candidate_intercept
    uncalibrated_macro_mae = identity_macro_metrics(
        y_true, y_pred, identities
    )["macro_mae"]
    candidate_macro_mae = identity_macro_metrics(
        y_true, candidate_prediction, identities
    )["macro_mae"]
    use_candidate = bool(
        candidate_macro_mae
        < uncalibrated_macro_mae - float(min_macro_mae_improvement)
    )
    return {
        "slope": candidate_slope if use_candidate else 1.0,
        "intercept": candidate_intercept if use_candidate else 0.0,
        "selected_affine": use_candidate,
        "candidate_slope": candidate_slope,
        "candidate_intercept": candidate_intercept,
        "raw_slope": raw_slope,
        "raw_intercept_after_slope_clip": raw_intercept,
        "slope_bounds": [slope_low, slope_high],
        "intercept_bounds": [intercept_low, intercept_high],
        "uncalibrated_validation_macro_mae": uncalibrated_macro_mae,
        "candidate_validation_macro_mae": candidate_macro_mae,
        "selected_validation_macro_mae": (
            candidate_macro_mae if use_candidate else uncalibrated_macro_mae
        ),
        "guard": "use affine only when validation identity-macro MAE improves",
    }


def apply_affine(y_pred: np.ndarray, calibration: Mapping[str, Any]) -> np.ndarray:
    prediction = np.asarray(y_pred, dtype=float)
    return (
        float(calibration["slope"]) * prediction
        + float(calibration["intercept"])
    )


def rr_band_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    identities: Sequence[str] | np.ndarray,
    *,
    edges: Sequence[float] = (6.0, 12.0, 18.0, 24.0, 30.0, 45.000001),
) -> dict[str, dict[str, float]]:
    """Report error and mean-regression bias in clinically readable RR bands."""

    y_true, y_pred = _finite_vectors(y_true, y_pred)
    identities = np.asarray(identities, dtype=str)
    if identities.shape != y_true.shape:
        raise ValueError("identity count does not match RR-band predictions")
    boundaries = np.asarray(edges, dtype=float)
    if len(boundaries) < 2 or not np.all(np.diff(boundaries) > 0):
        raise ValueError("RR-band edges must be strictly increasing")
    result: dict[str, dict[str, float]] = {}
    for band in range(len(boundaries) - 1):
        low = float(boundaries[band])
        high = float(boundaries[band + 1])
        selected = (y_true >= low) & (y_true < high)
        if not selected.any():
            continue
        label = f"{low:g}-{high:g}_bpm"
        metrics = regression_metrics(y_true[selected], y_pred[selected])
        metrics.update(
            identity_macro_metrics(
                y_true[selected], y_pred[selected], identities[selected]
            )
        )
        metrics.update(
            {
                "n_identities": float(len(np.unique(identities[selected]))),
                "mean_reference_bpm": float(np.mean(y_true[selected])),
                "mean_prediction_bpm": float(np.mean(y_pred[selected])),
                "mean_regression_bpm": float(
                    np.mean(y_pred[selected]) - np.mean(y_true[selected])
                ),
            }
        )
        result[label] = metrics
    return result


def compare_rr_band_metrics(
    uncalibrated: Mapping[str, Mapping[str, float]],
    calibrated: Mapping[str, Mapping[str, float]],
) -> dict[str, dict[str, float | bool]]:
    """Make affine improvements/regressions explicit for every RR band."""

    if set(uncalibrated) != set(calibrated):
        raise ValueError("RR-band schemas do not match")
    result: dict[str, dict[str, float | bool]] = {}
    for band in uncalibrated:
        before = uncalibrated[band]
        after = calibrated[band]
        mae_delta = float(after["mae"] - before["mae"])
        macro_mae_delta = float(after["macro_mae"] - before["macro_mae"])
        absolute_bias_delta = float(abs(after["bias"]) - abs(before["bias"]))
        result[band] = {
            "mae_delta_bpm_calibrated_minus_uncalibrated": mae_delta,
            "macro_mae_delta_bpm_calibrated_minus_uncalibrated": macro_mae_delta,
            "absolute_bias_delta_bpm_calibrated_minus_uncalibrated": absolute_bias_delta,
            "mae_improved": mae_delta < 0.0,
            "macro_mae_improved": macro_mae_delta < 0.0,
            "absolute_bias_improved": absolute_bias_delta < 0.0,
        }
    return result


def align_prediction_bundles(
    first: PredictionBundle,
    second: PredictionBundle,
) -> tuple[PredictionBundle, PredictionBundle]:
    """Return two bundles in matching cache-index order with strict audits."""

    if len(np.unique(first.index)) != len(first.index):
        raise ValueError("source A contains duplicate cache indices")
    if len(np.unique(second.index)) != len(second.index):
        raise ValueError("source B contains duplicate cache indices")
    first_order = np.argsort(first.index, kind="stable")
    second_order = np.argsort(second.index, kind="stable")
    first_sorted = PredictionBundle(
        **{
            field: np.asarray(getattr(first, field))[first_order]
            for field in PredictionBundle.__dataclass_fields__
        }
    )
    second_sorted = PredictionBundle(
        **{
            field: np.asarray(getattr(second, field))[second_order]
            for field in PredictionBundle.__dataclass_fields__
        }
    )
    if not np.array_equal(first_sorted.index, second_sorted.index):
        raise ValueError("source prediction cache indices do not match")
    for field in ("target", "observable", "reference_valid"):
        a = np.asarray(getattr(first_sorted, field))
        b = np.asarray(getattr(second_sorted, field))
        if not np.allclose(a, b, rtol=0.0, atol=1e-6, equal_nan=True):
            raise ValueError(f"source prediction field does not match: {field}")
    return first_sorted, second_sorted


def _subset_bundle(bundle: PredictionBundle, selected: np.ndarray) -> PredictionBundle:
    selected = np.asarray(selected)
    return PredictionBundle(
        **{
            field: np.asarray(getattr(bundle, field))[selected]
            for field in PredictionBundle.__dataclass_fields__
        }
    )


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _source_label(path: Path) -> str:
    value = re.sub(r"[^A-Za-z0-9]+", "_", path.name).strip("_").lower()
    return value or "source"


def _preprocessing_fingerprint(run_config: Mapping[str, Any]) -> dict[str, Any]:
    arguments = run_config["arguments"]
    return {
        "cache_dir": str(arguments["cache_dir"]),
        "map_branch": arguments["map_branch"],
        "input_branches": int(arguments["input_branches"]),
        "use_aux": bool(arguments["use_aux"]),
        "causal_history": bool(arguments["causal_history"]),
        "history_names": list(run_config.get("causal_history_feature_names", [])),
    }


def prepare_source_cache(
    run_config_a: Mapping[str, Any],
    run_config_b: Mapping[str, Any],
    *,
    cache_dir_override: Path | None = None,
) -> FeatureCache:
    """Reproduce the common feature preprocessing declared by both runs."""

    fingerprint_a = _preprocessing_fingerprint(run_config_a)
    fingerprint_b = _preprocessing_fingerprint(run_config_b)
    if fingerprint_a != fingerprint_b:
        raise ValueError(
            "source runs use different cache/map/history preprocessing and cannot "
            "share a locked input reconstruction"
        )
    cache_path = (
        cache_dir_override
        if cache_dir_override is not None
        else PROJECT_ROOT / fingerprint_a["cache_dir"]
    )
    cache = load_feature_cache(cache_path)
    stored_range_bins = int(cache.maps.shape[-1])
    branch = fingerprint_a["map_branch"]
    if branch == "both":
        maps = cache.maps
    elif branch == "raw":
        maps = cache.maps[..., : stored_range_bins // 2]
    elif branch == "phase":
        maps = cache.maps[..., stored_range_bins // 2 :]
    else:
        raise ValueError(f"unsupported map branch: {branch!r}")
    if not fingerprint_a["use_aux"]:
        aux = np.empty((len(cache.metadata), 0), dtype=np.float32)
    elif fingerprint_a["causal_history"]:
        aux, names = append_causal_history_features(cache.aux, cache.metadata)
        if names != fingerprint_a["history_names"]:
            raise RuntimeError("causal-history schema differs from the source run")
    else:
        aux = cache.aux
    prepared = FeatureCache(
        maps=maps,
        aux=aux,
        metadata=cache.metadata,
        frequencies_hz=cache.frequencies_hz,
    )
    expected_shape = run_config_a.get("cache_shape", {})
    if expected_shape:
        if list(prepared.maps.shape) != list(expected_shape.get("maps", [])):
            raise RuntimeError("reconstructed map shape differs from the source run")
        if list(prepared.aux.shape) != list(expected_shape.get("aux", [])):
            raise RuntimeError("reconstructed auxiliary shape differs from the source run")
    return prepared


def _checkpoint(path: Path) -> dict[str, Any]:
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
        raise KeyError(f"checkpoint {path} is missing fields: {missing}")
    if checkpoint["model_type"] != "snn":
        raise ValueError(f"checkpoint is not an SNN: {path}")
    return checkpoint


def _indices_for_identities(
    metadata: pd.DataFrame,
    identities: Sequence[str],
) -> np.ndarray:
    identity = metadata["identity"].astype(str).to_numpy()
    valid = metadata["reference_valid"].to_numpy(dtype=bool)
    return np.flatnonzero(valid & np.isin(identity, np.asarray(identities, dtype=str)))


def verify_checkpoint_scaler(
    checkpoint: Mapping[str, Any],
    cache: FeatureCache,
) -> dict[str, Any]:
    """Re-fit on checkpoint train identities and demand an exact reconstruction."""

    split = checkpoint["split"]
    train_index = _indices_for_identities(cache.metadata, split["train_identities"])
    expected_center, expected_scale = fit_aux_scaler(cache.aux, train_index)
    center = np.asarray(checkpoint["aux_center"].cpu(), dtype=np.float32)
    scale = np.asarray(checkpoint["aux_scale"].cpu(), dtype=np.float32)
    if center.shape != cache.aux.shape[1:] or scale.shape != cache.aux.shape[1:]:
        raise RuntimeError("checkpoint scaler dimension does not match reconstructed cache")
    center_error = float(np.max(np.abs(center - expected_center), initial=0.0))
    scale_error = float(np.max(np.abs(scale - expected_scale), initial=0.0))
    if not (
        np.array_equal(center, expected_center)
        and np.array_equal(scale, expected_scale)
    ):
        raise RuntimeError(
            "checkpoint scaler was not reproduced exactly from its declared "
            f"train identities (center max error={center_error}, "
            f"scale max error={scale_error})"
        )
    return {
        "verified_train_only": True,
        "train_rows": int(len(train_index)),
        "aux_dim": int(len(center)),
        "max_abs_center_error": center_error,
        "max_abs_scale_error": scale_error,
    }


@torch.inference_mode()
def infer_validation_checkpoint(
    checkpoint: Mapping[str, Any],
    cache: FeatureCache,
    indices: np.ndarray,
    *,
    device: torch.device,
    batch_size: int,
    workers: int,
    amp: bool,
) -> PredictionBundle:
    """Infer a locked checkpoint with its serialized train-only scaler."""

    center = np.asarray(checkpoint["aux_center"].cpu(), dtype=np.float32)
    scale = np.asarray(checkpoint["aux_scale"].cpu(), dtype=np.float32)
    aux_scaled = transform_aux(cache.aux, center, scale)
    loader = make_loader(
        cache,
        aux_scaled,
        np.asarray(indices, dtype=np.int64),
        batch_size=batch_size,
        workers=workers,
        device=device,
        seed=20260827 + int(checkpoint["fold"]),
        train=False,
    )
    # model_kwargs is intentionally consumed directly from the checkpoint;
    # no architecture defaults are re-derived from a run name or preset.
    model = build_model(checkpoint["model_type"], checkpoint["model_kwargs"])
    model.load_state_dict(checkpoint["model_state"])
    model = model.to(device)
    result = predict(model, loader, device, amp=amp)
    del model, loader, aux_scaled
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def _load_locked_test_bundle(
    run_dir: Path,
    fold: int,
    checkpoint: Mapping[str, Any],
    metadata: pd.DataFrame,
) -> PredictionBundle:
    """Load source test predictions only after fold selection is complete."""

    path = run_dir / f"fold_{fold}" / "snn_test_predictions.npz"
    bundle, fold_vector, signature = load_prediction_bundle(path)
    if signature != checkpoint["run_signature"]:
        raise RuntimeError(f"prediction/checkpoint signature mismatch: {path}")
    if not np.all(fold_vector == fold):
        raise RuntimeError(f"prediction file contains the wrong fold: {path}")
    expected = _indices_for_identities(
        metadata, checkpoint["split"]["test_identities"]
    )
    if not np.array_equal(np.sort(bundle.index), expected):
        raise RuntimeError(f"prediction rows do not match checkpoint test identities: {path}")
    return bundle


def _mixture_standard_deviation(
    prediction_a: np.ndarray,
    prediction_b: np.ndarray,
    std_a: np.ndarray,
    std_b: np.ndarray,
    weight_a: float,
) -> np.ndarray:
    mean = blend_prediction(prediction_a, prediction_b, weight_a)
    variance = weight_a * (
        np.square(std_a) + np.square(prediction_a - mean)
    ) + (1.0 - weight_a) * (
        np.square(std_b) + np.square(prediction_b - mean)
    )
    return np.sqrt(np.maximum(variance, 0.0))


def _calibrate_bundle(
    bundle: PredictionBundle,
    calibration: Mapping[str, Any],
) -> PredictionBundle:
    slope = float(calibration["slope"])
    return PredictionBundle(
        index=bundle.index,
        target=bundle.target,
        prediction=apply_affine(bundle.prediction, calibration).astype(np.float32),
        rr_std=(abs(slope) * bundle.rr_std).astype(np.float32),
        uncertainty=(abs(slope) * bundle.uncertainty).astype(np.float32),
        quality=bundle.quality,
        observable=bundle.observable,
        reference_valid=bundle.reference_valid,
        spike_rate=bundle.spike_rate,
        radar_weights=bundle.radar_weights,
    )


def _atomic_npz(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    temporary.replace(path)


def _validate_sources(
    run_a: Path,
    run_b: Path,
    config_a: Mapping[str, Any],
    config_b: Mapping[str, Any],
) -> int:
    if run_a.resolve() == run_b.resolve():
        raise ValueError("source run directories must differ")
    folds_a = int(config_a["arguments"]["folds"])
    folds_b = int(config_b["arguments"]["folds"])
    if folds_a != folds_b:
        raise ValueError("source runs use different outer-fold counts")
    for fold in range(folds_a):
        for run_dir in (run_a, run_b):
            for filename in ("snn_best.pt", "snn_test_predictions.npz"):
                path = run_dir / f"fold_{fold}" / filename
                if not path.is_file():
                    raise FileNotFoundError(path)
    return folds_a


def run(args: argparse.Namespace) -> dict[str, Any]:
    run_a = args.run_a.resolve()
    run_b = args.run_b.resolve()
    config_a = _load_json(run_a / "run_config.json")
    config_b = _load_json(run_b / "run_config.json")
    n_folds = _validate_sources(run_a, run_b, config_a, config_b)
    cache = prepare_source_cache(
        config_a, config_b, cache_dir_override=args.cache_dir
    )

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    amp = bool(args.amp and device.type == "cuda")
    torch.set_float32_matmul_precision("high")
    if args.num_threads is not None:
        torch.set_num_threads(args.num_threads)

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    label_a = _source_label(run_a)
    label_b = _source_label(run_b)
    if label_a == label_b:
        label_a, label_b = "source_a", "source_b"
    weight_grid = tuple(np.linspace(0.0, 1.0, args.weight_steps + 1).tolist())
    disagreement_grid = tuple(args.disagreement_grid)
    signature_payload = {
        "source_a_signature": config_a["run_signature"],
        "source_b_signature": config_b["run_signature"],
        "weight_grid": weight_grid,
        "disagreement_grid": disagreement_grid,
        "selection_coverages": tuple(args.selection_coverages),
    }
    signature = hashlib.sha256(
        json.dumps(signature_payload, sort_keys=True).encode("utf-8")
    ).hexdigest()[:16]

    ensemble_bundles: list[PredictionBundle] = []
    ensemble_uncalibrated_bundles: list[PredictionBundle] = []
    component_a_bundles: list[PredictionBundle] = []
    component_b_bundles: list[PredictionBundle] = []
    component_a_calibrated_bundles: list[PredictionBundle] = []
    component_b_calibrated_bundles: list[PredictionBundle] = []
    fold_vectors: list[np.ndarray] = []
    fold_weight_vectors: list[np.ndarray] = []
    fold_disagreement_vectors: list[np.ndarray] = []
    fold_reports: dict[str, Any] = {}

    for fold in range(n_folds):
        checkpoint_path_a = run_a / f"fold_{fold}" / "snn_best.pt"
        checkpoint_path_b = run_b / f"fold_{fold}" / "snn_best.pt"
        checkpoint_a = _checkpoint(checkpoint_path_a)
        checkpoint_b = _checkpoint(checkpoint_path_b)
        if int(checkpoint_a["fold"]) != fold or int(checkpoint_b["fold"]) != fold:
            raise RuntimeError(f"checkpoint fold metadata mismatch at fold {fold}")
        if checkpoint_a["run_signature"] != config_a["run_signature"]:
            raise RuntimeError(f"source A checkpoint signature mismatch at fold {fold}")
        if checkpoint_b["run_signature"] != config_b["run_signature"]:
            raise RuntimeError(f"source B checkpoint signature mismatch at fold {fold}")
        if checkpoint_a["split"] != checkpoint_b["split"]:
            raise RuntimeError(f"source split mismatch at fold {fold}")

        split = checkpoint_a["split"]
        validation_index = _indices_for_identities(
            cache.metadata, split["validation_identities"]
        )
        scaler_a = verify_checkpoint_scaler(checkpoint_a, cache)
        scaler_b = verify_checkpoint_scaler(checkpoint_b, cache)

        # Selection stage.  At this point no source outer-test prediction file
        # has been opened by this fold's code path.
        validation_a = infer_validation_checkpoint(
            checkpoint_a,
            cache,
            validation_index,
            device=device,
            batch_size=args.batch_size,
            workers=args.workers,
            amp=amp,
        )
        validation_b = infer_validation_checkpoint(
            checkpoint_b,
            cache,
            validation_index,
            device=device,
            batch_size=args.batch_size,
            workers=args.workers,
            amp=amp,
        )
        validation_a, validation_b = align_prediction_bundles(
            validation_a, validation_b
        )
        validation_identity = cache.metadata.iloc[validation_a.index][
            "identity"
        ].astype(str).to_numpy()
        weight_a, weight_search = select_blend_weight(
            validation_a.target,
            validation_a.prediction,
            validation_b.prediction,
            validation_identity,
            grid=weight_grid,
        )
        validation_blend = blend_prediction(
            validation_a.prediction, validation_b.prediction, weight_a
        )
        disagreement_coefficient, disagreement_search = (
            select_disagreement_coefficient(
                validation_a.target,
                validation_blend,
                validation_identity,
                validation_a.uncertainty,
                validation_b.uncertainty,
                validation_a.prediction,
                validation_b.prediction,
                weight_a=weight_a,
                coefficient_grid=disagreement_grid,
                coverages=args.selection_coverages,
            )
        )
        calibration_a = fit_identity_balanced_affine(
            validation_a.target,
            validation_a.prediction,
            validation_identity,
            slope_bounds=tuple(args.calibration_slope_bounds),
            intercept_bounds=tuple(args.calibration_intercept_bounds),
        )
        calibration_b = fit_identity_balanced_affine(
            validation_b.target,
            validation_b.prediction,
            validation_identity,
            slope_bounds=tuple(args.calibration_slope_bounds),
            intercept_bounds=tuple(args.calibration_intercept_bounds),
        )
        calibration_ensemble = fit_identity_balanced_affine(
            validation_a.target,
            validation_blend,
            validation_identity,
            slope_bounds=tuple(args.calibration_slope_bounds),
            intercept_bounds=tuple(args.calibration_intercept_bounds),
        )

        # Application stage.  The selection values above are now immutable.
        test_a = _load_locked_test_bundle(run_a, fold, checkpoint_a, cache.metadata)
        test_b = _load_locked_test_bundle(run_b, fold, checkpoint_b, cache.metadata)
        test_a, test_b = align_prediction_bundles(test_a, test_b)
        test_prediction = blend_prediction(
            test_a.prediction, test_b.prediction, weight_a
        )
        test_uncertainty = ensemble_uncertainty(
            test_a.uncertainty,
            test_b.uncertainty,
            test_a.prediction,
            test_b.prediction,
            weight_a=weight_a,
            disagreement_coefficient=disagreement_coefficient,
        )
        test_std = _mixture_standard_deviation(
            test_a.prediction,
            test_b.prediction,
            test_a.rr_std,
            test_b.rr_std,
            weight_a,
        )
        ensemble_uncalibrated = PredictionBundle(
            index=test_a.index,
            target=test_a.target,
            prediction=test_prediction.astype(np.float32),
            rr_std=test_std.astype(np.float32),
            uncertainty=test_uncertainty.astype(np.float32),
            quality=(
                weight_a * test_a.quality + (1.0 - weight_a) * test_b.quality
            ).astype(np.float32),
            observable=test_a.observable,
            reference_valid=test_a.reference_valid,
            spike_rate=(
                weight_a * test_a.spike_rate
                + (1.0 - weight_a) * test_b.spike_rate
            ).astype(np.float32),
            radar_weights=(
                weight_a * test_a.radar_weights
                + (1.0 - weight_a) * test_b.radar_weights
            ).astype(np.float32),
        )
        ensemble = _calibrate_bundle(
            ensemble_uncalibrated, calibration_ensemble
        )
        ensemble_bundles.append(ensemble)
        ensemble_uncalibrated_bundles.append(ensemble_uncalibrated)
        component_a_bundles.append(test_a)
        component_b_bundles.append(test_b)
        component_a_calibrated_bundles.append(
            _calibrate_bundle(test_a, calibration_a)
        )
        component_b_calibrated_bundles.append(
            _calibrate_bundle(test_b, calibration_b)
        )
        fold_vectors.append(np.full(len(ensemble), fold, dtype=np.int16))
        fold_weight_vectors.append(
            np.full(len(ensemble), weight_a, dtype=np.float32)
        )
        fold_disagreement_vectors.append(
            np.full(len(ensemble), disagreement_coefficient, dtype=np.float32)
        )

        fold_reports[str(fold)] = {
            "selection_lock": {
                "selection_data": "validation identities only",
                "outer_test_predictions_loaded_after_selection": True,
                "validation_identities": list(split["validation_identities"]),
                "outer_test_identities": list(split["test_identities"]),
                "validation_rows": int(len(validation_a)),
                "outer_test_rows": int(len(ensemble)),
            },
            "selected": {
                f"weight_{label_a}": weight_a,
                f"weight_{label_b}": 1.0 - weight_a,
                "disagreement_coefficient": disagreement_coefficient,
            },
            "validation_only_affine_calibration": {
                label_a: calibration_a,
                label_b: calibration_b,
                "ensemble": calibration_ensemble,
            },
            "validation_metrics": {
                label_a: {
                    **regression_metrics(
                        validation_a.target, validation_a.prediction
                    ),
                    **identity_macro_metrics(
                        validation_a.target,
                        validation_a.prediction,
                        validation_identity,
                    ),
                },
                label_b: {
                    **regression_metrics(
                        validation_b.target, validation_b.prediction
                    ),
                    **identity_macro_metrics(
                        validation_b.target,
                        validation_b.prediction,
                        validation_identity,
                    ),
                },
                "ensemble": {
                    **regression_metrics(validation_a.target, validation_blend),
                    **identity_macro_metrics(
                        validation_a.target,
                        validation_blend,
                        validation_identity,
                    ),
                },
                "ensemble_calibrated": {
                    **regression_metrics(
                        validation_a.target,
                        apply_affine(validation_blend, calibration_ensemble),
                    ),
                    **identity_macro_metrics(
                        validation_a.target,
                        apply_affine(validation_blend, calibration_ensemble),
                        validation_identity,
                    ),
                },
            },
            "scaler_audit": {label_a: scaler_a, label_b: scaler_b},
            "model_kwargs": {
                label_a: checkpoint_a["model_kwargs"],
                label_b: checkpoint_b["model_kwargs"],
            },
            "weight_search": weight_search,
            "disagreement_search": disagreement_search,
        }
        write_json(output_dir / f"fold_{fold}_selection.json", fold_reports[str(fold)])
        print(
            f"fold={fold} weight_{label_a}={weight_a:.2f} "
            f"disagreement={disagreement_coefficient:g} "
            f"validation_macro_mae="
            f"{fold_reports[str(fold)]['validation_metrics']['ensemble_calibrated']['macro_mae']:.4f}",
            flush=True,
        )

    def concatenate(bundles: Sequence[PredictionBundle]) -> PredictionBundle:
        joined = PredictionBundle(
            **{
                field: np.concatenate(
                    [np.asarray(getattr(bundle, field)) for bundle in bundles], axis=0
                )
                for field in PredictionBundle.__dataclass_fields__
            }
        )
        order = np.argsort(joined.index, kind="stable")
        return _subset_bundle(joined, order)

    ensemble_oof = concatenate(ensemble_bundles)
    ensemble_uncalibrated_oof = concatenate(ensemble_uncalibrated_bundles)
    component_a_oof = concatenate(component_a_bundles)
    component_b_oof = concatenate(component_b_bundles)
    component_a_calibrated_oof = concatenate(component_a_calibrated_bundles)
    component_b_calibrated_oof = concatenate(component_b_calibrated_bundles)
    raw_index = np.concatenate([bundle.index for bundle in ensemble_bundles])
    order = np.argsort(raw_index, kind="stable")
    folds = np.concatenate(fold_vectors)[order]
    weights = np.concatenate(fold_weight_vectors)[order]
    disagreement = np.concatenate(fold_disagreement_vectors)[order]
    if len(np.unique(ensemble_oof.index)) != len(ensemble_oof.index):
        raise RuntimeError("duplicate cache indices in ensemble OOF")
    expected_valid = int(cache.metadata["reference_valid"].to_numpy(dtype=bool).sum())
    if len(ensemble_oof) != expected_valid:
        raise RuntimeError(
            f"incomplete OOF: got {len(ensemble_oof)} rows, expected {expected_valid}"
        )

    npz_arrays = {
        # The core locked blend is the primary interoperable PredictionBundle.
        # Validation-affine output is retained as an explicit secondary field
        # because its OOF result can be worse despite the per-fold guard.
        **asdict(ensemble_uncalibrated_oof),
        "fold": folds,
        "run_signature": np.asarray(signature),
        "prediction_uncalibrated": ensemble_uncalibrated_oof.prediction,
        "prediction_calibrated": ensemble_oof.prediction,
        "uncertainty_uncalibrated": ensemble_uncalibrated_oof.uncertainty,
        "uncertainty_calibrated": ensemble_oof.uncertainty,
        f"prediction_{label_a}": component_a_oof.prediction,
        f"prediction_{label_b}": component_b_oof.prediction,
        f"prediction_{label_a}_calibrated": component_a_calibrated_oof.prediction,
        f"prediction_{label_b}_calibrated": component_b_calibrated_oof.prediction,
        f"uncertainty_{label_a}": component_a_oof.uncertainty,
        f"uncertainty_{label_b}": component_b_oof.uncertainty,
        f"uncertainty_{label_a}_calibrated": component_a_calibrated_oof.uncertainty,
        f"uncertainty_{label_b}_calibrated": component_b_calibrated_oof.uncertainty,
        f"rr_std_{label_a}": component_a_oof.rr_std,
        f"rr_std_{label_b}": component_b_oof.rr_std,
        f"quality_{label_a}": component_a_oof.quality,
        f"quality_{label_b}": component_b_oof.quality,
        f"weight_{label_a}": weights,
        "disagreement_coefficient": disagreement,
    }
    _atomic_npz(output_dir / "ensemble_oof.npz", **npz_arrays)

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
    ]
    rows = cache.metadata.iloc[ensemble_oof.index][columns].reset_index(drop=True)
    rows.insert(0, "cache_index", ensemble_oof.index)
    rows.insert(1, "fold", folds)
    rows["prediction_bpm"] = ensemble_uncalibrated_oof.prediction
    rows["prediction_uncalibrated_bpm"] = ensemble_uncalibrated_oof.prediction
    rows["prediction_calibrated_bpm"] = ensemble_oof.prediction
    rows[f"prediction_{label_a}_bpm"] = component_a_oof.prediction
    rows[f"prediction_{label_b}_bpm"] = component_b_oof.prediction
    rows[f"prediction_{label_a}_calibrated_bpm"] = (
        component_a_calibrated_oof.prediction
    )
    rows[f"prediction_{label_b}_calibrated_bpm"] = (
        component_b_calibrated_oof.prediction
    )
    rows[f"weight_{label_a}"] = weights
    rows["rr_std_bpm"] = ensemble_uncalibrated_oof.rr_std
    rows["uncertainty_score"] = ensemble_uncalibrated_oof.uncertainty
    rows["uncertainty_uncalibrated"] = ensemble_uncalibrated_oof.uncertainty
    rows[f"uncertainty_{label_a}"] = component_a_oof.uncertainty
    rows[f"uncertainty_{label_b}"] = component_b_oof.uncertainty
    rows["disagreement_bpm"] = np.abs(
        component_a_oof.prediction - component_b_oof.prediction
    )
    rows["disagreement_coefficient"] = disagreement
    rows["quality"] = ensemble_uncalibrated_oof.quality
    rows["spike_rate"] = ensemble_uncalibrated_oof.spike_rate
    rows.to_csv(output_dir / "ensemble_oof.csv", index=False)

    identities = cache.metadata.iloc[ensemble_oof.index]["identity"].astype(str).to_numpy()
    grouped = grouped_oof_metrics(
        ensemble_oof.target,
        ensemble_oof.prediction,
        identities,
        fold_ids=folds,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=20260827,
    )
    grouped_uncalibrated = grouped_oof_metrics(
        ensemble_uncalibrated_oof.target,
        ensemble_uncalibrated_oof.prediction,
        identities,
        fold_ids=folds,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=20260827,
    )
    risk = risk_coverage_curve(
        ensemble_oof.target,
        ensemble_oof.prediction,
        ensemble_oof.uncertainty,
        coverages=args.coverages,
        identities=identities,
    )
    risk_uncalibrated = risk_coverage_curve(
        ensemble_uncalibrated_oof.target,
        ensemble_uncalibrated_oof.prediction,
        ensemble_uncalibrated_oof.uncertainty,
        coverages=args.coverages,
        identities=identities,
    )
    component_metrics = {
        label_a: {
            "uncalibrated": grouped_oof_metrics(
                component_a_oof.target,
                component_a_oof.prediction,
                identities,
                fold_ids=folds,
                bootstrap_samples=args.bootstrap_samples,
                bootstrap_seed=20260827,
            ),
            "validation_affine": grouped_oof_metrics(
                component_a_calibrated_oof.target,
                component_a_calibrated_oof.prediction,
                identities,
                fold_ids=folds,
                bootstrap_samples=args.bootstrap_samples,
                bootstrap_seed=20260827,
            ),
        },
        label_b: {
            "uncalibrated": grouped_oof_metrics(
                component_b_oof.target,
                component_b_oof.prediction,
                identities,
                fold_ids=folds,
                bootstrap_samples=args.bootstrap_samples,
                bootstrap_seed=20260827,
            ),
            "validation_affine": grouped_oof_metrics(
                component_b_calibrated_oof.target,
                component_b_calibrated_oof.prediction,
                identities,
                fold_ids=folds,
                bootstrap_samples=args.bootstrap_samples,
                bootstrap_seed=20260827,
            ),
        },
    }
    band_metrics = {
        "ensemble_uncalibrated": rr_band_metrics(
            ensemble_uncalibrated_oof.target,
            ensemble_uncalibrated_oof.prediction,
            identities,
        ),
        "ensemble_validation_affine": rr_band_metrics(
            ensemble_oof.target, ensemble_oof.prediction, identities
        ),
        f"{label_a}_uncalibrated": rr_band_metrics(
            component_a_oof.target, component_a_oof.prediction, identities
        ),
        f"{label_a}_validation_affine": rr_band_metrics(
            component_a_calibrated_oof.target,
            component_a_calibrated_oof.prediction,
            identities,
        ),
        f"{label_b}_uncalibrated": rr_band_metrics(
            component_b_oof.target, component_b_oof.prediction, identities
        ),
        f"{label_b}_validation_affine": rr_band_metrics(
            component_b_calibrated_oof.target,
            component_b_calibrated_oof.prediction,
            identities,
        ),
    }
    band_comparison = {
        "ensemble": compare_rr_band_metrics(
            band_metrics["ensemble_uncalibrated"],
            band_metrics["ensemble_validation_affine"],
        ),
        label_a: compare_rr_band_metrics(
            band_metrics[f"{label_a}_uncalibrated"],
            band_metrics[f"{label_a}_validation_affine"],
        ),
        label_b: compare_rr_band_metrics(
            band_metrics[f"{label_b}_uncalibrated"],
            band_metrics[f"{label_b}_validation_affine"],
        ),
    }
    report: dict[str, Any] = {
        "method": "per-outer-fold validation-selected locked convex SNN blend",
        "primary_oof_variant": "uncalibrated locked blend",
        "selection_guarantee": (
            "For each fold, checkpoint inference on validation identities selected "
            "the blend and uncertainty disagreement coefficient before that fold's "
            "saved outer-test predictions were loaded."
        ),
        "run_signature": signature,
        "source_runs": {
            label_a: {
                "path": str(run_a),
                "run_signature": config_a["run_signature"],
            },
            label_b: {
                "path": str(run_b),
                "run_signature": config_b["run_signature"],
            },
        },
        "device": str(device),
        "amp": amp,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "complete_oof": True,
        "n_rows": int(len(ensemble_oof)),
        "n_identities": int(len(np.unique(identities))),
        "fold_selection": fold_reports,
        "grouped_metrics": {
            "uncalibrated": grouped_uncalibrated,
            "validation_affine": grouped,
        },
        "risk_coverage": {
            "uncalibrated": risk_uncalibrated,
            "validation_affine": risk,
        },
        "component_grouped_metrics": component_metrics,
        "rr_band_metrics": band_metrics,
        "rr_band_affine_comparison": band_comparison,
    }
    write_json(output_dir / "metrics.json", report)
    write_json(
        output_dir / "run_config.json",
        {
            "run_signature": signature,
            "source_a": str(run_a),
            "source_b": str(run_b),
            "source_labels": {"a": label_a, "b": label_b},
            "cache_dir": str(args.cache_dir) if args.cache_dir else None,
            "weight_steps": args.weight_steps,
            "disagreement_grid": disagreement_grid,
            "selection_coverages": list(args.selection_coverages),
            "report_coverages": list(args.coverages),
            "calibration_slope_bounds": list(args.calibration_slope_bounds),
            "calibration_intercept_bounds": list(args.calibration_intercept_bounds),
            "bootstrap_samples": args.bootstrap_samples,
            "device": str(device),
            "amp": amp,
        },
    )
    print(
        f"completed output={output_dir} "
        f"macro_mae={grouped_uncalibrated['identity_macro']['macro_mae']:.4f}",
        flush=True,
    )
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Locked validation-selected ensemble of two grouped SNN runs"
    )
    parser.add_argument(
        "--run-a",
        type=Path,
        default=Path("artifacts/runs/final_default_s12"),
    )
    parser.add_argument(
        "--run-b",
        type=Path,
        default=Path("artifacts/runs/final_compact_s8"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/runs/snn_nested_ensemble"),
    )
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--amp", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--num-threads", type=int)
    parser.add_argument(
        "--weight-steps",
        type=int,
        default=100,
        help="equal subdivisions of [0,1] (100 gives a 0.01 grid)",
    )
    parser.add_argument(
        "--disagreement-grid",
        type=float,
        nargs="+",
        default=list(DEFAULT_DISAGREEMENT_GRID),
    )
    parser.add_argument(
        "--selection-coverages",
        type=float,
        nargs="+",
        default=list(DEFAULT_SELECTION_COVERAGES),
    )
    parser.add_argument(
        "--calibration-slope-bounds",
        type=float,
        nargs=2,
        default=(0.7, 1.5),
        metavar=("MIN", "MAX"),
    )
    parser.add_argument(
        "--calibration-intercept-bounds",
        type=float,
        nargs=2,
        default=(-5.0, 5.0),
        metavar=("MIN", "MAX"),
    )
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
    if args.weight_steps < 1:
        parser.error("--weight-steps must be positive")
    if args.batch_size < 1 or args.workers < 0:
        parser.error("batch size must be positive and workers non-negative")
    if args.bootstrap_samples < 1:
        parser.error("--bootstrap-samples must be positive")
    if any(value < 0 or not math.isfinite(value) for value in args.disagreement_grid):
        parser.error("--disagreement-grid values must be finite and non-negative")
    if any(not 0 < value <= 1 for value in (*args.selection_coverages, *args.coverages)):
        parser.error("coverage values must be in (0,1]")
    if not any(value < 1 for value in args.selection_coverages):
        parser.error("selection coverages require at least one value below 1")
    if not (
        0 < args.calibration_slope_bounds[0]
        <= args.calibration_slope_bounds[1]
    ):
        parser.error("calibration slope bounds must be positive and ordered")
    if args.calibration_intercept_bounds[0] > args.calibration_intercept_bounds[1]:
        parser.error("calibration intercept bounds must be ordered")
    return args


def main(argv: Sequence[str] | None = None) -> None:
    run(parse_args(argv))


if __name__ == "__main__":
    main()
