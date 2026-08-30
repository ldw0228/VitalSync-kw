"""Evaluation utilities that treat identities, not overlapping windows, as units."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np


def _regression_vectors(
    y_true: np.ndarray, y_pred: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    target = np.asarray(y_true, dtype=float)
    prediction = np.asarray(y_pred, dtype=float)
    if target.shape != prediction.shape or target.ndim != 1:
        raise ValueError("y_true and y_pred must be equal-length vectors")
    if not len(target):
        raise ValueError("y_true and y_pred must not be empty")
    if not (np.isfinite(target).all() and np.isfinite(prediction).all()):
        raise ValueError("y_true and y_pred must contain only finite values")
    return target, prediction


def _group_vector(groups: Iterable[Any], length: int, name: str) -> np.ndarray:
    raw = np.asarray(list(groups), dtype=object)
    if raw.ndim != 1 or len(raw) != length:
        raise ValueError(f"{name} count does not match targets")
    for value in raw:
        if value is None or (isinstance(value, (float, np.floating)) and np.isnan(value)):
            raise ValueError(f"{name} contains missing values")
    return raw.astype(str)


def concordance_correlation_coefficient(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true, y_pred = _regression_vectors(y_true, y_pred)
    covariance = float(np.mean((y_true - y_true.mean()) * (y_pred - y_pred.mean())))
    denominator = float(y_true.var() + y_pred.var() + (y_true.mean() - y_pred.mean()) ** 2)
    return 2.0 * covariance / denominator if denominator > 0 else float("nan")


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true, y_pred = _regression_vectors(y_true, y_pred)
    error = y_pred - y_true
    absolute = np.abs(error)
    error_standard_deviation = float(np.std(error, ddof=1 if len(error) > 1 else 0))
    return {
        "n": float(len(y_true)),
        "mae": float(np.mean(absolute)),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "median_ae": float(np.median(absolute)),
        "p90_ae": float(np.quantile(absolute, 0.90)),
        "p95_ae": float(np.quantile(absolute, 0.95)),
        "p99_ae": float(np.quantile(absolute, 0.99)),
        "within_1": float(np.mean(absolute <= 1.0)),
        "within_2": float(np.mean(absolute <= 2.0)),
        "within_3": float(np.mean(absolute <= 3.0)),
        "catastrophic_over_5": float(np.mean(absolute > 5.0)),
        "bias": float(np.mean(error)),
        "loa_low": float(np.mean(error) - 1.96 * error_standard_deviation),
        "loa_high": float(np.mean(error) + 1.96 * error_standard_deviation),
        "ccc": concordance_correlation_coefficient(y_true, y_pred),
    }


def identity_macro_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    identities: Iterable[str],
) -> dict[str, float]:
    y_true, y_pred = _regression_vectors(y_true, y_pred)
    identities = _group_vector(identities, len(y_true), "identity")
    per_identity = [
        regression_metrics(y_true[identities == name], y_pred[identities == name])
        for name in np.unique(identities)
    ]
    keys = [key for key in per_identity[0] if key != "n"]
    result: dict[str, float] = {}
    for key in keys:
        metric_values = np.asarray([row[key] for row in per_identity], dtype=float)
        finite = metric_values[np.isfinite(metric_values)]
        result[f"macro_{key}"] = float(finite.mean()) if len(finite) else float("nan")
    return result


def risk_coverage_curve(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    uncertainty: np.ndarray,
    coverages: Iterable[float] = (1.0, 0.9, 0.8, 0.7, 0.5),
    *,
    identities: Iterable[str] | None = None,
) -> list[dict[str, float]]:
    """Return selective risk after keeping the least-uncertain predictions.

    Counts use ``ceil(n * requested_coverage)`` so a row never reports less
    coverage than requested.  When identities are supplied, each row also
    contains identity-macro metrics over the identities represented at that
    operating point.
    """

    y_true, y_pred = _regression_vectors(y_true, y_pred)
    uncertainty = np.asarray(uncertainty, dtype=float)
    if uncertainty.shape != y_true.shape:
        raise ValueError("inputs must have identical shapes")
    if not np.isfinite(uncertainty).all():
        raise ValueError("uncertainty must contain only finite values")
    identity_array = (
        None if identities is None else _group_vector(identities, len(y_true), "identity")
    )
    order = np.argsort(uncertainty, kind="stable")
    rows: list[dict[str, float]] = []
    for requested in coverages:
        if not 0 < requested <= 1:
            raise ValueError("coverage must be in (0, 1]")
        count = max(1, int(np.ceil(len(order) * requested)))
        selected = order[:count]
        metrics = regression_metrics(y_true[selected], y_pred[selected])
        metrics["requested_coverage"] = float(requested)
        metrics["coverage"] = count / len(order)
        metrics["uncertainty_threshold"] = float(uncertainty[selected].max())
        if identity_array is not None:
            metrics.update(
                identity_macro_metrics(
                    y_true[selected], y_pred[selected], identity_array[selected]
                )
            )
            metrics["n_identities"] = float(len(np.unique(identity_array[selected])))
        rows.append(metrics)
    return rows


def identity_cluster_bootstrap_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    identities: Iterable[str],
    *,
    metric: str = "mae",
    samples: int = 2000,
    confidence: float = 0.95,
    seed: int = 20260827,
) -> dict[str, float]:
    """Bootstrap a confidence interval for an identity-macro metric.

    Each draw resamples complete identities and then averages the chosen
    per-identity metric.  This makes overlapping windows within an identity a
    cluster instead of pretending that every window is an independent sample.
    """

    y_true, y_pred = _regression_vectors(y_true, y_pred)
    identities = _group_vector(identities, len(y_true), "identity")
    if samples <= 0:
        raise ValueError("samples must be positive")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be in (0, 1)")
    names = np.unique(identities)
    subject_metrics: list[float] = []
    for name in names:
        row = regression_metrics(y_true[identities == name], y_pred[identities == name])
        if metric not in row or metric == "n":
            raise KeyError(f"unsupported regression metric: {metric!r}")
        subject_metrics.append(row[metric])
    subject_values = np.asarray(subject_metrics, dtype=float)
    subject_values = subject_values[np.isfinite(subject_values)]
    if not len(subject_values):
        raise ValueError(f"metric {metric!r} is undefined for every identity")

    rng = np.random.default_rng(seed)
    draws = rng.choice(
        subject_values, size=(int(samples), len(subject_values)), replace=True
    ).mean(axis=1)
    alpha = (1.0 - confidence) / 2.0
    return {
        "estimate": float(subject_values.mean()),
        "lower": float(np.quantile(draws, alpha)),
        "upper": float(np.quantile(draws, 1.0 - alpha)),
        "confidence": float(confidence),
        "samples": float(samples),
        "n_identities": float(len(subject_values)),
    }


def clustered_bootstrap_mae(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    identities: Iterable[str],
    *,
    samples: int = 2000,
    seed: int = 20260827,
) -> tuple[float, float, float]:
    """Identity-cluster bootstrap CI for identity-macro MAE."""

    result = identity_cluster_bootstrap_ci(
        y_true,
        y_pred,
        identities,
        metric="mae",
        samples=samples,
        confidence=0.95,
        seed=seed,
    )
    return result["estimate"], result["lower"], result["upper"]


def grouped_oof_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    identities: Iterable[str],
    *,
    fold_ids: Iterable[Any] | None = None,
    bootstrap_samples: int = 2000,
    bootstrap_seed: int = 20260827,
) -> dict[str, Any]:
    """Summarize grouped out-of-fold predictions with identity-level audits.

    If ``fold_ids`` are supplied, every identity is required to occur in
    exactly one test fold.  This catches a common leakage error where windows
    from one person receive OOF predictions from different splits.
    """

    y_true, y_pred = _regression_vectors(y_true, y_pred)
    identity_array = _group_vector(identities, len(y_true), "identity")
    names = np.unique(identity_array)
    per_identity = {
        name: regression_metrics(
            y_true[identity_array == name], y_pred[identity_array == name]
        )
        for name in names
    }
    result: dict[str, Any] = {
        "overall": regression_metrics(y_true, y_pred),
        "identity_macro": identity_macro_metrics(y_true, y_pred, identity_array),
        "identity_cluster_bootstrap_mae": identity_cluster_bootstrap_ci(
            y_true,
            y_pred,
            identity_array,
            metric="mae",
            samples=bootstrap_samples,
            seed=bootstrap_seed,
        ),
        "n_identities": int(len(names)),
        "per_identity": per_identity,
    }

    if fold_ids is not None:
        fold_array = _group_vector(fold_ids, len(y_true), "fold")
        violations = {
            name: np.unique(fold_array[identity_array == name]).tolist()
            for name in names
            if len(np.unique(fold_array[identity_array == name])) != 1
        }
        if violations:
            raise ValueError(f"identities span multiple OOF folds: {violations}")
        result["per_fold"] = {
            fold: {
                "overall": regression_metrics(y_true[fold_array == fold], y_pred[fold_array == fold]),
                "identity_macro": identity_macro_metrics(
                    y_true[fold_array == fold],
                    y_pred[fold_array == fold],
                    identity_array[fold_array == fold],
                ),
                "n_identities": int(len(np.unique(identity_array[fold_array == fold]))),
            }
            for fold in np.unique(fold_array)
        }
    return result
