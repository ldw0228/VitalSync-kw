#!/usr/bin/env python3
"""Hash-bound, post-hoc v2-i3 failure decomposition for adaptive v3r1.

This program deliberately separates two evidence scopes:

* target-aware oracle and routing-regret diagnostics are computed only from the
  already-existing i3 discovery validation predictions for outer runs 3 and 4;
* the locked six-fold OOF is used only for a retrospective outcome breakdown
  and for verifying the frozen ``fail_closed_no_action`` execution path.

The result is transparent adaptive design evidence.  It is neither
confirmatory evidence nor an authorization under the chronologically invalid
pre-existing v3 design-only contract.  In particular, no locked outer-test
candidate oracle is computed.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIXED_SEEDS = (20260828, 20260829, 20260830)
DISCOVERY_OUTER_RUNS = (3, 4)
FULL_OUTER_FOLDS = (0, 1, 2, 3, 4, 5)
FACTOR_CLASSES = np.asarray((1.0, 2.0, 3.0, 4.0), dtype=np.float64)
CANDIDATE_SOURCE_NAMES = (
    "base",
    "direct_mode",
    "classical_x1",
    "classical_x2",
    "classical_x3",
    "classical_x4",
    "radar_peak_1",
    "radar_peak_2",
    "radar_peak_3",
)
CLASSICAL_SOURCE_SLICE = slice(2, 6)

# These are disclosed post-hoc thresholds, not falsely labelled predeclared
# thresholds.  Every item must pass independently for every fixed seed.
ADAPTIVE_ROUTING_ENTRY_CRITERIA = {
    "status": "post_hoc_transparent_thresholds_not_predeclared_confirmatory_gates",
    "scope": "each_fixed_seed_on_outer_3_and_4_discovery_validation_rows_only",
    "candidate_oracle_mae_improvement_min_bpm": 0.25,
    "candidate_oracle_within_1_bpm_coverage_min_fraction": 0.80,
    "candidate_oracle_mae_max_bpm": 1.0,
    "high_rr_candidate_oracle_mae_max_bpm": 2.0,
    "high_rr_candidate_oracle_mae_improvement_min_bpm": 0.50,
}

DEFAULT_PRIMARY = Path(
    "artifacts/runs/harmonic_candidate_set_snn_v2/hcs_locked_oof/"
    "primary_evaluation/locked_hcs_primary_evaluation.json"
)
DEFAULT_READINESS = Path(
    "artifacts/runs/harmonic_candidate_set_snn_v2/hcs_locked_oof/"
    "release_readiness/locked_hcs_release_readiness.json"
)
DEFAULT_LOCKED_ROOT = Path(
    "artifacts/runs/harmonic_candidate_set_snn_v2/hcs_locked_oof"
)
DEFAULT_DISCOVERY_ROOT = Path(
    "artifacts/runs/harmonic_candidate_set_snn_v2/hcs_discovery/i3_default"
)
DEFAULT_CACHE_ROOT = Path("artifacts/cache/harmonic_set_v2_fixed_i3_pretest_v2")
DEFAULT_COMPLETION = Path(
    "artifacts/runs/harmonic_candidate_set_snn_v2/hcs_fixed_i3_pretest_v2/"
    "fixed_runtime_completion_attestation.json"
)
DEFAULT_RUNTIME_SEAL = Path(
    "artifacts/campaigns/harmonic_candidate_set_snn_v2/nested_proposer/"
    "current_source_merged/fixed_i3_pretest_runtime_seal_v2.json"
)
DEFAULT_OLD_V3_CONTRACT = Path(
    "artifacts/campaigns/directed_harmonic_factor_expert_snn_v3/"
    "RETROSPECTIVE_CAMPAIGN_CONTRACT.json"
)
DEFAULT_OLD_V3_CONFIG = Path("configs/harmonic_factor_router_v3.yaml")
DEFAULT_OUTPUT = Path(
    "artifacts/campaigns/directed_harmonic_factor_expert_snn_v3/diagnostics/"
    "v2_i3_failure_decomposition_for_adaptive_v3r1.json"
)


def canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def semantic_sha256(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _absolute(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def _display_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return str(resolved)


def file_binding(path: Path) -> dict[str, Any]:
    path = _absolute(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    return {
        "path": _display_path(path),
        "bytes": path.stat().st_size,
        "sha256": file_sha256(path),
    }


def _load_json(path: Path, *, require_content_hash: bool = False) -> dict[str, Any]:
    path = _absolute(path)
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    if require_content_hash:
        claimed = value.get("content_sha256")
        payload = dict(value)
        payload.pop("content_sha256", None)
        observed = semantic_sha256(payload)
        if claimed != observed:
            raise ValueError(f"content_sha256 mismatch: {path}")
    return value


def _float_bit_equal(left: np.ndarray, right: np.ndarray) -> bool:
    left = np.ascontiguousarray(left)
    right = np.ascontiguousarray(right)
    return (
        left.dtype == right.dtype
        and left.shape == right.shape
        and left.tobytes(order="C") == right.tobytes(order="C")
    )


def _fraction(condition: np.ndarray) -> float:
    return float(np.mean(np.asarray(condition, dtype=np.float64)))


def _error_metrics(
    errors: np.ndarray,
    target: np.ndarray,
    identity: np.ndarray,
) -> dict[str, Any]:
    errors = np.asarray(errors, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    identity = np.asarray(identity).astype(str)
    if errors.ndim != 1 or errors.shape != target.shape or target.shape != identity.shape:
        raise ValueError("metric inputs must be aligned one-dimensional arrays")
    if not len(errors) or not np.isfinite(errors).all() or not np.isfinite(target).all():
        raise ValueError("metric inputs must be non-empty and finite")
    identities = sorted(set(identity.tolist()))
    identity_mae = {
        name: float(np.mean(errors[identity == name])) for name in identities
    }
    high = (target >= 25.0) & (target <= 35.0)
    result: dict[str, Any] = {
        "rows": int(len(errors)),
        "identities": int(len(identities)),
        "mae_bpm": float(np.mean(errors)),
        "identity_macro_mae_bpm": float(np.mean(list(identity_mae.values()))),
        "rmse_bpm": float(np.sqrt(np.mean(np.square(errors)))),
        "within_1_bpm_fraction": _fraction(errors <= 1.0),
        "within_2_bpm_fraction": _fraction(errors <= 2.0),
        "over_5_bpm_fraction": _fraction(errors > 5.0),
        "identity_mae_bpm": identity_mae,
        "high_rr_25_35": {
            "rows": int(high.sum()),
            "mae_bpm": float(np.mean(errors[high])) if high.any() else None,
            "within_1_bpm_fraction": _fraction(errors[high] <= 1.0)
            if high.any()
            else None,
            "within_2_bpm_fraction": _fraction(errors[high] <= 2.0)
            if high.any()
            else None,
            "over_5_bpm_fraction": _fraction(errors[high] > 5.0)
            if high.any()
            else None,
        },
    }
    return result


def prediction_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    identity: np.ndarray,
) -> dict[str, Any]:
    prediction = np.asarray(prediction, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if prediction.shape != target.shape or not np.isfinite(prediction).all():
        raise ValueError("prediction and target must be aligned and finite")
    return _error_metrics(np.abs(prediction - target), target, identity)


def _oracle_selection(
    candidate_bpm: np.ndarray,
    candidate_mask: np.ndarray,
    target: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    candidate_bpm = np.asarray(candidate_bpm, dtype=np.float64)
    candidate_mask = np.asarray(candidate_mask, dtype=bool)
    target = np.asarray(target, dtype=np.float64)
    if candidate_bpm.ndim != 2 or candidate_mask.shape != candidate_bpm.shape:
        raise ValueError("candidate arrays must share shape [rows,candidates]")
    if target.shape != (candidate_bpm.shape[0],):
        raise ValueError("target row count differs from candidate bank")
    valid = candidate_mask & np.isfinite(candidate_bpm)
    if not valid.any(axis=1).all():
        raise ValueError("candidate oracle has a row with no finite candidate")
    distances = np.where(valid, np.abs(candidate_bpm - target[:, None]), np.inf)
    index = np.argmin(distances, axis=1)  # stable lower-index tie break
    rows = np.arange(len(target))
    prediction = candidate_bpm[rows, index]
    error = distances[rows, index]
    return index.astype(np.int64), prediction, error


def _regret_metrics(
    fallback_error: np.ndarray,
    oracle_error: np.ndarray,
    target: np.ndarray,
) -> dict[str, Any]:
    fallback_error = np.asarray(fallback_error, dtype=np.float64)
    oracle_error = np.asarray(oracle_error, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    regret = fallback_error - oracle_error
    high = (target >= 25.0) & (target <= 35.0)
    return {
        "definition": "absolute_fallback_error_minus_absolute_oracle_error_bpm",
        "mean_bpm": float(np.mean(regret)),
        "median_bpm": float(np.median(regret)),
        "p10_bpm": float(np.quantile(regret, 0.10)),
        "p90_bpm": float(np.quantile(regret, 0.90)),
        "positive_fraction": _fraction(regret > 0.0),
        "at_least_0p25_bpm_fraction": _fraction(regret >= 0.25),
        "high_rr_25_35_mean_bpm": float(np.mean(regret[high]))
        if high.any()
        else None,
    }


def _confusion_matrix(
    truth: np.ndarray,
    prediction: np.ndarray,
    selector: np.ndarray,
) -> list[list[int]]:
    matrix = np.zeros((4, 4), dtype=np.int64)
    for actual, predicted in zip(truth[selector], prediction[selector], strict=True):
        matrix[int(actual), int(predicted)] += 1
    return matrix.tolist()


def _factor_diagnostics(
    *,
    classical_rr: np.ndarray,
    target: np.ndarray,
    fallback: np.ndarray,
    candidate_oracle_prediction: np.ndarray,
) -> dict[str, Any]:
    classical_rr = np.asarray(classical_rr, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    fallback = np.asarray(fallback, dtype=np.float64)
    candidate_oracle_prediction = np.asarray(
        candidate_oracle_prediction, dtype=np.float64
    )
    available = np.isfinite(classical_rr) & (classical_rr > 0.0)
    centers = classical_rr[:, None] * FACTOR_CLASSES[None, :]
    center_valid = (
        available[:, None]
        & np.isfinite(centers)
        & (centers >= 6.0)
        & (centers <= 45.0)
    )
    if not center_valid.any(axis=1).all():
        raise ValueError("classical factor centers unavailable for one or more rows")
    target_distance = np.where(center_valid, np.abs(centers - target[:, None]), np.inf)
    fallback_distance = np.where(
        center_valid, np.abs(centers - fallback[:, None]), np.inf
    )
    candidate_distance = np.where(
        center_valid,
        np.abs(centers - candidate_oracle_prediction[:, None]),
        np.inf,
    )
    target_label = np.argmin(target_distance, axis=1)
    fallback_label = np.argmin(fallback_distance, axis=1)
    candidate_label = np.argmin(candidate_distance, axis=1)
    rows = np.arange(len(target))
    target_best = target_distance[rows, target_label]
    fallback_best = fallback_distance[rows, fallback_label]
    candidate_best = candidate_distance[rows, candidate_label]
    confident = available & (target_best <= 2.0)
    target_counts = np.bincount(target_label[confident], minlength=4)
    return {
        "factor_classes": [1, 2, 3, 4],
        "stable_tie_break": "lower_factor_class",
        "target_factor_confidence_rule": "nearest classical_x1_x2_x3_x4 center within_2_bpm",
        "coverage": {
            "classical_available_fraction": _fraction(available),
            "target_nearest_factor_within_1_bpm_fraction": _fraction(target_best <= 1.0),
            "target_nearest_factor_within_2_bpm_fraction": _fraction(confident),
            "fallback_harmonic_explainable_within_1_bpm_fraction": _fraction(
                fallback_best <= 1.0
            ),
            "fallback_harmonic_explainable_within_2_bpm_fraction": _fraction(
                fallback_best <= 2.0
            ),
            "candidate_oracle_harmonic_explainable_within_1_bpm_fraction": _fraction(
                candidate_best <= 1.0
            ),
            "candidate_oracle_harmonic_explainable_within_2_bpm_fraction": _fraction(
                candidate_best <= 2.0
            ),
        },
        "confident_rows": int(confident.sum()),
        "target_class_counts": {
            str(index + 1): int(count) for index, count in enumerate(target_counts)
        },
        "target_vs_fallback_implied_confusion_counts": _confusion_matrix(
            target_label, fallback_label, confident
        ),
        "target_vs_candidate_oracle_implied_confusion_counts": _confusion_matrix(
            target_label, candidate_label, confident
        ),
        "fallback_implied_factor_accuracy_on_confident_rows": _fraction(
            fallback_label[confident] == target_label[confident]
        )
        if confident.any()
        else None,
        "candidate_oracle_implied_factor_accuracy_on_confident_rows": _fraction(
            candidate_label[confident] == target_label[confident]
        )
        if confident.any()
        else None,
    }


def _classical_rr_by_cache_index(metadata_path: Path, expected_rows: int) -> np.ndarray:
    """Read only the two explicitly allowed fields from the cache metadata."""

    output = np.full(expected_rows, np.nan, dtype=np.float64)
    seen = np.zeros(expected_rows, dtype=bool)
    with metadata_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"cache_index", "classical_rr_bpm"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(f"metadata lacks {sorted(required)}: {metadata_path}")
        for row in reader:
            index = int(row["cache_index"])
            if index < 0 or index >= expected_rows or seen[index]:
                raise ValueError("metadata cache_index is duplicate or out of range")
            seen[index] = True
            output[index] = float(row["classical_rr_bpm"])
    if not seen.all():
        raise ValueError("metadata cache_index is not an exact cover")
    return output


def _validate_cache_manifest(
    cache_dir: Path,
    *,
    expected_seed: int,
    expected_outer: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest_path = cache_dir / "manifest.json"
    manifest = _load_json(manifest_path, require_content_hash=True)
    if not manifest.get("complete"):
        raise ValueError(f"cache manifest is incomplete: {manifest_path}")
    outputs = manifest.get("outputs")
    if not isinstance(outputs, dict):
        raise ValueError("cache manifest outputs are missing")
    required = (
        "candidate_bpm",
        "candidate_mask",
        "candidate_source_mask",
        "candidate_primary_source",
        "feature_names",
        "metadata",
    )
    bindings: dict[str, Any] = {"manifest": file_binding(manifest_path)}
    for name in required:
        item = outputs.get(name)
        if not isinstance(item, dict) or not isinstance(item.get("filename"), str):
            raise ValueError(f"cache manifest lacks output {name!r}")
        path = cache_dir / item["filename"]
        binding = file_binding(path)
        if binding["sha256"] != item.get("sha256") or binding["bytes"] != item.get(
            "bytes"
        ):
            raise ValueError(f"cache output binding mismatch: {path}")
        bindings[name] = binding
    feature_names = _load_json(cache_dir / outputs["feature_names"]["filename"])
    if tuple(feature_names.get("candidate_source_names", ())) != CANDIDATE_SOURCE_NAMES:
        raise ValueError("candidate source order differs from the frozen cache schema")
    return manifest, bindings


def load_discovery_unit(
    *,
    seed: int,
    outer_run: int,
    discovery_root: Path,
    cache_root: Path,
) -> tuple[dict[str, Any], dict[str, np.ndarray], dict[str, Any]]:
    """Load one target-aware discovery-validation unit and its fixed cache."""

    if outer_run not in DISCOVERY_OUTER_RUNS:
        raise ValueError("candidate oracle is restricted to discovery outer runs 3 and 4")
    unit_dir = discovery_root / f"outer_{outer_run}_seed_{seed}"
    prediction_path = unit_dir / "validation_predictions.npz"
    cache_dir = cache_root / f"outer_{outer_run}_seed_{seed}"
    manifest, cache_bindings = _validate_cache_manifest(
        cache_dir, expected_seed=seed, expected_outer=outer_run
    )
    prediction_binding = file_binding(prediction_path)
    with np.load(prediction_path, allow_pickle=False) as predictions:
        required = {
            "cache_index",
            "target_rr_bpm",
            "identity",
            "fallback_rr_bpm",
            "source_rr_bpm",
            "final_rr_bpm",
            "applied_pull",
        }
        if not required.issubset(predictions.files):
            raise ValueError(f"discovery prediction keys are incomplete: {prediction_path}")
        values = {name: np.asarray(predictions[name]).copy() for name in required}
    rows = len(values["cache_index"])
    if not rows or any(np.asarray(value).shape != (rows,) for value in values.values()):
        raise ValueError("discovery validation arrays are not aligned vectors")
    if len(np.unique(values["cache_index"])) != rows:
        raise ValueError("discovery validation cache_index contains duplicates")
    if not _float_bit_equal(values["fallback_rr_bpm"], values["final_rr_bpm"]):
        raise ValueError("discovery final is not bit-exact fallback under no-action policy")
    if not np.all(np.asarray(values["applied_pull"]) == 0.0):
        raise ValueError("discovery applied_pull is not exact zero")
    row_count = int(manifest.get("row_count", -1))
    index = np.asarray(values["cache_index"], dtype=np.int64)
    if np.any(index < 0) or np.any(index >= row_count):
        raise ValueError("discovery cache_index is outside the fixed cache")

    candidate_bpm_all = np.load(cache_dir / "candidate_bpm.npy", mmap_mode="r")
    candidate_mask_all = np.load(cache_dir / "candidate_mask.npy", mmap_mode="r")
    source_mask_all = np.load(cache_dir / "candidate_source_mask.npy", mmap_mode="r")
    primary_all = np.load(cache_dir / "candidate_primary_source.npy", mmap_mode="r")
    if (
        candidate_bpm_all.shape != (row_count, 12)
        or candidate_mask_all.shape != (row_count, 12)
        or source_mask_all.shape != (row_count, 12, len(CANDIDATE_SOURCE_NAMES))
        or primary_all.shape != (row_count, 12)
    ):
        raise ValueError("fixed-i3 candidate cache has an unexpected shape")
    classical_all = _classical_rr_by_cache_index(
        cache_dir / "metadata.csv", expected_rows=row_count
    )
    raw = {
        "target": np.asarray(values["target_rr_bpm"], dtype=np.float64),
        "identity": np.asarray(values["identity"]).astype(str),
        "fallback": np.asarray(values["fallback_rr_bpm"], dtype=np.float64),
        "source": np.asarray(values["source_rr_bpm"], dtype=np.float64),
        "candidate_bpm": np.asarray(candidate_bpm_all[index], dtype=np.float64),
        "candidate_mask": np.asarray(candidate_mask_all[index], dtype=bool),
        "candidate_source_mask": np.asarray(source_mask_all[index], dtype=bool),
        "candidate_primary_source": np.asarray(primary_all[index], dtype=np.int16),
        "classical_rr": np.asarray(classical_all[index], dtype=np.float64),
    }
    if not (
        np.isfinite(raw["target"]).all()
        and np.isfinite(raw["fallback"]).all()
        and np.isfinite(raw["source"]).all()
    ):
        raise ValueError("discovery validation target or prediction is non-finite")
    bindings = {
        "validation_predictions": prediction_binding,
        "cache": cache_bindings,
    }
    observation = {
        "outer_run": int(outer_run),
        "validation_physical_fold": int((outer_run + 1) % 6),
        "seed": int(seed),
        "rows": rows,
        "final_bit_exact_fallback": True,
        "applied_pull_exact_zero": True,
        "source_bit_exact_fallback_fraction": _fraction(
            np.asarray(values["source_rr_bpm"]).view(np.uint32)
            == np.asarray(values["fallback_rr_bpm"]).view(np.uint32)
        ),
        "target_source": "existing_i3_default_validation_predictions_only",
        "locked_outer_test_target_opened_for_this_unit": False,
    }
    return observation, raw, bindings


def analyze_discovery_arrays(raw: Mapping[str, np.ndarray]) -> dict[str, Any]:
    target = raw["target"]
    identity = raw["identity"]
    fallback = raw["fallback"]
    candidate_index, candidate_prediction, candidate_error = _oracle_selection(
        raw["candidate_bpm"], raw["candidate_mask"], target
    )
    factor_candidate_mask = raw["candidate_mask"] & raw["candidate_source_mask"][
        ..., CLASSICAL_SOURCE_SLICE
    ].any(axis=-1)
    factor_index, factor_prediction, factor_error = _oracle_selection(
        raw["candidate_bpm"], factor_candidate_mask, target
    )
    fallback_error = np.abs(fallback - target)
    source_bits = raw["candidate_source_mask"][
        np.arange(len(target)), candidate_index
    ]
    primary = raw["candidate_primary_source"][
        np.arange(len(target)), candidate_index
    ]
    source_support_counts = {
        name: int(source_bits[:, index].sum())
        for index, name in enumerate(CANDIDATE_SOURCE_NAMES)
    }
    primary_counts = {
        name: int(np.sum(primary == index))
        for index, name in enumerate(CANDIDATE_SOURCE_NAMES)
    }
    primary_counts["padding_or_unknown"] = int(np.sum(primary < 0))
    return {
        "fallback": prediction_metrics(fallback, target, identity),
        "source_model": prediction_metrics(raw["source"], target, identity),
        "candidate_oracle": {
            **_error_metrics(candidate_error, target, identity),
            "coverage_within_0p5_bpm_fraction": _fraction(candidate_error <= 0.5),
            "coverage_within_1_bpm_fraction": _fraction(candidate_error <= 1.0),
            "coverage_within_2_bpm_fraction": _fraction(candidate_error <= 2.0),
            "selected_candidate_source_support_counts": source_support_counts,
            "selected_candidate_primary_source_counts": primary_counts,
            "diagnostic_upper_bound_only": True,
        },
        "classical_factor_candidate_oracle": {
            **_error_metrics(factor_error, target, identity),
            "coverage_within_0p5_bpm_fraction": _fraction(factor_error <= 0.5),
            "coverage_within_1_bpm_fraction": _fraction(factor_error <= 1.0),
            "coverage_within_2_bpm_fraction": _fraction(factor_error <= 2.0),
            "candidate_source_scope": [
                "classical_x1",
                "classical_x2",
                "classical_x3",
                "classical_x4",
            ],
            "diagnostic_upper_bound_only": True,
        },
        "routing_regret": {
            "candidate_oracle": _regret_metrics(
                fallback_error, candidate_error, target
            ),
            "classical_factor_candidate_oracle": _regret_metrics(
                fallback_error, factor_error, target
            ),
        },
        "classical_factor_class_confusion_and_coverage": _factor_diagnostics(
            classical_rr=raw["classical_rr"],
            target=target,
            fallback=fallback,
            candidate_oracle_prediction=candidate_prediction,
        ),
        "_oracle_indices_for_validation_only": {
            "candidate_index_min": int(candidate_index.min()),
            "candidate_index_max": int(candidate_index.max()),
            "factor_candidate_index_min": int(factor_index.min()),
            "factor_candidate_index_max": int(factor_index.max()),
        },
    }


def _concatenate_raw(units: Sequence[Mapping[str, np.ndarray]]) -> dict[str, np.ndarray]:
    keys = tuple(units[0].keys())
    if any(tuple(unit.keys()) != keys for unit in units):
        raise ValueError("discovery unit raw schemas differ")
    return {key: np.concatenate([unit[key] for unit in units], axis=0) for key in keys}


def evaluate_adaptive_routing_entry(
    seed_aggregates: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    per_seed: dict[str, Any] = {}
    for seed in map(str, FIXED_SEEDS):
        if seed not in seed_aggregates:
            raise ValueError(f"missing fixed-seed discovery aggregate: {seed}")
        value = seed_aggregates[seed]
        fallback = value["fallback"]
        candidate = value["candidate_oracle"]
        fallback_high = fallback["high_rr_25_35"]
        candidate_high = candidate["high_rr_25_35"]
        if not fallback_high["rows"] or not candidate_high["rows"]:
            raise ValueError("adaptive routing entry requires high-RR discovery rows")
        observed = {
            "candidate_oracle_mae_improvement_bpm": float(
                fallback["mae_bpm"] - candidate["mae_bpm"]
            ),
            "candidate_oracle_within_1_bpm_coverage_fraction": float(
                candidate["coverage_within_1_bpm_fraction"]
            ),
            "candidate_oracle_mae_bpm": float(candidate["mae_bpm"]),
            "high_rr_candidate_oracle_mae_bpm": float(candidate_high["mae_bpm"]),
            "high_rr_candidate_oracle_mae_improvement_bpm": float(
                fallback_high["mae_bpm"] - candidate_high["mae_bpm"]
            ),
        }
        gates = {
            "candidate_oracle_mae_improvement": observed[
                "candidate_oracle_mae_improvement_bpm"
            ]
            >= ADAPTIVE_ROUTING_ENTRY_CRITERIA[
                "candidate_oracle_mae_improvement_min_bpm"
            ],
            "candidate_oracle_within_1_bpm_coverage": observed[
                "candidate_oracle_within_1_bpm_coverage_fraction"
            ]
            >= ADAPTIVE_ROUTING_ENTRY_CRITERIA[
                "candidate_oracle_within_1_bpm_coverage_min_fraction"
            ],
            "candidate_oracle_mae": observed["candidate_oracle_mae_bpm"]
            <= ADAPTIVE_ROUTING_ENTRY_CRITERIA["candidate_oracle_mae_max_bpm"],
            "high_rr_candidate_oracle_mae": observed[
                "high_rr_candidate_oracle_mae_bpm"
            ]
            <= ADAPTIVE_ROUTING_ENTRY_CRITERIA[
                "high_rr_candidate_oracle_mae_max_bpm"
            ],
            "high_rr_candidate_oracle_mae_improvement": observed[
                "high_rr_candidate_oracle_mae_improvement_bpm"
            ]
            >= ADAPTIVE_ROUTING_ENTRY_CRITERIA[
                "high_rr_candidate_oracle_mae_improvement_min_bpm"
            ],
        }
        per_seed[seed] = {
            "observed": observed,
            "gates": gates,
            "all_gates_passed": bool(all(gates.values())),
        }
    supported = bool(all(value["all_gates_passed"] for value in per_seed.values()))
    return {
        "criteria": ADAPTIVE_ROUTING_ENTRY_CRITERIA,
        "criterion_inputs": (
            "outer_3_and_4_i3_default_discovery_validation_predictions_and_"
            "their_fixed_i3_candidate_caches_only"
        ),
        "locked_full_oof_outcomes_used_in_criterion": False,
        "per_seed": per_seed,
        "all_fixed_seeds_passed": supported,
        "adaptive_v3r1_routing_focused_design_entry_supported": supported,
        "interpretation": (
            "candidate-set feasibility plus fallback-to-oracle regret supports a "
            "routing-focused adaptive intervention; it does not identify a causal "
            "failure mechanism and does not estimate deployable performance"
        ),
        "causal_claim": False,
        "confirmatory_claim": False,
    }


def audit_locked_no_action_policy(locked_root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    plan_path = locked_root / "locked_oof_plan.json"
    seal_path = locked_root / "predictions_seal.json"
    joined_path = locked_root / "locked_hcs_oof_joined.npz"
    plan = _load_json(plan_path)
    seal = _load_json(seal_path)
    if plan.get("frozen_common_policy_fast_path") != "fail_closed_no_action":
        raise ValueError("locked common policy is not fail_closed_no_action")
    units = seal.get("units")
    if not isinstance(units, list) or len(units) != 18:
        raise ValueError("locked prediction seal must contain exactly 18 units")
    observed_keys: set[tuple[int, int]] = set()
    unit_results: list[dict[str, Any]] = []
    unit_bindings: list[dict[str, Any]] = []
    for unit in units:
        outer = int(unit["outer_fold"])
        seed = int(unit["seed"])
        key = (outer, seed)
        if key in observed_keys:
            raise ValueError("duplicate locked prediction unit")
        observed_keys.add(key)
        prediction_item = unit.get("prediction", {})
        prediction_path = Path(prediction_item["path"])
        prediction_path = (
            prediction_path
            if prediction_path.is_absolute()
            else PROJECT_ROOT / prediction_path
        )
        binding = file_binding(prediction_path)
        if binding["sha256"] != prediction_item.get("sha256") or binding[
            "bytes"
        ] != prediction_item.get("bytes"):
            raise ValueError("prediction seal byte binding mismatch")
        with np.load(prediction_path, allow_pickle=False) as data:
            fallback = np.asarray(data["fallback_rr_bpm"])
            source = np.asarray(data["source_rr_bpm"])
            final = np.asarray(data["final_rr_bpm"])
            applied = np.asarray(data["applied_pull"])
            if int(data["outer_fold"]) != outer or int(data["seed"]) != seed:
                raise ValueError("locked prediction scalar identity mismatch")
        bit_equal = _float_bit_equal(fallback, source) and _float_bit_equal(
            fallback, final
        )
        zero_pull = bool(np.isfinite(applied).all() and np.all(applied == 0.0))
        if not bit_equal or not zero_pull:
            raise ValueError("locked no-action unit is not bit exact")
        unit_results.append(
            {
                "outer_fold": outer,
                "seed": seed,
                "rows": int(len(fallback)),
                "fallback_source_final_bit_exact": True,
                "applied_pull_exact_zero": True,
                "passed": True,
            }
        )
        unit_bindings.append(binding)
    expected = {(outer, seed) for seed in FIXED_SEEDS for outer in FULL_OUTER_FOLDS}
    if observed_keys != expected:
        raise ValueError("locked units are not the exact fixed 6-fold x 3-seed cover")

    audit = {
        "common_policy": "fail_closed_no_action",
        "unit_count": 18,
        "all_18_units_passed": True,
        "fallback_source_final_bit_exact_all_units": True,
        "applied_pull_exact_zero_all_units": True,
        "joined_locked_oof_array_opened": False,
        "units": unit_results,
    }
    bindings = {
        "locked_plan": file_binding(plan_path),
        "predictions_seal": file_binding(seal_path),
        "joined_locked_oof": file_binding(joined_path),
        "sealed_unit_predictions": unit_bindings,
    }
    return audit, bindings


def locked_outcome_from_primary_report(primary: Mapping[str, Any]) -> dict[str, Any]:
    """Copy the evaluator's fold strata without reopening locked raw targets."""

    outcome: dict[str, Any] = {}
    per_seed = primary.get("per_seed")
    if not isinstance(per_seed, dict):
        raise ValueError("primary report lacks per_seed outcomes")
    for seed in FIXED_SEEDS:
        seed_value = per_seed.get(str(seed))
        try:
            fold = seed_value["candidates"]["locked_final"]["strata"]["fold"]
        except (KeyError, TypeError) as error:
            raise ValueError("primary report lacks locked-final fold strata") from error
        if set(fold) != {str(value) for value in FULL_OUTER_FOLDS}:
            raise ValueError("primary fold strata are not the exact six-fold cover")
        outcome[str(seed)] = fold
    return {
        "classification": "retrospective_locked_outer_test_outcome_breakdown_only",
        "source": "hash_bound_locked_primary_evaluation_json_fold_strata",
        "locked_raw_target_or_joined_array_opened": False,
        "design_selection_eligible": False,
        "candidate_or_factor_oracle_computed": False,
        "routing_entry_criterion_input": False,
        "per_seed_per_outer_fold": outcome,
    }


def _write_create_once_json(path: Path, report: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(
        report,
        sort_keys=True,
        indent=2,
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8") + b"\n"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    descriptor = os.open(path, flags, 0o444)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(serialized)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        try:
            path.unlink()
        except FileNotFoundError:
            pass
        raise
    path.chmod(0o444)


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    primary_path = _absolute(args.primary)
    readiness_path = _absolute(args.readiness)
    locked_root = _absolute(args.locked_root)
    discovery_root = _absolute(args.discovery_root)
    cache_root = _absolute(args.cache_root)
    completion_path = _absolute(args.fixed_completion)
    runtime_seal_path = _absolute(args.runtime_seal)
    old_contract_path = _absolute(args.old_v3_contract)
    old_config_path = _absolute(args.old_v3_config)

    primary = _load_json(primary_path, require_content_hash=True)
    readiness = _load_json(readiness_path, require_content_hash=True)
    completion = _load_json(completion_path, require_content_hash=True)
    if primary.get("all_fixed_seeds_point_gates_passed") is not False:
        raise ValueError("v2 fixed-i3 locked primary accuracy did not fail")
    if readiness.get("category_gate_status", {}).get("primary_accuracy") is not False:
        raise ValueError("release readiness does not bind the primary accuracy failure")
    if int(completion.get("completed_units", -1)) != 18:
        raise ValueError("fixed-i3 completion does not bind all 18 units")
    if completion.get("outer_test_opened") is not False:
        raise ValueError("fixed-i3 pretest completion says outer test was opened")

    locked_policy, locked_bindings = audit_locked_no_action_policy(locked_root)
    locked_outcome = locked_outcome_from_primary_report(primary)
    discovery_per_unit: dict[str, Any] = {}
    discovery_per_seed: dict[str, Any] = {}
    discovery_bindings: list[dict[str, Any]] = []
    all_seed_raw: list[Mapping[str, np.ndarray]] = []
    for seed in FIXED_SEEDS:
        seed_raw: list[Mapping[str, np.ndarray]] = []
        for outer in DISCOVERY_OUTER_RUNS:
            observation, raw, bindings = load_discovery_unit(
                seed=seed,
                outer_run=outer,
                discovery_root=discovery_root,
                cache_root=cache_root,
            )
            key = f"outer_{outer}_seed_{seed}"
            discovery_per_unit[key] = {
                "scope": observation,
                "diagnostics": analyze_discovery_arrays(raw),
            }
            discovery_bindings.append(
                {"outer_run": outer, "seed": seed, **bindings}
            )
            seed_raw.append(raw)
            all_seed_raw.append(raw)
        discovery_per_seed[str(seed)] = analyze_discovery_arrays(
            _concatenate_raw(seed_raw)
        )
    pooled_diagnostic = analyze_discovery_arrays(_concatenate_raw(all_seed_raw))
    adaptive_entry = evaluate_adaptive_routing_entry(discovery_per_seed)

    input_bindings = {
        "source": file_binding(Path(__file__)),
        "locked_primary_evaluation": file_binding(primary_path),
        "release_readiness": file_binding(readiness_path),
        "fixed_i3_completion": file_binding(completion_path),
        "fixed_i3_runtime_seal": file_binding(runtime_seal_path),
        "old_v3_design_only_contract": file_binding(old_contract_path),
        "old_v3_config": file_binding(old_config_path),
        "locked_policy_and_outcome": locked_bindings,
        "discovery_validation_and_fixed_i3_cache_units": discovery_bindings,
    }
    payload: dict[str, Any] = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "classification": (
            "posthoc_transparent_adaptive_v3r1_design_diagnostic_"
            "historical_cohort_not_confirmatory"
        ),
        "claim_boundary": {
            "commercial_claim_authorized": False,
            "confirmatory_claim_authorized": False,
            "causal_failure_mechanism_claimed": False,
            "old_v3_contract_authorization_claimed": False,
            "old_v3_contract_chronology_compliant": False,
            "required_route": (
                "a_new_additive_adaptive_retrospective_v3r1_authorization_must_"
                "bind_this_diagnostic_before_training"
            ),
            "maximum_interpretation": (
                "post_hoc_discovery_validation_feasibility_support_for_a_"
                "routing_focused_adaptive_design"
            ),
        },
        "evidence_scope": {
            "target_aware_design_diagnostics": {
                "outer_runs": [3, 4],
                "physical_validation_folds": [4, 5],
                "source": "existing_i3_default_validation_predictions_plus_fixed_i3_cache",
                "locked_outer_test_rows_used": False,
                "eligible_for_adaptive_design_reasoning": True,
            },
            "full_locked_oof": {
                "outer_folds": [0, 1, 2, 3, 4, 5],
                "use": "retrospective_outcome_breakdown_and_no_action_bit_audit_only",
                "eligible_for_routing_or_model_selection": False,
                "candidate_or_factor_oracle_computed": False,
            },
            "same_historical_cohort_repeatedly_observed": True,
            "post_hoc_thresholds_are_not_predeclared": True,
        },
        "v2_locked_entry_failure_binding": {
            "all_fixed_seeds_point_gates_passed": primary[
                "all_fixed_seeds_point_gates_passed"
            ],
            "fixed_seed_gate_status": primary["fixed_seed_gate_status"],
            "release_readiness_primary_accuracy_passed": readiness[
                "category_gate_status"
            ]["primary_accuracy"],
            "used_only_to_establish_v2_failure_and_report_outcome": True,
            "used_in_routing_dominance_criterion": False,
        },
        "locked_common_policy_audit": locked_policy,
        "full_locked_outer_fold_outcome_breakdown": locked_outcome,
        "discovery_validation_failure_decomposition": {
            "per_unit": discovery_per_unit,
            "per_seed_outer_3_and_4_aggregate": discovery_per_seed,
            "pooled_three_seed_diagnostic_descriptive_only": pooled_diagnostic,
            "cross_seed_pooling_used_for_entry_decision": False,
        },
        "adaptive_routing_entry_decision": adaptive_entry,
        "input_bindings": input_bindings,
    }
    payload["content_sha256"] = semantic_sha256(payload)
    return payload


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary", type=Path, default=DEFAULT_PRIMARY)
    parser.add_argument("--readiness", type=Path, default=DEFAULT_READINESS)
    parser.add_argument("--locked-root", type=Path, default=DEFAULT_LOCKED_ROOT)
    parser.add_argument("--discovery-root", type=Path, default=DEFAULT_DISCOVERY_ROOT)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--fixed-completion", type=Path, default=DEFAULT_COMPLETION)
    parser.add_argument("--runtime-seal", type=Path, default=DEFAULT_RUNTIME_SEAL)
    parser.add_argument("--old-v3-contract", type=Path, default=DEFAULT_OLD_V3_CONTRACT)
    parser.add_argument("--old-v3-config", type=Path, default=DEFAULT_OLD_V3_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report = build_report(args)
    output = _absolute(args.output)
    _write_create_once_json(output, report)
    print(
        json.dumps(
            {
                "output": _display_path(output),
                "content_sha256": report["content_sha256"],
                "adaptive_v3r1_routing_focused_design_entry_supported": report[
                    "adaptive_routing_entry_decision"
                ]["adaptive_v3r1_routing_focused_design_entry_supported"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
