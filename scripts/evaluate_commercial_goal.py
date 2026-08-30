#!/usr/bin/env python3
"""Audit a locked OOF RR candidate against the declared commercial goal.

The command is deliberately an acceptance *audit*, not a model selector.  It
never searches prediction columns, thresholds, non-overlap phases, or metric
variants.  The caller must name one prediction and (optionally) one uncertainty
column that were locked without outer-test labels.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from snn_rr.metrics import (  # noqa: E402
    identity_macro_metrics,
    regression_metrics,
    risk_coverage_curve,
)
from snn_rr.cache import load_feature_cache  # noqa: E402


GOAL_TARGETS = {
    "overall_mae_max_bpm": 1.0,
    "identity_macro_mae_max_bpm": 1.0,
    "overall_rmse_max_bpm": 1.8,
    "within_2_min_fraction": 0.90,
    "over_5_max_fraction": 0.03,
    "high_rr_25_35_mae_max_bpm": 2.0,
}
EXPECTED_VALID_REFERENCE_ROWS = 2327
EXPECTED_OUTER_FOLDS = 6
EXPECTED_IDENTITIES = 18
REQUIRED_RADAR_CONDITIONS = {
    "radars_123",
    "radars_12",
    "radars_13",
    "radars_23",
    "radar_1",
    "radar_2",
    "radar_3",
}
CALIBRATION_STATUSES = {
    "not_assessed",
    "ranking_only_not_interval_calibrated",
    "validation_locked_interval_calibrated",
}
CALIBRATION_PASS_STATUSES = {"validation_locked_interval_calibrated"}
REQUIRED_COLUMNS = {
    "cache_index",
    "fold",
    "identity",
    "session_id",
    "window_number",
    "window_start_s",
    "window_end_s",
    "rr_bpm",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _strict_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _strict_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_strict_json(item) for item in value]
    if isinstance(value, np.ndarray):
        return _strict_json(value.tolist())
    if isinstance(value, (np.floating, float)):
        result = float(value)
        if not np.isfinite(result):
            raise ValueError("report contains a non-finite float")
        return result
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    return value


def summarize_subset(frame: pd.DataFrame, prediction_column: str) -> dict[str, Any]:
    if frame.empty:
        raise ValueError("metric subset is empty")
    target = frame["rr_bpm"].to_numpy(dtype=float)
    prediction = frame[prediction_column].to_numpy(dtype=float)
    identity = frame["identity"].astype(str).to_numpy()
    return {
        "overall": regression_metrics(target, prediction),
        "identity_macro": identity_macro_metrics(target, prediction, identity),
        "n_identities": int(np.unique(identity).size),
    }


def greedy_nonoverlap_mask(frame: pd.DataFrame) -> np.ndarray:
    """Keep a deterministic maximal set of non-overlapping 32 s windows."""

    keep = np.zeros(len(frame), dtype=bool)
    for _, positions in frame.groupby("session_id", sort=False).indices.items():
        positions = np.asarray(positions, dtype=np.int64)
        order = positions[
            np.lexsort(
                (
                    frame.iloc[positions]["window_number"].to_numpy(),
                    frame.iloc[positions]["window_end_s"].to_numpy(),
                )
            )
        ]
        last_end = -np.inf
        for position in order:
            start = float(frame.iloc[position]["window_start_s"])
            if start >= last_end - 1e-9:
                keep[position] = True
                last_end = float(frame.iloc[position]["window_end_s"])
    return keep


def _load_expected_index_source(path: Path) -> pd.DataFrame:
    """Load an explicitly named, independently locked OOF index source."""

    suffix = path.suffix.lower()
    if suffix == ".csv":
        result = pd.read_csv(path)
    elif suffix == ".npz":
        with np.load(path, allow_pickle=False) as handle:
            index_key = "cache_index" if "cache_index" in handle else "index"
            if index_key not in handle:
                raise KeyError(f"expected-index NPZ is missing index/cache_index: {path}")
            columns: dict[str, np.ndarray] = {
                "cache_index": np.asarray(handle[index_key])
            }
            for source, destination in (
                ("fold", "fold"),
                ("target", "rr_bpm"),
                ("identity", "identity"),
                ("session_id", "session_id"),
                ("window_number", "window_number"),
            ):
                if source in handle:
                    columns[destination] = np.asarray(handle[source])
            result = pd.DataFrame(columns)
    else:
        raise ValueError("expected-index source must be a CSV or NPZ file")
    if "cache_index" not in result:
        if "index" not in result:
            raise KeyError("expected-index source is missing cache_index/index")
        result = result.rename(columns={"index": "cache_index"})
    return result


def load_canonical_expectation(
    cache_dir: Path,
    *,
    fold_assignments_path: Path | None,
    expected_index_source: Path | None,
    expected_rows: int,
    expected_folds: int,
    expected_identities: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build the exact valid-reference population and independently locked folds."""

    cache_dir = cache_dir.resolve()
    manifest_path = cache_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"canonical cache manifest missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    cache = load_feature_cache(cache_dir)
    if "reference_valid" not in cache.metadata:
        raise KeyError("canonical cache metadata is missing reference_valid")
    valid = cache.metadata["reference_valid"].to_numpy(dtype=bool)
    valid_index = np.flatnonzero(valid).astype(np.int64)
    manifest_valid = int(
        sum(
            int(item.get("valid_reference_count", 0))
            for item in manifest.get("sessions", [])
            if item.get("status") == "ok"
        )
    )
    manifest_windows = int(
        sum(
            int(item.get("window_count", 0))
            for item in manifest.get("sessions", [])
            if item.get("status") == "ok"
        )
    )
    if manifest_valid != len(valid_index):
        raise RuntimeError(
            "cache manifest valid-reference count differs from canonical metadata"
        )
    if manifest_windows != len(cache.metadata):
        raise RuntimeError("cache manifest window count differs from canonical metadata")
    if len(valid_index) != expected_rows:
        raise RuntimeError(
            f"canonical valid-reference count is {len(valid_index)}, expected {expected_rows}"
        )

    expected = cache.metadata.iloc[valid_index].copy().reset_index(drop=True)
    expected.insert(0, "cache_index", valid_index)
    canonical_identities = set(expected["identity"].astype(str))
    if len(canonical_identities) != expected_identities:
        raise RuntimeError(
            f"canonical valid-reference identity count is {len(canonical_identities)}, "
            f"expected {expected_identities}"
        )

    fold_source = "none"
    fold_source_sha256: str | None = None
    if fold_assignments_path is not None:
        fold_assignments_path = fold_assignments_path.resolve()
        assignment_document = json.loads(
            fold_assignments_path.read_text(encoding="utf-8")
        )
        identity_to_fold = assignment_document.get(
            "identity_to_fold", assignment_document
        )
        if not isinstance(identity_to_fold, Mapping):
            raise TypeError("fold-assignment document must contain an identity mapping")
        normalized_assignment = {
            str(identity): int(fold) for identity, fold in identity_to_fold.items()
        }
        if set(normalized_assignment) != canonical_identities:
            raise RuntimeError(
                "fold-assignment identities differ from canonical valid-reference identities"
            )
        observed_folds = set(normalized_assignment.values())
        if observed_folds != set(range(expected_folds)):
            raise RuntimeError(
                f"fold-assignment source has folds {sorted(observed_folds)}, "
                f"expected {list(range(expected_folds))}"
            )
        expected["fold"] = expected["identity"].astype(str).map(
            normalized_assignment
        )
        fold_source = str(fold_assignments_path)
        fold_source_sha256 = _sha256(fold_assignments_path)

    explicit_source_audit: dict[str, Any] | None = None
    if expected_index_source is not None:
        expected_index_source = expected_index_source.resolve()
        explicit = _load_expected_index_source(expected_index_source)
        explicit_index = pd.to_numeric(
            explicit["cache_index"], errors="raise"
        ).to_numpy(dtype=float)
        if not np.isfinite(explicit_index).all() or not np.equal(
            explicit_index, np.rint(explicit_index)
        ).all():
            raise ValueError("expected-index source contains non-integral indices")
        explicit_index = explicit_index.astype(np.int64)
        if len(np.unique(explicit_index)) != len(explicit_index):
            raise ValueError("expected-index source contains duplicate indices")
        if not np.array_equal(np.sort(explicit_index), valid_index):
            raise RuntimeError(
                "expected-index source differs from canonical valid-reference indices"
            )
        explicit_by_index = explicit.assign(cache_index=explicit_index).set_index(
            "cache_index"
        )
        canonical_by_index = expected.set_index("cache_index")
        for column in ("fold", "identity", "session_id", "window_number", "rr_bpm"):
            if column not in explicit_by_index:
                continue
            actual = explicit_by_index.loc[valid_index, column].to_numpy()
            if column == "fold" and "fold" not in canonical_by_index:
                expected["fold"] = pd.to_numeric(
                    explicit_by_index.loc[valid_index, "fold"], errors="raise"
                ).to_numpy(dtype=np.int64)
                canonical_by_index = expected.set_index("cache_index")
            wanted = canonical_by_index.loc[valid_index, column].to_numpy()
            if column in {"fold", "window_number"}:
                actual_numeric = np.asarray(actual, dtype=float)
                equal = bool(
                    np.isfinite(actual_numeric).all()
                    and np.equal(actual_numeric, np.rint(actual_numeric)).all()
                    and np.array_equal(
                        actual_numeric.astype(np.int64),
                        np.asarray(wanted, dtype=np.int64),
                    )
                )
            elif column == "rr_bpm":
                equal = np.allclose(
                    actual.astype(float), wanted.astype(float), atol=1e-5, rtol=0.0
                )
            else:
                equal = np.array_equal(actual.astype(str), wanted.astype(str))
            if not equal:
                raise RuntimeError(
                    f"expected-index source {column} differs from canonical expectation"
                )
        explicit_source_audit = {
            "path": str(expected_index_source),
            "sha256": _sha256(expected_index_source),
            "rows": int(len(explicit)),
            "exact_canonical_index_match": True,
        }

    if "fold" not in expected:
        raise RuntimeError(
            "no expected fold assignment is available; provide --fold-assignments "
            "or an expected-index source containing fold"
        )
    expected["fold"] = pd.to_numeric(expected["fold"], errors="raise").astype(int)
    if set(expected["fold"].unique()) != set(range(expected_folds)):
        raise RuntimeError("canonical expected rows do not cover every declared fold")

    audit = {
        "cache_dir": str(cache_dir),
        "cache_manifest": str(manifest_path),
        "cache_manifest_sha256": _sha256(manifest_path),
        "manifest_window_count": manifest_windows,
        "manifest_valid_reference_count": manifest_valid,
        "canonical_valid_reference_count": int(len(expected)),
        "canonical_identity_count": int(expected["identity"].nunique()),
        "expected_fold_count": expected_folds,
        "fold_assignments": fold_source,
        "fold_assignments_sha256": fold_source_sha256,
        "expected_index_source": explicit_source_audit,
    }
    return expected, audit


def validate_locked_oof(
    frame: pd.DataFrame,
    prediction_column: str,
    *,
    expected: pd.DataFrame | None = None,
    expected_rows: int | None = None,
    expected_folds: int | None = None,
    expected_identities: int | None = None,
) -> dict[str, Any]:
    missing = sorted(REQUIRED_COLUMNS - set(frame.columns))
    if missing:
        raise KeyError(f"candidate CSV is missing columns {missing}")
    if prediction_column not in frame:
        raise KeyError(f"candidate CSV is missing {prediction_column}")
    numeric = frame[["rr_bpm", prediction_column]].to_numpy(dtype=float)
    if not np.isfinite(numeric).all():
        raise ValueError("target/prediction contains non-finite values")
    raw_index = pd.to_numeric(frame["cache_index"], errors="raise").to_numpy(
        dtype=float
    )
    if not np.isfinite(raw_index).all() or not np.equal(
        raw_index, np.rint(raw_index)
    ).all():
        raise ValueError("candidate OOF cache indices must be finite integers")
    candidate_index = raw_index.astype(np.int64)
    if len(np.unique(candidate_index)) != len(candidate_index):
        raise ValueError("candidate OOF contains duplicate cache indices")
    raw_folds = pd.to_numeric(frame["fold"], errors="raise").to_numpy(dtype=float)
    if not np.isfinite(raw_folds).all() or not np.equal(
        raw_folds, np.rint(raw_folds)
    ).all():
        raise ValueError("candidate OOF folds must be finite integers")
    candidate_folds = raw_folds.astype(np.int64)
    identity_fold_counts = frame.groupby("identity")["fold"].nunique()
    if not (identity_fold_counts == 1).all():
        raise ValueError("an identity occurs in more than one outer-test fold")

    exact_index_match: bool | None = None
    exact_fold_assignment_match: bool | None = None
    canonical_metadata_match: bool | None = None
    if expected is not None:
        expected_index = expected["cache_index"].to_numpy(dtype=np.int64)
        if len(candidate_index) != len(expected_index) or not np.array_equal(
            np.sort(candidate_index), np.sort(expected_index)
        ):
            missing_index = np.setdiff1d(expected_index, candidate_index)
            extra_index = np.setdiff1d(candidate_index, expected_index)
            raise RuntimeError(
                "candidate is not the complete valid-reference OOF: "
                f"rows={len(candidate_index)}/{len(expected_index)}, "
                f"missing={len(missing_index)}, extra={len(extra_index)}"
            )
        exact_index_match = True
        candidate_by_index = frame.assign(
            cache_index=candidate_index, fold=candidate_folds
        ).set_index("cache_index")
        expected_by_index = expected.set_index("cache_index")
        aligned_candidate = candidate_by_index.loc[expected_index]
        aligned_expected = expected_by_index.loc[expected_index]
        if not np.array_equal(
            aligned_candidate["fold"].to_numpy(dtype=np.int64),
            aligned_expected["fold"].to_numpy(dtype=np.int64),
        ):
            raise RuntimeError("candidate outer-fold assignment differs from locked source")
        exact_fold_assignment_match = True
        for column in (
            "identity",
            "session_id",
            "window_number",
            "window_start_s",
            "window_end_s",
            "rr_bpm",
        ):
            actual = aligned_candidate[column].to_numpy()
            wanted = aligned_expected[column].to_numpy()
            if column in {
                "window_number",
                "window_start_s",
                "window_end_s",
                "rr_bpm",
            }:
                equal = np.allclose(
                    actual.astype(float), wanted.astype(float), atol=1e-5, rtol=0.0
                )
            else:
                equal = np.array_equal(actual.astype(str), wanted.astype(str))
            if not equal:
                raise RuntimeError(
                    f"candidate {column} differs from canonical cache metadata"
                )
        canonical_metadata_match = True

    if expected_rows is not None and len(frame) != expected_rows:
        raise RuntimeError(
            f"candidate row count is {len(frame)}, expected exactly {expected_rows}"
        )
    if expected_identities is not None and frame["identity"].nunique() != expected_identities:
        raise RuntimeError(
            f"candidate identity count is {frame['identity'].nunique()}, "
            f"expected exactly {expected_identities}"
        )
    observed_folds = set(candidate_folds.tolist())
    if expected_folds is not None and observed_folds != set(range(expected_folds)):
        raise RuntimeError(
            f"candidate folds are {sorted(observed_folds)}, "
            f"expected {list(range(expected_folds))}"
        )
    return {
        "rows": int(len(frame)),
        "identities": int(frame["identity"].nunique()),
        "folds": int(len(observed_folds)),
        "unique_cache_indices": True,
        "one_test_fold_per_identity": True,
        "finite_target_and_prediction": True,
        "exact_valid_reference_index_match": exact_index_match,
        "exact_locked_fold_assignment_match": exact_fold_assignment_match,
        "canonical_metadata_match": canonical_metadata_match,
        "candidate_complete": bool(
            expected is not None
            and exact_index_match
            and exact_fold_assignment_match
            and canonical_metadata_match
        ),
    }


def evaluate_goal_checks(
    full: Mapping[str, Any], high_rr: Mapping[str, Any]
) -> dict[str, bool]:
    overall = full["overall"]
    identity_macro = full["identity_macro"]
    return {
        "overall_mae": overall["mae"] <= GOAL_TARGETS["overall_mae_max_bpm"],
        "identity_macro_mae": identity_macro["macro_mae"]
        <= GOAL_TARGETS["identity_macro_mae_max_bpm"],
        "overall_rmse": overall["rmse"] <= GOAL_TARGETS["overall_rmse_max_bpm"],
        "within_2": overall["within_2"]
        >= GOAL_TARGETS["within_2_min_fraction"],
        "over_5": overall["catastrophic_over_5"]
        <= GOAL_TARGETS["over_5_max_fraction"],
        "high_rr_25_35_mae": high_rr["overall"]["mae"]
        <= GOAL_TARGETS["high_rr_25_35_mae_max_bpm"],
    }


def _e2e_summary(path: Path) -> dict[str, Any]:
    report = json.loads(path.read_text(encoding="utf-8"))
    input_record = report.get("input", {})
    top_level_warnings = [str(value) for value in report.get("warnings", [])]
    input_warnings = [str(value) for value in input_record.get("warnings", [])]
    # Preserve origin and also expose one de-duplicated list for acceptance.
    combined_warnings = list(dict.fromkeys(top_level_warnings + input_warnings))
    measurement_contract = report.get("measurement_contract", {})
    config_provenance = report.get("pipeline", {}).get(
        "checkpoint_config_provenance", {}
    )
    devices: dict[str, Any] = {}
    for device, record in report["devices"].items():
        paths = record["paths"]
        resident = paths["raw_window_in_memory"]["stages"]["total_ms"]
        value = {
            "device_name": record["device_name"],
            "raw_window_in_memory_p50_ms": resident["p50_ms"],
            "raw_window_in_memory_p95_ms": resident["p95_ms"],
        }
        if "warm_memmap_read_included" in paths:
            loaded = paths["warm_memmap_read_included"]["stages"]["total_ms"]
            value["warm_memmap_included_p50_ms"] = loaded["p50_ms"]
            value["warm_memmap_included_p95_ms"] = loaded["p95_ms"]
        devices[device] = value
    return {
        "path": str(path),
        "checkpoint": report["checkpoint"]["path"],
        "parameters": report["checkpoint"]["trainable_parameters"],
        "input_kind": input_record["kind"],
        "devices": devices,
        "warnings": combined_warnings,
        "top_level_warnings": top_level_warnings,
        "input_warnings": input_warnings,
        "production_feature_bit_exact": measurement_contract.get(
            "production_feature_bit_exact"
        ),
        "checkpoint_config_provenance": {
            "verified": config_provenance.get("verified") is True,
            "status": config_provenance.get("status", "missing"),
            "training_sha256": config_provenance.get("training_sha256"),
            "supplied_sha256": config_provenance.get("supplied_sha256"),
        },
    }


def _robustness_evidence(
    path: Path | None,
    *,
    expected_rows: int,
    expected_identities: int,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    if path is None:
        return None, {
            "passed": False,
            "status": "missing",
            "reason": "no radar-robustness report was supplied",
        }
    document = json.loads(path.read_text(encoding="utf-8"))
    conditions = document.get("radar_conditions")
    if not isinstance(conditions, Mapping):
        return None, {
            "passed": False,
            "status": "incomplete",
            "reason": "radar_conditions is missing",
            "path": str(path),
        }
    missing = sorted(REQUIRED_RADAR_CONDITIONS - set(conditions))
    incomplete: list[str] = []
    for name in sorted(REQUIRED_RADAR_CONDITIONS & set(conditions)):
        condition = conditions[name]
        try:
            rows = int(round(float(condition["overall"]["n"])))
            identities = int(condition["n_identities"])
        except (KeyError, TypeError, ValueError):
            incomplete.append(name)
            continue
        if rows != expected_rows or identities != expected_identities:
            incomplete.append(name)
    provenance = document.get("provenance_audit")
    provenance_passed = bool(
        isinstance(provenance, Mapping) and provenance.get("passed") is True
    )
    passed = not missing and not incomplete and provenance_passed
    return dict(conditions), {
        "passed": passed,
        "status": "complete" if passed else "incomplete",
        "path": str(path.resolve()),
        "sha256": _sha256(path),
        "required_conditions": sorted(REQUIRED_RADAR_CONDITIONS),
        "missing_conditions": missing,
        "incomplete_conditions": incomplete,
        "provenance_audit_passed": provenance_passed,
        "reason": (
            "all seven radar masks cover the exact audited OOF population and "
            "the robustness provenance audit passed"
            if passed
            else "robustness evidence is missing conditions, complete rows, or provenance"
        ),
    }


def _nonoverlap_evidence(
    nonoverlap: Mapping[str, Any], *, expected_identities: int
) -> dict[str, Any]:
    rows = int(round(float(nonoverlap["overall"]["n"])))
    identities = int(nonoverlap["n_identities"])
    passed = rows > 0 and identities == expected_identities
    return {
        "passed": passed,
        "status": "complete" if passed else "incomplete",
        "method": "deterministic greedy non-overlapping 32 s intervals per session",
        "rows": rows,
        "identities": identities,
        "expected_identities": expected_identities,
    }


def _calibration_evidence(status: str) -> dict[str, Any]:
    if status not in CALIBRATION_STATUSES:
        raise ValueError(f"unknown calibration status: {status}")
    passed = status in CALIBRATION_PASS_STATUSES
    return {
        "passed": passed,
        "status": status,
        "accepted_statuses": sorted(CALIBRATION_PASS_STATUSES),
        "reason": (
            "interval uncertainty was calibrated with a validation-locked procedure"
            if passed
            else "candidate uncertainty is not a calibrated RR interval"
        ),
    }


def _e2e_evidence(
    components: Sequence[Mapping[str, Any]], *, stride_budget_ms: float
) -> tuple[dict[str, Any], dict[str, Any]]:
    conservative: dict[str, Any] = {}
    if components:
        common_devices = set.intersection(
            *(set(component["devices"]) for component in components)
        )
    else:
        common_devices = set()
    for device in sorted(common_devices):
        conservative[device] = {
            "sum_of_component_e2e_p50_ms": float(
                sum(
                    component["devices"][device]["raw_window_in_memory_p50_ms"]
                    for component in components
                )
            ),
            "sum_of_component_e2e_p95_ms": float(
                sum(
                    component["devices"][device]["raw_window_in_memory_p95_ms"]
                    for component in components
                )
            ),
            "interpretation": (
                "conservative arithmetic sum of separately measured component "
                "quantiles; duplicates shared preprocessing and is not a directly "
                "measured ensemble quantile"
            ),
        }
    raw_input_complete = bool(components) and all(
        component.get("input_kind") in {"raw_window", "raw_file_window"}
        and component.get("devices")
        for component in components
    )
    warnings = [
        str(warning)
        for component in components
        for warning in component.get("warnings", [])
    ]
    warning_free = bool(components) and not warnings
    provenance_verified = bool(components) and all(
        component.get("checkpoint_config_provenance", {}).get("verified") is True
        for component in components
    )
    production_feature_bit_exact = bool(components) and all(
        component.get("production_feature_bit_exact") is True
        for component in components
    )
    window_boundary_warning = any(
        "window boundary" in warning.lower()
        or "pre-window repair state" in warning.lower()
        for warning in warnings
    )
    within_stride = bool(conservative) and all(
        float(record["sum_of_component_e2e_p95_ms"]) <= stride_budget_ms
        for record in conservative.values()
    )
    timing_complete = raw_input_complete and bool(conservative) and within_stride
    passed = bool(
        timing_complete
        and provenance_verified
        and production_feature_bit_exact
        and warning_free
    )
    if (
        timing_complete
        and provenance_verified
        and (not production_feature_bit_exact or window_boundary_warning)
    ):
        status = "timing_complete_not_feature_bit_exact"
    elif timing_complete and not provenance_verified:
        status = "timing_complete_provenance_unverified"
    elif raw_input_complete and bool(conservative) and not within_stride:
        status = "timing_complete_over_stride"
    elif passed:
        status = "complete_within_stride"
    else:
        status = "incomplete"
    evidence = {
        "passed": passed,
        "status": status,
        "component_count": len(components),
        "common_devices": sorted(common_devices),
        "raw_window_input_complete": raw_input_complete,
        "timing_complete": timing_complete,
        "warning_free": warning_free,
        "warnings": warnings,
        "checkpoint_config_provenance_verified": provenance_verified,
        "production_feature_bit_exact": production_feature_bit_exact,
        "window_boundary_warning": window_boundary_warning,
        "all_common_devices_within_stride": within_stride,
        "stride_budget_ms": stride_budget_ms,
    }
    benchmark = {
        "components": list(components),
        "conservative_sequential_estimate": conservative,
        "stride_budget_ms": stride_budget_ms,
    }
    return benchmark, evidence


def _render_markdown(report: Mapping[str, Any]) -> str:
    full = report["metrics"]["full"]
    high = report["metrics"]["high_rr_25_35"]
    checks = report["goal"]["checks"]
    rows = [
        (
            "Overall MAE",
            full["overall"]["mae"],
            f"≤ {GOAL_TARGETS['overall_mae_max_bpm']:.1f} bpm",
            checks["overall_mae"],
        ),
        (
            "Identity-macro MAE",
            full["identity_macro"]["macro_mae"],
            f"≤ {GOAL_TARGETS['identity_macro_mae_max_bpm']:.1f} bpm",
            checks["identity_macro_mae"],
        ),
        (
            "RMSE",
            full["overall"]["rmse"],
            f"≤ {GOAL_TARGETS['overall_rmse_max_bpm']:.1f} bpm",
            checks["overall_rmse"],
        ),
        (
            "Within ±2 bpm",
            100.0 * full["overall"]["within_2"],
            f"≥ {100 * GOAL_TARGETS['within_2_min_fraction']:.0f}%",
            checks["within_2"],
        ),
        (
            "Error >5 bpm",
            100.0 * full["overall"]["catastrophic_over_5"],
            f"≤ {100 * GOAL_TARGETS['over_5_max_fraction']:.0f}%",
            checks["over_5"],
        ),
        (
            "25–35 bpm MAE",
            high["overall"]["mae"],
            f"≤ {GOAL_TARGETS['high_rr_25_35_mae_max_bpm']:.1f} bpm",
            checks["high_rr_25_35_mae"],
        ),
    ]
    lines = [
        "# Commercial-goal audit",
        "",
        f"Candidate: `{report['candidate']['name']}`",
        "",
        "This is a retrospective engineering audit, not a commercial or medical claim.",
        "",
        "| Metric | Achieved | Target | Result |",
        "|---|---:|---:|:---:|",
    ]
    for name, achieved, target, passed in rows:
        unit = "%" if "%" in target else " bpm"
        lines.append(
            f"| {name} | {achieved:.3f}{unit} | {target} | "
            f"{'PASS' if passed else 'FAIL'} |"
        )
    nonoverlap = report["metrics"]["greedy_nonoverlap_32s"]["overall"]
    evidence = report["goal"]["required_evidence_checks"]
    lines.extend(
        [
            "",
            "## Required evidence gates",
            "",
            "| Evidence | Result | Status |",
            "|---|:---:|---|",
            (
                "| Complete radar-mask robustness | "
                f"{'PASS' if evidence['robustness_present_and_complete'] else 'FAIL'} | "
                f"{report['evidence']['robustness']['status']} |"
            ),
            (
                "| Non-overlap evaluation | "
                f"{'PASS' if evidence['nonoverlap_complete'] else 'FAIL'} | "
                f"{report['evidence']['nonoverlap']['status']} |"
            ),
            (
                "| Validation-locked interval calibration | "
                f"{'PASS' if evidence['calibration_validated'] else 'FAIL'} | "
                f"{report['evidence']['calibration']['status']} |"
            ),
            (
                "| Deployment-faithful E2E p95 within stride | "
                f"{'PASS' if evidence['e2e_within_stride'] else 'FAIL'} | "
                f"{report['evidence']['end_to_end']['status']} |"
            ),
            "",
            "## Dependence-aware view",
            "",
            f"The greedy non-overlap subset has n={int(nonoverlap['n'])}, "
            f"MAE={nonoverlap['mae']:.3f} bpm and RMSE={nonoverlap['rmse']:.3f} bpm.",
            "",
            "## Conclusion",
            "",
            (
                "All retrospective accuracy and required-evidence gates passed. "
                "External prospective validation is still required."
                if report["goal"]["retrospective_acceptance_pass"]
                else "The declared full-coverage commercial goal was not met."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def run(args: argparse.Namespace) -> dict[str, Any]:
    frame = pd.read_csv(args.candidate_csv)
    expected, expectation_audit = load_canonical_expectation(
        args.cache_dir,
        fold_assignments_path=args.fold_assignments,
        expected_index_source=args.expected_index_source,
        expected_rows=args.expected_valid_rows,
        expected_folds=args.expected_folds,
        expected_identities=args.expected_identities,
    )
    audit = validate_locked_oof(
        frame,
        args.prediction_column,
        expected=expected,
        expected_rows=args.expected_valid_rows,
        expected_folds=args.expected_folds,
        expected_identities=args.expected_identities,
    )
    audit["expectation"] = expectation_audit
    full = summarize_subset(frame, args.prediction_column)
    high_rows = (frame["rr_bpm"] >= 25.0) & (frame["rr_bpm"] < 35.0)
    high = summarize_subset(frame.loc[high_rows], args.prediction_column)
    mid_high_rows = (frame["rr_bpm"] >= 20.0) & (frame["rr_bpm"] < 35.0)
    mid_high = summarize_subset(frame.loc[mid_high_rows], args.prediction_column)
    nonoverlap = summarize_subset(
        frame.loc[greedy_nonoverlap_mask(frame)], args.prediction_column
    )
    phases = {
        str(phase): summarize_subset(
            frame.loc[(frame["window_number"].to_numpy(dtype=int) % 8) == phase],
            args.prediction_column,
        )
        for phase in range(8)
    }
    checks = evaluate_goal_checks(full, high)
    nonoverlap_evidence = _nonoverlap_evidence(
        nonoverlap, expected_identities=args.expected_identities
    )
    radar_conditions, robustness_evidence = _robustness_evidence(
        args.robustness_report,
        expected_rows=args.expected_valid_rows,
        expected_identities=args.expected_identities,
    )
    calibration_evidence = _calibration_evidence(args.calibration_status)

    uncertainty_report: dict[str, Any] | None = None
    if args.uncertainty_column:
        if args.uncertainty_column not in frame:
            raise KeyError(f"candidate CSV is missing {args.uncertainty_column}")
        uncertainty = frame[args.uncertainty_column].to_numpy(dtype=float)
        if not np.isfinite(uncertainty).all():
            raise ValueError("uncertainty contains non-finite values")
        uncertainty_report = {
            "role": "ranking score only; not a calibrated RR interval sigma",
            "risk_coverage": risk_coverage_curve(
                frame["rr_bpm"].to_numpy(dtype=float),
                frame[args.prediction_column].to_numpy(dtype=float),
                uncertainty,
                identities=frame["identity"].astype(str).to_numpy(),
            ),
        }

    components = [_e2e_summary(path.resolve()) for path in args.e2e_reports]
    end_to_end, e2e_evidence = _e2e_evidence(
        components, stride_budget_ms=args.stride_budget_ms
    )
    evidence_checks = {
        "robustness_present_and_complete": bool(robustness_evidence["passed"]),
        "nonoverlap_complete": bool(nonoverlap_evidence["passed"]),
        "calibration_validated": bool(calibration_evidence["passed"]),
        "e2e_within_stride": bool(e2e_evidence["passed"]),
    }

    report: dict[str, Any] = {
        "schema_version": 2,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "retrospective_engineering_audit_not_commercial_validation",
        "candidate": {
            "name": args.candidate_name,
            "csv": str(args.candidate_csv.resolve()),
            "csv_sha256": _sha256(args.candidate_csv.resolve()),
            "prediction_column": args.prediction_column,
            "uncertainty_column": args.uncertainty_column,
        },
        "provenance_audit": audit,
        "metrics": {
            "full": full,
            "high_rr_25_35": high,
            "rr_20_35": mid_high,
            "greedy_nonoverlap_32s": nonoverlap,
            "eight_fixed_nonoverlap_phases": phases,
        },
        "goal": {
            "targets": GOAL_TARGETS,
            # Retain the original names for downstream report compatibility.
            "checks": checks,
            "all_point_checks_pass": bool(all(checks.values())),
            "accuracy_checks": checks,
            "all_accuracy_checks_pass": bool(all(checks.values())),
            "required_evidence_checks": evidence_checks,
            "all_required_evidence_checks_pass": bool(
                all(evidence_checks.values())
            ),
            "retrospective_acceptance_pass": bool(
                all(checks.values()) and all(evidence_checks.values())
            ),
            "commercial_validation_pass": False,
            "commercial_validation_status": (
                "not_assessed_external_prospective_evidence_required"
            ),
        },
        "uncertainty": uncertainty_report,
        "evidence": {
            "robustness": robustness_evidence,
            "nonoverlap": nonoverlap_evidence,
            "calibration": calibration_evidence,
            "end_to_end": e2e_evidence,
        },
        "required_external_evidence": [
            "prospectively locked unseen cohort",
            "independent reference adjudication and demographic/placement coverage",
            "prospectively locked abstention threshold and achieved coverage",
            "packet-loss/corruption/placement-shift tests beyond ideal radar masking",
        ],
    }
    if radar_conditions is not None:
        report["radar_robustness"] = radar_conditions
    report["end_to_end_benchmark"] = end_to_end

    strict = _strict_json(report)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(strict, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    args.output_markdown.parent.mkdir(parents=True, exist_ok=True)
    args.output_markdown.write_text(_render_markdown(strict), encoding="utf-8")
    return strict


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidate-csv",
        type=Path,
        default=Path("artifacts/runs/ensemble_structured_exact/ensemble_oof.csv"),
    )
    parser.add_argument(
        "--candidate-name", default="validation-locked structured two-SNN ensemble"
    )
    parser.add_argument(
        "--prediction-column", default="prediction_uncalibrated_bpm"
    )
    parser.add_argument(
        "--uncertainty-column", default="uncertainty_uncalibrated"
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("artifacts/cache/rf32s"),
        help="canonical feature cache whose manifest defines valid-reference rows",
    )
    parser.add_argument(
        "--fold-assignments",
        type=Path,
        default=Path(
            "artifacts/runs/final_structured_aux_s12/fold_assignments.json"
        ),
        help="independently locked identity-to-outer-fold assignment JSON",
    )
    parser.add_argument(
        "--expected-index-source",
        type=Path,
        help="optional independent CSV/NPZ index source cross-checked with the cache",
    )
    parser.add_argument(
        "--expected-valid-rows", type=int, default=EXPECTED_VALID_REFERENCE_ROWS
    )
    parser.add_argument("--expected-folds", type=int, default=EXPECTED_OUTER_FOLDS)
    parser.add_argument(
        "--expected-identities", type=int, default=EXPECTED_IDENTITIES
    )
    parser.add_argument(
        "--robustness-report",
        type=Path,
        default=Path("artifacts/robustness/ensemble_structured_exact/report.json"),
    )
    parser.add_argument(
        "--e2e-reports",
        type=Path,
        nargs="*",
        default=[
            Path("artifacts/benchmarks/commercial/structured_aux_fold0_e2e.json"),
            Path("artifacts/benchmarks/commercial/structured_exact_fold0_e2e.json"),
        ],
    )
    parser.add_argument(
        "--stride-budget-ms",
        type=float,
        default=4000.0,
        help="maximum allowed conservative end-to-end p95 latency",
    )
    parser.add_argument(
        "--calibration-status",
        choices=sorted(CALIBRATION_STATUSES),
        default="ranking_only_not_interval_calibrated",
        help=(
            "explicit interval-calibration evidence status; ranking-only scores "
            "do not pass the calibration evidence gate"
        ),
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("artifacts/commercial_goal_report.json"),
    )
    parser.add_argument(
        "--output-markdown",
        type=Path,
        default=Path("artifacts/COMMERCIAL_GOAL_AUDIT.md"),
    )
    args = parser.parse_args(argv)
    if not args.candidate_csv.is_file():
        parser.error(f"candidate CSV not found: {args.candidate_csv}")
    if not (args.cache_dir / "manifest.json").is_file():
        parser.error(f"canonical cache manifest not found: {args.cache_dir}")
    if args.fold_assignments and not args.fold_assignments.is_file():
        parser.error(f"fold assignments not found: {args.fold_assignments}")
    if args.expected_index_source and not args.expected_index_source.is_file():
        parser.error(f"expected-index source not found: {args.expected_index_source}")
    if args.robustness_report and not args.robustness_report.is_file():
        parser.error(f"robustness report not found: {args.robustness_report}")
    missing_e2e = [str(path) for path in args.e2e_reports if not path.is_file()]
    if missing_e2e:
        parser.error(f"end-to-end reports not found: {missing_e2e}")
    if args.expected_valid_rows < 1 or args.expected_folds < 1 or args.expected_identities < 1:
        parser.error("expected population counts must be positive")
    if not np.isfinite(args.stride_budget_ms) or args.stride_budget_ms <= 0:
        parser.error("stride budget must be a positive finite value")
    return args


if __name__ == "__main__":
    result = run(parse_args())
    print(
        json.dumps(
            {
                "candidate": result["candidate"]["name"],
                "all_point_checks_pass": result["goal"][
                    "all_point_checks_pass"
                ],
                "all_required_evidence_checks_pass": result["goal"][
                    "all_required_evidence_checks_pass"
                ],
                "retrospective_acceptance_pass": result["goal"][
                    "retrospective_acceptance_pass"
                ],
            },
            ensure_ascii=False,
        )
    )
