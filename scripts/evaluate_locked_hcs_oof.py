#!/usr/bin/env python3
"""Primary, fail-closed evaluation of the single locked HCS OOF join.

This module is an evaluator, never a selector.  It validates the post-lock
evaluation authorization and the complete prediction/target provenance graph
before opening any target-bearing array.  It then evaluates each of the three
pre-specified seeds independently.  Seeds are never pooled, averaged, ranked,
or suppressed.

The outputs are create-once retrospective evidence.  They cannot authorize a
commercial claim and explicitly require confirmation in an independent
prospective cohort.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import secrets
import sys
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


SCHEMA_VERSION = 1
FOLDS = tuple(range(6))
FIXED_SEEDS = (20260828, 20260829, 20260830)
EXPECTED_SEED_COUNT = 3
EXPECTED_VALID_REFERENCE_ROWS = 2327
EXPECTED_IDENTITIES = 18
BOOTSTRAP_SAMPLES = 5000
BOOTSTRAP_SEED = 20260828
BOOTSTRAP_CONFIDENCE = 0.95

GOAL_TARGETS = {
    "overall_mae_max_bpm": 1.0,
    "identity_macro_mae_max_bpm": 1.0,
    "overall_rmse_max_bpm": 1.8,
    "within_2_min_fraction": 0.90,
    "over_5_max_fraction": 0.03,
    "high_rr_25_35_mae_max_bpm": 2.0,
}

CANDIDATES = {
    "fallback": "fallback_rr_bpm",
    "source": "source_rr_bpm",
    "locked_final": "final_rr_bpm",
}
PAIRED_COMPARISONS = {
    "source_minus_fallback": ("source", "fallback"),
    "locked_final_minus_fallback": ("locked_final", "fallback"),
    "locked_final_minus_source": ("locked_final", "source"),
}
RR_STRATA_SPEC = (
    {"name": "lt_12", "lower": None, "lower_inclusive": False, "upper": 12.0, "upper_inclusive": False},
    {"name": "12_to_lt_18", "lower": 12.0, "lower_inclusive": True, "upper": 18.0, "upper_inclusive": False},
    {"name": "18_to_lt_25", "lower": 18.0, "lower_inclusive": True, "upper": 25.0, "upper_inclusive": False},
    {"name": "25_to_35_inclusive", "lower": 25.0, "lower_inclusive": True, "upper": 35.0, "upper_inclusive": True},
    {"name": "gt_35", "lower": 35.0, "lower_inclusive": False, "upper": None, "upper_inclusive": False},
)
QC_NUMERIC_STRATA_SPEC = {
    "reference_quality": (0.5, 0.8),
    "reference_sigma_bpm": (1.0, 2.0),
    "spectral_concentration": (0.25, 0.5),
    "periodicity": (0.25, 0.5),
    "estimator_disagreement_bpm": (1.0, 3.0),
    "phase_residual_rad": (0.5, 1.25),
    "clip_fraction": (0.01, 0.05),
    "guard_clip_fraction": (0.01, 0.05),
    "plateau_fraction": (0.01, 0.05),
    "classical_confidence": (0.5, 0.8),
    "radar_peak_spread_bpm": (1.0, 3.0),
    "breath_count": (6.0, 12.0),
}
REQUIRED_TARGET_FIELDS = {
    "cache_index",
    "outer_fold",
    "target_rr_bpm",
    "identity",
    "reference_valid",
    "session_id",
    "window_number",
    "protocol",
    "window_start_s",
    "window_end_s",
}
REQUIRED_JOINED_FIELDS = {
    "cache_index",
    "outer_fold",
    "seed",
    "target_rr_bpm",
    "identity",
    *CANDIDATES.values(),
}
UNCERTAINTY_FIELDS = (
    "uncertainty_uncalibrated",
    "uncertainty_bpm",
    "fallback_std_bpm",
    "source_scale_bpm",
)
METRIC_FIELDS = (
    "mae",
    "identity_macro_mae",
    "rmse",
    "within_2_fraction",
    "over_5_fraction",
    "tail_25_35_mae",
)
CSV_COLUMNS = (
    "record_type",
    "seed",
    "candidate",
    "comparison",
    "scope",
    "stratum_type",
    "stratum",
    "rows",
    "identities",
    "tail_25_35_rows",
    *METRIC_FIELDS,
    "improved_fraction",
    "tied_fraction",
    "worsened_fraction",
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LOCKED_ROOT = (
    PROJECT_ROOT / "artifacts/runs/harmonic_candidate_set_snn_v2/hcs_locked_oof"
)
DEFAULT_EVALUATION_SPEC = (
    PROJECT_ROOT
    / "artifacts/campaigns/harmonic_candidate_set_snn_v2/locked_primary_evaluation_spec.json"
)


class LockedPrimaryEvaluationError(RuntimeError):
    """An authorization, provenance, topology, or publication check failed."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def array_sha256(value: Any) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii") + b"\0")
    digest.update(json.dumps(array.shape, separators=(",", ":")).encode("ascii"))
    digest.update(b"\0")
    digest.update(array.tobytes())
    return digest.hexdigest()


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def _read_json(
    path: Path,
    label: str,
    *,
    require_content_hash: bool = False,
) -> dict[str, Any]:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise LockedPrimaryEvaluationError(f"{label} repeats JSON key {key!r}")
            result[key] = value
        return result

    def reject_nonfinite(value: str) -> None:
        raise LockedPrimaryEvaluationError(f"{label} contains non-finite JSON number {value}")

    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=reject_duplicates,
            parse_constant=reject_nonfinite,
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise LockedPrimaryEvaluationError(f"invalid {label}: {path} ({exc})") from exc
    if not isinstance(value, dict):
        raise LockedPrimaryEvaluationError(f"{label} root must be an object: {path}")
    if require_content_hash:
        payload = dict(value)
        expected = str(payload.pop("content_sha256", "")).lower()
        if not _is_sha256(expected) or canonical_json_sha256(payload) != expected:
            raise LockedPrimaryEvaluationError(f"{label} content_sha256 mismatch")
    return value


def evaluation_spec_document(
    *,
    expected_rows: int = EXPECTED_VALID_REFERENCE_ROWS,
    expected_identities: int = EXPECTED_IDENTITIES,
    bootstrap_samples: int = BOOTSTRAP_SAMPLES,
    bootstrap_seed: int = BOOTSTRAP_SEED,
    bootstrap_confidence: float = BOOTSTRAP_CONFIDENCE,
) -> dict[str, Any]:
    """Return the exact target-independent primary evaluation specification."""

    integer_values = (expected_rows, expected_identities, bootstrap_samples, bootstrap_seed)
    if any(isinstance(value, bool) or not isinstance(value, int) for value in integer_values):
        raise LockedPrimaryEvaluationError("evaluation-spec integer fields must be integers")
    if expected_rows < 1 or expected_identities < 2 or bootstrap_samples < 1 or bootstrap_seed < 0:
        raise LockedPrimaryEvaluationError("evaluation-spec integer fields are out of range")
    if not isinstance(bootstrap_confidence, (int, float)) or isinstance(
        bootstrap_confidence, bool
    ) or not 0.0 < float(bootstrap_confidence) < 1.0:
        raise LockedPrimaryEvaluationError("evaluation-spec confidence must lie in (0,1)")
    return {
        "schema_version": SCHEMA_VERSION,
        "classification": "locked_hcs_oof_primary_evaluation_specification",
        "must_be_frozen_before_outer_test_inference": True,
        "target_values_or_target_bearing_artifacts_used_to_build_spec": False,
        "commercial_claim_authorized": False,
        "commercial_performance_proven": False,
        "prospective_confirmation_required": True,
        "independent_prospective_cohort_evaluated": False,
        "population": {
            "valid_reference_rows_per_seed": expected_rows,
            "physical_identity_count": expected_identities,
            "outer_folds": list(FOLDS),
            "fixed_seeds": list(FIXED_SEEDS),
            "seed_count": EXPECTED_SEED_COUNT,
            "each_seed_must_be_evaluated_independently": True,
            "cross_seed_pooling_ranking_or_suppression_allowed": False,
        },
        "prediction_candidates": dict(CANDIDATES),
        "primary_candidate": "locked_final",
        "point_gates": {
            "overall_mae": {
                "metric": "mae",
                "operator": "<=",
                "threshold": GOAL_TARGETS["overall_mae_max_bpm"],
                "unit": "bpm",
            },
            "identity_macro_mae": {
                "metric": "identity_macro_mae",
                "operator": "<=",
                "threshold": GOAL_TARGETS["identity_macro_mae_max_bpm"],
                "unit": "bpm",
            },
            "overall_rmse": {
                "metric": "rmse",
                "operator": "<=",
                "threshold": GOAL_TARGETS["overall_rmse_max_bpm"],
                "unit": "bpm",
            },
            "within_2_fraction": {
                "metric": "within_2_fraction",
                "operator": ">=",
                "threshold": GOAL_TARGETS["within_2_min_fraction"],
                "unit": "fraction",
            },
            "over_5_fraction": {
                "metric": "over_5_fraction",
                "operator": "<=",
                "threshold": GOAL_TARGETS["over_5_max_fraction"],
                "unit": "fraction",
            },
            "tail_25_35_mae": {
                "metric": "tail_25_35_mae",
                "operator": "<=",
                "threshold": GOAL_TARGETS["high_rr_25_35_mae_max_bpm"],
                "unit": "bpm",
                "tail_lower_bpm_inclusive": 25.0,
                "tail_upper_bpm_inclusive": 35.0,
            },
        },
        "nonoverlap": {
            "greedy": {
                "algorithm_id": "per_session_earliest_end_greedy_v1",
                "group_key": "session_id",
                "stable_order_keys": ["window_end_s", "window_number", "cache_index"],
                "accept_rule": "window_start_s >= previous_kept_window_end_s - 1e-9",
                "interval_tolerance_s": 1.0e-9,
                "uses_target_or_prediction_values": False,
            },
            "fixed_phases": {
                "definition": "window_number modulo 8",
                "modulus": 8,
                "phases": list(range(8)),
                "phase_search_or_suppression_allowed": False,
            },
        },
        "strata": {
            "categorical": {
                "identity": "identity",
                "fold": "outer_fold",
                "session": "session_id",
                "protocol": "protocol",
            },
            "rr_bands": [dict(value) for value in RR_STRATA_SPEC],
            "qc_numeric_three_band_thresholds": {
                name: {"lower_cut": cuts[0], "upper_cut": cuts[1]}
                for name, cuts in QC_NUMERIC_STRATA_SPEC.items()
            },
            "qc_boolean": {"radar_observable": [False, True]},
            "missing_qc_policy": "report_unavailable_never_drop_rows_silently",
            "post_target_quantile_or_bin_fitting_allowed": False,
        },
        "bootstrap": {
            "unit": "physical_identity",
            "samples": bootstrap_samples,
            "confidence": float(bootstrap_confidence),
            "interval": "two-sided percentile",
            "rng": "numpy.default_rng_PCG64",
            "base_seed": bootstrap_seed,
            "per_seed_seed_derivation": (
                "first 64 bits of SHA-256('locked-hcs-primary:<base>:<seed>') modulo 2^32"
            ),
            "same_resamples_for_candidates_and_paired_deltas": True,
            "resample_count_equals_physical_identity_count": True,
            "cross_seed_bootstrap_pooling_allowed": False,
        },
        "paired_comparisons": {
            name: {"challenger": pair[0], "reference": pair[1]}
            for name, pair in PAIRED_COMPARISONS.items()
        },
        "uncertainty": {
            "recognized_fields": list(UNCERTAINTY_FIELDS),
            "role": "diagnostic_ranking_only_not_calibrated_interval",
            "fixed_risk_coverages": [1.0, 0.9, 0.8, 0.5],
            "calibration_fit_allowed": False,
            "threshold_fit_allowed": False,
            "model_or_candidate_selection_allowed": False,
        },
        "post_target_prohibitions": {
            "prediction_column_search": True,
            "gate_or_threshold_change": True,
            "stratum_or_phase_search": True,
            "seed_pooling_ranking_or_suppression": True,
            "calibration_or_uncertainty_fit": True,
            "bootstrap_spec_change": True,
        },
    }


def freeze_evaluation_spec(
    output: Path,
    *,
    expected_rows: int = EXPECTED_VALID_REFERENCE_ROWS,
    expected_identities: int = EXPECTED_IDENTITIES,
    bootstrap_samples: int = BOOTSTRAP_SAMPLES,
    bootstrap_seed: int = BOOTSTRAP_SEED,
    bootstrap_confidence: float = BOOTSTRAP_CONFIDENCE,
) -> dict[str, Any]:
    """Publish the create-once specification without touching outer-test artifacts."""

    destination = output.expanduser().resolve()
    if destination.exists():
        raise LockedPrimaryEvaluationError(
            f"immutable evaluation specification already exists: {destination}"
        )
    document = evaluation_spec_document(
        expected_rows=expected_rows,
        expected_identities=expected_identities,
        bootstrap_samples=bootstrap_samples,
        bootstrap_seed=bootstrap_seed,
        bootstrap_confidence=bootstrap_confidence,
    )
    document["content_sha256"] = canonical_json_sha256(document)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = _temporary_path(destination)
    try:
        _write_json(temporary, document)
        _publish_exclusive(temporary, destination)
        destination.chmod(0o444)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
    published = _read_json(
        destination, "published primary evaluation specification", require_content_hash=True
    )
    if published != document:
        raise LockedPrimaryEvaluationError("published evaluation specification changed")
    return bind_file(destination)


def _load_evaluation_spec(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise LockedPrimaryEvaluationError(
            "immutable primary evaluation specification is absent; evaluation forbidden"
        )
    document = _read_json(
        resolved, "primary evaluation specification", require_content_hash=True
    )
    payload = dict(document)
    payload.pop("content_sha256", None)
    population = payload.get("population")
    bootstrap = payload.get("bootstrap")
    if not isinstance(population, Mapping) or not isinstance(bootstrap, Mapping):
        raise LockedPrimaryEvaluationError("evaluation specification population/bootstrap is absent")
    try:
        expected = evaluation_spec_document(
            expected_rows=int(population["valid_reference_rows_per_seed"]),
            expected_identities=int(population["physical_identity_count"]),
            bootstrap_samples=int(bootstrap["samples"]),
            bootstrap_seed=int(bootstrap["base_seed"]),
            bootstrap_confidence=float(bootstrap["confidence"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise LockedPrimaryEvaluationError("evaluation specification values are invalid") from exc
    if canonical_json_sha256(payload) != canonical_json_sha256(expected):
        raise LockedPrimaryEvaluationError(
            "evaluation specification differs from the exact frozen primary protocol"
        )
    return document, bind_file(resolved)


def _resolve(value: Any, *, relative_to: Path, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise LockedPrimaryEvaluationError(f"{label}.path must be a non-empty string")
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = relative_to / path
    return path.resolve()


def bind_file(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise LockedPrimaryEvaluationError(f"expected file is absent: {resolved}")
    return {
        "path": str(resolved),
        "sha256": sha256_file(resolved),
        "bytes": resolved.stat().st_size,
    }


def _binding_shape(raw: Any, *, relative_to: Path, label: str) -> dict[str, Any]:
    """Validate a binding without touching the bound file."""

    if not isinstance(raw, Mapping):
        raise LockedPrimaryEvaluationError(f"missing file binding: {label}")
    path = _resolve(raw.get("path"), relative_to=relative_to, label=label)
    expected = str(raw.get("sha256", "")).lower()
    if not _is_sha256(expected):
        raise LockedPrimaryEvaluationError(f"invalid SHA-256 binding: {label}")
    declared = raw.get("bytes")
    if declared is not None and (
        isinstance(declared, bool) or not isinstance(declared, int) or declared < 0
    ):
        raise LockedPrimaryEvaluationError(f"invalid byte-size binding: {label}")
    result = {"path": str(path), "sha256": expected}
    if declared is not None:
        result["bytes"] = declared
    return result


def _verify_binding(raw: Any, *, relative_to: Path, label: str) -> dict[str, Any]:
    binding = _binding_shape(raw, relative_to=relative_to, label=label)
    path = Path(binding["path"])
    if not path.is_file():
        raise LockedPrimaryEvaluationError(f"bound file is absent: {label} ({path})")
    size = path.stat().st_size
    if "bytes" in binding and binding["bytes"] != size:
        raise LockedPrimaryEvaluationError(f"file byte-size binding mismatch: {label}")
    if sha256_file(path) != binding["sha256"]:
        raise LockedPrimaryEvaluationError(f"file SHA-256 binding mismatch: {label} ({path})")
    return {"path": str(path), "sha256": binding["sha256"], "bytes": size}


def _verify_binding_tree(value: Any, *, relative_to: Path, label: str) -> list[dict[str, Any]]:
    """Rehash every mapping that declares both ``path`` and ``sha256``."""

    found: list[dict[str, Any]] = []
    if isinstance(value, Mapping):
        if "path" in value and "sha256" in value:
            found.append(_verify_binding(value, relative_to=relative_to, label=label))
            return found
        for key, child in value.items():
            found.extend(
                _verify_binding_tree(
                    child,
                    relative_to=relative_to,
                    label=f"{label}.{key}",
                )
            )
    elif isinstance(value, list):
        for position, child in enumerate(value):
            found.extend(
                _verify_binding_tree(
                    child,
                    relative_to=relative_to,
                    label=f"{label}[{position}]",
                )
            )
    return found


def _same_binding(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    return (
        Path(str(left["path"])).resolve() == Path(str(right["path"])).resolve()
        and str(left["sha256"]) == str(right["sha256"])
        and int(left.get("bytes", Path(str(left["path"])).stat().st_size))
        == int(right.get("bytes", Path(str(right["path"])).stat().st_size))
    )


def _scalar_integer(value: Any, *, label: str) -> int:
    array = np.asarray(value)
    if array.ndim != 0 or array.dtype.kind not in "iu" or array.dtype.kind == "b":
        raise LockedPrimaryEvaluationError(f"{label} must be an integer scalar")
    return int(array.item())


def _strict_integer_vector(value: Any, *, label: str) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != 1 or array.dtype.kind not in "iu" or array.dtype.kind == "b":
        raise LockedPrimaryEvaluationError(f"{label} must be an integer vector")
    return array.astype(np.int64, copy=False)


def _load_npz(path: Path, *, label: str) -> dict[str, np.ndarray]:
    try:
        with np.load(path, allow_pickle=False) as archive:
            return {name: np.asarray(archive[name]) for name in archive.files}
    except (OSError, ValueError) as exc:
        raise LockedPrimaryEvaluationError(f"invalid {label}: {path} ({exc})") from exc


def _array_schema(arrays: Mapping[str, np.ndarray]) -> dict[str, dict[str, Any]]:
    return {
        name: {
            "dtype": np.asarray(value).dtype.str,
            "shape": list(np.asarray(value).shape),
            "array_sha256": array_sha256(value),
        }
        for name, value in sorted(arrays.items())
    }


def _validate_prediction_npz(path: Path, *, fold: int, seed: int) -> np.ndarray:
    arrays = _load_npz(path, label=f"sealed prediction {fold}/{seed}")
    required = {
        "cache_index",
        "outer_fold",
        "seed",
        "fallback_rr_bpm",
        "source_rr_bpm",
        "final_rr_bpm",
        "target_joined",
    }
    if not required <= set(arrays):
        raise LockedPrimaryEvaluationError(
            f"sealed prediction fields are missing: {fold}/{seed}"
        )
    if _scalar_integer(arrays["outer_fold"], label="prediction outer_fold") != fold:
        raise LockedPrimaryEvaluationError("sealed prediction outer-fold identity mismatch")
    if _scalar_integer(arrays["seed"], label="prediction seed") != seed:
        raise LockedPrimaryEvaluationError("sealed prediction seed identity mismatch")
    joined = np.asarray(arrays["target_joined"])
    if joined.ndim != 0 or joined.dtype.kind != "b" or bool(joined.item()):
        raise LockedPrimaryEvaluationError("sealed prediction is not label-free")
    index = _strict_integer_vector(arrays["cache_index"], label="prediction cache_index")
    if len(index) == 0 or (index < 0).any() or not np.all(index[1:] > index[:-1]):
        raise LockedPrimaryEvaluationError(
            "sealed prediction cache_index must be non-empty, unique, and sorted"
        )
    for field in CANDIDATES.values():
        value = np.asarray(arrays[field])
        if value.shape != index.shape or value.dtype.kind not in "fc":
            raise LockedPrimaryEvaluationError(f"sealed prediction field shape/type: {field}")
        if not np.isfinite(value).all() or (value <= 0).any():
            raise LockedPrimaryEvaluationError(f"sealed prediction field values: {field}")
    return index.copy()


def _authorize_target_access(
    *, root: Path, evaluation_lock: Path
) -> tuple[dict[str, Any], dict[str, Any], dict[tuple[int, int], dict[str, Any]]]:
    """Validate the evaluation authorization without opening target-bearing files."""

    lock_path = evaluation_lock.expanduser().resolve()
    try:
        lock_path.relative_to(root)
    except ValueError as exc:
        raise LockedPrimaryEvaluationError(
            "evaluation lock must reside under --locked-oof-root"
        ) from exc
    if not lock_path.is_file():
        raise LockedPrimaryEvaluationError(
            "evaluation_lock.json is absent; target access is forbidden"
        )
    lock = _read_json(lock_path, "evaluation lock")
    if (
        lock.get("schema_version") != SCHEMA_VERSION
        or lock.get("classification") != "locked_hcs_oof_single_target_join_seal"
        or lock.get("target_join_count") != 1
        or lock.get("commercial_claim_authorized") is not False
        or lock.get("prospective_confirmation_required") is not True
    ):
        raise LockedPrimaryEvaluationError("evaluation lock authorization is invalid")
    # Validate all target-bearing binding *shapes* without stat/open/hash access.
    _binding_shape(lock.get("target_artifact"), relative_to=lock_path.parent, label="target")
    outputs = lock.get("outputs")
    if not isinstance(outputs, Mapping) or set(outputs) != {"joined_oof", "metrics"}:
        raise LockedPrimaryEvaluationError("evaluation lock output topology is invalid")
    _binding_shape(outputs["joined_oof"], relative_to=lock_path.parent, label="joined OOF")
    _binding_shape(outputs["metrics"], relative_to=lock_path.parent, label="locked metrics")

    seal_binding = _verify_binding(
        lock.get("predictions_seal"),
        relative_to=lock_path.parent,
        label="predictions seal",
    )
    seal_path = Path(seal_binding["path"])
    seal = _read_json(seal_path, "predictions seal")
    units = seal.get("units")
    if (
        seal.get("schema_version") != SCHEMA_VERSION
        or seal.get("classification")
        != "locked_hcs_oof_all_label_free_predictions_sealed"
        or seal.get("target_artifact_opened_before_seal") is not False
        or seal.get("target_join_authorized") is not True
        or seal.get("unit_count") != len(FOLDS) * EXPECTED_SEED_COUNT
        or seal.get("outer_folds") != list(FOLDS)
        or not isinstance(units, list)
        or len(units) != len(FOLDS) * EXPECTED_SEED_COUNT
    ):
        raise LockedPrimaryEvaluationError("predictions seal is incomplete or invalid")

    pretest_path = root / "pretest_lock.json"
    expected_pretest = str(seal.get("pretest_lock_sha256", "")).lower()
    if not _is_sha256(expected_pretest) or not pretest_path.is_file():
        raise LockedPrimaryEvaluationError("predictions seal pretest lock is absent")
    if sha256_file(pretest_path) != expected_pretest:
        raise LockedPrimaryEvaluationError("predictions seal pretest lock hash mismatch")
    pretest = _read_json(pretest_path, "pretest lock")
    _verify_binding_tree(pretest, relative_to=pretest_path.parent, label="pretest lock")

    inventory: dict[tuple[int, int], dict[str, Any]] = {}
    for position, raw in enumerate(units):
        if not isinstance(raw, Mapping):
            raise LockedPrimaryEvaluationError("predictions seal unit must be an object")
        try:
            fold = int(raw["outer_fold"])
            seed = int(raw["seed"])
        except (KeyError, TypeError, ValueError) as exc:
            raise LockedPrimaryEvaluationError("predictions seal unit identity is invalid") from exc
        if fold not in FOLDS or (fold, seed) in inventory:
            raise LockedPrimaryEvaluationError("predictions seal has duplicate/invalid unit identity")
        derived_binding = _verify_binding(
            raw.get("derived_lock"),
            relative_to=seal_path.parent,
            label=f"derived lock {fold}/{seed}",
        )
        prediction_binding = _verify_binding(
            raw.get("prediction"),
            relative_to=seal_path.parent,
            label=f"sealed prediction {fold}/{seed}",
        )
        derived_path = Path(derived_binding["path"])
        derived = _read_json(derived_path, f"derived lock {fold}/{seed}")
        if (
            derived.get("schema_version") != SCHEMA_VERSION
            or derived.get("classification") != "locked_hcs_oof_derived_test_inference"
            or derived.get("outer_fold") != fold
            or derived.get("seed") != seed
            or derived.get("target_artifact_opened") is not False
            or derived.get("pretest_lock_sha256") != expected_pretest
        ):
            raise LockedPrimaryEvaluationError(f"derived lock identity invariant: {fold}/{seed}")
        verified = _verify_binding_tree(
            derived,
            relative_to=derived_path.parent,
            label=f"derived lock {fold}/{seed}",
        )
        # Stage receipts are themselves bound by the derived lock.  Follow the
        # receipt boundary as well so its output and stdout/stderr-log bindings
        # are rehashed instead of trusting only the receipt file hash.
        receipts_raw = derived.get("stage_receipts")
        if receipts_raw is not None:
            if not isinstance(receipts_raw, list):
                raise LockedPrimaryEvaluationError(
                    f"derived stage receipt topology is invalid: {fold}/{seed}"
                )
            for receipt_position, receipt_raw in enumerate(receipts_raw):
                receipt_binding = _verify_binding(
                    receipt_raw,
                    relative_to=derived_path.parent,
                    label=f"stage receipt {fold}/{seed}/{receipt_position}",
                )
                receipt_path = Path(receipt_binding["path"])
                receipt = _read_json(
                    receipt_path,
                    f"stage receipt {fold}/{seed}/{receipt_position}",
                )
                if (
                    receipt.get("schema_version") != SCHEMA_VERSION
                    or receipt.get("classification") != "locked_hcs_oof_stage_receipt"
                ):
                    raise LockedPrimaryEvaluationError(
                        f"stage receipt invariant is invalid: {fold}/{seed}/{receipt_position}"
                    )
                verified.extend(
                    _verify_binding_tree(
                        receipt,
                        relative_to=receipt_path.parent,
                        label=f"stage receipt {fold}/{seed}/{receipt_position}",
                    )
                )
        sealed_raw = derived.get("sealed_prediction")
        sealed = _binding_shape(
            sealed_raw,
            relative_to=derived_path.parent,
            label=f"derived sealed prediction {fold}/{seed}",
        )
        if not _same_binding(sealed, prediction_binding):
            raise LockedPrimaryEvaluationError(
                f"prediction seal/derived prediction binding mismatch: {fold}/{seed}"
            )
        index = _validate_prediction_npz(
            Path(prediction_binding["path"]), fold=fold, seed=seed
        )
        inventory[(fold, seed)] = {
            "fold": fold,
            "seed": seed,
            "derived_lock": derived_binding,
            "prediction": prediction_binding,
            "cache_index": index,
            "verified_bound_artifact_count": len(verified),
        }
    seeds = sorted({seed for _, seed in inventory})
    if len(seeds) != EXPECTED_SEED_COUNT or set(inventory) != {
        (fold, seed) for seed in seeds for fold in FOLDS
    }:
        raise LockedPrimaryEvaluationError(
            "predictions seal must be an exact six-fold x three-seed Cartesian product"
        )
    return lock, seal_binding, inventory


def _verify_target_schema(
    target: Mapping[str, np.ndarray], receipt: Mapping[str, Any]
) -> dict[str, dict[str, Any]]:
    schema = receipt.get("target_schema")
    observed = _array_schema(target)
    if not isinstance(schema, Mapping) or set(schema) != set(observed):
        raise LockedPrimaryEvaluationError("target receipt schema differs from target NPZ")
    for name, actual in observed.items():
        declared = schema.get(name)
        if not isinstance(declared, Mapping) or any(
            declared.get(field) != actual[field]
            for field in ("dtype", "shape", "array_sha256")
        ):
            raise LockedPrimaryEvaluationError(f"target array schema/hash mismatch: {name}")
    return observed


def _fold_assignment_map(document: Mapping[str, Any]) -> dict[str, int]:
    raw = document.get("identity_to_fold", document)
    if not isinstance(raw, Mapping):
        raise LockedPrimaryEvaluationError("fold assignment identity map is absent")
    result: dict[str, int] = {}
    for identity, fold_raw in raw.items():
        if not isinstance(identity, str) or not identity:
            continue
        if isinstance(fold_raw, bool) or not isinstance(fold_raw, int):
            continue
        result[identity] = int(fold_raw)
    return result


def _validate_context(
    *,
    locked_oof_root: Path,
    evaluation_lock: Path,
    target_receipt: Path,
    expected_rows: int,
    expected_identities: int,
    expected_seeds: Sequence[int],
) -> tuple[dict[int, dict[str, np.ndarray]], dict[str, Any]]:
    root = locked_oof_root.expanduser().resolve()
    lock_path = evaluation_lock.expanduser().resolve()

    # This is the authorization boundary.  No target, joined, metric, target
    # receipt, or target metadata file is resolved/stat'd/opened above it.
    lock, seal_binding, prediction_inventory = _authorize_target_access(
        root=root, evaluation_lock=lock_path
    )

    # The evaluation lock is now authorized.  Rehash every one of its declared
    # bindings, including the target-bearing artifacts.
    receipt_path = target_receipt.expanduser().resolve()
    lock_bindings = _verify_binding_tree(
        lock, relative_to=lock_path.parent, label="evaluation lock"
    )
    target_binding = _verify_binding(
        lock.get("target_artifact"), relative_to=lock_path.parent, label="target artifact"
    )
    outputs = lock["outputs"]
    joined_binding = _verify_binding(
        outputs["joined_oof"], relative_to=lock_path.parent, label="joined OOF"
    )
    metrics_binding = _verify_binding(
        outputs["metrics"], relative_to=lock_path.parent, label="locked metrics"
    )

    if not receipt_path.is_file():
        raise LockedPrimaryEvaluationError("canonical target receipt is absent")
    receipt = _read_json(
        receipt_path, "canonical target receipt", require_content_hash=True
    )
    if (
        receipt.get("schema_version") != SCHEMA_VERSION
        or receipt.get("classification")
        != "retrospective_locked_hcs_canonical_target_artifact_receipt"
        or receipt.get("target_artifact_created_once") is not True
        or receipt.get("target_artifact_overwrite_allowed") is not False
        or receipt.get("commercial_claim_authorized") is not False
        or receipt.get("prospective_confirmation_required") is not True
    ):
        raise LockedPrimaryEvaluationError("canonical target receipt invariant is invalid")
    receipt_bindings = _verify_binding_tree(
        receipt, relative_to=receipt_path.parent, label="target receipt"
    )
    receipt_target = _verify_binding(
        receipt.get("target_artifact"),
        relative_to=receipt_path.parent,
        label="receipt target artifact",
    )
    if not _same_binding(receipt_target, target_binding):
        raise LockedPrimaryEvaluationError(
            "evaluation lock and target receipt bind different target artifacts"
        )
    source_bindings = receipt.get("source_bindings")
    if not isinstance(source_bindings, Mapping):
        raise LockedPrimaryEvaluationError("target receipt source bindings are absent")
    receipt_seal = _verify_binding(
        source_bindings.get("predictions_seal"),
        relative_to=receipt_path.parent,
        label="receipt predictions seal",
    )
    if not _same_binding(receipt_seal, seal_binding):
        raise LockedPrimaryEvaluationError(
            "evaluation lock and target receipt bind different prediction seals"
        )

    topology = receipt.get("prediction_topology")
    if not isinstance(topology, Mapping):
        raise LockedPrimaryEvaluationError("target receipt prediction topology is absent")
    seeds_raw = topology.get("seeds")
    if not isinstance(seeds_raw, list) or any(
        isinstance(value, bool) or not isinstance(value, int) for value in seeds_raw
    ):
        raise LockedPrimaryEvaluationError("target receipt seeds are invalid")
    seeds = sorted(int(value) for value in seeds_raw)
    if (
        len(seeds) != EXPECTED_SEED_COUNT
        or len(set(seeds)) != EXPECTED_SEED_COUNT
        or seeds != sorted(int(value) for value in expected_seeds)
        or topology.get("folds") != list(FOLDS)
        or topology.get("unit_count") != len(FOLDS) * EXPECTED_SEED_COUNT
        or set(prediction_inventory) != {(fold, seed) for seed in seeds for fold in FOLDS}
    ):
        raise LockedPrimaryEvaluationError("receipt/seal seed-fold topology mismatch")
    inventory_raw = receipt.get("prediction_inventory")
    if not isinstance(inventory_raw, list) or len(inventory_raw) != len(prediction_inventory):
        raise LockedPrimaryEvaluationError("target receipt prediction inventory is incomplete")
    inventory_keys: set[tuple[int, int]] = set()
    for raw in inventory_raw:
        if not isinstance(raw, Mapping):
            raise LockedPrimaryEvaluationError("target receipt prediction inventory is invalid")
        key = (int(raw.get("outer_fold", -1)), int(raw.get("seed", -1)))
        if key not in prediction_inventory or key in inventory_keys:
            raise LockedPrimaryEvaluationError("target receipt inventory unit mismatch")
        inventory_keys.add(key)
        for field in ("derived_lock", "prediction"):
            bound = _verify_binding(
                raw.get(field), relative_to=receipt_path.parent, label=f"receipt {field} {key}"
            )
            if not _same_binding(bound, prediction_inventory[key][field]):
                raise LockedPrimaryEvaluationError(f"receipt/seal {field} mismatch: {key}")
        declared_rows = raw.get("prediction_rows")
        if declared_rows is not None and declared_rows != len(
            prediction_inventory[key]["cache_index"]
        ):
            raise LockedPrimaryEvaluationError(f"receipt prediction row mismatch: {key}")
        declared_index_hash = raw.get("prediction_index_sha256")
        if declared_index_hash is not None and declared_index_hash != array_sha256(
            prediction_inventory[key]["cache_index"]
        ):
            raise LockedPrimaryEvaluationError(f"receipt prediction index hash mismatch: {key}")

    locked_metrics = _read_json(Path(metrics_binding["path"]), "locked metrics")
    if (
        locked_metrics.get("schema_version") != SCHEMA_VERSION
        or locked_metrics.get("classification") != "retrospective_locked_hcs_oof_evaluation"
        or locked_metrics.get("commercial_claim_authorized") is not False
        or locked_metrics.get("prospective_confirmation_required") is not True
    ):
        raise LockedPrimaryEvaluationError("locked metrics invariant is invalid")

    target = _load_npz(Path(target_binding["path"]), label="canonical target artifact")
    joined = _load_npz(Path(joined_binding["path"]), label="locked joined OOF")
    target_schema = _verify_target_schema(target, receipt)
    missing_target = sorted(REQUIRED_TARGET_FIELDS - set(target))
    missing_joined = sorted(REQUIRED_JOINED_FIELDS - set(joined))
    if missing_target or missing_joined:
        raise LockedPrimaryEvaluationError(
            f"locked arrays lack required fields (target={missing_target}, joined={missing_joined})"
        )
    target_rows = len(np.asarray(target["cache_index"]))
    if any(np.asarray(value).shape != (target_rows,) for value in target.values()):
        raise LockedPrimaryEvaluationError("target arrays do not share one row shape")
    target_index = _strict_integer_vector(target["cache_index"], label="target cache_index")
    target_fold = _strict_integer_vector(target["outer_fold"], label="target outer_fold")
    if not np.array_equal(target_index, np.arange(target_rows, dtype=np.int64)):
        raise LockedPrimaryEvaluationError("target cache_index is not exact contiguous cache order")
    if set(np.unique(target_fold).tolist()) != set(FOLDS):
        raise LockedPrimaryEvaluationError("target does not cover all fixed folds")
    target_rr = np.asarray(target["target_rr_bpm"], dtype=np.float64)
    reference_valid_raw = np.asarray(target["reference_valid"])
    if reference_valid_raw.dtype.kind != "b":
        raise LockedPrimaryEvaluationError("target reference_valid must be boolean")
    reference_valid = reference_valid_raw.astype(bool, copy=False)
    if not np.isfinite(target_rr).all() or (target_rr <= 0).any():
        raise LockedPrimaryEvaluationError("target RR contains non-finite/non-positive values")
    valid = reference_valid & np.isfinite(target_rr)
    if int(valid.sum()) != expected_rows or receipt.get("valid_reference_rows") != expected_rows:
        raise LockedPrimaryEvaluationError(
            f"valid-reference population is {int(valid.sum())}, expected {expected_rows}"
        )
    if receipt.get("row_count") != target_rows:
        raise LockedPrimaryEvaluationError("target receipt row count mismatch")

    identity = np.asarray(target["identity"]).astype(str)
    session = np.asarray(target["session_id"]).astype(str)
    protocol = np.asarray(target["protocol"]).astype(str)
    window = _strict_integer_vector(target["window_number"], label="target window_number")
    start = np.asarray(target["window_start_s"], dtype=np.float64)
    end = np.asarray(target["window_end_s"], dtype=np.float64)
    if (
        any(not value or value != value.strip() for value in identity.tolist())
        or any(not value or value != value.strip() for value in session.tolist())
        or any(not value or value != value.strip() for value in protocol.tolist())
        or (window < 0).any()
        or not np.isfinite(start).all()
        or not np.isfinite(end).all()
        or (end <= start).any()
    ):
        raise LockedPrimaryEvaluationError("target semantic lineage is invalid")
    if len(set(zip(session.tolist(), window.tolist(), strict=True))) != target_rows:
        raise LockedPrimaryEvaluationError("target repeats a session/window semantic key")
    valid_identities = sorted(set(identity[valid].tolist()))
    if len(valid_identities) != expected_identities:
        raise LockedPrimaryEvaluationError(
            f"valid identities are {len(valid_identities)}, expected {expected_identities}"
        )
    identity_to_fold: dict[str, int] = {}
    for name in sorted(set(identity.tolist())):
        owned = set(target_fold[identity == name].tolist())
        if len(owned) != 1:
            raise LockedPrimaryEvaluationError(f"physical identity crosses folds: {name}")
        identity_to_fold[name] = int(next(iter(owned)))
    for name in sorted(set(session.tolist())):
        if len(set(identity[session == name].tolist())) != 1:
            raise LockedPrimaryEvaluationError(f"session crosses physical identities: {name}")

    fold_binding_raw = source_bindings.get("fold_assignments")
    if isinstance(fold_binding_raw, Mapping):
        fold_binding = _verify_binding(
            fold_binding_raw,
            relative_to=receipt_path.parent,
            label="receipt fold assignments",
        )
        fold_document = _read_json(Path(fold_binding["path"]), "fold assignments")
        assigned = _fold_assignment_map(fold_document)
        if any(assigned.get(name) != fold for name, fold in identity_to_fold.items()):
            raise LockedPrimaryEvaluationError("target identity/fold differs from cache authority")

    # Every seed/fold prediction must match the exact full cache ownership.
    for seed in seeds:
        union: list[np.ndarray] = []
        for fold in FOLDS:
            index = prediction_inventory[(fold, seed)]["cache_index"]
            expected = target_index[target_fold == fold]
            if not np.array_equal(index, expected):
                raise LockedPrimaryEvaluationError(
                    f"prediction cache/fold ownership mismatch: {fold}/{seed}"
                )
            union.append(index)
        if not np.array_equal(np.sort(np.concatenate(union)), target_index):
            raise LockedPrimaryEvaluationError(f"prediction cache coverage mismatch: seed {seed}")

    joined_rows = len(np.asarray(joined["cache_index"]))
    if any(np.asarray(value).shape != (joined_rows,) for value in joined.values()):
        raise LockedPrimaryEvaluationError("joined arrays do not share one row shape")
    if joined_rows != expected_rows * EXPECTED_SEED_COUNT:
        raise LockedPrimaryEvaluationError("joined row count is not rows x three fixed seeds")
    joined_index = _strict_integer_vector(joined["cache_index"], label="joined cache_index")
    joined_fold = _strict_integer_vector(joined["outer_fold"], label="joined outer_fold")
    joined_seed = _strict_integer_vector(joined["seed"], label="joined seed")
    joined_target = np.asarray(joined["target_rr_bpm"])
    joined_identity = np.asarray(joined["identity"]).astype(str)
    valid_index = target_index[valid]
    lookup = {int(value): position for position, value in enumerate(target_index)}
    frames: dict[int, dict[str, np.ndarray]] = {}
    for seed in seeds:
        selected = np.flatnonzero(joined_seed == seed)
        if len(selected) != expected_rows:
            raise LockedPrimaryEvaluationError(f"joined seed row count mismatch: {seed}")
        order = np.argsort(joined_index[selected], kind="stable")
        rows = selected[order]
        index = joined_index[rows]
        if not np.array_equal(index, valid_index):
            raise LockedPrimaryEvaluationError(
                f"joined seed is not the exact valid cache cover: {seed}"
            )
        positions = np.asarray([lookup[int(value)] for value in index], dtype=np.int64)
        if (
            not np.array_equal(joined_fold[rows], target_fold[positions])
            or not np.array_equal(joined_identity[rows], identity[positions])
            or not np.array_equal(joined_target[rows], np.asarray(target["target_rr_bpm"])[positions])
        ):
            raise LockedPrimaryEvaluationError(
                f"joined target/fold/cache/identity lineage mismatch: seed {seed}"
            )
        frame = {
            name: np.asarray(value)[positions]
            for name, value in target.items()
        }
        frame["seed"] = np.full(expected_rows, seed, dtype=np.int64)
        for name, value in joined.items():
            if name not in {"cache_index", "outer_fold", "seed", "target_rr_bpm", "identity"}:
                frame[name] = np.asarray(value)[rows]
        for field in CANDIDATES.values():
            prediction = np.asarray(frame[field], dtype=np.float64)
            if not np.isfinite(prediction).all() or (prediction <= 0).any():
                raise LockedPrimaryEvaluationError(f"invalid joined prediction: {seed}/{field}")
        frames[seed] = frame

    _cross_check_locked_metrics(locked_metrics, frames)
    context = {
        "bindings": {
            "evaluation_lock": bind_file(lock_path),
            "predictions_seal": seal_binding,
            "target_receipt": bind_file(receipt_path),
            "target_artifact": target_binding,
            "joined_oof": joined_binding,
            "locked_metrics": metrics_binding,
        },
        "all_evaluation_lock_bindings_verified": len(lock_bindings),
        "all_target_receipt_bindings_verified": len(receipt_bindings),
        "target_array_schema": target_schema,
        "joined_array_schema": _array_schema(joined),
        "seeds": seeds,
        "identity_to_fold": identity_to_fold,
        "full_cache_rows": target_rows,
        "valid_reference_rows_per_seed": expected_rows,
        "identity_count": len(valid_identities),
        "fold_count": len(FOLDS),
        "seed_count": len(seeds),
        "exact_seed_fold_cache_identity_alignment": True,
        "target_access_authorized_only_after_evaluation_lock_validation": True,
    }
    return frames, context


def _metric_summary(
    target: np.ndarray, prediction: np.ndarray, identity: np.ndarray
) -> dict[str, Any]:
    target = np.asarray(target, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    identity = np.asarray(identity).astype(str)
    if not (target.shape == prediction.shape == identity.shape) or target.ndim != 1:
        raise LockedPrimaryEvaluationError("metric arrays must share one vector shape")
    if len(target) == 0:
        return {
            "rows": 0,
            "identities": 0,
            "tail_25_35_rows": 0,
            **{name: None for name in METRIC_FIELDS},
        }
    error = np.abs(prediction - target)
    squared = np.square(prediction - target)
    names = sorted(set(identity.tolist()))
    per_identity_mae = [float(error[identity == name].mean()) for name in names]
    tail = (target >= 25.0) & (target <= 35.0)
    return {
        "rows": int(len(target)),
        "identities": len(names),
        "mae": float(error.mean()),
        "identity_macro_mae": float(np.mean(per_identity_mae)),
        "rmse": float(np.sqrt(squared.mean())),
        "within_2_fraction": float(np.mean(error <= 2.0)),
        "over_5_fraction": float(np.mean(error > 5.0)),
        "tail_25_35_rows": int(tail.sum()),
        "tail_25_35_mae": float(error[tail].mean()) if tail.any() else None,
    }


def _locked_metric_record(
    target: np.ndarray, prediction: np.ndarray, identity: np.ndarray
) -> dict[str, Any]:
    summary = _metric_summary(target, prediction, identity)
    error = np.abs(np.asarray(prediction, float) - np.asarray(target, float))
    names = sorted(set(np.asarray(identity).astype(str).tolist()))
    return {
        "rows": summary["rows"],
        "mae": summary["mae"],
        "identity_macro_mae": summary["identity_macro_mae"],
        "identity_mae": {
            name: float(error[np.asarray(identity).astype(str) == name].mean()) for name in names
        },
        "rmse": summary["rmse"],
        "within_2": summary["within_2_fraction"],
        "catastrophic_over_5": summary["over_5_fraction"],
        "tail_25_35_mae": summary["tail_25_35_mae"],
    }


def _numbers_close(left: Any, right: Any) -> bool:
    if left is None or right is None:
        return left is None and right is None
    try:
        return math.isclose(float(left), float(right), rel_tol=1e-12, abs_tol=1e-12)
    except (TypeError, ValueError):
        return False


def _cross_check_locked_metrics(
    locked: Mapping[str, Any], frames: Mapping[int, Mapping[str, np.ndarray]]
) -> None:
    per_seed = locked.get("per_seed")
    if not isinstance(per_seed, Mapping) or set(per_seed) != {str(seed) for seed in frames}:
        raise LockedPrimaryEvaluationError("locked metrics seed topology mismatch")
    for seed, frame in frames.items():
        group = per_seed[str(seed)]
        if not isinstance(group, Mapping) or set(group) != set(CANDIDATES):
            raise LockedPrimaryEvaluationError(f"locked metrics candidate topology: {seed}")
        for candidate, field in CANDIDATES.items():
            expected = _locked_metric_record(
                frame["target_rr_bpm"], frame[field], frame["identity"]
            )
            observed = group[candidate]
            if not isinstance(observed, Mapping) or observed.get("rows") != expected["rows"]:
                raise LockedPrimaryEvaluationError(f"locked metric row mismatch: {seed}/{candidate}")
            for field_name in (
                "mae",
                "identity_macro_mae",
                "rmse",
                "within_2",
                "catastrophic_over_5",
                "tail_25_35_mae",
            ):
                if not _numbers_close(observed.get(field_name), expected[field_name]):
                    raise LockedPrimaryEvaluationError(
                        f"locked metric value mismatch: {seed}/{candidate}/{field_name}"
                    )
            identity_observed = observed.get("identity_mae")
            if not isinstance(identity_observed, Mapping) or set(identity_observed) != set(
                expected["identity_mae"]
            ):
                raise LockedPrimaryEvaluationError(
                    f"locked metric identity topology mismatch: {seed}/{candidate}"
                )
            for identity, value in expected["identity_mae"].items():
                if not _numbers_close(identity_observed[identity], value):
                    raise LockedPrimaryEvaluationError(
                        f"locked metric identity value mismatch: {seed}/{candidate}/{identity}"
                    )


def _greedy_nonoverlap_mask(frame: Mapping[str, np.ndarray]) -> np.ndarray:
    rows = len(frame["cache_index"])
    keep = np.zeros(rows, dtype=bool)
    session = np.asarray(frame["session_id"]).astype(str)
    start = np.asarray(frame["window_start_s"], dtype=float)
    end = np.asarray(frame["window_end_s"], dtype=float)
    window = np.asarray(frame["window_number"], dtype=np.int64)
    index = np.asarray(frame["cache_index"], dtype=np.int64)
    for name in dict.fromkeys(session.tolist()):
        positions = np.flatnonzero(session == name)
        order = positions[np.lexsort((index[positions], window[positions], end[positions]))]
        last_end = -math.inf
        for position in order:
            if start[position] >= last_end - 1.0e-9:
                keep[position] = True
                last_end = end[position]
    return keep


def _intervals_nonoverlap(frame: Mapping[str, np.ndarray], mask: np.ndarray) -> bool:
    session = np.asarray(frame["session_id"]).astype(str)
    start = np.asarray(frame["window_start_s"], dtype=float)
    end = np.asarray(frame["window_end_s"], dtype=float)
    for name in set(session[mask].tolist()):
        positions = np.flatnonzero(mask & (session == name))
        order = positions[np.argsort(start[positions], kind="stable")]
        if len(order) > 1 and np.any(start[order[1:]] < end[order[:-1]] - 1.0e-9):
            return False
    return True


def _qc_strata(frame: Mapping[str, np.ndarray]) -> dict[str, dict[str, np.ndarray]]:
    rows = len(frame["cache_index"])
    result: dict[str, dict[str, np.ndarray]] = {}

    def numeric(name: str, cuts: tuple[float, float]) -> None:
        if name not in frame:
            return
        value = np.asarray(frame[name], dtype=float)
        if value.shape != (rows,) or not np.isfinite(value).all():
            raise LockedPrimaryEvaluationError(f"QC field is invalid: {name}")
        low, high = cuts
        result[name] = {
            f"lt_{low:g}": value < low,
            f"{low:g}_to_lt_{high:g}": (value >= low) & (value < high),
            f"ge_{high:g}": value >= high,
        }

    for field, cuts in QC_NUMERIC_STRATA_SPEC.items():
        numeric(field, cuts)
    if "radar_observable" in frame:
        raw = np.asarray(frame["radar_observable"])
        if raw.shape != (rows,) or raw.dtype.kind != "b":
            raise LockedPrimaryEvaluationError("QC field is invalid: radar_observable")
        result["radar_observable"] = {"false": ~raw, "true": raw}
    return result


def _strata(frame: Mapping[str, np.ndarray]) -> dict[str, dict[str, np.ndarray]]:
    rows = len(frame["cache_index"])
    result: dict[str, dict[str, np.ndarray]] = {}
    for kind, field in (
        ("identity", "identity"),
        ("fold", "outer_fold"),
        ("session", "session_id"),
        ("protocol", "protocol"),
    ):
        values = np.asarray(frame[field]).astype(str)
        result[kind] = {name: values == name for name in sorted(set(values.tolist()))}
    rr = np.asarray(frame["target_rr_bpm"], dtype=float)
    result["rr_band"] = {
        "lt_12": rr < 12.0,
        "12_to_lt_18": (rr >= 12.0) & (rr < 18.0),
        "18_to_lt_25": (rr >= 18.0) & (rr < 25.0),
        "25_to_35_inclusive": (rr >= 25.0) & (rr <= 35.0),
        "gt_35": rr > 35.0,
    }
    qc = _qc_strata(frame)
    if qc:
        for field, groups in qc.items():
            result[f"qc:{field}"] = groups
    else:
        result["qc:unavailable"] = {"no_qc_fields_in_target_artifact": np.zeros(rows, bool)}
    return result


def _gate_decision(metrics: Mapping[str, Any]) -> dict[str, Any]:
    checks = {
        "overall_mae": {
            "value": metrics["mae"],
            "operator": "<=",
            "threshold": GOAL_TARGETS["overall_mae_max_bpm"],
        },
        "identity_macro_mae": {
            "value": metrics["identity_macro_mae"],
            "operator": "<=",
            "threshold": GOAL_TARGETS["identity_macro_mae_max_bpm"],
        },
        "overall_rmse": {
            "value": metrics["rmse"],
            "operator": "<=",
            "threshold": GOAL_TARGETS["overall_rmse_max_bpm"],
        },
        "within_2_fraction": {
            "value": metrics["within_2_fraction"],
            "operator": ">=",
            "threshold": GOAL_TARGETS["within_2_min_fraction"],
        },
        "over_5_fraction": {
            "value": metrics["over_5_fraction"],
            "operator": "<=",
            "threshold": GOAL_TARGETS["over_5_max_fraction"],
        },
        "tail_25_35_mae": {
            "value": metrics["tail_25_35_mae"],
            "operator": "<=",
            "threshold": GOAL_TARGETS["high_rr_25_35_mae_max_bpm"],
        },
    }
    for record in checks.values():
        value = record["value"]
        record["passed"] = bool(
            value is not None
            and (
                float(value) <= float(record["threshold"])
                if record["operator"] == "<="
                else float(value) >= float(record["threshold"])
            )
        )
    return {"all_point_gates_passed": all(item["passed"] for item in checks.values()), "checks": checks}


def _paired_summary(
    target: np.ndarray,
    challenger: np.ndarray,
    reference: np.ndarray,
    identity: np.ndarray,
) -> dict[str, Any]:
    challenger_metrics = _metric_summary(target, challenger, identity)
    reference_metrics = _metric_summary(target, reference, identity)
    delta = {
        name: (
            None
            if challenger_metrics[name] is None or reference_metrics[name] is None
            else float(challenger_metrics[name] - reference_metrics[name])
        )
        for name in METRIC_FIELDS
    }
    challenger_error = np.abs(np.asarray(challenger, float) - np.asarray(target, float))
    reference_error = np.abs(np.asarray(reference, float) - np.asarray(target, float))
    difference = challenger_error - reference_error
    tolerance = 1.0e-12
    return {
        "delta_definition": "challenger_metric_minus_reference_metric",
        "negative_mae_rmse_tail_delta_is_better": True,
        "positive_within_2_delta_is_better": True,
        "negative_over_5_delta_is_better": True,
        "rows": int(len(target)),
        "identities": int(len(set(np.asarray(identity).astype(str).tolist()))),
        "tail_25_35_rows": int(np.sum((target >= 25.0) & (target <= 35.0))),
        **delta,
        "improved_fraction": float(np.mean(difference < -tolerance)),
        "tied_fraction": float(np.mean(np.abs(difference) <= tolerance)),
        "worsened_fraction": float(np.mean(difference > tolerance)),
        "mean_paired_absolute_error_delta_bpm": float(difference.mean()),
    }


def _cluster_components(
    target: np.ndarray, prediction: np.ndarray, identity: np.ndarray, names: Sequence[str]
) -> dict[str, np.ndarray]:
    error = np.abs(prediction - target)
    tail = (target >= 25.0) & (target <= 35.0)
    return {
        "rows": np.asarray([np.sum(identity == name) for name in names], float),
        "abs_sum": np.asarray([np.sum(error[identity == name]) for name in names], float),
        "squared_sum": np.asarray(
            [np.sum(np.square(prediction[identity == name] - target[identity == name])) for name in names],
            float,
        ),
        "within_count": np.asarray(
            [np.sum(error[identity == name] <= 2.0) for name in names], float
        ),
        "over_count": np.asarray(
            [np.sum(error[identity == name] > 5.0) for name in names], float
        ),
        "tail_count": np.asarray(
            [np.sum((identity == name) & tail) for name in names], float
        ),
        "tail_abs_sum": np.asarray(
            [np.sum(error[(identity == name) & tail]) for name in names], float
        ),
        "identity_mae": np.asarray([np.mean(error[identity == name]) for name in names], float),
    }


def _bootstrap_values(
    components: Mapping[str, np.ndarray], weights: np.ndarray
) -> dict[str, np.ndarray]:
    row_count = weights @ components["rows"]
    tail_count = weights @ components["tail_count"]
    identities = weights.shape[1]
    return {
        "mae": (weights @ components["abs_sum"]) / row_count,
        "identity_macro_mae": (weights @ components["identity_mae"]) / identities,
        "rmse": np.sqrt((weights @ components["squared_sum"]) / row_count),
        "within_2_fraction": (weights @ components["within_count"]) / row_count,
        "over_5_fraction": (weights @ components["over_count"]) / row_count,
        "tail_25_35_mae": np.divide(
            weights @ components["tail_abs_sum"],
            tail_count,
            out=np.full(len(weights), np.nan, float),
            where=tail_count > 0,
        ),
    }


def _percentile_interval(
    values: np.ndarray, *, estimate: float | None, samples: int, confidence: float
) -> dict[str, Any] | None:
    finite = np.asarray(values, float)
    finite = finite[np.isfinite(finite)]
    if estimate is None or len(finite) == 0:
        return None
    alpha = (1.0 - confidence) / 2.0
    lower, upper = np.quantile(finite, [alpha, 1.0 - alpha])
    return {
        "estimate": float(estimate),
        "lower": float(lower),
        "upper": float(upper),
        "confidence": float(confidence),
        "bootstrap_unit": "physical_identity",
        "samples_requested": int(samples),
        "samples_finite": int(len(finite)),
    }


def _derived_bootstrap_seed(base: int, seed: int) -> int:
    digest = hashlib.sha256(f"locked-hcs-primary:{base}:{seed}".encode("utf-8")).hexdigest()
    return int(digest[:16], 16) % (2**32)


def _bootstrap_seed_report(
    frame: Mapping[str, np.ndarray],
    *,
    seed: int,
    samples: int,
    base_seed: int,
    confidence: float,
) -> dict[str, Any]:
    target = np.asarray(frame["target_rr_bpm"], float)
    identity = np.asarray(frame["identity"]).astype(str)
    names = sorted(set(identity.tolist()))
    if len(names) < 2:
        raise LockedPrimaryEvaluationError("identity bootstrap needs at least two identities")
    derived_seed = _derived_bootstrap_seed(base_seed, seed)
    rng = np.random.default_rng(derived_seed)
    weights = rng.multinomial(
        len(names), np.full(len(names), 1.0 / len(names)), size=samples
    ).astype(float)
    sampled: dict[str, dict[str, np.ndarray]] = {}
    candidate_intervals: dict[str, Any] = {}
    for candidate, field in CANDIDATES.items():
        prediction = np.asarray(frame[field], float)
        components = _cluster_components(target, prediction, identity, names)
        values = _bootstrap_values(components, weights)
        sampled[candidate] = values
        point = _metric_summary(target, prediction, identity)
        candidate_intervals[candidate] = {
            name: _percentile_interval(
                values[name], estimate=point[name], samples=samples, confidence=confidence
            )
            for name in METRIC_FIELDS
        }
    comparison_intervals: dict[str, Any] = {}
    for name, (challenger, reference) in PAIRED_COMPARISONS.items():
        point = _paired_summary(
            target,
            np.asarray(frame[CANDIDATES[challenger]], float),
            np.asarray(frame[CANDIDATES[reference]], float),
            identity,
        )
        comparison_intervals[name] = {
            field: _percentile_interval(
                sampled[challenger][field] - sampled[reference][field],
                estimate=point[field],
                samples=samples,
                confidence=confidence,
            )
            for field in METRIC_FIELDS
        }
    return {
        "fixed_spec": {
            "unit": "physical_identity",
            "identity_count": len(names),
            "samples": samples,
            "confidence": confidence,
            "interval": "two-sided percentile",
            "rng": "numpy.default_rng_PCG64",
            "base_seed": base_seed,
            "per_seed_derived_seed": derived_seed,
            "seed_derivation": "first 64 bits of SHA-256('locked-hcs-primary:<base>:<seed>') modulo 2^32",
            "same_cluster_resamples_shared_across_candidates_and_paired_deltas": True,
            "cross_model_seed_pooling": False,
        },
        "candidate_intervals": candidate_intervals,
        "paired_delta_intervals": comparison_intervals,
    }


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values), dtype=float)
    position = 0
    while position < len(values):
        end = position + 1
        while end < len(values) and values[order[end]] == values[order[position]]:
            end += 1
        ranks[order[position:end]] = 0.5 * (position + end - 1) + 1.0
        position = end
    return ranks


def _correlation(left: np.ndarray, right: np.ndarray) -> float | None:
    left = np.asarray(left, float)
    right = np.asarray(right, float)
    if len(left) < 2 or float(np.std(left)) == 0.0 or float(np.std(right)) == 0.0:
        return None
    return float(np.corrcoef(left, right)[0, 1])


def _uncertainty_diagnostics(frame: Mapping[str, np.ndarray]) -> dict[str, Any]:
    fields = [name for name in UNCERTAINTY_FIELDS if name in frame]
    if not fields:
        return {
            "available": False,
            "role": "diagnostic_only",
            "calibration_fit_performed": False,
            "threshold_fit_performed": False,
            "fields": {},
        }
    target = np.asarray(frame["target_rr_bpm"], float)
    prediction = np.asarray(frame["final_rr_bpm"], float)
    identity = np.asarray(frame["identity"]).astype(str)
    absolute_error = np.abs(prediction - target)
    result: dict[str, Any] = {}
    for field in fields:
        uncertainty = np.asarray(frame[field], float)
        if uncertainty.shape != target.shape or not np.isfinite(uncertainty).all() or (
            uncertainty < 0
        ).any():
            raise LockedPrimaryEvaluationError(f"uncertainty diagnostic field is invalid: {field}")
        order = np.argsort(uncertainty, kind="stable")
        risk_coverage: dict[str, Any] = {}
        for coverage in (1.0, 0.9, 0.8, 0.5):
            count = max(1, int(math.ceil(coverage * len(order))))
            selected = order[:count]
            risk_coverage[f"{coverage:.1f}"] = {
                "requested_coverage": coverage,
                "actual_coverage": float(count / len(order)),
                "uncertainty_threshold_is_observed_order_statistic_not_fitted": float(
                    uncertainty[order[count - 1]]
                ),
                "metrics": _metric_summary(
                    target[selected], prediction[selected], identity[selected]
                ),
            }
        result[field] = {
            "role": "ranking_diagnostic_only_not_a_calibrated_interval",
            "pearson_with_absolute_error": _correlation(uncertainty, absolute_error),
            "spearman_with_absolute_error": _correlation(
                _average_ranks(uncertainty), _average_ranks(absolute_error)
            ),
            "risk_coverage": risk_coverage,
        }
    return {
        "available": True,
        "role": "diagnostic_only",
        "calibration_fit_performed": False,
        "interval_coverage_claim_performed": False,
        "threshold_fit_performed": False,
        "fields": result,
    }


def _metric_csv_record(
    *,
    seed: int,
    candidate: str,
    scope: str,
    stratum_type: str,
    stratum: str,
    metrics: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "record_type": "metric",
        "seed": seed,
        "candidate": candidate,
        "comparison": "",
        "scope": scope,
        "stratum_type": stratum_type,
        "stratum": stratum,
        "rows": metrics["rows"],
        "identities": metrics["identities"],
        "tail_25_35_rows": metrics["tail_25_35_rows"],
        **{name: metrics[name] for name in METRIC_FIELDS},
        "improved_fraction": "",
        "tied_fraction": "",
        "worsened_fraction": "",
    }


def _delta_csv_record(
    *, seed: int, comparison: str, metrics: Mapping[str, Any]
) -> dict[str, Any]:
    return {
        "record_type": "paired_delta",
        "seed": seed,
        "candidate": "",
        "comparison": comparison,
        "scope": "full",
        "stratum_type": "all",
        "stratum": "all",
        "rows": metrics["rows"],
        "identities": metrics["identities"],
        "tail_25_35_rows": metrics["tail_25_35_rows"],
        **{name: metrics[name] for name in METRIC_FIELDS},
        "improved_fraction": metrics["improved_fraction"],
        "tied_fraction": metrics["tied_fraction"],
        "worsened_fraction": metrics["worsened_fraction"],
    }


def _evaluate_seed(
    frame: Mapping[str, np.ndarray],
    *,
    seed: int,
    bootstrap_samples: int,
    bootstrap_seed: int,
    bootstrap_confidence: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    target = np.asarray(frame["target_rr_bpm"], float)
    identity = np.asarray(frame["identity"]).astype(str)
    rows = len(target)
    all_rows = np.ones(rows, dtype=bool)
    nonoverlap = _greedy_nonoverlap_mask(frame)
    phases = {
        str(phase): np.asarray(frame["window_number"], dtype=np.int64) % 8 == phase
        for phase in range(8)
    }
    strata = _strata(frame)
    csv_rows: list[dict[str, Any]] = []
    candidates: dict[str, Any] = {}
    for candidate, field in CANDIDATES.items():
        prediction = np.asarray(frame[field], float)
        full = _metric_summary(target, prediction, identity)
        greedy = _metric_summary(
            target[nonoverlap], prediction[nonoverlap], identity[nonoverlap]
        )
        phase_results = {
            phase: _metric_summary(target[mask], prediction[mask], identity[mask])
            for phase, mask in phases.items()
        }
        stratum_results: dict[str, Any] = {}
        for kind, groups in strata.items():
            stratum_results[kind] = {
                name: _metric_summary(target[mask], prediction[mask], identity[mask])
                for name, mask in groups.items()
            }
        candidates[candidate] = {
            "full": full,
            "greedy_nonoverlap_32s": greedy,
            "eight_fixed_window_phases": phase_results,
            "strata": stratum_results,
        }
        csv_rows.append(
            _metric_csv_record(
                seed=seed,
                candidate=candidate,
                scope="full",
                stratum_type="all",
                stratum="all",
                metrics=full,
            )
        )
        csv_rows.append(
            _metric_csv_record(
                seed=seed,
                candidate=candidate,
                scope="greedy_nonoverlap_32s",
                stratum_type="all",
                stratum="all",
                metrics=greedy,
            )
        )
        for phase, metrics in phase_results.items():
            csv_rows.append(
                _metric_csv_record(
                    seed=seed,
                    candidate=candidate,
                    scope=f"fixed_window_phase_{phase}",
                    stratum_type="window_phase_mod_8",
                    stratum=phase,
                    metrics=metrics,
                )
            )
        for kind, groups in stratum_results.items():
            for name, metrics in groups.items():
                csv_rows.append(
                    _metric_csv_record(
                        seed=seed,
                        candidate=candidate,
                        scope="stratum",
                        stratum_type=kind,
                        stratum=name,
                        metrics=metrics,
                    )
                )
    comparisons: dict[str, Any] = {}
    for name, (challenger, reference) in PAIRED_COMPARISONS.items():
        summary = _paired_summary(
            target,
            np.asarray(frame[CANDIDATES[challenger]], float),
            np.asarray(frame[CANDIDATES[reference]], float),
            identity,
        )
        comparisons[name] = {
            "challenger": challenger,
            "reference": reference,
            **summary,
        }
        csv_rows.append(_delta_csv_record(seed=seed, comparison=name, metrics=summary))
    phase_nonoverlap = {
        phase: _intervals_nonoverlap(frame, mask) for phase, mask in phases.items()
    }
    report = {
        "seed": seed,
        "seed_evaluated_independently": True,
        "cross_seed_pooling_performed": False,
        "candidates": candidates,
        "paired_candidate_deltas": comparisons,
        "locked_final_goal": _gate_decision(candidates["locked_final"]["full"]),
        "nonoverlap_audit": {
            "greedy_method": "deterministic maximal-by-earliest-end within session",
            "greedy_rows": int(nonoverlap.sum()),
            "greedy_intervals_nonoverlapping": _intervals_nonoverlap(frame, nonoverlap),
            "fixed_phase_definition": "window_number modulo 8",
            "fixed_phase_count": len(phases),
            "fixed_phase_rows": {phase: int(mask.sum()) for phase, mask in phases.items()},
            "fixed_phase_intervals_nonoverlapping": phase_nonoverlap,
            "all_eight_fixed_phases_reported": set(phases) == {str(value) for value in range(8)},
        },
        "identity_cluster_bootstrap": _bootstrap_seed_report(
            frame,
            seed=seed,
            samples=bootstrap_samples,
            base_seed=bootstrap_seed,
            confidence=bootstrap_confidence,
        ),
        "uncertainty_diagnostics": _uncertainty_diagnostics(frame),
    }
    return report, csv_rows


def _temporary_path(destination: Path) -> Path:
    return destination.with_name(
        f".{destination.name}.{os.getpid()}.{secrets.token_hex(8)}.tmp"
    )


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    payload = json.dumps(
        value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False
    ) + "\n"
    with path.open("x", encoding="utf-8") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def _format_csv(value: Any) -> str:
    if value is None or value == "":
        return ""
    if isinstance(value, (bool, np.bool_)):
        return "true" if bool(value) else "false"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        number = float(value)
        if not math.isfinite(number):
            raise LockedPrimaryEvaluationError("metrics CSV contains a non-finite number")
        return np.format_float_positional(number, unique=True, trim="-")
    return str(value)


def _write_csv(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(CSV_COLUMNS), lineterminator="\n")
        writer.writeheader()
        for row in rows:
            if set(row) != set(CSV_COLUMNS):
                raise LockedPrimaryEvaluationError("metrics CSV row schema is inconsistent")
            writer.writerow({name: _format_csv(row[name]) for name in CSV_COLUMNS})
        stream.flush()
        os.fsync(stream.fileno())


def _publish_exclusive(temporary: Path, destination: Path) -> None:
    try:
        os.link(temporary, destination)
    except FileExistsError as exc:
        raise LockedPrimaryEvaluationError(
            f"immutable evaluation output already exists: {destination}"
        ) from exc


def evaluate_locked_oof(
    *,
    locked_oof_root: Path,
    evaluation_lock: Path,
    target_receipt: Path,
    evaluation_spec: Path,
    output_dir: Path,
    report_output: Path,
    csv_output: Path,
    receipt_output: Path,
    expected_rows: int = EXPECTED_VALID_REFERENCE_ROWS,
    expected_identities: int = EXPECTED_IDENTITIES,
    bootstrap_samples: int = BOOTSTRAP_SAMPLES,
    bootstrap_seed: int = BOOTSTRAP_SEED,
    bootstrap_confidence: float = BOOTSTRAP_CONFIDENCE,
    orchestrator_command: Sequence[str] = (),
) -> dict[str, Any]:
    """Validate, evaluate fixed seeds independently, and publish immutable evidence."""

    integer_options = {
        "expected_rows": expected_rows,
        "expected_identities": expected_identities,
        "bootstrap_samples": bootstrap_samples,
        "bootstrap_seed": bootstrap_seed,
    }
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value < (0 if name == "bootstrap_seed" else 1)
        for name, value in integer_options.items()
    ):
        raise LockedPrimaryEvaluationError("integer evaluation options are invalid")
    if not 0.0 < bootstrap_confidence < 1.0:
        raise LockedPrimaryEvaluationError("bootstrap confidence must lie in (0,1)")
    output_root = output_dir.expanduser().resolve()
    report_path = report_output.expanduser().resolve()
    csv_path = csv_output.expanduser().resolve()
    receipt_path = receipt_output.expanduser().resolve()
    if len({report_path, csv_path, receipt_path}) != 3 or any(
        path.parent != output_root for path in (report_path, csv_path, receipt_path)
    ):
        raise LockedPrimaryEvaluationError(
            "report, CSV, and receipt must be distinct direct children of --output-dir"
        )
    if any(path.exists() for path in (report_path, csv_path, receipt_path)):
        raise LockedPrimaryEvaluationError("immutable evaluation output already exists")

    # The immutable, target-independent protocol is the first input opened.
    # A missing, altered, or non-canonical spec therefore fails before the
    # evaluation lock can authorize any target-bearing artifact access.
    spec, spec_binding = _load_evaluation_spec(evaluation_spec)
    population_spec = spec["population"]
    bootstrap_spec = spec["bootstrap"]
    if (
        expected_rows != population_spec["valid_reference_rows_per_seed"]
        or expected_identities != population_spec["physical_identity_count"]
        or bootstrap_samples != bootstrap_spec["samples"]
        or bootstrap_seed != bootstrap_spec["base_seed"]
        or not math.isclose(
            float(bootstrap_confidence),
            float(bootstrap_spec["confidence"]),
            rel_tol=0.0,
            abs_tol=0.0,
        )
    ):
        raise LockedPrimaryEvaluationError(
            "runtime evaluation options differ from the immutable evaluation specification"
        )
    frames, context = _validate_context(
        locked_oof_root=locked_oof_root,
        evaluation_lock=evaluation_lock,
        target_receipt=target_receipt,
        expected_rows=expected_rows,
        expected_identities=expected_identities,
        expected_seeds=population_spec["fixed_seeds"],
    )
    context["bindings"]["evaluation_spec"] = spec_binding
    context["evaluation_spec_verified_before_any_target_bearing_artifact_access"] = True
    seed_reports: dict[str, Any] = {}
    csv_rows: list[dict[str, Any]] = []
    for seed in context["seeds"]:
        seed_report, records = _evaluate_seed(
            frames[seed],
            seed=seed,
            bootstrap_samples=bootstrap_samples,
            bootstrap_seed=bootstrap_seed,
            bootstrap_confidence=bootstrap_confidence,
        )
        seed_reports[str(seed)] = seed_report
        csv_rows.extend(records)
    all_seed_gates = {
        str(seed): seed_reports[str(seed)]["locked_final_goal"]["all_point_gates_passed"]
        for seed in context["seeds"]
    }
    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "classification": "retrospective_locked_hcs_oof_primary_evaluation",
        "commercial_claim_authorized": False,
        "commercial_performance_proven": False,
        "prospective_confirmation_required": True,
        "independent_prospective_cohort_evaluated": False,
        "selection_or_retraining_performed": False,
        "seed_ranking_or_suppression_performed": False,
        "cross_seed_pooling_performed": False,
        "calibration_fit_performed": False,
        "uncertainty_used_for_model_or_threshold_selection": False,
        "goal_targets": GOAL_TARGETS,
        "evaluation_specification": spec_binding,
        "all_prespecified_fixed_seeds_must_pass": True,
        "all_fixed_seeds_point_gates_passed": bool(all(all_seed_gates.values())),
        "fixed_seed_gate_status": all_seed_gates,
        "provenance_audit": context,
        "per_seed": seed_reports,
        "orchestrator_command": list(orchestrator_command),
    }
    report["content_sha256"] = canonical_json_sha256(report)

    output_root.mkdir(parents=True, exist_ok=True)
    report_tmp = _temporary_path(report_path)
    csv_tmp = _temporary_path(csv_path)
    receipt_tmp = _temporary_path(receipt_path)
    try:
        _write_json(report_tmp, report)
        _write_csv(csv_tmp, csv_rows)
        output_bindings = {
            "report": {
                "path": str(report_path),
                "sha256": sha256_file(report_tmp),
                "bytes": report_tmp.stat().st_size,
            },
            "metrics_csv": {
                "path": str(csv_path),
                "sha256": sha256_file(csv_tmp),
                "bytes": csv_tmp.stat().st_size,
            },
        }
        receipt: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "classification": "retrospective_locked_hcs_oof_primary_evaluation_receipt",
            "commercial_claim_authorized": False,
            "commercial_performance_proven": False,
            "prospective_confirmation_required": True,
            "independent_prospective_cohort_evaluated": False,
            "outputs_create_once": True,
            "output_overwrite_allowed": False,
            "all_declared_input_artifacts_rehashed_before_metric_evaluation": True,
            "target_access_authorized_by_validated_evaluation_lock": True,
            "one_report_per_prespecified_seed_no_pooling": True,
            "bootstrap_spec_fixed_before_metric_computation": True,
            "calibration_fit_performed": False,
            "inputs": context["bindings"],
            "outputs": output_bindings,
            "metrics_csv_rows": len(csv_rows),
            "seeds": context["seeds"],
            "orchestrator_command": list(orchestrator_command),
        }
        receipt["content_sha256"] = canonical_json_sha256(receipt)
        _write_json(receipt_tmp, receipt)

        # A crash between links is a fail-closed partial publication; no file is
        # replaced, and a subsequent run refuses to overwrite the evidence.
        _publish_exclusive(report_tmp, report_path)
        report_path.chmod(0o444)
        _publish_exclusive(csv_tmp, csv_path)
        csv_path.chmod(0o444)
        _publish_exclusive(receipt_tmp, receipt_path)
        receipt_path.chmod(0o444)
    finally:
        for temporary in (report_tmp, csv_tmp, receipt_tmp):
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass
    published = _read_json(
        receipt_path, "published primary evaluation receipt", require_content_hash=True
    )
    if bind_file(report_path) != published["outputs"]["report"] or bind_file(csv_path) != published[
        "outputs"
    ]["metrics_csv"]:
        raise LockedPrimaryEvaluationError("published output differs from immutable receipt")
    return published


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="mode", required=True)
    freeze = subparsers.add_parser(
        "freeze-spec",
        help="create the immutable target-independent evaluation protocol before inference",
    )
    freeze.add_argument("--output", type=Path, default=DEFAULT_EVALUATION_SPEC)
    freeze.add_argument("--expected-rows", type=int, default=EXPECTED_VALID_REFERENCE_ROWS)
    freeze.add_argument("--expected-identities", type=int, default=EXPECTED_IDENTITIES)
    freeze.add_argument("--bootstrap-samples", type=int, default=BOOTSTRAP_SAMPLES)
    freeze.add_argument("--bootstrap-seed", type=int, default=BOOTSTRAP_SEED)
    freeze.add_argument("--bootstrap-confidence", type=float, default=BOOTSTRAP_CONFIDENCE)

    evaluate = subparsers.add_parser(
        "evaluate", help="evaluate the single locked join under the frozen protocol"
    )
    evaluate.add_argument("--locked-oof-root", type=Path, default=DEFAULT_LOCKED_ROOT)
    evaluate.add_argument("--evaluation-lock", type=Path)
    evaluate.add_argument("--target-receipt", type=Path)
    evaluate.add_argument("--evaluation-spec", type=Path, default=DEFAULT_EVALUATION_SPEC)
    evaluate.add_argument("--output-dir", type=Path)
    evaluate.add_argument("--report-output", type=Path)
    evaluate.add_argument("--csv-output", type=Path)
    evaluate.add_argument("--receipt-output", type=Path)
    evaluate.add_argument("--expected-rows", type=int, default=EXPECTED_VALID_REFERENCE_ROWS)
    evaluate.add_argument("--expected-identities", type=int, default=EXPECTED_IDENTITIES)
    evaluate.add_argument("--bootstrap-samples", type=int, default=BOOTSTRAP_SAMPLES)
    evaluate.add_argument("--bootstrap-seed", type=int, default=BOOTSTRAP_SEED)
    evaluate.add_argument("--bootstrap-confidence", type=float, default=BOOTSTRAP_CONFIDENCE)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.mode == "freeze-spec":
        try:
            result = freeze_evaluation_spec(
                args.output,
                expected_rows=args.expected_rows,
                expected_identities=args.expected_identities,
                bootstrap_samples=args.bootstrap_samples,
                bootstrap_seed=args.bootstrap_seed,
                bootstrap_confidence=args.bootstrap_confidence,
            )
        except (LockedPrimaryEvaluationError, OSError, ValueError) as exc:
            parser.error(str(exc))
        print(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False))
        return 0

    root = args.locked_oof_root.expanduser().resolve()
    evaluation_lock = args.evaluation_lock or root / "evaluation_lock.json"
    target_receipt = args.target_receipt or root / "canonical_locked_hcs_targets_receipt.json"
    output_dir = (args.output_dir or root / "primary_evaluation").expanduser().resolve()
    report_output = args.report_output or output_dir / "locked_hcs_primary_evaluation.json"
    csv_output = args.csv_output or output_dir / "locked_hcs_primary_metrics.csv"
    receipt_output = args.receipt_output or output_dir / "locked_hcs_primary_evaluation_receipt.json"
    command = [
        str(Path(sys.executable).resolve()),
        str(Path(__file__).resolve()),
        *(argv or sys.argv[1:]),
    ]
    try:
        result = evaluate_locked_oof(
            locked_oof_root=root,
            evaluation_lock=evaluation_lock,
            target_receipt=target_receipt,
            evaluation_spec=args.evaluation_spec,
            output_dir=output_dir,
            report_output=report_output,
            csv_output=csv_output,
            receipt_output=receipt_output,
            expected_rows=args.expected_rows,
            expected_identities=args.expected_identities,
            bootstrap_samples=args.bootstrap_samples,
            bootstrap_seed=args.bootstrap_seed,
            bootstrap_confidence=args.bootstrap_confidence,
            orchestrator_command=command,
        )
    except (LockedPrimaryEvaluationError, OSError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
