#!/usr/bin/env python3
"""Evaluate the pre-test-frozen HCS uncertainty protocol, without refitting.

The ordering in this evaluator is intentional and security relevant.  It first
validates the immutable primary specification, the complete cross-fitted
calibration, the label-free prediction seal, and the target-free uncertainty
seal (including every declared file and array hash).  Only after that boundary
has succeeded is the canonical evaluation lock passed to the primary evaluator
to authorize opening the target and joined OOF artifacts.

This is a retrospective evaluator, not a selector.  It never fits a scale,
chooses a threshold, ranks seeds, or changes a point prediction.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import secrets
from statistics import NormalDist
import sys
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = PROJECT_ROOT / "artifacts/runs/harmonic_candidate_set_snn_v2/hcs_locked_oof"
DEFAULT_SPEC = (
    PROJECT_ROOT
    / "artifacts/campaigns/harmonic_candidate_set_snn_v2/locked_primary_evaluation_spec.json"
)
DEFAULT_UNCERTAINTY_EVALUATION_SPEC = (
    PROJECT_ROOT
    / "artifacts/campaigns/harmonic_candidate_set_snn_v2/locked_uncertainty_evaluation_spec.json"
)
DEFAULT_CALIBRATION = (
    PROJECT_ROOT
    / "artifacts/campaigns/harmonic_candidate_set_snn_v2/nested_proposer/current_source_merged/uncertainty_calibration.json"
)

SCHEMA_VERSION = 1
FOLDS = tuple(range(6))
INTERVAL_COVERAGES = (0.50, 0.80, 0.90, 0.95)
SELECTIVE_COVERAGES = (0.50, 0.80, 0.90, 1.00)
PHASES = tuple(range(8))
STD_FLOOR_BPM = 0.25
EXPECTED_UNCERTAINTY_FIELDS = frozenset(
    {
        "cache_index",
        "outer_fold",
        "seed",
        "final_rr_bpm",
        "fallback_std_bpm",
        "source_scale_bpm",
        "selected_probability",
        "margin",
        "normalized_entropy",
        "quality",
        "valid_candidate_count",
        "fallback_available",
        "source_available",
    }
)
GATE_NAMES = frozenset(
    {
        "all_seeds_required",
        "conformal_max_absolute_calibration_error_all_levels",
        "conformal_90_marginal_coverage_min",
        "conformal_90_identity_macro_coverage_min",
        "conformal_90_fixed_phase_0_coverage_min",
        "conformal_90_mean_full_width_bpm_max",
        "conformal_90_p95_full_width_bpm_max",
        "selective_80_mae_bpm_max",
        "selective_80_catastrophic_over_5_max",
    }
)
CSV_COLUMNS = (
    "record_type",
    "seed",
    "method",
    "nominal_coverage",
    "intended_acceptance_coverage",
    "scope",
    "phase",
    "rows",
    "identities",
    "coverage",
    "identity_macro_coverage",
    "worst_identity_coverage",
    "mean_full_width_bpm",
    "p95_full_width_bpm",
    "calibration_error",
    "achieved_coverage",
    "mae",
    "identity_macro_mae",
    "rmse",
    "within_2_fraction",
    "over_5_fraction",
    "tail_25_35_mae",
    "gate_name",
    "operator",
    "threshold",
    "observed",
    "passed",
)


class LockedUncertaintyEvaluationError(RuntimeError):
    """A frozen-input, authorization, alignment, or publication check failed."""


def _load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise LockedUncertaintyEvaluationError(f"cannot import required evaluator: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


PRIMARY = _load_module(
    "evaluate_locked_hcs_oof_for_uncertainty",
    PROJECT_ROOT / "scripts/evaluate_locked_hcs_oof.py",
)
SEALER = _load_module(
    "seal_locked_hcs_uncertainty_inputs_for_evaluation",
    PROJECT_ROOT / "scripts/seal_locked_hcs_uncertainty_inputs.py",
)
FREEZER = _load_module(
    "freeze_locked_hcs_uncertainty_evaluation_spec_for_evaluation",
    PROJECT_ROOT / "scripts/freeze_locked_hcs_uncertainty_evaluation_spec.py",
)


def _fail(message: str) -> None:
    raise LockedUncertaintyEvaluationError(message)


def _read_json(path: Path, label: str, *, content_hash: bool = False) -> dict[str, Any]:
    try:
        return PRIMARY._read_json(path, label, require_content_hash=content_hash)
    except RuntimeError as exc:
        raise LockedUncertaintyEvaluationError(str(exc)) from exc


def _verify_binding(raw: Any, *, relative_to: Path, label: str) -> dict[str, Any]:
    try:
        return PRIMARY._verify_binding(raw, relative_to=relative_to, label=label)
    except RuntimeError as exc:
        raise LockedUncertaintyEvaluationError(str(exc)) from exc


def _verify_tree(value: Any, *, relative_to: Path, label: str) -> list[dict[str, Any]]:
    try:
        return PRIMARY._verify_binding_tree(value, relative_to=relative_to, label=label)
    except RuntimeError as exc:
        raise LockedUncertaintyEvaluationError(str(exc)) from exc


def _same_binding(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    try:
        return bool(PRIMARY._same_binding(left, right))
    except (KeyError, OSError, TypeError, ValueError):
        return False


def _finite_number(value: Any, *, label: str, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _fail(f"{label} must be a finite number")
    result = float(value)
    if not math.isfinite(result) or (minimum is not None and result < minimum):
        _fail(f"{label} is out of range")
    return result


def _validate_calibration(
    path: Path, *, expected_seeds: Sequence[int]
) -> tuple[dict[str, Any], dict[str, Any], int]:
    resolved = path.expanduser().resolve()
    calibration = _read_json(resolved, "pretest uncertainty calibration", content_hash=True)
    expected_keys = {(fold, int(seed)) for seed in expected_seeds for fold in FOLDS}
    if (
        calibration.get("schema_version") != SCHEMA_VERSION
        or calibration.get("classification")
        != "locked_pretest_cross_fitted_proposer_uncertainty_calibration"
        or calibration.get("commercial_claim_authorized") is not False
        or calibration.get("prospective_confirmation_required") is not True
        or calibration.get("outer_test_opened") is not False
        or calibration.get("outer_test_record_count") != 0
        or calibration.get("target_artifact_opened") is not False
        or calibration.get("point_prediction_modified") is not False
        or calibration.get("folds") != list(FOLDS)
        or calibration.get("seeds") != [int(seed) for seed in expected_seeds]
        or calibration.get("unit_count") != len(expected_keys)
    ):
        _fail("pretest uncertainty calibration invariants are invalid")
    method = calibration.get("fixed_method")
    if (
        not isinstance(method, Mapping)
        or method.get("phase_modulus") != 8
        or method.get("phase_value") != 0
        or float(method.get("std_floor_bpm", -1.0)) != STD_FLOOR_BPM
        or method.get("interval_coverages") != list(INTERVAL_COVERAGES)
        or method.get("selective_coverages") != list(SELECTIVE_COVERAGES)
        or method.get("no_test_time_fit_or_threshold_selection") is not True
        or method.get("formal_exchangeability_claim") is not False
    ):
        _fail("pretest calibration fixed method differs from the evaluator protocol")
    gates = calibration.get("fixed_evaluation_gates")
    if not isinstance(gates, Mapping) or set(gates) != GATE_NAMES:
        _fail("pretest calibration fixed gates are incomplete or changed")
    if gates.get("all_seeds_required") is not True:
        _fail("pretest calibration does not require every fixed seed")
    for name in GATE_NAMES - {"all_seeds_required"}:
        _finite_number(gates[name], label=f"fixed gate {name}", minimum=0.0)
    units = calibration.get("units")
    if not isinstance(units, list) or len(units) != len(expected_keys):
        _fail("pretest calibration unit list is incomplete")
    observed: set[tuple[int, int]] = set()
    for position, unit in enumerate(units):
        if not isinstance(unit, Mapping):
            _fail("pretest calibration unit must be an object")
        try:
            key = (int(unit["outer_fold"]), int(unit["seed"]))
        except (KeyError, TypeError, ValueError) as exc:
            raise LockedUncertaintyEvaluationError("invalid calibration unit identity") from exc
        if key not in expected_keys or key in observed:
            _fail(f"duplicate or unexpected calibration unit: {key}")
        observed.add(key)
        if (
            unit.get("source_unit_count") != 5
            or not isinstance(unit.get("source_units"), list)
            or len(unit["source_units"]) != 5
            or unit.get("source_identity_count") != 15
            or not isinstance(unit.get("source_identities"), list)
            or len(set(map(str, unit["source_identities"]))) != 15
            or int(unit.get("source_rows_valid_phase_0", 0)) < 1
            or float(unit.get("std_floor_bpm", -1.0)) != STD_FLOOR_BPM
        ):
            _fail(f"calibration source topology is invalid: {key}")
        source_names: set[str] = set()
        for source_position, source in enumerate(unit["source_units"]):
            if not isinstance(source, Mapping):
                _fail(f"calibration source unit must be an object: {key}/{source_position}")
            source_name = str(source.get("name", ""))
            if not source_name or source_name in source_names or not str(source.get("role", "")):
                _fail(f"calibration source unit identity is invalid: {key}/{source_position}")
            source_names.add(source_name)
            for artifact in ("manifest", "checkpoint", "all_window_prediction"):
                _verify_binding(
                    source.get(artifact),
                    relative_to=resolved.parent,
                    label=f"calibration {key}/{source_name}/{artifact}",
                )
        intervals = unit.get("interval_calibration")
        thresholds = unit.get("selective_thresholds")
        if not isinstance(intervals, Mapping) or set(intervals) != {
            f"{value:.2f}" for value in INTERVAL_COVERAGES
        }:
            _fail(f"calibration intervals are incomplete: {key}")
        if not isinstance(thresholds, Mapping) or set(thresholds) != {
            f"{value:.2f}" for value in SELECTIVE_COVERAGES
        }:
            _fail(f"selective thresholds are incomplete: {key}")
        for coverage in INTERVAL_COVERAGES:
            entry = intervals[f"{coverage:.2f}"]
            if (
                not isinstance(entry, Mapping)
                or float(entry.get("nominal_coverage", -1.0)) != coverage
            ):
                _fail(f"calibration interval identity differs: {key}/{coverage}")
            _finite_number(
                entry.get("normalized_absolute_error_quantile"),
                label=f"conformal quantile {key}/{coverage}",
                minimum=0.0,
            )
        for coverage in SELECTIVE_COVERAGES:
            entry = thresholds[f"{coverage:.2f}"]
            if (
                not isinstance(entry, Mapping)
                or float(entry.get("intended_acceptance_coverage", -1.0)) != coverage
            ):
                _fail(f"selective threshold identity differs: {key}/{coverage}")
            _finite_number(
                entry.get("rr_std_threshold_bpm"),
                label=f"selective threshold {key}/{coverage}",
                minimum=0.0,
            )
    if observed != expected_keys:
        _fail("pretest calibration is not the exact fold/seed Cartesian product")
    verified = _verify_tree(calibration, relative_to=resolved.parent, label="calibration")
    return calibration, PRIMARY.bind_file(resolved), len(verified)


def _validate_predictions_seal(
    path: Path, *, root: Path, expected_seeds: Sequence[int]
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    dict[tuple[int, int], dict[str, np.ndarray]],
    int,
]:
    resolved = path.expanduser().resolve()
    seal = _read_json(resolved, "primary predictions seal")
    units = seal.get("units")
    expected_keys = {(fold, int(seed)) for seed in expected_seeds for fold in FOLDS}
    if (
        seal.get("schema_version") != SCHEMA_VERSION
        or seal.get("classification") != "locked_hcs_oof_all_label_free_predictions_sealed"
        or seal.get("target_artifact_opened_before_seal") is not False
        or seal.get("target_join_authorized") is not True
        or seal.get("unit_count") != len(expected_keys)
        or seal.get("outer_folds") != list(FOLDS)
        or not isinstance(units, list)
        or len(units) != len(expected_keys)
    ):
        _fail("primary predictions seal is incomplete or invalid")
    verified = _verify_tree(seal, relative_to=resolved.parent, label="predictions seal")
    expected_pretest = str(seal.get("pretest_lock_sha256", "")).lower()
    pretest_path = root / "pretest_lock.json"
    if len(expected_pretest) != 64 or PRIMARY.sha256_file(pretest_path) != expected_pretest:
        _fail("predictions seal/pretest lock hash mismatch")
    pretest = _read_json(pretest_path, "pretest lock")
    verified.extend(_verify_tree(pretest, relative_to=pretest_path.parent, label="pretest lock"))
    observed: set[tuple[int, int]] = set()
    unit_arrays: dict[tuple[int, int], dict[str, np.ndarray]] = {}
    for raw in units:
        if not isinstance(raw, Mapping):
            _fail("predictions seal unit must be an object")
        try:
            key = (int(raw["outer_fold"]), int(raw["seed"]))
        except (KeyError, TypeError, ValueError) as exc:
            raise LockedUncertaintyEvaluationError("invalid prediction unit identity") from exc
        if key not in expected_keys or key in observed:
            _fail(f"duplicate or unexpected prediction unit: {key}")
        observed.add(key)
        try:
            arrays, record = SEALER._load_unit(raw, root=root)
        except RuntimeError as exc:
            raise LockedUncertaintyEvaluationError(str(exc)) from exc
        if (int(record["outer_fold"]), int(record["seed"])) != key:
            _fail("prediction unit identity changed while being reopened")
        unit_arrays[key] = arrays
        derived_binding = _verify_binding(
            raw.get("derived_lock"), relative_to=resolved.parent, label=f"derived lock {key}"
        )
        derived_path = Path(derived_binding["path"])
        derived = _read_json(derived_path, f"derived lock {key}")
        verified.extend(
            _verify_tree(derived, relative_to=derived_path.parent, label=f"derived lock {key}")
        )
        receipts = derived.get("stage_receipts", [])
        if not isinstance(receipts, list):
            _fail(f"derived stage receipt topology is invalid: {key}")
        for receipt_position, receipt_raw in enumerate(receipts):
            receipt_binding = _verify_binding(
                receipt_raw,
                relative_to=derived_path.parent,
                label=f"stage receipt {key}/{receipt_position}",
            )
            receipt_path = Path(receipt_binding["path"])
            receipt = _read_json(receipt_path, f"stage receipt {key}/{receipt_position}")
            if receipt.get("classification") != "locked_hcs_oof_stage_receipt":
                _fail(f"stage receipt invariant is invalid: {key}/{receipt_position}")
            verified.extend(
                _verify_tree(
                    receipt,
                    relative_to=receipt_path.parent,
                    label=f"stage receipt {key}/{receipt_position}",
                )
            )
    if observed != expected_keys:
        _fail("predictions seal is not the exact fold/seed Cartesian product")
    return seal, PRIMARY.bind_file(resolved), unit_arrays, len(verified)


def _validate_uncertainty_seal(
    path: Path,
    *,
    calibration_binding: Mapping[str, Any],
    predictions_binding: Mapping[str, Any],
    prediction_unit_arrays: Mapping[tuple[int, int], Mapping[str, np.ndarray]],
    expected_seeds: Sequence[int],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, np.ndarray], int]:
    resolved = path.expanduser().resolve()
    seal = _read_json(resolved, "uncertainty input seal", content_hash=True)
    units = seal.get("units")
    expected_keys = {(fold, int(seed)) for seed in expected_seeds for fold in FOLDS}
    if (
        seal.get("schema_version") != SCHEMA_VERSION
        or seal.get("classification") != "locked_hcs_all_target_free_uncertainty_inputs_sealed"
        or seal.get("commercial_claim_authorized") is not False
        or seal.get("prospective_confirmation_required") is not True
        or seal.get("target_artifact_opened_before_seal") is not False
        or seal.get("target_fields_present") is not False
        or seal.get("point_prediction_modified") is not False
        or seal.get("no_action_primary_bit_exact_verified") is not True
        or seal.get("unit_count") != len(FOLDS) * len(expected_seeds)
        or seal.get("folds") != list(FOLDS)
        or seal.get("seeds") != [int(seed) for seed in expected_seeds]
        or not isinstance(units, list)
        or len(units) != len(expected_keys)
    ):
        _fail("uncertainty input seal invariants are invalid")
    seal_calibration = _verify_binding(
        seal.get("pretest_calibration"), relative_to=resolved.parent, label="sealed calibration"
    )
    seal_predictions = _verify_binding(
        seal.get("predictions_seal"), relative_to=resolved.parent, label="sealed predictions"
    )
    if not _same_binding(seal_calibration, calibration_binding):
        _fail("uncertainty seal and evaluator bind different calibrations")
    if not _same_binding(seal_predictions, predictions_binding):
        _fail("uncertainty seal and evaluator bind different prediction seals")
    verified = _verify_tree(seal, relative_to=resolved.parent, label="uncertainty seal")
    archive_binding = _verify_binding(
        seal.get("uncertainty_archive"), relative_to=resolved.parent, label="uncertainty archive"
    )
    try:
        arrays = PRIMARY._load_npz(Path(archive_binding["path"]), label="uncertainty archive")
    except RuntimeError as exc:
        raise LockedUncertaintyEvaluationError(str(exc)) from exc
    if set(arrays) != EXPECTED_UNCERTAINTY_FIELDS:
        _fail("uncertainty archive field set differs from the sealed protocol")
    observed_units: set[tuple[int, int]] = set()
    declared_unit_rows = 0
    for unit in units:
        if not isinstance(unit, Mapping):
            _fail("uncertainty seal unit must be an object")
        try:
            key = (int(unit["outer_fold"]), int(unit["seed"]))
            unit_rows = int(unit["rows"])
        except (KeyError, TypeError, ValueError) as exc:
            raise LockedUncertaintyEvaluationError("invalid uncertainty unit identity") from exc
        if key not in expected_keys or key in observed_units or unit_rows < 1:
            _fail(f"duplicate or invalid uncertainty seal unit: {key}")
        observed_units.add(key)
        declared_unit_rows += unit_rows
        if key not in prediction_unit_arrays or unit_rows != len(
            prediction_unit_arrays[key]["cache_index"]
        ):
            _fail(f"uncertainty seal unit row binding differs: {key}")
        for artifact in ("derived_lock", "raw_hcs_prediction", "sealed_prediction"):
            _verify_binding(
                unit.get(artifact),
                relative_to=resolved.parent,
                label=f"uncertainty seal {key}/{artifact}",
            )
    if observed_units != expected_keys:
        _fail("uncertainty seal unit topology is not the exact fold/seed product")
    schema = seal.get("array_schema")
    if not isinstance(schema, Mapping) or set(schema) != set(arrays):
        _fail("uncertainty seal array schema is incomplete")
    rows = len(np.asarray(arrays["cache_index"]))
    if rows < 1 or seal.get("row_count") != rows or any(
        np.asarray(value).shape != (rows,) for value in arrays.values()
    ) or declared_unit_rows != rows:
        _fail("uncertainty archive row topology is invalid")
    for name, value in arrays.items():
        array = np.asarray(value)
        declared = schema.get(name)
        if not isinstance(declared, Mapping) or (
            declared.get("dtype") != array.dtype.str
            or declared.get("shape") != list(array.shape)
            or declared.get("sha256") != PRIMARY.array_sha256(array)
        ):
            _fail(f"uncertainty array schema/hash mismatch: {name}")
    index = np.asarray(arrays["cache_index"])
    fold = np.asarray(arrays["outer_fold"])
    seed = np.asarray(arrays["seed"])
    if index.dtype.kind not in "iu" or fold.dtype.kind not in "iu" or seed.dtype.kind not in "iu":
        _fail("uncertainty index/fold/seed fields must be integer vectors")
    float_fields = (
        "final_rr_bpm",
        "fallback_std_bpm",
        "source_scale_bpm",
        "selected_probability",
        "margin",
        "normalized_entropy",
        "quality",
    )
    for name in float_fields:
        value = np.asarray(arrays[name])
        if value.dtype.kind not in "fc" or not np.isfinite(value).all():
            _fail(f"uncertainty field is not finite floating point: {name}")
    if (
        (arrays["final_rr_bpm"] <= 0).any()
        or (arrays["fallback_std_bpm"] < 0).any()
        or (arrays["source_scale_bpm"] < 0).any()
        or (arrays["selected_probability"] < 0).any()
        or (arrays["selected_probability"] > 1).any()
        or (arrays["normalized_entropy"] < 0).any()
        or (arrays["valid_candidate_count"] < 0).any()
        or np.asarray(arrays["fallback_available"]).dtype.kind != "b"
        or np.asarray(arrays["source_available"]).dtype.kind != "b"
    ):
        _fail("uncertainty values violate their sealed domains")
    expected_index: np.ndarray | None = None
    rows_per_seed = int(seal.get("rows_per_seed", -1))
    for seed_value in expected_seeds:
        selected = np.flatnonzero(seed.astype(np.int64) == int(seed_value))
        if len(selected) != rows_per_seed:
            _fail(f"uncertainty rows-per-seed differs: {seed_value}")
        current = index[selected].astype(np.int64)
        if len(np.unique(current)) != len(current):
            _fail(f"uncertainty cache indices repeat: {seed_value}")
        order = np.argsort(current, kind="stable")
        current = current[order]
        if expected_index is None:
            expected_index = current
        elif not np.array_equal(current, expected_index):
            _fail("uncertainty cache-index coverage differs across seeds")
        if set(fold[selected].astype(int).tolist()) != set(FOLDS):
            _fail(f"uncertainty seed does not cover every outer fold: {seed_value}")
    if not np.isin(seed.astype(np.int64), np.asarray(expected_seeds, dtype=np.int64)).all():
        _fail("uncertainty archive contains an unexpected seed")
    # Reconstruct the exact sealer concatenation from all 18 independently
    # reopened raw/point units.  This proves that the archive hash does not
    # merely describe a self-consistent but unrelated array payload.
    order = [(fold, int(seed_value)) for seed_value in expected_seeds for fold in FOLDS]
    reconstructed = {
        name: np.concatenate([np.asarray(prediction_unit_arrays[key][name]) for key in order])
        for name in EXPECTED_UNCERTAINTY_FIELDS
    }
    for name in EXPECTED_UNCERTAINTY_FIELDS:
        observed_array = np.asarray(arrays[name])
        expected_array = np.asarray(reconstructed[name])
        if (
            observed_array.dtype != expected_array.dtype
            or observed_array.shape != expected_array.shape
            or observed_array.tobytes() != expected_array.tobytes()
        ):
            _fail(f"uncertainty archive differs from rederived 18-unit input: {name}")
    return seal, PRIMARY.bind_file(resolved), arrays, len(verified)


def _validate_pre_target_inputs(
    *,
    uncertainty_evaluation_spec: Path,
    evaluation_spec: Path,
    calibration_path: Path,
    predictions_seal_path: Path,
    uncertainty_seal_path: Path,
    locked_oof_root: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, np.ndarray], dict[str, Any]]:
    """Validate all target-free inputs.  Never touch predictions before the dedicated spec."""

    # This dedicated secondary protocol is the first artifact opened.  Its
    # loader rehashes the unchanged primary spec, completed calibration, and
    # every implementation source.  No prediction, uncertainty, evaluation
    # lock, target receipt, target array, or joined OOF is accessed before it.
    try:
        secondary_spec, secondary_spec_binding = FREEZER.load_uncertainty_evaluation_spec(
            uncertainty_evaluation_spec,
            expected_primary_spec_path=evaluation_spec,
            expected_calibration_path=calibration_path,
        )
    except RuntimeError as exc:
        raise LockedUncertaintyEvaluationError(str(exc)) from exc
    try:
        spec, spec_binding = PRIMARY._load_evaluation_spec(evaluation_spec)
    except RuntimeError as exc:
        raise LockedUncertaintyEvaluationError(str(exc)) from exc
    seeds = [int(value) for value in spec["population"]["fixed_seeds"]]
    calibration, calibration_binding, calibration_rehashed = _validate_calibration(
        calibration_path, expected_seeds=seeds
    )
    predictions, predictions_binding, prediction_unit_arrays, predictions_rehashed = _validate_predictions_seal(
        predictions_seal_path,
        root=locked_oof_root.expanduser().resolve(),
        expected_seeds=seeds,
    )
    uncertainty, uncertainty_binding, arrays, uncertainty_rehashed = _validate_uncertainty_seal(
        uncertainty_seal_path,
        calibration_binding=calibration_binding,
        predictions_binding=predictions_binding,
        prediction_unit_arrays=prediction_unit_arrays,
        expected_seeds=seeds,
    )
    audit = {
        "secondary_uncertainty_evaluation_spec": secondary_spec_binding,
        "evaluation_spec": spec_binding,
        "calibration": calibration_binding,
        "predictions_seal": predictions_binding,
        "uncertainty_inputs_seal": uncertainty_binding,
        "uncertainty_archive": dict(uncertainty["uncertainty_archive"]),
        "calibration_declared_bindings_rehashed": calibration_rehashed,
        "prediction_declared_bindings_rehashed": predictions_rehashed,
        "uncertainty_declared_bindings_rehashed": uncertainty_rehashed,
        "secondary_protocol_role": secondary_spec["protocol_relationship"]["role"],
        "primary_uncertainty_contract_overridden": False,
        "dedicated_secondary_spec_verified_before_prediction_uncertainty_or_target_access": True,
        "all_target_free_inputs_verified_before_evaluation_lock_access": True,
        "all_uncertainty_array_schema_and_hashes_verified": True,
    }
    return spec, calibration, predictions, arrays, audit


def _align_uncertainty(
    arrays: Mapping[str, np.ndarray], frames: Mapping[int, Mapping[str, np.ndarray]]
) -> dict[int, dict[str, np.ndarray]]:
    result: dict[int, dict[str, np.ndarray]] = {}
    all_seed = np.asarray(arrays["seed"], dtype=np.int64)
    for seed, frame in frames.items():
        selected = np.flatnonzero(all_seed == int(seed))
        raw_index = np.asarray(arrays["cache_index"], dtype=np.int64)[selected]
        order = np.argsort(raw_index, kind="stable")
        selected = selected[order]
        raw_index = raw_index[order]
        if len(np.unique(raw_index)) != len(raw_index):
            _fail(f"uncertainty rows repeat cache indices for seed {seed}")
        frame_index = np.asarray(frame["cache_index"], dtype=np.int64)
        positions = np.searchsorted(raw_index, frame_index)
        if (
            len(frame_index) == 0
            or (positions >= len(raw_index)).any()
            or not np.array_equal(raw_index[positions], frame_index)
        ):
            _fail(f"uncertainty/OOF cache-index cover differs for seed {seed}")
        rows = selected[positions]
        if not np.array_equal(
            np.asarray(arrays["outer_fold"], dtype=np.int64)[rows],
            np.asarray(frame["outer_fold"], dtype=np.int64),
        ):
            _fail(f"uncertainty/OOF fold parity failed for seed {seed}")
        sealed_final = np.asarray(arrays["final_rr_bpm"])[rows]
        joined_final = np.asarray(frame["final_rr_bpm"])
        if (
            sealed_final.dtype != joined_final.dtype
            or sealed_final.shape != joined_final.shape
            or sealed_final.tobytes() != joined_final.tobytes()
        ):
            _fail(f"uncertainty/OOF final_rr bit parity failed for seed {seed}")
        aligned = {name: np.asarray(value)[rows] for name, value in arrays.items()}
        if not np.array_equal(aligned["cache_index"].astype(np.int64), frame_index):
            _fail(f"uncertainty cache-index parity failed for seed {seed}")
        result[int(seed)] = aligned
    if set(result) != set(frames):
        _fail("uncertainty and joined OOF seed sets differ")
    return result


def _coverage_summary(
    covered: np.ndarray,
    width: np.ndarray,
    identity: np.ndarray,
    nominal: float,
    mask: np.ndarray | None = None,
) -> dict[str, Any]:
    selected = np.ones(len(covered), dtype=bool) if mask is None else np.asarray(mask, bool)
    if selected.shape != covered.shape or not selected.any():
        return {
            "rows": 0,
            "identities": 0,
            "empirical_coverage": None,
            "identity_macro_coverage": None,
            "worst_identity_coverage": None,
            "mean_full_width_bpm": None,
            "p95_full_width_bpm": None,
            "absolute_calibration_error": None,
        }
    observed_id = np.asarray(identity).astype(str)[selected]
    observed_covered = np.asarray(covered, bool)[selected]
    observed_width = np.asarray(width, float)[selected]
    by_identity = [
        float(np.mean(observed_covered[observed_id == name]))
        for name in sorted(set(observed_id.tolist()))
    ]
    coverage = float(np.mean(observed_covered))
    return {
        "rows": int(selected.sum()),
        "identities": len(by_identity),
        "empirical_coverage": coverage,
        "identity_macro_coverage": float(np.mean(by_identity)),
        "worst_identity_coverage": float(np.min(by_identity)),
        "mean_full_width_bpm": float(np.mean(observed_width)),
        "p95_full_width_bpm": float(np.quantile(observed_width, 0.95)),
        "absolute_calibration_error": abs(coverage - nominal),
    }


def _calibration_lookup(calibration: Mapping[str, Any]) -> dict[tuple[int, int], Mapping[str, Any]]:
    return {
        (int(unit["outer_fold"]), int(unit["seed"])): unit
        for unit in calibration["units"]
    }


def _interval_evaluation(
    *,
    frame: Mapping[str, np.ndarray],
    uncertainty: Mapping[str, np.ndarray],
    calibration_lookup: Mapping[tuple[int, int], Mapping[str, Any]],
    seed: int,
) -> tuple[dict[str, Any], dict[tuple[str, float], tuple[np.ndarray, np.ndarray]]]:
    target = np.asarray(frame["target_rr_bpm"], dtype=np.float64)
    prediction = np.asarray(frame["final_rr_bpm"], dtype=np.float64)
    identity = np.asarray(frame["identity"]).astype(str)
    fold = np.asarray(frame["outer_fold"], dtype=np.int64)
    phase = np.asarray(frame["window_number"], dtype=np.int64) % 8
    std = np.asarray(uncertainty["fallback_std_bpm"], dtype=np.float64)
    if not np.isfinite(std).all() or (std < 0).any():
        _fail(f"invalid aligned uncertainty scale for seed {seed}")
    error = np.abs(prediction - target)
    result: dict[str, Any] = {"normal_uncalibrated": {}, "normalized_conformal": {}}
    raw: dict[tuple[str, float], tuple[np.ndarray, np.ndarray]] = {}
    for coverage in INTERVAL_COVERAGES:
        normal_z = NormalDist().inv_cdf((1.0 + coverage) / 2.0)
        normal_half = normal_z * std
        conformal_quantile = np.empty(len(target), dtype=np.float64)
        for fold_value in FOLDS:
            unit = calibration_lookup[(fold_value, seed)]
            conformal_quantile[fold == fold_value] = float(
                unit["interval_calibration"][f"{coverage:.2f}"][
                    "normalized_absolute_error_quantile"
                ]
            )
        conformal_half = conformal_quantile * np.maximum(std, STD_FLOOR_BPM)
        for method, half_width in (
            ("normal_uncalibrated", normal_half),
            ("normalized_conformal", conformal_half),
        ):
            covered = error <= half_width
            full_width = 2.0 * half_width
            phases = {
                str(value): _coverage_summary(
                    covered, full_width, identity, coverage, phase == value
                )
                for value in PHASES
            }
            result[method][f"{coverage:.2f}"] = {
                "nominal_coverage": coverage,
                "interval_center": "locked_final_rr_bpm",
                "half_width_rule": (
                    "normal_two_sided_z_times_fallback_std_bpm"
                    if method == "normal_uncalibrated"
                    else "frozen_fold_seed_quantile_times_max_fallback_std_bpm_0.25"
                ),
                "marginal": _coverage_summary(
                    covered, full_width, identity, coverage
                ),
                "fixed_phase_0": phases["0"],
                "fixed_phases": phases,
            }
            raw[(method, coverage)] = (covered, full_width)
    return result, raw


def _selective_evaluation(
    *,
    frame: Mapping[str, np.ndarray],
    uncertainty: Mapping[str, np.ndarray],
    calibration_lookup: Mapping[tuple[int, int], Mapping[str, Any]],
    seed: int,
) -> tuple[dict[str, Any], dict[float, np.ndarray]]:
    fold = np.asarray(frame["outer_fold"], dtype=np.int64)
    std = np.asarray(uncertainty["fallback_std_bpm"], dtype=np.float64)
    result: dict[str, Any] = {}
    masks: dict[float, np.ndarray] = {}
    for coverage in SELECTIVE_COVERAGES:
        threshold = np.empty(len(std), dtype=np.float64)
        by_fold: dict[str, float] = {}
        for fold_value in FOLDS:
            value = float(
                calibration_lookup[(fold_value, seed)]["selective_thresholds"][
                    f"{coverage:.2f}"
                ]["rr_std_threshold_bpm"]
            )
            threshold[fold == fold_value] = value
            by_fold[str(fold_value)] = value
        accepted = std <= threshold
        masks[coverage] = accepted
        metrics = PRIMARY._metric_summary(
            np.asarray(frame["target_rr_bpm"])[accepted],
            np.asarray(frame["final_rr_bpm"])[accepted],
            np.asarray(frame["identity"])[accepted],
        )
        result[f"{coverage:.2f}"] = {
            "intended_acceptance_coverage": coverage,
            "accept_rule": "fallback_std_bpm <= frozen outer-fold/seed threshold",
            "threshold_bpm_by_outer_fold": by_fold,
            "accepted_rows": int(accepted.sum()),
            "total_rows": len(accepted),
            "achieved_coverage": float(np.mean(accepted)),
            "metrics": metrics,
        }
    return result, masks


def _percentile(values: np.ndarray, estimate: float | None, confidence: float) -> dict[str, Any] | None:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if estimate is None or not len(finite):
        return None
    alpha = (1.0 - confidence) / 2.0
    limits = np.quantile(finite, [alpha, 1.0 - alpha])
    return {
        "estimate": float(estimate),
        "lower": float(limits[0]),
        "upper": float(limits[1]),
        "confidence": confidence,
        "bootstrap_unit": "physical_identity",
        "samples_finite": int(len(finite)),
    }


def _cluster_bootstrap(
    *,
    frame: Mapping[str, np.ndarray],
    interval_raw: Mapping[tuple[str, float], tuple[np.ndarray, np.ndarray]],
    interval_report: Mapping[str, Any],
    selective_masks: Mapping[float, np.ndarray],
    selective_report: Mapping[str, Any],
    seed: int,
    samples: int,
    base_seed: int,
    confidence: float,
) -> dict[str, Any]:
    identity = np.asarray(frame["identity"]).astype(str)
    names = sorted(set(identity.tolist()))
    code = np.asarray([names.index(value) for value in identity], dtype=np.int64)
    digest = hashlib.sha256(
        f"locked-hcs-uncertainty:{base_seed}:{seed}".encode("utf-8")
    ).hexdigest()
    derived_seed = int(digest[:16], 16) % (2**32)
    rng = np.random.default_rng(derived_seed)
    weights = rng.multinomial(
        len(names), np.full(len(names), 1.0 / len(names)), size=samples
    ).astype(np.float64)
    identity_rows = np.bincount(code, minlength=len(names)).astype(np.float64)
    row_denominator = weights @ identity_rows
    interval_ci: dict[str, Any] = {"normal_uncalibrated": {}, "normalized_conformal": {}}
    for (method, coverage), (covered, width) in interval_raw.items():
        covered_by_identity = np.bincount(
            code, weights=np.asarray(covered, float), minlength=len(names)
        )
        width_by_identity = np.bincount(
            code, weights=np.asarray(width, float), minlength=len(names)
        )
        identity_coverage = covered_by_identity / identity_rows
        marginal = (weights @ covered_by_identity) / row_denominator
        macro = (weights @ identity_coverage) / len(names)
        mean_width = (weights @ width_by_identity) / row_denominator
        worst = np.asarray(
            [np.min(identity_coverage[row > 0]) for row in weights], dtype=np.float64
        )
        # Weighted p95 over row widths, in modest blocks to bound memory.
        width_order = np.argsort(np.asarray(width), kind="stable")
        ordered_width = np.asarray(width, float)[width_order]
        ordered_code = code[width_order]
        p95 = np.empty(samples, dtype=np.float64)
        for start in range(0, samples, 128):
            stop = min(samples, start + 128)
            cumulative = np.cumsum(weights[start:stop, ordered_code], axis=1)
            ranks = np.ceil(0.95 * row_denominator[start:stop])
            positions = np.argmax(cumulative >= ranks[:, None], axis=1)
            p95[start:stop] = ordered_width[positions]
        point = interval_report[method][f"{coverage:.2f}"]["marginal"]
        interval_ci[method][f"{coverage:.2f}"] = {
            "empirical_coverage": _percentile(
                marginal, point["empirical_coverage"], confidence
            ),
            "identity_macro_coverage": _percentile(
                macro, point["identity_macro_coverage"], confidence
            ),
            "worst_identity_coverage": _percentile(
                worst, point["worst_identity_coverage"], confidence
            ),
            "mean_full_width_bpm": _percentile(
                mean_width, point["mean_full_width_bpm"], confidence
            ),
            "p95_full_width_bpm": _percentile(
                p95, point["p95_full_width_bpm"], confidence
            ),
            "absolute_calibration_error": _percentile(
                np.abs(marginal - coverage),
                point["absolute_calibration_error"],
                confidence,
            ),
        }
    target = np.asarray(frame["target_rr_bpm"], float)
    prediction = np.asarray(frame["final_rr_bpm"], float)
    absolute = np.abs(prediction - target)
    selective_ci: dict[str, Any] = {}
    for coverage, accepted in selective_masks.items():
        accepted_float = np.asarray(accepted, float)
        accepted_by_identity = np.bincount(code, weights=accepted_float, minlength=len(names))
        accepted_rows = weights @ accepted_by_identity
        accepted_identity = accepted_by_identity > 0
        abs_by_identity = np.bincount(
            code, weights=accepted_float * absolute, minlength=len(names)
        )
        square_by_identity = np.bincount(
            code, weights=accepted_float * absolute**2, minlength=len(names)
        )
        within_by_identity = np.bincount(
            code, weights=accepted_float * (absolute <= 2.0), minlength=len(names)
        )
        over_by_identity = np.bincount(
            code, weights=accepted_float * (absolute > 5.0), minlength=len(names)
        )
        tail = (target >= 25.0) & (target <= 35.0)
        tail_rows_by_identity = np.bincount(
            code, weights=accepted_float * tail, minlength=len(names)
        )
        tail_abs_by_identity = np.bincount(
            code, weights=accepted_float * tail * absolute, minlength=len(names)
        )
        achieved = accepted_rows / row_denominator
        mae = np.divide(
            weights @ abs_by_identity,
            accepted_rows,
            out=np.full(samples, np.nan),
            where=accepted_rows > 0,
        )
        rmse = np.sqrt(
            np.divide(
                weights @ square_by_identity,
                accepted_rows,
                out=np.full(samples, np.nan),
                where=accepted_rows > 0,
            )
        )
        within = np.divide(
            weights @ within_by_identity,
            accepted_rows,
            out=np.full(samples, np.nan),
            where=accepted_rows > 0,
        )
        over = np.divide(
            weights @ over_by_identity,
            accepted_rows,
            out=np.full(samples, np.nan),
            where=accepted_rows > 0,
        )
        per_identity_mae = np.divide(
            abs_by_identity,
            accepted_by_identity,
            out=np.zeros(len(names)),
            where=accepted_by_identity > 0,
        )
        macro_denominator = weights @ accepted_identity.astype(float)
        macro = np.divide(
            weights @ per_identity_mae,
            macro_denominator,
            out=np.full(samples, np.nan),
            where=macro_denominator > 0,
        )
        tail_rows = weights @ tail_rows_by_identity
        tail_mae = np.divide(
            weights @ tail_abs_by_identity,
            tail_rows,
            out=np.full(samples, np.nan),
            where=tail_rows > 0,
        )
        point = selective_report[f"{coverage:.2f}"]
        metrics = point["metrics"]
        selective_ci[f"{coverage:.2f}"] = {
            "achieved_coverage": _percentile(
                achieved, point["achieved_coverage"], confidence
            ),
            "mae": _percentile(mae, metrics["mae"], confidence),
            "identity_macro_mae": _percentile(
                macro, metrics["identity_macro_mae"], confidence
            ),
            "rmse": _percentile(rmse, metrics["rmse"], confidence),
            "within_2_fraction": _percentile(
                within, metrics["within_2_fraction"], confidence
            ),
            "over_5_fraction": _percentile(
                over, metrics["over_5_fraction"], confidence
            ),
            "tail_25_35_mae": _percentile(
                tail_mae, metrics["tail_25_35_mae"], confidence
            ),
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
            "seed_derivation": "first 64 bits of SHA-256('locked-hcs-uncertainty:<base>:<seed>') modulo 2^32",
            "cross_seed_pooling": False,
        },
        "intervals": interval_ci,
        "selective": selective_ci,
    }


def _gate_report(
    interval_report: Mapping[str, Any],
    selective_report: Mapping[str, Any],
    gates: Mapping[str, Any],
) -> dict[str, Any]:
    conformal = interval_report["normalized_conformal"]
    c90 = conformal["0.90"]
    selective_80 = selective_report["0.80"]["metrics"]
    if selective_80.get("mae") is None or selective_80.get("over_5_fraction") is None:
        _fail("frozen selective-80 rule accepted no evaluable rows")
    observed = {
        "conformal_max_absolute_calibration_error_all_levels": max(
            float(entry["marginal"]["absolute_calibration_error"])
            for entry in conformal.values()
        ),
        "conformal_90_marginal_coverage_min": float(
            c90["marginal"]["empirical_coverage"]
        ),
        "conformal_90_identity_macro_coverage_min": float(
            c90["marginal"]["identity_macro_coverage"]
        ),
        "conformal_90_fixed_phase_0_coverage_min": float(
            c90["fixed_phase_0"]["empirical_coverage"]
        ),
        "conformal_90_mean_full_width_bpm_max": float(
            c90["marginal"]["mean_full_width_bpm"]
        ),
        "conformal_90_p95_full_width_bpm_max": float(
            c90["marginal"]["p95_full_width_bpm"]
        ),
        "selective_80_mae_bpm_max": float(selective_80["mae"]),
        "selective_80_catastrophic_over_5_max": float(
            selective_80["over_5_fraction"]
        ),
    }
    decisions: dict[str, Any] = {}
    for name, value in observed.items():
        threshold = float(gates[name])
        operator = ">=" if name.endswith("_min") else "<="
        passed = value >= threshold if operator == ">=" else value <= threshold
        decisions[name] = {
            "observed": value,
            "operator": operator,
            "threshold": threshold,
            "passed": bool(passed),
        }
    return {"all_gates_passed": all(item["passed"] for item in decisions.values()), "gates": decisions}


def _empty_csv_row() -> dict[str, Any]:
    return {name: "" for name in CSV_COLUMNS}


def _csv_rows(seed: int, report: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for method, levels in report["intervals"].items():
        for coverage_raw, entry in levels.items():
            coverage = float(coverage_raw)
            scopes = [("marginal", "", entry["marginal"])] + [
                ("fixed_phase", str(phase), entry["fixed_phases"][str(phase)])
                for phase in PHASES
            ]
            for scope, phase, summary in scopes:
                row = _empty_csv_row()
                row.update(
                    {
                        "record_type": "interval",
                        "seed": seed,
                        "method": method,
                        "nominal_coverage": coverage,
                        "scope": scope,
                        "phase": phase,
                        "rows": summary["rows"],
                        "identities": summary["identities"],
                        "coverage": summary["empirical_coverage"],
                        "identity_macro_coverage": summary["identity_macro_coverage"],
                        "worst_identity_coverage": summary["worst_identity_coverage"],
                        "mean_full_width_bpm": summary["mean_full_width_bpm"],
                        "p95_full_width_bpm": summary["p95_full_width_bpm"],
                        "calibration_error": summary["absolute_calibration_error"],
                    }
                )
                rows.append(row)
    for coverage_raw, entry in report["selective"].items():
        row = _empty_csv_row()
        metrics = entry["metrics"]
        row.update(
            {
                "record_type": "selective",
                "seed": seed,
                "method": "frozen_rr_std_threshold",
                "intended_acceptance_coverage": float(coverage_raw),
                "scope": "marginal",
                "rows": entry["accepted_rows"],
                "identities": metrics["identities"],
                "achieved_coverage": entry["achieved_coverage"],
                "mae": metrics["mae"],
                "identity_macro_mae": metrics["identity_macro_mae"],
                "rmse": metrics["rmse"],
                "within_2_fraction": metrics["within_2_fraction"],
                "over_5_fraction": metrics["over_5_fraction"],
                "tail_25_35_mae": metrics["tail_25_35_mae"],
            }
        )
        rows.append(row)
    for name, decision in report["fixed_gate_decision"]["gates"].items():
        row = _empty_csv_row()
        row.update(
            {
                "record_type": "gate",
                "seed": seed,
                "scope": "all_valid_rows",
                "gate_name": name,
                "operator": decision["operator"],
                "threshold": decision["threshold"],
                "observed": decision["observed"],
                "passed": decision["passed"],
            }
        )
        rows.append(row)
    return rows


def _format_csv(value: Any) -> str:
    if value == "" or value is None:
        return ""
    if isinstance(value, (bool, np.bool_)):
        return "true" if bool(value) else "false"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        number = float(value)
        if not math.isfinite(number):
            _fail("uncertainty CSV contains a non-finite number")
        return np.format_float_positional(number, unique=True, trim="-")
    return str(value)


def _write_csv(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(CSV_COLUMNS), lineterminator="\n")
        writer.writeheader()
        for row in rows:
            if set(row) != set(CSV_COLUMNS):
                _fail("uncertainty CSV row schema differs")
            writer.writerow({name: _format_csv(row[name]) for name in CSV_COLUMNS})
        stream.flush()
        os.fsync(stream.fileno())


def _temporary(destination: Path) -> Path:
    return destination.with_name(
        f".{destination.name}.{os.getpid()}.{secrets.token_hex(8)}.tmp"
    )


def _publish(temporary: Path, destination: Path) -> None:
    try:
        os.link(temporary, destination)
    except FileExistsError as exc:
        raise LockedUncertaintyEvaluationError(
            f"immutable uncertainty output already exists: {destination}"
        ) from exc


def evaluate_locked_uncertainty(
    *,
    locked_oof_root: Path,
    uncertainty_evaluation_spec: Path,
    evaluation_spec: Path,
    calibration_path: Path,
    predictions_seal: Path,
    uncertainty_seal: Path,
    evaluation_lock: Path,
    target_receipt: Path,
    output_dir: Path,
    report_output: Path,
    csv_output: Path,
    receipt_output: Path,
    orchestrator_command: Sequence[str] = (),
) -> dict[str, Any]:
    output_root = output_dir.expanduser().resolve()
    destinations = tuple(
        path.expanduser().resolve() for path in (report_output, csv_output, receipt_output)
    )
    if len(set(destinations)) != 3 or any(path.parent != output_root for path in destinations):
        _fail("uncertainty report, CSV, and receipt must be distinct output-dir children")
    if any(path.exists() for path in destinations):
        _fail("immutable uncertainty evaluation output already exists")

    # Do not resolve/stat/open evaluation_lock, target_receipt, target, or joined
    # OOF above this line.  The entire uncertainty protocol is authorized first.
    spec, calibration, _, uncertainty_arrays, pretarget_audit = _validate_pre_target_inputs(
        uncertainty_evaluation_spec=uncertainty_evaluation_spec,
        evaluation_spec=evaluation_spec,
        calibration_path=calibration_path,
        predictions_seal_path=predictions_seal,
        uncertainty_seal_path=uncertainty_seal,
        locked_oof_root=locked_oof_root,
    )
    population = spec["population"]
    bootstrap = spec["bootstrap"]
    try:
        frames, primary_context = PRIMARY._validate_context(
            locked_oof_root=locked_oof_root,
            evaluation_lock=evaluation_lock,
            target_receipt=target_receipt,
            expected_rows=int(population["valid_reference_rows_per_seed"]),
            expected_identities=int(population["physical_identity_count"]),
            expected_seeds=population["fixed_seeds"],
        )
    except RuntimeError as exc:
        raise LockedUncertaintyEvaluationError(str(exc)) from exc
    aligned = _align_uncertainty(uncertainty_arrays, frames)
    lookup = _calibration_lookup(calibration)
    per_seed: dict[str, Any] = {}
    csv_rows: list[dict[str, Any]] = []
    for seed in primary_context["seeds"]:
        intervals, interval_raw = _interval_evaluation(
            frame=frames[seed],
            uncertainty=aligned[seed],
            calibration_lookup=lookup,
            seed=seed,
        )
        selective, selective_masks = _selective_evaluation(
            frame=frames[seed],
            uncertainty=aligned[seed],
            calibration_lookup=lookup,
            seed=seed,
        )
        gate = _gate_report(intervals, selective, calibration["fixed_evaluation_gates"])
        seed_report = {
            "seed": seed,
            "rows": len(frames[seed]["cache_index"]),
            "identity_count": len(set(np.asarray(frames[seed]["identity"]).astype(str).tolist())),
            "exact_seed_cache_fold_final_rr_bit_parity": True,
            "intervals": intervals,
            "selective": selective,
            "identity_cluster_bootstrap": _cluster_bootstrap(
                frame=frames[seed],
                interval_raw=interval_raw,
                interval_report=intervals,
                selective_masks=selective_masks,
                selective_report=selective,
                seed=seed,
                samples=int(bootstrap["samples"]),
                base_seed=int(bootstrap["base_seed"]),
                confidence=float(bootstrap["confidence"]),
            ),
            "fixed_gate_decision": gate,
        }
        per_seed[str(seed)] = seed_report
        csv_rows.extend(_csv_rows(seed, seed_report))
    seed_status = {
        str(seed): bool(per_seed[str(seed)]["fixed_gate_decision"]["all_gates_passed"])
        for seed in primary_context["seeds"]
    }
    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "classification": "retrospective_locked_hcs_uncertainty_evaluation",
        "commercial_claim_authorized": False,
        "commercial_performance_proven": False,
        "prospective_confirmation_required": True,
        "independent_prospective_cohort_evaluated": False,
        "selection_retraining_or_test_time_fitting_performed": False,
        "interval_scale_or_threshold_refit_performed": False,
        "seed_pooling_ranking_or_suppression_performed": False,
        "point_prediction_modified": False,
        "protocol_role": "separate_secondary_retrospective_engineering_protocol",
        "primary_uncertainty_contract_overridden": False,
        "uncertainty_evaluation_specification": pretarget_audit[
            "secondary_uncertainty_evaluation_spec"
        ],
        "frozen_evaluation_gates": calibration["fixed_evaluation_gates"],
        "all_prespecified_fixed_seeds_must_pass": True,
        "all_fixed_seed_uncertainty_gates_passed": bool(all(seed_status.values())),
        "fixed_seed_gate_status": seed_status,
        "pretarget_provenance_audit": pretarget_audit,
        "canonical_target_context": primary_context,
        "per_seed": per_seed,
        "limitations": [
            "cross-fitted pretest calibration transferred to final outer-test checkpoints is not a formal exchangeability guarantee",
            "overlapping windows and identity clustering limit nominal interval interpretation",
            "retrospective results cannot authorize a commercial claim without an independent prospective cohort",
        ],
        "orchestrator_command": list(orchestrator_command),
    }
    report["content_sha256"] = PRIMARY.canonical_json_sha256(report)

    output_root.mkdir(parents=True, exist_ok=True)
    report_path, csv_path, receipt_path = destinations
    temporary = tuple(_temporary(path) for path in destinations)
    try:
        PRIMARY._write_json(temporary[0], report)
        _write_csv(temporary[1], csv_rows)
        outputs = {
            "report": {
                "path": str(report_path),
                "sha256": PRIMARY.sha256_file(temporary[0]),
                "bytes": temporary[0].stat().st_size,
            },
            "metrics_csv": {
                "path": str(csv_path),
                "sha256": PRIMARY.sha256_file(temporary[1]),
                "bytes": temporary[1].stat().st_size,
            },
        }
        receipt: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "classification": "retrospective_locked_hcs_uncertainty_evaluation_receipt",
            "commercial_claim_authorized": False,
            "commercial_performance_proven": False,
            "prospective_confirmation_required": True,
            "independent_prospective_cohort_evaluated": False,
            "protocol_role": "separate_secondary_retrospective_engineering_protocol",
            "primary_uncertainty_contract_overridden": False,
            "outputs_create_once": True,
            "output_overwrite_allowed": False,
            "all_target_free_inputs_rehashed_before_evaluation_lock_access": True,
            "target_access_authorized_by_canonical_evaluation_lock": True,
            "no_test_time_fit_selection_seed_pooling_or_prediction_change": True,
            "uncertainty_evaluation_specification": pretarget_audit[
                "secondary_uncertainty_evaluation_spec"
            ],
            "inputs": {
                **pretarget_audit,
                **primary_context["bindings"],
            },
            "outputs": outputs,
            "metrics_csv_rows": len(csv_rows),
            "seeds": primary_context["seeds"],
            "orchestrator_command": list(orchestrator_command),
        }
        receipt["content_sha256"] = PRIMARY.canonical_json_sha256(receipt)
        PRIMARY._write_json(temporary[2], receipt)
        for source, destination in zip(temporary, destinations, strict=True):
            _publish(source, destination)
            destination.chmod(0o444)
    finally:
        for path in temporary:
            path.unlink(missing_ok=True)
    published = _read_json(receipt_path, "published uncertainty receipt", content_hash=True)
    if (
        PRIMARY.bind_file(report_path) != published["outputs"]["report"]
        or PRIMARY.bind_file(csv_path) != published["outputs"]["metrics_csv"]
    ):
        _fail("published uncertainty output differs from its immutable receipt")
    return published


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--locked-oof-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument(
        "--uncertainty-evaluation-spec",
        type=Path,
        default=DEFAULT_UNCERTAINTY_EVALUATION_SPEC,
    )
    parser.add_argument("--evaluation-spec", type=Path, default=DEFAULT_SPEC)
    parser.add_argument("--calibration", type=Path, default=DEFAULT_CALIBRATION)
    parser.add_argument("--predictions-seal", type=Path)
    parser.add_argument("--uncertainty-seal", type=Path)
    parser.add_argument("--evaluation-lock", type=Path)
    parser.add_argument("--target-receipt", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--report-output", type=Path)
    parser.add_argument("--csv-output", type=Path)
    parser.add_argument("--receipt-output", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = args.locked_oof_root.expanduser().resolve()
    output = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else root / "uncertainty_evaluation"
    )
    command = [str(Path(__file__).resolve()), *(sys.argv[1:] if argv is None else argv)]
    try:
        result = evaluate_locked_uncertainty(
            locked_oof_root=root,
            uncertainty_evaluation_spec=args.uncertainty_evaluation_spec,
            evaluation_spec=args.evaluation_spec,
            calibration_path=args.calibration,
            predictions_seal=(
                args.predictions_seal if args.predictions_seal is not None else root / "predictions_seal.json"
            ),
            uncertainty_seal=(
                args.uncertainty_seal if args.uncertainty_seal is not None else root / "uncertainty_inputs_seal.json"
            ),
            evaluation_lock=(
                args.evaluation_lock if args.evaluation_lock is not None else root / "evaluation_lock.json"
            ),
            target_receipt=(
                args.target_receipt
                if args.target_receipt is not None
                else root / "canonical_locked_hcs_targets_receipt.json"
            ),
            output_dir=output,
            report_output=(
                args.report_output if args.report_output is not None else output / "uncertainty_report.json"
            ),
            csv_output=(
                args.csv_output if args.csv_output is not None else output / "uncertainty_metrics.csv"
            ),
            receipt_output=(
                args.receipt_output if args.receipt_output is not None else output / "uncertainty_receipt.json"
            ),
            orchestrator_command=command,
        )
    except (LockedUncertaintyEvaluationError, RuntimeError, ValueError, KeyError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
