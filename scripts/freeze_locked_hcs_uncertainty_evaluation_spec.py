#!/usr/bin/env python3
"""Freeze the secondary HCS uncertainty protocol before target access.

The primary evaluation specification intentionally treats uncertainty only as
a ranking diagnostic.  This create-once document does not alter that contract.
It authorizes a separate, secondary engineering analysis whose interval scales
and selective thresholds were already frozen from the identity-disjoint
pretest calibration.  The specification binds the unchanged primary protocol,
the completed calibration, and every implementation source needed to reproduce
the secondary analysis.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import secrets
import sys
from typing import Any, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LOCKED_ROOT = (
    PROJECT_ROOT / "artifacts/runs/harmonic_candidate_set_snn_v2/hcs_locked_oof"
)
DEFAULT_PRIMARY_SPEC = (
    PROJECT_ROOT
    / "artifacts/campaigns/harmonic_candidate_set_snn_v2/locked_primary_evaluation_spec.json"
)
DEFAULT_CALIBRATION = (
    PROJECT_ROOT
    / "artifacts/campaigns/harmonic_candidate_set_snn_v2/nested_proposer/current_source_merged/uncertainty_calibration.json"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "artifacts/campaigns/harmonic_candidate_set_snn_v2/locked_uncertainty_evaluation_spec.json"
)

SCHEMA_VERSION = 1
FOLDS = tuple(range(6))
INTERVAL_COVERAGES = (0.50, 0.80, 0.90, 0.95)
SELECTIVE_COVERAGES = (0.50, 0.80, 0.90, 1.00)
PHASES = tuple(range(8))
STD_FLOOR_BPM = 0.25
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
DEFAULT_SOURCE_PATHS = {
    "uncertainty_specification_builder": Path(__file__).resolve(),
    "primary_evaluator": PROJECT_ROOT / "scripts/evaluate_locked_hcs_oof.py",
    "calibration_builder": PROJECT_ROOT / "scripts/build_locked_proposer_uncertainty_calibration.py",
    "uncertainty_input_sealer": PROJECT_ROOT / "scripts/seal_locked_hcs_uncertainty_inputs.py",
    "uncertainty_evaluator": PROJECT_ROOT / "scripts/evaluate_locked_hcs_uncertainty.py",
}
TARGET_BEARING_NAMES = (
    "evaluation_lock.json",
    "locked_hcs_oof_joined.npz",
    "canonical_locked_hcs_targets.npz",
    "canonical_locked_hcs_targets_receipt.json",
)


class UncertaintyEvaluationSpecError(RuntimeError):
    """The secondary protocol cannot be frozen or verified safely."""


def _load_primary() -> Any:
    path = PROJECT_ROOT / "scripts/evaluate_locked_hcs_oof.py"
    spec = importlib.util.spec_from_file_location(
        "evaluate_locked_hcs_oof_for_uncertainty_spec", path
    )
    if spec is None or spec.loader is None:
        raise UncertaintyEvaluationSpecError(f"cannot import primary evaluator: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


PRIMARY = _load_primary()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
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


def bind_file(path: Path) -> dict[str, Any]:
    raw = path.expanduser()
    if raw.is_symlink():
        raise UncertaintyEvaluationSpecError(f"bound source cannot be a symlink: {raw}")
    resolved = raw.resolve()
    if not resolved.is_file():
        raise UncertaintyEvaluationSpecError(f"bound source is absent: {resolved}")
    return {
        "path": str(resolved),
        "sha256": sha256_file(resolved),
        "bytes": resolved.stat().st_size,
    }


def _same_binding(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    return (
        Path(str(left.get("path", ""))).resolve()
        == Path(str(right.get("path", ""))).resolve()
        and str(left.get("sha256", "")) == str(right.get("sha256", ""))
        and int(left.get("bytes", -1)) == int(right.get("bytes", -2))
    )


def _read_json(path: Path, label: str, *, require_content_hash: bool) -> dict[str, Any]:
    try:
        return PRIMARY._read_json(
            path, label, require_content_hash=require_content_hash
        )
    except RuntimeError as exc:
        raise UncertaintyEvaluationSpecError(str(exc)) from exc


def _validate_calibration(
    path: Path, *, fixed_seeds: Sequence[int]
) -> tuple[dict[str, Any], dict[str, Any]]:
    resolved = path.expanduser().resolve()
    calibration = _read_json(
        resolved, "completed pretest uncertainty calibration", require_content_hash=True
    )
    expected_keys = {(fold, int(seed)) for seed in fixed_seeds for fold in FOLDS}
    units = calibration.get("units")
    method = calibration.get("fixed_method")
    gates = calibration.get("fixed_evaluation_gates")
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
        or calibration.get("seeds") != [int(seed) for seed in fixed_seeds]
        or calibration.get("unit_count") != len(expected_keys)
        or not isinstance(units, list)
        or len(units) != len(expected_keys)
        or not isinstance(method, Mapping)
        or not isinstance(gates, Mapping)
    ):
        raise UncertaintyEvaluationSpecError("completed calibration invariants are invalid")
    if (
        method.get("phase_modulus") != 8
        or method.get("phase_value") != 0
        or float(method.get("std_floor_bpm", -1.0)) != STD_FLOOR_BPM
        or method.get("interval_coverages") != list(INTERVAL_COVERAGES)
        or method.get("selective_coverages") != list(SELECTIVE_COVERAGES)
        or method.get("no_test_time_fit_or_threshold_selection") is not True
        or method.get("formal_exchangeability_claim") is not False
        or set(gates) != GATE_NAMES
        or gates.get("all_seeds_required") is not True
    ):
        raise UncertaintyEvaluationSpecError("calibration method or fixed gates differ")
    for name in GATE_NAMES - {"all_seeds_required"}:
        value = gates[name]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) < 0.0
        ):
            raise UncertaintyEvaluationSpecError(f"calibration gate is invalid: {name}")
    observed: set[tuple[int, int]] = set()
    for unit in units:
        if not isinstance(unit, Mapping):
            raise UncertaintyEvaluationSpecError("calibration unit must be an object")
        try:
            key = (int(unit["outer_fold"]), int(unit["seed"]))
        except (KeyError, TypeError, ValueError) as exc:
            raise UncertaintyEvaluationSpecError("calibration unit identity is invalid") from exc
        if key not in expected_keys or key in observed:
            raise UncertaintyEvaluationSpecError(
                f"calibration unit topology differs: {key}"
            )
        observed.add(key)
        intervals = unit.get("interval_calibration")
        thresholds = unit.get("selective_thresholds")
        if (
            not isinstance(intervals, Mapping)
            or set(intervals) != {f"{value:.2f}" for value in INTERVAL_COVERAGES}
            or not isinstance(thresholds, Mapping)
            or set(thresholds) != {f"{value:.2f}" for value in SELECTIVE_COVERAGES}
        ):
            raise UncertaintyEvaluationSpecError(
                f"calibration unit rule topology differs: {key}"
            )
    if observed != expected_keys:
        raise UncertaintyEvaluationSpecError("calibration is not the exact 18-unit cover")
    return calibration, bind_file(resolved)


def uncertainty_evaluation_spec_document(
    *,
    primary_spec: Mapping[str, Any],
    primary_spec_binding: Mapping[str, Any],
    calibration: Mapping[str, Any],
    calibration_binding: Mapping[str, Any],
    source_bindings: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Build the exact secondary protocol from already verified frozen inputs."""

    population = primary_spec["population"]
    bootstrap = primary_spec["bootstrap"]
    primary_uncertainty = primary_spec["uncertainty"]
    if (
        primary_uncertainty.get("role")
        != "diagnostic_ranking_only_not_calibrated_interval"
        or primary_uncertainty.get("calibration_fit_allowed") is not False
        or primary_uncertainty.get("threshold_fit_allowed") is not False
        or primary_uncertainty.get("model_or_candidate_selection_allowed") is not False
    ):
        raise UncertaintyEvaluationSpecError(
            "primary uncertainty contract is not the expected diagnostic-only protocol"
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "classification": "locked_hcs_secondary_uncertainty_evaluation_specification",
        "commercial_claim_authorized": False,
        "commercial_performance_proven": False,
        "prospective_confirmation_required": True,
        "independent_prospective_cohort_evaluated": False,
        "must_be_frozen_before_target_or_target_bearing_artifact_access": True,
        "target_or_target_bearing_artifact_opened_to_build_spec": False,
        "outer_test_prediction_or_uncertainty_artifact_opened_to_build_spec": False,
        "protocol_relationship": {
            "role": "separate_secondary_retrospective_engineering_protocol",
            "primary_uncertainty_contract_overridden": False,
            "primary_point_evaluation_or_gates_modified": False,
            "primary_diagnostic_only_uncertainty_claim_preserved": True,
            "secondary_interval_results_are_part_of_primary_evaluation": False,
            "formal_conformal_exchangeability_or_coverage_guarantee_claimed": False,
            "reason": "all interval scales and selective thresholds were frozen in a separate identity-disjoint pretest calibration before target access",
        },
        "population": {
            "valid_reference_rows_per_seed": int(
                population["valid_reference_rows_per_seed"]
            ),
            "physical_identity_count": int(population["physical_identity_count"]),
            "outer_folds": list(population["outer_folds"]),
            "fixed_seeds": list(population["fixed_seeds"]),
            "seed_count": int(population["seed_count"]),
            "each_seed_evaluated_independently": True,
            "cross_seed_pooling_ranking_or_suppression_allowed": False,
        },
        "fixed_methods": {
            "point_prediction": "locked_final_rr_bpm_unmodified",
            "uncertainty_scale": "fallback_std_bpm_from_target_free_uncertainty_seal",
            "normal_uncalibrated": {
                "coverages": list(INTERVAL_COVERAGES),
                "center": "locked_final_rr_bpm",
                "half_width": "NormalDist.inv_cdf((1+coverage)/2)*fallback_std_bpm",
                "scale_floor_applied": False,
            },
            "normalized_conformal": {
                "coverages": list(INTERVAL_COVERAGES),
                "center": "locked_final_rr_bpm",
                "half_width": "frozen_outer_fold_seed_normalized_absolute_error_quantile*max(fallback_std_bpm,0.25)",
                "std_floor_bpm": STD_FLOOR_BPM,
                "quantile_source": "bound completed pretest calibration only",
                "test_time_refit_allowed": False,
                "formal_exchangeability_claim": False,
            },
            "selective": {
                "intended_acceptance_coverages": list(SELECTIVE_COVERAGES),
                "accept_rule": "fallback_std_bpm <= frozen outer-fold/seed rr_std threshold",
                "threshold_source": "bound completed pretest calibration only",
                "test_time_threshold_fit_or_selection_allowed": False,
            },
            "reported_interval_statistics": [
                "marginal_coverage",
                "identity_macro_coverage",
                "worst_identity_coverage",
                "mean_full_width_bpm",
                "p95_full_width_bpm",
                "absolute_calibration_error",
            ],
            "reported_selective_metrics": [
                "achieved_coverage",
                "mae",
                "identity_macro_mae",
                "rmse",
                "within_2_fraction",
                "over_5_fraction",
                "tail_25_35_mae",
            ],
        },
        "fixed_phases": {
            "definition": "window_number modulo 8",
            "modulus": 8,
            "phases": list(PHASES),
            "phase_0_reported_separately": True,
            "phase_search_ranking_or_suppression_allowed": False,
        },
        "fixed_evaluation_gates": dict(calibration["fixed_evaluation_gates"]),
        "identity_cluster_bootstrap": {
            "unit": "physical_identity",
            "samples": int(bootstrap["samples"]),
            "confidence": float(bootstrap["confidence"]),
            "interval": "two-sided percentile",
            "rng": "numpy.default_rng_PCG64",
            "base_seed": int(bootstrap["base_seed"]),
            "resample_count_equals_physical_identity_count": True,
            "per_seed_seed_derivation": "first 64 bits of SHA-256('locked-hcs-uncertainty:<base>:<seed>') modulo 2^32",
            "cross_seed_pooling_allowed": False,
        },
        "post_target_prohibitions": {
            "method_or_gate_change": True,
            "calibration_or_interval_scale_fit": True,
            "selective_threshold_fit_or_search": True,
            "phase_or_stratum_search": True,
            "seed_pooling_ranking_or_suppression": True,
            "point_prediction_change": True,
        },
        "bound_inputs": {
            "unchanged_primary_evaluation_spec": dict(primary_spec_binding),
            "completed_pretest_uncertainty_calibration": dict(calibration_binding),
            "implementation_sources": {
                name: dict(binding) for name, binding in sorted(source_bindings.items())
            },
        },
        "calibration_content_sha256": str(calibration["content_sha256"]),
        "limitations": [
            "this separate secondary protocol does not supersede the primary diagnostic-only uncertainty contract",
            "checkpoint transfer and clustered overlapping windows prevent a formal coverage guarantee",
            "commercial release still requires an independent prospective cohort",
        ],
    }


def _expected_document(
    *,
    primary_spec_path: Path,
    calibration_path: Path,
    source_paths: Mapping[str, Path],
) -> dict[str, Any]:
    try:
        primary_spec, primary_binding = PRIMARY._load_evaluation_spec(primary_spec_path)
    except RuntimeError as exc:
        raise UncertaintyEvaluationSpecError(str(exc)) from exc
    fixed_seeds = [int(value) for value in primary_spec["population"]["fixed_seeds"]]
    calibration, calibration_binding = _validate_calibration(
        calibration_path, fixed_seeds=fixed_seeds
    )
    if set(source_paths) != set(DEFAULT_SOURCE_PATHS):
        raise UncertaintyEvaluationSpecError("implementation source topology differs")
    source_bindings = {name: bind_file(path) for name, path in source_paths.items()}
    return uncertainty_evaluation_spec_document(
        primary_spec=primary_spec,
        primary_spec_binding=primary_binding,
        calibration=calibration,
        calibration_binding=calibration_binding,
        source_bindings=source_bindings,
    )


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


def freeze_uncertainty_evaluation_spec(
    *,
    output_path: Path,
    locked_oof_root: Path,
    primary_spec_path: Path,
    calibration_path: Path,
    source_paths: Mapping[str, Path] | None = None,
) -> dict[str, Any]:
    destination = output_path.expanduser().resolve()
    if destination.exists():
        raise UncertaintyEvaluationSpecError(
            f"immutable uncertainty evaluation specification already exists: {destination}"
        )
    root = locked_oof_root.expanduser().resolve()
    for name in TARGET_BEARING_NAMES:
        if (root / name).exists():
            raise UncertaintyEvaluationSpecError(
                f"secondary uncertainty specification must be frozen before target access: {name}"
            )
    document = _expected_document(
        primary_spec_path=primary_spec_path.expanduser().resolve(),
        calibration_path=calibration_path.expanduser().resolve(),
        source_paths=(DEFAULT_SOURCE_PATHS if source_paths is None else source_paths),
    )
    document["content_sha256"] = canonical_json_sha256(document)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = _temporary_path(destination)
    try:
        _write_json(temporary, document)
        try:
            os.link(temporary, destination)
        except FileExistsError as exc:
            raise UncertaintyEvaluationSpecError(
                f"immutable uncertainty specification appeared concurrently: {destination}"
            ) from exc
        destination.chmod(0o444)
    finally:
        temporary.unlink(missing_ok=True)
    published = _read_json(
        destination,
        "published uncertainty evaluation specification",
        require_content_hash=True,
    )
    if published != document:
        raise UncertaintyEvaluationSpecError("published uncertainty specification changed")
    return bind_file(destination)


def load_uncertainty_evaluation_spec(
    path: Path,
    *,
    expected_primary_spec_path: Path,
    expected_calibration_path: Path,
    source_paths: Mapping[str, Path] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Rebuild and byte-compare the exact protocol without touching predictions."""

    raw_path = path.expanduser()
    if raw_path.is_symlink():
        raise UncertaintyEvaluationSpecError(
            "secondary uncertainty evaluation specification cannot be a symlink"
        )
    resolved = raw_path.resolve()
    if not resolved.is_file():
        raise UncertaintyEvaluationSpecError(
            "immutable secondary uncertainty evaluation specification is absent"
        )
    if resolved.stat().st_mode & 0o777 != 0o444:
        raise UncertaintyEvaluationSpecError(
            "secondary uncertainty evaluation specification is not mode 0444"
        )
    observed = _read_json(
        resolved, "secondary uncertainty evaluation specification", require_content_hash=True
    )
    expected = _expected_document(
        primary_spec_path=expected_primary_spec_path.expanduser().resolve(),
        calibration_path=expected_calibration_path.expanduser().resolve(),
        source_paths=(DEFAULT_SOURCE_PATHS if source_paths is None else source_paths),
    )
    observed_payload = dict(observed)
    observed_payload.pop("content_sha256", None)
    if observed_payload != expected:
        raise UncertaintyEvaluationSpecError(
            "secondary uncertainty evaluation specification or a bound input has drifted"
        )
    binding = bind_file(resolved)
    bound = observed["bound_inputs"]
    primary_binding = bind_file(expected_primary_spec_path)
    calibration_binding = bind_file(expected_calibration_path)
    if not _same_binding(bound["unchanged_primary_evaluation_spec"], primary_binding):
        raise UncertaintyEvaluationSpecError("secondary spec primary binding differs")
    if not _same_binding(
        bound["completed_pretest_uncertainty_calibration"], calibration_binding
    ):
        raise UncertaintyEvaluationSpecError("secondary spec calibration binding differs")
    return observed, binding


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--locked-oof-root", type=Path, default=DEFAULT_LOCKED_ROOT)
    parser.add_argument("--primary-evaluation-spec", type=Path, default=DEFAULT_PRIMARY_SPEC)
    parser.add_argument("--calibration", type=Path, default=DEFAULT_CALIBRATION)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = freeze_uncertainty_evaluation_spec(
            output_path=args.output,
            locked_oof_root=args.locked_oof_root,
            primary_spec_path=args.primary_evaluation_spec,
            calibration_path=args.calibration,
        )
    except (UncertaintyEvaluationSpecError, RuntimeError, ValueError, KeyError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
