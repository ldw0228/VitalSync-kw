#!/usr/bin/env python3
"""Freeze cross-fitted proposer uncertainty calibration before outer-test access.

The calibration cohort is the exact 5-way, identity-disjoint proposer prediction
stack for each outer-fold/seed unit.  Every contributing prediction was produced
for identities excluded from both that checkpoint's training and early-stopping
partitions.  Only fixed phase-0 (``window_number % 8 == 0``) valid-reference rows
are used, so the fitted ratios are not multiplied by the eight overlapping
window phases.

The result is a pre-test engineering calibration rule.  It is deliberately not
described as a formal exchangeability guarantee: the rule transfers between
cross-fitted checkpoints and the final checkpoint, and rows remain clustered by
physical identity/session.  Independent prospective calibration is still a
release requirement.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PLAN = (
    PROJECT_ROOT
    / "artifacts/campaigns/harmonic_candidate_set_snn_v2/nested_proposer/full_oof_non_test/control/plan.json"
)
DEFAULT_INDEX = (
    PROJECT_ROOT
    / "artifacts/campaigns/harmonic_candidate_set_snn_v2/nested_proposer/current_source_merged/index.json"
)
DEFAULT_RETRAIN_AUDIT = (
    PROJECT_ROOT
    / "artifacts/campaigns/harmonic_candidate_set_snn_v2/nested_proposer/current_source_merged/retrain_impact_audit.json"
)
DEFAULT_GOVERNANCE = (
    PROJECT_ROOT
    / "artifacts/campaigns/harmonic_candidate_set_snn_v2/nested_proposer/current_source_merged/governance_attestation.json"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "artifacts/campaigns/harmonic_candidate_set_snn_v2/nested_proposer/current_source_merged/uncertainty_calibration.json"
)

SCHEMA_VERSION = 1
FOLDS = tuple(range(6))
SEEDS = (20260828, 20260829, 20260830)
COVERAGES = (0.50, 0.80, 0.90, 0.95)
SELECTIVE_COVERAGES = (0.50, 0.80, 0.90, 1.00)
STD_FLOOR_BPM = 0.25
PHASE_MODULUS = 8
PHASE_VALUE = 0
REQUIRED_ARRAYS = frozenset(
    {
        "cache_index",
        "session_id",
        "identity",
        "window_number",
        "reference_valid",
        "reference_rr_bpm",
        "prediction",
        "rr_std",
        "fold_id",
        "checkpoint_sha256",
        "split_manifest_file_sha256",
        "split_manifest_content_sha256",
        "strict_retrospective",
        "strict_nested_prediction_role",
    }
)


class CalibrationBuildError(RuntimeError):
    """A pre-test calibration input or publication invariant failed."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def canonical_content_sha256(value: Mapping[str, Any]) -> str:
    payload = dict(value)
    payload.pop("content_sha256", None)
    return canonical_json_sha256(payload)


def bind_file(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    if not resolved.is_file() or resolved.is_symlink():
        raise CalibrationBuildError(f"bound input must be a regular non-symlink file: {resolved}")
    return {
        "path": str(resolved),
        "sha256": sha256_file(resolved),
        "bytes": resolved.stat().st_size,
    }


def _read_json(path: Path, label: str) -> tuple[dict[str, Any], dict[str, Any]]:
    resolved = path.expanduser().resolve()
    binding = bind_file(resolved)
    try:
        value = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CalibrationBuildError(f"invalid {label}: {resolved} ({exc})") from exc
    if not isinstance(value, dict):
        raise CalibrationBuildError(f"{label} root must be an object")
    if "content_sha256" in value and canonical_content_sha256(value) != value.get(
        "content_sha256"
    ):
        raise CalibrationBuildError(f"{label} content hash mismatch")
    return value, binding


def _load_fixed_runner_module() -> Any:
    path = PROJECT_ROOT / "scripts/run_fixed_i3_pretest_campaign.py"
    spec = importlib.util.spec_from_file_location(
        "run_fixed_i3_pretest_campaign_for_calibration", path
    )
    if spec is None or spec.loader is None:
        raise CalibrationBuildError(f"cannot import index validator: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _scalar(archive: Mapping[str, Any], name: str) -> Any:
    value = np.asarray(archive[name])
    if value.ndim != 0:
        raise CalibrationBuildError(f"prediction field {name} must be scalar")
    return value.item()


def _higher_quantile(values: np.ndarray, probability: float) -> float:
    array = np.sort(np.asarray(values, dtype=np.float64))
    if array.ndim != 1 or not len(array) or not np.isfinite(array).all():
        raise CalibrationBuildError("quantile input must be a non-empty finite vector")
    if not 0.0 < probability <= 1.0:
        raise CalibrationBuildError("quantile probability must lie in (0, 1]")
    # Split-conformal finite-sample rank, one-indexed and clipped at n.
    rank = min(len(array), int(math.ceil((len(array) + 1) * probability)))
    return float(array[rank - 1])


def _ordinary_higher_quantile(values: np.ndarray, probability: float) -> float:
    array = np.sort(np.asarray(values, dtype=np.float64))
    if array.ndim != 1 or not len(array) or not np.isfinite(array).all():
        raise CalibrationBuildError("selective quantile input must be finite")
    if probability >= 1.0:
        return float(array[-1])
    index = int(math.ceil(probability * len(array))) - 1
    return float(array[max(0, min(index, len(array) - 1))])


def _same_binding(
    left: Any,
    right: Any,
    *,
    label: str,
    require_content: bool = False,
) -> None:
    if not isinstance(left, Mapping) or not isinstance(right, Mapping):
        raise CalibrationBuildError(f"missing {label} binding")
    try:
        same = (
            Path(str(left.get("path", ""))).expanduser().resolve()
            == Path(str(right.get("path", ""))).expanduser().resolve()
            and str(left.get("sha256", "")) == str(right.get("sha256", ""))
            and len(str(left.get("sha256", ""))) == 64
            and int(left.get("bytes", -1)) == int(right.get("bytes", -2))
        )
    except (TypeError, ValueError) as exc:
        raise CalibrationBuildError(f"malformed {label} binding") from exc
    if require_content:
        same = (
            same
            and len(str(left.get("content_sha256", ""))) == 64
            and left.get("content_sha256") == right.get("content_sha256")
        )
    if not same:
        raise CalibrationBuildError(f"{label} binding mismatch")


def _live_binding(raw: Any, label: str) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        raise CalibrationBuildError(f"missing {label} binding")
    live = bind_file(Path(str(raw.get("path", ""))))
    _same_binding(raw, live, label=f"{label} live")
    return live


def _load_runtime_seal_module() -> Any:
    path = PROJECT_ROOT / "scripts/seal_runtime_inputs.py"
    spec = importlib.util.spec_from_file_location(
        "seal_runtime_inputs_for_locked_calibration", path
    )
    if spec is None or spec.loader is None:
        raise CalibrationBuildError(f"cannot import runtime seal verifier: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _validate_attestations(
    audit_path: Path, governance_path: Path, index_binding: Mapping[str, Any]
) -> dict[str, Any]:
    audit, audit_binding = _read_json(audit_path, "retrain impact audit")
    if (
        audit.get("schema_version") != 1
        or audit.get("classification")
        != "retrospective_nested_proposer_retrain_impact_audit"
        or audit.get("commercial_claim_authorized") is not False
        or audit.get("outer_test_opened") is not False
        or int(audit.get("outer_test_record_count", -1)) != 0
        or audit.get("target_or_reference_accessed") is not False
        or audit.get("source_campaigns_hash_complete") is not True
        or audit.get("source_plans_compatible") is not True
        or len(str(audit.get("content_sha256", ""))) != 64
        or canonical_content_sha256(audit) != audit.get("content_sha256")
    ):
        raise CalibrationBuildError("retrain impact audit is not a sealed non-test audit")
    governance, governance_binding = _read_json(
        governance_path, "nested governance attestation"
    )
    if (
        governance.get("classification")
        != "retrospective_nested_proposer_governance_attestation"
        or governance.get("commercial_claim_authorized") is not False
        or governance.get("prospective_confirmation_required") is not True
        or governance.get("outer_test_artifact_evaluated_during_non_test_campaigns")
        is not False
        or governance.get("target_metric_used_for_proposer_fit_or_selection") is not False
    ):
        raise CalibrationBuildError("governance attestation invariants are invalid")
    verified_documents = governance.get("verified_documents")
    execution = governance.get("execution_provenance")
    runtime_attestations = governance.get("runtime_input_attestations")
    if (
        governance.get("execution_provenance_complete") is not True
        or not isinstance(verified_documents, Mapping)
        or not isinstance(execution, Mapping)
        or not isinstance(runtime_attestations, Mapping)
        or execution.get("required") is not True
        or execution.get("merged_nested_binding_verified") is not True
        or execution.get("execution_attestation_live_rehashed") is not True
        or execution.get("authoritative_runtime_seal_live_rehashed") is not True
        or execution.get("supervisor_live_rehashed") is not True
        or execution.get("retrain_index_30_of_30_verified") is not True
        or execution.get("canonical_supervisor_and_unit_command_verified") is not True
    ):
        raise CalibrationBuildError("governance execution provenance is incomplete")

    merged_binding = verified_documents.get("merged_index")
    _same_binding(
        merged_binding,
        index_binding,
        label="governance merged index",
        require_content=True,
    )

    # Bind the retrain-impact decision to the exact source graph that
    # governance live-verified.  A self-declared audit (even one with a valid
    # canonical content hash) must not be able to force or suppress retraining
    # without naming the same immutable plans and indexes.
    audit_inputs = audit.get("inputs")
    if not isinstance(audit_inputs, Mapping) or set(audit_inputs) != {
        "full_plan", "main_index", "retrain_plan", "retrain_index"
    }:
        raise CalibrationBuildError("retrain impact audit input topology differs")
    governance_sources = governance.get("verified_retrain_impact_sources")
    if not isinstance(governance_sources, Mapping) or set(governance_sources) != {
        "full_plan", "main_index", "retrain_plan", "retrain_index"
    }:
        raise CalibrationBuildError("governance lacks retrain-impact source closure")
    for name in sorted(audit_inputs):
        _same_binding(
            audit_inputs.get(name),
            governance_sources.get(name),
            label=f"retrain impact source {name}",
            require_content=True,
        )
        source_document, live_source_binding = _read_json(
            Path(str(governance_sources[name].get("path", ""))),
            f"live retrain impact source {name}",
        )
        live_source_with_content = {
            **live_source_binding,
            "content_sha256": source_document.get("content_sha256"),
        }
        _same_binding(
            governance_sources[name],
            live_source_with_content,
            label=f"live retrain impact source {name}",
            require_content=True,
        )
    expected_plan_hashes = {
        "full": governance_sources["full_plan"]["content_sha256"],
        "retrain": governance_sources["retrain_plan"]["content_sha256"],
    }
    if audit.get("plan_content_sha256") != expected_plan_hashes:
        raise CalibrationBuildError("retrain impact plan hashes differ from governance")
    comparison = audit.get("comparison")
    if (
        not isinstance(comparison, Mapping)
        or int(comparison.get("comparison_units", -1)) != 30
        or int(comparison.get("changed_prediction_units", -1)) < 1
        or int(comparison.get("changed_prediction_units", -1)) > 30
        or int(comparison.get("changed_checkpoint_units", -1)) < 0
        or int(comparison.get("changed_checkpoint_units", -1)) > 30
        or int(comparison.get("force_retrain_unit_count", -1)) != 6
        or comparison.get("force_retrain_units_cli_value")
        != "3:20260828,3:20260829,3:20260830,4:20260828,4:20260829,4:20260830"
        or comparison.get("force_retrain_argument")
        != [
            "--force-retrain-units",
            "3:20260828,3:20260829,3:20260830,4:20260828,4:20260829,4:20260830",
        ]
        or comparison.get("checkpoint_change_alone_forces_hcs_retrain") is not False
        or comparison.get("prediction_paths_ignored_after_bound_file_validation") is not True
        or not isinstance(comparison.get("units"), list)
        or len(comparison["units"]) != 30
    ):
        raise CalibrationBuildError("retrain impact comparison is incomplete or disconnected")

    execution_attestation = execution.get("execution_attestation")
    _same_binding(
        verified_documents.get("retrain_execution_attestation"),
        execution_attestation,
        label="governance execution attestation",
        require_content=True,
    )
    _live_binding(execution_attestation, "governance execution attestation")
    if not isinstance(execution_attestation, Mapping):
        raise CalibrationBuildError("missing governance execution attestation")
    receipt, _ = _read_json(
        Path(str(execution_attestation.get("path", ""))),
        "governance execution attestation",
    )
    if (
        receipt.get("schema_version") != 1
        or receipt.get("classification")
        != "sealed_non_test_proposer_execution_attestation"
        or receipt.get("outer_test_opened") is not False
        or int(receipt.get("outer_test_record_count", -1)) != 0
        or receipt.get("commercial_claim_authorized") is not False
        or int(receipt.get("expected_units", -1)) != 30
        or int(receipt.get("completed_units", -1)) != 30
        or int(receipt.get("invocations_this_resume", -1)) != 30
        or receipt.get("one_new_unit_per_invocation") is not True
        or receipt.get("runtime_seal_verified_before_and_after_every_invocation")
        is not True
        or receipt.get("content_sha256")
        != execution_attestation.get("content_sha256")
    ):
        raise CalibrationBuildError("governance execution receipt is not a sealed 30/30 receipt")

    completion = execution.get("completion_evidence")
    if (
        not isinstance(completion, Mapping)
        or int(completion.get("expected_units", -1)) != 30
        or int(completion.get("completed_units", -1)) != 30
        or int(completion.get("invocations_this_resume", -1)) != 30
        or completion.get("single_supervisor_execution_covered_all_units") is not True
        or completion.get("one_new_unit_per_invocation") is not True
        or completion.get("runtime_seal_verified_before_and_after_every_invocation")
        is not True
    ):
        raise CalibrationBuildError("governance completion evidence is not 30/30")
    _same_binding(
        completion.get("campaign_index"),
        receipt.get("campaign_index"),
        label="governance completion campaign index",
        require_content=True,
    )

    supervisor = execution.get("supervisor")
    _same_binding(
        verified_documents.get("retrain_execution_supervisor"),
        supervisor,
        label="governance execution supervisor document",
    )
    _same_binding(
        supervisor,
        receipt.get("supervisor"),
        label="governance execution supervisor receipt",
    )
    _live_binding(supervisor, "governance execution supervisor")

    selected_seal = execution.get("authoritative_runtime_input_seal")
    _same_binding(
        runtime_attestations.get("f3_f4_retrain_authoritative_prelaunch"),
        selected_seal,
        label="governance authoritative runtime seal document",
        require_content=True,
    )
    _same_binding(
        selected_seal,
        receipt.get("runtime_input_seal"),
        label="governance authoritative runtime seal receipt",
        require_content=True,
    )
    _live_binding(selected_seal, "governance authoritative runtime seal")
    if not isinstance(selected_seal, Mapping):
        raise CalibrationBuildError("missing governance authoritative runtime seal")
    runtime_verifier = _load_runtime_seal_module()
    live_runtime = runtime_verifier.verify(
        Path(str(selected_seal.get("path", ""))).expanduser().resolve()
    )
    try:
        verified_files_match = (
            int(live_runtime.get("verified_files", -1))
            == int(selected_seal.get("verified_files", -2))
            == int(receipt["runtime_input_seal"].get("verified_files", -3))
        )
    except (AttributeError, TypeError, ValueError) as exc:
        raise CalibrationBuildError("runtime verified-file evidence is malformed") from exc
    if (
        live_runtime.get("content_sha256") != selected_seal.get("content_sha256")
        or selected_seal.get("attestation_phase") != "prelaunch"
        or selected_seal.get("payloads_rehashed_during_this_audit") is not True
        or not verified_files_match
    ):
        raise CalibrationBuildError("authoritative runtime seal live evidence mismatch")
    return {"retrain_impact_audit": audit_binding, "governance_attestation": governance_binding}


def _prediction_rows(
    unit: Mapping[str, Any], *, outer_fold: int, seed: int
) -> dict[str, np.ndarray]:
    prediction_raw = unit.get("all_window_prediction")
    checkpoint_raw = unit.get("checkpoint")
    manifest_raw = unit.get("manifest")
    if not isinstance(prediction_raw, Mapping) or not isinstance(checkpoint_raw, Mapping):
        raise CalibrationBuildError("validated proposer unit lacks bound artifacts")
    prediction = Path(str(prediction_raw.get("path", ""))).expanduser().resolve()
    checkpoint = Path(str(checkpoint_raw.get("path", ""))).expanduser().resolve()
    manifest = Path(str(manifest_raw.get("path", ""))).expanduser().resolve()
    if sha256_file(prediction) != str(prediction_raw.get("sha256", "")):
        raise CalibrationBuildError(f"prediction binding changed: {prediction}")
    if sha256_file(checkpoint) != str(checkpoint_raw.get("sha256", "")):
        raise CalibrationBuildError(f"checkpoint binding changed: {checkpoint}")
    manifest_document, _ = _read_json(manifest, "calibration split manifest")
    expected_manifest_content_sha256 = str(
        manifest_document.get("content_sha256", "")
    )
    if len(expected_manifest_content_sha256) != 64:
        raise CalibrationBuildError("split manifest lacks a canonical content hash")
    try:
        with np.load(prediction, allow_pickle=False) as archive:
            missing = sorted(REQUIRED_ARRAYS - set(archive.files))
            if missing:
                raise CalibrationBuildError(f"prediction fields missing: {missing}")
            if not bool(_scalar(archive, "strict_retrospective")) or not bool(
                _scalar(archive, "strict_nested_prediction_role")
            ):
                raise CalibrationBuildError("calibration prediction lacks strict nested role")
            if str(_scalar(archive, "checkpoint_sha256")) != str(
                checkpoint_raw.get("sha256", "")
            ):
                raise CalibrationBuildError("prediction/checkpoint digest mismatch")
            if str(_scalar(archive, "split_manifest_file_sha256")) != str(
                unit["manifest"]["sha256"]
            ):
                raise CalibrationBuildError("prediction/manifest file digest mismatch")
            if str(_scalar(archive, "split_manifest_content_sha256")) != str(
                expected_manifest_content_sha256
            ):
                raise CalibrationBuildError("prediction/manifest content digest mismatch")
            arrays = {
                name: np.asarray(archive[name]).copy()
                for name in (
                    "cache_index",
                    "session_id",
                    "identity",
                    "window_number",
                    "reference_valid",
                    "reference_rr_bpm",
                    "prediction",
                    "rr_std",
                )
            }
    except (OSError, ValueError) as exc:
        raise CalibrationBuildError(f"invalid prediction archive: {prediction} ({exc})") from exc
    rows = len(arrays["cache_index"])
    if rows == 0 or any(value.shape != (rows,) for value in arrays.values()):
        raise CalibrationBuildError("prediction calibration arrays have inconsistent shapes")
    if len(np.unique(arrays["cache_index"].astype(np.int64))) != rows:
        raise CalibrationBuildError("prediction calibration cache indices are duplicated")
    if not np.isfinite(arrays["prediction"]).all() or not np.isfinite(
        arrays["rr_std"]
    ).all():
        raise CalibrationBuildError("prediction or rr_std contains non-finite values")
    if (np.asarray(arrays["rr_std"], float) < 0.0).any():
        raise CalibrationBuildError("rr_std contains negative values")
    identities = manifest_document.get("identities")
    if not isinstance(identities, Mapping):
        raise CalibrationBuildError("split manifest lacks identity partitions")
    predicted = set(map(str, identities.get("prediction", ())))
    train = set(map(str, identities.get("train", ())))
    validation = set(map(str, identities.get("validation", ())))
    observed = set(np.asarray(arrays["identity"]).astype(str).tolist())
    if not predicted or observed != predicted or predicted & (train | validation):
        raise CalibrationBuildError("calibration prediction identity ownership is not disjoint")
    arrays["outer_fold"] = np.full(rows, outer_fold, dtype=np.int16)
    arrays["seed"] = np.full(rows, seed, dtype=np.int64)
    return arrays


def _calibrate_group(
    units: Sequence[Mapping[str, Any]], *, outer_fold: int, seed: int
) -> dict[str, Any]:
    parts = [
        _prediction_rows(unit, outer_fold=outer_fold, seed=seed) for unit in units
    ]
    names = tuple(parts[0])
    arrays = {name: np.concatenate([part[name] for part in parts]) for name in names}
    if len(np.unique(arrays["cache_index"].astype(np.int64))) != len(
        arrays["cache_index"]
    ):
        raise CalibrationBuildError("five-way calibration stack is not an exact row partition")
    valid = np.asarray(arrays["reference_valid"], dtype=bool)
    target = np.asarray(arrays["reference_rr_bpm"], dtype=np.float64)
    prediction = np.asarray(arrays["prediction"], dtype=np.float64)
    std = np.asarray(arrays["rr_std"], dtype=np.float64)
    phase = np.asarray(arrays["window_number"], dtype=np.int64) % PHASE_MODULUS == PHASE_VALUE
    selected = valid & np.isfinite(target) & np.isfinite(prediction) & np.isfinite(std) & phase
    if int(selected.sum()) < 100:
        raise CalibrationBuildError(
            f"too few non-overlap calibration rows for {outer_fold}/{seed}: {selected.sum()}"
        )
    selected_identity = np.asarray(arrays["identity"])[selected].astype(str)
    if len(set(selected_identity.tolist())) != 15:
        raise CalibrationBuildError("calibration group must cover exactly 15 non-test identities")
    scale = np.maximum(std[selected], STD_FLOOR_BPM)
    error = np.abs(prediction[selected] - target[selected])
    score = error / scale
    intervals: dict[str, Any] = {}
    for coverage in COVERAGES:
        quantile = _higher_quantile(score, coverage)
        covered = error <= quantile * scale
        identity_coverage = {
            identity: float(np.mean(covered[selected_identity == identity]))
            for identity in sorted(set(selected_identity.tolist()))
        }
        intervals[f"{coverage:.2f}"] = {
            "nominal_coverage": coverage,
            "normalized_absolute_error_quantile": quantile,
            "finite_sample_rank": min(
                len(score), int(math.ceil((len(score) + 1) * coverage))
            ),
            "calibration_empirical_coverage": float(np.mean(covered)),
            "calibration_identity_macro_coverage": float(
                np.mean(list(identity_coverage.values()))
            ),
        }
    thresholds = {
        f"{coverage:.2f}": {
            "intended_acceptance_coverage": coverage,
            "rr_std_threshold_bpm": _ordinary_higher_quantile(std[selected], coverage),
        }
        for coverage in SELECTIVE_COVERAGES
    }
    bindings = []
    for unit in units:
        bindings.append(
            {
                "name": str(unit["name"]),
                "role": str(unit["role"]),
                "manifest": dict(unit["manifest"]),
                "checkpoint": dict(unit["checkpoint"]),
                "all_window_prediction": dict(unit["all_window_prediction"]),
            }
        )
    return {
        "outer_fold": outer_fold,
        "seed": seed,
        "source_unit_count": len(units),
        "source_units": bindings,
        "source_rows_all": int(len(target)),
        "source_rows_valid_phase_0": int(selected.sum()),
        "source_identity_count": len(set(selected_identity.tolist())),
        "source_identities": sorted(set(selected_identity.tolist())),
        "std_floor_bpm": STD_FLOOR_BPM,
        "interval_calibration": intervals,
        "selective_thresholds": thresholds,
        "calibration_score_summary": {
            "median": float(np.median(score)),
            "p90": _ordinary_higher_quantile(score, 0.90),
            "p95": _ordinary_higher_quantile(score, 0.95),
            "maximum": float(np.max(score)),
        },
    }


def _atomic_immutable_json(path: Path, value: Mapping[str, Any]) -> None:
    destination = path.expanduser().resolve()
    payload = (
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        + "\n"
    ).encode("utf-8")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if destination.is_symlink():
            raise CalibrationBuildError(
                f"immutable calibration must not be a symlink: {destination}"
            )
        if destination.read_bytes() != payload:
            raise CalibrationBuildError(f"immutable calibration already differs: {destination}")
        destination.chmod(0o444)
        return
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, destination)
        except FileExistsError as exc:
            raise CalibrationBuildError(f"immutable calibration appeared concurrently: {destination}") from exc
        destination.chmod(0o444)
    finally:
        temporary.unlink(missing_ok=True)


def build_calibration(
    *,
    plan_path: Path,
    index_path: Path,
    retrain_audit_path: Path,
    governance_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    destination = output_path.expanduser().resolve()
    # Fail closed if the locked test path has already crossed its target join.
    locked_root = PROJECT_ROOT / "artifacts/runs/harmonic_candidate_set_snn_v2/hcs_locked_oof"
    if (locked_root / "evaluation_lock.json").exists() or (
        locked_root / "locked_hcs_oof_joined.npz"
    ).exists():
        raise CalibrationBuildError("calibration must be frozen before outer-test target join")
    fixed = _load_fixed_runner_module()
    try:
        groups, source_bindings = fixed.validate_non_test_plan_index(
            plan_path.expanduser().resolve(), index_path.expanduser().resolve()
        )
    except RuntimeError as exc:
        raise CalibrationBuildError(str(exc)) from exc
    index_binding = source_bindings["index"]
    attestations = _validate_attestations(
        retrain_audit_path.expanduser().resolve(),
        governance_path.expanduser().resolve(),
        index_binding,
    )
    calibrated = []
    for outer_fold in FOLDS:
        for seed in SEEDS:
            group = groups.get((outer_fold, seed))
            if not isinstance(group, Mapping) or group.get("status") != "ready":
                raise CalibrationBuildError(f"proposer group is not ready: {outer_fold}/{seed}")
            units = group.get("units")
            if not isinstance(units, list) or len(units) != 5:
                raise CalibrationBuildError("each calibration group must bind five units")
            calibrated.append(
                _calibrate_group(units, outer_fold=outer_fold, seed=seed)
            )
    document: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "classification": "locked_pretest_cross_fitted_proposer_uncertainty_calibration",
        "commercial_claim_authorized": False,
        "prospective_confirmation_required": True,
        "outer_test_opened": False,
        "outer_test_record_count": 0,
        "target_artifact_opened": False,
        "point_prediction_modified": False,
        "calibration_scope": "uncertainty_intervals_and_prefit_selective_thresholds_only",
        "folds": list(FOLDS),
        "seeds": list(SEEDS),
        "unit_count": len(calibrated),
        "fixed_method": {
            "source": "five identity-disjoint cross-fitted non-test proposer predictions per outer-fold/seed",
            "row_filter": "reference_valid and finite and window_number modulo 8 equals 0",
            "phase_modulus": PHASE_MODULUS,
            "phase_value": PHASE_VALUE,
            "std_floor_bpm": STD_FLOOR_BPM,
            "interval_coverages": list(COVERAGES),
            "interval_score": "abs(prediction-reference)/max(rr_std,0.25 bpm)",
            "interval_quantile": "ceil((n+1)*coverage)-th sorted score, clipped to n",
            "selective_coverages": list(SELECTIVE_COVERAGES),
            "selective_rule": "accept iff test rr_std <= frozen calibration higher-quantile threshold",
            "no_test_time_fit_or_threshold_selection": True,
            "formal_exchangeability_claim": False,
        },
        "fixed_evaluation_gates": {
            "all_seeds_required": True,
            "conformal_max_absolute_calibration_error_all_levels": 0.07,
            "conformal_90_marginal_coverage_min": 0.88,
            "conformal_90_identity_macro_coverage_min": 0.85,
            "conformal_90_fixed_phase_0_coverage_min": 0.85,
            "conformal_90_mean_full_width_bpm_max": 6.0,
            "conformal_90_p95_full_width_bpm_max": 10.0,
            "selective_80_mae_bpm_max": 1.0,
            "selective_80_catastrophic_over_5_max": 0.03,
        },
        "inputs": {
            **source_bindings,
            **attestations,
            "builder": bind_file(Path(__file__)),
        },
        "units": calibrated,
        "limitations": [
            "checkpoint-transfer and clustered-window dependence prevent a formal marginal-coverage guarantee",
            "this is retrospective engineering calibration, not independent prospective validation",
            "calibration does not alter or rescue the locked primary point prediction",
        ],
    }
    document["content_sha256"] = canonical_content_sha256(document)
    _atomic_immutable_json(destination, document)
    return document


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--retrain-audit", type=Path, default=DEFAULT_RETRAIN_AUDIT)
    parser.add_argument("--governance", type=Path, default=DEFAULT_GOVERNANCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = build_calibration(
            plan_path=args.plan,
            index_path=args.index,
            retrain_audit_path=args.retrain_audit,
            governance_path=args.governance,
            output_path=args.output,
        )
    except (CalibrationBuildError, RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
