#!/usr/bin/env python3
"""Fail-closed aggregate release-readiness decision for locked HCS evidence.

``freeze-spec`` is a target-free operation.  It fixes every evidence role,
threshold, source digest, CUDA applicability rule, and the no-selection policy
before the canonical target can be released.  ``evaluate`` reads and verifies
that specification first, then only aggregates already-published evidence.  It
never opens prediction or target arrays and never recomputes target metrics.

The strongest positive result emitted here is an *internal retrospective
engineering candidate*.  A commercial release is unconditionally blocked
until a separately locked independent prospective cohort exists.
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
import stat
import sys
from typing import Any, Iterable, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCHEMA_VERSION = 1
FOLDS = tuple(range(6))
SEEDS = (20260828, 20260829, 20260830)
MASKS = (
    "radars_123", "radars_12", "radars_13", "radars_23",
    "radar_1", "radar_2", "radar_3",
)

ACCURACY_GATES = {
    "overall_mae_max_bpm": 1.0,
    "identity_macro_mae_max_bpm": 1.0,
    "overall_rmse_max_bpm": 1.8,
    "within_2_min_fraction": 0.90,
    "over_5_max_fraction": 0.03,
    "high_rr_25_35_mae_max_bpm": 2.0,
}
ACCURACY_CHECKS = {
    "overall_mae": ("<=", ACCURACY_GATES["overall_mae_max_bpm"]),
    "identity_macro_mae": ("<=", ACCURACY_GATES["identity_macro_mae_max_bpm"]),
    "overall_rmse": ("<=", ACCURACY_GATES["overall_rmse_max_bpm"]),
    "within_2_fraction": (">=", ACCURACY_GATES["within_2_min_fraction"]),
    "over_5_fraction": ("<=", ACCURACY_GATES["over_5_max_fraction"]),
    "tail_25_35_mae": ("<=", ACCURACY_GATES["high_rr_25_35_mae_max_bpm"]),
}
UNCERTAINTY_GATES = {
    "all_seeds_required": True,
    "conformal_max_absolute_calibration_error_all_levels": 0.07,
    "conformal_90_marginal_coverage_min": 0.88,
    "conformal_90_identity_macro_coverage_min": 0.85,
    "conformal_90_fixed_phase_0_coverage_min": 0.85,
    "conformal_90_mean_full_width_bpm_max": 6.0,
    "conformal_90_p95_full_width_bpm_max": 10.0,
    "selective_80_mae_bpm_max": 1.0,
    "selective_80_catastrophic_over_5_max": 0.03,
}
ENGINEERING_GATES = {
    "cpu_warm_p99_ms_max": 250.0,
    "cuda_warm_p99_ms_max": 50.0,
    "stride_budget_ms": 4000.0,
    "p99_stride_budget_fraction_max": 0.10,
    "checkpoint_bytes_max": 50 * 1024 * 1024,
    "parameter_count_max": 5_000_000,
    "cpu_process_peak_rss_bytes_max": 2 * 1024**3,
    "cuda_peak_reserved_bytes_max": 1024**3,
    "spike_rate_diagnostic_min": 0.01,
    "spike_rate_diagnostic_max": 0.20,
    "spike_rate_unavailable_policy": "reported_not_applicable_without_failure",
}
READINESS_ENGINEERING_GATES = {
    "spike_telemetry_required_for_every_unit": True,
    "spike_rate_must_be_finite": True,
    "spike_rate_minimum_inclusive": 0.01,
    "spike_rate_maximum_inclusive": 0.20,
    "missing_or_unavailable_spike_policy": "fail_internal_candidate",
}
MANDATORY_STREAMING_GATES = (
    "locked_artifact_hashes",
    "whole_chunk_one_window_parity",
    "explicit_session_reset",
    "finite_outputs",
    "seven_nonempty_structural_radar_masks",
    "no_candidate_structural_fallback",
    "corrupt_input_fail_closed",
)

PRIMARY_ROOT = PROJECT_ROOT / "artifacts/runs/harmonic_candidate_set_snn_v2/hcs_locked_oof"
MASK_ROOT = PROJECT_ROOT / "artifacts/runs/harmonic_candidate_set_snn_v2/hcs_locked_radar_masks"
CAMPAIGN_ROOT = PROJECT_ROOT / "artifacts/campaigns/harmonic_candidate_set_snn_v2"
PRETEST_ROOT = PROJECT_ROOT / "artifacts/runs/harmonic_candidate_set_snn_v2/hcs_fixed_i3_pretest"
STREAMING_ROOT = PROJECT_ROOT / "artifacts/runs/harmonic_candidate_set_snn_v2/hcs_streaming_deployment"
BENCHMARK_ROOT = PROJECT_ROOT / "artifacts/benchmarks/locked_proposer_deployment"

DEFAULT_SPEC = CAMPAIGN_ROOT / "locked_hcs_release_readiness_spec.json"
DEFAULT_OUTPUT_DIR = PRIMARY_ROOT / "release_readiness"
DEFAULT_ROLES: dict[str, Path | None] = {
    "target_release_receipt": PRIMARY_ROOT / "canonical_locked_hcs_targets_release_receipt.json",
    "pretarget_release_lock": PRIMARY_ROOT / "pretarget_release_lock.json",
    "primary_evaluation_lock": PRIMARY_ROOT / "evaluation_lock.json",
    "canonical_target": PRIMARY_ROOT / "canonical_locked_hcs_targets.npz",
    "canonical_target_receipt": PRIMARY_ROOT / "canonical_locked_hcs_targets_receipt.json",
    "joined_output": PRIMARY_ROOT / "locked_hcs_oof_joined.npz",
    "predictions_seal": PRIMARY_ROOT / "predictions_seal.json",
    "release_evaluation_execution_attestation": (
        PRIMARY_ROOT / "release_evaluation_execution_attestation.json"
    ),
    "primary_report": PRIMARY_ROOT / "primary_evaluation/locked_hcs_primary_evaluation.json",
    "primary_receipt": PRIMARY_ROOT / "primary_evaluation/locked_hcs_primary_evaluation_receipt.json",
    "radar_report": MASK_ROOT / "evaluation/locked_hcs_radar_masks_evaluation.json",
    "radar_receipt": MASK_ROOT / "evaluation/locked_hcs_radar_masks_evaluation_receipt.json",
    "uncertainty_spec": CAMPAIGN_ROOT / "locked_uncertainty_evaluation_spec.json",
    "primary_evaluation_spec": CAMPAIGN_ROOT / "locked_primary_evaluation_spec.json",
    "uncertainty_report": PRIMARY_ROOT / "uncertainty_evaluation/uncertainty_report.json",
    "uncertainty_receipt": PRIMARY_ROOT / "uncertainty_evaluation/uncertainty_receipt.json",
    "streaming_complete_seal": STREAMING_ROOT / "complete_seal.json",
    "radar_mask_complete_seal": MASK_ROOT / "complete_seal.json",
    "uncertainty_inputs_seal": PRIMARY_ROOT / "uncertainty_inputs_seal.json",
    "proposer_cpu_complete_seal": BENCHMARK_ROOT / "cpu/complete_seal.json",
    "proposer_cuda_complete_seal": BENCHMARK_ROOT / "cuda/complete_seal.json",
    "fixed_i3_runtime_seal": CAMPAIGN_ROOT / "nested_proposer/current_source_merged/fixed_i3_pretest_runtime_seal.json",
    "fixed_runtime_completion": PRETEST_ROOT / "fixed_runtime_completion_attestation.json",
    "postlock_runtime_guard": PRIMARY_ROOT / "postlock_runtime_guard_attestation.json",
    "radar_mask_runtime_guard": MASK_ROOT / "radar_mask_runtime_guard_attestation.json",
    "commercial_execution_plan": (
        PROJECT_ROOT / "artifacts/COMMERCIAL_SNN_CONTINUOUS_EXECUTION_PLAN_V4.md"
    ),
}
SOURCE_PATHS = {
    "release_readiness_evaluator": Path(__file__).resolve(),
    "primary_evaluator": PROJECT_ROOT / "scripts/evaluate_locked_hcs_oof.py",
    "radar_mask_evaluator": PROJECT_ROOT / "scripts/evaluate_locked_hcs_radar_masks.py",
    "uncertainty_spec_builder": PROJECT_ROOT / "scripts/freeze_locked_hcs_uncertainty_evaluation_spec.py",
    "uncertainty_evaluator": PROJECT_ROOT / "scripts/evaluate_locked_hcs_uncertainty.py",
    "streaming_campaign": PROJECT_ROOT / "scripts/run_locked_hcs_streaming_campaign.py",
    "proposer_deployment_benchmark": PROJECT_ROOT / "scripts/benchmark_locked_proposer_deployment_v3.py",
    "pretarget_release_lock_creator": PROJECT_ROOT / "scripts/create_locked_hcs_pretarget_release_lock.py",
    "target_after_release_builder": PROJECT_ROOT / "scripts/build_locked_hcs_targets_after_release_lock.py",
    "release_evaluation_orchestrator": PROJECT_ROOT / "scripts/run_release_locked_hcs_evaluation.py",
    "python_executable": Path(sys.executable).resolve(),
}


class ReleaseReadinessError(RuntimeError):
    """An evidence role, immutable binding, or fixed gate is invalid."""


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def bind_file(path: Path) -> dict[str, Any]:
    raw = path.expanduser()
    if raw.is_symlink():
        raise ReleaseReadinessError(f"bound artifact is a symlink: {raw}")
    resolved = raw.resolve()
    if not resolved.is_file():
        raise ReleaseReadinessError(f"bound artifact is absent: {resolved}")
    return {"path": str(resolved), "sha256": sha256_file(resolved), "bytes": resolved.stat().st_size}


def _json_value(path: Path, label: str) -> Any:
    def pairs(values: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in values:
            if key in result:
                raise ReleaseReadinessError(f"{label} repeats JSON key {key!r}")
            result[key] = value
        return result

    def nonfinite(value: str) -> None:
        raise ReleaseReadinessError(f"{label} contains non-finite value {value}")

    try:
        value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=pairs,
                           parse_constant=nonfinite)
    except (OSError, json.JSONDecodeError) as exc:
        raise ReleaseReadinessError(f"invalid {label}: {path} ({exc})") from exc
    return value


def _json(path: Path, label: str) -> dict[str, Any]:
    value = _json_value(path, label)
    if not isinstance(value, dict):
        raise ReleaseReadinessError(f"{label} must be a JSON object")
    return value


def _validate_content(document: Mapping[str, Any], label: str) -> None:
    if "content_sha256" not in document:
        return
    payload = dict(document)
    recorded = str(payload.pop("content_sha256"))
    # Runtime input inventories intentionally exclude their wall-clock creation
    # timestamp from the semantic digest (see seal_runtime_inputs.py).  Keep
    # this compatibility rule classification-scoped so no other document can
    # silently omit a field from its authenticated content.
    if document.get("classification") == "supplemental_runtime_input_byte_inventory":
        payload.pop("created_utc", None)
    if len(recorded) != 64 or canonical_sha256(payload) != recorded:
        raise ReleaseReadinessError(f"{label} content hash mismatch")


def _same_binding(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    try:
        return (Path(str(left["path"])).resolve() == Path(str(right["path"])).resolve()
                and str(left["sha256"]) == str(right["sha256"])
                and int(left["bytes"]) == int(right["bytes"]))
    except (KeyError, TypeError, ValueError):
        return False


def _verify_binding(raw: Any, *, label: str) -> dict[str, Any]:
    if not isinstance(raw, Mapping) or not {"path", "sha256"} <= set(raw):
        raise ReleaseReadinessError(f"missing binding: {label}")
    bound_path = Path(str(raw["path"])).expanduser()
    lexical_path = bound_path if bound_path.is_absolute() else (Path.cwd() / bound_path)
    if lexical_path.is_symlink():
        # The deployment protocols deliberately retain the venv launcher path
        # while hashing the interpreter bytes.  Permit only that one canonical
        # launcher symlink; all other bound symlinks remain fail-closed.
        allowed_launcher = PROJECT_ROOT / ".venv/bin/python"
        if lexical_path.absolute() != allowed_launcher.absolute():
            raise ReleaseReadinessError(f"bound artifact is a symlink: {lexical_path}")
        resolved_target = lexical_path.resolve()
        if not resolved_target.is_file():
            raise ReleaseReadinessError(
                f"bound Python launcher target is not a regular file: {lexical_path}"
            )
        observed = {
            "path": str(lexical_path.absolute()),
            "sha256": sha256_file(resolved_target),
            "bytes": resolved_target.stat().st_size,
        }
    else:
        observed = bind_file(lexical_path)
    if str(raw.get("sha256")) != observed["sha256"] or (
        "bytes" in raw and int(raw["bytes"]) != observed["bytes"]
    ):
        raise ReleaseReadinessError(f"binding changed: {label}")
    return observed


def _is_binding(value: Any) -> bool:
    return isinstance(value, Mapping) and {"path", "sha256"} <= set(value)


def _verify_closure(document: Mapping[str, Any], *, label: str,
                    visited: set[Path] | None = None) -> None:
    """Rehash every reachable binding and every reachable JSON content digest."""
    _validate_content(document, label)
    seen = visited if visited is not None else set()

    classification = str(document.get("classification", ""))

    def walk(value: Any, name: str, *, descend_json: bool = True) -> None:
        if _is_binding(value):
            binding = _verify_binding(value, label=name)
            path = Path(binding["path"])
            if descend_json and path.suffix.lower() == ".json" and path not in seen:
                seen.add(path)
                nested = _json_value(path, name)
                if isinstance(nested, Mapping):
                    _verify_closure(nested, label=name, visited=seen)
                elif isinstance(nested, list):
                    # JSON arrays (for example epoch-history payloads) are
                    # legitimate byte-bound artifacts.  Traverse any bindings
                    # they contain without requiring an object-level digest.
                    walk(nested, name)
            return
        if isinstance(value, Mapping):
            for key, item in value.items():
                child_descend = descend_json and not (
                    classification
                    == "non_test_proposer_execution_runtime_seal_supersession"
                    and key == "superseded_runtime_seals"
                )
                walk(item, f"{name}.{key}", descend_json=child_descend)
        elif isinstance(value, list):
            for index, item in enumerate(value):
                walk(item, f"{name}[{index}]", descend_json=descend_json)

    walk(document, label)


def _require_mode_0444(path: Path, label: str) -> None:
    raw = path.expanduser()
    if raw.is_symlink():
        raise ReleaseReadinessError(f"{label} must be a regular mode-0444 file: {raw}")
    resolved = raw.resolve()
    if not resolved.is_file() or stat.S_IMODE(resolved.stat().st_mode) != 0o444:
        raise ReleaseReadinessError(f"{label} must be a regular mode-0444 file: {resolved}")


def _content_document(value: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(value)
    result.pop("content_sha256", None)
    result["content_sha256"] = canonical_sha256(result)
    return result


def _load_bound_json(binding: Any, label: str) -> tuple[dict[str, Any], dict[str, Any]]:
    verified = _verify_binding(binding, label=label)
    document = _json(Path(verified["path"]), label)
    _verify_closure(document, label=label)
    return document, verified


def _resolve_roles(roles: Mapping[str, Path | None]) -> dict[str, str | None]:
    if set(roles) != set(DEFAULT_ROLES):
        raise ReleaseReadinessError("aggregate input-role topology differs")
    return {
        name: None if path is None else str(path.expanduser().resolve())
        for name, path in sorted(roles.items())
    }


def _source_bindings() -> dict[str, dict[str, Any]]:
    return {name: bind_file(path) for name, path in sorted(SOURCE_PATHS.items())}


def _assert_class(document: Mapping[str, Any], expected: str, label: str) -> None:
    if document.get("classification") != expected:
        raise ReleaseReadinessError(f"{label} classification differs")


def _bound_plan(seal: Mapping[str, Any], label: str) -> dict[str, Any]:
    plan, _ = _load_bound_json(seal.get("plan"), f"{label}.plan")
    return plan


def _streaming_cuda_required(streaming: Mapping[str, Any]) -> bool:
    plan = _bound_plan(streaming, "streaming seal")
    freeze, freeze_binding = _load_bound_json(plan.get("freeze_spec"), "streaming freeze spec")
    if not _same_binding(freeze_binding, streaming.get("freeze_spec", {})):
        raise ReleaseReadinessError("streaming plan/seal freeze-spec binding differs")
    _assert_class(freeze, "locked_hcs_streaming_deployment_freeze_spec", "streaming freeze spec")
    value = freeze.get("runtime_identity", {}).get("cuda_available_at_freeze")
    if not isinstance(value, bool):
        raise ReleaseReadinessError("streaming freeze spec lacks CUDA applicability")
    return value


def _deployment_device(seal: Mapping[str, Any], label: str) -> str:
    plan = _bound_plan(seal, label)
    device = str(plan.get("runtime", {}).get("device_type", ""))
    if device not in {"cpu", "cuda"}:
        raise ReleaseReadinessError(f"{label} plan device is invalid")
    return device


def freeze_spec(*, output: Path, roles: Mapping[str, Path | None]) -> dict[str, Any]:
    """Publish the immutable aggregate protocol without touching any target artifact."""
    destination = output.expanduser().resolve()
    if destination.exists():
        raise ReleaseReadinessError(f"immutable release-readiness spec already exists: {destination}")
    resolved = _resolve_roles(roles)
    # These are the only documents opened here.  They are target-free and must
    # already be sealed.  All target-derived/report roles must still be absent.
    pretarget_names = (
        "uncertainty_spec", "primary_evaluation_spec", "streaming_complete_seal",
        "proposer_cpu_complete_seal", "predictions_seal", "radar_mask_complete_seal",
        "uncertainty_inputs_seal",
        "commercial_execution_plan",
        "fixed_i3_runtime_seal",
        "fixed_runtime_completion", "postlock_runtime_guard", "radar_mask_runtime_guard",
    )
    pretarget: dict[str, dict[str, Any]] = {}
    documents: dict[str, dict[str, Any]] = {}
    for name in pretarget_names:
        path = Path(str(resolved[name]))
        _require_mode_0444(path, name)
        if path.suffix.lower() == ".json":
            document = _json(path, name)
            _verify_closure(document, label=name)
            documents[name] = document
        pretarget[name] = bind_file(path)
    _assert_class(documents["uncertainty_spec"], "locked_hcs_secondary_uncertainty_evaluation_specification", "uncertainty spec")
    _assert_class(documents["primary_evaluation_spec"], "locked_hcs_oof_primary_evaluation_specification", "primary evaluation spec")
    _assert_class(documents["streaming_complete_seal"], "locked_hcs_streaming_deployment_all_18_complete_seal", "streaming seal")
    _assert_class(documents["predictions_seal"], "locked_hcs_oof_all_label_free_predictions_sealed", "primary predictions seal")
    _assert_class(documents["radar_mask_complete_seal"], "locked_hcs_all_seven_radar_mask_predictions_sealed", "radar-mask complete seal")
    _assert_class(documents["uncertainty_inputs_seal"], "locked_hcs_all_target_free_uncertainty_inputs_sealed", "uncertainty inputs seal")
    _assert_class(documents["proposer_cpu_complete_seal"], "locked_proposer_deployment_all_18_complete_seal", "CPU deployment seal")
    if _deployment_device(documents["proposer_cpu_complete_seal"], "CPU deployment seal") != "cpu":
        raise ReleaseReadinessError("CPU deployment role is not a CPU campaign")
    cuda_required = _streaming_cuda_required(documents["streaming_complete_seal"])
    cuda_path_raw = resolved["proposer_cuda_complete_seal"]
    if cuda_required:
        if cuda_path_raw is None or not Path(cuda_path_raw).is_file():
            raise ReleaseReadinessError("CUDA was available at freeze; a CUDA proposer seal is required")
        cuda_path = Path(cuda_path_raw)
        _require_mode_0444(cuda_path, "CUDA deployment seal")
        cuda_doc = _json(cuda_path, "CUDA deployment seal")
        _verify_closure(cuda_doc, label="CUDA deployment seal")
        _assert_class(cuda_doc, "locked_proposer_deployment_all_18_complete_seal", "CUDA deployment seal")
        if _deployment_device(cuda_doc, "CUDA deployment seal") != "cuda":
            raise ReleaseReadinessError("CUDA deployment role is not a CUDA campaign")
        pretarget["proposer_cuda_complete_seal"] = bind_file(cuda_path)
    elif cuda_path_raw is not None and Path(cuda_path_raw).exists():
        raise ReleaseReadinessError("CUDA seal supplied although frozen CUDA applicability is false")

    future_roles = (
        "pretarget_release_lock", "target_release_receipt", "primary_evaluation_lock",
        "release_evaluation_execution_attestation",
        "canonical_target", "canonical_target_receipt", "joined_output",
        "primary_report", "primary_receipt", "radar_report", "radar_receipt",
        "uncertainty_report", "uncertainty_receipt",
    )
    for name in future_roles:
        if Path(str(resolved[name])).exists():
            raise ReleaseReadinessError(
                f"freeze-spec must precede target access/evaluation publication: {name} exists"
            )

    value = _content_document({
        "schema_version": SCHEMA_VERSION,
        "classification": "locked_hcs_release_readiness_aggregate_specification",
        "must_be_frozen_before_target_access": True,
        "target_or_target_bearing_artifact_opened": False,
        "input_roles": resolved,
        "pretarget_input_bindings": pretarget,
        "fixed_seeds": list(SEEDS),
        "fixed_folds": list(FOLDS),
        "fixed_radar_masks": list(MASKS),
        "exact_gates": {
            "accuracy": ACCURACY_GATES,
            "uncertainty": UNCERTAINTY_GATES,
            "engineering": ENGINEERING_GATES,
            "release_readiness_engineering": READINESS_ENGINEERING_GATES,
            "mandatory_streaming": list(MANDATORY_STREAMING_GATES),
        },
        "cuda_applicability": {
            "source": "streaming_freeze_spec.runtime_identity.cuda_available_at_freeze",
            "required": cuda_required,
            "separate_cuda_proposer_complete_seal_required_when_true": True,
            "missing_required_cuda_policy": "fail_closed",
        },
        "decision_rules": {
            "all_three_fixed_seeds_required_independently": True,
            "all_six_accuracy_gates_per_seed_required": True,
            "all_seven_masks_per_seed_required": True,
            "all_uncertainty_gates_per_seed_required": True,
            "all_18_streaming_units_required": True,
            "all_18_cpu_proposer_units_required": True,
            "all_18_cuda_proposer_units_required_when_applicable": True,
            "model_seed_mask_ranking_selection_pooling_averaging_suppression_allowed": False,
            "missing_or_tampered_input_policy": "fail_closed_no_output",
        },
        "prospective_release_policy": {
            "independent_prospective_cohort_present": False,
            "commercial_release_ready_must_equal": False,
            "blocked_reason": "independent prospective cohort has not been collected and locked",
        },
        "implementation_sources": _source_bindings(),
    })
    _publish_json_create_once(destination, value)
    _require_mode_0444(destination, "release-readiness specification")
    return value


def _load_spec(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    raw = path.expanduser()
    _require_mode_0444(raw, "release-readiness specification")
    resolved = raw.resolve()
    spec = _json(resolved, "release-readiness specification")
    _validate_content(spec, "release-readiness specification")
    _assert_class(spec, "locked_hcs_release_readiness_aggregate_specification", "release-readiness specification")
    expected_decisions = {
        "all_three_fixed_seeds_required_independently": True,
        "all_six_accuracy_gates_per_seed_required": True,
        "all_seven_masks_per_seed_required": True,
        "all_uncertainty_gates_per_seed_required": True,
        "all_18_streaming_units_required": True,
        "all_18_cpu_proposer_units_required": True,
        "all_18_cuda_proposer_units_required_when_applicable": True,
        "model_seed_mask_ranking_selection_pooling_averaging_suppression_allowed": False,
        "missing_or_tampered_input_policy": "fail_closed_no_output",
    }
    expected_prospective = {
        "independent_prospective_cohort_present": False,
        "commercial_release_ready_must_equal": False,
        "blocked_reason": "independent prospective cohort has not been collected and locked",
    }
    roles = spec.get("input_roles")
    if (not isinstance(roles, Mapping) or set(roles) != set(DEFAULT_ROLES)
            or spec.get("must_be_frozen_before_target_access") is not True
            or spec.get("target_or_target_bearing_artifact_opened") is not False
            or spec.get("fixed_seeds") != list(SEEDS)
            or spec.get("fixed_folds") != list(FOLDS)
            or spec.get("fixed_radar_masks") != list(MASKS)
            or spec.get("decision_rules") != expected_decisions
            or spec.get("prospective_release_policy") != expected_prospective
            or spec.get("exact_gates") != {
                "accuracy": ACCURACY_GATES, "uncertainty": UNCERTAINTY_GATES,
                "engineering": ENGINEERING_GATES,
                "release_readiness_engineering": READINESS_ENGINEERING_GATES,
                "mandatory_streaming": list(MANDATORY_STREAMING_GATES),
            }):
        raise ReleaseReadinessError("release-readiness specification drifted")
    expected_sources = spec.get("implementation_sources")
    if not isinstance(expected_sources, Mapping) or set(expected_sources) != set(SOURCE_PATHS):
        raise ReleaseReadinessError("release-readiness source topology differs")
    for name, path_value in SOURCE_PATHS.items():
        observed = bind_file(path_value)
        if not _same_binding(observed, expected_sources[name]):
            raise ReleaseReadinessError(f"release-readiness implementation source changed: {name}")
    pretarget = spec.get("pretarget_input_bindings")
    if not isinstance(pretarget, Mapping):
        raise ReleaseReadinessError("release-readiness pretarget bindings are absent")
    cuda = spec.get("cuda_applicability")
    if not isinstance(cuda, Mapping) or not isinstance(cuda.get("required"), bool):
        raise ReleaseReadinessError("release-readiness CUDA applicability is absent")
    expected_pretarget = {
        "uncertainty_spec", "primary_evaluation_spec", "streaming_complete_seal",
        "proposer_cpu_complete_seal", "predictions_seal", "radar_mask_complete_seal",
        "uncertainty_inputs_seal",
        "commercial_execution_plan",
        "fixed_i3_runtime_seal", "fixed_runtime_completion", "postlock_runtime_guard",
        "radar_mask_runtime_guard",
    } | ({"proposer_cuda_complete_seal"} if cuda["required"] else set())
    if set(pretarget) != expected_pretarget:
        raise ReleaseReadinessError("release-readiness pretarget binding topology differs")
    for name, binding in pretarget.items():
        observed = _verify_binding(binding, label=f"pretarget input {name}")
        if not _same_binding(observed, binding):
            raise ReleaseReadinessError(f"pretarget input changed: {name}")
        if Path(observed["path"]).suffix.lower() == ".json":
            document = _json(Path(observed["path"]), f"pretarget input {name}")
            _verify_closure(document, label=f"pretarget input {name}")
    streaming, _ = _load_bound_json(
        pretarget["streaming_complete_seal"], "frozen readiness streaming seal"
    )
    derived_cuda = _streaming_cuda_required(streaming)
    expected_cuda = {
        "source": "streaming_freeze_spec.runtime_identity.cuda_available_at_freeze",
        "required": derived_cuda,
        "separate_cuda_proposer_complete_seal_required_when_true": True,
        "missing_required_cuda_policy": "fail_closed",
    }
    if cuda != expected_cuda:
        raise ReleaseReadinessError("release-readiness frozen CUDA policy differs")
    return spec, bind_file(resolved)


def load_release_readiness_spec(path: Path = DEFAULT_SPEC) -> tuple[dict[str, Any], dict[str, Any]]:
    """Public target-free loader used by the pretarget release-lock creator."""
    return _load_spec(path)


def _gate_pass(value: Any, operator: str, threshold: float) -> bool:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    if not math.isfinite(number):
        return False
    return number <= threshold if operator == "<=" else number >= threshold


def _accuracy_decision(decision: Any, label: str) -> bool:
    if not isinstance(decision, Mapping) or set(decision.get("checks", {})) != set(ACCURACY_CHECKS):
        raise ReleaseReadinessError(f"{label} accuracy gate topology differs")
    passed = []
    for name, (operator, threshold) in ACCURACY_CHECKS.items():
        row = decision["checks"][name]
        if (not isinstance(row, Mapping) or row.get("operator") != operator
                or float(row.get("threshold", math.nan)) != threshold):
            raise ReleaseReadinessError(f"{label} accuracy threshold differs: {name}")
        expected = _gate_pass(row.get("value"), operator, threshold)
        if row.get("passed") is not expected:
            raise ReleaseReadinessError(f"{label} accuracy decision is inconsistent: {name}")
        passed.append(expected)
    result = all(passed)
    if decision.get("all_point_gates_passed") is not result:
        raise ReleaseReadinessError(f"{label} aggregate accuracy decision is inconsistent")
    return result


def _validate_primary(report: Mapping[str, Any]) -> tuple[bool, list[dict[str, Any]]]:
    _assert_class(report, "retrospective_locked_hcs_oof_primary_evaluation", "primary report")
    if (report.get("commercial_claim_authorized") is not False
            or report.get("commercial_performance_proven") is not False
            or report.get("prospective_confirmation_required") is not True
            or report.get("independent_prospective_cohort_evaluated") is not False
            or report.get("goal_targets") != ACCURACY_GATES
            or report.get("selection_or_retraining_performed") is not False
            or report.get("seed_ranking_or_suppression_performed") is not False
            or report.get("cross_seed_pooling_performed") is not False):
        raise ReleaseReadinessError("primary evaluation permits drift or selection")
    per_seed = report.get("per_seed")
    if not isinstance(per_seed, Mapping) or set(per_seed) != {str(seed) for seed in SEEDS}:
        raise ReleaseReadinessError("primary report fixed-seed topology differs")
    rows = []
    statuses = {}
    for seed in SEEDS:
        seed_report = per_seed[str(seed)]
        if (seed_report.get("seed") != seed or seed_report.get("seed_evaluated_independently") is not True
                or seed_report.get("cross_seed_pooling_performed") is not False):
            raise ReleaseReadinessError(f"primary seed independence differs: {seed}")
        status = _accuracy_decision(seed_report.get("locked_final_goal"), f"primary seed {seed}")
        statuses[str(seed)] = status
        rows.append({"category": "primary_accuracy", "condition": str(seed), "passed": status})
    if report.get("fixed_seed_gate_status") != statuses or report.get(
        "all_fixed_seeds_point_gates_passed") is not all(statuses.values()):
        raise ReleaseReadinessError("primary fixed-seed aggregate differs")
    return all(statuses.values()), rows


def _validate_radar(report: Mapping[str, Any]) -> tuple[bool, list[dict[str, Any]]]:
    _assert_class(report, "retrospective_locked_hcs_all_radar_masks_evaluation", "radar report")
    if (report.get("commercial_claim_authorized") is not False
            or report.get("commercial_performance_proven") is not False
            or report.get("prospective_confirmation_required") is not True
            or report.get("independent_prospective_cohort_evaluated") is not False
            or report.get("mask_selection_or_ranking_performed") is not False
            or report.get("seed_pooling_ranking_or_suppression_performed") is not False
            or report.get("all_seven_masks_are_required_fixed_conditions") is not True):
        raise ReleaseReadinessError("radar-mask report permits selection or omission")
    parity = report.get("radars_123_primary_parity_gate")
    if (not isinstance(parity, Mapping) or parity.get("required") is not True
            or parity.get("all_fixed_seeds_passed") is not True
            or set(parity.get("per_seed", {})) != {str(seed) for seed in SEEDS}):
        raise ReleaseReadinessError("full-radar primary parity gate failed or is incomplete")
    for seed in SEEDS:
        item = parity["per_seed"][str(seed)]
        if (item.get("passed") is not True or item.get("locked_final_metrics_exact") is not True
                or item.get("array_bit_exact") != {
                    "fallback_rr_bpm": True, "source_rr_bpm": True, "final_rr_bpm": True
                }):
            raise ReleaseReadinessError(f"full-radar parity differs: seed {seed}")
    per_seed = report.get("per_seed")
    if not isinstance(per_seed, Mapping) or set(per_seed) != {str(seed) for seed in SEEDS}:
        raise ReleaseReadinessError("radar fixed-seed topology differs")
    rows = []
    statuses = []
    for seed in SEEDS:
        item = per_seed[str(seed)]
        masks = item.get("radar_masks")
        if (item.get("seed") != seed or item.get("seed_evaluated_independently") is not True
                or item.get("all_seven_masks_reported_without_selection") is not True
                or not isinstance(masks, Mapping) or set(masks) != set(MASKS)):
            raise ReleaseReadinessError(f"radar mask preservation differs: seed {seed}")
        mask_status = []
        for mask in MASKS:
            status = _accuracy_decision(masks[mask].get("fixed_point_goal_gate"), f"radar {seed}/{mask}")
            mask_status.append(status)
            rows.append({"category": "radar_mask_accuracy", "condition": f"{seed}/{mask}", "passed": status})
        aggregate = all(mask_status)
        if item.get("all_seven_masks_fixed_point_gates_passed") is not aggregate:
            raise ReleaseReadinessError(f"radar mask aggregate differs: seed {seed}")
        statuses.append(aggregate)
    result = all(statuses)
    if report.get("all_masks_all_fixed_seeds_point_gates_passed") is not result:
        raise ReleaseReadinessError("radar all-mask aggregate differs")
    return result, rows


def _validate_uncertainty(report: Mapping[str, Any], spec: Mapping[str, Any]) -> tuple[bool, list[dict[str, Any]]]:
    _assert_class(report, "retrospective_locked_hcs_uncertainty_evaluation", "uncertainty report")
    if (report.get("commercial_claim_authorized") is not False
            or report.get("commercial_performance_proven") is not False
            or report.get("prospective_confirmation_required") is not True
            or report.get("independent_prospective_cohort_evaluated") is not False
            or report.get("frozen_evaluation_gates") != UNCERTAINTY_GATES
            or report.get("selection_retraining_or_test_time_fitting_performed") is not False
            or report.get("interval_scale_or_threshold_refit_performed") is not False
            or report.get("seed_pooling_ranking_or_suppression_performed") is not False
            or report.get("point_prediction_modified") is not False):
        raise ReleaseReadinessError("uncertainty report permits fitting, selection, or gate drift")
    if spec.get("fixed_evaluation_gates") != UNCERTAINTY_GATES:
        raise ReleaseReadinessError("dedicated uncertainty specification gate drift")
    per_seed = report.get("per_seed")
    if not isinstance(per_seed, Mapping) or set(per_seed) != {str(seed) for seed in SEEDS}:
        raise ReleaseReadinessError("uncertainty fixed-seed topology differs")
    statuses: dict[str, bool] = {}
    rows = []
    expected_names = set(UNCERTAINTY_GATES) - {"all_seeds_required"}
    for seed in SEEDS:
        decision = per_seed[str(seed)].get("fixed_gate_decision")
        gates = decision.get("gates", {}) if isinstance(decision, Mapping) else {}
        if set(gates) != expected_names:
            raise ReleaseReadinessError(f"uncertainty gate topology differs: {seed}")
        values = []
        for name in sorted(expected_names):
            row = gates[name]
            operator = ">=" if name.endswith("_min") else "<="
            threshold = float(UNCERTAINTY_GATES[name])
            if row.get("operator") != operator or float(row.get("threshold", math.nan)) != threshold:
                raise ReleaseReadinessError(f"uncertainty threshold differs: {seed}/{name}")
            expected = _gate_pass(row.get("observed"), operator, threshold)
            if row.get("passed") is not expected:
                raise ReleaseReadinessError(f"uncertainty decision inconsistent: {seed}/{name}")
            values.append(expected)
        status = all(values)
        if decision.get("all_gates_passed") is not status:
            raise ReleaseReadinessError(f"uncertainty seed aggregate differs: {seed}")
        statuses[str(seed)] = status
        rows.append({"category": "uncertainty", "condition": str(seed), "passed": status})
    result = all(statuses.values())
    if report.get("fixed_seed_gate_status") != statuses or report.get(
        "all_fixed_seed_uncertainty_gates_passed") is not result:
        raise ReleaseReadinessError("uncertainty all-seed aggregate differs")
    return result, rows


def _bool_gate(row: Any, *, value_key: str, limit_key: str, limit: float,
               operator: str, label: str) -> bool:
    if not isinstance(row, Mapping) or float(row.get(limit_key, math.nan)) != float(limit):
        raise ReleaseReadinessError(f"{label} fixed threshold differs")
    expected = _gate_pass(row.get(value_key), operator, float(limit))
    if row.get("pass") is not expected:
        raise ReleaseReadinessError(f"{label} decision is inconsistent")
    return expected


def _validate_streaming_engineering(gates: Mapping[str, Any], label: str) -> bool:
    engineering = gates.get("engineering")
    if not isinstance(engineering, Mapping):
        raise ReleaseReadinessError(f"{label} streaming engineering gates are absent")
    latency = engineering.get("latency")
    if not isinstance(latency, Mapping) or set(latency) != {"cpu", "cuda"}:
        raise ReleaseReadinessError(f"{label} streaming latency topology differs")
    applicable_latency: list[bool] = []
    for device, maximum in (
        ("cpu", ENGINEERING_GATES["cpu_warm_p99_ms_max"]),
        ("cuda", ENGINEERING_GATES["cuda_warm_p99_ms_max"]),
    ):
        row = latency[device]
        if not isinstance(row, Mapping) or row.get("applicable") not in {True, False}:
            raise ReleaseReadinessError(f"{label} {device} latency applicability differs")
        if float(row.get("maximum_ms", math.nan)) != maximum or float(
            row.get("stride_fraction_maximum", math.nan)
        ) != ENGINEERING_GATES["p99_stride_budget_fraction_max"]:
            raise ReleaseReadinessError(f"{label} {device} latency threshold differs")
        if row["applicable"]:
            p99 = row.get("value_ms")
            expected = _gate_pass(p99, "<=", maximum)
            stride = float(p99) / ENGINEERING_GATES["stride_budget_ms"]
            if (row.get("pass") is not expected
                    or not math.isclose(float(row.get("stride_fraction", math.nan)), stride,
                                        rel_tol=1e-12, abs_tol=1e-12)
                    or row.get("stride_pass") is not (
                        stride <= ENGINEERING_GATES["p99_stride_budget_fraction_max"]
                    )):
                raise ReleaseReadinessError(f"{label} {device} latency decision differs")
            applicable_latency.append(bool(row["pass"] and row["stride_pass"]))
        elif (row.get("value_ms") is not None or row.get("stride_fraction") is not None
              or row.get("pass") is not True or row.get("stride_pass") is not True):
            raise ReleaseReadinessError(f"{label} non-applicable {device} latency is not neutral")

    checks = [
        _bool_gate(engineering.get("checkpoint_bytes"), value_key="value", limit_key="maximum",
                   limit=ENGINEERING_GATES["checkpoint_bytes_max"], operator="<=",
                   label=f"{label} checkpoint bytes"),
        _bool_gate(engineering.get("parameter_count"), value_key="value", limit_key="maximum",
                   limit=ENGINEERING_GATES["parameter_count_max"], operator="<=",
                   label=f"{label} parameter count"),
    ]
    for name, maximum in (
        ("cpu_process_peak_rss_bytes", ENGINEERING_GATES["cpu_process_peak_rss_bytes_max"]),
        ("cuda_peak_reserved_bytes", ENGINEERING_GATES["cuda_peak_reserved_bytes_max"]),
    ):
        row = engineering.get(name)
        if not isinstance(row, Mapping) or float(row.get("maximum", math.nan)) != maximum:
            raise ReleaseReadinessError(f"{label} {name} threshold differs")
        if row.get("applicable"):
            expected = _gate_pass(row.get("value"), "<=", maximum)
            if row.get("pass") is not expected:
                raise ReleaseReadinessError(f"{label} {name} decision differs")
            checks.append(expected)
        elif row.get("value") is not None or row.get("pass") is not True:
            raise ReleaseReadinessError(f"{label} non-applicable {name} is not neutral")
        else:
            checks.append(True)
    spike = engineering.get("spike_rate_diagnostic")
    if (not isinstance(spike, Mapping)
            or float(spike.get("minimum", math.nan)) != ENGINEERING_GATES["spike_rate_diagnostic_min"]
            or float(spike.get("maximum", math.nan)) != ENGINEERING_GATES["spike_rate_diagnostic_max"]
            or spike.get("unavailable_policy") != ENGINEERING_GATES["spike_rate_unavailable_policy"]):
        raise ReleaseReadinessError(f"{label} spike diagnostic contract differs")
    if spike.get("applicable"):
        telemetry_available = True
        expected_spike = (
            math.isfinite(float(spike.get("value", math.nan)))
            and
            ENGINEERING_GATES["spike_rate_diagnostic_min"]
            <= float(spike.get("value", math.nan))
            <= ENGINEERING_GATES["spike_rate_diagnostic_max"]
        )
    else:
        telemetry_available = False
        expected_spike = spike.get("value") is None
    if spike.get("pass") is not expected_spike:
        raise ReleaseReadinessError(f"{label} spike diagnostic decision differs")
    checks.append(expected_spike)
    frozen_result = all(applicable_latency + checks)
    if gates.get("all_applicable_engineering_pass") is not frozen_result:
        raise ReleaseReadinessError(f"{label} streaming engineering aggregate differs")
    # The underlying immutable campaign deliberately treats unavailable spike
    # telemetry as diagnostic-only.  Release readiness is stricter: a V4
    # internal candidate requires a real finite operating-band value per unit.
    return frozen_result and telemetry_available and expected_spike


def _validate_streaming(seal: Mapping[str, Any]) -> tuple[bool, list[dict[str, Any]]]:
    _assert_class(seal, "locked_hcs_streaming_deployment_all_18_complete_seal", "streaming seal")
    if (seal.get("commercial_performance_claim_authorized") is not False
            or seal.get("prospective_cohort_required_for_commercial_claim") is not True):
        raise ReleaseReadinessError("streaming seal weakens the prospective blocker")
    plan = _bound_plan(seal, "streaming seal")
    freeze, _ = _load_bound_json(plan.get("freeze_spec"), "streaming freeze spec")
    expected_streaming_gates = {
        "cpu_stateful_one_window_warm_p99_ms_max": ENGINEERING_GATES["cpu_warm_p99_ms_max"],
        "cuda_stateful_one_window_warm_p99_ms_max": ENGINEERING_GATES["cuda_warm_p99_ms_max"],
        "stride_budget_ms": ENGINEERING_GATES["stride_budget_ms"],
        "p99_stride_budget_fraction_max": ENGINEERING_GATES["p99_stride_budget_fraction_max"],
        "checkpoint_bytes_max": ENGINEERING_GATES["checkpoint_bytes_max"],
        "parameter_count_max": ENGINEERING_GATES["parameter_count_max"],
        "cpu_process_peak_rss_bytes_max": ENGINEERING_GATES["cpu_process_peak_rss_bytes_max"],
        "cuda_peak_reserved_bytes_max": ENGINEERING_GATES["cuda_peak_reserved_bytes_max"],
        "spike_rate_diagnostic_min": ENGINEERING_GATES["spike_rate_diagnostic_min"],
        "spike_rate_diagnostic_max": ENGINEERING_GATES["spike_rate_diagnostic_max"],
        "spike_rate_unavailable_policy": ENGINEERING_GATES["spike_rate_unavailable_policy"],
    }
    if (freeze.get("engineering_gates") != expected_streaming_gates
            or freeze.get("mandatory_gates") != list(MANDATORY_STREAMING_GATES)):
        raise ReleaseReadinessError("streaming frozen gate contract differs")
    units = seal.get("units")
    if (seal.get("unit_count") != 18 or seal.get("all_18_receipts_validated") is not True
            or seal.get("unit_selection_or_ranking_performed") is not False
            or not isinstance(units, list) or len(units) != 18):
        raise ReleaseReadinessError("streaming complete-seal topology differs")
    keys = {(int(unit.get("outer_fold", -1)), int(unit.get("seed", -1))) for unit in units}
    if keys != {(fold, seed) for fold in FOLDS for seed in SEEDS}:
        raise ReleaseReadinessError("streaming fixed fold/seed cover differs")
    statuses = []
    mandatory_statuses = []
    engineering_statuses = []
    rows = []
    for unit in units:
        receipt, _ = _load_bound_json(unit.get("receipt"), f"streaming receipt {unit.get('unit_id')}")
        gates = receipt.get("gates", {})
        mandatory = gates.get("mandatory", {})
        if not isinstance(mandatory, Mapping) or set(mandatory) != set(MANDATORY_STREAMING_GATES):
            raise ReleaseReadinessError(
                f"streaming mandatory-gate topology differs: {unit.get('unit_id')}"
            )
        mandatory_pass = all(value is True for value in mandatory.values())
        if gates.get("all_mandatory_pass") is not mandatory_pass:
            raise ReleaseReadinessError(
                f"streaming mandatory aggregate differs: {unit.get('unit_id')}"
            )
        readiness_engineering_pass = _validate_streaming_engineering(
            gates, f"streaming unit {unit.get('unit_id')}"
        )
        frozen_engineering_pass = bool(gates.get("all_applicable_engineering_pass"))
        mandatory_statuses.append(mandatory_pass)
        engineering_statuses.append(frozen_engineering_pass)
        frozen_integrated = mandatory_pass and frozen_engineering_pass
        if gates.get("unit_integrated_pass") is not frozen_integrated:
            raise ReleaseReadinessError(
                f"streaming integrated receipt decision differs: {unit.get('unit_id')}"
            )
        if unit.get("unit_integrated_pass") is not frozen_integrated:
            raise ReleaseReadinessError(f"streaming unit seal/receipt gate differs: {unit.get('unit_id')}")
        status = mandatory_pass and readiness_engineering_pass
        statuses.append(status)
        rows.append({"category": "streaming", "condition": str(unit.get("unit_id")), "passed": status})
    result = all(statuses)
    if (seal.get("all_mandatory_gates_pass") is not all(mandatory_statuses)
            or seal.get("all_applicable_engineering_gates_pass") is not all(engineering_statuses)
            or seal.get("integrated_pass") is not all(
                mandatory and engineering
                for mandatory, engineering in zip(
                    mandatory_statuses, engineering_statuses, strict=True
                )
            )):
        raise ReleaseReadinessError("streaming complete-seal aggregate differs")
    return result, rows


def _validate_deployment_gate_details(gates: Any, *, device: str, label: str) -> bool:
    if not isinstance(gates, Mapping) or not isinstance(gates.get("results"), Mapping):
        raise ReleaseReadinessError(f"{label} deployment gates are absent")
    results = gates["results"]
    expected_names = {
        "warm_p99_device_limit", "warm_p99_stride_fraction", "checkpoint_bytes",
        "parameter_count", "cpu_process_peak_rss", "spike_rate_diagnostic",
    } | ({"cuda_peak_reserved"} if device == "cuda" else set())
    if set(results) != expected_names:
        raise ReleaseReadinessError(f"{label} deployment gate topology differs")
    p99_pass = _bool_gate(
            results["warm_p99_device_limit"], value_key="value_ms", limit_key="maximum_ms",
            limit=(ENGINEERING_GATES["cpu_warm_p99_ms_max"] if device == "cpu"
                   else ENGINEERING_GATES["cuda_warm_p99_ms_max"]), operator="<=",
            label=f"{label} warm p99",
        )
    stride_pass = _bool_gate(
            results["warm_p99_stride_fraction"], value_key="value", limit_key="maximum",
            limit=ENGINEERING_GATES["p99_stride_budget_fraction_max"], operator="<=",
            label=f"{label} stride fraction",
        )
    expected_fraction = float(results["warm_p99_device_limit"]["value_ms"]) / ENGINEERING_GATES[
        "stride_budget_ms"
    ]
    if not math.isclose(
        float(results["warm_p99_stride_fraction"]["value"]), expected_fraction,
        rel_tol=1e-12, abs_tol=1e-12,
    ):
        raise ReleaseReadinessError(f"{label} p99/stride fraction differs")
    checks = [
        p99_pass,
        stride_pass,
        _bool_gate(
            results["checkpoint_bytes"], value_key="value", limit_key="maximum",
            limit=ENGINEERING_GATES["checkpoint_bytes_max"], operator="<=",
            label=f"{label} checkpoint bytes",
        ),
        _bool_gate(
            results["parameter_count"], value_key="value", limit_key="maximum",
            limit=ENGINEERING_GATES["parameter_count_max"], operator="<=",
            label=f"{label} parameter count",
        ),
        _bool_gate(
            results["cpu_process_peak_rss"], value_key="value_bytes", limit_key="maximum_bytes",
            limit=ENGINEERING_GATES["cpu_process_peak_rss_bytes_max"], operator="<=",
            label=f"{label} CPU peak RSS",
        ),
    ]
    if device == "cuda":
        checks.append(_bool_gate(
            results["cuda_peak_reserved"], value_key="value_bytes", limit_key="maximum_bytes",
            limit=ENGINEERING_GATES["cuda_peak_reserved_bytes_max"], operator="<=",
            label=f"{label} CUDA peak reserved",
        ))
    spike = results["spike_rate_diagnostic"]
    if (not isinstance(spike, Mapping)
            or float(spike.get("minimum", math.nan)) != ENGINEERING_GATES["spike_rate_diagnostic_min"]
            or float(spike.get("maximum", math.nan)) != ENGINEERING_GATES["spike_rate_diagnostic_max"]
            or spike.get("unavailable_policy") != ENGINEERING_GATES["spike_rate_unavailable_policy"]):
        raise ReleaseReadinessError(f"{label} spike diagnostic contract differs")
    if spike.get("available"):
        telemetry_available = True
        spike_pass = (
            math.isfinite(float(spike.get("value", math.nan)))
            and
            ENGINEERING_GATES["spike_rate_diagnostic_min"]
            <= float(spike.get("value", math.nan))
            <= ENGINEERING_GATES["spike_rate_diagnostic_max"]
        )
    else:
        telemetry_available = False
        spike_pass = spike.get("value") is None
    if spike.get("pass") is not spike_pass:
        raise ReleaseReadinessError(f"{label} spike diagnostic decision differs")
    checks.append(spike_pass)
    frozen_result = all(checks)
    if gates.get("all_applicable_pass") is not frozen_result:
        raise ReleaseReadinessError(f"{label} deployment gate aggregate differs")
    return frozen_result and telemetry_available and spike_pass


def _validate_deployment(seal: Mapping[str, Any], *, expected_device: str) -> tuple[bool, list[dict[str, Any]]]:
    _assert_class(seal, "locked_proposer_deployment_all_18_complete_seal", f"{expected_device} deployment seal")
    if seal.get("commercial_performance_claim_authorized") is not False:
        raise ReleaseReadinessError(f"{expected_device} deployment seal authorizes a commercial claim")
    if _deployment_device(seal, f"{expected_device} deployment seal") != expected_device:
        raise ReleaseReadinessError(f"{expected_device} deployment role/device differs")
    plan = _bound_plan(seal, f"{expected_device} deployment seal")
    freeze, _ = _load_bound_json(plan.get("freeze_spec"), f"{expected_device} deployment freeze spec")
    expected_frozen = {
        "cpu_raw_resident_warm_p99_ms_max": ENGINEERING_GATES["cpu_warm_p99_ms_max"],
        "cuda_raw_resident_warm_p99_ms_max": ENGINEERING_GATES["cuda_warm_p99_ms_max"],
        "p99_stride_budget_fraction_max": ENGINEERING_GATES["p99_stride_budget_fraction_max"],
        "checkpoint_bytes_max": ENGINEERING_GATES["checkpoint_bytes_max"],
        "parameter_count_max": ENGINEERING_GATES["parameter_count_max"],
        "cpu_process_peak_rss_bytes_max": ENGINEERING_GATES["cpu_process_peak_rss_bytes_max"],
        "cuda_peak_reserved_bytes_max": ENGINEERING_GATES["cuda_peak_reserved_bytes_max"],
        "spike_rate_diagnostic_min": ENGINEERING_GATES["spike_rate_diagnostic_min"],
        "spike_rate_diagnostic_max": ENGINEERING_GATES["spike_rate_diagnostic_max"],
        "spike_rate_unavailable_policy": ENGINEERING_GATES["spike_rate_unavailable_policy"],
    }
    if freeze.get("engineering_gates") != expected_frozen:
        raise ReleaseReadinessError(f"{expected_device} deployment frozen gates differ")
    if float(freeze.get("measurement", {}).get("stride_budget_ms", math.nan)) != ENGINEERING_GATES[
        "stride_budget_ms"
    ]:
        raise ReleaseReadinessError(f"{expected_device} deployment stride budget differs")
    units = seal.get("units")
    if (seal.get("unit_count") != 18 or seal.get("all_18_reported") is not True
            or seal.get("best_unit_selection_performed") is not False
            or seal.get("unit_ranking_performed") is not False
            or not isinstance(units, list) or len(units) != 18):
        raise ReleaseReadinessError(f"{expected_device} deployment topology differs")
    keys = {(int(unit.get("outer_fold", -1)), int(unit.get("seed", -1))) for unit in units}
    if keys != {(fold, seed) for fold in FOLDS for seed in SEEDS}:
        raise ReleaseReadinessError(f"{expected_device} deployment fixed cover differs")
    statuses = []
    frozen_statuses = []
    rows = []
    for unit in units:
        receipt, _ = _load_bound_json(unit.get("receipt"), f"{expected_device} deployment receipt")
        if receipt.get("runtime", {}).get("device_type") != expected_device:
            raise ReleaseReadinessError(f"{expected_device} unit runtime differs")
        status = _validate_deployment_gate_details(
            receipt.get("engineering_gates"), device=expected_device,
            label=f"{expected_device} unit {unit.get('unit_id')}",
        )
        frozen_status = bool(receipt.get("engineering_gates", {}).get("all_applicable_pass"))
        frozen_statuses.append(frozen_status)
        if unit.get("engineering_gates_pass") is not frozen_status:
            raise ReleaseReadinessError(f"{expected_device} unit seal/receipt gate differs")
        statuses.append(status)
        rows.append({"category": f"proposer_{expected_device}", "condition": str(unit.get("unit_id")), "passed": status})
    result = all(statuses)
    frozen_result = all(frozen_statuses)
    if seal.get("all_applicable_engineering_gates_pass") is not frozen_result:
        raise ReleaseReadinessError(f"{expected_device} deployment aggregate differs")
    return result, rows


def _receipt_binds_report(receipt: Mapping[str, Any], report_path: Path, label: str) -> None:
    raw = receipt.get("outputs", {}).get("report")
    observed = bind_file(report_path)
    if not _same_binding(raw, observed):
        raise ReleaseReadinessError(f"{label} receipt does not bind its report")


def _load_release_runner_module() -> Any:
    name = "run_release_locked_hcs_evaluation"
    existing = sys.modules.get(name)
    if existing is not None:
        return existing
    path = SOURCE_PATHS["release_evaluation_orchestrator"]
    specification = importlib.util.spec_from_file_location(name, path)
    if specification is None or specification.loader is None:
        raise ReleaseReadinessError(f"cannot import release evaluation orchestrator: {path}")
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    try:
        specification.loader.exec_module(module)
    except Exception:
        sys.modules.pop(name, None)
        raise
    return module


def _receipt_output_path(
    receipt: Mapping[str, Any], *, name: str, label: str
) -> Path:
    outputs = receipt.get("outputs")
    if not isinstance(outputs, Mapping):
        raise ReleaseReadinessError(f"{label} receipt outputs are absent")
    binding = _verify_binding(outputs.get(name), label=f"{label} receipt output {name}")
    return Path(binding["path"])


def _output_group(
    *, report: Path, csv_path: Path, receipt: Path, label: str
) -> Path:
    parents = {path.expanduser().resolve().parent for path in (report, csv_path, receipt)}
    if len(parents) != 1:
        raise ReleaseReadinessError(f"{label} outputs do not share the fixed output directory")
    return next(iter(parents))


def _validate_execution_attestation(
    attestation: Mapping[str, Any], *, spec_binding: Mapping[str, Any],
    roles: Mapping[str, Path], documents: Mapping[str, Mapping[str, Any]],
) -> None:
    _assert_class(
        attestation, "locked_hcs_release_evaluation_execution_attestation",
        "release evaluation execution attestation",
    )
    if (attestation.get("commercial_claim_authorized") is not False
            or attestation.get("prospective_confirmation_required") is not True
            or attestation.get("target_re_evaluation_performed") is not False
            or attestation.get("all_steps_executed_once") is not True
            or attestation.get("all_inputs_and_outputs_live_rehashed") is not True):
        raise ReleaseReadinessError("release evaluation execution invariants differ")
    bindings = {
        "release_readiness_spec": spec_binding,
        "pretarget_release_lock": bind_file(roles["pretarget_release_lock"]),
        "target_release_receipt": bind_file(roles["target_release_receipt"]),
        "canonical_target": bind_file(roles["canonical_target"]),
        "canonical_target_receipt": bind_file(roles["canonical_target_receipt"]),
        "predictions_seal": bind_file(roles["predictions_seal"]),
        "evaluation_lock": bind_file(roles["primary_evaluation_lock"]),
        "primary_evaluation_spec": bind_file(roles["primary_evaluation_spec"]),
        "uncertainty_evaluation_spec": bind_file(roles["uncertainty_spec"]),
        "radar_mask_complete_seal": bind_file(roles["radar_mask_complete_seal"]),
        "uncertainty_inputs_seal": bind_file(roles["uncertainty_inputs_seal"]),
    }
    for name, expected in bindings.items():
        if not _same_binding(attestation.get(name, {}), expected):
            raise ReleaseReadinessError(f"execution attestation binding differs: {name}")
    expected_source = bind_file(SOURCE_PATHS["release_evaluation_orchestrator"])
    if not _same_binding(attestation.get("execution_source", {}), expected_source):
        raise ReleaseReadinessError("execution attestation runner source differs")
    evaluations = attestation.get("evaluations")
    if not isinstance(evaluations, Mapping) or set(evaluations) != {
        "primary", "radar_masks", "uncertainty"
    }:
        raise ReleaseReadinessError("execution attestation evaluation topology differs")
    expected_evaluations = {
        "primary": ("primary_report", "primary_receipt"),
        "radar_masks": ("radar_report", "radar_receipt"),
        "uncertainty": ("uncertainty_report", "uncertainty_receipt"),
    }
    for name, (report_role, receipt_role) in expected_evaluations.items():
        item = evaluations[name]
        if (not isinstance(item, Mapping) or set(item) != {"report", "receipt"}
                or not _same_binding(item.get("report", {}), bind_file(roles[report_role]))
                or not _same_binding(item.get("receipt", {}), bind_file(roles[receipt_role]))):
            raise ReleaseReadinessError(f"execution attestation evaluation binding differs: {name}")

    primary_receipt = documents["primary_receipt"]
    radar_receipt = documents["radar_receipt"]
    uncertainty_receipt = documents["uncertainty_receipt"]
    uncertainty_spec = documents["uncertainty_spec"]
    bound_inputs = uncertainty_spec.get("bound_inputs")
    if not isinstance(bound_inputs, Mapping):
        raise ReleaseReadinessError("uncertainty specification bound inputs are absent")
    calibration = _verify_binding(
        bound_inputs.get("completed_pretest_uncertainty_calibration"),
        label="uncertainty specification calibration",
    )
    primary_csv = _receipt_output_path(
        primary_receipt, name="metrics_csv", label="primary"
    )
    radar_csv = _receipt_output_path(radar_receipt, name="metrics_csv", label="radar")
    uncertainty_csv = _receipt_output_path(
        uncertainty_receipt, name="metrics_csv", label="uncertainty"
    )
    runner = _load_release_runner_module()
    pretarget = _json(roles["pretarget_release_lock"], "pretarget release lock")
    _verify_closure(pretarget, label="pretarget release lock")
    boundaries = pretarget.get("boundaries")
    if not isinstance(boundaries, Mapping):
        raise ReleaseReadinessError("pretarget release boundaries are absent")
    radar_boundary = boundaries.get("radar_masks")
    uncertainty_boundary = boundaries.get("uncertainty")
    if not isinstance(radar_boundary, Mapping) or not isinstance(
        uncertainty_boundary, Mapping
    ):
        raise ReleaseReadinessError("pretarget radar/uncertainty boundary is absent")
    uncertainty_audit = documents["uncertainty_report"].get(
        "pretarget_provenance_audit"
    )
    if not isinstance(uncertainty_audit, Mapping):
        raise ReleaseReadinessError(
            "uncertainty report target-free provenance audit is absent"
        )
    authorization = {
        "canonical_target_receipt": bind_file(roles["canonical_target_receipt"]),
        "canonical_target": bind_file(roles["canonical_target"]),
        "predictions_seal": bind_file(roles["predictions_seal"]),
        "primary_evaluation_spec": bind_file(roles["primary_evaluation_spec"]),
        "uncertainty_evaluation_spec": bind_file(roles["uncertainty_spec"]),
        "uncertainty_calibration": calibration,
        "radar_mask_complete_seal": bind_file(roles["radar_mask_complete_seal"]),
        "radar_mask_plan": _verify_binding(
            radar_boundary.get("plan"), label="pretarget radar-mask plan"
        ),
        "radar_mask_preexecution_lock": _verify_binding(
            radar_boundary.get("preexecution_lock"),
            label="pretarget radar-mask preexecution lock",
        ),
        "uncertainty_inputs_seal": bind_file(roles["uncertainty_inputs_seal"]),
        "uncertainty_archive": _verify_binding(
            uncertainty_boundary.get("uncertainty_archive"),
            label="pretarget uncertainty archive",
        ),
        "uncertainty_pretarget_audit": dict(uncertainty_audit),
        "joined_oof": bind_file(roles["joined_output"]),
        "locked_metrics": _verify_binding(
            documents["primary_evaluation_lock"].get("outputs", {}).get("metrics"),
            label="primary evaluation lock metrics",
        ),
    }
    evaluation_lock = bind_file(roles["primary_evaluation_lock"])
    try:
        for name, receipt in (
            ("primary", primary_receipt),
            ("radar_masks", radar_receipt),
            ("uncertainty", uncertainty_receipt),
        ):
            inputs = receipt.get("inputs")
            if not isinstance(inputs, Mapping):
                raise ReleaseReadinessError(f"{name} receipt inputs are absent")
            runner.validate_evaluator_receipt_edges(
                name=name,
                inputs=inputs,
                authorization=authorization,
                evaluation_lock_binding=evaluation_lock,
            )
    except Exception as exc:
        if isinstance(exc, ReleaseReadinessError):
            raise
        raise ReleaseReadinessError(
            f"execution attestation evaluator input edge differs: {exc}"
        ) from exc

    frozen = attestation.get("frozen_argv")
    executed = attestation.get("executed_commands")
    if frozen != executed:
        raise ReleaseReadinessError("execution attestation frozen/executed argv differ")
    primary_output_dir = _output_group(
        report=roles["primary_report"], csv_path=primary_csv,
        receipt=roles["primary_receipt"], label="primary",
    )
    radar_output_dir = _output_group(
        report=roles["radar_report"], csv_path=radar_csv,
        receipt=roles["radar_receipt"], label="radar",
    )
    uncertainty_output_dir = _output_group(
        report=roles["uncertainty_report"], csv_path=uncertainty_csv,
        receipt=roles["uncertainty_receipt"], label="uncertainty",
    )
    evidence = {
        "raw_join_source": PROJECT_ROOT / "scripts/run_locked_hcs_oof.py",
        "primary_source": SOURCE_PATHS["primary_evaluator"],
        "radar_source": SOURCE_PATHS["radar_mask_evaluator"],
        "uncertainty_source": SOURCE_PATHS["uncertainty_evaluator"],
        "primary_root": roles["joined_output"].parent,
        "mask_root": roles["radar_mask_complete_seal"].parent,
        "canonical_target": roles["canonical_target"],
        "evaluation_lock": roles["primary_evaluation_lock"],
        "canonical_target_receipt": roles["canonical_target_receipt"],
        "primary_evaluation_spec": roles["primary_evaluation_spec"],
        "uncertainty_evaluation_spec": roles["uncertainty_spec"],
        "uncertainty_calibration": Path(calibration["path"]),
        "predictions_seal": roles["predictions_seal"],
        "uncertainty_inputs_seal": roles["uncertainty_inputs_seal"],
        "primary_output_dir": primary_output_dir,
        "primary_report": roles["primary_report"],
        "primary_csv": primary_csv,
        "primary_receipt": roles["primary_receipt"],
        "radar_output_dir": radar_output_dir,
        "radar_report": roles["radar_report"],
        "radar_csv": radar_csv,
        "radar_receipt": roles["radar_receipt"],
        "uncertainty_output_dir": uncertainty_output_dir,
        "uncertainty_report": roles["uncertainty_report"],
        "uncertainty_csv": uncertainty_csv,
        "uncertainty_receipt": roles["uncertainty_receipt"],
    }
    try:
        runner.validate_frozen_argv_from_evidence(frozen, evidence=evidence)
    except Exception as exc:
        raise ReleaseReadinessError(
            f"execution attestation exact fixed argv differs: {exc}"
        ) from exc


def _verify_release_and_guards(spec: Mapping[str, Any], spec_binding: Mapping[str, Any],
                               roles: Mapping[str, Path]) -> dict[str, Any]:
    for name in (
        "target_release_receipt", "pretarget_release_lock", "primary_evaluation_lock",
        "fixed_i3_runtime_seal", "fixed_runtime_completion", "postlock_runtime_guard",
        "radar_mask_runtime_guard",
    ):
        _require_mode_0444(roles[name], name)
    release_receipt = _json(roles["target_release_receipt"], "target release receipt")
    _verify_closure(release_receipt, label="target release receipt")
    _assert_class(release_receipt, "locked_hcs_canonical_targets_built_after_pretarget_release", "target release receipt")
    if (release_receipt.get("commercial_claim_authorized") is not False
            or release_receipt.get("prospective_confirmation_required") is not True
            or release_receipt.get("release_lock_revalidated_before_target_builder_call") is not True):
        raise ReleaseReadinessError("target release receipt invariants differ")
    pretarget_binding = bind_file(roles["pretarget_release_lock"])
    if not _same_binding(release_receipt.get("pretarget_release_lock", {}), pretarget_binding):
        raise ReleaseReadinessError("target release receipt/pretarget lock binding differs")
    pretarget = _json(roles["pretarget_release_lock"], "pretarget release lock")
    _verify_closure(pretarget, label="pretarget release lock")
    _assert_class(pretarget, "locked_hcs_pretarget_release_lock", "pretarget release lock")
    readiness = pretarget.get("frozen_specs", {}).get("release_readiness")
    if (not isinstance(readiness, Mapping)
            or not _same_binding(readiness.get("binding", {}), spec_binding)
            or readiness.get("content_sha256") != spec.get("content_sha256")
            or readiness.get("target_or_target_bearing_artifact_opened") is not False
            or readiness.get("commercial_release_ready_must_equal") is not False
            or readiness.get("prospective_confirmation_required") is not True):
        raise ReleaseReadinessError(
            "pretarget release lock does not exactly bind the release-readiness specification"
        )
    locations = pretarget.get("locations", {})
    if Path(str(locations.get("release_readiness_spec", ""))).resolve() != Path(
        str(spec_binding["path"])
    ).resolve():
        raise ReleaseReadinessError("pretarget release lock readiness-spec location differs")
    closure = pretarget.get("boundaries", {}).get("runtime_payload_closure", {})
    guard_keys = {
        "fixed_i3_runtime_seal": "runtime_input_seal",
        "fixed_runtime_completion": "fixed_pretest_completion_attestation",
        "postlock_runtime_guard": "postlock_runtime_guard_attestation",
        "radar_mask_runtime_guard": "radar_mask_runtime_guard_attestation",
    }
    for role, key in guard_keys.items():
        if not _same_binding(closure.get(key, {}), bind_file(roles[role])):
            raise ReleaseReadinessError(f"pretarget runtime closure differs: {role}")
    lock = _json(roles["primary_evaluation_lock"], "primary evaluation lock")
    _verify_closure(lock, label="primary evaluation lock")
    _assert_class(lock, "locked_hcs_oof_single_target_join_seal", "primary evaluation lock")
    if (lock.get("target_join_count") != 1 or lock.get("commercial_claim_authorized") is not False
            or not _same_binding(lock.get("target_artifact", {}), release_receipt.get("canonical_target", {}))):
        raise ReleaseReadinessError("primary evaluation lock target binding differs")
    lock_outputs = lock.get("outputs")
    if not isinstance(lock_outputs, Mapping) or set(lock_outputs) != {
        "joined_oof", "metrics"
    }:
        raise ReleaseReadinessError("primary evaluation lock output topology differs")
    _verify_binding(lock_outputs["metrics"], label="primary evaluation lock metrics")
    if (not _same_binding(release_receipt.get("canonical_target", {}), bind_file(roles["canonical_target"]))
            or not _same_binding(release_receipt.get("canonical_target_receipt", {}),
                                 bind_file(roles["canonical_target_receipt"]))
            or not _same_binding(lock_outputs["joined_oof"],
                                 bind_file(roles["joined_output"]))):
        raise ReleaseReadinessError("canonical target/joined boundary roles differ")
    return {"target_release_receipt": bind_file(roles["target_release_receipt"]),
            "pretarget_release_lock": pretarget_binding,
            "primary_evaluation_lock": bind_file(roles["primary_evaluation_lock"])}


def evaluate(*, spec_path: Path, roles: Mapping[str, Path | None], output_dir: Path,
             report_output: Path, csv_output: Path, receipt_output: Path,
             orchestrator_command: Sequence[str] = ()) -> dict[str, Any]:
    # Absolute ordering boundary: the immutable aggregate spec, its sources,
    # and its target-free input hashes are validated before any target-derived
    # role is resolved, stat'ed, or opened below.
    spec, spec_binding = _load_spec(spec_path)
    frozen_roles = spec.get("input_roles")
    observed_roles = _resolve_roles(roles)
    if observed_roles != frozen_roles:
        raise ReleaseReadinessError("evaluate input roles differ from the frozen specification")
    resolved = {name: Path(value) for name, value in observed_roles.items() if value is not None}
    cuda_required = bool(spec["cuda_applicability"]["required"])
    # Recheck the immutable publication contract for the entire frozen role
    # topology at consumption time.  Content hashes alone do not detect a
    # writable-mode drift, and limiting this to the JSON reports leaves the
    # target, joined output, pretarget inputs, seals, and CSV-independent
    # receipts outside the aggregate boundary.
    for name, path in resolved.items():
        if name == "proposer_cuda_complete_seal" and not cuda_required:
            continue
        _require_mode_0444(path, name)
    destinations = tuple(path.expanduser().resolve() for path in (report_output, csv_output, receipt_output))
    root = output_dir.expanduser().resolve()
    if len(set(destinations)) != 3 or any(path.parent != root for path in destinations):
        raise ReleaseReadinessError("aggregate outputs must be distinct direct output-dir children")
    if any(path.exists() for path in destinations):
        raise ReleaseReadinessError("immutable aggregate output already exists")

    release_bindings = _verify_release_and_guards(spec, spec_binding, resolved)
    top_documents: dict[str, dict[str, Any]] = {}
    for name in (
        "primary_evaluation_lock", "primary_report", "primary_receipt",
        "radar_report", "radar_receipt",
        "uncertainty_spec", "uncertainty_report", "uncertainty_receipt",
        "streaming_complete_seal", "proposer_cpu_complete_seal",
        "release_evaluation_execution_attestation",
    ):
        _require_mode_0444(resolved[name], name)
        document = _json(resolved[name], name)
        _verify_closure(document, label=name)
        top_documents[name] = document
    if cuda_required:
        path = resolved.get("proposer_cuda_complete_seal")
        if path is None:
            raise ReleaseReadinessError("frozen-required CUDA deployment seal is missing")
        _require_mode_0444(path, "CUDA deployment seal")
        cuda = _json(path, "CUDA deployment seal")
        _verify_closure(cuda, label="CUDA deployment seal")
        top_documents["proposer_cuda_complete_seal"] = cuda

    _assert_class(top_documents["primary_receipt"], "retrospective_locked_hcs_oof_primary_evaluation_receipt", "primary receipt")
    _assert_class(top_documents["radar_receipt"], "retrospective_locked_hcs_all_radar_masks_evaluation_receipt", "radar receipt")
    _assert_class(top_documents["uncertainty_receipt"], "retrospective_locked_hcs_uncertainty_evaluation_receipt", "uncertainty receipt")
    for name in ("primary_receipt", "radar_receipt", "uncertainty_receipt"):
        receipt_document = top_documents[name]
        if (receipt_document.get("commercial_claim_authorized") is not False
                or receipt_document.get("commercial_performance_proven") is not False
                or receipt_document.get("prospective_confirmation_required") is not True
                or receipt_document.get("independent_prospective_cohort_evaluated") is not False
                or receipt_document.get("outputs_create_once") is not True
                or receipt_document.get("output_overwrite_allowed") is not False):
            raise ReleaseReadinessError(f"{name} immutability/prospective invariants differ")
    _receipt_binds_report(top_documents["primary_receipt"], resolved["primary_report"], "primary")
    _receipt_binds_report(top_documents["radar_receipt"], resolved["radar_report"], "radar")
    _receipt_binds_report(top_documents["uncertainty_receipt"], resolved["uncertainty_report"], "uncertainty")
    if (top_documents["primary_receipt"].get("seeds") != list(SEEDS)
            or top_documents["uncertainty_receipt"].get("seeds") != list(SEEDS)):
        raise ReleaseReadinessError("evaluation receipt fixed seeds differ")
    _validate_execution_attestation(
        top_documents["release_evaluation_execution_attestation"],
        spec_binding=spec_binding, roles=resolved, documents=top_documents,
    )
    if (not _same_binding(top_documents["primary_report"].get(
            "evaluation_specification", {}), bind_file(resolved["primary_evaluation_spec"]))
            or not _same_binding(top_documents["radar_report"].get(
                "evaluation_specification", {}), bind_file(resolved["primary_evaluation_spec"]))):
        raise ReleaseReadinessError("primary/radar report evaluation-spec binding differs")
    if not _same_binding(top_documents["uncertainty_report"].get(
            "uncertainty_evaluation_specification", {}), bind_file(resolved["uncertainty_spec"])):
        raise ReleaseReadinessError("uncertainty report/dedicated spec binding differs")

    category_status: dict[str, bool] = {}
    csv_rows: list[dict[str, Any]] = []
    category_status["primary_accuracy"], rows = _validate_primary(top_documents["primary_report"]); csv_rows += rows
    category_status["all_radar_masks"], rows = _validate_radar(top_documents["radar_report"]); csv_rows += rows
    category_status["uncertainty_calibration"], rows = _validate_uncertainty(
        top_documents["uncertainty_report"], top_documents["uncertainty_spec"]); csv_rows += rows
    category_status["streaming"], rows = _validate_streaming(top_documents["streaming_complete_seal"]); csv_rows += rows
    category_status["proposer_cpu"], rows = _validate_deployment(
        top_documents["proposer_cpu_complete_seal"], expected_device="cpu"); csv_rows += rows
    if cuda_required:
        category_status["proposer_cuda"], rows = _validate_deployment(
            top_documents["proposer_cuda_complete_seal"], expected_device="cuda"); csv_rows += rows
    else:
        category_status["proposer_cuda"] = True
        csv_rows.append({"category": "proposer_cuda", "condition": "not_applicable_frozen", "passed": True})

    candidate_ready = all(category_status.values())
    bound_input_paths = {
        name: path for name, path in resolved.items()
        if name != "proposer_cuda_complete_seal" or cuda_required
    }
    report = _content_document({
        "schema_version": SCHEMA_VERSION,
        "classification": "locked_hcs_release_readiness_aggregate_evaluation",
        "evaluation_specification": spec_binding,
        "all_evidence_roles_rehashed_without_target_array_access": True,
        "target_or_prediction_array_opened_or_re_evaluated": False,
        "model_seed_mask_ranking_selection_pooling_averaging_suppression_performed": False,
        "fixed_seeds": list(SEEDS),
        "fixed_masks": list(MASKS),
        "exact_goal_thresholds": spec["exact_gates"],
        "cuda_applicability": spec["cuda_applicability"],
        "category_gate_status": category_status,
        "all_required_categories_passed": candidate_ready,
        "internal_retrospective_engineering_candidate_ready": candidate_ready,
        "commercial_release_ready": False,
        "commercial_release_blocked_reasons": [
            "no separately locked independent prospective cohort has been evaluated",
            "latency and memory measurements are current-host engineering evidence, not target-device guarantees",
            "all accuracy, robustness, calibration, and deployment evidence is retrospective",
        ],
        "independent_prospective_cohort_evaluated": False,
        "prospective_confirmation_required": True,
        "commercial_claim_authorized": False,
        "provenance_locks": release_bindings,
        "input_bindings": {
            name: bind_file(path) for name, path in sorted(bound_input_paths.items())
        },
        "orchestrator_command": list(orchestrator_command),
    })
    root.mkdir(parents=True, exist_ok=True)
    report_tmp = _temporary(destinations[0]); csv_tmp = _temporary(destinations[1]); receipt_tmp = _temporary(destinations[2])
    try:
        _write_json_exclusive(report_tmp, report)
        _write_csv_exclusive(csv_tmp, csv_rows)
        outputs = {
            "report": {"path": str(destinations[0]), "sha256": sha256_file(report_tmp), "bytes": report_tmp.stat().st_size},
            "gate_csv": {"path": str(destinations[1]), "sha256": sha256_file(csv_tmp), "bytes": csv_tmp.stat().st_size},
        }
        receipt = _content_document({
            "schema_version": SCHEMA_VERSION,
            "classification": "locked_hcs_release_readiness_aggregate_receipt",
            "outputs_create_once": True,
            "output_overwrite_allowed": False,
            "spec_verified_before_any_target_derived_input_access": True,
            "all_bindings_content_hashes_and_source_closure_reverified": True,
            "target_or_prediction_arrays_opened": False,
            "no_ranking_selection_pooling_averaging_or_suppression": True,
            "internal_retrospective_engineering_candidate_ready": candidate_ready,
            "commercial_release_ready": False,
            "prospective_confirmation_required": True,
            "inputs": {"release_readiness_spec": spec_binding,
                       **{
                           name: bind_file(path)
                           for name, path in sorted(bound_input_paths.items())
                       }},
            "outputs": outputs,
            "gate_csv_rows": len(csv_rows),
            "orchestrator_command": list(orchestrator_command),
        })
        _write_json_exclusive(receipt_tmp, receipt)
        for source, destination in zip((report_tmp, csv_tmp, receipt_tmp), destinations, strict=True):
            _publish_existing_temp(source, destination)
    finally:
        for path in (report_tmp, csv_tmp, receipt_tmp):
            path.unlink(missing_ok=True)
    published = _json(destinations[2], "published release-readiness receipt")
    _validate_content(published, "published release-readiness receipt")
    if not _same_binding(published["outputs"]["report"], bind_file(destinations[0])):
        raise ReleaseReadinessError("published aggregate report differs from receipt")
    return published


def _temporary(destination: Path) -> Path:
    return destination.with_name(f".{destination.name}.{os.getpid()}.{secrets.token_hex(8)}.tmp")


def _write_json_exclusive(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
        stream.flush(); os.fsync(stream.fileno())


def _write_csv_exclusive(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["category", "condition", "passed"], lineterminator="\n")
        writer.writeheader()
        for row in rows:
            if set(row) != {"category", "condition", "passed"}:
                raise ReleaseReadinessError("aggregate CSV row topology differs")
            writer.writerow({**row, "passed": "true" if row["passed"] else "false"})
        stream.flush(); os.fsync(stream.fileno())


def _publish_existing_temp(source: Path, destination: Path) -> None:
    try:
        os.link(source, destination)
    except FileExistsError as exc:
        raise ReleaseReadinessError(f"immutable aggregate output exists: {destination}") from exc
    destination.chmod(0o444)


def _publish_json_create_once(destination: Path, value: Mapping[str, Any]) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = _temporary(destination)
    try:
        _write_json_exclusive(temporary, value)
        _publish_existing_temp(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _roles_from_args(args: argparse.Namespace) -> dict[str, Path | None]:
    return {name: getattr(args, name) for name in DEFAULT_ROLES}


def _add_role_arguments(parser: argparse.ArgumentParser) -> None:
    for name, default in DEFAULT_ROLES.items():
        option = "--" + name.replace("_", "-")
        parser.add_argument(option, dest=name, type=Path, default=default)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    freeze = sub.add_parser("freeze-spec")
    freeze.add_argument("--output", type=Path, default=DEFAULT_SPEC)
    _add_role_arguments(freeze)
    evaluate_parser = sub.add_parser("evaluate")
    evaluate_parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    evaluate_parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    evaluate_parser.add_argument("--report-output", type=Path)
    evaluate_parser.add_argument("--csv-output", type=Path)
    evaluate_parser.add_argument("--receipt-output", type=Path)
    _add_role_arguments(evaluate_parser)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    command = [str(Path(sys.executable).resolve()), str(Path(__file__).resolve()), *(argv or sys.argv[1:])]
    try:
        if args.mode == "freeze-spec":
            result = freeze_spec(output=args.output, roles=_roles_from_args(args))
        else:
            output = args.output_dir.expanduser().resolve()
            result = evaluate(
                spec_path=args.spec, roles=_roles_from_args(args), output_dir=output,
                report_output=args.report_output or output / "locked_hcs_release_readiness.json",
                csv_output=args.csv_output or output / "locked_hcs_release_readiness_gates.csv",
                receipt_output=args.receipt_output or output / "locked_hcs_release_readiness_receipt.json",
                orchestrator_command=command,
            )
    except (ReleaseReadinessError, OSError, ValueError, KeyError, TypeError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
